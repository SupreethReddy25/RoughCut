"""Application context: data directories, DB engine, job runner, locks, per-user policies."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func
from sqlmodel import Session, select

from app import config
from app.convert import event_from_row, suggestion_from_row
from app.db import make_engine
from app.edl.build import load_processed
from app.jobs import JobRunner, LockMap
from app.models import EventRow, PolicyState, StylePosterior, SuggestionRow, now
from app.pipeline.run import run_pipeline
from app.pipeline.vision import FaceSample
from app.policy.linear import LinTS
from app.policy.log_replay import Learner, replay
from app.policy.prior import Prior, StyleMixture, load_shipped_priors
from app.policy.reward import LEARNED_KINDS
from app.policy.store import PolicyStore
from app.schemas import UnitsFile

# Product settings of the per-user bandit (tuned in the simulator, see reports/figures and the Phase 3 notes):
# a stronger prior than the PLAN default lam=2 keeps ONE click from re-scoring the whole video (with lam=2 a single
# phrase cut flipped ~30 confident phrase suggestions), and a softer style posterior (sigma=3 instead of 0.7) stops
# the style mixture from collapsing onto one prototype after a single event. Simulator accuracy is unchanged.
PRODUCT_LAMBDA = 10.0
STYLE_SIGMA = 3.0

log = logging.getLogger(__name__)
IngestFn = Callable[[list[Path], str, str, Path], UnitsFile]


def default_ingest(inputs: list[Path], video_id: str, video_type: str, out_dir: Path) -> UnitsFile:
    return run_pipeline(inputs, video_id, video_type, out_dir, progress=lambda *_: None)


@dataclass
class Analysis:
    uf: UnitsFile
    faces: list[FaceSample]
    word_feats: dict[int, dict[str, float]]


def _config_key(shipped) -> str:
    """Fingerprint of everything a cached learner depends on besides the event log."""
    import hashlib
    import json

    h = hashlib.sha1(f"{PRODUCT_LAMBDA}|{STYLE_SIGMA}".encode())
    if shipped is not None:
        generic, protos = shipped
        h.update(json.dumps([generic.to_dict(), {k: p.to_dict() for k, p in sorted(protos.items())}],
                            sort_keys=True).encode())
    return h.hexdigest()[:16]


class PolicyManager:
    """One LinTS learner (+ style mixture) per user, always consistent with the user's event log.

    The learner is updated online by each request; before use it is checked against a fingerprint of the user's
    non-undone learning events (count + id sum). A mismatch (undo, redo, a failed request, another process) makes
    it rebuild from the log, so the cache can never silently drift (PLAN 5, principle 2).
    """

    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.store = PolicyStore(ctx.data_dir / "policies")
        self.config_key = _config_key(ctx.shipped_priors)
        self._cache: dict[int, tuple[Learner, tuple[int, int]]] = {}
        self._guard = threading.Lock()

    # -- construction
    def _new_learner(self, user_id: int) -> Learner:
        shipped = self.ctx.shipped_priors
        if shipped is not None:
            generic, protos = shipped
            mixture = StyleMixture(protos, generic, sigma=STYLE_SIGMA)
            return Learner(LinTS(mixture.mean_prior(), lam=PRODUCT_LAMBDA, seed=user_id), mixture)
        return Learner(LinTS(Prior(), lam=PRODUCT_LAMBDA, seed=user_id), None)

    @staticmethod
    def fingerprint(session: Session, user_id: int) -> tuple[int, int]:
        n, total = session.exec(select(func.count(EventRow.id), func.coalesce(func.sum(EventRow.id), 0)).where(
            EventRow.user_id == user_id, EventRow.undone == False,
            EventRow.kind.in_(LEARNED_KINDS))).one()
        return int(n), int(total)

    def _rebuild(self, session: Session, user_id: int) -> Learner:
        learner = self._new_learner(user_id)
        rows = session.exec(select(EventRow).where(EventRow.user_id == user_id, EventRow.undone == False,
                                                   EventRow.kind.in_(LEARNED_KINDS)).order_by(EventRow.id)).all()
        sug_ids = {r.suggestion_id for r in rows if r.suggestion_id is not None}
        sugs = {}
        if sug_ids:
            for r in session.exec(select(SuggestionRow).where(SuggestionRow.id.in_(sug_ids))).all():
                sugs[r.id] = suggestion_from_row(r)
        replay(learner, sugs, [event_from_row(r) for r in rows])
        return learner

    # -- API
    def get(self, session: Session, user_id: int) -> Learner:
        fp = self.fingerprint(session, user_id)
        with self._guard:
            hit = self._cache.get(user_id)
            if hit is not None and hit[1] == fp:
                return hit[0]
        learner = self._try_load(session, user_id, fp) or self._rebuild(session, user_id)
        with self._guard:
            self._cache[user_id] = (learner, fp)
        return learner

    def _try_load(self, session: Session, user_id: int, fp: tuple[int, int]) -> Learner | None:
        learner = self._new_learner(user_id)
        expect = {"n": fp[0], "sum": fp[1], "cfg": self.config_key}
        try:
            if not self.store.load_into(user_id, learner.policy, n_events=fp[0], expect=expect):
                return None
            if learner.mixture is not None and not self.store.load_mixture(user_id, learner.mixture):
                return None
        except Exception:
            log.warning("policy cache of user %s is unreadable; rebuilding from the event log", user_id, exc_info=True)
            return None
        return learner

    def commit(self, session: Session, user_id: int, learner: Learner) -> None:
        """Call after the request's DB changes are flushed/committed: re-fingerprint and persist the cache."""
        fp = self.fingerprint(session, user_id)
        with self._guard:
            self._cache[user_id] = (learner, fp)
        self.store.save(user_id, learner.policy, n_events=fp[0],
                        extra={"n": fp[0], "sum": fp[1], "cfg": self.config_key})
        row = session.exec(select(PolicyState).where(PolicyState.user_id == user_id,
                                                     PolicyState.algo == learner.policy.name)).first()
        path = str(self.store.root / f"user{user_id}_{learner.policy.name}.npz")
        if row is None:
            session.add(PolicyState(user_id=user_id, algo=learner.policy.name, blob_path=path, n_events=fp[0],
                                    version=learner.policy.version))
        else:
            row.n_events, row.version, row.blob_path = fp[0], learner.policy.version, path
            session.add(row)
        if learner.mixture is not None:
            self.store.save_mixture(user_id, learner.mixture)
            sp = session.get(StylePosterior, user_id)
            payload = {"weights": learner.mixture.weights(), "n_events": learner.mixture.n_events}
            if sp is None:
                session.add(StylePosterior(user_id=user_id, weights_json=payload, frozen=learner.mixture.frozen))
            else:
                sp.weights_json, sp.frozen, sp.updated_at = payload, learner.mixture.frozen, now()
                session.add(sp)

    def invalidate(self, user_id: int) -> None:
        with self._guard:
            self._cache.pop(user_id, None)


class AppContext:
    def __init__(self, data_dir: str | Path | None = None, ingest_fn: IngestFn | None = None, workers: int = 2,
                 shipped_priors="auto"):
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR
        self.processed_dir = self.data_dir / "processed"
        self.uploads_dir = self.data_dir / "uploads"
        self.exports_dir = self.data_dir / "exports"
        for d in (self.processed_dir, self.uploads_dir, self.exports_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.engine = make_engine(self.data_dir / "db.sqlite")
        self.jobs = JobRunner(workers)
        self.locks = LockMap()
        self.ingest_fn = ingest_fn or default_ingest
        self.shipped_priors = load_shipped_priors() if shipped_priors == "auto" else shipped_priors
        self._analysis: dict[int, Analysis] = {}
        self._analysis_guard = threading.Lock()
        self.policies = PolicyManager(self)

    def session(self) -> Session:
        return Session(self.engine, expire_on_commit=False)

    def video_dir(self, video_pk: int) -> Path:
        return self.processed_dir / f"v_{video_pk}"

    def analysis(self, video_pk: int) -> Analysis:
        with self._analysis_guard:
            a = self._analysis.get(video_pk)
        if a is None:
            uf, faces, wf = load_processed(self.video_dir(video_pk))
            a = Analysis(uf, faces, wf)
            with self._analysis_guard:
                self._analysis[video_pk] = a
        return a

    def drop_analysis(self, video_pk: int) -> None:
        with self._analysis_guard:
            self._analysis.pop(video_pk, None)

    def close(self) -> None:
        self.jobs.shutdown()
        self.engine.dispose()
