"""Project lifecycle: create + ingest job, list/get, suggest, seen, finalize, explain, render + downloads."""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

from sqlalchemy import func
from sqlmodel import Session, select

from app import config
from app import music as music_lib
from app.api.io import FinalizeOut, ProfileOut, ProjectOut, ProjectSummary, UserOut
from app.appctx import AppContext
from app.convert import event_from_row, row_from_suggestion, suggestion_from_row
from app.models import (
    EventRow,
    Metric,
    SuggestionRow,
    TextOverlayRow,
    UnitRow,
    User,
    Video,
    WordRow,
    vid_str,
)
from app.pipeline.ingest import probe
from app.pipeline.transcribe import LANGUAGE_WARN_PROB
from app.policy.explain import explain
from app.policy.preferences import describe_preferences
from app.policy.reward import DEFAULT, LEARNED_KINDS, updates_for
from app.policy.style import STYLES
from app.render.run import render as render_edl
from app.rl.hook import bandit_from_log
from app.rl.rhythm import RhythmAgent, episodes_from_log
from app.rl.rhythm import explain as explain_rhythm
from app.schemas import OutputSpec
from app.session import suggest as run_suggest
from app.state import sanitize_defaults
from app.svc import progress as progress_lib
from app.svc.bundle import ApiError, Bundle, event_dict, load_bundle, queue_out, state_out, style_out
from app.svc.events import arms_of, require_ready
from app.svc.stats import aggregate, video_stats

log = logging.getLogger(__name__)
PRESETS_PATH = Path(__file__).resolve().parents[1] / "render" / "presets.json"
VIDEO_TYPES = ("tutorial", "vlog", "opinion", "other")


def load_presets() -> dict:
    return {k: v for k, v in json.loads(PRESETS_PATH.read_text(encoding="utf-8")).items() if not k.startswith("_")}


# --------------------------------------------------------------------------- users
def create_user(session: Session, name: str) -> User:
    u = User(name=name.strip())
    if not u.name:
        raise ApiError(422, "name must not be blank")
    session.add(u)
    session.commit()
    return u


TRUST_WINDOW = 2  # finalized, reviewed videos the trust score looks at
TRUST_READY = 0.85  # trust needed for a batch video to be called "Ready to post"
READY_MAX_CORRECTIONS = 3.0  # ... together with fewer expected corrections than this


def trust_score(session: Session, user_id: int) -> float | None:
    """PLAN 8.8: how often the user accepted what was suggested, over their last 2 finalized videos (the reviewed
    suggestions only: a video that was exported without a review says nothing about it). None until one exists."""
    rows = session.exec(select(Metric).where(Metric.user_id == user_id, Metric.n_reviewed > 0,
                                             Metric.exported_via_autopilot == False)
                        .order_by(Metric.video_index.desc()).limit(TRUST_WINDOW)).all()  # type: ignore[attr-defined]
    reviewed = sum(m.n_reviewed for m in rows)
    return sum(m.n_accepted for m in rows) / reviewed if reviewed else None


def profile(ctx: AppContext, session: Session, user_id: int) -> ProfileOut:
    """What the agent has learned about this user (PLAN 16.3): style, sentences, editing statistics, curves, trust."""
    user = session.get(User, user_id)
    if user is None:
        raise ApiError(404, f"user {user_id} not found")
    videos = {v.id: v for v in session.exec(select(Video).where(Video.user_id == user_id)).all()}
    metrics = session.exec(select(Metric).where(Metric.user_id == user_id).order_by(Metric.video_index)).all()
    with ctx.locks(("user", user_id)):
        learner = ctx.policies.get(session, user_id)
        style = style_out(learner)
        n_learning = ctx.policies.fingerprint(session, user_id)[0]
        evidence = dict(session.exec(
            select(SuggestionRow.decision_type, func.count(EventRow.id)).join(SuggestionRow, SuggestionRow.id == EventRow.suggestion_id)
            .where(EventRow.user_id == user_id, EventRow.undone == False,
                   EventRow.kind.in_(LEARNED_KINDS)).group_by(SuggestionRow.decision_type)).all())
        preferences = [{"decision_type": p.decision_type, "text": p.text}
                       for p in describe_preferences(learner.policy, evidence)]
    finished = [videos[m.video_id] for m in metrics if m.video_id in videos]
    per_video = []
    for v in finished:
        try:
            st = video_stats(load_bundle(ctx, session, v.id))
        except Exception:  # a video whose files are gone must not break the profile
            log.warning("no statistics for project %s", v.id, exc_info=True)
            continue
        if st is not None:
            per_video.append(st)
    return ProfileOut(
        user=UserOut(id=user.id, name=user.name, created_at=user.created_at), n_projects=len(videos),
        n_finalized=len(metrics), n_learning_events=n_learning, style_mixture=style,
        trust=trust_score(session, user_id), trust_threshold=TRUST_READY, preferences=preferences,
        stats=aggregate(per_video), style_labels={k: {"label": v[0], "blurb": v[1]} for k, v in STYLES.items()},
        n_decisions_by_type=evidence,
        metrics=[{"video_index": m.video_index, "project_id": m.video_id, "title": videos[m.video_id].title,
                  "acceptance": m.acceptance, "n_decisions": m.n_decisions, "n_reviewed": m.n_reviewed,
                  "n_accepted": m.n_accepted, "n_corrected": m.n_corrected, "n_manual_actions": m.n_manual_actions,
                  "seconds_active": m.seconds_active, "duration": videos[m.video_id].duration,
                  "manual_per_minute": m.n_manual_actions / max(videos[m.video_id].duration / 60.0, 1e-9)}
                 for m in metrics if m.video_id in videos])


# --------------------------------------------------------------------------- create + ingest
BATCH_MAX = 5  # autopilot videos that may wait for / be in analysis at once (PLAN 8.8)


def create_project(ctx: AppContext, session: Session, user_id: int, inputs: list[Path], names: list[str],
                   video_type: str, target: float | None, title: str, autopilot: bool = False) -> Video:
    user = session.get(User, user_id)
    if user is None:
        raise ApiError(404, f"user {user_id} not found")
    if autopilot:
        waiting = session.exec(select(Video).where(Video.user_id == user_id, Video.autopilot == True,
                                                   Video.status.in_(("uploaded", "processing")))).all()  # type: ignore[attr-defined]
        if len(waiting) >= BATCH_MAX:
            raise ApiError(429, f"at most {BATCH_MAX} batch videos can be waiting at once: wait for one to finish")
    if not inputs:
        raise ApiError(422, "at least one video file is required")
    if video_type not in VIDEO_TYPES:
        raise ApiError(422, f"type must be one of {VIDEO_TYPES}")
    if target is not None and not 5 <= target <= 180:
        raise ApiError(422, "target duration must be between 5 and 180 seconds")
    try:
        total = sum(probe(p).duration for p in inputs)
    except Exception:  # unreadable here: the analysis job reports it properly
        total = 0.0
    if total > config.MAX_TOTAL_SECONDS:
        raise ApiError(422, f"This video is {int(total) // 60}:{int(total) % 60:02d} long. RoughCut handles up to "
                            f"{config.MAX_TOTAL_SECONDS // 60} minutes in total: trim it and upload again.")
    v = Video(user_id=user_id, title=(title.strip() or names[0]), sources_json=[str(p) for p in inputs],
              video_type=video_type, target_duration=target, status="uploaded", autopilot=autopilot,
              settings_json={"defaults": sanitize_defaults(user.defaults_json)})  # the user's remembered settings
    session.add(v)
    session.commit()
    pk = v.id
    # files were saved under uploads/<tmp>; move them next to the project so the pipeline input is stable
    dest = ctx.uploads_dir / vid_str(pk)
    moved = []
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(inputs):
            q = dest / f"{i:02d}_{Path(names[i]).name}"
            p.replace(q)
            moved.append(q)
    except OSError as e:  # never leave a project that no job will ever pick up
        v.status, v.error = "failed", f"could not store the upload: {e}"[:500]
        session.add(v)
        session.commit()
        raise ApiError(500, v.error) from None
    v.sources_json = [str(q) for q in moved]
    session.add(v)
    session.commit()
    job = ctx.jobs.submit(lambda: _ingest(ctx, pk, moved, video_type), kind="ingest", meta={"project": pk})
    with ctx.session() as s2:
        v2 = s2.get(Video, pk)
        v2.settings_json = {**v2.settings_json, "ingest_job": job.id, "expected_seconds": total}
        s2.add(v2)
        s2.commit()
    session.refresh(v)
    return v


def _set_status(ctx: AppContext, pk: int, status: str, error: str | None = None, **fields) -> None:
    with ctx.session() as s:
        v = s.get(Video, pk)
        v.status, v.error, v.updated_at = status, error, time.time()
        for k, val in fields.items():
            setattr(v, k, val)
        s.add(v)
        s.commit()


def _ingest(ctx: AppContext, pk: int, inputs: list[Path], video_type: str) -> None:
    """Runs on the ingest pool (ONE worker), so analyses never overlap."""
    _set_status(ctx, pk, "processing")
    try:
        out = ctx.video_dir(pk)
        uf = ctx.ingest_fn(inputs, vid_str(pk), video_type, out)
        if not any(u.kind == "speech" for u in uf.units):
            raise RuntimeError("no speech was detected in the video: there is nothing to edit")
        _thumbnail(out)
        progress_lib.learn(ctx, pk, uf.duration)
        src_dims = _source_dims(out)
        with ctx.session() as s:
            for model in (UnitRow, WordRow):  # a re-run (restart recovery) starts from a clean slate
                for row in s.exec(select(model).where(model.video_id == pk)).all():
                    s.delete(row)
            s.flush()
            s.add_all(UnitRow(video_id=pk, unit_id=u.id, idx=u.idx, kind=u.kind, parent_unit_id=u.parent_unit_id,
                              start=u.start, end=u.end, text=u.text, features_json=dict(u.features)) for u in uf.units)
            s.add_all(WordRow(video_id=pk, word_id=w.id, idx=w.idx, unit_id=w.unit_id, start=w.start, end=w.end,
                              text_original=w.text_original, text_display=w.text_display, conf=w.conf,
                              is_filler=w.is_filler) for w in uf.words)
            s.commit()
        ctx.drop_analysis(pk)
        with ctx.session() as s:
            v = s.get(Video, pk)
            v.settings_json = {**v.settings_json, "source": src_dims}
            s.add(v)
            s.commit()
        _set_status(ctx, pk, "ready", duration=uf.duration, source_map_json=list(uf.source_map),
                    feature_version=uf.feature_version)
        _autopilot_suggest(ctx, pk)
    except Exception as e:
        _set_status(ctx, pk, "failed", f"{type(e).__name__}: {e}"[:500])
        raise


def _autopilot_suggest(ctx: AppContext, pk: int) -> None:
    """A batch (autopilot) video is auto-edited as soon as it is analysed (PLAN 8.8). A failure here must not fail the
    analysis: the suggestions are then simply made when the project is opened."""
    try:
        with ctx.session() as s:
            if s.get(Video, pk).autopilot:
                suggest(ctx, s, pk)
    except Exception:
        log.warning("autopilot suggestions for project %s failed; they will be made when it is opened", pk, exc_info=True)


def _source_dims(video_dir: Path) -> dict:
    """Pixel size of source.mp4 (the crop maths of the browser preview needs it); 1920x1080 if there is no media."""
    default = {"width": 1920, "height": 1080}
    source = video_dir / "source.mp4"
    if not source.exists():
        return default
    try:
        info = probe(source)
        return {"width": info.width, "height": info.height}
    except Exception:  # best effort: a probe problem must not fail the analysis that just succeeded
        log.warning("could not probe %s; assuming 1920x1080", source, exc_info=True)
        return default


def _thumbnail(video_dir: Path) -> None:
    """Best effort: a frame for the projects list (the pipeline's source.mp4 may not exist in tests)."""
    src = video_dir / "source.mp4"
    if not src.exists():
        return
    try:
        subprocess.run([config.ffmpeg_path(), "-y", "-loglevel", "error", "-ss", "1", "-i", str(src), "-frames:v", "1",
                        "-vf", "scale=-2:320", str(video_dir / "thumb.jpg")], check=False, timeout=60,
                       capture_output=True)
    except (OSError, subprocess.SubprocessError, RuntimeError):
        log.warning("thumbnail failed for %s", video_dir)


# --------------------------------------------------------------------------- list / get
def summary(ctx: AppContext, v: Video, auto: dict | None = None) -> ProjectSummary:
    return ProjectSummary(id=v.id, video_id=vid_str(v.id), user_id=v.user_id, title=v.title, status=v.status,
                          error=v.error, duration=v.duration, video_type=v.video_type,
                          target_duration=v.target_duration, created_at=v.created_at, updated_at=v.updated_at,
                          has_thumbnail=(ctx.video_dir(v.id) / "thumb.jpg").exists(), autopilot=v.autopilot,
                          **(auto or {}))


def autopilot_badge(ctx: AppContext, session: Session, v: Video, trust: float | None) -> dict:
    """PLAN 8.8: a batch video that nobody has touched yet is "ready" to post when the agent expects fewer than 3
    corrections (sum of 1 - confidence over what is still pending) AND the user trusts it (trust >= 0.85); otherwise it
    "needs review" (n_review = the uncertain decisions the editor opens on)."""
    if not v.autopilot or v.status != "ready":
        return {}
    b = load_bundle(ctx, session, v.id)
    pending = [s for s in b.active if b.status_of(s) == "pending"]
    if not pending:
        return {}
    expected = sum(1.0 - s.confidence for s in pending)
    ready = expected < READY_MAX_CORRECTIONS and trust is not None and trust >= TRUST_READY
    return {"badge": "ready" if ready else "review", "n_review": len(queue_out(b).needs_review),
            "expected_corrections": round(expected, 2)}


def list_projects(ctx: AppContext, session: Session, user_id: int) -> list[ProjectSummary]:
    if session.get(User, user_id) is None:
        raise ApiError(404, f"user {user_id} not found")
    rows = session.exec(select(Video).where(Video.user_id == user_id).order_by(Video.updated_at.desc(),  # type: ignore[attr-defined]
                                                                               Video.id.desc())).all()  # type: ignore[attr-defined]
    trust = trust_score(session, user_id)
    return [summary(ctx, v, autopilot_badge(ctx, session, v, trust)) for v in rows]


def get_project(ctx: AppContext, session: Session, pk: int) -> ProjectOut:
    b = load_bundle(ctx, session, pk)
    with ctx.locks(("user", b.video.user_id)):  # the learner is shared and mutated by other requests
        learner = ctx.policies.get(session, b.video.user_id)
        state = state_out(b, learner).model_dump()
    job = ctx.jobs.get(b.video.settings_json.get("ingest_job", ""))
    uf = b.analysis.uf if b.analysis is not None else None
    auto = autopilot_badge(ctx, session, b.video, trust_score(session, b.video.user_id))
    data = {**summary(ctx, b.video, auto).model_dump(), **state}
    return ProjectOut(**data, source_map=list(b.video.source_map_json), units=list(uf.units) if uf else [],
                      words=list(uf.words) if uf else [], seen_unit_ids=sorted(b.seen_units),
                      exports=dict(b.video.settings_json.get("exports", {})), job=job.public() if job else None,
                      source=dict(b.video.settings_json.get("source", {"width": 1920, "height": 1080})),
                      progress=progress_lib.snapshot(ctx, b.video) if b.video.status in ("uploaded", "processing") else None,
                      rating=next((int(e.payload_json["value"]) for e in reversed(b.live) if e.kind == "rating"), None))


# --------------------------------------------------------------------------- suggest
def suggest(ctx: AppContext, session: Session, pk: int):
    with ctx.locks(("video", pk)):
        b = load_bundle(ctx, session, pk)
        require_ready(b)
        uid = b.video.user_id
        with ctx.locks(("user", uid)):
            learner = ctx.policies.get(session, uid)
            if b.sug_rows:  # idempotent: the suggestions of a video are made once
                return state_out(b, learner, message="Suggestions already exist.")
            auto = b.video.autopilot
            rhythm, hook = _agents(session, uid, pk, auto) if ctx.agents else (None, None)
            res = run_suggest(b.analysis.uf, b.analysis.word_feats, learner.policy, video_type=b.video.video_type,
                              source="autopilot" if auto else "policy", greedy=auto,
                              target=b.video.target_duration, seed=pk, rhythm=rhythm, hook=hook)
            rows = [row_from_suggestion(s, pk, mean_best=mb) for s, mb in zip(res.suggestions, res.mean_best, strict=True)]
            session.add_all(rows)
            session.flush()
            ctx.policies.commit(session, uid, learner)
            session.commit()
            return state_out(load_bundle(ctx, session, pk), learner,
                             message=language_warning(ctx, pk) or f"{len(rows)} suggestions made; "
                             f"{len(res.skipped_fillers)} fillers left alone (no clean gap to cut in).")


def _agents(session: Session, uid: int, pk: int, greedy: bool):
    """The creator's rhythm and hook agents, rebuilt from everything they have logged."""
    vids = session.exec(select(Video.id).where(Video.user_id == uid)).all()
    rows = session.exec(select(SuggestionRow).where(SuggestionRow.video_id.in_(vids),  # type: ignore[attr-defined]
                                                    SuggestionRow.superseded == False,
                                                    SuggestionRow.decision_type.in_(("rhythm", "hook")))  # type: ignore[attr-defined]
                        .order_by(SuggestionRow.id)).all()
    events = session.exec(select(EventRow).where(EventRow.video_id.in_(vids), EventRow.undone == False)  # type: ignore[attr-defined]
                          .order_by(EventRow.id)).all()
    rhythm = RhythmAgent(seed=pk, greedy=greedy)
    for ep in episodes_from_log([r for r in rows if r.decision_type == "rhythm"], events):
        rhythm.add_episode(ep)
    return rhythm, bandit_from_log([r for r in rows if r.decision_type == "hook"], events, seed=pk, greedy=greedy)


def language_warning(ctx: AppContext, pk: int) -> str | None:
    """Analysis is English-only: say so when the speech is clearly something else, rather than show a made-up edit."""
    try:
        tr = json.loads((ctx.video_dir(pk) / "transcript.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    lang, prob = tr.get("detected_language"), tr.get("detected_language_prob", 0.0)
    if lang and lang != "en" and prob >= LANGUAGE_WARN_PROB:
        return (f"This video sounds like it is not in English ({lang}, {prob:.0%} sure). RoughCut's analysis is English "
                "only, so the transcript and the cuts below are unreliable.")
    return None


def recover_interrupted(ctx: AppContext) -> int:
    """Projects left 'uploaded'/'processing' by a crash or restart: run their ingest again (or fail them clearly)."""
    n = 0
    with ctx.session() as s:
        for v in s.exec(select(Video).where(Video.status.in_(("uploaded", "processing")))).all():  # type: ignore[attr-defined]
            inputs = [Path(p) for p in v.sources_json]
            if inputs and all(p.exists() for p in inputs):
                job = ctx.jobs.submit(lambda pk=v.id, ins=inputs, vt=v.video_type: _ingest(ctx, pk, ins, vt),
                                      kind="ingest", meta={"project": v.id, "recovered": True})
                v.settings_json = {**v.settings_json, "ingest_job": job.id}
                n += 1
            else:
                v.status, v.error = "failed", "interrupted by a restart and the uploaded files are gone: upload again"
            s.add(v)
        s.commit()
    return n


NOT_EXPLAINABLE = {
    "rhythm": "",
    "hook": "Picked from this video's strongest lines, by what you have preferred as an opener so far.",
    "pace": "Picked from how fast this phrase is spoken and the pace you have kept on phrases like it.",
    "music_timing": "Picked from the kind of video and how long its first sentence is.",
}


def explain_suggestion(ctx: AppContext, session: Session, pk: int, sid: int) -> dict:
    b = load_bundle(ctx, session, pk)
    s = b.sug_by_id.get(sid)
    if s is None:
        raise ApiError(404, f"suggestion {sid} not found in this project")
    with ctx.locks(("user", b.video.user_id)):
        if s.decision_type in NOT_EXPLAINABLE:
            text = (explain_rhythm(s.context_raw_json, b.action_of(s)) if s.decision_type == "rhythm"
                    else NOT_EXPLAINABLE[s.decision_type])
            return {"suggestion_id": sid, "text": text, "confidence": s.confidence, "reasons": []}
        if s.decision_type == "phrase" and s.context_raw_json.get("crew_score", 0.0) >= 0.5:
            return {"suggestion_id": sid, "text": "This sounds like talk to the crew rather than to your viewer, so it was "
                    "cut. Restore it if it belongs in the video.", "confidence": s.confidence, "reasons": []}
        learner = ctx.policies.get(session, b.video.user_id)
        ex = explain(learner.policy, s.decision_type, s.context_raw_json, s.proposed_action, s.confidence)
    return {"suggestion_id": sid, "text": ex.text, "confidence": s.confidence,
            "reasons": [{"feature": f, "contribution": c, "sentence": t} for f, c, t in ex.reasons]}


# --------------------------------------------------------------------------- seen / finalize
def mark_seen(ctx: AppContext, session: Session, pk: int, unit_ids: list[int]) -> dict:
    with ctx.locks(("video", pk)):
        v = session.get(Video, pk)
        if v is None:
            raise ApiError(404, f"project {pk} not found")
        rows = session.exec(select(UnitRow).where(UnitRow.video_id == pk, UnitRow.unit_id.in_(set(unit_ids)))).all()  # type: ignore[attr-defined]
        known = {r.unit_id for r in rows}
        unknown = sorted(set(unit_ids) - known)
        if unknown:
            raise ApiError(422, f"unknown unit ids {unknown[:10]}")
        for r in rows:
            if not r.seen:
                r.seen = True
                session.add(r)
        session.commit()
        return {"marked": len(rows)}


def finalize(ctx: AppContext, session: Session, pk: int, seconds_active: float) -> FinalizeOut:
    with ctx.locks(("video", pk)):
        b = load_bundle(ctx, session, pk)
        require_ready(b)
        uid = b.video.user_id
        with ctx.locks(("user", uid)):
            try:
                out = _finalize(ctx, session, b, uid, seconds_active)
                session.commit()
                return out
            except ApiError:
                session.rollback()
                raise
            except BaseException:
                session.rollback()
                ctx.policies.invalidate(uid)
                raise


def _finalize(ctx: AppContext, session: Session, b: Bundle, uid: int, seconds_active: float) -> FinalizeOut:
    learner = ctx.policies.get(session, uid)
    have_implicit = {e.suggestion_id for e in b.live if e.kind == "implicit"}
    created: list[EventRow] = []
    for s in b.active:  # untouched + seen -> implicit accept (+0.5); never seen -> no signal
        unit = b.unit_of(s)
        if b.is_touched(s) or s.id in have_implicit or unit is None or unit not in b.seen_units:
            continue
        ups = updates_for("implicit", s.decision_type, s.proposed_action, arms_of(s), None, DEFAULT)
        created.append(EventRow(user_id=uid, video_id=b.video.id, kind="implicit", suggestion_id=s.id, payload_json={},
                                reward=ups[0].reward, weight=ups[0].weight))
    session.add_all(created)
    session.flush()
    for e in created:
        learner.apply_event(suggestion_from_row(b.sug_by_id[e.suggestion_id]), event_from_row(e))
    b2 = load_bundle(ctx, session, b.video.id)
    n_dec = len(b2.active)
    n_reviewed = sum(1 for s in b2.active if b2.is_touched(s) or b2.unit_of(s) in b2.seen_units
                     or b2.status_of(s) != "pending")
    n_acc = sum(1 for s in b2.active if b2.status_of(s) == "accepted")
    n_cor = sum(1 for s in b2.active if b2.status_of(s) == "corrected")
    manual_kinds = {"fine_cut", "crop", "caption_text", "text_overlay", "speed", "setting"}
    n_manual = sum(1 for e in b2.live if e.kind in manual_kinds or e.payload_json.get("via") == "manual")
    m = session.exec(select(Metric).where(Metric.video_id == b.video.id)).first()
    if m is None:
        idx = len(session.exec(select(Metric).where(Metric.user_id == uid)).all()) + 1
        m = Metric(user_id=uid, video_id=b.video.id, video_index=idx)
    m.n_decisions, m.n_reviewed, m.n_accepted, m.n_corrected = n_dec, n_reviewed, n_acc, n_cor
    m.n_manual_actions, m.acceptance = n_manual, (n_acc / n_reviewed if n_reviewed else 0.0)
    m.seconds_active, m.ts = seconds_active, time.time()
    session.add(m)
    if b.video.status in ("ready", "exported"):
        b.video.status = "edited"
    b.video.updated_at = time.time()
    session.add(b.video)
    session.flush()
    ctx.policies.commit(session, uid, learner)
    st = state_out(b2, learner, events=[event_dict(e) for e in created])
    return FinalizeOut(**st.model_dump(), metrics={
        "video_index": m.video_index, "n_decisions": n_dec, "n_reviewed": n_reviewed, "n_accepted": n_acc,
        "n_corrected": n_cor, "n_manual_actions": n_manual, "acceptance": m.acceptance,
        "seconds_active": seconds_active}, implicit_events=len(created))


# --------------------------------------------------------------------------- render / download
def export_dir(ctx: AppContext, pk: int) -> Path:
    return ctx.exports_dir / vid_str(pk)


def sync_text_overlays(session: Session, b: Bundle) -> None:
    """Keep the TextOverlay table equal to the overlays of the current edit state."""
    for r in session.exec(select(TextOverlayRow).where(TextOverlayRow.video_id == b.video.id)).all():
        session.delete(r)
    for ov in b.state.texts.values():
        session.add(TextOverlayRow(video_id=b.video.id, overlay_id=ov.id, text=ov.text, t0=ov.t0, t1=ov.t1,
                                   position=ov.position, y=ov.y, style_id=ov.style_id))


def start_render(ctx: AppContext, session: Session, pk: int, preset: str) -> dict:
    presets = load_presets()
    if preset not in presets:
        raise ApiError(422, f"unknown preset {preset!r}; choose one of {sorted(presets)}")
    pr = presets[preset]
    if pr["fps"] != 30:
        raise ApiError(500, "presets must be 30 fps: the normalised source is 30 fps")
    with ctx.locks(("video", pk)):
        b = load_bundle(ctx, session, pk)
        require_ready(b)
        opts = b.state.build_options(OutputSpec(width=pr["width"], height=pr["height"], fps=pr["fps"]))
        from app.edl.build import build_edl

        edl = build_edl(b.analysis.uf, b.state.decisions, b.analysis.faces, opts)
        if not edl.clips:
            raise ApiError(409, "nothing is kept in this edit, so there is nothing to render")
        music_path = None
        if edl.audio.music is not None:
            music_path = music_lib.track_path(ctx.data_dir, b.video.user_id, edl.audio.music.track_id)
            if music_path is None:
                raise ApiError(409, f"the music track {edl.audio.music.track_id!r} is not available any more: "
                                    "choose another track or remove the music")
        last_event = max((e.id for e in b.live if e.kind != "rating"), default=0)
        sync_text_overlays(session, b)
        warn = edl.duration > pr["max_duration_s"]
        source = ctx.video_dir(pk) / "source.mp4"
        out_dir = export_dir(ctx, pk)

        def work() -> dict:
            out_dir.mkdir(parents=True, exist_ok=True)
            with ctx.locks(("render", pk, preset)):  # two renders of one preset must not write the same files
                res = render_edl(edl, source, out_dir / f"{preset}.mp4", out_dir / f"{preset}.srt", crf=pr["crf"],
                                 audio_kbps=pr["audio_kbps"], progress=lambda *_: None, music_path=music_path)
            info = {"event_id": last_event, "duration": res["duration"], "loudness": res["loudness"]["I"],
                    "ts": time.time(), "over_platform_limit": warn}
            with ctx.session() as s:
                v = s.get(Video, pk)
                v.settings_json = {**v.settings_json, "exports": {**v.settings_json.get("exports", {}), preset: info}}
                if latest_live_event(s, pk) == last_event and v.status in ("ready", "edited"):
                    v.status = "exported"
                v.updated_at = time.time()
                s.add(v)
                s.commit()
            return info

        session.commit()  # before the job starts: it writes the same database
        job = ctx.jobs.submit(work, kind="render", meta={"project": pk, "preset": preset})
        return {"job": job.public(), "preset": preset, "duration": edl.duration,
                "warning": f"{edl.duration:.0f}s is longer than {pr['label']} allows ({pr['max_duration_s']}s)"
                if warn else None}


def latest_live_event(session: Session, pk: int) -> int:
    """Id of the newest non-undone event of a project (0 if none): identifies the edit a render was made from."""
    return max(session.exec(select(EventRow.id).where(EventRow.video_id == pk, EventRow.undone == False,
                                                      EventRow.kind != "rating")).all(), default=0)


def download_path(ctx: AppContext, session: Session, pk: int, fmt: str, preset: str | None) -> tuple[Path, str, dict]:
    v = session.get(Video, pk)
    if v is None:
        raise ApiError(404, f"project {pk} not found")
    if fmt not in ("mp4", "srt"):
        raise ApiError(422, "fmt must be mp4 or srt (png covers arrive with the cover-image feature)")
    exports = v.settings_json.get("exports", {})
    if not exports:
        raise ApiError(404, "nothing has been rendered yet: POST /render first")
    name = preset or max(exports, key=lambda k: exports[k]["ts"])
    if name not in exports:
        raise ApiError(404, f"no render for preset {name!r}")
    path = export_dir(ctx, pk) / f"{name}.{fmt}"
    if not path.exists():
        raise ApiError(404, f"{path.name} is missing: render again")
    return path, name, {"stale": latest_live_event(session, pk) != exports[name]["event_id"]}
