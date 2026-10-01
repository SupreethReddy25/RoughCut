"""Loading everything about one video (rows + analysis + derived edit state) and turning it into API output."""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlmodel import Session, select

from app.api.io import QueueOut, StateOut, StyleOut, SuggestionOut
from app.appctx import Analysis, AppContext
from app.edl.build import build_edl
from app.models import EventRow, SuggestionRow, UnitRow, Video, vid_str
from app.policy.log_replay import Learner
from app.state import EditState, current_action, rebuild

EXPLICIT = ("accept", "reject", "override")  # events that mean "the user looked at this suggestion and decided"
NEEDS_EYE_BELOW = 0.7  # PLAN 7.3


class ApiError(Exception):
    def __init__(self, status: int, detail: str, **extra):
        super().__init__(detail)
        self.status, self.detail, self.extra = status, detail, extra


@dataclass
class Bundle:
    video: Video
    analysis: Analysis
    sug_rows: list[SuggestionRow]
    ev_rows: list[EventRow]
    seen_units: set[int]
    active: list[SuggestionRow] = field(default_factory=list)  # not superseded
    live: list[EventRow] = field(default_factory=list)  # not undone
    sug_by_id: dict[int, SuggestionRow] = field(default_factory=dict)
    explicit: dict[int, list[EventRow]] = field(default_factory=dict)  # suggestion id -> its live explicit events
    implicit_for: set[int] = field(default_factory=set)  # suggestions with a live implicit accept
    state: EditState = field(default_factory=EditState)

    def action_of(self, s: SuggestionRow) -> str:
        return current_action(self.state, s.decision_type, s.target_id, s.context_raw_json)

    def unit_of(self, s: SuggestionRow) -> int | None:
        """The unit a suggestion is about (for 'seen' tracking). Emphasis: the unit of its first candidate word."""
        if s.decision_type != "emphasis":
            return s.target_id
        wid = s.context_raw_json.get("c0_word_id")
        if wid is None:
            return None
        w = self.analysis.uf.words[int(wid)]  # word id == index in a UnitsFile
        return w.unit_id

    def status_of(self, s: SuggestionRow) -> str:
        kinds = {e.kind for e in self.explicit.get(s.id, [])}
        if kinds & {"reject", "override"}:
            return "corrected"
        if "accept" in kinds or s.id in self.implicit_for:
            return "accepted"
        return "pending"

    def is_touched(self, s: SuggestionRow) -> bool:
        return bool(self.explicit.get(s.id))


def load_bundle(ctx: AppContext, session: Session, video_pk: int) -> Bundle:
    video = session.get(Video, video_pk)
    if video is None:
        raise ApiError(404, f"project {video_pk} not found")
    sugs = list(session.exec(select(SuggestionRow).where(SuggestionRow.video_id == video_pk)
                             .order_by(SuggestionRow.id)).all())
    evs = list(session.exec(select(EventRow).where(EventRow.video_id == video_pk).order_by(EventRow.id)).all())
    seen = set(session.exec(select(UnitRow.unit_id).where(UnitRow.video_id == video_pk,
                                                          UnitRow.seen == True)).all())
    b = Bundle(video, ctx.analysis(video_pk) if video.status in ("ready", "edited", "exported") else None,  # type: ignore[arg-type]
               sugs, evs, seen)
    b.active = [s for s in sugs if not s.superseded]
    b.live = [e for e in evs if not e.undone]
    b.sug_by_id = {s.id: s for s in sugs}
    for e in b.live:
        if e.kind in EXPLICIT and e.suggestion_id is not None:
            b.explicit.setdefault(e.suggestion_id, []).append(e)
        elif e.kind == "implicit" and e.suggestion_id is not None:
            b.implicit_for.add(e.suggestion_id)
    if b.analysis is not None:
        b.state = rebuild(b.analysis.uf, b.active, b.live)
    return b


# --------------------------------------------------------------------------- undo / redo availability
def undoable(e: EventRow) -> bool:
    return e.kind != "implicit"  # finalize's implicit accepts are bookkeeping, not user actions


def undo_target(b: Bundle) -> EventRow | None:
    live = [e for e in b.ev_rows if not e.undone and undoable(e)]
    return live[-1] if live else None


def redo_target(b: Bundle) -> EventRow | None:
    live = [e.id for e in b.ev_rows if not e.undone and undoable(e)]
    floor = max(live) if live else 0
    cands = [e for e in b.ev_rows if e.undone and undoable(e) and e.id > floor]
    return cands[0] if cands else None  # the earliest undone one: events come back in the order they were made


# --------------------------------------------------------------------------- output
def suggestion_out(b: Bundle, s: SuggestionRow) -> SuggestionOut:
    unit = b.unit_of(s)
    ctx = s.context_raw_json
    cand_ids = [int(ctx[f"c{k}_word_id"]) for k in range(int(ctx.get("n_cand", 0)))] if s.decision_type == "emphasis" else []
    return SuggestionOut(
        id=s.id, decision_type=s.decision_type, target_id=s.target_id, proposed_action=s.proposed_action,
        current_action=b.action_of(s), candidates=list(s.candidates_json), candidate_word_ids=cand_ids,
        confidence=s.confidence, propensity=s.propensity, explored=s.explored, source=s.source,
        status=b.status_of(s), unit_id=unit, seen=unit in b.seen_units if unit is not None else False)


def queue_out(b: Bundle) -> QueueOut:
    """PLAN 7.3: pending suggestions only; priority = 1 - confidence; 'needs your eye' = confidence < 0.7."""
    pending = [s for s in b.active if b.status_of(s) == "pending"]
    pending.sort(key=lambda s: (-(1 - s.confidence), s.id))
    return QueueOut(needs_review=[s.id for s in pending if s.confidence < NEEDS_EYE_BELOW],
                    confident=[s.id for s in pending if s.confidence >= NEEDS_EYE_BELOW])


def style_out(learner: Learner | None) -> StyleOut | None:
    if learner is None or learner.mixture is None:
        return None
    m = learner.mixture
    return StyleOut(weights=m.weights(), n_events=m.n_events, frozen=m.frozen)


def build_edl_for(b: Bundle):
    if b.analysis is None:
        return None
    return build_edl(b.analysis.uf, b.state.decisions, b.analysis.faces, b.state.build_options())


def state_out(b: Bundle, learner: Learner | None, **extra) -> StateOut:
    ud, rd = undo_target(b), redo_target(b)
    return StateOut(
        status=b.video.status, settings=b.state.settings(),
        suggestions=[suggestion_out(b, s) for s in b.active], queue=queue_out(b), edl=build_edl_for(b),
        style_mixture=style_out(learner), can_undo=ud is not None, can_redo=rd is not None,
        n_events=len(b.live), **extra)


def event_dict(e: EventRow) -> dict:
    return {"id": e.id, "kind": e.kind, "suggestion_id": e.suggestion_id, "payload": e.payload_json,
            "reward": e.reward, "weight": e.weight, "undone": e.undone, "ts": e.ts, "video_id": vid_str(e.video_id)}
