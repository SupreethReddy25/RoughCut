"""DB rows <-> the frozen pydantic contracts in schemas.py."""
from __future__ import annotations

from app.models import EventRow, SuggestionRow, vid_str
from app.schemas import Event, Suggestion


def suggestion_from_row(r: SuggestionRow) -> Suggestion:
    return Suggestion(
        id=r.id, video_id=vid_str(r.video_id), target_id=r.target_id, decision_type=r.decision_type,  # type: ignore[arg-type]
        proposed_action=r.proposed_action, candidates=list(r.candidates_json), context_raw=dict(r.context_raw_json),
        propensity=r.propensity, confidence=r.confidence, explored=r.explored, source=r.source,  # type: ignore[arg-type]
        policy_name=r.policy_name, policy_version=r.policy_version, feature_version=r.feature_version,
        superseded=r.superseded)


def row_from_suggestion(s: Suggestion, video_pk: int, **extra) -> SuggestionRow:
    return SuggestionRow(
        video_id=video_pk, target_id=s.target_id, decision_type=s.decision_type, proposed_action=s.proposed_action,
        candidates_json=list(s.candidates), context_raw_json=dict(s.context_raw), propensity=s.propensity,
        confidence=s.confidence, explored=s.explored, source=s.source, policy_name=s.policy_name,
        policy_version=s.policy_version, feature_version=s.feature_version, superseded=s.superseded, **extra)


def event_from_row(r: EventRow) -> Event:
    return Event(id=r.id, user_id=r.user_id, video_id=vid_str(r.video_id), kind=r.kind,  # type: ignore[arg-type]
                 suggestion_id=r.suggestion_id, payload=dict(r.payload_json), reward=r.reward, weight=r.weight,
                 undone=r.undone, ts=r.ts)
