"""Live re-scoring and global nudges (PLAN 16.2, 6.4).

After a learnable event the posterior mean is re-run on suggestions the user has not touched (and not seen yet);
a suggestion is replaced only if the model's *opinion* changed (its posterior-mean argmax now differs from the one it
was made with) AND the new best action differs from what is proposed. A nudge re-runs every untouched suggestion with
the accumulated score offsets. Replacements are NEW rows (the old row is marked superseded and remembers which event
did it), so undo is exact: see events.undo.

The replacement action is the deterministic posterior-mean argmax (propensity 1.0, policy name `<algo>:rescore` /
`<algo>:nudge`); the review-queue confidence is the Thompson probability of that action.
"""
from __future__ import annotations

from sqlmodel import Session

from app.api.io import ChangeOut
from app.models import SuggestionRow
from app.policy.log_replay import Learner
from app.svc.bundle import Bundle

MARGIN = 0.5  # a replacement must be better than the current action by this much (posterior-mean reward units)


def total_offsets(nudges: list[dict]) -> dict[str, dict[str, float]]:
    """Sum the per-arm score offsets of all active nudge events."""
    out: dict[str, dict[str, float]] = {}
    for n in nudges:
        for dt, arms in n["offsets"].items():
            for arm, v in arms.items():
                out.setdefault(dt, {}).setdefault(arm, 0.0)
                out[dt][arm] += v
    return out


def rescore(session: Session, b: Bundle, learner: Learner, event_id: int, nudge: bool) -> list[ChangeOut]:
    policy = learner.policy
    offsets = total_offsets(b.state.nudges)
    changes: list[ChangeOut] = []
    for s in b.active:
        if b.is_touched(s):
            continue
        unit = b.unit_of(s)
        if not nudge and unit is not None and unit in b.seen_units:
            continue  # the user has already looked at it: don't move things under their eyes
        ctx, dt = s.context_raw_json, s.decision_type
        off = offsets.get(dt)
        scores = policy.mean_scores(dt, ctx)
        base_best = max(scores, key=lambda a: scores[a])  # the model's opinion without any nudge
        for arm, o in (off or {}).items():
            if arm in scores:
                scores[arm] += o
        best = max(scores, key=lambda a: scores[a])
        current = b.action_of(s)
        if best == current or scores[best] - scores.get(current, scores[best]) < MARGIN:
            continue  # nothing better, or not clearly better (no flicker on near-ties)
        if nudge:
            if best == base_best and s.source != "nudge":
                continue  # differs from the model's opinion only through exploration, not through the nudge
        elif best == s.mean_best:
            continue  # the model's opinion has not changed since this suggestion was made (it was an exploration)
        conf = policy.choose(dt, ctx, off).probs.get(best, 0.5)
        new = SuggestionRow(
            video_id=s.video_id, target_id=s.target_id, decision_type=dt, proposed_action=best,
            candidates_json=list(s.candidates_json), context_raw_json=dict(ctx), propensity=1.0, confidence=conf,
            explored=False, source="nudge" if nudge else "policy",
            policy_name=f"{policy.name}:{'nudge' if nudge else 'rescore'}", policy_version=policy.version,
            feature_version=s.feature_version, mean_best=best, caused_by_event=event_id, rescored_from=s.id)
        session.add(new)
        session.flush()
        s.superseded, s.superseded_by_event = True, event_id
        session.add(s)
        changes.append(ChangeOut(old_id=s.id, new_id=new.id, decision_type=dt, target_id=s.target_id,
                                 from_action=s.proposed_action, to_action=best))
    return changes
