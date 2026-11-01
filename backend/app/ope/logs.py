"""Logged suggestions + events -> `Logged` decisions for off-policy evaluation (same code for real logs and the
simulator).

Reward = "the creator kept the suggestion": accept / untouched-but-seen at finalize (implicit) = 1, reject /
override to another action = 0. Only suggestions made by the exploring policy are used (`source == "policy"`):
nudge re-runs and autopilot (greedy, propensity 1) are not samples of the logging policy (PLAN 7.4). A
suggestion that was never seen has no verdict and is left out. When a suggestion has several non-undone
verdicts (accept, then override), the last one counts.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from app.ope.estimators import Logged
from app.policy import context
from app.schemas import Event, Suggestion

VERDICT = {"accept": 1.0, "implicit": 1.0, "reject": 0.0}


def reward_of(ev: Event, proposed: str) -> float | None:
    if ev.kind in VERDICT:
        return VERDICT[ev.kind]
    if ev.kind == "override":
        final = (ev.payload or {}).get("final_action")
        return 1.0 if final == proposed else 0.0
    return None


def logged_from(suggestions: Mapping[int, Suggestion], events: Iterable[Event],
                truth: Mapping[int, dict[str, float]] | None = None) -> list[Logged]:
    """`truth` (simulator only): suggestion id -> {arm: E[reward | x, arm]}."""
    last: dict[int, float] = {}
    for ev in events:
        if ev.undone or ev.suggestion_id is None or ev.suggestion_id not in suggestions:
            continue
        r = reward_of(ev, suggestions[ev.suggestion_id].proposed_action)
        if r is not None:
            last[ev.suggestion_id] = r
    out = []
    for sid, r in last.items():
        s = suggestions[sid]
        if s.source != "policy":
            continue
        dt = s.decision_type
        arms = list(s.candidates) if dt == "emphasis" and s.candidates else context.arms_for(dt, s.context_raw)
        groups = s.video_id if isinstance(s.video_id, int) else hash(s.video_id) % (2 ** 31)
        out.append(Logged(dt, dict(s.context_raw), arms, s.proposed_action, float(s.propensity), r, groups,
                          dict((truth or {}).get(sid, {}))))
    return out


def true_reward(arms: list[str], ideal: str, noise: float, p_override: float) -> dict[str, float]:
    """Simulator: E[kept | x, a]. The persona's verdict is its ideal action, except with prob `noise` a random
    other arm; when the verdict differs from the proposal it overrides with prob `p_override`, else leaves it."""
    k = len(arms)
    out = {}
    for a in arms:
        agree = (1 - noise) if a == ideal else (noise / (k - 1) if k > 1 else 0.0)
        out[a] = agree + (1 - agree) * (1 - p_override)
    return out
