"""Event log -> policy state (PLAN 5, principle 2): policy state is a cache that can always be recomputed.

`Learner` is the ONE place where an event becomes bandit updates. The simulator, the API and `replay()` all go
through it, so "replay equals online" holds by construction (and is tested). Undone events are skipped, which is
exactly how undo / redo work: flip the flag and replay.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

from app.policy import context
from app.policy.base import Policy
from app.policy.reward import DEFAULT, LEARNED_KINDS, RewardConfig, Update, updates_for
from app.policy.style import StyleMixture
from app.schemas import Event, Suggestion


class Learner:
    """A policy plus (optionally) the style-mixture prior that is updated alongside it."""

    def __init__(self, policy: Policy, mixture: StyleMixture | None = None, cfg: RewardConfig = DEFAULT):
        self.policy, self.cfg = policy, cfg
        # the style prior only means something to policies that have a prior to swap (the linear bandits)
        self.mixture = mixture if hasattr(policy, "set_prior") else None
        if self.mixture is not None:
            policy.set_prior(self.mixture.mean_prior())

    def apply(self, decision_type: str, ctx: dict[str, float], proposed: str, kind: str,
              final: str | None = None, arms: list[str] | None = None, manual: bool = False) -> list[Update]:
        """Turn one event on a suggestion into updates and apply them. Returns what was applied."""
        if decision_type in context.AGENT_TYPES:
            return []  # those agents are rebuilt from the log (app/rl)
        arms = arms or context.arms_for(decision_type, ctx)
        ups = updates_for(kind, decision_type, proposed, arms, final, self.cfg, manual)
        for u in ups:
            self.policy.update(decision_type, ctx, u.arm, u.reward, u.weight)
        if ups and self.mixture is not None and not self.mixture.frozen:
            self.mixture.observe(decision_type, ctx, [(u.arm, u.reward) for u in ups])
            self.policy.set_prior(self.mixture.mean_prior())  # once per event, not per update
        return ups

    def apply_event(self, sug: Suggestion, ev: Event) -> list[Update]:
        if ev.undone or ev.kind not in LEARNED_KINDS:
            return []
        arms = list(sug.candidates) or None
        # a user who changes their mind twice moves FROM the action currently in the edit, not from the original
        # proposal: the API records that action as `from_action` whenever it differs
        proposed = ev.payload.get("from_action") or sug.proposed_action
        return self.apply(sug.decision_type, sug.context_raw, proposed, ev.kind,
                          ev.payload.get("final_action"), arms, manual=ev.payload.get("via") == "manual")


def replay(learner: Learner, suggestions: dict[int, Suggestion], events: Iterable[Event]) -> int:
    """Apply the non-undone events in order. Returns the number of bandit updates applied."""
    n = 0
    for ev in events:
        if ev.undone or ev.kind not in LEARNED_KINDS:
            continue
        if ev.suggestion_id is None or ev.suggestion_id not in suggestions:
            raise KeyError(f"event {ev.id} refers to unknown suggestion {ev.suggestion_id}")
        n += len(learner.apply_event(suggestions[ev.suggestion_id], ev))
    return n


def recompute_from_log(make_learner: Callable[[], Learner], suggestions: dict[int, Suggestion],
                       events: Iterable[Event]) -> Learner:
    """A fresh learner rebuilt from the log (used by undo/redo, prior swaps, schema changes)."""
    learner = make_learner()
    replay(learner, suggestions, events)
    return learner
