"""RulePolicy (PLAN 7.1): fixed thresholds, i.e. what a simple editor's one-click features do.

Cut pauses > 0.5 s, cut earlier takes, remove um/uh only, no zoom, emphasise the longest candidate word.
Deterministic and stateless (propensity = confidence = 1).
"""
from __future__ import annotations

from app.policy import context
from app.policy.base import Choice, Policy

PAUSE_CUT_OVER = 0.5
EARLIER_TAKE_CUT = 0.5


class RulePolicy(Policy):
    name = "rule"

    def decide(self, decision_type: str, ctx: dict[str, float]) -> str:
        if decision_type == "pause":
            return "REMOVE" if ctx.get("gap_seconds", 0.0) > PAUSE_CUT_OVER else "KEEP"
        if decision_type == "phrase":
            return "CUT" if ctx.get("is_earlier_take", 0.0) >= EARLIER_TAKE_CUT else "KEEP"
        if decision_type == "filler":
            return "REMOVE" if (ctx.get("tok_um") == 1.0 or ctx.get("tok_uh") == 1.0) else "KEEP"
        if decision_type == "zoom":
            return "NONE"
        if decision_type == "emphasis":
            n = int(ctx.get("n_cand", 0))
            if n == 0:
                return "NONE"
            return f"CAND_{max(range(n), key=lambda k: (ctx.get(f'c{k}_word_len', 0.0), -k))}"
        raise ValueError(f"unknown decision type {decision_type!r}")

    def mean_scores(self, decision_type: str, ctx: dict[str, float]) -> dict[str, float]:
        pick = self.decide(decision_type, ctx)
        return {a: float(a == pick) for a in context.arms_for(decision_type, ctx)}

    def choose(self, decision_type: str, ctx: dict[str, float],
               offsets: dict[str, float] | None = None) -> Choice:
        scores = self.mean_scores(decision_type, ctx)
        for a, o in (offsets or {}).items():  # a global nudge can still tip a rule
            if a in scores:
                scores[a] += o
        action = max(scores, key=lambda a: scores[a])
        return Choice(action, 1.0, 1.0, False, {a: float(a == action) for a in scores})
