"""Rule-based decisions for the EDL/render CLI: simply the RulePolicy (PLAN 7.1) run through the shared session.

Cut pauses > 0.5 s, cut earlier takes, remove um/uh (clean cuts only), no zoom, emphasise the longest candidate word.
"""
from __future__ import annotations

from app.edl.build import Decisions
from app.policy.rule import RulePolicy
from app.schemas import UnitsFile
from app.session import suggest


def rule_decisions(uf: UnitsFile, word_features: dict[int, dict[str, float]] | None = None) -> Decisions:
    return suggest(uf, word_features, RulePolicy()).decisions
