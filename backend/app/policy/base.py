"""Policy interface (PLAN 7.1): every policy maps (decision_type, context_raw) -> Choice and learns by update().

`choose` never mutates learned state, so the same call can be used to score without side effects (live
re-scoring uses `mean_scores`). Randomness comes only from the policy's own generator.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from app.policy import context


@dataclass
class Choice:
    action: str
    propensity: float  # probability the policy picks `action` in this context (logged for OPE)
    confidence: float  # drives the review queue (PLAN 7.3); for Thompson sampling == propensity
    explored: bool = False  # chose something other than the posterior-mean argmax
    probs: dict[str, float] = field(default_factory=dict)


class Policy(ABC):
    name = "base"
    version = 0

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.n_updates = 0

    def arms(self, decision_type: str, ctx: dict[str, float]) -> list[str]:
        return context.arms_for(decision_type, ctx)

    @abstractmethod
    def choose(self, decision_type: str, ctx: dict[str, float],
               offsets: dict[str, float] | None = None) -> Choice:
        """Pick an action. `offsets` adds a per-arm score offset (global nudges, PLAN 6.4 / 8.10)."""

    def update(self, decision_type: str, ctx: dict[str, float], arm: str, reward: float,
               weight: float = 1.0) -> None:
        """Learn from one (context, arm, reward). Stateless policies ignore it."""

    @abstractmethod
    def mean_scores(self, decision_type: str, ctx: dict[str, float]) -> dict[str, float]:
        """Posterior-mean (or rule) score of every arm, no exploration."""

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


def softmax_confidence(scores: np.ndarray, idx: int, tau: float = 0.25) -> float:
    """Confidence for deterministic policies: softmax probability of the chosen arm."""
    z = (scores - scores.max()) / tau
    e = np.exp(z)
    return float(e[idx] / e.sum())
