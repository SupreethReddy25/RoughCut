"""Priors (PLAN 7.1-7.2): weight sets and ridge fits for style prototypes (the style posterior is in style.py).

A `Prior` is one full set of weights (theta per decision type and arm; D5 has one shared theta). LinTS uses it as
the mean of its prior (b0 = lambda * theta_prior). Prototypes are ridge fits on *full-information* labels
(+1 for the arm a persona wants, -1 for the others), i.e. what the expected reward of each arm should be.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from app.policy import context

SHARED = "*"  # arm key of the single shared theta of the action-feature form (emphasis)


def key_of(decision_type: str, arm: str) -> tuple[str, str]:
    return (decision_type, SHARED) if decision_type == "emphasis" else (decision_type, arm)


class Prior:
    def __init__(self, weights: dict[tuple[str, str], np.ndarray] | None = None):
        self.weights: dict[tuple[str, str], np.ndarray] = dict(weights or {})

    def theta(self, decision_type: str, arm: str) -> np.ndarray:
        k = key_of(decision_type, arm)
        w = self.weights.get(k)
        return np.zeros(context.dim(decision_type)) if w is None else w

    def score(self, decision_type: str, ctx: dict[str, float], arm: str) -> float:
        x = context.vectorize_arm(ctx, arm) if decision_type == "emphasis" else context.vectorize(decision_type, ctx)
        return float(self.theta(decision_type, arm) @ x)

    def to_dict(self) -> dict[str, list[float]]:
        return {f"{dt}|{arm}": w.tolist() for (dt, arm), w in self.weights.items()}

    @classmethod
    def from_dict(cls, d: dict[str, list[float]]) -> Prior:
        out = {}
        for k, v in d.items():
            dt, arm = k.split("|", 1)
            out[(dt, arm)] = np.asarray(v, dtype=float)
        return cls(out)

    @classmethod
    def mix(cls, priors: list[Prior], weights: np.ndarray) -> Prior:
        keys = {k for p in priors for k in p.weights}
        out = {}
        for k in keys:
            d = context.dim(k[0])
            out[k] = sum(float(w) * p.weights.get(k, np.zeros(d)) for p, w in zip(priors, weights, strict=True))
        return cls(out)


def fit_prior(labeled: Iterable[tuple[str, dict[str, float], str]], alpha: float = 1.0) -> Prior:
    """Ridge fit from (decision_type, context_raw, wanted_arm) triples."""
    xtx: dict[tuple[str, str], np.ndarray] = {}
    xty: dict[tuple[str, str], np.ndarray] = {}
    for dt, ctx, wanted in labeled:
        if dt == "emphasis":
            for arm in context.arms_for(dt, ctx):
                x = context.vectorize_arm(ctx, arm)
                k = key_of(dt, arm)
                xtx[k] = xtx.get(k, 0.0) + np.outer(x, x)
                xty[k] = xty.get(k, 0.0) + (1.0 if arm == wanted else -1.0) * x
        else:
            x = context.vectorize(dt, ctx)
            o = np.outer(x, x)
            for arm in context.ARMS[dt]:
                k = (dt, arm)
                xtx[k] = xtx.get(k, 0.0) + o
                xty[k] = xty.get(k, 0.0) + (1.0 if arm == wanted else -1.0) * x
    out = {}
    for k, m in xtx.items():
        out[k] = np.linalg.solve(m + alpha * np.eye(m.shape[0]), xty[k])
    return Prior(out)


# --------------------------------------------------------------------------- shipped priors (assets/priors)
def load_shipped_priors(directory=None) -> tuple[Prior, dict[str, Prior]] | None:
    """(pooled generic prior, {style name: prototype}) from assets/priors, or None if they were not built.

    Built by scripts/build_prototypes.py from the persona simulator; see PLAN 7.2.
    """
    import json
    from pathlib import Path

    from app import config

    d = Path(directory) if directory else config.ASSETS_DIR / "priors"
    try:
        generic = Prior.from_dict(json.loads((d / "generic.json").read_text(encoding="utf-8")))
        protos = json.loads((d / "prototypes.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return generic, {name: Prior.from_dict(w) for name, w in protos.items()}
