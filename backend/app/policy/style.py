"""Style prior (PLAN 7.2, U5): which of the known editing styles does this creator resemble?

`StyleMixture` is the style posterior over the shipped prototypes (plus the pooled generic prior):
`log w_k += -(r - score_k(x, a))^2 / (2 sigma^2)` for every update of a user event. Two things come out of it:

- the *posterior* (`weights()`, sigma = 0.7 as in the PLAN): "Your style so far: 70% Snappy" and the answer to
  "has the style been identified?" (a weight above 0.8);
- the *prior mean* (`mean_prior()`) the bandit starts from. A sharp posterior would swap the whole prior after a
  single click, so the product tempers it (`prior_sigma` > `sigma`): the prior moves gently while the displayed
  posterior still sharpens as fast as the evidence allows. The simulator experiments leave `prior_sigma` unset
  (no tempering), which is the PLAN's plain model.

The mixture keeps updating while the user has fewer than `freeze_after` events, then freezes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.policy.prior import Prior

POOLED = "pooled"
IDENTIFIED_ABOVE = 0.8  # PLAN Phase 6: "Snappy is identified (> 0.8) within ~30 decisions"
SHOW_FROM = 0.10  # a style below this share is not worth a mention
SHOW_AT_MOST = 2
MIN_EVENTS_FOR_SHARES = 8  # below this the posterior is sharp but rests on a few clicks: name a leaning, no numbers

# display name and one line of what the style means, for the profile page (names are the persona / prototype keys)
STYLES: dict[str, tuple[str, str]] = {
    "snappy": ("Snappy", "tight pace: short pauses, fillers and retakes cut, quick punch-ins"),
    "relaxed": ("Relaxed", "breathing room: pauses and natural pacing kept, few cuts"),
    "storyteller": ("Storyteller", "keeps the narrative and the pauses that carry it, zooms on key moments"),
    "minimalist": ("Minimalist", "only the essential phrases, aiming for a short video"),
    "focused": ("Focused", "stays on topic: cuts tangents and repeats, highlights keywords"),
    "blend": ("Blend", "a bit of snappy and a bit of storyteller"),
    POOLED: ("Generic", "what most creators do, before it learns about you"),
}


def style_label(name: str) -> str:
    return STYLES.get(name, (name.replace("_", " ").title(), ""))[0]


class StyleMixture:
    """Style posterior over prototypes. See the module docstring."""

    def __init__(self, prototypes: dict[str, Prior], pooled: Prior, sigma: float = 0.7, freeze_after: int = 200,
                 prior_sigma: float | None = None):
        if sigma <= 0 or (prior_sigma is not None and prior_sigma < sigma):
            raise ValueError("sigma must be positive and prior_sigma at least sigma")
        self.names = [POOLED, *prototypes]
        self.priors = [pooled, *prototypes.values()]
        self.sigma = sigma
        self.prior_sigma = prior_sigma
        self.freeze_after = freeze_after
        self.logw = np.zeros(len(self.priors))
        self.n_events = 0
        self._mean: Prior | None = None

    @property
    def frozen(self) -> bool:
        return self.n_events >= self.freeze_after

    def observe(self, decision_type: str, ctx: dict[str, float], pairs: list[tuple[str, float]]) -> None:
        """One user EVENT = a list of (arm, reward) updates; it counts once towards the freeze threshold."""
        if self.frozen:
            return
        for k, p in enumerate(self.priors):
            for arm, reward in pairs:
                self.logw[k] -= (reward - p.score(decision_type, ctx, arm)) ** 2 / (2 * self.sigma ** 2)
        self.n_events += 1
        self._mean = None

    @staticmethod
    def _softmax(logw: np.ndarray) -> np.ndarray:
        w = np.exp(logw - logw.max())
        return w / w.sum()

    def weights(self) -> dict[str, float]:
        """The posterior over styles (sums to 1)."""
        return dict(zip(self.names, (float(v) for v in self._softmax(self.logw)), strict=True))

    def prior_weights(self) -> np.ndarray:
        """The weights the prior mean is built from: the posterior, tempered to `prior_sigma` when that is set."""
        if self.prior_sigma is None:
            return self._softmax(self.logw)
        return self._softmax(self.logw * (self.sigma / self.prior_sigma) ** 2)

    def mean_prior(self) -> Prior:
        if self._mean is None:
            self._mean = Prior.mix(self.priors, self.prior_weights())
        return self._mean


@dataclass
class StyleSummary:
    text: str  # "Your style so far: 70% Snappy · 25% Storyteller"
    top: str | None  # the most likely named style (None while the generic prior still leads)
    identified: bool  # that style's weight is above IDENTIFIED_ABOVE
    shares: list[tuple[str, float]]  # the styles named in `text`, most likely first


def summarize(weights: dict[str, float], n_events: int) -> StyleSummary:
    """The sentence of PLAN 7.2 and whether the style has been identified."""
    if n_events <= 0:
        return StyleSummary("No style yet: review a few suggestions and RoughCut will learn how you like to edit.",
                            None, False, [])
    named = sorted(((n, w) for n, w in weights.items() if n != POOLED), key=lambda nw: -nw[1])
    top = named[0][0] if named and named[0][1] >= weights.get(POOLED, 0.0) else None
    shown = [(n, w) for n, w in named[:SHOW_AT_MOST] if w >= SHOW_FROM]
    identified = bool(named) and named[0][1] > IDENTIFIED_ABOVE
    if not shown or top is None:
        return StyleSummary("Your style so far: not clear yet, still learning.", None, False, [])
    if n_events < MIN_EVENTS_FOR_SHARES:
        return StyleSummary(f"Your style so far: early guess, leaning {style_label(top)}.", top, False, shown)
    text = "Your style so far: " + " · ".join(f"{round(100 * w)}% {style_label(n)}" for n, w in shown)
    return StyleSummary(text, top, identified, shown)
