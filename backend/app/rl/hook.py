"""Hook selection: which line opens the video as its on-screen title.

The candidates are the first sentence plus the two strongest other sentences that survive the edit. A small linear
Thompson-sampling bandit over the candidates' own features (a question? a number? does it address "you"?) learns what
this creator likes as an opener, so what it learns transfers to new videos. The footage itself is not reordered: the
chosen line appears as the title over the first seconds. Rebuilt from the log on every use, like the rhythm agent.
"""
from __future__ import annotations

import re

import numpy as np

from app.policy.base import Choice
from app.schemas import UnitsFile

MAX_CANDIDATES = 3
MAX_WORDS = 8
FEATURES = ("bias", "is_first", "has_number", "is_question", "has_you", "n_words", "position", "centrality")
PRIOR = {"bias": 0.1, "is_first": 0.5, "is_question": 0.3, "has_number": 0.2, "has_you": 0.1}
LAMBDA = 2.0
V = 0.25
_END = (".", "?", "!", "…")


def candidates(uf: UnitsFile) -> list[dict]:
    """[{text, is_first, has_number, is_question, has_you, n_words, position, centrality}] with the first sentence first."""
    speech = {u.id: u for u in uf.units if u.kind == "speech"}
    skip = {i for i, u in speech.items() if u.features.get("is_earlier_take", 0.0) >= 0.5}
    sentences: list[list] = [[]]
    for w in sorted(uf.words, key=lambda w: w.idx):
        if w.is_filler or w.unit_id in skip or w.unit_id not in speech:
            continue
        sentences[-1].append(w)
        if w.text_display.strip().endswith(_END):
            sentences.append([])
    total = max(uf.duration, 1e-6)
    out = []
    for ws in sentences:
        if len(ws) < 3:
            continue
        raw = " ".join(w.text_display.strip() for w in ws[:MAX_WORDS])
        text = raw.rstrip(",;:.!?…- ").strip()
        low = " ".join(w.text_display.strip().lower() for w in ws)
        ids = {w.unit_id for w in ws}
        out.append({"text": text[0].upper() + text[1:] if text else text, "is_first": float(not out),
                    "has_number": float(bool(re.search(r"\d", low))), "is_question": float(ws[-1].text_display.strip().endswith("?")),
                    "has_you": float(bool(re.search(r"\byou(r|'ll|'re)?\b", low))), "n_words": min(len(ws), 16) / 16.0,
                    "position": ws[0].start / total,
                    "centrality": float(np.mean([speech[i].features.get("topic_centrality", 0.0) for i in ids]))})
    if not out:
        return []
    def strength(c: dict) -> float:
        return c["centrality"] + .4 * c["is_question"] + .3 * c["has_number"] + .15 * c["has_you"] - .3 * c["position"]
    rest = sorted(out[1:], key=lambda c: -strength(c))[:MAX_CANDIDATES - 1]
    return [out[0], *rest]


def context_of(cands: list[dict]) -> dict[str, float]:
    ctx: dict[str, float] = {"n_cand": float(len(cands))}
    for k, c in enumerate(cands):
        for f in FEATURES[1:]:
            ctx[f"c{k}_{f}"] = float(c[f])
    return ctx


def arms_of(ctx: dict[str, float]) -> list[str]:
    return [f"CAND_{k}" for k in range(int(ctx.get("n_cand", 0)))] + ["NONE"]


def _x(ctx: dict[str, float], arm: str) -> np.ndarray:
    k = arm.split("_")[1]
    return np.array([1.0, *(float(ctx.get(f"c{k}_{f}", 0.0)) for f in FEATURES[1:])])


class HookBandit:
    name = "hookTS"

    def __init__(self, seed: int = 0, greedy: bool = False):
        self.rng = np.random.default_rng(seed)
        self.greedy = greedy
        d = len(FEATURES)
        self.prior = np.array([PRIOR.get(f, 0.0) for f in FEATURES])
        self.A = LAMBDA * np.eye(d)
        self.b = LAMBDA * self.prior

    def update(self, ctx: dict[str, float], arm: str, reward: float) -> None:
        if arm == "NONE":
            return
        x = _x(ctx, arm)
        self.A += np.outer(x, x)
        self.b += reward * x

    def choose(self, ctx: dict[str, float]) -> tuple[Choice, str]:
        """(choice, posterior-mean best). Never proposes NONE by itself: having no title is the creator's call."""
        arms = [a for a in arms_of(ctx) if a != "NONE"]
        X = np.stack([_x(ctx, a) for a in arms])
        ainv = np.linalg.inv(self.A)
        mean = ainv @ self.b
        draws = self.rng.multivariate_normal(mean, V ** 2 * ainv, size=100)
        best = (draws @ X.T).argmax(axis=1)
        freq = np.bincount(best, minlength=len(arms)) / len(best)
        mean_k = int(np.argmax(X @ mean))
        k = mean_k if self.greedy else int(best[0])
        probs = dict(zip(arms, freq.tolist(), strict=True))
        return Choice(arms[k], float(max(freq[k], .01)), float(freq[k]), k != mean_k, probs), arms[mean_k]


def bandit_from_log(rows: list, events: list, seed: int = 0, greedy: bool = False) -> HookBandit:
    """A creator's hook bandit from their hook suggestions and non-undone events."""
    bandit = HookBandit(seed, greedy)
    last: dict[int, tuple[str, dict]] = {}
    for e in events:
        if e.suggestion_id is not None and e.kind in ("accept", "reject", "override"):
            last[e.suggestion_id] = (e.kind, e.payload_json)
    for r in rows:
        kind, payload = last.get(r.id, (None, {}))
        if kind == "accept":
            bandit.update(r.context_raw_json, r.proposed_action, 1.0)
        elif kind is not None:
            bandit.update(r.context_raw_json, r.proposed_action, -1.0)
            final = payload.get("final_action") or payload.get("applied")
            if final and final != r.proposed_action:
                bandit.update(r.context_raw_json, final, 1.0)
    return bandit
