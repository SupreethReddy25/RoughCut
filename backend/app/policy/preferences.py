"""What the agent has learned about a creator, as sentences (PLAN 16.3 "Sentences from top-weight features").

The sentences are read off the user's *learned policy*, not guessed from counts: probe contexts are scored with the
posterior-mean weights and the sentence says where the best action flips ("You remove pauses longer than ~0.6 s").
A decision type is only described once it has `MIN_EVIDENCE` learned updates, so the profile never states a
preference the agent has no evidence for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.pipeline import fillers as fl
from app.policy import context
from app.policy.base import Policy

MIN_EVIDENCE = 6  # learned updates of a decision type before it is described
FILLER_MARGIN = 0.25  # a filler word is mentioned when its two arms differ by at least this much
EMPHASIS_MIN_WEIGHT = 0.08

TOKEN_NAMES = {"um": "um", "uh": "uh", "like": "like", "you_know": "you know", "so": "so", "basically": "basically",
               "actually": "actually", "i_mean": "I mean"}
HIGHLIGHT = {"is_number": "numbers", "is_capitalized_mid_sentence": "names", "tfidf_in_transcript": "keywords",
             "word_loudness_z": "words you stress", "is_negation": "negations"}


@dataclass
class Preference:
    decision_type: str
    text: str


def _best(policy: Policy, dt: str, ctx: dict[str, float]) -> str:
    scores = policy.mean_scores(dt, ctx)
    return max(scores, key=lambda a: scores[a])


def _ctx(unit: dict[str, float], hist: dict[str, float] | None = None, video_type: str = "vlog") -> dict[str, float]:
    return context.build_context(unit, {"hist_output_ratio": 1.0, **(hist or {})}, video_type)


def _pauses(policy: Policy) -> list[str]:
    gaps = [round(g, 2) for g in np.arange(0.2, 3.01, 0.1)]
    best = [_best(policy, "pause", _ctx({"gap_seconds": g, "after_sentence_end": 1.0, "position": 0.5})) for g in gaps]

    def first_stable(group: set[str]) -> float | None:
        """The shortest gap from which the best action is in `group` for every longer gap too."""
        for i, g in enumerate(gaps):
            if all(b in group for b in best[i:]):
                return g
        return None

    cut = first_stable({"REMOVE"})
    tight = first_stable({"REMOVE", "TIGHT_15", "TIGHT_40"})
    if cut is not None and cut <= 2.0:
        extra = ""
        if tight is not None and tight < cut - 0.15:
            extra = ", and tighten the shorter ones" if tight <= gaps[0] + 0.05 else f", and tighten the ones above ~{tight:.1f} s"
        return [f"You remove pauses longer than ~{cut:.1f} s{extra}."]
    if tight is not None and tight <= 2.0:
        if tight <= gaps[0] + 0.05:
            return ["You tighten almost every pause rather than removing it."]
        return [f"You tighten pauses longer than ~{tight:.1f} s rather than removing them."]
    return ["You mostly keep pauses as they are."]


def _phrases(policy: Policy) -> list[str]:
    base = {"duration": 3.0, "avg_word_conf": 0.9, "position": 0.5}
    if _best(policy, "phrase", _ctx(base)) == "CUT":
        return ["You tend to cut phrases unless something makes them worth keeping."]
    reasons = []
    for label, change in (("an earlier take when you repeat yourself", {"is_earlier_take": 1.0, "retake_score": 0.95}),
                          ("speech that is unclear", {"avg_word_conf": 0.3}),
                          ("lines that are mostly filler words", {"filler_ratio": 0.7, "starts_with_filler": 1.0}),
                          ("lines that trail off", {"is_incomplete": 1.0})):
        if _best(policy, "phrase", _ctx({**base, **change})) == "CUT":
            reasons.append(label)
    if not reasons:
        return ["You rarely cut whole phrases."]
    return ["You cut " + _join(reasons) + "."]


def _fillers(policy: Policy) -> list[str]:
    removed, kept = [], []
    for tok in fl.TOKENS:
        if tok == "other":
            continue
        ctx = _ctx({f"tok_{tok}": 1.0, "pos_mid": 1.0, "gap_before": 0.2, "gap_after": 0.2, "token_duration": 0.3,
                    "word_conf": 0.9})
        s = policy.mean_scores("filler", ctx)
        diff = s["REMOVE"] - s["KEEP"]
        if diff >= FILLER_MARGIN:
            removed.append(TOKEN_NAMES[tok])
        elif diff <= -FILLER_MARGIN:
            kept.append(TOKEN_NAMES[tok])
    quote = lambda xs: _join([f"'{x}'" for x in xs])
    many = 5  # a list this long is "most of them"
    if len(removed) >= many and not kept:
        return ["You remove every filler word."]
    if len(removed) >= many:
        return [f"You remove most filler words but keep {quote(kept)}."]
    if len(kept) >= many and removed:
        return [f"You remove {quote(removed)} but keep the other filler words."]
    if len(kept) >= many:
        return ["You keep filler words."]
    if removed and kept:
        return [f"You remove {quote(removed)} but keep {quote(kept)}."]
    if removed:
        return [f"You remove {quote(removed)}."]
    if kept:
        return [f"You keep {quote(kept)}."]
    return []


def _zooms(policy: Policy) -> list[str]:
    grid = [(z, s) for z in (0.0, 1.0, 2.0) for s in (2.0, 10.0)]
    best = [_best(policy, "zoom", _ctx({"duration": 3.0, "loudness_z": z, "face_size": 0.3},
                                       {"hist_since_last_zoom": s})) for z, s in grid]
    if all(b == "NONE" for b in best):
        return ["You rarely zoom."]
    loud = _best(policy, "zoom", _ctx({"duration": 3.0, "loudness_z": 2.0}, {"hist_since_last_zoom": 10.0}))
    calm = _best(policy, "zoom", _ctx({"duration": 3.0, "loudness_z": 0.0}, {"hist_since_last_zoom": 10.0}))
    if loud != "NONE" and calm == "NONE":
        return ["You zoom in on phrases where you sound emphatic."]
    if all(b != "NONE" for b in best):
        return ["You like to zoom in, often."]
    return ["You zoom in now and then."]


def _emphasis(policy: Policy) -> list[str]:
    theta = getattr(policy, "theta", None)
    if theta is None:
        return []
    w = theta("emphasis", "CAND_0")
    names = context.feature_names("emphasis")
    ranked = sorted(((float(w[i]), names[i][5:]) for i in range(len(names)) if names[i][5:] in HIGHLIGHT
                     and names[i].startswith("cand:")), reverse=True)
    top = [HIGHLIGHT[n] for v, n in ranked[:2] if v >= EMPHASIS_MIN_WEIGHT]
    return [f"You highlight {_join(top)} in captions."] if top else []


def _join(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def describe_preferences(policy: Policy, evidence: dict[str, int]) -> list[Preference]:
    """The sentences for every decision type with enough evidence (`evidence` = learned updates per type)."""
    out: list[Preference] = []
    for dt, fn in (("pause", _pauses), ("phrase", _phrases), ("filler", _fillers), ("zoom", _zooms),
                   ("emphasis", _emphasis)):
        if evidence.get(dt, 0) >= MIN_EVIDENCE:
            out += [Preference(dt, s) for s in fn(policy)]
    return out
