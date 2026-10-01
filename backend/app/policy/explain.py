"""Explanations (PLAN 7.7): top-3 feature contributions of the chosen arm over the runner-up, as sentences.

For a linear score the contribution of feature i to "action a beats runner-up r" is (theta_a - theta_r)_i * x_i.
Features without a readable template are skipped, so the sentence never says something meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.policy import context
from app.policy.base import Policy
from app.policy.linear import GlobalPolicy, LinearPolicy
from app.policy.rule import RulePolicy

VERB = {"CUT": "cut", "KEEP": "keep", "REMOVE": "remove", "TIGHT_40": "tighten to 0.4 s", "TIGHT_15": "tighten to 0.15 s",
        "NONE": "leave alone", "Z115": "zoom in a little on", "Z130": "zoom in a lot on"}
SUBJECT = {"phrase": "lines like this", "pause": "pauses like this", "zoom": "phrases like this",
           "filler": "these", "emphasis": "words like this"}
TOKEN_NAMES = {"um": "um", "uh": "uh", "like": "like", "you_know": "you know", "so": "so", "basically": "basically",
               "actually": "actually", "i_mean": "I mean", "other": "filler"}


@dataclass
class Explanation:
    text: str
    reasons: list[tuple[str, float, str]] = field(default_factory=list)  # (feature, contribution, sentence)
    confidence: float | None = None


def parse_bin(name: str) -> tuple[str, float | None, float | None]:
    """'gap_seconds>=2.0' -> ('gap_seconds', 2.0, None); 'x<0.3' -> ('x', None, 0.3); 'x[0.3,0.6)' -> ('x', 0.3, 0.6)."""
    if "[" in name:
        key, rng = name.split("[")
        a, b = rng.rstrip(")").split(",")
        return key, float(a), float(b)
    if ">=" in name:
        key, a = name.split(">=")
        return key, float(a), None
    key, b = name.split("<")
    return key, None, float(b)


def _bin_sentence(name: str, ctx: dict[str, float]) -> str | None:
    key, a, b = parse_bin(name)
    return _range_sentence(key, a, b, ctx)


def _range_sentence(key: str, a: float | None, b: float | None, ctx: dict[str, float]) -> str | None:
    v = ctx.get(key, 0.0)
    if key == "gap_seconds":
        return f"This pause is {v:.1f} s long"
    if key == "duration":
        return "It is very short" if b is not None and a is None else "It is quite long" if b is None else None
    if key == "retake_score":
        return "You say almost the same thing again right after" if a is not None and a >= 0.5 else None
    if key == "avg_word_conf":
        return "The speech there was unclear" if b is not None and b <= 0.5 else None
    if key == "topic_centrality":
        if b is not None and b <= 0.3 and a is None:
            return "It drifts away from your main topic"
        return "It is central to your topic" if a is not None and b is None and a >= 0.6 else None
    if key == "filler_ratio":
        return "It is mostly filler words" if a is not None and b is None and a >= 0.4 else None
    if key == "loudness_z":
        return "You sound emphatic here" if a is not None and b is None else None
    if key == "hist_since_last_zoom":
        return "It has been a while since the last zoom" if a is not None and b is None and a >= 6 else (
            "The last zoom was just a moment ago" if b is not None and a is None else None)
    return None


def describe(decision_type: str, name: str, ctx: dict[str, float]) -> str | None:
    """A readable reason for the feature `name` (None if it has no template)."""
    if name.startswith("cand:"):
        f = name[5:]
        return {"is_number": "It is a number", "is_capitalized_mid_sentence": "It is a proper noun",
                "tfidf_in_transcript": "It is a keyword of your topic", "word_loudness_z": "You stress it vocally",
                "is_negation": "It is a negation", "is_last_in_sentence": "It ends the sentence"}.get(f)
    if name.startswith("none:"):
        return "The line has a number in it" if name == "none:line_has_number" else None
    if any(c in name for c in "<>["):
        return _bin_sentence(name, ctx)
    if name.startswith("log_gap"):
        return f"This pause is {ctx.get('gap_seconds', 0.0):.1f} s long"
    if name.startswith("tok_"):
        tok = name[4:]
        gb, ga = ctx.get("gap_before", 0.0), ctx.get("gap_after", 0.0)
        clean = " and sits between clean gaps" if min(gb, ga) >= 0.1 else ""
        return f"This '{TOKEN_NAMES.get(tok, tok)}' is a filler{clean}"
    return {
        "after_sentence_end": "It follows a finished sentence", "before_new_sentence": "A new sentence starts after it",
        "prev_has_filler": "The line before it ended in a filler", "is_take_boundary": "It is where two takes join",
        "gap_rms": "The pause is not fully silent", "gap_voiced_ratio": "Something was said inside the pause",
        "is_earlier_take": "You repeat this line right after (an earlier take)", "is_incomplete": "It trails off mid-sentence",
        "retake_score": "You say almost the same thing again right after",
        "semantic_redundancy": "It says nearly the same as another line",
        "starts_with_filler": "It starts with a filler", "filler_ratio": "It is mostly filler words",
        "nonword_voiced_ratio": "There is voiced sound that is not words", "is_first": "It opens the video",
        "is_last": "It closes the video", "has_emphasis_punct": "It ends with ? or !",
        "likely_verb_like": "Here 'like' is probably a real word", "pos_start": "It starts the phrase",
        "pos_end": "It ends the phrase", "hist_prev_zoomed": "The previous phrase was zoomed",
        "topic_centrality": "It is central to your topic", "face_present": "Your face is in frame",
        "loudness_z": "You sound emphatic here",
    }.get(name)


def _theta(policy: Policy, decision_type: str, arm: str) -> np.ndarray | None:
    if isinstance(policy, LinearPolicy):
        return policy.theta(decision_type, arm)
    if isinstance(policy, GlobalPolicy):
        return policy.prior.theta(decision_type, arm)
    return None


def explain(policy: Policy, decision_type: str, ctx: dict[str, float], action: str,
            confidence: float | None = None, top: int = 3) -> Explanation:
    """Why `action` was (or would be) chosen for this context."""
    conf = f" ({confidence * 100:.0f}% sure)" if confidence is not None else ""
    if isinstance(policy, RulePolicy) or _theta(policy, decision_type, action) is None:
        return Explanation(f"A fixed rule suggests to {VERB.get(action, action)} {SUBJECT[decision_type]}.{conf}",
                           confidence=confidence)
    scores = policy.mean_scores(decision_type, ctx)
    others = [a for a in scores if a != action]
    runner = max(others, key=lambda a: scores[a]) if others else None
    names = context.feature_names(decision_type)
    if decision_type == "emphasis":
        xa = context.vectorize_arm(ctx, action)
        xr = context.vectorize_arm(ctx, runner) if runner else np.zeros_like(xa)
        th = _theta(policy, decision_type, action)
        contrib = th * xa - th * xr
    else:
        x = context.vectorize(decision_type, ctx)
        ta = _theta(policy, decision_type, action)
        tr = _theta(policy, decision_type, runner) if runner else np.zeros_like(ta)
        contrib = (ta - tr) * x
    reasons: list[tuple[str, float, str]] = []
    for i in np.argsort(-contrib):
        if contrib[i] <= 1e-9 or len(reasons) >= top:
            break
        sentence = describe(decision_type, names[i], ctx)
        if sentence and all(sentence != r[2] for r in reasons):
            reasons.append((names[i], float(contrib[i]), sentence))
    if decision_type == "emphasis" and action != "NONE" and any(r[0] == "cand:is_number" for r in reasons):
        tail = "You tend to highlight numbers."
    else:
        tail = f"You usually {VERB.get(action, action)} {SUBJECT[decision_type]}."
    why = "; ".join(r[2] for r in reasons)
    text = (f"{why}. {tail}" if why else tail) + conf
    return Explanation(text, reasons, confidence)
