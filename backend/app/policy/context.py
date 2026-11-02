"""Decision contexts (PLAN 6.3): raw named features -> small normalised vectors the linear policies use.

`context_raw` (stored on every Suggestion) holds *raw named* values: the unit's static features, the dynamic
history features, a video-type one-hot and a bias. `vectorize()` turns it into the model input, so a policy can
always be rebuilt from the log (PLAN 5, principle 2) and the vectorisation is the only place that knows scales.

Bins are one-hot over edges chosen to line up with the thresholds people actually use (0.5 s, 0.8 s, 4 s, ...);
the PLAN's own bin edges are a subset of them.

D5 (caption emphasis) is an action-feature bandit: each candidate word k has its own raw block `c{k}_*`, and the
NONE arm has line-level features. Both map into ONE shared vector space (shared theta).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from app.pipeline import fillers as fl
from app.pipeline.semantic import STOPWORDS
from app.schemas import (
    FILLER_ACTIONS,
    MUSIC_TIMING_ACTIONS,
    PACE_ACTIONS,
    PAUSE_ACTIONS,
    PHRASE_ACTIONS,
    RHYTHM_ACTIONS,
    ZOOM_ACTIONS,
    Word,
)

CONTEXT_VERSION = 1
VIDEO_TYPES = ("tutorial", "vlog", "opinion", "other")
DISJOINT = ("phrase", "pause", "zoom", "filler")
ARMS: dict[str, list[str]] = {
    "phrase": list(PHRASE_ACTIONS),
    "pause": list(PAUSE_ACTIONS),
    "zoom": list(ZOOM_ACTIONS),
    "filler": list(FILLER_ACTIONS),
    "rhythm": list(RHYTHM_ACTIONS),
    "pace": list(PACE_ACTIONS),
    "music_timing": list(MUSIC_TIMING_ACTIONS),
}
# decided by their own agents (app/rl), so the bandit neither models nor updates them
AGENT_TYPES = ("rhythm", "hook")
MAX_CANDIDATES = 3
EMPHASIS_ARMS = ["NONE"] + [f"CAND_{k}" for k in range(MAX_CANDIDATES)]
ZOOM_NEVER = 30.0  # hist_since_last_zoom before the first zoom (seconds)

# ---- spec entries: ("lin", key, scale) | ("log", key, scale) | ("bin", key, edges)
_TYPE = [("lin", f"type_{t}", 1.0) for t in VIDEO_TYPES] + [("lin", "bias", 1.0)]

SPECS: dict[str, list[tuple]] = {
    "phrase": [
        ("lin", "duration", 6.0), ("lin", "speech_rate", 5.0), ("lin", "avg_word_conf", 1.0),
        ("lin", "loudness_z", 3.0), ("lin", "filler_ratio", 1.0), ("lin", "starts_with_filler", 1.0),
        ("lin", "nonword_voiced_ratio", 1.0), ("lin", "retake_score", 1.0), ("lin", "is_earlier_take", 1.0),
        ("lin", "is_incomplete", 1.0), ("lin", "position", 1.0), ("lin", "is_first", 1.0),
        ("lin", "is_last", 1.0), ("lin", "face_present", 1.0), ("lin", "semantic_redundancy", 1.0),
        ("lin", "topic_centrality", 1.0), ("lin", "hist_output_ratio", 1.0), ("lin", "hist_since_last_cut", 10.0),
        ("bin", "duration", (1.0, 2.5, 4.0)), ("bin", "retake_score", (0.5, 0.85)),
        ("bin", "avg_word_conf", (0.5, 0.8)), ("bin", "topic_centrality", (0.3, 0.6)),
        ("bin", "filler_ratio", (0.2, 0.4)),
    ] + _TYPE,
    "pause": [
        ("log", "gap_seconds", 1.5), ("bin", "gap_seconds", (0.3, 0.5, 0.6, 0.8, 1.0, 2.0)),
        ("lin", "after_sentence_end", 1.0), ("lin", "before_new_sentence", 1.0), ("lin", "gap_rms", 1.0),
        ("lin", "gap_voiced_ratio", 1.0), ("lin", "prev_has_filler", 1.0), ("lin", "is_take_boundary", 1.0),
        ("lin", "position", 1.0), ("lin", "hist_output_ratio", 1.0),
    ] + _TYPE,
    "zoom": [
        ("lin", "duration", 6.0), ("lin", "loudness_z", 3.0), ("lin", "speech_rate", 5.0),
        ("lin", "has_emphasis_punct", 1.0), ("lin", "is_first", 1.0), ("lin", "face_size", 1.0),
        ("lin", "topic_centrality", 1.0), ("lin", "hist_prev_zoomed", 1.0),
        ("bin", "hist_since_last_zoom", (2.0, 4.0, 6.0, 10.0)), ("bin", "loudness_z", (0.5, 1.0)),
        ("bin", "topic_centrality", (0.3, 0.6)), ("bin", "duration", (1.5, 3.0)),
    ] + _TYPE,
    "filler": [("lin", f"tok_{t}", 1.0) for t in fl.TOKENS] + [
        ("lin", "pos_start", 1.0), ("lin", "pos_mid", 1.0), ("lin", "pos_end", 1.0),
        ("lin", "token_duration", 0.5), ("lin", "gap_before", 1.0), ("lin", "gap_after", 1.0),
        ("lin", "word_conf", 1.0), ("lin", "likely_verb_like", 1.0), ("lin", "loudness_z", 3.0),
    ] + _TYPE,
    "pace": [
        ("lin", "speech_rate", 5.0), ("lin", "duration", 6.0), ("lin", "loudness_z", 3.0), ("lin", "filler_ratio", 1.0),
        ("lin", "topic_centrality", 1.0), ("lin", "is_first", 1.0), ("lin", "position", 1.0),
        ("bin", "speech_rate", (2.0, 3.0, 4.0)),
    ] + _TYPE,
    "music_timing": [("lin", "video_minutes", 3.0), ("lin", "first_sentence_seconds", 10.0)] + _TYPE,
}

# D5: per-candidate block and the NONE-arm block (all live in one shared space)
CAND_KEYS = [("is_number", 1.0), ("is_capitalized_mid_sentence", 1.0), ("word_len", 10.0),
             ("tfidf_in_transcript", 1.0), ("word_loudness_z", 3.0), ("position_in_line", 1.0),
             ("is_last_in_sentence", 1.0), ("is_negation", 1.0)]
NONE_KEYS = [("line_len", 4.0), ("line_has_number", 1.0), ("bias", 1.0)] + [(f"type_{t}", 1.0) for t in VIDEO_TYPES]
EMPH_DIM = len(CAND_KEYS) + 1 + len(NONE_KEYS)  # candidate features, is_candidate flag, NONE features


def _bin_names(key: str, edges: Sequence[float]) -> list[str]:
    names, lo = [], None
    for e in edges:
        names.append(f"{key}<{e}" if lo is None else f"{key}[{lo},{e})")
        lo = e
    names.append(f"{key}>={lo}")
    return names


def _compile(spec: list[tuple]) -> list[str]:
    out: list[str] = []
    for kind, key, arg in spec:
        out += _bin_names(key, arg) if kind == "bin" else [key if kind == "lin" else f"log_{key}"]
    return out


NAMES: dict[str, list[str]] = {dt: _compile(sp) for dt, sp in SPECS.items()}
NAMES["emphasis"] = ([f"cand:{k}" for k, _ in CAND_KEYS] + ["cand:is_candidate"]
                     + [f"none:{k}" for k, _ in NONE_KEYS])
DIMS = {dt: len(n) for dt, n in NAMES.items()}


def dim(decision_type: str) -> int:
    return DIMS[decision_type]


def feature_names(decision_type: str) -> list[str]:
    return NAMES[decision_type]


def vectorize(decision_type: str, ctx: dict[str, float]) -> np.ndarray:
    """Model input for the four disjoint decision types."""
    out = np.zeros(DIMS[decision_type])
    i = 0
    for kind, key, arg in SPECS[decision_type]:
        v = float(ctx.get(key, 0.0))
        if kind == "lin":
            out[i] = max(-3.0, min(3.0, v / arg))
            i += 1
        elif kind == "log":
            out[i] = math.log1p(max(v, 0.0)) / arg
            i += 1
        else:  # one-hot bin
            j = 0
            while j < len(arg) and v >= arg[j]:
                j += 1
            out[i + j] = 1.0
            i += len(arg) + 1
    return out


def arms_for(decision_type: str, ctx: dict[str, float] | None = None) -> list[str]:
    if decision_type == "emphasis":
        n = int((ctx or {}).get("n_cand", 0))
        return EMPHASIS_ARMS[: 1 + n]
    if decision_type == "hook":
        return [f"CAND_{k}" for k in range(int((ctx or {}).get("n_cand", 0)))] + ["NONE"]
    return ARMS[decision_type]


def vectorize_arm(ctx: dict[str, float], arm: str) -> np.ndarray:
    """Action-feature vector of one D5 arm (candidate word k, or NONE) in the shared space."""
    x = np.zeros(EMPH_DIM)
    nc = len(CAND_KEYS)
    if arm == "NONE":
        for j, (k, scale) in enumerate(NONE_KEYS):
            x[nc + 1 + j] = max(-3.0, min(3.0, float(ctx.get(k, 0.0)) / scale))
        return x
    k = int(arm.split("_")[1])
    for j, (key, scale) in enumerate(CAND_KEYS):
        x[j] = max(-3.0, min(3.0, float(ctx.get(f"c{k}_{key}", 0.0)) / scale))
    x[nc] = 1.0
    return x


def arm_matrix(ctx: dict[str, float], arms: list[str]) -> np.ndarray:
    return np.stack([vectorize_arm(ctx, a) for a in arms])


# --------------------------------------------------------------------------- building contexts
def type_onehot(video_type: str) -> dict[str, float]:
    t = video_type if video_type in VIDEO_TYPES else "other"
    return {f"type_{v}": float(v == t) for v in VIDEO_TYPES}


def build_context(unit_features: dict[str, float], hist: dict[str, float], video_type: str) -> dict[str, float]:
    """context_raw for a D1-D4 decision: static unit features + dynamic history + video type + bias."""
    ctx = dict(unit_features)
    ctx.update(hist)
    ctx.update(type_onehot(video_type))
    ctx["bias"] = 1.0
    return ctx


def emphasis_candidates(words: list[Word], word_feats: dict[int, dict[str, float]]) -> list[int]:
    """Indices (into `words`) of the top-3 non-stopword words by a simple tfidf + loudness heuristic."""
    scored = []
    for i, w in enumerate(words):
        wf = word_feats.get(w.id, {})
        if w.is_filler or wf.get("is_stopword", 0.0) or not any(ch.isalnum() for ch in w.text_original):
            continue
        t = w.text_original.lower().strip(".,;:!?\"'")
        if t in STOPWORDS:
            continue
        score = wf.get("tfidf_in_transcript", 0.0) + 0.15 * max(wf.get("word_loudness_z", 0.0), 0.0)
        scored.append((-score, i))
    scored.sort()
    return [i for _, i in scored[:MAX_CANDIDATES]]


def build_emphasis_context(words: list[Word], word_feats: dict[int, dict[str, float]],
                           video_type: str) -> tuple[dict[str, float], list[int]] | None:
    """(context_raw, candidate word ids) for one caption line, or None when it has no candidate word."""
    cand = emphasis_candidates(words, word_feats)
    if not cand:
        return None
    n = len(words)
    ctx: dict[str, float] = {"n_cand": float(len(cand)), "line_len": float(n)}
    ctx["line_has_number"] = float(any(word_feats.get(w.id, {}).get("is_number", 0.0) for w in words))
    ctx.update(type_onehot(video_type))
    ctx["bias"] = 1.0
    ids = []
    for k, i in enumerate(cand):
        w = words[i]
        wf = word_feats.get(w.id, {})
        for key, _ in CAND_KEYS:
            if key != "position_in_line":
                ctx[f"c{k}_{key}"] = float(wf.get(key, 0.0))
        ctx[f"c{k}_position_in_line"] = i / (n - 1) if n > 1 else 0.0
        ctx[f"c{k}_word_id"] = float(w.id)
        ids.append(w.id)
    return ctx, ids
