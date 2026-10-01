"""Filler detection (PLAN 6.3 D4, 12): lexicon + position/ambiguity rules.

A filler group is one filler token (um, uh, like, you know, so, basically, actually, i mean, ...)
located by word index. Every group is reported; ambiguity is exposed as `likely_verb_like`, and the
cut-safety rule as `clean_gap` (cut from prev word end + 20 ms to next word start - 20 ms; if either
neighbouring gap is < 30 ms the cut would clip neighbours, so the policy layer should default to KEEP).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas import Word

CUT_PAD = 0.02
MIN_CLEAN_GAP = 0.03

UM = {"um", "umm", "uhm", "ummm", "hmm", "hm", "mm", "mhm"}  # -> token "um"
UH = {"uh", "uhh", "uhhh", "er", "err", "erm", "ah", "eh"}  # -> token "uh"
SINGLE = {"like": "like", "so": "so", "basically": "basically", "actually": "actually"}
BIGRAMS = {("you", "know"): "you_know", ("i", "mean"): "i_mean"}
TOKENS = ["um", "uh", "like", "you_know", "so", "basically", "actually", "i_mean", "other"]

# "like" preceded by these is probably the verb/preposition ("I like it", "would like to", "to like")
VERB_LIKE_PREV = {"i", "you", "we", "they", "would", "don't", "dont", "do", "to", "didn't", "didnt", "doesn't",
                  "he", "she", "who", "people", "really"}
# "you know" preceded by these is a real clause ("do you know", "as you know")
KNOW_PREV = {"do", "did", "if", "as", "that", "what", "don't", "dont", "didn't", "didnt", "and", "but"}


def norm(text: str) -> str:
    t = text.lower().replace("’", "'")
    return re.sub(r"^[^\w']+|[^\w']+$", "", t)


@dataclass
class FillerGroup:
    word_idxs: list[int]  # indices into the word list (1 or 2)
    token: str  # one of TOKENS
    span_idx: int  # speech span containing it
    likely_verb_like: bool = False


def _span_lookup(spans: list[tuple[int, int]], n_words: int) -> list[int]:
    out = [-1] * n_words
    for k, (a, b) in enumerate(spans):
        for i in range(a, b):
            out[i] = k
    return out


def find_fillers(words: list[Word], spans: list[tuple[int, int]]) -> list[FillerGroup]:
    """Mark `is_filler` on words (in place) and return the filler groups in time order."""
    which = _span_lookup(spans, len(words))
    groups: list[FillerGroup] = []
    i = 0
    n = len(words)
    while i < n:
        if which[i] < 0:
            i += 1
            continue
        s0 = spans[which[i]][0]
        t = norm(words[i].text_original)
        nxt = norm(words[i + 1].text_original) if i + 1 < n and which[i + 1] == which[i] else None
        prev = norm(words[i - 1].text_original) if i > s0 else None
        prev_raw = words[i - 1].text_original if i > s0 else ""
        g: FillerGroup | None = None
        if nxt is not None and (t, nxt) in BIGRAMS:
            tok = BIGRAMS[(t, nxt)]
            if not (tok == "you_know" and prev in KNOW_PREV):
                g = FillerGroup([i, i + 1], tok, which[i])
        elif t in UM:
            g = FillerGroup([i], "um", which[i])
        elif t in UH:
            g = FillerGroup([i], "uh", which[i])
        elif t in SINGLE:
            tok = SINGLE[t]
            if tok == "so":
                # discourse "so": phrase-initial, or right after a comma (not "so much", "so that")
                if i == s0 or prev_raw.endswith(","):
                    g = FillerGroup([i], "so", which[i])
            elif tok == "like":
                g = FillerGroup([i], "like", which[i], likely_verb_like=(prev in VERB_LIKE_PREV))
            else:
                g = FillerGroup([i], tok, which[i])
        if g:
            for j in g.word_idxs:
                words[j].is_filler = True
            groups.append(g)
            i = g.word_idxs[-1] + 1
        else:
            i += 1
    return groups


def clean_gap(words: list[Word], g: FillerGroup) -> tuple[float, float, bool]:
    """(gap_before, gap_after, clean). Gaps to the neighbouring words (1.0 s sentinel at the video edges)."""
    a, b = g.word_idxs[0], g.word_idxs[-1]
    before = words[a].start - words[a - 1].end if a > 0 else 1.0
    after = words[b + 1].start - words[b].end if b + 1 < len(words) else 1.0
    return before, after, (before >= MIN_CLEAN_GAP and after >= MIN_CLEAN_GAP)
