"""Filler detection (PLAN 6.3 D4, 12): lexicon + position/ambiguity rules.

A filler group is one filler token (um, uh, like, you know, so, basically, actually, i mean, ...)
located by word index. Every group is reported; ambiguity is exposed as `likely_verb_like`, and the
cut-safety rule as `clean_gap` (cut from prev word end + 20 ms to next word start - 20 ms). A neighbouring gap under
30 ms is still safe when the audio dips at that word boundary (fast speech has no silence, but it has valleys between
words); with no gap and no dip the cut would clip a neighbour, so the policy layer leaves the filler in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from app.schemas import Word

CUT_PAD = 0.02
MIN_CLEAN_GAP = 0.03
VALLEY_DB = 6.0  # a word boundary is cuttable when the level there is this far below the speech on both sides
VALLEY_REACH = 0.02  # s searched around the boundary for the quietest frame (the word timing is good to about this)
VALLEY_SPEECH = 0.1  # s of speech on each side the dip is compared with
SNAP_REACH = 0.08  # s a filler's boundary may move to land in a dip
MIN_WORD_KEPT = 0.04  # never shrink a word below this while moving a boundary

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
VERB_LIKE_PREV |= {"look", "looks", "looked", "feel", "feels", "felt", "seem", "seems", "sound", "sounds", "something",
                   "things", "stuff", "anything", "nothing", "more", "much"}
# hedges, mapped to the "other" token: "kind of" / "sort of" unless it names a kind ("what kind of camera")
HEDGES = {("kind", "of"), ("sort", "of")}
HEDGE_REAL_PREV = {"what", "this", "that", "the", "a", "any", "some", "which", "one", "every", "each", "same",
                   "different", "another", "other", "my", "your", "our", "their", "his", "her", "its", "these", "those"}
# sentence-initial discourse markers followed by a comma ("Well, ...", "Okay, so ...")
OPENERS = {"well", "okay", "ok", "alright", "right", "anyway", "anyways"}
# a word said twice in a row is a stutter ("I I think", "the the") unless doubling is normal English
DOUBLE_OK = {"that", "had", "very", "really", "so", "no", "yeah", "yes", "bye", "ha", "go", "now", "more", "far",
             "long", "many", "much", "is", "do", "um", "uh"}


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
        if (nxt is not None and t == nxt and t not in DOUBLE_OK and t not in SINGLE and t not in UM | UH
                and any(ch.isalpha() for ch in t) and words[i + 1].start - words[i].end < 0.5):
            # stutter: the first of the two copies goes. It is the same kind of disfluency as "uh" (no creator keeps
            # it), so it shares that token and what has been learned about it.
            g = FillerGroup([i], "uh", which[i])
        elif nxt is not None and (t, nxt) in HEDGES:
            if prev not in HEDGE_REAL_PREV:
                g = FillerGroup([i, i + 1], "other", which[i])
        elif t in OPENERS and i == s0 and words[i].text_original.rstrip().endswith(","):
            g = FillerGroup([i], "other", which[i])
        elif nxt is not None and (t, nxt) in BIGRAMS:
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


def is_valley(energy, t: float) -> bool:
    """The audio dips at time `t` compared with the speech just before and after it."""
    fs = energy.frame_s
    k = int(round(t / fs))
    r, sp = int(round(VALLEY_REACH / fs)), max(int(round(VALLEY_SPEECH / fs)), 1)
    db = energy.db
    low = db[max(k - r, 0):k + r + 1]
    left, right = db[max(k - r - sp, 0):max(k - r, 0)], db[k + r + 1:k + r + 1 + sp]
    if low.size == 0 or left.size == 0 or right.size == 0:
        return False
    return float(low.min()) <= min(float(left.max()), float(right.max())) - VALLEY_DB


def snap_to_valleys(words: list[Word], groups: list[FillerGroup], energy) -> None:
    """In place: where a filler touches its neighbour, move that shared boundary onto the audio's dip when one is close
    (word timing is good to tens of ms; the cut is only clean if it lands in the dip)."""
    fs = energy.frame_s
    reach = int(round(SNAP_REACH / fs))

    def snap(a: Word, b: Word) -> None:
        if b.start - a.end >= MIN_CLEAN_GAP:
            return
        k = int(round((a.end + b.start) / 2 / fs))
        lo, hi = max(k - reach, 0), min(k + reach + 1, len(energy.db))
        if hi <= lo:
            return
        t = (lo + int(np.argmin(energy.db[lo:hi]))) * fs + fs / 2
        if is_valley(energy, t) and a.start + MIN_WORD_KEPT < t < b.end - MIN_WORD_KEPT:
            a.end = b.start = round(t, 4)

    for g in groups:
        first, last = g.word_idxs[0], g.word_idxs[-1]
        if first > 0:
            snap(words[first - 1], words[first])
        if last + 1 < len(words):
            snap(words[last], words[last + 1])


def clean_gap(words: list[Word], g: FillerGroup, energy=None) -> tuple[float, float, bool]:
    """(gap_before, gap_after, clean). Gaps to the neighbouring words (1.0 s sentinel at the video edges). With the
    `energy` track, a boundary with no gap counts as clean when the audio dips there."""
    a, b = g.word_idxs[0], g.word_idxs[-1]
    before = words[a].start - words[a - 1].end if a > 0 else 1.0
    after = words[b + 1].start - words[b].end if b + 1 < len(words) else 1.0
    ok_before = before >= MIN_CLEAN_GAP or (energy is not None and is_valley(energy, words[a].start))
    ok_after = after >= MIN_CLEAN_GAP or (energy is not None and is_valley(energy, words[b].end))
    if b + 1 < len(words) and norm(words[b].text_original) == norm(words[b + 1].text_original):
        ok_after = True  # a stutter: what follows the cut is the same sound, so nothing can be clipped audibly
    return before, after, (ok_before and ok_after)
