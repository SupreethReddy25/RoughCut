"""Speech-unit segmentation (PLAN 6.1, 12): split at pauses >= 0.25 s and sentence punctuation,
merge units < 0.4 s into a close neighbour, cap at ~6 s. Never crosses a take boundary.

Works on word index ranges: a span is (i0, i1) meaning words[i0:i1].
"""
from __future__ import annotations

import re

from app.schemas import Word

SPLIT_GAP = 0.25
MIN_LEN = 0.4
MERGE_GAP = 0.6  # only merge a short unit into a neighbour this close (else it is a real standalone phrase)
MAX_LEN = 6.0
MIN_HALF = 1.0  # when splitting an over-long unit, keep both halves at least this long
SENT_END = re.compile(r"[.?!][\"')\]]*$")
SOFT_BREAK = re.compile(r"[,;:][\"')\]]*$")


def is_sentence_end(text: str) -> bool:
    return bool(SENT_END.search(text)) or text.endswith("…")  # "..." counts: the speaker stopped there


def take_of(t: float, bounds: list[float]) -> int:
    return sum(1 for b in bounds if t >= b)


def clamp_to_takes(words: list[Word], bounds: list[float]) -> None:
    """Clip each word to the time range of the take holding its midpoint (words must not straddle a join)."""
    if not bounds:
        return
    edges = [0.0, *bounds, float("inf")]
    for w in words:
        k = take_of((w.start + w.end) / 2, bounds)
        lo, hi = edges[k], edges[k + 1]
        w.start = max(w.start, lo)
        w.end = max(min(w.end, hi), w.start + 0.02)


def _dur(words, s):
    return words[s[1] - 1].end - words[s[0]].start


def _gap(words, a, b):
    return words[b[0]].start - words[a[1] - 1].end


def _split_long(words: list[Word], s: tuple[int, int], max_len: float) -> list[tuple[int, int]]:
    if _dur(words, s) <= max_len or s[1] - s[0] < 2:
        return [s]
    i0, i1 = s
    best, best_score = None, None
    for i in range(i0 + 1, i1):  # split between words i-1 and i
        left, right = words[i - 1].end - words[i0].start, words[i1 - 1].end - words[i].start
        if left < MIN_HALF or right < MIN_HALF:
            continue
        score = (words[i].start - words[i - 1].end) + (0.1 if SOFT_BREAK.search(words[i - 1].text_original) else 0)
        if best_score is None or score > best_score:
            best, best_score = i, score
    if best is None:  # no balanced candidate: split at the middle word
        mid = (words[i0].start + words[i1 - 1].end) / 2
        best = min(range(i0 + 1, i1), key=lambda i: abs(words[i].start - mid))
    return _split_long(words, (i0, best), max_len) + _split_long(words, (best, i1), max_len)


def segment_speech(words: list[Word], take_bounds: list[float] = (), *, split_gap: float = SPLIT_GAP,
                   min_len: float = MIN_LEN, merge_gap: float = MERGE_GAP,
                   max_len: float = MAX_LEN) -> list[tuple[int, int]]:
    if not words:
        return []
    bounds = list(take_bounds)
    takes = [take_of((w.start + w.end) / 2, bounds) for w in words]
    # 1+2: split at takes, long gaps and sentence-final punctuation
    spans: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(words) + 1):
        if (i == len(words) or takes[i] != takes[i - 1]
                or (words[i].start - words[i - 1].end) >= split_gap
                or is_sentence_end(words[i - 1].text_original)):
            spans.append((start, i))
            start = i
    # 3: merge too-short units into the nearer neighbour (same take, close in time, result <= max_len)
    changed = True
    while changed and len(spans) > 1:
        changed = False
        for k, s in enumerate(spans):
            if _dur(words, s) >= min_len:
                continue
            cands = []
            for j in (k - 1, k + 1):
                if 0 <= j < len(spans) and takes[spans[j][0]] == takes[s[0]]:
                    a, b = sorted((s, spans[j]))
                    if _gap(words, a, b) <= merge_gap and _dur(words, (a[0], b[1])) <= max_len:
                        cands.append((_gap(words, a, b), j))
            if not cands:
                continue
            j = min(cands)[1]
            a, b = sorted((k, j))
            spans[a:b + 1] = [(spans[a][0], spans[b][1])]
            changed = True
            break
    # 4: cap length
    out: list[tuple[int, int]] = []
    for s in spans:
        out += _split_long(words, s, max_len)
    return out
