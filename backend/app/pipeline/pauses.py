"""Energy track, word-edge refinement and pause spans (PLAN 6.1, 12).

Whisper word timestamps routinely stretch words into adjacent silence. We compute 20 ms RMS frames,
pick a per-video silence threshold from percentiles, and snap word edges to the actual speech/silence
transition (only by trimming silence off the edges, or by extending at most `tail_ext` of voiced tail).
"""
from __future__ import annotations

import math
import wave
from dataclasses import dataclass

import numpy as np

from app.schemas import Word

FRAME_S = 0.02
MIN_PAUSE = 0.15  # word gap that counts as a pause unit (PLAN 6.1)
MIN_KEEP = 0.04  # never trim a word below this
TAIL_EXT = 0.4  # max contiguous voiced run we add back onto a word edge (Whisper ends words ~0.2-0.3 s early before pauses)
THR_FRAC = 0.3  # threshold = floor + THR_FRAC * (speech - floor)
THR_MIN_DB = -60.0  # never call anything quieter than this "voiced" (digital-silence clips)
DB_FLOOR = -90.0


@dataclass
class Energy:
    rms: np.ndarray  # linear RMS per frame
    db: np.ndarray
    frame_s: float
    thr: float  # silence threshold (dB): db < thr is silent
    floor_db: float  # noise-floor estimate (p10)
    speech_db: float  # typical speech level (p90)

    @property
    def voiced(self) -> np.ndarray:
        return self.db >= self.thr

    def span(self, t0: float, t1: float) -> slice:
        """Frames whose centre lies in [t0, t1)."""
        a = max(0, math.ceil(t0 / self.frame_s - 0.5))
        b = max(a, math.ceil(t1 / self.frame_s - 0.5))
        return slice(a, min(b, len(self.db)))

    def level_db(self, t0: float, t1: float) -> float:
        r = self.rms[self.span(t0, t1)]
        if r.size == 0:
            return DB_FLOOR
        return float(max(DB_FLOOR, 20 * math.log10(math.sqrt(float(np.mean(r.astype(np.float64) ** 2))) + 1e-9)))

    def level_db_words(self, words: list[Word]) -> float:
        """Level over the frames inside the given words only (excludes the silence between them)."""
        parts = [self.rms[self.span(w.start, w.end)] for w in words]
        r = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        if r.size == 0:
            return DB_FLOOR
        return float(max(DB_FLOOR, 20 * math.log10(math.sqrt(float(np.mean(r.astype(np.float64) ** 2))) + 1e-9)))

    def rel_level(self, t0: float, t1: float) -> float:
        """0 = noise floor, 1 = typical speech level."""
        d = max(self.speech_db - self.floor_db, 1e-6)
        return float(np.clip((self.level_db(t0, t1) - self.floor_db) / d, 0.0, 1.5))

    def voiced_mask(self, t0: float, t1: float) -> np.ndarray:
        return self.voiced[self.span(t0, t1)]


def load_wav(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError("expected 16-bit wav")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def energy_from_rms(rms: np.ndarray, frame_s: float = FRAME_S) -> Energy:
    """Energy track from per-frame linear RMS (silence threshold from the per-video percentiles)."""
    rms = np.asarray(rms, dtype=np.float64)
    db = np.maximum(20 * np.log10(rms + 1e-9), DB_FLOOR)
    if len(rms) == 0:
        return Energy(rms.astype(np.float32), db, frame_s, THR_MIN_DB, DB_FLOOR, DB_FLOOR + 40)
    lo, hi = float(np.percentile(db, 10)), float(np.percentile(db, 90))
    thr = max(lo + THR_FRAC * (hi - lo), lo + 6.0, THR_MIN_DB)
    return Energy(rms.astype(np.float32), db, frame_s, thr, lo, hi)


def energy_from_samples(x: np.ndarray, sr: int, frame_s: float = FRAME_S) -> Energy:
    n = round(sr * frame_s)
    nf = len(x) // n
    fr = x[: nf * n].reshape(nf, n).astype(np.float64)
    return energy_from_rms(np.sqrt((fr ** 2).mean(axis=1)) if nf else np.zeros(0), frame_s)


def energy_from_wav(path) -> Energy:
    x, sr = load_wav(path)
    return energy_from_samples(x, sr)


# ------------------------------------------------------------------ word-edge refinement
def _trim_end(t: float, floor_t: float, en: Energy, ceil_t: float) -> float:
    """New end for a word currently ending at t: drop trailing silence, else add a short voiced tail."""
    fs, v = en.frame_s, en.voiced
    k = int(t // fs) - 1  # last frame fully before t
    if k < 0 or k >= len(v):
        return t
    if not v[k]:
        while k >= 0 and k * fs >= floor_t and not v[k]:
            k -= 1
        return min(t, max((k + 1) * fs, floor_t))
    k += 1  # frame containing/after t
    new = t
    while k < len(v) and v[k] and (k + 1) * fs <= min(t + TAIL_EXT, ceil_t):
        new = (k + 1) * fs
        k += 1
    return new


def _trim_start(t: float, ceil_t: float, en: Energy, floor_t: float) -> float:
    """New start for a word currently starting at t: drop leading silence, else add a short voiced lead."""
    fs, v = en.frame_s, en.voiced
    k = math.ceil(t / fs)  # first frame fully after t
    if k >= len(v):
        return t
    if not v[k]:
        while k < len(v) and (k + 1) * fs <= ceil_t and not v[k]:
            k += 1
        return max(t, min(k * fs, ceil_t))
    k -= 1
    new = t
    while k >= 0 and v[k] and k * fs >= max(t - TAIL_EXT, floor_t):
        new = k * fs
        k -= 1
    return new


def refine_word_edges(words: list[Word], en: Energy) -> None:
    """In place. Snap each word's edges to the audio's speech/silence transitions."""
    n = len(words)
    for i, w in enumerate(words):
        prev_end = words[i - 1].end if i else 0.0
        next_start = words[i + 1].start if i + 1 < n else float("inf")
        w.start = min(_trim_start(w.start, w.end - MIN_KEEP, en, prev_end), w.end - MIN_KEEP)
        w.end = max(_trim_end(w.end, w.start + MIN_KEEP, en, next_start), w.start + MIN_KEEP)
        w.start, w.end = round(w.start, 4), round(w.end, 4)


# ------------------------------------------------------------------ pause spans
@dataclass
class PauseSpan:
    start: float
    end: float
    prev_span: int  # index into the speech-span list
    next_span: int
    is_take_boundary: bool = False


def build_pause_spans(words: list[Word], spans: list[tuple[int, int]], take_bounds: list[float] = (),
                      min_gap: float = MIN_PAUSE) -> list[PauseSpan]:
    """Pause unit between consecutive speech spans when the gap >= min_gap; always at a take boundary."""
    out: list[PauseSpan] = []
    for k in range(len(spans) - 1):
        gs = words[spans[k][1] - 1].end
        ge = words[spans[k + 1][0]].start
        boundary = any(gs - 1e-6 <= t <= ge + 1e-6 for t in take_bounds)
        if (ge - gs) >= min_gap or boundary:
            out.append(PauseSpan(gs, max(ge, gs), k, k + 1, boundary))
    return out
