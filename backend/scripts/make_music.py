"""Synthesises the bundled background-music loops (PLAN 8.6) into backend/assets/music/.

The tracks are generated here from scratch (sines, a few filtered saws, noise percussion), so they are original works
with no third-party licence: they are released as CC0 (see assets/music/LICENSES.md). Each loop is a whole number of
bars; the tail of every note is wrapped around to the start, so the loop point is seamless.

    python backend/scripts/make_music.py

Needs numpy and the project's ffmpeg (MP3 encoding). The output is deterministic.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from app import config

OUT = BACKEND / "assets" / "music"
SR = 44100
BARS = 16


def hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12)


def adsr(n: int, a: float, d: float, s: float, r: float, hold: int | None = None) -> np.ndarray:
    """Envelope of n samples: attack a s, decay d s to level s, release r s starting at `hold` samples."""
    hold = n if hold is None else min(hold, n)
    t = np.arange(n) / SR
    env = np.where(t < a, t / max(a, 1e-4), 1.0 - (1.0 - s) * np.minimum((t - a) / max(d, 1e-4), 1.0))
    if hold < n:
        rel = np.clip(1.0 - (np.arange(n) - hold) / max(r * SR, 1.0), 0.0, 1.0)
        env = env * np.where(np.arange(n) >= hold, rel, 1.0)
    return env


def osc(kind: str, f: float, n: int, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SR
    x = 2 * np.pi * f * t + phase
    if kind == "sine":
        return np.sin(x)
    if kind == "tri":
        return 2 / np.pi * np.arcsin(np.sin(x))
    if kind == "saw":  # band-limited enough once low-passed
        return 2 * ((f * t + phase / (2 * np.pi)) % 1.0) - 1
    raise ValueError(kind)


def lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    """One-pole low-pass (cheap, smooth)."""
    a = float(np.exp(-2 * np.pi * cutoff / SR))
    return lfilter([1 - a], [1, -a], x)


class Mix:
    """A stereo buffer of `BARS` bars with wrap-around mixing (seamless loop)."""

    def __init__(self, bpm: float):
        self.beat = 60.0 / bpm
        self.n = round(BARS * 4 * self.beat * SR)
        self.buf = np.zeros((self.n, 2))

    def add(self, t: float, sig: np.ndarray, gain: float = 1.0, pan: float = 0.0) -> None:
        i = round(t * SR) % self.n
        l_gain, r_gain = gain * (1 - max(pan, 0)), gain * (1 + min(pan, 0))
        end = i + len(sig)
        first = min(len(sig), self.n - i)
        self.buf[i:i + first, 0] += sig[:first] * l_gain
        self.buf[i:i + first, 1] += sig[:first] * r_gain
        rest = len(sig) - first
        while rest > 0:  # wrap the tail to the start (a note longer than the loop wraps repeatedly)
            take = min(rest, self.n)
            seg = sig[len(sig) - rest:len(sig) - rest + take]
            self.buf[:take, 0] += seg * l_gain
            self.buf[:take, 1] += seg * r_gain
            rest -= take
        del end


def note(kind: str, midi: float, dur: float, a=0.02, d=0.1, s=0.7, r=0.2, cutoff: float | None = None,
         detune: float = 0.0) -> np.ndarray:
    n = int((dur + r) * SR)
    sig = osc(kind, hz(midi), n)
    if detune:
        sig = 0.5 * (sig + osc(kind, hz(midi) * 2 ** (detune / 1200), n, 0.7))
    if cutoff:
        sig = lowpass(sig, cutoff)
    return sig * adsr(n, a, d, s, r, int(dur * SR))


def kick() -> np.ndarray:
    n = int(0.28 * SR)
    t = np.arange(n) / SR
    f = 45 + 90 * np.exp(-t * 28)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 11)


def hat(rng: np.random.Generator, dur: float = 0.05) -> np.ndarray:
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    hp = np.diff(noise, prepend=0.0)  # crude high-pass
    return hp * np.exp(-np.arange(n) / SR * 70)


def snare(rng: np.random.Generator) -> np.ndarray:
    n = int(0.18 * SR)
    t = np.arange(n) / SR
    return (rng.standard_normal(n) * 0.6 + np.sin(2 * np.pi * 190 * t)) * np.exp(-t * 22)


MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]


def chord(root: int, degree: int, scale: list[int], size: int = 3, seventh: bool = False) -> list[int]:
    """Stacked thirds on a scale degree (0-based)."""
    steps = [0, 2, 4] + ([6] if seventh else [])
    out = []
    for s in steps[:max(size, 3)]:
        d = degree + s
        out.append(root + scale[d % 7] + 12 * (d // 7))
    return out


def render(spec: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bpm, root, scale = spec["bpm"], spec["root"], MAJOR if spec["mode"] == "major" else MINOR
    m = Mix(bpm)
    bar = 4 * m.beat
    prog = spec["prog"]  # scale degrees, one per bar, repeated
    for b in range(BARS):
        deg = prog[b % len(prog)]
        t0 = b * bar
        ch = chord(root, deg, scale, 3, spec.get("seventh", False))
        if spec.get("pad"):
            for k, midi in enumerate(ch):
                m.add(t0, note(spec["pad"], midi, bar * 0.98, a=0.35, d=0.4, s=0.8, r=0.6, cutoff=spec.get("pad_cut", 1800),
                               detune=7), 0.11, pan=(k - 1) * 0.35)
        if spec.get("bass"):
            pat = spec["bass"]
            if pat == "whole":
                m.add(t0, note("sine", ch[0] - 24, bar * 0.95, a=0.05, r=0.3), 0.30)
            elif pat == "eighths":
                for k in range(8):
                    m.add(t0 + k * m.beat / 2, note("sine", ch[0] - 24 + (12 if k % 4 == 3 else 0), m.beat * 0.45, a=0.01, d=0.08, s=0.5, r=0.08), 0.26)
            elif pat == "offbeat":
                for k in range(4):
                    m.add(t0 + (k + 0.5) * m.beat, note("sine", ch[0] - 24, m.beat * 0.4, a=0.01, r=0.1), 0.28)
        if spec.get("arp"):
            order = spec.get("arp_order", [0, 1, 2, 1])
            step = spec.get("arp_step", 0.5)
            n_steps = int(4 / step)
            for k in range(n_steps):
                midi = ch[order[k % len(order)] % len(ch)] + 12 * spec.get("arp_oct", 1)
                m.add(t0 + k * step * m.beat, note(spec["arp"], midi, step * m.beat * 0.9, a=0.005, d=0.15, s=0.2, r=0.25,
                                                   cutoff=3500 if spec["arp"] == "saw" else None), 0.12, pan=0.4 if k % 2 else -0.4)
        if spec.get("bell") and b % 2 == 0:
            for k in range(2):
                midi = ch[(b + k) % len(ch)] + 24
                m.add(t0 + (k * 2 + 1) * m.beat, note("sine", midi, 1.5, a=0.005, d=0.9, s=0.0, r=0.4), 0.08, pan=0.5 - k)
        drums = spec.get("drums")
        if drums:
            for k in range(4):
                if k in (0, 2) or drums == "four":
                    m.add(t0 + k * m.beat, kick(), 0.45 if drums != "soft" else 0.28)
                if k in (1, 3) and drums in ("full", "soft"):
                    m.add(t0 + k * m.beat, snare(rng), 0.18 if drums == "full" else 0.10)
            for k in range(8):
                m.add(t0 + k * m.beat / 2, hat(rng), 0.06 if k % 2 else 0.035, pan=0.3)
    peak = np.abs(m.buf).max()
    return m.buf / max(peak, 1e-9) * 0.8


TRACKS = [
    {"id": "sunrise", "name": "Sunrise", "mood": "calm", "bpm": 80, "root": 48, "mode": "major", "prog": [0, 4, 5, 3], "pad": "tri", "arp": "sine", "bass": "whole"},
    {"id": "lofi_drift", "name": "Lo-fi Drift", "mood": "chill", "bpm": 76, "root": 45, "mode": "minor", "prog": [0, 5, 3, 4], "seventh": True, "pad": "tri", "pad_cut": 1400, "bass": "offbeat", "drums": "soft"},
    {"id": "bright_steps", "name": "Bright Steps", "mood": "upbeat", "bpm": 112, "root": 55, "mode": "major", "prog": [0, 4, 5, 3], "arp": "tri", "arp_step": 0.5, "bass": "eighths", "drums": "full"},
    {"id": "neon_pulse", "name": "Neon Pulse", "mood": "energetic", "bpm": 120, "root": 52, "mode": "minor", "prog": [0, 0, 5, 6], "pad": "saw", "pad_cut": 1200, "arp": "saw", "arp_step": 0.25, "arp_order": [0, 2, 1, 2], "bass": "eighths", "drums": "four"},
    {"id": "quiet_hours", "name": "Quiet Hours", "mood": "mellow", "bpm": 66, "root": 50, "mode": "minor", "prog": [0, 5, 2, 6], "pad": "sine", "bell": True, "bass": "whole"},
    {"id": "coffee_break", "name": "Coffee Break", "mood": "light", "bpm": 92, "root": 53, "mode": "major", "prog": [0, 3, 4, 0], "arp": "tri", "arp_step": 1.0, "arp_order": [0, 2, 1], "arp_oct": 1, "bass": "offbeat", "drums": "soft"},
    {"id": "rise_up", "name": "Rise Up", "mood": "motivating", "bpm": 100, "root": 50, "mode": "major", "prog": [0, 5, 3, 4], "pad": "saw", "pad_cut": 1600, "bass": "eighths", "drums": "full"},
    {"id": "night_drive", "name": "Night Drive", "mood": "moody", "bpm": 104, "root": 47, "mode": "minor", "prog": [0, 5, 6, 4], "pad": "saw", "pad_cut": 900, "arp": "sine", "arp_step": 0.5, "bass": "offbeat", "drums": "four"},
]


def encode(buf: np.ndarray, path: Path) -> None:
    pcm = (np.clip(buf, -1, 1) * 32767).astype("<i2").tobytes()
    p = subprocess.run([config.ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(SR),
                        "-ac", "2", "-i", "-", "-c:a", "libmp3lame", "-b:a", "112k", "-metadata", "title=" + path.stem,
                        "-metadata", "comment=CC0, generated by make_music.py", str(path)],
                       input=pcm, capture_output=True, check=False)
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed: " + p.stderr.decode("utf-8", "replace")[-800:])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, spec in enumerate(TRACKS):
        buf = render(spec, seed=100 + i)
        encode(buf, OUT / f"{spec['id']}.mp3")
        meta.append({"id": f"lib_{spec['id']}", "name": spec["name"], "mood": spec["mood"], "bpm": spec["bpm"],
                     "file": f"{spec['id']}.mp3", "duration": round(len(buf) / SR, 2)})
        print(f"{spec['id']}: {len(buf) / SR:.1f}s")
    (OUT / "tracks.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT / "LICENSES.md").write_text(
        "# Bundled music\n\nEvery track in this folder was synthesised from scratch by `backend/scripts/make_music.py` "
        "(sine / triangle / filtered-saw oscillators and noise percussion). They contain no samples and no third-party "
        "material, so they are original works dedicated to the public domain under "
        "[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).\n\n"
        + "\n".join(f"- `{t['file']}`: {t['name']} ({t['mood']}, {t['bpm']} bpm)" for t in meta) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
