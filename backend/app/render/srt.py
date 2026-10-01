"""SRT export (PLAN 8.2): one cue per caption line, plain text in output time."""
from __future__ import annotations

from app.schemas import EDL


def _ts(t: float) -> str:
    ms = max(0, round(t * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(edl: EDL) -> str:
    cues = []
    for i, line in enumerate(edl.captions.lines, 1):
        text = " ".join(w.text for w in line.words)
        cues.append(f"{i}\n{_ts(line.t0)} --> {_ts(line.t1)}\n{text}\n")
    return "\n".join(cues)
