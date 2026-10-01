"""Golden fixtures for the frontend's parity tests: the browser re-implements the EDL time maths (remap) and the
crop window (crop_rect) so the preview matches the export; these fixtures are what the Python code returns, and
`frontend/src/lib/remap.test.ts` / `crop.test.ts` assert the TypeScript returns the same numbers.

    python backend/scripts/make_frontend_fixtures.py        (backend/tests/test_frontend_fixtures.py checks it is current)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.edl.remap import Remap
from app.render.graph import SourceInfo, crop_rect
from app.schemas import Clip, Point

OUT = Path(__file__).resolve().parents[2] / "frontend" / "src" / "lib" / "fixtures" / "parity.json"

REMAP_CASES = [
    {"ranges": [[1.0, 3.0], [5.0, 6.0], [8.0, 10.0]], "speed": 1.0},
    {"ranges": [[1.0, 3.0], [5.0, 6.0], [8.0, 10.0]], "speed": 1.25},
    {"ranges": [[0.0, 2.5]], "speed": 1.15},
    {"ranges": [[0.48333, 1.41667], [1.58333, 3.3], [4.28333, 6.975]], "speed": 1.1},
]
SRC_PROBES = [-1.0, 0.0, 0.2, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.6, 5.0, 5.5, 6.0, 7.0, 9.0, 10.0, 20.0]
OUT_PROBES = [-1.0, 0.0, 0.3, 1.0, 1.5, 2.0, 3.0, 4.0, 100.0]

CLIPS = [
    {"zoom": 1.0, "cx": 0.5, "cy": 0.5}, {"zoom": 1.15, "cx": 0.5, "cy": 0.5}, {"zoom": 1.3, "cx": 0.2, "cy": 0.7},
    {"zoom": 1.0, "cx": 0.0, "cy": 0.0}, {"zoom": 1.15, "cx": 1.0, "cy": 1.0}, {"zoom": 1.3, "cx": 0.3123, "cy": 0.4177},
]
SOURCES = [(1920, 1080), (1080, 1920), (1280, 720), (1080, 2400), (854, 480), (1440, 1080), (2160, 3840)]


def build() -> dict:
    remap = []
    for case in REMAP_CASES:
        r = Remap([tuple(x) for x in case["ranges"]], case["speed"])
        remap.append({
            **case, "duration": r.duration,
            "src_to_out": [{"t": t, "snap": s, "out": r.src_to_out(t, s)} for t in SRC_PROBES
                           for s in ("nearest", "next", "prev")],
            "out_to_src": [{"o": o, "src": r.out_to_src(o)} for o in OUT_PROBES],
            "kept": [{"t": t, "kept": r.is_kept(t)} for t in SRC_PROBES],
        })
    crop = []
    for c in CLIPS:
        for w, h in SOURCES:
            rect = crop_rect(Clip(src_start=0, src_end=1, zoom=c["zoom"], crop_center=Point(x=c["cx"], y=c["cy"])),
                             SourceInfo(w, h), 1080, 1920)
            crop.append({**c, "src": [w, h], "out": [1080, 1920], "rect": list(rect)})
    return {"remap": remap, "crop": crop}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1), encoding="utf-8", newline="\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
