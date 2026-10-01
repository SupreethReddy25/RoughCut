"""Rule-based EDL CLI (testing aid).

    python -m app.edl.run data/processed/clip_01_main [--speed 1.1] [--style bold_pop] [--out edl.json]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app import config
from app.edl.build import BuildOptions, build_edl, load_processed
from app.edl.rules import rule_decisions


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build a rule-based EDL from a processed video folder")
    ap.add_argument("video", help="data/processed/<id> folder (or just the id)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--style", default="bold_pop", choices=["bold_pop", "clean", "karaoke", "minimal"])
    ap.add_argument("--position", default="lower_third", choices=["lower_third", "center", "top"])
    ap.add_argument("--out", default=None, help="edl.json path (default: <video folder>/edl.json)")
    a = ap.parse_args(argv)
    d = Path(a.video)
    if not d.is_dir():
        d = config.PROCESSED_DIR / a.video
    uf, faces, wf = load_processed(d)
    edl = build_edl(uf, rule_decisions(uf, wf), faces,
                    BuildOptions(speed=a.speed, caption_style=a.style, caption_position=a.position))
    out = Path(a.out) if a.out else d / "edl.json"
    out.write_text(edl.model_dump_json(indent=1), encoding="utf-8")
    print(f"{out}: {len(edl.clips)} clips, {len(edl.captions.lines)} caption lines, "
          f"duration {edl.duration:.2f}s (source {uf.duration:.2f}s)")


if __name__ == "__main__":
    main()
