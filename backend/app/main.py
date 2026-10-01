from __future__ import annotations

import subprocess

from fastapi import FastAPI

from . import config

app = FastAPI(title="RoughCut")


def _ffmpeg_status() -> dict:
    try:
        out = subprocess.run(
            [config.ffmpeg_path(), "-hide_banner", "-filters"], capture_output=True, text=True, timeout=20, check=False
        ).stdout
        needed = ["subtitles", "sidechaincompress", "loudnorm", "afftdn", "ebur128"]
        return {"ok": True, "path": config.ffmpeg_path(), "missing_filters": [f for f in needed if f" {f} " not in out]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "ffmpeg": _ffmpeg_status()}
