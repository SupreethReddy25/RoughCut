"""Paths and constants. ffmpeg lookup: FFMPEG_DIR env/.env -> tools/ffmpeg/bin -> PATH."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ["ROUGHCUT_DATA_DIR"]) if os.environ.get("ROUGHCUT_DATA_DIR") else REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "db.sqlite"
ASSETS_DIR = REPO_ROOT / "backend" / "assets"

MAX_TOTAL_SECONDS = 180
MAX_TOTAL_BYTES = 500 * 1024 * 1024


def _load_dotenv() -> None:
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


def _find_tool(name: str) -> str | None:
    exe = name + (".exe" if os.name == "nt" else "")
    candidates = []
    if os.environ.get("FFMPEG_DIR"):
        candidates.append(Path(os.environ["FFMPEG_DIR"]) / exe)
    candidates.append(REPO_ROOT / "tools" / "ffmpeg" / "bin" / exe)
    for c in candidates:
        if c.exists():
            return str(c)
    return shutil.which(name)


def ffmpeg_path() -> str:
    p = _find_tool("ffmpeg")
    if not p:
        raise RuntimeError(
            "ffmpeg not found. Put it in tools/ffmpeg/bin, add it to PATH, or set FFMPEG_DIR in .env"
        )
    return p


def ffprobe_path() -> str:
    p = _find_tool("ffprobe")
    if not p:
        raise RuntimeError("ffprobe not found (see ffmpeg_path hint)")
    return p
