"""Background-music library (PLAN 8.6): the bundled tracks (assets/music) plus a user's own uploads.

Track ids: `lib_<name>` for a bundled track, `up_<user id>_<hash>` for an upload. An upload is only visible to the user
who made it. Files live in assets/music/ and data/music/u<user id>/.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path

from app import config

LIB_DIR = Path(__file__).resolve().parents[1] / "assets" / "music"
UPLOAD_EXTS = (".mp3", ".wav")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MIN_SECONDS = 3.0
MAX_SECONDS = 15 * 60.0
_ID = re.compile(r"^(lib_[A-Za-z0-9_]+|up_(\d+)_[0-9a-f]{8})$")


class MusicError(ValueError):
    """A bad upload or an unknown track (the API turns it into a 4xx)."""


@lru_cache(maxsize=1)
def library() -> tuple[dict, ...]:
    path = LIB_DIR / "tracks.json"
    if not path.exists():
        return ()
    return tuple({**t, "source": "library"} for t in json.loads(path.read_text(encoding="utf-8"))
                 if (LIB_DIR / t["file"]).exists())


def user_dir(data_dir: Path, user_id: int) -> Path:
    return data_dir / "music" / f"u{user_id}"


def _uploads(data_dir: Path, user_id: int) -> list[dict]:
    out = []
    for meta in sorted(user_dir(data_dir, user_id).glob("up_*.json")):
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (meta.parent / m.get("file", "")).exists():
            out.append({**m, "source": "upload"})
    return out


def list_tracks(data_dir: Path, user_id: int) -> list[dict]:
    """Every track the user can pick: bundled first, then their uploads. {id, name, mood, duration, source}."""
    return [{"id": t["id"], "name": t["name"], "mood": t.get("mood", ""), "duration": t["duration"], "source": t["source"]}
            for t in [*library(), *_uploads(data_dir, user_id)]]


def track_path(data_dir: Path, user_id: int, track_id: str) -> Path | None:
    """The audio file of a track the user may use, or None (unknown id, someone else's upload, file gone)."""
    m = _ID.match(track_id or "")
    if m is None:
        return None
    if track_id.startswith("lib_"):
        t = next((t for t in library() if t["id"] == track_id), None)
        return LIB_DIR / t["file"] if t else None
    if int(m.group(2)) != user_id:
        return None
    meta = user_dir(data_dir, user_id) / f"{track_id}.json"
    try:
        f = json.loads(meta.read_text(encoding="utf-8"))["file"]
    except (OSError, ValueError, KeyError):
        return None
    p = meta.parent / f
    return p if p.exists() else None


def _probe_audio(path: Path) -> float:
    p = subprocess.run([config.ffprobe_path(), "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
                        str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    try:
        info = json.loads(p.stdout)
        if not any(s.get("codec_type") == "audio" for s in info["streams"]):
            raise MusicError("the file has no audio")
        dur = float(info["format"]["duration"])
    except (ValueError, KeyError, TypeError):
        raise MusicError("this is not a readable audio file") from None
    return dur


def save_upload(data_dir: Path, user_id: int, filename: str, data: bytes) -> dict:
    """Validate and store an uploaded mp3 / wav; returns its track entry."""
    ext = Path(filename or "").suffix.lower()
    if ext not in UPLOAD_EXTS:
        raise MusicError(f"only {', '.join(UPLOAD_EXTS)} files are supported")
    if not data:
        raise MusicError("the file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise MusicError(f"the file is larger than {MAX_UPLOAD_BYTES // 2**20} MB")
    tid = f"up_{user_id}_{hashlib.sha1(data).hexdigest()[:8]}"
    d = user_dir(data_dir, user_id)
    d.mkdir(parents=True, exist_ok=True)
    audio = d / f"{tid}{ext}"
    tmp = d / f"{tid}.tmp{ext}"
    tmp.write_bytes(data)
    try:
        dur = _probe_audio(tmp)
        if not MIN_SECONDS <= dur <= MAX_SECONDS:
            raise MusicError(f"the track must be between {MIN_SECONDS:.0f} seconds and {MAX_SECONDS / 60:.0f} minutes long")
        tmp.replace(audio)
    finally:
        tmp.unlink(missing_ok=True)
    name = Path(filename).stem[:60] or "My track"
    entry = {"id": tid, "name": name, "mood": "", "duration": round(dur, 2), "file": audio.name}
    (d / f"{tid}.json").write_text(json.dumps(entry), encoding="utf-8")
    return {"id": tid, "name": name, "mood": "", "duration": entry["duration"], "source": "upload"}
