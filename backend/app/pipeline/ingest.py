"""Ingest (PLAN 8.9, 12): probe -> normalize (CFR 30 fps, rotation honored, 48 kHz) -> join -> proxy + wav.

Writes into out_dir:  source.mp4, proxy.mp4 (720p-ish, -g 15), audio.wav (16 kHz mono).
Multi-clip join is a single ffmpeg concat-filter pass so every input is normalized and joined without
re-timestamping artefacts; `source_map` records where each input landed in source.mp4.
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from app import config

log = logging.getLogger(__name__)

FPS = 30
AUDIO_RATE = 48000
PROXY_SHORT_SIDE = 720


@dataclass
class ProbeInfo:
    path: Path
    width: int  # after rotation
    height: int
    duration: float  # max(video, audio) stream duration
    fps: float
    has_audio: bool
    rotation: int = 0
    nb_frames: int | None = None  # video frames, from the container header (exact for our CFR mp4s)


@dataclass
class IngestResult:
    source: Path
    proxy: Path
    wav: Path
    duration: float
    width: int
    height: int
    source_map: list[dict] = field(default_factory=list)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    log.debug("run: %s", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-2000:]}")
    return p


def _rotation(stream: dict) -> int:
    rot = stream.get("tags", {}).get("rotate")
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = sd["rotation"]
    try:
        return round(float(rot)) % 360 if rot is not None else 0
    except (TypeError, ValueError):
        return 0


def _frac(s: str) -> float:
    try:
        n, d = s.split("/")
        return float(n) / float(d) if float(d) else 0.0
    except (ValueError, AttributeError):
        return 0.0


def probe(path: str | Path) -> ProbeInfo:
    path = Path(path)
    p = _run([config.ffprobe_path(), "-v", "error", "-print_format", "json", "-show_streams",
              "-show_format", str(path)])
    info = json.loads(p.stdout)
    vs = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if vs is None:
        raise ValueError(f"{path}: no video stream")
    aus = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    rot = _rotation(vs)
    w, h = int(vs["width"]), int(vs["height"])
    if rot in (90, 270):
        w, h = h, w
    durs = [float(s["duration"]) for s in (vs, aus) if s and s.get("duration") not in (None, "N/A")]
    if not durs and info.get("format", {}).get("duration"):
        durs = [float(info["format"]["duration"])]
    if not durs:
        raise ValueError(f"{path}: cannot determine duration")
    try:
        nb = int(vs["nb_frames"])
    except (KeyError, ValueError):
        nb = None
    return ProbeInfo(path, w, h, max(durs), _frac(vs.get("avg_frame_rate", "0/1")), aus is not None, rot, nb)


def _even(x: int) -> int:
    return max(2, x - (x % 2))


def ingest(inputs: list[str | Path], out_dir: str | Path) -> IngestResult:
    """Normalize + join `inputs` into out_dir. Returns paths, duration and the join map."""
    if not inputs:
        raise ValueError("no inputs")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    infos = [probe(p) for p in inputs]
    total = sum(i.duration for i in infos)
    total_bytes = sum(Path(i.path).stat().st_size for i in infos)
    if total > config.MAX_TOTAL_SECONDS:
        raise ValueError(f"total duration {total:.0f}s exceeds limit {config.MAX_TOTAL_SECONDS}s")
    if total_bytes > config.MAX_TOTAL_BYTES:
        raise ValueError("total size exceeds limit")

    W, H = _even(infos[0].width), _even(infos[0].height)  # all clips conform to the first clip's frame
    cmd = [config.ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error"]
    for i in infos:
        cmd += ["-i", str(i.path)]
    parts, concat_in = [], ""
    n_in = len(infos)
    extra = n_in  # next input index for generated silence
    for k, i in enumerate(infos):
        parts.append(
            f"[{k}:v]fps={FPS},scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v{k}]"
        )
        if i.has_audio:
            parts.append(f"[{k}:a]aresample={AUDIO_RATE}:async=1,"
                         f"aformat=sample_fmts=fltp:channel_layouts=stereo[a{k}]")
        else:  # silent clip: synthesize audio of the same length
            cmd += ["-f", "lavfi", "-t", f"{i.duration:.3f}", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo"]
            parts.append(f"[{extra}:a]aformat=sample_fmts=fltp:channel_layouts=stereo[a{k}]")
            extra += 1
        concat_in += f"[v{k}][a{k}]"
    parts.append(f"{concat_in}concat=n={n_in}:v=1:a=1[v][a]")
    source = out_dir / "source.mp4"
    cmd += ["-filter_complex", ";".join(parts), "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-g", str(FPS), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", str(AUDIO_RATE), "-movflags", "+faststart", str(source)]
    _run(cmd)

    # source_map: where each input landed (durations snapped to the 30 fps grid)
    smap, t = [], 0.0
    for i in infos:
        d = round(i.duration * FPS) / FPS
        smap.append({"file": i.path.name, "src_start": 0.0, "src_end": round(d, 4), "out_start": round(t, 4)})
        t += d
    final = probe(source)
    if abs(final.duration - t) > 0.15:
        log.warning("joined duration %.3f differs from source_map total %.3f", final.duration, t)

    # proxy: ~720p short side, short GOP for responsive seeking in the browser
    proxy = out_dir / "proxy.mp4"
    vf = []
    if min(W, H) > PROXY_SHORT_SIDE:
        vf = ["-vf", f"scale=-2:{PROXY_SHORT_SIDE}" if W >= H else f"scale={PROXY_SHORT_SIDE}:-2"]
    _run([config.ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), *vf,
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-g", "15", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(proxy)])

    wav = out_dir / "audio.wav"
    _run([config.ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-vn",
          "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])
    # source.mp4 is CFR 30 fps, so its length is its frame count / 30 (stream/container durations can overshoot by
    # a few frames, and a clip trimmed past the last real frame would render short)
    duration = round(final.nb_frames / FPS, 4) if final.nb_frames else final.duration
    return IngestResult(source, proxy, wav, duration, W, H, smap)
