"""ffmpeg filter graph for an EDL (PLAN 12 "Render" row). The source is CFR at output fps (ingest guarantees it).

Per clip: trim/atrim + 8 ms afade in/out + 9:16 crop at crop_center (window shrunk by zoom) taken from the
SOURCE (not the proxy) + scale to 1080x1920 -> concat -> speed (setpts / atempo) -> ASS burn-in -> audio chain.
The graph is written to a file and passed with `-/filter_complex`, so clip count is not bounded by the Windows
command-line length.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.schemas import EDL, Clip

FADE = 0.008  # 8 ms at every cut (PLAN 8.3)


@dataclass
class SourceInfo:
    width: int
    height: int


def _even(x: float, lo: int = 2) -> int:
    return max(lo, round(x / 2.0) * 2)


def crop_rect(clip: Clip, src: SourceInfo, out_w: int, out_h: int) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the crop window in source pixels. The window has the output aspect ratio, fills the
    frame at zoom 1 (portrait sources: nothing is cropped away), shrinks by `zoom`, is centred on crop_center
    and clamped to stay in-frame."""
    ar = out_w / out_h
    bw, bh = src.height * ar, float(src.height)  # fit by height ...
    if bw > src.width:  # ... unless the source is narrower than the output aspect: fit by width
        bw, bh = float(src.width), src.width / ar
    z = max(clip.zoom, 1.0)
    cw, ch = min(_even(bw / z), src.width), min(_even(bh / z), src.height)
    x = min(max(clip.crop_center.x * src.width - cw / 2, 0), src.width - cw)
    y = min(max(clip.crop_center.y * src.height - ch / 2, 0), src.height - ch)
    return int(x) // 2 * 2, int(y) // 2 * 2, cw, ch


def fpath(p: str | Path) -> str:
    """A path as a quoted filtergraph option value (forward slashes, drive colon escaped)."""
    return "'" + str(p).replace("\\", "/").replace(":", "\\:") + "'"


def build_filter_complex(edl: EDL, src: SourceInfo, *, ass_path: str | Path | None = None,
                         fonts_dir: str | Path | None = None, audio_chain: list[str] | None = None,
                         video: bool = True) -> tuple[str, str | None, str]:
    """Returns (graph, video_label|None, audio_label). `audio_chain` filters are appended after the speed step
    (it normally ends with a loudnorm). video=False builds the audio-only graph used for loudnorm pass 1."""
    if not edl.clips:
        raise ValueError("EDL has no clips")
    ow, oh, fps, speed = edl.output.width, edl.output.height, edl.output.fps, edl.output.speed
    parts, cat = [], ""
    for i, c in enumerate(edl.clips):
        dur = c.src_end - c.src_start
        fd = min(FADE, dur / 2)
        if video:
            x, y, w, h = crop_rect(c, src, ow, oh)
            # frame-index trim: exact frame counts (clip edges are on the 30 fps grid, see edl/build.py)
            parts.append(f"[0:v]trim=start_frame={round(c.src_start * fps)}:end_frame={round(c.src_end * fps)},"
                         f"setpts=PTS-STARTPTS,"
                         f"crop={w}:{h}:{x}:{y},scale={ow}:{oh}:flags=lanczos,setsar=1[v{i}]")
            cat += f"[v{i}]"
        parts.append(f"[0:a]atrim=start={c.src_start:.4f}:end={c.src_end:.4f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:st=0:d={fd:.4f},afade=t=out:st={dur - fd:.4f}:d={fd:.4f}[a{i}]")
        cat += f"[a{i}]"
    n = len(edl.clips)
    parts.append(f"{cat}concat=n={n}:v={1 if video else 0}:a=1" + ("[vc][ac]" if video else "[ac]"))

    vlabel = None
    if video:
        chain = [f"setpts=PTS/{speed:.6f}"] if abs(speed - 1.0) > 1e-9 else []
        chain.append(f"fps={fps}")
        if ass_path is not None:
            ass = f"ass=filename={fpath(ass_path)}" + (f":fontsdir={fpath(fonts_dir)}" if fonts_dir else "")
            chain.append(ass)
        parts.append("[vc]" + ",".join(chain) + "[vo]")
        vlabel = "vo"
    achain = []
    if abs(speed - 1.0) > 1e-9:
        achain.append(f"atempo={speed:.6f}")
    achain += audio_chain or []
    parts.append("[ac]" + (",".join(achain) if achain else "anull") + "[ao]")
    return ";\n".join(parts), vlabel, "ao"
