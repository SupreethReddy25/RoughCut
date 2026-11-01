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
SFX_SECONDS = 0.22


@dataclass
class SourceInfo:
    width: int
    height: int


MUSIC_FADE = 1.0  # seconds of fade in at the start and fade out at the end of the music (PLAN 8.6)
DUCK_REF_DB = -19.0  # level of loud speech at the sidechain, dBFS: the voice the requested depth is calibrated for
DUCK_ATTACK_MS = 20
DUCK_RELEASE_MS = 300


@dataclass
class MusicMix:
    """Background music under the voice. The music file is input 1 of the ffmpeg command, looped (-stream_loop -1)."""

    gain_db: float
    duck_db: float
    duration: float  # output duration to fill
    start: float = 0.0  # output time the music comes in


def duck_ratio(duck_db: float) -> float:
    """Compressor ratio for a requested duck depth: 8 (PLAN 8.6) at -10 dB, gentler / harder either side."""
    return max(1.5, min(20.0, round(abs(duck_db) * 0.8, 2)))


def duck_threshold(duck_db: float) -> float:
    """Sidechain threshold (linear) that makes the compressor take `duck_db` dB off the music when the voice is at
    DUCK_REF_DB: gain reduction = (voice level - threshold) * (1 - 1/ratio). At the default -10 dB this is the PLAN's
    0.03; a quieter voice is ducked a little less, a louder one a little more."""
    r = duck_ratio(duck_db)
    return round(10 ** ((DUCK_REF_DB - abs(duck_db) / (1 - 1 / r)) / 20), 4)


def music_chain(mix: MusicMix) -> list[str]:
    """Filter-graph lines that turn input 1 + the voice [vox] into the mix [mix]."""
    d = max(mix.duration - mix.start, 0.1)
    fade = min(MUSIC_FADE, d / 2)
    lines = ["[vox]aformat=channel_layouts=stereo,asplit=2[vkey][vmix]",
             (f"[1:a]aformat=channel_layouts=stereo,aresample=48000,atrim=0:{d:.4f},asetpts=PTS-STARTPTS,"
             f"volume={mix.gain_db:.1f}dB,afade=t=in:st=0:d={fade:.3f},afade=t=out:st={d - fade:.4f}:d={fade:.3f}"
              + (f",adelay={round(mix.start * 1000)}:all=1" if mix.start > 0 else "") + "[mus]")]
    if mix.duck_db < 0:
        lines.append(f"[mus][vkey]sidechaincompress=threshold={duck_threshold(mix.duck_db)}:ratio={duck_ratio(mix.duck_db)}:"
                     f"attack={DUCK_ATTACK_MS}:release={DUCK_RELEASE_MS}[musd]")
        music = "musd"
    else:
        lines.append("[vkey]anullsink")
        music = "mus"
    lines.append(f"[vmix][{music}]amix=inputs=2:duration=first:normalize=0[mix]")
    return lines


def sfx_chain(edl: EDL) -> list[str]:
    """Soft whooshes (filtered noise, nothing to license) mixed into the voice [vox] -> [voxs]."""
    lines, labels = [], ""
    for i, cue in enumerate(edl.audio.sfx):
        lines.append(f"anoisesrc=d={SFX_SECONDS}:c=pink:a=0.3:r=48000,aformat=channel_layouts=stereo,highpass=f=600,"
                     f"lowpass=f=6000,afade=t=in:st=0:d=0.05,afade=t=out:st=0.05:d={SFX_SECONDS - 0.05:.2f},"
                     f"adelay={round(cue.t0 * 1000)}:all=1[sfx{i}]")
        labels += f"[sfx{i}]"
    lines.append(f"[vox]{labels}amix=inputs={len(edl.audio.sfx) + 1}:duration=first:normalize=0[voxs]")
    return lines


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
                         video: bool = True, music: MusicMix | None = None,
                         post_chain: list[str] | None = None) -> tuple[str, str | None, str]:
    """Returns (graph, video_label|None, audio_label). `audio_chain` filters are appended after the speed step
    (the voice chain); with `music` the voice and the music are then mixed and `post_chain` (normally the loudnorm)
    runs on the mix. Without music `post_chain` simply continues the chain, as before. video=False builds the
    audio-only graph used for loudnorm pass 1."""
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
                         f"setpts=(PTS-STARTPTS)/{c.speed:.6f},"
                         f"crop={w}:{h}:{x}:{y},scale={ow}:{oh}:flags=lanczos,setsar=1[v{i}]")
            cat += f"[v{i}]"
        parts.append(f"[0:a]atrim=start={c.src_start:.4f}:end={c.src_end:.4f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:st=0:d={fd:.4f},afade=t=out:st={dur - fd:.4f}:d={fd:.4f}"
                     + (f",atempo={c.speed:.6f}" if abs(c.speed - 1.0) > 1e-9 else "") + f"[a{i}]")
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
    sfx = sfx_chain(edl) if edl.audio.sfx else []
    if music is None:
        if sfx:
            parts.append("[ac]" + (",".join(achain) if achain else "anull") + "[vox]")
            parts += sfx
            parts.append("[voxs]" + (",".join(post_chain) if post_chain else "anull") + "[ao]")
        else:
            achain += post_chain or []
            parts.append("[ac]" + (",".join(achain) if achain else "anull") + "[ao]")
    else:
        parts.append("[ac]" + (",".join(achain) if achain else "anull") + "[vox]")
        if sfx:
            parts += sfx
            parts.append("[voxs]anull[vox2]")
            parts += [ln.replace("[vox]", "[vox2]", 1) for ln in music_chain(music)]
        else:
            parts += music_chain(music)
        parts.append("[mix]" + (",".join(post_chain) if post_chain else "anull") + "[ao]")
    return ";\n".join(parts), vlabel, "ao"
