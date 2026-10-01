"""Render CLI (PLAN 14 Phase 1):  python -m app.render.run <edl.json> [--source source.mp4] [--out out.mp4]

Writes out.mp4 (1080x1920 H.264 High CRF 20, AAC 48 kHz 192 kbps) and out.srt next to the EDL by default.
Source defaults to data/processed/<video_id>/source.mp4.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from app import config
from app.pipeline.ingest import probe
from app.render import audio
from app.render.ass import FONTS_DIR, build_ass
from app.render.graph import SourceInfo, build_filter_complex
from app.render.srt import build_srt
from app.schemas import EDL


def _ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    p = subprocess.run([config.ffmpeg_path(), "-y", "-hide_banner", "-nostats", *args],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + p.stderr[-3000:])
    return p


def render(edl: EDL, source: str | Path, out_mp4: str | Path, out_srt: str | Path | None = None, *,
           denoise: str = "auto", preset: str = "fast", progress=print) -> dict:
    """Render the EDL. Returns {duration, loudness{I,LRA,TP}, denoise, loudnorm_pass1}."""
    source, out_mp4 = Path(source), Path(out_mp4)
    info = probe(source)
    src = SourceInfo(info.width, info.height)
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="roughcut_render_") as td:
        td = Path(td)
        ass_path = None
        ass_text = build_ass(edl)
        if "Dialogue:" in ass_text:
            ass_path = td / "overlay.ass"
            ass_path.write_text(ass_text, encoding="utf-8")

        enhance = edl.audio.enhance
        chain: list[str] = []
        stats: dict | None = None
        use_denoise = False
        if enhance:
            if denoise == "auto":
                wav = source.with_name("audio.wav")
                use_denoise = audio.should_denoise(audio.source_noise_floor(wav) if wav.exists() else None)
            else:
                use_denoise = denoise == "on"
            pre = audio.enhance_chain(use_denoise)
            g1, _, a1 = build_filter_complex(edl, src, audio_chain=pre + [audio.loudnorm_pass1(edl.audio.target_lufs)],
                                             video=False)
            (td / "pass1.txt").write_text(g1, encoding="utf-8")
            p1 = _ffmpeg(["-i", str(source), "-/filter_complex", str(td / "pass1.txt"), "-map", f"[{a1}]",
                          "-f", "null", "-"])
            stats = audio.parse_loudnorm_json(p1.stderr)
            progress(f"loudnorm pass 1: measured {stats['input_i']} LUFS, TP {stats['input_tp']}, "
                     f"LRA {stats['input_lra']}  (denoise={'on' if use_denoise else 'off'})")
            chain = pre + [audio.loudnorm_pass2(stats, edl.audio.target_lufs), "aresample=48000"]

        graph, vlabel, alabel = build_filter_complex(edl, src, ass_path=ass_path, fonts_dir=FONTS_DIR,
                                                     audio_chain=chain, video=True)
        (td / "graph.txt").write_text(graph, encoding="utf-8")
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        _ffmpeg(["-i", str(source), "-/filter_complex", str(td / "graph.txt"),
                 "-map", f"[{vlabel}]", "-map", f"[{alabel}]",
                 "-c:v", "libx264", "-preset", preset, "-crf", "20", "-profile:v", "high", "-pix_fmt", "yuv420p",
                 "-r", str(edl.output.fps), "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                 "-t", f"{edl.duration:.3f}",  # AAC padding would otherwise make the container ~0.1 s longer
                 "-movflags", "+faststart", str(out_mp4)])
    if out_srt:
        Path(out_srt).write_text(build_srt(edl), encoding="utf-8")
    dur = probe(out_mp4).duration
    loud = audio.measure_loudness(out_mp4)
    progress(f"rendered {out_mp4} in {time.time() - t0:.1f}s: duration {dur:.3f}s (EDL {edl.duration:.3f}s), "
             f"loudness {loud['I']:.1f} LUFS, TP {loud['TP']:.1f} dBFS, LRA {loud['LRA']:.1f}")
    return {"duration": dur, "loudness": loud, "denoise": use_denoise, "loudnorm_pass1": stats}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Render an EDL to out.mp4 + out.srt")
    ap.add_argument("edl", help="edl.json")
    ap.add_argument("--source", default=None, help="source.mp4 (default data/processed/<video_id>/source.mp4)")
    ap.add_argument("--out", default=None, help="output mp4 (default: out.mp4 next to the EDL)")
    ap.add_argument("--srt", default=None, help="output srt (default: same name as the mp4)")
    ap.add_argument("--denoise", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--preset", default="fast", help="x264 preset")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    edl_path = Path(a.edl)
    edl = EDL.model_validate_json(edl_path.read_text(encoding="utf-8"))
    source = Path(a.source) if a.source else config.PROCESSED_DIR / edl.video_id / "source.mp4"
    out = Path(a.out) if a.out else edl_path.with_name("out.mp4")
    srt = Path(a.srt) if a.srt else out.with_suffix(".srt")
    render(edl, source, out, srt, denoise=a.denoise, preset=a.preset)


if __name__ == "__main__":
    main()
