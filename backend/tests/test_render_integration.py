"""Real renders (ffmpeg + libass): rule-based EDL -> 1080x1920 mp4. Slow: `-m "not slow"` skips.

Uses data/processed/<id> if the analysis pipeline was already run, otherwise runs it into a temp dir.
"""
from __future__ import annotations

import json
import subprocess

import pytest

from app import config
from app.edl.build import BuildOptions, build_edl, load_processed
from app.edl.rules import rule_decisions
from app.pipeline.run import run_pipeline
from app.render import audio
from app.render.run import render

RAW = config.RAW_DIR
CLIPS = {
    "clip_01_main": [RAW / "clip_01_main.mp4"],
    "clip_02_take_a_clip_03_take_b": [RAW / "clip_02_take_a.mp4", RAW / "clip_03_take_b.mp4"],
}


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    def get(name: str):
        existing = config.PROCESSED_DIR / name
        if (existing / "units.json").exists() and (existing / "source.mp4").exists():
            return existing
        if not all(p.exists() for p in CLIPS[name]):
            pytest.skip("run scripts/make_test_clips.py first")
        out = tmp_path_factory.mktemp(name)
        run_pipeline(CLIPS[name], name, "vlog", out, progress=lambda *_: None)
        return out
    return get


def probe_video(path):
    p = subprocess.run([config.ffprobe_path(), "-v", "error", "-select_streams", "v:0", "-count_frames",
                        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames,codec_name,profile",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(p.stdout)["streams"][0]


@pytest.mark.slow
@pytest.mark.parametrize("name", list(CLIPS))
def test_rule_based_render(name, processed, tmp_path):
    d = processed(name)
    uf, faces, wf = load_processed(d)
    edl = build_edl(uf, rule_decisions(uf, wf), faces, BuildOptions(caption_style="bold_pop"))
    assert edl.clips and edl.captions.lines
    out = tmp_path / "out.mp4"
    res = render(edl, d / "source.mp4", out, tmp_path / "out.srt", progress=lambda *_: None)

    v = probe_video(out)
    assert (v["width"], v["height"], v["r_frame_rate"]) == (1080, 1920, "30/1")
    assert v["codec_name"] == "h264" and v["profile"] == "High"
    # acceptance: rendered duration == EDL duration +- 0.1 s (video frame count is exact by construction)
    assert abs(res["duration"] - edl.duration) <= 0.1
    assert abs(int(v["nb_read_frames"]) / 30 - edl.duration) <= 1 / 30 + 1e-6
    # acceptance: integrated loudness -14 +- 1 LUFS
    loud = audio.measure_loudness(out)
    print(f"{name}: duration {res['duration']:.3f}s (EDL {edl.duration:.3f}s), {loud['I']:.2f} LUFS, TP {loud['TP']:.2f}")
    assert abs(loud["I"] - (-14.0)) <= 1.0
    assert loud["TP"] <= -1.0
    assert (tmp_path / "out.srt").read_text(encoding="utf-8").startswith("1\n")


@pytest.mark.slow
def test_render_with_speed_and_texts(processed, tmp_path):
    from app.schemas import TextOverlay

    d = processed("clip_01_main")
    uf, faces, wf = load_processed(d)
    opt = BuildOptions(speed=1.25, caption_style="karaoke", caption_position="center",
                       texts=[TextOverlay(id="t1", text="3 editing mistakes", t0=0.0, t1=3.0)])
    edl = build_edl(uf, rule_decisions(uf, wf), faces, opt)
    res = render(edl, d / "source.mp4", tmp_path / "fast.mp4", progress=lambda *_: None)
    assert abs(res["duration"] - edl.duration) <= 0.1
    assert abs(res["loudness"]["I"] + 14.0) <= 1.0
