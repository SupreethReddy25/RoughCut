"""End-to-end analysis on the synthetic clip_01 (real ffmpeg + Whisper + MiniLM). Slow: `-m "not slow"` skips."""
from __future__ import annotations

import json

import pytest

from app import config
from app.pipeline.run import run_pipeline
from app.schemas import UnitsFile

CLIP = config.RAW_DIR / "clip_01_main.mp4"


@pytest.mark.slow
@pytest.mark.skipif(not CLIP.exists(), reason="run scripts/make_test_clips.py first")
def test_clip_01_end_to_end(tmp_path):
    uf = run_pipeline([CLIP], "t_clip01", "vlog", tmp_path, progress=lambda *_: None)
    for name in ("source.mp4", "proxy.mp4", "audio.wav", "transcript.json", "embeddings.npy", "units.json"):
        assert (tmp_path / name).exists(), name
    again = UnitsFile.model_validate_json((tmp_path / "units.json").read_text(encoding="utf-8"))
    assert again.video_type == "vlog" and abs(again.duration - 66.5) < 0.3

    speech = [u for u in uf.units if u.kind == "speech"]
    pauses = [u for u in uf.units if u.kind == "pause"]
    filler = [u for u in uf.units if u.kind == "filler"]

    # planted retake: abandoned "Fast cuts keep viewers watching for the..." followed by the full take
    early = [u for u in speech if u.text.lower().startswith("fast cuts keep viewers watching for the")]
    assert len(early) == 1
    assert early[0].features["retake_score"] >= 0.85 and early[0].features["is_earlier_take"] == 1
    later = [u for u in speech if "much longer" in u.text.lower()]
    assert later and later[0].features["is_earlier_take"] == 0

    # scripted long pauses (1.1-1.9 s) are detected as pause units
    assert sum(p.end - p.start >= 1.5 for p in pauses) >= 6
    # at least a few um/uh/like/you-know made it through Whisper and were flagged
    assert len(filler) >= 5

    # synthetic clip has no face: graceful defaults
    assert all(u.features["face_present"] == 0 and u.features["face_cx"] == 0.5 for u in speech)
    emb = json.loads((tmp_path / "face_track.json").read_text())
    assert emb["faces_found"] == 0
