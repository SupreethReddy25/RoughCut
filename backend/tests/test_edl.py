"""Remap math + EDL builder on hand-made word lists (no models, no ffmpeg)."""
from __future__ import annotations

import pytest

from app.edl.build import PAD, BuildOptions, Decisions, build_edl, group_caption_lines
from app.edl.remap import Remap, subtract_ranges
from app.pipeline import fillers as fl
from app.pipeline import segment
from app.pipeline.features import build_units
from app.pipeline.pauses import build_pause_spans
from app.pipeline.vision import FaceSample
from app.schemas import EDL
from tests.test_pipeline_units import chain

FPS = 30


# ---------------------------------------------------------------- remap math
def test_subtract_ranges():
    assert subtract_ranges((0, 10), [(2, 3), (2.5, 4), (9, 12)]) == [(0, 2), (4, 9)]
    assert subtract_ranges((0, 10), []) == [(0, 10)]
    assert subtract_ranges((0, 10), [(0, 10)]) == []


def test_remap_with_cuts():
    r = Remap([(1.0, 3.0), (5.0, 6.0), (8.0, 10.0)])  # kept 2 + 1 + 2 = 5 s
    assert r.duration == 5.0
    assert r.src_to_out(1.0) == 0.0 and r.src_to_out(2.5) == 1.5
    assert r.src_to_out(5.0) == 2.0 and r.src_to_out(5.5) == 2.5
    assert r.src_to_out(9.0) == 4.0 and r.src_to_out(10.0) == 5.0
    # inside a cut: snap rules
    assert r.src_to_out(4.0, "next") == 2.0 and r.src_to_out(4.0, "prev") == 2.0
    assert r.src_to_out(3.5, "nearest") == 2.0 and r.src_to_out(4.6, "nearest") == 2.0
    assert r.src_to_out(0.2, "next") == 0.0  # before the first clip
    assert r.src_to_out(20, "prev") == 5.0  # after the last
    assert r.is_kept(2.0) and not r.is_kept(4.0)


def test_remap_speed_scales_output_times():
    r = Remap([(1.0, 3.0), (5.0, 6.0)], speed=1.5)
    assert r.duration == pytest.approx(2.0)
    assert r.src_to_out(5.0) == pytest.approx(2.0 / 1.5)
    assert r.src_to_out(5.75) == pytest.approx(2.75 / 1.5)


def test_remap_roundtrip():
    r = Remap([(1.0, 3.0), (5.0, 6.0), (8.0, 10.0)], speed=1.25)
    for t in (1.0, 2.2, 5.3, 8.0, 9.9):
        assert r.out_to_src(r.src_to_out(t)) == pytest.approx(t)
    assert r.out_to_src(-1) == 1.0 and r.out_to_src(1e9) == 10.0


# ---------------------------------------------------------------- builder
def make_units(parts, *, take_bound=None, fillers=True):
    w = chain(*parts)
    bounds = [take_bound] if take_bound else []
    spans = segment.segment_speech(w, bounds)
    groups = fl.find_fillers(w, spans) if fillers else []
    ps = build_pause_spans(w, spans, bounds)
    uf, _ = build_units(w, spans, ps, groups, video_id="v", duration=w[-1].end + 1.0)
    return uf, w


def ids(uf, kind):
    return [u.id for u in uf.units if u.kind == kind]


def clip_len(edl):
    return sum(c.src_end - c.src_start for c in edl.clips)


def test_keep_everything_is_one_padded_clip():
    uf, w = make_units([("one two three four", 0.5), ("five six seven eight", 0.5)])
    edl = build_edl(uf, Decisions())
    assert len(edl.clips) == 1
    c = edl.clips[0]
    assert abs(c.src_start - (w[0].start - PAD)) <= 1 / FPS and abs(c.src_end - (w[-1].end + PAD)) <= 1 / FPS
    assert edl.duration == pytest.approx(clip_len(edl), abs=1e-3)
    EDL.model_validate_json(edl.model_dump_json())


@pytest.mark.parametrize("action,target", [("REMOVE", 0.03), ("TIGHT_15", 0.15), ("TIGHT_40", 0.40)])
def test_pause_actions_retain_target_silence(action, target):
    uf, w = make_units([("one two three four", 0.5), ("five six seven eight", 1.5)])
    dec = Decisions(actions={ids(uf, "pause")[0]: action})
    edl = build_edl(uf, dec)
    assert len(edl.clips) == 2
    kept_gap = edl.clips[1].src_start - edl.clips[0].src_end  # source time dropped
    retained = (w[4].start - w[3].end) - kept_gap
    assert retained == pytest.approx(2 * PAD + target, abs=2 / FPS)
    # the cut never clips speech: both words keep >= PAD of margin (less one frame of grid snapping)
    assert edl.clips[0].src_end >= w[3].end + PAD - 1 / FPS and edl.clips[1].src_start <= w[4].start - PAD + 1 / FPS


def test_short_pause_not_stretched():
    uf, _ = make_units([("one two three four", 0.5), ("five six seven eight", 0.3)])
    edl = build_edl(uf, Decisions(actions={ids(uf, "pause")[0]: "TIGHT_40"}))
    assert len(edl.clips) == 1  # 0.3 s gap < 2*pad + 0.40: nothing to remove


def test_cut_phrase_removed_from_audio_and_captions():
    uf, w = make_units([("alpha beta gamma delta.", 0.5), ("oops this was wrong.", 0.6), ("final good take here", 0.6)])
    speech = ids(uf, "speech")
    dec = Decisions(actions={speech[1]: "CUT"})
    edl = build_edl(uf, dec)
    words = [x.text for line in edl.captions.lines for x in line.words]
    assert "oops" not in words and "wrong" not in words and "alpha" in words and "final" in words
    cut_start, cut_end = w[4].start, w[7].end
    assert not any(c.src_start < cut_end - 0.1 and c.src_end > cut_start + 0.1 for c in edl.clips)
    assert speech[1] not in {i for c in edl.clips for i in c.unit_ids}


def test_filler_cut_uses_20ms_boundaries():
    uf, w = make_units([("so I think um we should go now", 0.5)], fillers=True)
    um_unit = next(u for u in uf.units if u.kind == "filler" and u.text == "um")
    edl = build_edl(uf, Decisions(actions={um_unit.id: "REMOVE"}))
    assert len(edl.clips) == 2
    prev_w = next(x for x in w if x.text_original == "think")
    next_w = next(x for x in w if x.text_original == "we")
    assert edl.clips[0].src_end == pytest.approx(prev_w.end + 0.02, abs=1 / FPS)
    assert edl.clips[1].src_start == pytest.approx(next_w.start - 0.02, abs=1 / FPS)
    assert "um" not in [x.text for line in edl.captions.lines for x in line.words]


def test_zoom_change_splits_clip_at_gap_midpoint():
    uf, w = make_units([("one two three four.", 0.5), ("five six seven eight.", 0.1)])
    sp = ids(uf, "speech")
    edl = build_edl(uf, Decisions(zoom={sp[0]: "NONE", sp[1]: "Z115"}))
    assert [c.zoom for c in edl.clips] == [1.0, 1.15]
    mid = (w[3].end + w[4].start) / 2
    assert edl.clips[0].src_end == edl.clips[1].src_start == pytest.approx(mid, abs=1 / FPS)
    same = build_edl(uf, Decisions(zoom={sp[0]: "Z115", sp[1]: "Z115"}))
    assert len(same.clips) == 1 and same.clips[0].zoom == 1.15


def test_crop_center_median_with_fallback_to_previous_clip():
    uf, w = make_units([("one two three four", 0.5), ("five six seven eight", 2.0), ("nine ten eleven twelve", 2.0)])
    dec = Decisions(actions={p: "REMOVE" for p in ids(uf, "pause")})
    t0 = w[0].start
    faces = [FaceSample(t0 + 0.1 * i, 0.4 + 0.01 * i, 0.3, 0.2, 0.3, 0.9) for i in range(5)]  # only in clip 1
    edl = build_edl(uf, dec, faces)
    assert len(edl.clips) == 3
    assert edl.clips[0].crop_center.x == pytest.approx(0.42, abs=1e-3) and edl.clips[0].crop_center.y == 0.3
    assert (edl.clips[1].crop_center.x, edl.clips[1].crop_center.y) == (edl.clips[0].crop_center.x, 0.3)
    assert edl.clips[2].crop_center == edl.clips[0].crop_center
    no_face = build_edl(uf, dec, [])
    assert all((c.crop_center.x, c.crop_center.y) == (0.5, 0.5) for c in no_face.clips)


def test_clip_edges_on_frame_grid_and_duration_consistent():
    uf, _ = make_units([("one two three four", 0.5), ("five six seven eight", 1.5), ("nine ten eleven twelve", 1.2)])
    dec = Decisions(actions={p: "REMOVE" for p in ids(uf, "pause")})
    for speed in (1.0, 1.15, 1.25):
        edl = build_edl(uf, dec, options=BuildOptions(speed=speed))
        for c in edl.clips:
            assert c.src_start * FPS == pytest.approx(round(c.src_start * FPS), abs=1e-3)
            assert c.src_end * FPS == pytest.approx(round(c.src_end * FPS), abs=1e-3)
        assert edl.duration == pytest.approx(clip_len(edl) / speed, abs=1e-3)
        assert edl.output.speed == speed
        assert edl.duration == pytest.approx(Remap.from_edl(edl).duration, abs=1e-3)


def test_caption_times_are_output_times_and_scale_with_speed():
    uf, w = make_units([("one two three four", 0.5), ("five six seven eight", 2.0)])
    dec = Decisions(actions={ids(uf, "pause")[0]: "REMOVE"})
    e1 = build_edl(uf, dec)
    e2 = build_edl(uf, dec, options=BuildOptions(speed=2.0))
    r1 = Remap.from_edl(e1)
    five = next(x for x in w if x.text_original == "five")
    cw = next(x for line in e1.captions.lines for x in line.words if x.text == "five")
    assert cw.t0 == pytest.approx(r1.src_to_out(five.start), abs=2e-3)
    cw2 = next(x for line in e2.captions.lines for x in line.words if x.text == "five")
    assert cw2.t0 == pytest.approx(cw.t0 / 2, abs=2e-3)
    last = e1.captions.lines[-1].words[-1]
    assert last.t1 <= e1.duration + 1e-6


def test_caption_line_rules():
    def item(text, t0, t1):
        return {"raw": text, "text": text.strip(".,"), "t0": t0, "t1": t1, "emph": False}

    # <= 4 words
    items = [item(f"w{i}", i * 0.2, i * 0.2 + 0.18) for i in range(9)]
    assert [len(x) for x in group_caption_lines(items)] == [4, 4, 1]
    # <= 1.4 s
    items = [item(f"w{i}", i * 0.5, i * 0.5 + 0.45) for i in range(4)]
    assert [len(x) for x in group_caption_lines(items)] == [2, 2]
    # break at a pause (>= 0.25 s of output silence) and at sentence end
    items = [item("a", 0, 0.2), item("b", 0.2, 0.4), item("c", 0.8, 1.0), item("d.", 1.0, 1.2), item("e", 1.2, 1.4)]
    assert [[w["raw"] for w in g] for g in group_caption_lines(items)] == [["a", "b"], ["c", "d."], ["e"]]
    # a comma breaks only once the line has 2+ words
    items = [item("a,", 0, 0.2), item("b", 0.2, 0.4), item("c,", 0.4, 0.6), item("d", 0.6, 0.8)]
    assert [[w["raw"] for w in g] for g in group_caption_lines(items)] == [["a,", "b", "c,"], ["d"]]


def test_take_boundary_pause_removal_joins_clips():
    w = chain(("take one speech here", 0.5), ("take two speech here", 1.0))
    bound = (w[3].end + w[4].start) / 2
    uf, _ = make_units([("take one speech here", 0.5), ("take two speech here", 1.0)], take_bound=bound)
    tb = next(u for u in uf.units if u.kind == "pause")
    assert tb.features["is_take_boundary"] == 1
    edl = build_edl(uf, Decisions(actions={tb.id: "REMOVE"}))
    assert len(edl.clips) == 2 and edl.clips[1].src_start - edl.clips[0].src_end > 0.5
