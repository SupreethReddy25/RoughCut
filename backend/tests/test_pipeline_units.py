"""Segmentation / pauses / fillers / retake / unit assembly on hand-made word lists (no models, no ffmpeg)."""
from __future__ import annotations

import numpy as np

from app.pipeline import fillers as fl
from app.pipeline import pauses, segment
from app.pipeline.features import build_units, content_tokens, retake_scores
from app.pipeline.pauses import Energy, build_pause_spans, energy_from_samples, refine_word_edges
from app.schemas import UnitsFile, Word


def mk(spec: list[tuple[str, float, float]]) -> list[Word]:
    return [Word(id=i, idx=i, unit_id=-1, start=s, end=e, text_original=t, text_display=t)
            for i, (t, s, e) in enumerate(spec)]


def seq(text: str, start: float = 0.0, dur: float = 0.3, gap: float = 0.05) -> list[Word]:
    """Words of `text`, each `dur` long with `gap` between them."""
    out, t = [], start
    for tok in text.split():
        out.append((tok, t, t + dur))
        t += dur + gap
    return mk(out)


def chain(*parts: tuple[str, float]) -> list[Word]:
    """Concatenate seq() pieces, (text, gap_before) each; ids are renumbered."""
    words, t = [], 0.0
    for text, gap in parts:
        w = seq(text, t + gap)
        words += w
        t = w[-1].end
    for i, w in enumerate(words):
        w.id = w.idx = i
    return words


def texts(words, spans):
    return [" ".join(w.text_original for w in words[a:b]) for a, b in spans]


# ---------------------------------------------------------------- segmentation
def test_split_at_pause_but_not_small_gap():
    w = chain(("one two three four", 0.0), ("five six seven eight", 0.3))
    assert texts(w, segment.segment_speech(w)) == ["one two three four", "five six seven eight"]
    w = chain(("one two three four", 0.0), ("five six seven eight", 0.1))
    assert len(segment.segment_speech(w)) == 1


def test_split_at_sentence_punctuation_even_without_pause():
    w = seq("I like it. Then we left early today")  # no audible gap at all
    assert texts(w, segment.segment_speech(w)) == ["I like it.", "Then we left early today"]


def test_ellipsis_is_a_sentence_end():
    assert segment.is_sentence_end("intro...") and segment.is_sentence_end("done.") and segment.is_sentence_end("why?")
    assert not segment.is_sentence_end("about,") and not segment.is_sentence_end("so")


def test_short_unit_merges_only_when_close():
    close = chain(("Okay,", 0.0), ("this is the real content here", 0.3))
    assert len(segment.segment_speech(close)) == 1
    far = chain(("Okay,", 0.0), ("this is the real content here", 1.0))
    assert len(segment.segment_speech(far)) == 2  # standalone phrase, real pause after it


def test_cap_at_six_seconds_splits_at_biggest_gap():
    w = seq("w " * 24, dur=0.3, gap=0.05)  # 24 words ~ 8.5 s, no pause >= 0.25 s
    for x in w[12:]:  # open a 0.2 s gap (< split_gap) in the middle so it is the preferred cut
        x.start += 0.15
        x.end += 0.15
    spans = segment.segment_speech(w)
    assert len(spans) == 2 and spans[0][1] == spans[1][0] == 12
    assert all(w[b - 1].end - w[a].start <= 6.0 for a, b in spans)
    # contiguous, complete coverage
    assert spans[0][0] == 0 and spans[-1][1] == 24


def test_never_crosses_take_boundary():
    w = chain(("alpha beta gamma delta", 0.0), ("epsilon zeta eta theta", 0.02))
    bound = w[3].end + 0.01
    assert len(segment.segment_speech(w)) == 1
    spans = segment.segment_speech(w, [bound])
    assert spans == [(0, 4), (4, 8)]


# ---------------------------------------------------------------- pauses
def test_pause_units_between_spans_and_take_boundary():
    w = chain(("one two three four", 0.0), ("five six seven eight", 0.12), ("nine ten eleven twelve", 0.4))
    spans = [(0, 4), (4, 8), (8, 12)]
    ps = build_pause_spans(w, spans)
    assert len(ps) == 1  # 0.12 gap < 0.15 -> no pause unit; 0.4 gap -> pause
    assert (ps[0].prev_span, ps[0].next_span) == (1, 2)
    assert abs((ps[0].end - ps[0].start) - 0.4) < 1e-6 and not ps[0].is_take_boundary
    # a take boundary is ALWAYS a pause unit, even with no audible gap
    bound = (w[3].end + w[4].start) / 2
    ps = build_pause_spans(w, spans, [bound])
    assert [(p.prev_span, p.is_take_boundary) for p in ps] == [(0, True), (1, False)]


def _tone_energy() -> Energy:
    sr = 16000
    x = np.zeros(sr * 4, dtype=np.float32)
    t = np.arange(sr) / sr
    for s in (0.0, 2.0):  # speech 0-1 s and 2-3 s, silence in between and after
        x[int(s * sr):int(s * sr) + sr] = 0.2 * np.sin(2 * np.pi * 220 * t)
    return energy_from_samples(x, sr)


def test_refine_trims_silence_off_word_edges():
    en = _tone_energy()
    # Whisper stretched A into the silence (ends 1.45) and started B early (1.6); true speech is 0-1 and 2-3
    w = mk([("hello", 0.0, 1.45), ("world", 1.6, 3.0)])
    refine_word_edges(w, en)
    assert abs(w[0].end - 1.0) <= 0.04 and abs(w[1].start - 2.0) <= 0.04
    assert w[0].start <= 0.04 and abs(w[1].end - 3.0) <= 0.04


def test_refine_restores_early_cut_word_tail():
    en = _tone_energy()
    # Whisper ended A at 0.8 but voiced audio continues to 1.0; B starts late at 2.2
    w = mk([("hello", 0.0, 0.8), ("world", 2.2, 3.0)])
    refine_word_edges(w, en)
    assert abs(w[0].end - 1.0) <= 0.04
    assert abs(w[1].start - 2.0) <= 0.04


def test_energy_threshold_and_levels():
    en = _tone_energy()
    assert en.thr >= pauses.THR_MIN_DB
    assert en.voiced_mask(0.1, 0.9).all() and not en.voiced_mask(1.2, 1.8).any()
    assert en.rel_level(0.1, 0.9) > 0.9 and en.rel_level(1.2, 1.8) < 0.05


# ---------------------------------------------------------------- fillers
def find(text: str, **kw):
    w = seq(text, **kw)
    spans = [(0, len(w))]
    return w, fl.find_fillers(w, spans)


def test_filler_lexicon_and_bigrams():
    w, g = find("so um I was uh thinking you know about it")
    toks = [(x.token, [w[i].text_original for i in x.word_idxs]) for x in g]
    assert toks == [("so", ["so"]), ("um", ["um"]), ("uh", ["uh"]), ("you_know", ["you", "know"])]
    assert [x.text_original for x in w if x.is_filler] == ["so", "um", "uh", "you", "know"]


def test_filler_so_only_at_discourse_position():
    _, g = find("it was so much fun")
    assert g == []
    w = seq("well okay, so that works")
    assert [x.token for x in fl.find_fillers(w, [(0, len(w))])] == ["so"]


def test_like_verb_vs_filler_and_you_know_clause():
    _, g = find("I like pizza")
    assert [(x.token, x.likely_verb_like) for x in g] == [("like", True)]
    _, g = find("it was like really big")
    assert [(x.token, x.likely_verb_like) for x in g] == [("like", False)]
    _, g = find("do you know the answer")
    assert g == []


def test_clean_gap_rule():
    w = mk([("it", 0.0, 0.2), ("um", 0.21, 0.5), ("was", 0.6, 0.8)])  # 10 ms before the filler, 100 ms after
    g = fl.find_fillers(w, [(0, 3)])[0]
    before, after, clean = fl.clean_gap(w, g)
    assert abs(before - 0.01) < 1e-9 and abs(after - 0.1) < 1e-9 and clean is False
    w = mk([("it", 0.0, 0.2), ("um", 0.3, 0.5), ("was", 0.6, 0.8)])
    assert fl.clean_gap(w, fl.find_fillers(w, [(0, 3)])[0])[2] is True


# ---------------------------------------------------------------- retakes
def toks(s: str) -> list[str]:
    return content_tokens(seq(s))


def test_planted_retake_scores_high_and_is_earlier():
    phrases = [
        "So the second thing is the pacing of your cuts.",
        "Fast cuts keep viewers watching for the",
        "Fast cuts keep viewers watching for much longer.",
        "Third always add captions because most people watch without sound.",
    ]
    score, earlier = retake_scores([toks(p) for p in phrases])
    assert score[1] >= 0.95 and earlier[1] == 1
    assert earlier[2] == 0 and score[2] < 0.6  # the complete retake is NOT the earlier take
    assert earlier[0] == 0 and earlier[3] == 0


def test_retake_needs_content_and_window():
    # filler-ish / tiny phrases must not match everything
    score, earlier = retake_scores([toks("okay so"), toks("okay so this is the long phrase about editing")])
    assert earlier[0] == 0 and score[0] < 0.85
    # a match 4 phrases away is outside the 3-phrase window
    p = ["red blue green yellow", "alpha beta gamma delta", "one two three four", "up down left right",
         "red blue green yellow orange"]
    score, earlier = retake_scores([toks(x) for x in p])
    assert earlier[0] == 0
    _, earlier = retake_scores([toks(x) for x in p[:4] + ["red blue green yellow orange"]], window=4)
    assert earlier[0] == 1


# ---------------------------------------------------------------- assembly
def test_build_units_structure_and_schema():
    w = chain(("um so this is take one", 0.0), ("and then the second take starts here", 1.9))
    w[0].text_original = w[0].text_display = "um,"
    bound = (w[5].end + w[6].start) / 2  # take boundary between "one" and "and"
    spans = segment.segment_speech(w, [bound])
    groups = fl.find_fillers(w, spans)
    ps = build_pause_spans(w, spans, [bound])
    uf, wf = build_units(w, spans, ps, groups, video_id="v_t", duration=10.0, video_type="vlog",
                         source_map=[{"file": "a.mp4", "src_start": 0, "src_end": 5, "out_start": 0}])
    UnitsFile.model_validate(uf.model_dump())
    kinds = [u.kind for u in uf.units]
    assert kinds == ["speech", "filler", "filler", "pause", "speech"]
    assert [u.id for u in uf.units] == list(range(len(uf.units)))
    sp0, f_um, f_so, pause, sp1 = uf.units
    assert f_um.parent_unit_id == f_so.parent_unit_id == sp0.id
    assert f_um.features["tok_um"] == 1 and f_um.features["pos_start"] == 1
    assert pause.features["is_take_boundary"] == 1 and pause.features["gap_seconds"] > 1.5
    assert pause.features["gap_bin_1.0_2.0"] == 1 and pause.features["prev_has_filler"] == 0
    assert sp0.features["starts_with_filler"] == 1 and sp0.features["is_first"] == 1
    assert sp1.features["is_last"] == 1 and sp0.features["face_present"] == 0
    assert (sp0.features["face_cx"], sp0.features["face_cy"]) == (0.5, 0.5)  # no-face default crop centre
    assert all(x.unit_id in (sp0.id, sp1.id) for x in uf.words) and len(uf.words) == len(w)
    assert all(np.isfinite(v) for u in uf.units for v in u.features.values())
    assert set(wf[0]) >= {"is_number", "tfidf_in_transcript", "word_loudness_z", "is_negation"}
