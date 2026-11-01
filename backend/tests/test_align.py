"""Forced alignment (Viterbi over CTC frames, no model needed) and the hallucination filter."""
from __future__ import annotations

import numpy as np

from app.pipeline import align, pauses, transcribe
from app.schemas import Word


def _word(i, text, s, e):
    return Word(id=i, idx=i, unit_id=-1, start=s, end=e, text_original=text, text_display=text)


def _emissions(path: list[int], vocab: int = 4) -> np.ndarray:
    em = np.full((len(path), vocab), np.log(0.02))
    em[np.arange(len(path)), path] = np.log(0.94)
    return em


def test_viterbi_finds_the_token_frames_through_blanks():
    em = _emissions([0, 0, 1, 1, 0, 0, 0, 2, 0])  # blank = 0; tokens 1 and 2
    states, is_tok, score = align.viterbi(em, [1, 2], blank=0)
    assert [t for t, x in enumerate(is_tok) if x and states[t] == 1] == [2, 3]
    assert [t for t, x in enumerate(is_tok) if x and states[t] == 2] == [7]
    assert score > align.MIN_BLOCK_SCORE


def test_viterbi_refuses_more_tokens_than_frames():
    assert align.viterbi(_emissions([0, 1]), [1, 2, 3], blank=0) is None
    assert align.viterbi(_emissions([0, 1]), [], blank=0) is None


def test_blocks_split_at_long_gaps():
    w = [_word(0, "a", 0, 1), _word(1, "b", 1, 2), _word(2, "c", 3.5, 4)]
    assert align._blocks(w) == [(0, 2), (2, 3)]


def test_alignment_off_or_failing_keeps_whisper_timing(monkeypatch, tmp_path):
    w = [_word(0, "hello", 0.5, 1.0)]
    monkeypatch.setenv("ROUGHCUT_ALIGN", "0")
    assert align.align_words(w, tmp_path / "none.wav")[0].start == 0.5
    monkeypatch.setenv("ROUGHCUT_ALIGN", "1")
    assert align.align_words(w, tmp_path / "missing.wav")[0].start == 0.5  # no audio: never raises


def test_words_over_silence_are_dropped_and_renumbered():
    sr = 16000
    x = np.zeros(sr * 3, dtype=np.float32)
    x[:sr] = 0.3 * np.sin(np.arange(sr) * 0.1)  # voice in the first second only
    en = pauses.energy_from_samples(x, sr)
    w = [_word(0, "real", 0.2, 0.6), _word(1, "ghost", 2.0, 2.5), _word(2, "words", 0.7, 0.9)]
    kept = transcribe.drop_hallucinations(w, en)
    assert [k.text_original for k in kept] == ["real", "words"] and [k.id for k in kept] == [0, 1]


def _word(i, text, start, end):
    return Word(id=i, idx=i, unit_id=-1, start=start, end=end, text_original=text, text_display=text, conf=1.0)


def test_find_disfluencies_recovers_only_stutters_and_hesitations():
    words = [_word(0, "I", 0.0, 0.2), _word(1, "think", 0.3, 0.6), _word(2, "this", 0.7, 0.9)]
    heard = [("I", 0, 2, -0.01), ("I", 10, 12, -0.01), ("THINK", 20, 30, -0.01), ("UM", 32, 40, -0.1),
             ("THIS", 45, 50, -0.01), ("BANANA", 55, 60, -0.01)]
    found = align.find_disfluencies(heard, words, 0.0)
    assert [(i, t) for i, t, _, _ in found] == [(0, "I"), (2, "um")]  # the misheard word is not trusted
    thin = [("I", 0, 2, -0.01), ("THINK", 20, 30, -0.01), ("UM", 32, 34, -0.1), ("THIS", 45, 50, -0.01)]
    assert align.find_disfluencies(thin, words, 0.0) == []  # a 40 ms "um" is a misread consonant
    unsure = [("I", 0, 2, -0.9), ("I", 10, 12, -0.9), ("THINK", 20, 30, -0.01), ("THIS", 45, 50, -0.01)]
    assert align.find_disfluencies(unsure, words, 0.0) == []


def test_greedy_words_reads_ctc_frames():
    vocab = {0: "<pad>", 1: "|", 2: "H", 3: "I"}
    frames = [2, 2, 0, 3, 1, 1, 3, 0]  # "HI" then "I"
    em = np.full((len(frames), 4), -10.0)
    for t, k in enumerate(frames):
        em[t, k] = -0.1
    assert [w[0] for w in align.greedy_words(em, vocab, 0, 1)] == ["HI", "I"]


def test_an_unsure_lone_first_word_is_aligned_with_what_follows():
    sure, unsure = _word(0, "So,", 1.3, 1.7), _word(0, "So,", 1.3, 1.7)
    sure.conf, unsure.conf = 0.9, 0.24
    rest = [_word(1, "hello", 3.0, 3.3), _word(2, "there", 3.3, 3.6)]
    assert align._blocks([sure, *rest]) == [(0, 1), (1, 3)]  # Whisper is sure: the gap is real
    assert align._blocks([unsure, *rest]) == [(0, 3)]  # it may have mistimed the word: let the audio decide
