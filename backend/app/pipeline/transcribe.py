"""Transcription (PLAN 12): faster-whisper base.en int8 CPU, word timestamps, disfluency prompt."""
from __future__ import annotations

import logging
import os
import re

import numpy as np

from app import config  # noqa: F401  (loads .env)
from app.schemas import Word

log = logging.getLogger(__name__)

# small.en is clearly more accurate than base.en (words, names, word timings) at ~2-3x the CPU time;
# set ROUGHCUT_WHISPER_MODEL=base.en in .env for faster analysis on slow machines.
MODEL_SIZE = os.environ.get("ROUGHCUT_WHISPER_MODEL") or "small.en"
# Whisper otherwise drops many um/uh; this prompt nudges it to keep disfluencies (PLAN 12).
INITIAL_PROMPT = "Umm, let me think, like, hmm... Okay, so, uh, here's what I, like, mean."

_models: dict[tuple, object] = {}


def _get_model(size: str, device: str, compute_type: str):
    key = (size, device, compute_type)
    if key not in _models:
        from faster_whisper import WhisperModel

        _models[key] = WhisperModel(size, device=device, compute_type=compute_type)
    return _models[key]


def transcribe(wav_path, model_size: str = MODEL_SIZE, device: str = "cpu", compute_type: str = "int8",
               initial_prompt: str = INITIAL_PROMPT) -> dict:
    """Returns {model, language, duration, segments:[{start,end,text}], words:[{start,end,word,probability}]}."""
    model = _get_model(model_size, device, compute_type)
    segs, info = model.transcribe(
        str(wav_path), language="en", word_timestamps=True, initial_prompt=initial_prompt,
        beam_size=10,  # 5 misread words on real footage ("send it to" -> "this is the entity of")
        vad_filter=False,  # VAD would also strip the fillers we want to see
        temperature=0.0,  # no sampling fallback: the same file always gives the same transcript
        # Conditioning on the previous window is what makes Whisper loop ("okay, okay, okay, ...") and echo the
        # disfluency prompt over silence/music; each window is decoded on its own instead. Long silent stretches
        # that produced a suspicious segment are skipped (needs word timestamps).
        condition_on_previous_text=False, hallucination_silence_threshold=2.0,
    )
    out_segs, out_words = [], []
    for s in segs:
        out_segs.append({"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()})
        for w in s.words or []:
            out_words.append({"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word,
                              "probability": round(float(w.probability), 4)})
    return {"model": model_size, "language": info.language, "duration": round(info.duration, 3),
            "initial_prompt": initial_prompt, "segments": out_segs, "words": out_words}


MIN_WORD_DUR = 0.02


def words_from_transcript(tr: dict) -> list[Word]:
    """Whisper words -> schema Words. Sanitizes timing (non-negative duration, no overlaps); drops blanks."""
    words: list[Word] = []
    prev_end = 0.0
    for w in tr["words"]:
        text = w["word"].strip()
        if not text or not re.search(r"\w", text):
            if words and text:  # lone punctuation token: glue onto the previous word
                words[-1].text_original += text
                words[-1].text_display += text
            continue
        start = max(float(w["start"]), prev_end)
        end = max(float(w["end"]), start + MIN_WORD_DUR)
        i = len(words)
        words.append(Word(id=i, idx=i, unit_id=-1, start=start, end=end, text_original=text,
                          text_display=text, conf=float(w.get("probability", 1.0))))
        prev_end = end
    return words


LANGUAGE_MODEL = "small"  # multilingual: tiny mislabels accented English; small got our test clips right
LANGUAGE_WARN_PROB = 0.7


def detect_language(wav_path) -> tuple[str, float] | None:
    """(language, probability) of the speech, or None if the model is unavailable. Only used to warn: analysis is
    English-only, and a different language would otherwise silently give a made-up transcript."""
    try:
        from faster_whisper.audio import decode_audio

        lang, prob, _ = _get_model(LANGUAGE_MODEL, "cpu", "int8").detect_language(decode_audio(str(wav_path)))
        return lang, float(prob)
    except Exception:
        log.warning("language detection unavailable", exc_info=True)
        return None


def preload() -> None:
    """Load the speech models up front, so the first upload is as fast as the later ones."""
    _get_model(MODEL_SIZE, "cpu", "int8")
    _get_model(LANGUAGE_MODEL, "cpu", "int8")


HALLU_PAD = 0.05  # s of context around a word when checking that there is any voice at all
HALLU_MIN_VOICED = 0.1  # share of voiced frames below which a "word" is a hallucination over silence


def drop_hallucinations(words: list[Word], energy) -> list[Word]:
    """Drop words that have (almost) no voiced audio under them: Whisper invents text over silence and music
    beds, and every edit decision downstream would be built on it. Ids/idx are renumbered."""
    keep = []
    for w in words:
        m = energy.voiced_mask(max(0.0, w.start - HALLU_PAD), w.end + HALLU_PAD)
        if m.size == 0 or float(np.mean(m)) >= HALLU_MIN_VOICED:
            keep.append(w)
    if len(keep) < len(words):
        log.info("dropped %d words with no voice under them", len(words) - len(keep))
    for i, w in enumerate(keep):
        w.id = w.idx = i
    return keep
