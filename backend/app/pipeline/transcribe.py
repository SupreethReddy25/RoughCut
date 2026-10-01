"""Transcription (PLAN 12): faster-whisper base.en int8 CPU, word timestamps, disfluency prompt."""
from __future__ import annotations

import logging
import re

from app.schemas import Word

log = logging.getLogger(__name__)

MODEL_SIZE = "base.en"
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
        beam_size=5, vad_filter=False,  # VAD would also strip the fillers we want to see
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
