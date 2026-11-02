"""Forced alignment of Whisper's words with a wav2vec2 CTC model (the WhisperX approach).

Whisper's word timestamps come from cross-attention and drift by 0.1-0.5 s: it stretches words over silences
("you'll" swallowing the pause before a restart), which hides pauses inside words and leaves dead air in the
edit. Here the transcript Whisper produced is kept, and only its timing is re-derived: the audio around a block
of words goes through wav2vec2 (character CTC, 20 ms frames), and a Viterbi pass finds the best monotonic path
of the words' characters through those frames. Each word gets the frames of its first..last character.

Words with nothing to align (digits, symbols) are placed between their aligned neighbours. A block whose path is
implausible (very low per-frame score) keeps Whisper's timing. If the model cannot be loaded (offline, no
space) the whole step is skipped with a warning: alignment improves timing but nothing depends on it.
Set ROUGHCUT_ALIGN=0 in .env to switch it off.

Whisper also tidies speech: it can write "I I think" as "I think" and drop an "um". wav2vec2 transcribes what it
hears, so its own greedy reading of the block is compared with Whisper's words, and a disfluency only it heard (a
hesitation sound, or an exact repeat of the neighbouring word) is put back before alignment. Nothing else from that
reading is trusted: its spelling is far worse than Whisper's.
"""
from __future__ import annotations

import logging
import os
import re
from difflib import SequenceMatcher

import numpy as np

from app.schemas import Word

log = logging.getLogger(__name__)

MODEL_NAME = "facebook/wav2vec2-base-960h"
SR = 16000
FRAME_S = 0.02  # wav2vec2 stride (320 samples at 16 kHz)
BLOCK_MAX_S = 20.0  # words are aligned in blocks of at most this much audio
BLOCK_BREAK_GAP = 0.8  # ... split where Whisper sees a gap at least this long
MARGIN = 0.4  # s of extra audio searched on each side of a block
MIN_WORD = 0.04
MIN_BLOCK_SCORE = -6.0  # mean log-prob per path frame below which the block keeps Whisper's timing

HESITATIONS = {"UM": "um", "UMM": "um", "UH": "uh", "UHM": "um", "ER": "uh", "AH": "uh", "HM": "hmm", "MM": "hmm"}
LOW_CONF_WORD = 0.5  # a word Whisper is this unsure of may be mistimed: it does not end a block by itself
RECOVER_WINDOW_S = 60.0  # s of audio read at once
RECOVER_MIN_S = 0.08  # a hesitation must last at least this long: shorter is a misread consonant (CTC peaks are thin)
RECOVER_MIN_CONF = -0.5  # mean log-prob per frame a recovered word needs

_model = None


def enabled() -> bool:
    from app import config  # noqa: F401  (loads .env)

    return os.environ.get("ROUGHCUT_ALIGN", "1").strip().lower() not in ("0", "false", "no", "off")


def _load():
    global _model
    if _model is None:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        proc = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).eval()
        torch.set_grad_enabled(False)
        _model = (proc, model)
    return _model


def preload() -> None:
    if enabled():
        _load()


def _emissions(x: np.ndarray) -> np.ndarray:
    """(frames, vocab) log-probs for a mono 16 kHz float signal."""
    import torch

    proc, model = _load()
    inp = proc(x, sampling_rate=SR, return_tensors="pt").input_values
    with torch.inference_mode():
        logits = model(inp).logits[0]
    return torch.log_softmax(logits, dim=-1).numpy().astype(np.float64)


def _chars(text: str) -> str:
    return re.sub(r"[^A-Z']", "", text.upper().replace("’", "'"))


def viterbi(em: np.ndarray, tokens: list[int], blank: int) -> tuple[list[int], list[bool], float] | None:
    """Best monotonic path of `tokens` through the frames of `em`. A frame either emits the next token, stays
    on the current token, or is blank. Returns (state per frame = number of tokens emitted so far, whether the
    frame is a token frame, mean log-prob along the path), or None when the tokens do not fit the frames."""
    T, N = len(em), len(tokens)
    if N == 0 or T < N:
        return None
    tok = np.asarray(tokens)
    dp = np.full((T + 1, N + 1), -np.inf)
    dp[0, 0] = 0.0
    move = np.zeros((T, N + 1), dtype=np.int8)  # 0 blank-stay, 1 token-stay, 2 advance
    for t in range(T):
        e = em[t]
        blank_stay = dp[t] + e[blank]
        tok_stay = np.full(N + 1, -np.inf)
        tok_stay[1:] = dp[t, 1:] + e[tok]
        adv = np.full(N + 1, -np.inf)
        adv[1:] = dp[t, :-1] + e[tok]
        stack = np.stack([blank_stay, tok_stay, adv])
        move[t] = stack.argmax(axis=0)
        dp[t + 1] = stack.max(axis=0)
    if not np.isfinite(dp[T, N]):
        return None
    states, is_tok = [0] * T, [False] * T
    j = N
    for t in range(T - 1, -1, -1):
        m = move[t, j]
        states[t], is_tok[t] = j, m != 0
        if m == 2:
            j -= 1
    return states, is_tok, float(dp[T, N] / T)


def _blocks(words: list[Word]) -> list[tuple[int, int]]:
    out, a = [], 0
    for i in range(1, len(words) + 1):
        unsure_lead = i < len(words) and i - a == 1 and words[a].conf < LOW_CONF_WORD
        if (i == len(words) or (words[i].start - words[i - 1].end >= BLOCK_BREAK_GAP and not unsure_lead)
                or words[i].end - words[a].start > BLOCK_MAX_S):
            out.append((a, i))
            a = i
    return out


def _align_block(x: np.ndarray, words: list[Word], t0: float, t1: float, vocab: dict[str, int], blank: int,
                 sep: int | None) -> list[tuple[float, float] | None] | None:
    seg = x[int(t0 * SR):int(t1 * SR)]
    if len(seg) < SR // 10:
        return None
    em = _emissions(seg)
    tokens, owner = [], []  # owner[k] = index of the word token k spells (-1 for a separator)
    for i, w in enumerate(words):
        cs = [vocab[c] for c in _chars(w.text_original) if c in vocab]
        if not cs:
            continue
        if tokens and sep is not None:
            tokens.append(sep)
            owner.append(-1)
        tokens += cs
        owner += [i] * len(cs)
    res = viterbi(em, tokens, blank)
    if res is None:
        return None
    states, is_tok, score = res
    if score < MIN_BLOCK_SCORE:
        log.info("alignment block %.1f-%.1f s scored %.2f: keeping Whisper timing", t0, t1, score)
        return None
    span: dict[int, list[int]] = {}
    for t, (j, tk) in enumerate(zip(states, is_tok, strict=True)):
        if tk and j > 0 and owner[j - 1] >= 0:
            span.setdefault(owner[j - 1], []).append(t)
    return [(t0 + min(span[i]) * FRAME_S, t0 + (max(span[i]) + 1) * FRAME_S) if i in span else None
            for i in range(len(words))]


def greedy_words(em: np.ndarray, id_to_char: dict[int, str], blank: int, sep: int | None
                 ) -> list[tuple[str, int, int, float]]:
    """wav2vec2's own reading of the frames: [(WORD, first frame, last frame + 1, mean log-prob of its frames)]."""
    out: list[tuple[str, int, int, float]] = []
    chars: list[str] = []
    scores: list[float] = []
    first, last, prev = 0, -1, blank
    for t, k in enumerate(int(v) for v in em.argmax(axis=1)):
        if k == sep:
            if chars:
                out.append(("".join(chars), first, last + 1, float(np.mean(scores))))
            chars, scores = [], []
        elif k != blank:
            if not chars:
                first = t
            if k != prev or not chars:
                chars.append(id_to_char.get(k, ""))
            last = t
            scores.append(float(em[t, k]))
        prev = k
    if chars:
        out.append(("".join(chars), first, last + 1, float(np.mean(scores))))
    return out


def find_disfluencies(heard: list[tuple[str, int, int, float]], words: list[Word], t0: float
                      ) -> list[tuple[int, str, float, float]]:
    """Disfluencies in `heard` that Whisper's `words` lack: [(insert before words[index], text, start, end)]."""
    ours = [_chars(w.text_original) for w in words]
    theirs = [h[0] for h in heard]
    out = []
    for op, i1, _i2, j1, j2 in SequenceMatcher(None, ours, theirs, autojunk=False).get_opcodes():
        if op != "insert":
            continue
        for j in range(j1, j2):
            tok, fa, fb, conf = heard[j]
            if conf < RECOVER_MIN_CONF:
                continue
            if tok in HESITATIONS:
                if (fb - fa) * FRAME_S < RECOVER_MIN_S:
                    continue
                text = HESITATIONS[tok]
            elif (j + 1 < len(theirs) and theirs[j + 1] == tok) or (j > 0 and theirs[j - 1] == tok):
                text = "I" if tok == "I" else tok.lower()  # the extra copy of a stutter ("I I", "the the")
            else:
                continue
            out.append((i1, text, t0 + fa * FRAME_S, t0 + fb * FRAME_S))
    return out


def recover_disfluencies(words: list[Word], x: np.ndarray, vocab: dict[str, int], blank: int, sep: int | None) -> int:
    """In place: insert the disfluencies Whisper tidied away, renumbering the words. Returns how many."""
    id_to_char = {v: k for k, v in vocab.items()}
    dur = len(x) / SR
    n = 0
    chunks: list[tuple[int, int]] = []  # long windows: the reading of a short block misses repeats a longer one hears
    for a, z in _blocks(words):
        if chunks and words[z - 1].end - words[chunks[-1][0]].start <= RECOVER_WINDOW_S:
            chunks[-1] = (chunks[-1][0], z)
        else:
            chunks.append((a, z))
    for a, z in reversed(chunks):  # back to front, so earlier indices stay valid
        t0 = max(words[a].start - MARGIN, 0.0)
        seg = x[int(t0 * SR):int(min(words[z - 1].end + MARGIN, dur) * SR)]
        if len(seg) < SR // 10:
            continue
        heard = greedy_words(_emissions(seg), id_to_char, blank, sep)
        for i, text, s, e in reversed(find_disfluencies(heard, words[a:z], t0)):
            log.info("recovered disfluency %r at %.2f s", text, s)
            words.insert(a + i, Word(id=0, idx=0, unit_id=-1, start=s, end=e, text_original=text, text_display=text,
                                     conf=0.5))
            n += 1
    if n:
        for i, w in enumerate(words):
            w.id = w.idx = i
    return n


def align_words(words: list[Word], wav_path) -> list[Word]:
    """In place (and returned): re-time `words` against the audio. Never raises; on failure keeps the input."""
    if not words or not enabled():
        return words
    try:
        from app.pipeline.pauses import load_wav

        x, sr = load_wav(wav_path)
        if sr != SR:
            raise ValueError(f"expected {SR} Hz audio, got {sr}")
        proc, _ = _load()
        vocab = proc.tokenizer.get_vocab()
        blank = vocab.get(proc.tokenizer.pad_token, 0)
        sep = vocab.get("|")
    except Exception:
        log.warning("forced alignment unavailable; keeping Whisper word timing", exc_info=True)
        return words
    dur = len(x) / SR
    try:
        recover_disfluencies(words, x, vocab, blank, sep)
    except Exception:
        log.warning("disfluency recovery failed; keeping Whisper's words", exc_info=True)
    blocks = _blocks(words)
    n_done = 0
    for b, (a, z) in enumerate(blocks):
        lo = words[blocks[b - 1][1] - 1].end if b else 0.0
        hi = words[blocks[b + 1][0]].start if b + 1 < len(blocks) else dur
        t0 = max(lo, words[a].start - MARGIN, 0.0)
        t1 = min(hi, words[z - 1].end + MARGIN, dur)
        try:
            spans = _align_block(x, words[a:z], t0, t1, vocab, blank, sep)
        except Exception:
            log.warning("alignment of block %.1f-%.1f s failed", t0, t1, exc_info=True)
            spans = None
        if spans is None:
            continue
        n_done += 1
        # words without characters to align go between their aligned neighbours, in Whisper's proportions
        for k, w in enumerate(words[a:z]):
            if spans[k] is not None:
                w.start, w.end = spans[k]
        for k, w in enumerate(words[a:z]):
            if spans[k] is None:
                prev_end = next((words[a + q].end for q in range(k - 1, -1, -1) if spans[q] is not None), t0)
                next_start = next((words[a + q].start for q in range(k + 1, z - a) if spans[q] is not None), t1)
                w.start = min(max(w.start, prev_end), next_start - MIN_WORD)
                w.end = max(min(w.end, next_start), w.start + MIN_WORD)
    # keep the list monotonic and non-overlapping
    prev = 0.0
    for w in words:
        w.start = round(max(w.start, prev), 4)
        w.end = round(max(w.end, w.start + MIN_WORD), 4)
        prev = w.end
    log.info("aligned %d/%d blocks", n_done, len(blocks))
    return words
