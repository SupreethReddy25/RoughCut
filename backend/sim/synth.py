"""Synthetic talking-head videos for the persona simulator (no media, no models).

A video is generated as a *timed transcript* with the mistakes creators make (fillers, abandoned takes followed by
a retake, tangents, long pauses, join points between takes) and then pushed through the REAL analysis code
(segmentation, filler detection, pause spans, `build_units` with a synthetic energy track, face track and phrase
embeddings). So the simulator's units.json has exactly the same features and structure as the pipeline's.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from app.pipeline import fillers as fl
from app.pipeline import segment
from app.pipeline.features import build_units
from app.pipeline.pauses import FRAME_S, build_pause_spans, energy_from_rms
from app.pipeline.vision import FaceSample
from app.schemas import UnitsFile, Word

STOP = ["the", "a", "and", "to", "of", "in", "is", "that", "it", "for", "you", "this", "with", "on", "are", "but",
        "so", "we", "my", "your", "can", "was", "have", "just", "be", "if", "at", "as"]
GENERIC = ["thing", "really", "video", "time", "day", "people", "start", "better", "keep", "make", "look", "think",
           "great", "easy", "problem", "idea", "first", "again", "always", "important", "simple", "actually", "learn",
           "share", "story", "moment", "reason", "change", "bring", "feel", "world", "little", "enough", "stay",
           "write", "watch", "build", "choose", "believe", "remember", "ready", "happen", "answer", "question"]
TOPIC_WORDS = {
    "tutorial": ["timeline", "export", "settings", "layer", "render", "software", "shortcut", "color", "plugin",
                 "keyframe", "preset", "resolution", "bitrate", "codec", "caption", "template"],
    "vlog": ["morning", "coffee", "trip", "friends", "city", "weekend", "cooking", "gym", "routine", "puppy",
             "airport", "hotel", "market", "sunset", "train", "festival"],
    "opinion": ["industry", "problem", "honestly", "society", "money", "trend", "future", "debate", "culture",
                "audience", "algorithm", "creators", "burnout", "platform", "policy", "quality"],
}
TOPIC_WORDS["other"] = TOPIC_WORDS["tutorial"][:6] + TOPIC_WORDS["vlog"][:5] + TOPIC_WORDS["opinion"][:5]
NUMBERS = ["3", "five", "10", "40%", "two", "100", "seven", "twenty"]
PROPER = ["Premiere", "Instagram", "Tokyo", "Sarah", "YouTube", "Adobe", "London", "Netflix"]
NEGATIONS = ["not", "never", "no", "don't"]
FILLERS = [(("um",), 0.25), (("uh",), 0.20), (("like",), 0.20), (("you", "know"), 0.12), (("so",), 0.08),
           (("basically",), 0.05), (("actually",), 0.05), (("i", "mean"), 0.05)]
DANGLING = ["the", "and", "to", "of", "for", "with"]
FILLER_ONLY = [("Um",), ("Uh",), ("So", "yeah"), ("Okay", "so")]
EMB_DIM = 16


@dataclass
class SynthVideo:
    uf: UnitsFile
    word_feats: dict[int, dict[str, float]]
    video_type: str
    seed: int
    index: int = 0

    @property
    def minutes(self) -> float:
        return self.uf.duration / 60.0


def _unit(rng: np.random.Generator, dim: int = EMB_DIM) -> np.ndarray:
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def _phrase_words(rng, n: int, topical: list[str]) -> list[str]:
    out = []
    for i in range(n):
        r = rng.random()
        if r < 0.38:
            w = str(rng.choice(STOP))
        elif r < 0.62:
            w = str(rng.choice(topical))
        elif r < 0.67:
            w = str(rng.choice(NUMBERS))
        elif r < 0.72:
            w = str(rng.choice(PROPER))
        elif r < 0.75:
            w = str(rng.choice(NEGATIONS))
        else:
            w = str(rng.choice(GENERIC))
        out.append(w)
    if not out[0][:1].isdigit():
        out[0] = out[0][:1].upper() + out[0][1:]
    return out


def _end_punct(rng) -> str:
    r = rng.random()
    return "." if r < 0.58 else "?" if r < 0.68 else "!" if r < 0.76 else "," if r < 0.96 else ""


def _pause_after(rng, punct: str, abandoned: bool) -> float:
    if abandoned:
        return float(rng.uniform(0.5, 1.6))
    r = rng.random()
    if punct in ".?!":
        return float(np.clip(rng.lognormal(np.log(0.75), 0.6), 0.28, 3.2)) if r < 0.9 else float(rng.uniform(1.5, 3.0))
    return float(np.clip(rng.lognormal(np.log(0.38), 0.45), 0.26, 1.6))


def make_video(seed: int, video_type: str | None = None, index: int = 0, n_phrases: int = 30,
               video_id: str | None = None) -> SynthVideo:
    rng = np.random.default_rng(seed)
    vtype = video_type or str(rng.choice(["tutorial", "vlog", "opinion"]))
    topical_pool = TOPIC_WORDS[vtype]
    topical = list(topical_pool)
    off_pool = [w for k, ws in TOPIC_WORDS.items() if k != vtype and k != "other" for w in ws]
    main_topic, off_topic = _unit(rng), _unit(rng)

    # ---- 1. the script: phrases = word strings + per-word filler flags + embedding + flags
    script: list[dict] = []
    for _ in range(n_phrases):
        r = rng.random()
        tangent = r > 0.92
        n = int(rng.integers(3, 13))
        vocab = list(rng.choice(off_pool, size=6, replace=False)) if tangent else topical
        words_ = _phrase_words(rng, n, vocab)
        ph = {"words": words_, "fl": [False] * len(words_), "punct": _end_punct(rng),
              "emb": off_topic if tangent else main_topic, "abandoned": False, "filler_only": False,
              "lowconf": bool(rng.random() < 0.06)}
        if r < 0.10 and n >= 5:  # an abandoned first attempt precedes the full phrase: same words, cut off
            k = max(3, int(np.ceil(0.6 * n)))
            ab = list(words_[:k]) + ([str(rng.choice(DANGLING))] if rng.random() < 0.6 else [])
            script.append({"words": ab, "fl": [False] * len(ab), "punct": "", "emb": ph["emb"], "abandoned": True,
                           "filler_only": False, "lowconf": False})
        script.append(ph)
        if rng.random() < 0.04:
            only = list(FILLER_ONLY[int(rng.integers(len(FILLER_ONLY)))])
            script.append({"words": only, "fl": [True] * len(only), "punct": ",", "emb": main_topic,
                           "abandoned": False, "filler_only": True, "lowconf": False})
    # ---- 2. fillers inside phrases
    for ph in script:
        if ph["filler_only"] or ph["abandoned"] or rng.random() > 0.32:
            continue
        for _ in range(1 if rng.random() < 0.8 else 2):
            tok = FILLERS[int(rng.choice(len(FILLERS), p=[f[1] for f in FILLERS]))][0]
            pos = rng.choice(["start", "mid", "end"], p=[0.4, 0.4, 0.2])
            ws, fls = ph["words"], ph["fl"]
            at = 0 if pos == "start" else len(ws) if pos == "end" else int(rng.integers(1, max(2, len(ws))))
            ins = list(tok)
            if rng.random() < 0.6:
                ins[-1] += ","
            if at == 0:  # the sentence capital moves onto a phrase-initial filler
                ins[0] = ins[0].capitalize()
                if ws[0] not in PROPER:
                    ws[0] = ws[0][:1].lower() + ws[0][1:]
            ph["words"] = ws[:at] + ins + ws[at:]
            ph["fl"] = fls[:at] + [True] * len(ins) + fls[at:]

    # ---- 3. timing
    words: list[Word] = []
    word_phrase: list[int] = []
    t = 0.4
    take_idx: set[int] = set()
    if rng.random() < 0.3 and len(script) > 12:
        take_idx = {int(i) for i in rng.choice(range(5, len(script) - 3), size=int(rng.integers(1, 3)),
                                               replace=False)}
    take_bounds: list[float] = []
    voiced_blobs: list[tuple[float, float]] = []
    word_db: list[tuple[float, float, float]] = []
    for pi, ph in enumerate(script):
        loud = float(rng.normal(-21, 3.5))
        n_w = len(ph["words"])
        for wi, w in enumerate(ph["words"]):
            filler = ph["fl"][wi]
            dur = float(np.clip(0.12 + 0.045 * len(w.strip(".,?!")) + rng.normal(0, 0.03), 0.1, 0.6))
            text = w
            if wi == n_w - 1:  # phrase-final punctuation (a cut-off take trails off with "...")
                base = w.rstrip(".,?!")
                text = base + ("..." if ph["abandoned"] and rng.random() < 0.3 else ph["punct"])
            i = len(words)
            conf = float(rng.uniform(0.25, 0.55)) if ph["lowconf"] else float(np.clip(rng.normal(0.93, 0.05), 0.5, 1.0))
            if filler and rng.random() < 0.3:
                conf = float(rng.uniform(0.3, 0.8))
            words.append(Word(id=i, idx=i, unit_id=-1, start=round(t, 3), end=round(t + dur, 3), text_original=text,
                              text_display=text, conf=conf))
            word_phrase.append(pi)
            word_db.append((t, t + dur, loud + float(rng.normal(0, 2.0)) - (3.0 if filler else 0.0)))
            gap = 0.0
            if wi < n_w - 1:
                near_filler = filler or ph["fl"][wi + 1]
                if near_filler and rng.random() < 0.15:
                    gap = 0.0  # co-articulated filler: no clean gap to cut in
                else:
                    gap = float(rng.choice([0.0, 0.03, 0.06, 0.1, 0.15], p=[0.2, 0.25, 0.25, 0.2, 0.1]))
            t += dur + gap
        if pi + 1 < len(script):
            pa = _pause_after(rng, ph["punct"], ph["abandoned"])
            if pi + 1 in take_idx:
                pa = max(pa, float(rng.uniform(1.0, 2.2)))
                take_bounds.append(round(t + pa / 2, 3))
            if rng.random() < 0.05:
                voiced_blobs.append((t + 0.1, t + 0.1 + float(rng.uniform(0.15, 0.4))))
            t += pa
    duration = round(t + 0.5, 3)

    # ---- 4. synthetic energy track
    nf = int(duration / FRAME_S) + 1
    rms = 10 ** ((-58 + rng.normal(0, 3, nf)) / 20)
    for a, b, db in word_db:
        f0, f1 = int(a / FRAME_S), max(int(b / FRAME_S), int(a / FRAME_S) + 1)
        rms[f0:f1] = 10 ** (db / 20) * rng.uniform(0.8, 1.2, f1 - f0)
    for a, b in voiced_blobs:
        rms[int(a / FRAME_S):int(b / FRAME_S)] = 10 ** (-33 / 20)
    energy = energy_from_rms(rms)

    # ---- 5. real analysis code: segmentation, fillers, pauses, units + features
    segment.clamp_to_takes(words, take_bounds)
    spans = segment.segment_speech(words, take_bounds)
    groups = fl.find_fillers(words, spans)
    pauses = build_pause_spans(words, spans, take_bounds)
    embs = []
    for a, b in spans:
        v = np.mean([script[word_phrase[i]]["emb"] for i in range(a, b)], axis=0) + 0.15 * rng.standard_normal(EMB_DIM)
        embs.append(v / np.linalg.norm(v))
    p_face = float(rng.uniform(0.7, 1.0))
    size = float(np.clip(rng.normal(0.25, 0.04), 0.1, 0.5))
    faces, n_sampled = [], 0
    for k in range(int(duration * 5)):
        n_sampled += 1
        if rng.random() < p_face:
            faces.append(FaceSample(k / 5, 0.5 + float(rng.normal(0, 0.03)), 0.4 + float(rng.normal(0, 0.02)),
                                    size + float(rng.normal(0, 0.01)), size * 1.3, 0.9))
    bounds = [0.0, *take_bounds, duration]
    smap = [{"file": f"take{i + 1}.mp4", "src_start": 0.0, "src_end": round(b - a, 3), "out_start": round(a, 3)}
            for i, (a, b) in enumerate(pairwise(bounds))]
    vid = video_id or f"sim_{vtype}_{seed}"
    uf, wf = build_units(words, spans, pauses, groups, video_id=vid, duration=duration, video_type=vtype,
                         source_map=smap, energy=energy, face_samples=faces, face_sampled=n_sampled,
                         emb=np.array(embs))
    return SynthVideo(uf, wf, vtype, seed, index)
