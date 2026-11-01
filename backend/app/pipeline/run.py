"""Analysis pipeline CLI (PLAN 14, Phase 1 analysis half).

    python -m app.pipeline.run data/raw/a.mp4 [b.mp4 ...] --type vlog [--id my_video]

Writes data/processed/{id}/: source.mp4, proxy.mp4, audio.wav, transcript.json, embeddings.npy, units.json
plus two sidecars: word_features.json (per-word D5 features) and face_track.json (5 fps YuNet samples).
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from app import config
from app.pipeline import align, features, fillers, ingest, pauses, segment, semantic, transcribe, vision
from app.schemas import UnitsFile

log = logging.getLogger(__name__)


def default_video_id(inputs: list[Path]) -> str:
    stems = [p.stem for p in inputs]
    return stems[0] if len(stems) == 1 else "_".join(stems)


PEAK_BARS = 56


def _peaks(en) -> list[float]:
    """The video's real waveform as PEAK_BARS heights from 0 to 1."""
    if len(en.db) == 0:
        return []
    chunks = np.array_split(en.db, PEAK_BARS)
    span = max(en.speech_db - en.floor_db, 1e-6)
    # the mean, not the peak: speech nearly always peaks at the same level, which draws a flat wall
    return [round(float(np.clip(((c.mean() - en.floor_db) / span) ** 2, 0.04, 1.0)), 3) for c in chunks]


def _keywords(words) -> list[str]:
    """The words the talk keeps coming back to, most frequent first."""
    counts: dict[str, int] = {}
    for w in words:
        t = fillers.norm(w.text_original)
        if len(t) > 3 and t not in semantic.STOPWORDS and not w.is_filler:
            counts[t] = counts.get(t, 0) + 1
    return [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:8]]


def _preview_frame(video, out: Path) -> None:
    """A frame for the analysing screen and the projects list. Best effort."""
    try:
        subprocess.run([config.ffmpeg_path(), "-y", "-loglevel", "error", "-ss", "1", "-i", str(video), "-frames:v", "1",
                        "-vf", "scale=-2:320", str(out / "thumb.jpg")], check=False, timeout=60, capture_output=True)
    except (OSError, subprocess.SubprocessError, RuntimeError):
        log.warning("preview frame failed for %s", out)


def run_pipeline(inputs: list[str | Path], video_id: str | None = None, video_type: str = "other",
                 out_dir: str | Path | None = None, whisper_model: str = transcribe.MODEL_SIZE,
                 progress=print) -> UnitsFile:
    paths = [Path(p) for p in inputs]
    video_id = video_id or default_video_id(paths)
    out = Path(out_dir) if out_dir else config.PROCESSED_DIR / video_id
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    def step(name: str, t0: float, extra: str = "") -> None:
        progress(f"[{time.time() - t0:5.1f}s] {name} {extra}".rstrip())

    marks: dict = {"started": {}, "done": {}, "current": None, "live": {}}

    def publish() -> None:
        try:
            tmp = out / "progress.tmp"
            tmp.write_text(json.dumps(marks), encoding="utf-8")
            tmp.replace(out / "progress.json")
        except OSError:
            pass

    def mark(stage: str | None) -> None:
        """Stage timings for the analysing screen (svc/progress.py). Best effort: never fails an analysis."""
        now = time.time()
        if marks["current"]:
            marks["done"][marks["current"]] = round(now - marks["started"][marks["current"]], 2)
        marks["current"] = stage
        if stage:
            marks["started"][stage] = now
        publish()

    mark("ingest")

    t0 = time.time()
    ing = ingest.ingest(paths, out)
    step("ingest", t0, f"{len(paths)} clip(s) -> {ing.width}x{ing.height}, {ing.duration:.2f}s")
    take_bounds = [m["out_start"] for m in ing.source_map[1:]]

    en = pauses.energy_from_wav(ing.wav)
    _preview_frame(ing.proxy, out)
    marks["live"]["peaks"] = _peaks(en)
    marks["live"]["frame"] = (out / "thumb.jpg").exists()
    marks["live"]["heard"] = []
    mark("transcribe")
    t0 = time.time()

    def heard(text: str, at: float, total: float) -> None:
        marks["live"]["heard"] = [*marks["live"]["heard"], text][-6:]
        marks["live"]["heard_fraction"] = round(at / total, 3) if total else 0.0
        publish()

    tr = transcribe.transcribe(ing.wav, model_size=whisper_model, on_text=heard)
    detected = transcribe.detect_language(ing.wav)
    if detected:
        tr["detected_language"], tr["detected_language_prob"] = detected[0], round(detected[1], 3)
    (out / "transcript.json").write_text(json.dumps(tr, indent=1, ensure_ascii=False), encoding="utf-8")
    step("transcribe", t0, f"{len(tr['words'])} words")
    if not tr["words"]:
        progress("warning: no speech detected - the video has nothing to edit")

    mark("align")
    t0 = time.time()
    words = transcribe.words_from_transcript(tr)
    align.align_words(words, ing.wav)
    words = transcribe.drop_hallucinations(words, en)  # after alignment: Whisper's own timing can be seconds off
    pauses.refine_word_edges(words, en)
    segment.clamp_to_takes(words, take_bounds)
    for w in words:  # Whisper can place the last word slightly past the end of the media
        w.end = min(w.end, ing.duration)
        w.start = min(w.start, w.end - 0.02)
    spans = segment.segment_speech(words, take_bounds)
    groups = fillers.find_fillers(words, spans)
    fillers.snap_to_valleys(words, groups, en)
    pause_spans = pauses.build_pause_spans(words, spans, take_bounds)
    marks["live"].update(n_words=len(words), n_phrases=len(spans), n_pauses=len(pause_spans), n_fillers=len(groups),
                         pause_marks=[[round(p.start / ing.duration, 4), round(p.end / ing.duration, 4)] for p in pause_spans],
                         filler_marks=[round(words[g.word_idxs[0]].start / ing.duration, 4) for g in groups],
                         filler_words=[words[g.word_idxs[0]].text_original.strip(",.") for g in groups][:12],
                         keywords=_keywords(words))
    publish()
    step("pauses+segment+fillers", t0,
         f"{len(spans)} phrases, {len(pause_spans)} pauses, {len(groups)} fillers (thr {en.thr:.1f} dB)")

    mark("vision")
    t0 = time.time()
    samples, n_sampled = vision.track_faces(ing.proxy)
    if samples:
        marks["live"]["face"] = {"x": round(float(np.median([s.cx for s in samples])), 3),
                                 "y": round(float(np.median([s.cy for s in samples])), 3), "seen": round(len(samples) / max(n_sampled, 1), 2),
                                 "source_ar": round(ing.width / ing.height, 3)}
    (out / "face_track.json").write_text(json.dumps(vision.track_to_json(samples, n_sampled)), encoding="utf-8")
    step("vision", t0, f"{len(samples)}/{int(n_sampled)} sampled frames had a face")

    mark("semantic")
    t0 = time.time()
    texts = [" ".join(w.text_display for w in words[a:b] if not w.is_filler) or " ".join(
        w.text_display for w in words[a:b]) for a, b in spans]
    publish()
    emb = semantic.embed_texts(texts)
    np.save(out / "embeddings.npy", emb)
    step("semantic", t0, f"embeddings {emb.shape}")

    uf, word_feats = features.build_units(
        words, spans, pause_spans, groups, video_id=video_id, duration=ing.duration, video_type=video_type,
        source_map=ing.source_map, energy=en, face_samples=samples, face_sampled=n_sampled, emb=emb)
    uf = UnitsFile.model_validate(uf.model_dump())  # fail loudly if it ever drifts from the frozen contract
    (out / "units.json").write_text(uf.model_dump_json(indent=1), encoding="utf-8")
    (out / "word_features.json").write_text(
        json.dumps({str(k): v for k, v in word_feats.items()}, indent=0), encoding="utf-8")
    mark(None)
    progress(f"done in {time.time() - t_all:.1f}s -> {out}")
    return uf


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="RoughCut analysis pipeline")
    ap.add_argument("inputs", nargs="+", help="one or more video files (joined in order)")
    ap.add_argument("--type", default="other", choices=["tutorial", "vlog", "opinion", "other"])
    ap.add_argument("--id", default=None, help="video id / output folder name (default: from file names)")
    ap.add_argument("--out", default=None, help="output dir (default data/processed/<id>)")
    ap.add_argument("--model", default=transcribe.MODEL_SIZE, help="faster-whisper model size")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING)
    run_pipeline(a.inputs, a.id, a.type, a.out, a.model)


if __name__ == "__main__":
    main()
