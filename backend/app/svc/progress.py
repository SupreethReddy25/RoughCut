"""Progress and a time estimate for the analysing screen.

The pipeline writes when each stage started (pipeline/run.py). The estimate per stage is base + rate * video seconds,
with the rate learned from this machine's earlier analyses (exponential moving average in data/eta.json), so it gets
better with use and it adapts to how fast this PC is. A stage that runs past its estimate pushes the estimate out, so
the countdown never sits at zero while work is still going on.
"""
from __future__ import annotations

import json
import time

from app.appctx import AppContext
from app.models import Video

# (key, label, seconds regardless of length, seconds per second of video before anything has been learned)
STAGES = [
    ("ingest", "Preparing the footage", 2.0, 0.20),
    ("transcribe", "Listening to every word", 4.0, 0.45),
    ("align", "Finding pauses, fillers and retakes", 3.0, 0.35),
    ("vision", "Finding the face to frame", 2.0, 0.12),
    ("semantic", "Reading for meaning", 3.0, 0.05),
]
LEARNING_RATE = 0.4
OVERRUN = 1.15  # a stage past its estimate is expected to need this much more than it has taken so far


def _eta_file(ctx: AppContext):
    return ctx.data_dir / "eta.json"


def _rates(ctx: AppContext) -> dict[str, float]:
    try:
        return json.loads(_eta_file(ctx).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _marks(ctx: AppContext, pk: int) -> dict:
    try:
        return json.loads((ctx.video_dir(pk) / "progress.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def learn(ctx: AppContext, pk: int, duration: float) -> None:
    """After an analysis: move the learned rates towards how long this one really took. Best effort."""
    done = _marks(ctx, pk).get("done", {})
    if not done or duration <= 1:
        return
    rates = _rates(ctx)
    for key, _, base, default in STAGES:
        if key in done:
            seen = max(done[key] - base, 0.0) / duration
            rates[key] = round((1 - LEARNING_RATE) * rates.get(key, default) + LEARNING_RATE * seen, 4)
    try:
        _eta_file(ctx).write_text(json.dumps(rates), encoding="utf-8")
    except OSError:
        pass


def snapshot(ctx: AppContext, v: Video) -> dict:
    """{stages: [{key, label, state, seconds}], fraction, seconds_left, video_seconds, learned}."""
    marks = _marks(ctx, v.id)
    rates = _rates(ctx)
    now = time.time()
    video_seconds = float(v.settings_json.get("expected_seconds") or 0.0)
    current, done, started = marks.get("current"), marks.get("done", {}), marks.get("started", {})
    stages, total, finished = [], 0.0, 0.0
    for key, label, base, default in STAGES:
        est = base + rates.get(key, default) * video_seconds
        if key in done:
            state, seconds, spent = "done", done[key], done[key]
        elif key == current:
            spent = max(now - started.get(key, now), 0.0)
            est = max(est, spent * OVERRUN)
            state, seconds = "running", spent
        else:
            state, seconds, spent = "waiting", est, 0.0
        length = done[key] if key in done else est
        total += length
        finished += spent
        stages.append({"key": key, "label": label, "state": state, "seconds": round(seconds, 1)})
    return {"stages": stages, "fraction": round(min(finished / total, 0.99), 3) if total else 0.0,
            "seconds_left": round(max(total - finished, 0.0), 1), "video_seconds": video_seconds,
            "learned": bool(rates), "waiting": not marks, "live": marks.get("live", {})}
