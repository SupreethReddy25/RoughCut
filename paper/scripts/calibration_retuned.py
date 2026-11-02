"""Confidence quality of the retuned settings against the defaults, seeds 1-6: paper/data/final/calibration.json.

    set PYTHONPATH=backend;paper\\scripts
    backend\\.venv\\Scripts\\python.exe paper\\scripts\\calibration_retuned.py
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from calibration import table
from sim.experiments.common import make_learner, style_mixture, videos_for
from sim.personas import BASE_PERSONAS, make_persona
from sim.session import run_user

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {"default": {}, "retuned": json.loads((ROOT / "data" / "final" / "tune.json").read_text(encoding="utf-8"))["best"]["lints"]}


def task(t):
    name, seed, kw = t
    learner = make_learner("lints", None, seed, 0.995, mixture=style_mixture((name,)), **kw)
    run = run_user(make_persona(name), learner, videos_for(seed, 10), seed, keep_log=True)
    return [(s.confidence, float(s.proposed_action == tr["ideal"]), tr["video"]) for s, tr in zip(run.suggestions, run.truth, strict=True)]


if __name__ == "__main__":
    out = {}
    for label, kw in SETTINGS.items():
        tasks = [(p, s, kw) for p in BASE_PERSONAS for s in range(1, 7)]
        with ProcessPoolExecutor(max_workers=2) as ex:
            rows = [r for rr in ex.map(task, tasks, chunksize=2) for r in rr]
        out[label] = table(rows)
        print(label, {k: round(out[label][k], 3) for k in ("ece", "share_below_0.7", "acc_below_0.7", "acc_above_0.7", "auc_error")}, flush=True)
    (ROOT / "data" / "final" / "calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
