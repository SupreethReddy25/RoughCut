"""Does exploration pay? Acceptance during training counts every exploratory proposal as a mistake, so it cannot answer
that alone. Each variant trains on 10 videos, then its posterior mean is frozen and scored on 4 unseen videos.
Three feedback conditions: normal; reluctant creators who rarely correct; a 30 percent review budget.

    set PYTHONPATH=backend
    backend\\.venv\\Scripts\\python.exe paper\\scripts\\run_exploration.py
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

from sim.experiments.common import make_learner, style_mixture, videos_for
from sim.personas import BASE_PERSONAS, make_persona
from sim.session import GreedyView, evaluate, run_user

OUT = Path(__file__).resolve().parents[1] / "data" / "exploration.json"
VARIANTS = {"Greedy (v=0, no floor)": ("lints", {"v": 0.0, "eps_floor": 0.0}), "Thompson v=0.05": ("lints", {"v": 0.05}),
            "Thompson v=0.25": ("lints", {"v": 0.25}), "Thompson v=0.5": ("lints", {"v": 0.5}),
            "LinUCB": ("linucb", {}), "eps-greedy": ("eps", {})}
CONDITIONS = {"normal": {}, "reluctant": {"p_override": 0.2}, "sparse review": {"review_budget": 0.3}}


def task(t: dict) -> dict:
    name, seed = t["persona"], t["seed"]
    persona = make_persona(name, **CONDITIONS[t["cond"]])
    algo, kw = VARIANTS[t["variant"]]
    learner = make_learner(algo, None, seed, 0.995, mixture=style_mixture((name,)), **kw)
    run = run_user(persona, learner, videos_for(seed, 10), seed)
    train = float(np.mean([v.acceptance for v in run.videos]))
    held = evaluate(GreedyView(learner.policy).choose, make_persona(name), videos_for(seed, 4, offset=500), 100)
    return {**t, "train": train, "held": held["overall"], "held_by_type": {k: v for k, v in held.items() if k != "overall"}}


if __name__ == "__main__":
    for p in BASE_PERSONAS:
        style_mixture((p,))
    tasks = [{"persona": p, "seed": s, "variant": v, "cond": c} for c in CONDITIONS for v in VARIANTS for p in BASE_PERSONAS
             for s in range(1, 9)]
    t0 = time.time()
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    done = {(r["persona"], r["seed"], r["variant"], r["cond"]) for r in out}
    todo = [t for t in tasks if (t["persona"], t["seed"], t["variant"], t["cond"]) not in done]
    for attempt in range(6):  # a worker can be killed from outside: keep what finished and carry on
        if not todo:
            break
        try:
            with ProcessPoolExecutor(max_workers=6) as ex:
                for start in range(0, len(todo), 48):
                    out += list(ex.map(task, todo[start:start + 48], chunksize=2))
                    OUT.write_text(json.dumps(out), encoding="utf-8")
                    print(len(out), len(tasks), round(time.time() - t0), flush=True)
        except BrokenProcessPool:
            print("worker pool lost, resuming", flush=True)
        done = {(r["persona"], r["seed"], r["variant"], r["cond"]) for r in out}
        todo = [t for t in tasks if (t["persona"], t["seed"], t["variant"], t["cond"]) not in done]
    print("done" if not todo else f"incomplete: {len(todo)} left", round(time.time() - t0))
