"""Computational cost of the learner: decision latency, update, full-log replay, planning, parameters and state size.

    set PYTHONPATH=backend
    backend\\.venv\\Scripts\\python.exe paper\\scripts\\measure_runtime.py
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread: what one request costs
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from app.policy import context
from app.rl.plan import EpisodePlanner
from app.session import EditSession
from sim.experiments.common import make_learner, style_mixture, videos_for
from sim.personas import make_persona
from sim.session import run_user

OUT = Path(__file__).resolve().parents[1] / "data" / "runtime.json"


def main() -> None:
    learner = make_learner("lints", None, 1, 0.995, mixture=style_mixture(("snappy",)))
    t0 = time.perf_counter()
    run = run_user(make_persona("snappy"), learner, videos_for(1, 5), 1, keep_log=True)
    train_s = time.perf_counter() - t0
    n_decisions = sum(sum(v.n.values()) for v in run.videos)
    pol = learner.policy
    test = videos_for(1, 3, offset=500)
    choose, by_type = [], {}
    for v in test:
        sess = EditSession(v.uf, v.word_feats, v.video_type)
        for item in sess.items():
            t = time.perf_counter()
            c = pol.choose(item.decision_type, item.ctx)
            dt = time.perf_counter() - t
            choose.append(dt)
            by_type.setdefault(item.decision_type, []).append(dt)
            sess.commit(item, c.action)
    upd = []
    v = test[0]
    sess = EditSession(v.uf, v.word_feats, v.video_type)
    for item in sess.items():
        a = pol.choose(item.decision_type, item.ctx).action
        t = time.perf_counter()
        pol.update(item.decision_type, item.ctx, a, 1.0)
        upd.append(time.perf_counter() - t)
        sess.commit(item, a)
    plan = []
    for v in test:
        for target in (20.0, 30.0, 45.0):
            t = time.perf_counter()
            EpisodePlanner(v.uf, v.video_type, pol, target, seed=1)
            plan.append(time.perf_counter() - t)
    params = {dt: context.dim(dt) * (1 if dt == "emphasis" else len(context.ARMS[dt])) for dt in ("phrase", "pause", "zoom", "filler", "emphasis")}
    state_bytes = sum(m.A_d.size * 8 + m.b_d.size * 8 for m in pol.models.values())
    ms = lambda a: {"mean": float(np.mean(a) * 1e3), "p95": float(np.percentile(a, 95) * 1e3)}
    out = {"cpu": platform.processor(), "cores_logical": os.cpu_count(), "threads": 1,
           "five_videos_s": train_s, "decisions_in_five_videos": n_decisions, "s_per_decision_including_sim": train_s / n_decisions,
           "choose_ms": ms(choose), "choose_ms_by_type": {k: ms(v) for k, v in by_type.items()}, "update_ms": ms(upd),
           "plan_episode_ms": ms(plan), "plan_stages": [len(EpisodePlanner(test[0].uf, test[0].video_type, pol, 30.0, seed=1).stages)],
           "dims": {dt: context.dim(dt) for dt in params}, "params_by_type": params, "params_total": sum(params.values()),
           "state_kib": state_bytes / 1024, "n_models": len(pol.models)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
