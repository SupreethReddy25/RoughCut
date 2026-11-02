"""Numbers for the features added after the first draft: paper/data/new_features.json.

Final bandit settings on fresh seeds, E11 as paired comparisons, the cost of the new agents.

    set PYTHONPATH=backend
    backend\\.venv\\Scripts\\python.exe paper\\scripts\\new_features.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from app.appctx import (
    PRODUCT_EPS,
    PRODUCT_EPS_BY_TYPE,
    PRODUCT_LAMBDA,
    PRODUCT_LAMBDA_BY_TYPE,
    PRODUCT_V,
)
from app.rl.hook import HookBandit, candidates, context_of
from app.rl.rhythm import ARMS, D, Episode, RhythmAgent, Step
from app.session import EditSession
from scipy import stats
from sim.experiments.common import (
    acc_matrix,
    run_learning_task,
    run_parallel,
    videos_for,
)
from sim.personas import BASE_PERSONAS

ROOT = Path(__file__).resolve().parents[1] / "data"
BASE = {"lam": PRODUCT_LAMBDA, "lam_by_type": PRODUCT_LAMBDA_BY_TYPE, "eps_by_type": PRODUCT_EPS_BY_TYPE}
SETTINGS = {"earlier (v=0.25, eps=0.02)": {**BASE, "v": 0.25}, "final": {**BASE, "v": PRODUCT_V, "eps_floor": PRODUCT_EPS}}
SEEDS = range(301, 321)  # never used to choose anything


def t_ci(d: np.ndarray) -> float:
    return float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    return {"a": float(a.mean()), "b": float(b.mean()), "diff": float(d.mean()), "ci": t_ci(d), "n": len(d),
            "p": float(stats.ttest_rel(a, b).pvalue) if d.std() > 0 else 1.0}


def bandit_settings() -> dict:
    tasks = [{"persona": p, "algo": "lints", "seed": s, "n_videos": 6, "tag": (n, "plain"), "prior": "mixture", "gamma": 0.995, "policy_kw": kw}
             for n, kw in SETTINGS.items() for p in BASE_PERSONAS for s in SEEDS]
    tasks += [{"persona": "drift", "algo": "lints", "seed": s, "n_videos": 12, "tag": (n, "drift"), "prior": "generic", "gamma": 0.995,
               "policy_kw": kw, "exclude": ("relaxed", "snappy"), "persona_kw": {"at": 6}} for n, kw in SETTINGS.items() for s in SEEDS]
    res = run_parallel(run_learning_task, tasks, workers=6, progress=False)

    def mat(n, cond, dt=None):
        rs = sorted((r for r in res if r["tag"] == (n, cond)), key=lambda r: (r["persona"], r["seed"]))
        return acc_matrix(rs, dt)

    old, new = SETTINGS
    return {"seeds": [SEEDS[0], SEEDS[-1]], "video_1": paired(mat(new, "plain")[:, 0], mat(old, "plain")[:, 0]),
            "video_3": paired(mat(new, "plain")[:, 2], mat(old, "plain")[:, 2]),
            "video_6": paired(mat(new, "plain")[:, 5], mat(old, "plain")[:, 5]),
            "keep_cut_video_6": paired(np.nan_to_num(mat(new, "plain", "phrase")[:, 5]), np.nan_to_num(mat(old, "plain", "phrase")[:, 5])),
            "drift_plus_1": paired(mat(new, "drift")[:, 6], mat(old, "drift")[:, 6]),
            "drift_plus_3": paired(mat(new, "drift")[:, 8], mat(old, "drift")[:, 8]),
            "drift_plus_6": paired(mat(new, "drift")[:, 11], mat(old, "drift")[:, 11])}


def e11_paired() -> dict:
    raw = json.loads((ROOT / "e" / "e11_raw.json").read_text(encoding="utf-8"))
    out: dict = {}
    for other in ("bandit", "rule"):
        for metric in ("acc", "rating"):
            for when, k in (("video_3", 3), ("final", -1)):
                key = lambda r: (r["persona"], r["seed"])
                a = sorted((r for r in raw if r["algo"] == "rlsvi"), key=key)
                b = sorted((r for r in raw if r["algo"] == other), key=key)
                out[f"rlsvi_vs_{other}|{metric}|{when}"] = paired(np.array([r["curve"][k][metric] for r in a]),
                                                                  np.array([r["curve"][k][metric] for r in b]))
    return out


def cost() -> dict:
    agent = RhythmAgent(seed=1)
    v = videos_for(1, 1)[0]
    sess = EditSession(v.uf, v.word_feats, v.video_type, agents=True)
    agent.begin_video()
    ctxs = []
    for item in sess.items():
        if item.decision_type == "rhythm":
            ctxs.append(item.ctx)
        sess.commit(item, item.arms[0] if item.decision_type != "emphasis" else "NONE")
    for _ in range(10):  # a creator with 10 videos behind them
        agent.add_episode(Episode([Step(c, "NONE", 1.0) for c in ctxs], rating=1.0))
    times = []
    for _ in range(5):
        t = time.perf_counter()
        agent.begin_video()
        times.append(time.perf_counter() - t)
    t = time.perf_counter()
    for c in ctxs:
        agent.choose(c)
    choose = (time.perf_counter() - t) / max(len(ctxs), 1)
    hook = HookBandit(seed=1)
    ctx = context_of(candidates(v.uf))
    t = time.perf_counter()
    for _ in range(20):
        hook.choose(ctx)
    return {"rlsvi_features": D, "rlsvi_arms": len(ARMS), "rlsvi_params": D * len(ARMS), "rlsvi_begin_video_ms_10_videos": float(np.mean(times) * 1e3),
            "rlsvi_choose_ms": choose * 1e3, "hook_choose_ms": (time.perf_counter() - t) / 20 * 1e3, "steps_per_video": len(ctxs)}


if __name__ == "__main__":
    out = {"cost": cost(), "e11": e11_paired()}
    print(json.dumps(out["cost"]), flush=True)
    out["bandit_settings"] = bandit_settings()
    (ROOT / "new_features.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["bandit_settings"], indent=1), flush=True)
