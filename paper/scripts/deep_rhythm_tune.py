"""Give DQN and PPO a fair chance on visual rhythm: tune on seeds 201-203, then rerun the best settings on seeds 101-120.
Updates the rhythm rows of paper/data/deep.json.

    set PYTHONPATH=backend;paper\\scripts
    backend\\.venv\\Scripts\\python.exe paper\\scripts\\deep_rhythm_tune.py
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from deep_baselines import OUT, PERSONAS, WORKERS, rhythm_task

GRID = {"DQN": [{"lr": lr, "eps": e} for lr, e in product((3e-4, 1e-3, 3e-3), (0.05, 0.1, 0.2))],
        "PPO": [{"lr": lr, "epochs": n} for lr, n in product((3e-4, 1e-3, 3e-3), (4, 20, 60))]}

if __name__ == "__main__":
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        tasks = [{"algo": a, "warm": w, "persona": p, "seed": s, "videos": 8, "kw": kw, "cfg": json.dumps(kw)}
                 for a, g in GRID.items() for kw in g for w in (False, True) for p in PERSONAS for s in (201, 202, 203)]
        res = list(ex.map(rhythm_task, tasks, chunksize=2))
        best, table = {}, {}
        for a, g in GRID.items():
            for w in (False, True):
                sc = {json.dumps(kw): float(np.mean([r["curve"][-1]["acc"] for r in res if r["algo"] == a and r["warm"] == w and r["cfg"] == json.dumps(kw)]))
                      for kw in g}
                k = max(sc, key=sc.get)
                best[(a, w)], table[f"{a}|{w}"] = json.loads(k), sc
                print(a, w, "best", k, round(sc[k], 3), flush=True)
        final = [{"algo": a, "warm": w, "persona": p, "seed": 100 + s, "videos": 8, "kw": kw}
                 for (a, w), kw in best.items() for p in PERSONAS for s in range(1, 21)]
        rhythm = list(ex.map(rhythm_task, final, chunksize=2))
    d = json.loads(OUT.read_text(encoding="utf-8"))
    d["rhythm"], d["rhythm_tuning"] = rhythm, {"table": table, "best": {f"{a}|{w}": kw for (a, w), kw in best.items()}}
    OUT.write_text(json.dumps(d), encoding="utf-8")
    print("done", round(time.time() - t0), flush=True)
