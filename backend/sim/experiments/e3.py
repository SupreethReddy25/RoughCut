"""E3 (PLAN 10): is it personalised?  Train LinTS on persona P, evaluate the frozen policy against persona Q.

Training uses no prior (what is learned comes only from P's feedback). Evaluation proposes with the posterior mean
(no exploration) on held-out videos and counts how often the proposal equals Q's ideal action. The matrix is
"diagonal-dominant" when each persona Q is served best by the policy trained on Q (column check) and each policy
does best on the persona it was trained on (row check).

    python -m sim.experiments.e3 [--seeds 10] [--train-videos 10] [--test-videos 5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.experiments import plots
from sim.experiments.common import make_learner, run_parallel, videos_for
from sim.personas import BASE_PERSONAS, make_persona
from sim.session import GreedyView, evaluate, run_user


def e3_task(task: dict) -> dict:
    seed = task["seed"]
    learner = make_learner("lints", None, seed, gamma=0.995)
    run_user(make_persona(task["train"]), learner, videos_for(seed, task["train_videos"]), seed)
    view = GreedyView(learner.policy)
    test = videos_for(seed, task["test_videos"], offset=500)
    return {"train": task["train"], "seed": seed,
            "eval": {q: evaluate(view.choose, make_persona(q), test) for q in BASE_PERSONAS}}


def run(seeds: int = 10, train_videos: int = 10, test_videos: int = 5, workers: int | None = None) -> list[dict]:
    tasks = [{"train": p, "seed": s, "train_videos": train_videos, "test_videos": test_videos}
             for p in BASE_PERSONAS for s in range(1, seeds + 1)]
    return run_parallel(e3_task, tasks, workers)


def matrix(results: list[dict], key: str = "overall") -> np.ndarray:
    """M[i, j] = mean acceptance of the policy trained on persona i, judged by persona j."""
    M = np.zeros((len(BASE_PERSONAS), len(BASE_PERSONAS)))
    for i, p in enumerate(BASE_PERSONAS):
        rs = [r for r in results if r["train"] == p]
        for j, q in enumerate(BASE_PERSONAS):
            M[i, j] = np.mean([r["eval"][q][key] for r in rs])
    return M


def dominance(M: np.ndarray) -> dict:
    n = len(M)
    col = [bool(M[j, j] >= M[:, j].max() - 1e-12) for j in range(n)]  # persona j is best served by policy j
    row = [bool(M[i, i] >= M[i, :].max() - 1e-12) for i in range(n)]  # policy i does best on persona i
    return {"columns": dict(zip(BASE_PERSONAS, col, strict=True)), "rows": dict(zip(BASE_PERSONAS, row, strict=True)),
            "column_dominant": all(col), "row_dominant": all(row)}


def report(results: list[dict], out_dir: Path | None = None) -> dict:
    M = matrix(results)
    out = {"personas": list(BASE_PERSONAS), "matrix": M.tolist(), "dominance": dominance(M),
           "mean_diagonal": float(np.mean(np.diag(M))),
           "mean_off_diagonal": float((M.sum() - np.trace(M)) / (M.size - len(M)))}
    d = out_dir or plots.FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / "e3_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    fig, ax = plots.plt.subplots(figsize=(6.5, 5.5))
    im = plots.heatmap(ax, M, [p.capitalize() for p in BASE_PERSONAS], [p.capitalize() for p in BASE_PERSONAS],
                       "E3: acceptance of policy trained on P (rows), judged by Q (columns)")
    ax.set_ylabel("trained on persona")
    ax.set_xlabel("evaluated against persona")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plots.save(fig, "e3_cross_persona.png", out_dir)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="E3: cross-persona matrix")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--train-videos", type=int, default=10)
    ap.add_argument("--test-videos", type=int, default=5)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    out = report(run(a.seeds, a.train_videos, a.test_videos, a.workers))
    M = np.array(out["matrix"])
    print("rows = trained on, columns = judged by")
    print(f"{'':<13}" + "".join(f"{p[:9]:>11}" for p in BASE_PERSONAS))
    for p, row in zip(BASE_PERSONAS, M, strict=True):
        print(f"{p:<13}" + "".join(f"{v:>11.3f}" for v in row))
    d = out["dominance"]
    print(f"\nmean diagonal {out['mean_diagonal']:.3f}  vs mean off-diagonal {out['mean_off_diagonal']:.3f}")
    print(f"column-dominant (each persona best served by its own policy): {d['column_dominant']} {d['columns']}")
    print(f"row-dominant (each policy best on its own persona):           {d['row_dominant']} {d['rows']}")


if __name__ == "__main__":
    main()
