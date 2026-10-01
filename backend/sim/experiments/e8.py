"""E8 (PLAN 10): are we asking the right questions?  Review budget 10-100 %, uncertainty vs random vs timeline.

A simulated creator only reviews a fraction of the suggestions of each video (`sim.review`). With the same patience
the three strategies differ in WHICH suggestions get looked at:

    uncertainty  the app's "Needs your eye" order (least confident first)
    random       a random subset
    timeline     from the start of the video until the patience is used up

Metrics (mean over personas, seeds and videos 2..N, 95% CI over seeds x personas):
    posted quality   share of decisions that are right in the video that gets posted
    errors caught    share of the wrong proposals that the review found and corrected
    learned          acceptance of the proposals of the LAST two videos (what the policy has learned)

    python -m sim.experiments.e8 [--seeds 8] [--videos 6]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.experiments import plots
from sim.experiments.common import ci95, generic_prior, make_learner, run_parallel, videos_for
from sim.personas import BASE_PERSONAS, make_persona
from sim.review import STRATEGIES, run_budgeted

BUDGETS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
LABEL = {"uncertainty": "uncertainty queue (ours)", "random": "random", "timeline": "timeline order"}
COLOR = {"uncertainty": "#D55E00", "random": "#999999", "timeline": "#0072B2"}


def e8_task(task: dict) -> dict:
    name, seed = task["persona"], task["seed"]
    learner = make_learner("lints", generic_prior((name,)), seed, 0.995)
    videos = run_budgeted(make_persona(name), learner, videos_for(seed, task["videos"]), seed, task["budget"],
                          task["strategy"])
    return {"persona": name, "seed": seed, "strategy": task["strategy"], "budget": task["budget"],
            "videos": [vars(v) for v in videos]}


def run(seeds: int = 8, n_videos: int = 6, workers: int | None = None, budgets=BUDGETS) -> list[dict]:
    tasks = [{"persona": p, "seed": s, "videos": n_videos, "budget": b, "strategy": st}
             for p in BASE_PERSONAS for st in STRATEGIES for b in budgets for s in range(1, seeds + 1)]
    return run_parallel(e8_task, tasks, workers)


def metrics(r: dict) -> dict[str, float]:
    vs = r["videos"][1:]  # video 1 is the cold start of every strategy; the budget matters from video 2 on
    wrong = sum(v["wrong"] for v in vs)
    return {"posted": float(np.mean([v["final_correct"] / v["n"] for v in vs])),
            "caught": sum(v["wrong_fixed"] for v in vs) / wrong if wrong else float("nan"),
            "learned": float(np.mean([v["correct"] / v["n"] for v in r["videos"][-2:]]))}


def table(results: list[dict], budgets=BUDGETS) -> dict:
    out: dict = {}
    for st in STRATEGIES:
        out[st] = {}
        for key in ("posted", "caught", "learned"):
            means, halves = [], []
            for b in budgets:
                x = np.array([metrics(r)[key] for r in results if r["strategy"] == st and r["budget"] == b])
                m, h = ci95(x)
                means.append(float(m))
                halves.append(float(h))
            out[st][key] = {"mean": means, "ci95": halves}
    return out


def report(results: list[dict], out_dir: Path | None = None, budgets=BUDGETS) -> dict:
    t = table(results, budgets)
    summary = {"budgets": list(budgets), "table": t,
               # the headline: at each budget below 100 %, how much of the posted-quality gap to full review does the
               # uncertainty queue close compared with random review?
               "uncertainty_beats_random_at": [b for i, b in enumerate(budgets) if b < 1.0
                                               and t["uncertainty"]["posted"]["mean"][i] > t["random"]["posted"]["mean"][i]],
               "uncertainty_beats_timeline_at": [b for i, b in enumerate(budgets) if b < 1.0
                                                 and t["uncertainty"]["posted"]["mean"][i] > t["timeline"]["posted"]["mean"][i]]}
    d = out_dir or plots.FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / "e8_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    x = np.array(budgets) * 100
    fig, axes = plots.plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, key, title in zip(axes, ("posted", "caught", "learned"),
                              ("quality of the posted video", "wrong suggestions caught by the review",
                               "acceptance of the last 2 videos (learned)"), strict=True):
        for st in STRATEGIES:
            m, h = np.array(t[st][key]["mean"]), np.array(t[st][key]["ci95"])
            plots.curve(ax, x, m, h, LABEL[st], COLOR[st])
        plots.style_axes(ax, title, "review budget (% of suggestions looked at)")
        ax.set_xticks(x)
    axes[0].legend(loc="lower right")
    fig.suptitle("E8: asking the right questions (mean ± 95% CI over seeds × personas)")
    plots.save(fig, "e8_review_budget.png", out_dir)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="E8: review budget vs review strategy")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--videos", type=int, default=6)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    s = report(run(a.seeds, a.videos, a.workers))
    print(f"{'budget':>8}" + "".join(f"{st:>14}" for st in STRATEGIES) + "   (quality of the posted video)")
    for i, b in enumerate(s["budgets"]):
        print(f"{b * 100:>7.0f}%" + "".join(f"{s['table'][st]['posted']['mean'][i]:>14.3f}" for st in STRATEGIES))
    print("uncertainty beats random at budgets:", s["uncertainty_beats_random_at"])
    print("uncertainty beats timeline at budgets:", s["uncertainty_beats_timeline_at"])


if __name__ == "__main__":
    main()
