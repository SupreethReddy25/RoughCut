"""E9 (PLAN 7.4, 10): off-policy evaluation. Can we tell how good a DIFFERENT policy would have been, using only
the feedback logged while LinTS was making the suggestions?

For every persona x seed a creator edits `--videos` videos with LinTS suggesting (the logging policy pi_b; every
suggestion stores its Thompson-sampling propensity). Then a set of target policies is evaluated OFFLINE on those
logs with DM / IPS / SNIPS / DR (app/ope). In the simulator the true value of each target on the same contexts is
known exactly (the persona's ideal action and feedback noise), so each estimator's bias, RMSE, confidence-interval
coverage, and whether it ranks the policies correctly can be measured.

Targets: the learned LinTS (posterior mean after the run), the fixed rule baseline (what simple editors do), the
generic prior, and each style prototype (= "what if this creator were edited in style X").

    python -m sim.experiments.e9 [--seeds 6] [--videos 6]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from app.ope.estimators import ESTIMATORS, evaluate, greedy_target
from app.ope.logs import logged_from, true_reward
from app.policy.linear import GlobalPolicy
from app.policy.rule import RulePolicy
from sim.experiments import plots
from sim.experiments.common import generic_prior, make_learner, prototype, run_parallel, videos_for
from sim.personas import BASE_PERSONAS, make_persona
from sim.session import run_user

STYLES = ("snappy", "relaxed", "storyteller", "minimalist", "focused")
COLOR = {"DM": "#999999", "IPS": "#0072B2", "SNIPS": "#009E73", "DR": "#E69F00", "SNDR": "#D55E00"}


def _det(choose):
    """A deterministic target from `choose(dt, ctx) -> action`."""
    return lambda dt, ctx, arms: {a: float(a == choose(dt, ctx)) for a in arms}


def e9_task(task: dict) -> dict:
    name, seed = task["persona"], task["seed"]
    prior = generic_prior((name,))
    learner = make_learner("lints", prior, seed, 0.995)
    run = run_user(make_persona(name), learner, videos_for(seed, task["videos"]), seed, keep_log=True)
    sugs = {s.id: s for s in run.suggestions}
    truth = {}
    for s, t in zip(run.suggestions, run.truth, strict=True):
        arms = list(s.candidates) if s.decision_type == "emphasis" and s.candidates else None
        from app.policy import context

        truth[s.id] = true_reward(arms or context.arms_for(s.decision_type, s.context_raw), t["ideal"], t["noise"],
                                  t["p_override"])
    data = logged_from(sugs, run.events, truth)
    rule = RulePolicy(seed)
    targets = {"LinTS learned (ours)": greedy_target(learner.policy),
               "rule (simple editor)": _det(lambda dt, ctx: rule.choose(dt, ctx).action),
               "generic prior": greedy_target(GlobalPolicy(prior))}
    for st in STYLES:
        if st != name:
            targets[f"style: {st}"] = greedy_target(GlobalPolicy(prototype(st)))
    rows = []
    for tname, target in targets.items():
        r = evaluate(data, target, n_boot=task["boot"], seed=seed)
        rows.append({"target": tname, "truth": r.truth, "est": r.estimates, "ci": r.ci, "ess": r.ess,
                     "max_w": r.max_weight, "n": r.n})
    return {"persona": name, "seed": seed, "rows": rows}


def run(seeds: int = 6, n_videos: int = 6, boot: int = 300, workers: int | None = None) -> list[dict]:
    tasks = [{"persona": p, "seed": s, "videos": n_videos, "boot": boot}
             for p in BASE_PERSONAS for s in range(1, seeds + 1)]
    return run_parallel(e9_task, tasks, workers)


def _kendall(a: list[float], b: list[float]) -> float:
    n, c, d = len(a), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            c += s > 0
            d += s < 0
    return (c - d) / max(c + d, 1)


def report(results: list[dict], out_dir: Path | None = None) -> dict:
    stats: dict = {}
    for e in ESTIMATORS:
        err = np.array([r["est"][e] - r["truth"] for res in results for r in res["rows"]])
        cover = np.array([r["ci"][e][0] <= r["truth"] <= r["ci"][e][1] for res in results for r in res["rows"]])
        picks, taus = [], []
        for res in results:
            tr = [r["truth"] for r in res["rows"]]
            es = [r["est"][e] for r in res["rows"]]
            picks.append(int(np.nanargmax(es)) == int(np.argmax(tr)))
            taus.append(_kendall(es, tr))
        stats[e] = {"bias": float(np.nanmean(err)), "rmse": float(np.sqrt(np.nanmean(err ** 2))),
                    "ci95_coverage": float(np.mean(cover)), "picks_best_policy": float(np.mean(picks)),
                    "kendall_tau": float(np.mean(taus))}
    ess = float(np.mean([r["ess"] / r["n"] for res in results for r in res["rows"] if r["n"]]))
    n_logged = float(np.mean([res["rows"][0]["n"] for res in results]))
    by_target: dict = {}
    for res in results:
        for r in res["rows"]:
            key = r["target"] if not r["target"].startswith("style:") else "style prototypes (other creators)"
            by_target.setdefault(key, {"truth": [], **{e: [] for e in ESTIMATORS}})
            by_target[key]["truth"].append(r["truth"])
            for e in ESTIMATORS:
                by_target[key][e].append(r["est"][e])
    targets = {k: {kk: float(np.nanmean(v)) for kk, v in d.items()} for k, d in by_target.items()}
    summary = {"estimators": stats, "mean_logged_decisions": n_logged, "mean_ess_share": ess, "targets": targets}
    d = out_dir or plots.FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / "e9_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    fig, axes = plots.plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    fig.subplots_adjust(wspace=0.75)
    ax = axes[0]
    for e in ESTIMATORS:
        tr = [r["truth"] for res in results for r in res["rows"]]
        es = [r["est"][e] for res in results for r in res["rows"]]
        ax.scatter(tr, es, s=9, alpha=0.55, color=COLOR[e], label=e)
    lo, hi = ax.get_xlim()
    ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--")
    plots.style_axes(ax, "estimated value (offline)", "true value (simulator)")
    ax.legend(loc="upper left")
    ax.set_title("each dot = one target policy for one creator")
    ax = axes[1]
    x = np.arange(len(ESTIMATORS))
    ax.bar(x - 0.2, [stats[e]["rmse"] for e in ESTIMATORS], 0.4, color=[COLOR[e] for e in ESTIMATORS], label="RMSE")
    ax.bar(x + 0.2, [abs(stats[e]["bias"]) for e in ESTIMATORS], 0.4, color=[COLOR[e] for e in ESTIMATORS], alpha=0.4,
           label="|bias|")
    ax.set_xticks(x, ESTIMATORS)
    plots.style_axes(ax, "error of the value estimate")
    ax.legend()
    ax.set_title("accuracy of the estimators")
    ax = axes[2]
    names = list(targets)
    y = np.arange(len(names))
    ax.barh(y - 0.2, [targets[k]["truth"] for k in names], 0.4, color="#444444", label="true")
    ax.barh(y + 0.2, [targets[k]["SNDR"] for k in names], 0.4, color=COLOR["SNDR"], label="SNDR estimate (offline)")
    ax.set_yticks(y, names)
    ax.set_xlim(0.4, 1.0)
    plots.style_axes(ax, None, "expected share of suggestions the creator keeps")
    ax.legend(loc="upper right")
    ax.set_title("which policy would have been best?")
    fig.suptitle("E9: off-policy evaluation from LinTS's logged propensities")
    plots.save(fig, "e9_ope.png", out_dir)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="E9: off-policy evaluation in the simulator")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--videos", type=int, default=6)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    s = report(run(a.seeds, a.videos, a.boot, a.workers))
    print(f"logged decisions per creator: {s['mean_logged_decisions']:.0f}; ESS share {s['mean_ess_share']:.2f}")
    print(f"{'':8}{'bias':>9}{'RMSE':>9}{'CI cov':>9}{'best?':>9}{'tau':>8}")
    for e, v in s["estimators"].items():
        print(f"{e:8}{v['bias']:>9.3f}{v['rmse']:>9.3f}{v['ci95_coverage']:>9.2f}{v['picks_best_policy']:>9.2f}"
              f"{v['kendall_tau']:>8.2f}")
    for k, v in s["targets"].items():
        print(f"  {k:36} true {v['truth']:.3f}  SNDR {v['SNDR']:.3f}  DR {v['DR']:.3f}  SNIPS {v['SNIPS']:.3f}"
              f"  IPS {v['IPS']:.3f}  DM {v['DM']:.3f}")


if __name__ == "__main__":
    main()
