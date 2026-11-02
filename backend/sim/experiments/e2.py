"""E2 (PLAN 10): does the style prior fix cold start?

Two questions, both answered in the simulator:

1. Cold start: LinTS with {no prior, pooled generic prior, style-mixture prior}, leave-one-persona-out (the persona's
   own prototype is NOT in the prior), acceptance on videos 1-3.
2. Identification: how many user decisions (feedback events) until the style posterior puts more than 0.8 on the
   persona's own style? Measured with the plain PLAN model (sigma = 0.7, no tempering) and with the PRODUCT settings
   (sigma = 0.7 posterior, prior tempered with sigma = 3, lambda = 10), the one the app really runs.

    python -m sim.experiments.e2 [--seeds 10] [--videos 6] [--id-videos 3]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from app.policy.log_replay import Learner
from app.policy.style import IDENTIFIED_ABOVE
from sim.experiments import plots
from sim.experiments.common import (
    acc_matrix,
    ci95,
    make_policy,
    run_learning_task,
    run_parallel,
    style_mixture,
    videos_for,
)
from sim.personas import BASE_PERSONAS, make_persona
from sim.session import run_user

PRIORS = ("none", "generic", "mixture")
PRIOR_LABEL = {"none": "no prior", "generic": "pooled generic prior", "mixture": "style-mixture prior"}
PRIOR_COLOR = {"none": "#999999", "generic": "#0072B2", "mixture": "#D55E00"}
EARLY = 3  # "acceptance on videos 1-3"
PRODUCT = {"lam": 10.0, "lam_by_type": {"phrase": 30.0}, "prior_sigma": 3.0}  # = app/appctx.py
MODELS = {"plain": {}, "product": PRODUCT}  # the PLAN model vs the settings the app runs


class _Tracking(Learner):
    """A learner that records the posterior weight of one style after every feedback event."""

    def __init__(self, *a, true: str, **k):
        super().__init__(*a, **k)
        self.true = true
        self.curve: list[float] = []

    def apply(self, *a, **k):
        ups = super().apply(*a, **k)
        if ups and self.mixture is not None:
            self.curve.append(self.mixture.weights()[self.true])
        return ups


def identification_task(task: dict) -> dict:
    """One persona, one seed: weight of the true style after each event. The persona's own prototype IS available."""
    name, seed, kw = task["persona"], task["seed"], dict(MODELS[task["model"]])
    mix = style_mixture(())
    mix.prior_sigma = kw.pop("prior_sigma", None)
    learner = _Tracking(make_policy("lints", mix.mean_prior(), seed, **kw), mix, true=name)
    run_user(make_persona(name), learner, videos_for(seed, task["videos"]), seed)
    return {"persona": name, "seed": seed, "model": task["model"], "curve": learner.curve}


def decisions_to_identify(curve: list[float], thr: float = IDENTIFIED_ABOVE, stable: bool = False) -> int | None:
    """Number of feedback events until the weight first exceeds `thr` (the PLAN's criterion); with `stable` until it
    exceeds it for good (never falls back below `thr` to the end of the run). None if that never happens."""
    if not stable:
        return next((i for i, w in enumerate(curve, 1) if w > thr), None)
    last_below = 0
    for i, w in enumerate(curve, 1):
        if w <= thr:
            last_below = i
    return last_below + 1 if last_below < len(curve) else None


def run(seeds: int = 10, n_videos: int = 6, id_videos: int = 3, workers: int | None = None) -> dict:
    cold = [{"persona": p, "algo": "lints", "seed": s, "n_videos": n_videos, "prior": pr, "tag": pr}
            for p in BASE_PERSONAS for pr in PRIORS for s in range(1, seeds + 1)]
    ident = [{"persona": p, "seed": s, "videos": id_videos, "model": m}
             for p in BASE_PERSONAS for m in MODELS for s in range(1, seeds + 1)]
    return {"cold": run_parallel(run_learning_task, cold, workers),
            "ident": run_parallel(identification_task, ident, workers)}


def cold_start_table(cold: list[dict]) -> dict:
    """{prior: {"early": mean acceptance of videos 1-3, "by_persona": {...}, "mean_curve": [...]}}."""
    out = {}
    for pr in PRIORS:
        rs = [r for r in cold if r["tag"] == pr]
        acc = acc_matrix(rs)
        out[pr] = {"early": float(np.nanmean(acc[:, :EARLY])), "mean_curve": np.nanmean(acc, axis=0).tolist(),
                   "by_persona": {p: float(np.nanmean(acc_matrix([r for r in rs if r["persona"] == p])[:, :EARLY]))
                                  for p in BASE_PERSONAS}}
    return out


def identification_table(ident: list[dict]) -> dict:
    out: dict = {}
    for model in MODELS:
        per = {}
        for p in BASE_PERSONAS:
            rs = [r for r in ident if r["model"] == model and r["persona"] == p]
            n = [decisions_to_identify(r["curve"]) for r in rs]
            found = [x for x in n if x is not None]
            stable = [x for x in (decisions_to_identify(r["curve"], stable=True) for r in rs) if x is not None]
            per[p] = {"median": float(np.median(found)) if found else None, "max": max(found) if found else None,
                      "identified": f"{len(found)}/{len(n)}",
                      "stable_median": float(np.median(stable)) if stable else None}
        medians = [v["median"] for v in per.values() if v["median"] is not None]
        out[model] = {"by_persona": per, "median_over_personas": float(np.median(medians)) if medians else None}
    return out


def report(res: dict, out_dir: Path | None = None) -> dict:
    cold, ident = res["cold"], res["ident"]
    table = {"cold_start": cold_start_table(cold), "identification": identification_table(ident),
             "threshold": IDENTIFIED_ABOVE}
    d = out_dir or plots.FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / "e2_summary.json").write_text(json.dumps(table, indent=1), encoding="utf-8")

    n_videos = len(cold[0]["videos"])
    x = np.arange(1, n_videos + 1)
    fig, ax = plots.plt.subplots(figsize=(8, 4.8))
    for pr in PRIORS:
        m, h = ci95(acc_matrix([r for r in cold if r["tag"] == pr]))
        plots.curve(ax, x, m, h, PRIOR_LABEL[pr], PRIOR_COLOR[pr], lw=2.2)
    ax.axvspan(0.6, EARLY + 0.4, color="k", alpha=0.05)
    plots.style_axes(ax, "acceptance", "video #", (0.3, 1.02))
    ax.legend(loc="lower right")
    ax.set_title("E2: cold start, leave-one-persona-out (mean ± 95% CI over seeds × personas)")
    plots.save(fig, "e2_cold_start.png", out_dir)

    fig, axes = plots.plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    colors = plots.plt.get_cmap("tab10")
    for ax, model, title in zip(axes, MODELS, ("PLAN model (sigma 0.7)", "product (prior tempered, lambda 10)"), strict=True):
        for i, p in enumerate(BASE_PERSONAS):
            curves = [r["curve"] for r in ident if r["model"] == model and r["persona"] == p]
            n = min(len(c) for c in curves)
            m = np.mean([c[:n] for c in curves], axis=0)
            ax.plot(np.arange(1, n + 1), m, color=colors(i), lw=1.8, label=p)
        ax.axhline(IDENTIFIED_ABOVE, color="k", ls="--", lw=1)
        ax.axvline(30, color="k", ls=":", lw=1)
        ax.set_xlim(0, 60)
        plots.style_axes(ax, None, "feedback events", (0, 1.02))
        ax.set_title(title)
    axes[0].set_ylabel("posterior weight of the persona's own style")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("E2: how fast is the style identified?  (dashed: 0.8, dotted: 30 events)")
    plots.save(fig, "e2_identification.png", out_dir)
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description="E2: style prior vs cold start; decisions to identify the style")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--videos", type=int, default=6)
    ap.add_argument("--id-videos", type=int, default=3)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    t = report(run(a.seeds, a.videos, a.id_videos, a.workers))
    print("cold start, mean acceptance of videos 1-3 (leave-one-persona-out):")
    for pr in PRIORS:
        print(f"  {PRIOR_LABEL[pr]:<24} {t['cold_start'][pr]['early']:.3f}")
    print(f"events until the true style has weight > {IDENTIFIED_ABOVE}:")
    for model, v in t["identification"].items():
        print(f"  {model:<8} median over personas {v['median_over_personas']}  " +
              "  ".join(f"{p}:{d['median']}({d['identified']})" for p, d in v["by_persona"].items()))
        print(f"  {'':<8} ... and for good: " + "  ".join(f"{p}:{d['stable_median']}" for p, d in v["by_persona"].items()))


if __name__ == "__main__":
    main()
