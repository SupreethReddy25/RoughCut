"""E1 (PLAN 10): does it learn each creator?

Every persona x {Rule, Global, eps-greedy, LinUCB, LinTS}, 10 videos, all decision types, 10 seeds.
Metrics: acceptance vs video index (overall + per decision type), corrections per minute, cumulative regret.
Rule / Global / eps / LinUCB / LinTS all see the same videos; Global and the learners start from the same generic
prior fit on the OTHER personas (leave-one-persona-out), so nothing is leaked from the persona being evaluated.

    python -m sim.experiments.e1 [--seeds 10] [--videos 10] [--workers 16]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.experiments import plots
from sim.experiments.common import (
    ALGO_LABEL,
    ALGOS,
    acc_matrix,
    ci95,
    corrections_per_minute,
    regret_curve,
    run_learning_task,
    run_parallel,
)
from sim.personas import BASE_PERSONAS
from sim.session import DECISION_TYPES

CHECK_AT = 5  # "by video 5"
CEILING = 0.95  # a baseline at/above this on a decision type IS the persona's rule there: it cannot be beaten
CEILING_REACHED = 0.90  # ... so LinTS must instead get there (>= this) by the last video


def run(seeds: int = 10, n_videos: int = 10, workers: int | None = None, personas=BASE_PERSONAS,
        algos=ALGOS, policy_kw: dict | None = None) -> list[dict]:
    tasks = [{"persona": p, "algo": a, "seed": s, "n_videos": n_videos, "gamma": 0.995,
              "policy_kw": (policy_kw or {}) if a in ("lints", "eps", "linucb") else {}, "tag": "e1"}
             for p in personas for a in algos for s in range(1, seeds + 1)]
    return run_parallel(run_learning_task, tasks, workers)


def select(results: list[dict], persona: str, algo: str) -> list[dict]:
    return [r for r in results if r["persona"] == persona and r["algo"] == algo]


def summary_table(results: list[dict], at: int = CHECK_AT) -> dict:
    """mean acceptance at video `at` per persona / algo / decision type (+ overall)."""
    out: dict = {}
    personas = sorted({r["persona"] for r in results}, key=list(BASE_PERSONAS).index)
    for p in personas:
        out[p] = {}
        for a in ALGOS:
            rs = select(results, p, a)
            if not rs:
                continue
            row = {"overall": float(np.nanmean(acc_matrix(rs)[:, at - 1]))}
            for dt in DECISION_TYPES:
                m = acc_matrix(rs, dt)[:, at - 1]
                row[dt] = float(np.nanmean(m)) if not np.all(np.isnan(m)) else float("nan")
            out[p][a] = row
    return out


def _cell(results: list[dict], persona: str, algo: str, dt: str | None, at: int) -> np.ndarray:
    return acc_matrix(select(results, persona, algo), dt)[:, at - 1]


def check(results: list[dict]) -> dict:
    """The Phase 2 acceptance criterion: LinTS beats Rule and Global by video 5 for every persona, overall and per
    decision type. Per cell the comparison is statistical: "ahead" (LinTS mean higher), "tied" (the baseline's mean
    lies inside LinTS's 95% CI over seeds) or "behind". A baseline already at the persona's exact rule (>= CEILING on
    that type) cannot be beaten, so there LinTS must instead reach CEILING_REACHED by the last video.
    overall_ok: LinTS ahead overall for every persona/baseline.  by_type_ok: no per-type cell is "behind"."""
    last_video = len(results[0]["videos"])
    overall_ok, by_type_ok = True, True
    counts = {"ahead": 0, "tied": 0, "behind": 0, "ceiling": 0}
    details: list[str] = []
    personas = sorted({r["persona"] for r in results}, key=list(BASE_PERSONAS).index)
    for p in personas:
        for base in ("rule", "global"):
            if not select(results, p, base):
                continue
            ts, bs = _cell(results, p, "lints", None, CHECK_AT), _cell(results, p, base, None, CHECK_AT)
            if not np.nanmean(ts) > np.nanmean(bs):
                overall_ok = False
                details.append(f"{p}: overall LinTS {np.nanmean(ts):.3f} <= {base} {np.nanmean(bs):.3f}")
            for dt in DECISION_TYPES:
                ts, bs = _cell(results, p, "lints", dt, CHECK_AT), _cell(results, p, base, dt, CHECK_AT)
                if np.all(np.isnan(ts)) or np.all(np.isnan(bs)):
                    continue
                m, b = float(np.nanmean(ts)), float(np.nanmean(bs))
                half = float(np.nan_to_num(ci95(ts)[1], nan=0.0))
                if m > b:
                    counts["ahead"] += 1
                elif b >= CEILING:
                    final = float(np.nanmean(_cell(results, p, "lints", dt, last_video)))
                    counts["ceiling"] += 1
                    ok = final >= CEILING_REACHED
                    by_type_ok &= ok
                    details.append(f"{p}/{dt}: {base} {b:.3f} IS the persona's rule; LinTS {m:.3f} at video {CHECK_AT} "
                                   f"-> {final:.3f} at video {last_video}" + ("" if ok else "  (NOT converged)"))
                elif b <= m + half:
                    counts["tied"] += 1
                    details.append(f"{p}/{dt}: LinTS {m:.3f} vs {base} {b:.3f}: tied (inside ±{half:.3f} CI)")
                else:
                    counts["behind"] += 1
                    by_type_ok = False
                    details.append(f"{p}/{dt}: LinTS {m:.3f} vs {base} {b:.3f}: BEHIND (outside ±{half:.3f} CI)")
    return {"overall_ok": overall_ok, "by_type_ok": by_type_ok, "cells": counts, "details": details}


def plot(results: list[dict], out_dir: Path | None = None) -> list[Path]:
    personas = sorted({r["persona"] for r in results}, key=list(BASE_PERSONAS).index)
    n_videos = len(results[0]["videos"])
    x = np.arange(1, n_videos + 1)
    paths = []
    fig, axes = plots.plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    fig.subplots_adjust(hspace=0.35)
    for ax, p in zip(axes.ravel(), personas, strict=False):
        for a in ALGOS:
            rs = select(results, p, a)
            if rs:
                m, h = ci95(acc_matrix(rs))
                plots.curve(ax, x, m, h, ALGO_LABEL[a], plots.PALETTE[a], lw=2.4 if a == "lints" else 1.5)
        ax.axvline(CHECK_AT, color="k", ls=":", lw=0.8)
        ax.set_title(p.capitalize())
        plots.style_axes(ax, "acceptance" if p in (personas[0], personas[3]) else None, "video #", (0, 1.02))
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("E1: acceptance vs video index (mean ± 95% CI over seeds; dotted = video 5)")
    paths.append(plots.save(fig, "e1_acceptance_overall.png", out_dir))

    fig, axes = plots.plt.subplots(len(DECISION_TYPES), len(personas), figsize=(2.6 * len(personas), 2.2 * len(DECISION_TYPES)),
                                   sharex=True, sharey=True)
    for i, dt in enumerate(DECISION_TYPES):
        for j, p in enumerate(personas):
            ax = axes[i, j]
            for a in ALGOS:
                rs = select(results, p, a)
                if rs:
                    m, h = ci95(acc_matrix(rs, dt))
                    plots.curve(ax, x, m, h, ALGO_LABEL[a], plots.PALETTE[a], lw=2.0 if a == "lints" else 1.0)
            ax.axvline(CHECK_AT, color="k", ls=":", lw=0.6)
            plots.style_axes(ax, dt if j == 0 else None, "video #" if i == len(DECISION_TYPES) - 1 else None, (0, 1.05))
            ax.set_xticks([1, 5, 10] if n_videos >= 10 else range(1, n_videos + 1))
            if i == 0:
                ax.set_title(p.capitalize(), fontsize=10)
    axes[0, 0].legend(fontsize=6, loc="lower right")
    fig.subplots_adjust(top=0.93, hspace=0.25)
    fig.suptitle("E1: acceptance per decision type")
    paths.append(plots.save(fig, "e1_acceptance_by_type.png", out_dir))

    fig, axes = plots.plt.subplots(1, 2, figsize=(11, 4))
    for a in ALGOS:
        cpm = np.mean([np.nanmean(corrections_per_minute(select(results, p, a)), axis=0) for p in personas], axis=0)
        reg = np.mean([regret_curve(select(results, p, a)).mean(axis=0) for p in personas], axis=0)
        axes[0].plot(x, cpm, color=plots.PALETTE[a], label=ALGO_LABEL[a], marker="o", ms=3, lw=2.2 if a == "lints" else 1.4)
        axes[1].plot(x, reg, color=plots.PALETTE[a], label=ALGO_LABEL[a], marker="o", ms=3, lw=2.2 if a == "lints" else 1.4)
    plots.style_axes(axes[0], "corrections per minute of footage", "video #")
    plots.style_axes(axes[1], "cumulative regret (proposals ≠ ideal)", "video #")
    axes[0].legend(fontsize=8)
    fig.suptitle("E1: effort and regret (mean over personas and seeds)")
    paths.append(plots.save(fig, "e1_effort_regret.png", out_dir))
    return paths


def report(results: list[dict], out_dir: Path | None = None) -> dict:
    table = summary_table(results)
    verdict = check(results)
    out = {"check_at_video": CHECK_AT, "table": table, "verdict": verdict}
    d = out_dir or plots.FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / "e1_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def print_table(table: dict) -> None:
    cols = ["overall", *DECISION_TYPES]
    print(f"\nAcceptance at video {CHECK_AT} (mean over seeds)")
    print(f"{'persona':<12}{'policy':<14}" + "".join(f"{c:>10}" for c in cols))
    for p, row in table.items():
        for a in ALGOS:
            if a in row:
                print(f"{p:<12}{ALGO_LABEL[a]:<14}" + "".join(f"{row[a][c]:>10.3f}" for c in cols))
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="E1: does it learn each creator?")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--videos", type=int, default=10)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    results = run(a.seeds, a.videos, a.workers)
    raw = Path(__file__).resolve().parents[3] / "data" / "experiments"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "e1_runs.json").write_text(json.dumps(results), encoding="utf-8")
    out = report(results)
    print_table(out["table"])
    for pth in plot(results):
        print("wrote", pth)
    v = out["verdict"]
    print(f"\nLinTS beats Rule & Global overall by video {CHECK_AT} for every persona: {v['overall_ok']}")
    print(f"... and per decision type (no cell behind; baselines that ARE the persona's rule: LinTS must converge): "
          f"{v['by_type_ok']}")
    print(f"per-type cells vs Rule/Global: {v['cells']}")
    for line in v["details"]:
        print("  -", line)

if __name__ == "__main__":
    main()
