"""E4 (PLAN 10): drift. A Relaxed creator turns Snappy at video 6; LinTS with discount gamma in {1, 0.995, 0.98}.

Metric: acceptance per video against the persona's CURRENT ideal (the recovery curve) and how many videos after
the switch it takes to get back to RECOVERED acceptance.

    python -m sim.experiments.e4 [--seeds 10] [--videos 14]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.experiments import plots
from sim.experiments.common import acc_matrix, ci95, run_learning_task, run_parallel

GAMMAS = (1.0, 0.995, 0.98)
SWITCH = 6
RECOVERED = 0.80  # Snappy is the harder persona: its steady state is ~0.85, so 0.80 = "recovered"
COLORS = {1.0: "#999999", 0.995: "#D55E00", 0.98: "#0072B2"}


def run(seeds: int = 10, n_videos: int = 14, workers: int | None = None, gammas=GAMMAS) -> list[dict]:
    tasks = [{"persona": "drift", "algo": "lints", "seed": s, "n_videos": n_videos, "gamma": g,
              "exclude": ("relaxed", "snappy"), "tag": g, "persona_kw": {"at": SWITCH}}
             for g in gammas for s in range(1, seeds + 1)]
    return run_parallel(run_learning_task, tasks, workers)


def recovery(mean_curve: np.ndarray, switch: int = SWITCH, thr: float = RECOVERED) -> int | None:
    """Videos after the switch (switch video = 0) until acceptance is >= thr again; None if it never is."""
    for k in range(switch - 1, len(mean_curve)):
        if mean_curve[k] >= thr:
            return k - (switch - 1)
    return None


def report(results: list[dict], out_dir: Path | None = None) -> dict:
    gammas = sorted({r["tag"] for r in results}, reverse=True)
    n_videos = len(results[0]["videos"])
    x = np.arange(1, n_videos + 1)
    summary: dict = {}
    fig, ax = plots.plt.subplots(figsize=(8.5, 4.8))
    for g in gammas:
        rs = [r for r in results if r["tag"] == g]
        m, h = ci95(acc_matrix(rs))
        plots.curve(ax, x, m, h, f"γ = {g}", COLORS.get(g, None) or "k", lw=2.2)
        summary[str(g)] = {"mean_curve": m.tolist(), "at_switch": float(m[SWITCH - 1]),
                           "before_switch_mean": float(m[SWITCH - 3:SWITCH - 1].mean()),
                           "videos_to_recover": recovery(m)}
    ax.axvline(SWITCH - 0.5, color="k", ls="--", lw=1)
    ax.text(SWITCH - 0.4, 0.02, "creator switches\nRelaxed → Snappy", fontsize=8, va="bottom")
    plots.style_axes(ax, "acceptance vs current preference", "video #", (0, 1.02))
    ax.legend(loc="lower right")
    ax.set_title("E4: recovery after preference drift (mean ± 95% CI over seeds)")
    d = out_dir or plots.FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    plots.save(fig, "e4_drift.png", out_dir)
    (d / "e4_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="E4: drift recovery")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--videos", type=int, default=14)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    out = report(run(a.seeds, a.videos, a.workers))
    print(f"{'gamma':<8}{'before':>9}{'at switch':>11}{f'videos to ≥ {RECOVERED:.2f}':>18}")
    for g, s in out.items():
        print(f"{g:<8}{s['before_switch_mean']:>9.3f}{s['at_switch']:>11.3f}{s['videos_to_recover']!s:>18}")


if __name__ == "__main__":
    main()
