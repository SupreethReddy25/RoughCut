"""Figure and table numbers from the final fair comparison: figures/long.pdf and data/final_numbers.json.

    backend\\.venv\\Scripts\\python.exe paper\\scripts\\final_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
long = json.loads((ROOT / "data" / "final" / "long.json").read_text(encoding="utf-8"))
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.3, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": .5, "legend.frameon": False})
by: dict = {}
for r in long:
    by.setdefault(r["label"], []).append(r)


def acc(v):
    return sum(v["correct"].values()) / sum(v["n"].values())


numbers = {}
for lab, rs in by.items():
    A = np.array([[acc(v) for v in r["videos"]] for r in rs])
    E = np.array([[sum(v["n"].values()) - sum(v["correct"].values()) for v in r["videos"]] for r in rs])
    N = np.array([[sum(v["n"].values()) for v in r["videos"]] for r in rs])
    steps = []
    for a, n in zip(A[:, :10], N[:, :10]):
        hit = np.where(a >= .85)[0]
        steps.append(float(n[:hit[0] + 1].sum()) if len(hit) else np.nan)
    numbers[lab] = {"v1": float(A[:, 0].mean()), "v5": float(A[:, 4].mean()), "v10": float(A[:, 9].mean()), "v10_ci": float(2.001 * A[:, 9].std(ddof=1) / np.sqrt(len(A))), "v50": float(A[:, 49].mean()),
                    "v50_ci": float(2.001 * A[:, 49].std(ddof=1) / np.sqrt(len(A))), "success": float((A[:, 5:10] >= .85).mean()), "steps": float(np.nanmedian(steps)) if not np.isnan(steps).all() else None,
                    "regret10": float(E[:, :10].sum(1).mean()), "regret10_ci": float(2.001 * E[:, :10].sum(1).std(ddof=1) / np.sqrt(len(E))), "regret50": float(E.sum(1).mean()),
                    "regret50_ci": float(2.001 * E.sum(1).std(ddof=1) / np.sqrt(len(E)))}
(ROOT / "data" / "final_numbers.json").write_text(json.dumps(numbers, indent=1), encoding="utf-8")
for k, v in numbers.items():
    print(k, {a: round(b, 3) if b is not None else None for a, b in v.items()})

STYLE = (("Proposed", "Proposed, default", "#D55E00", "-"), ("Proposed, retuned", "Proposed, retuned", "#7F2704", "-"), ("LinTS", "LinTS, generic prior", "#0072B2", "-"),
         ("Neural", "Neural, blank", "#009E73", "--"), ("Neural, pretrained", "Neural, pretrained", "#56B4E9", "--"), ("NeuralLinear, pretrained", "NeuralLinear, pretrained", "#CC79A7", "--"))
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
for key, name, color, ls in STYLE:
    rs = by[key]
    A = np.array([[acc(v) for v in r["videos"]] for r in rs])
    E = np.array([[sum(v["n"].values()) - sum(v["correct"].values()) for v in r["videos"]] for r in rs]).cumsum(axis=1)
    x = np.arange(1, 51)
    smooth = np.array([A[:, max(0, i - 2):i + 1].mean() for i in range(50)])
    axes[0].plot(x, smooth, color=color, ls=ls, label=name)
    axes[1].plot(x, E.mean(axis=0), color=color, ls=ls, label=name)
axes[0].set(xlabel="Video", ylabel="Acceptance (3-video mean)", title="(a) Acceptance", ylim=(.68, .95))
axes[1].set(xlabel="Video", ylabel="Cumulative wrong proposals", title="(b) Regret")
axes[0].title.set_fontsize(8)
axes[1].title.set_fontsize(8)
h, l = axes[0].get_legend_handles_labels()
fig.tight_layout(pad=.3, rect=(0, .12, 1, 1))
fig.legend(h, l, loc="lower center", ncol=3)
fig.savefig(ROOT / "figures" / "long.pdf", bbox_inches="tight", pad_inches=.02)
