"""Does the logged confidence predict mistakes? Reliability of p(a|x) against the persona's ideal action, and the extra
figures for the paper (reliability, learning curve per creator).

    set PYTHONPATH=backend
    backend\\.venv\\Scripts\\python.exe paper\\scripts\\calibration.py
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sim.experiments.common import make_learner, style_mixture, videos_for
from sim.personas import BASE_PERSONAS, make_persona
from sim.session import run_user

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 1.3, "lines.markersize": 3.5, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": .5,
                     "legend.frameon": False})
EDGES = np.array([0, .3, .5, .7, .8, .9, .95, .99, 1.0001])


def task(t):
    name, seed, v = t
    learner = make_learner("lints", None, seed, 0.995, mixture=style_mixture((name,)), v=v)
    run = run_user(make_persona(name), learner, videos_for(seed, 10), seed, keep_log=True)
    return [(s.confidence, float(s.proposed_action == tr["ideal"]), tr["video"]) for s, tr in zip(run.suggestions, run.truth, strict=True)]


def table(rows):
    a = np.array(rows)
    out, ece = [], 0.0
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if m.sum():
            out.append({"lo": float(lo), "hi": float(min(hi, 1)), "n": int(m.sum()), "conf": float(a[m, 0].mean()), "acc": float(a[m, 1].mean())})
            ece += m.mean() * abs(a[m, 0].mean() - a[m, 1].mean())
    below = a[:, 0] < .7
    from scipy import stats

    return {"bins": out, "ece": float(ece), "n": len(a), "share_below_0.7": float(below.mean()), "acc_below_0.7": float(a[below, 1].mean()),
            "acc_above_0.7": float(a[~below, 1].mean()), "auc_error": float(stats.mannwhitneyu(a[a[:, 1] == 1, 0], a[a[:, 1] == 0, 0]).statistic / ((a[:, 1] == 1).sum() * (a[:, 1] == 0).sum()))}


if __name__ == "__main__":
    out = {}
    for v in (0.05, 0.25, 0.5):
        tasks = [(p, s, v) for p in BASE_PERSONAS for s in range(1, 7)]
        with ProcessPoolExecutor(max_workers=6) as ex:
            rows = [r for rr in ex.map(task, tasks, chunksize=2) for r in rr]
        out[str(v)] = table(rows)
        print(v, {k: out[str(v)][k] for k in ("ece", "share_below_0.7", "acc_below_0.7", "acc_above_0.7", "auc_error")}, flush=True)
    (ROOT / "data" / "calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(3.36, 2.0))
    ax.plot([0, 1], [0, 1], color="black", lw=.6, ls="--", label="Perfect calibration")
    for v, c, m in (("0.05", "#0072B2", "^"), ("0.25", "#D55E00", "o"), ("0.5", "#009E73", "s")):
        b = out[v]["bins"]
        ax.plot([x["conf"] for x in b], [x["acc"] for x in b], color=c, marker=m, label=f"v = {v}")
    ax.set(xlabel="Confidence of the suggestion", ylabel="Share that match the creator")
    ax.legend(loc="lower right")
    fig.savefig(ROOT / "figures" / "calibration.pdf", bbox_inches="tight", pad_inches=.02)

    main = json.loads((ROOT / "data" / "main.json").read_text(encoding="utf-8"))
    T = ("phrase", "pause", "filler", "zoom", "emphasis")
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 3.0), sharex=True, sharey=True)
    for ax, p in zip(axes.flat, BASE_PERSONAS):
        for lab, c, m, name in (("Rule", "#888888", "s", "Rule"), ("LinUCB", "#009E73", "D", "LinUCB"), ("LinTS", "#0072B2", "o", "LinTS, generic prior"), ("Proposed", "#D55E00", "*", "Proposed")):
            rs = [r for r in main if r["label"] == lab and r["persona"] == p]
            A = np.array([[sum(v["ok"][t] for t in T) / sum(v["n"][t] for t in T) for v in r["videos"]] for r in rs])
            h = 2.262 * A.std(axis=0, ddof=1) / np.sqrt(len(A))
            ax.plot(range(1, 11), A.mean(axis=0), color=c, marker=m, label=name)
            ax.fill_between(range(1, 11), A.mean(axis=0) - h, A.mean(axis=0) + h, color=c, alpha=.15, lw=0)
        ax.set_title(p.capitalize(), fontsize=8)
        ax.set_xticks([1, 5, 10])
    for ax in axes[1]:
        ax.set_xlabel("Video")
    for ax in axes[:, 0]:
        ax.set_ylabel("Acceptance")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(pad=.3)
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.07), fontsize=7)
    fig.savefig(ROOT / "figures" / "by_persona.pdf", bbox_inches="tight", pad_inches=.02)
    print("done")
