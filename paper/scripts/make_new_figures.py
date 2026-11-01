"""Figures and numbers for the deep baselines and visual rhythm: figures/deep.pdf, figures/rhythm.pdf, data/deep_numbers.json.

    backend\\.venv\\Scripts\\python.exe paper\\scripts\\make_new_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
D, F = ROOT / "data", ROOT / "figures"
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.3, "lines.markersize": 3.2, "axes.grid": True,
                     "grid.color": "#dddddd", "grid.linewidth": 0.5, "legend.frameon": False})
T = ("phrase", "pause", "filler", "zoom", "emphasis")


def ci(a):
    a = np.asarray(a, dtype=float)
    return a.mean(axis=0), stats.t.ppf(.975, len(a) - 1) * a.std(axis=0, ddof=1) / np.sqrt(len(a))


def acc(r):
    return np.array([sum(v["ok"][t] for t in T) / sum(v["n"][t] for t in T) for v in r["videos"]])


def steps(r):
    return np.cumsum([sum(v["n"][t] for t in T) for v in r["videos"]])


def regret(r):
    return float(sum(sum(v["n"][t] - v["ok"][t] for t in T) for v in r["videos"]))


def ret(r):
    v = r["videos"][-1]
    n = sum(v["n"][t] for t in T)
    return (v["acc"] + .5 * v["imp"] - v["corr"]) / n


deep = json.loads((D / "deep.json").read_text(encoding="utf-8"))
main = json.loads((D / "main.json").read_text(encoding="utf-8"))
rows = [r for r in main if r["label"] in ("Rule", "LinUCB", "LinTS", "Proposed")] + deep["decisions"]
N: dict = {"chosen": deep["chosen"], "tuning": deep["tuning"], "decisions": {}, "rhythm": {}}
key = lambda r: (r["persona"], r["seed"])  # noqa: E731
base = sorted((r for r in rows if r["label"] == "Proposed"), key=key)
for lab in dict.fromkeys(r["label"] for r in rows):
    rs = sorted((r for r in rows if r["label"] == lab), key=key)
    A = np.array([acc(r) for r in rs])
    first = [float(steps(r)[np.flatnonzero(acc(r) >= .85)[0]]) if (acc(r) >= .85).any() else np.nan for r in rs]
    d = np.array([regret(a) - regret(b) for a, b in zip(rs, base, strict=True)])
    dv = A[:, 9] - np.array([acc(b)[9] for b in base])
    de = A[:, :3].mean(axis=1) - np.array([acc(b)[:3].mean() for b in base])
    N["decisions"][lab] = {
        "v1": float(A[:, 0].mean()), "v3": float(A[:, 2].mean()), "v5": float(A[:, 4].mean()), "v10": float(A[:, 9].mean()), "v10_ci": float(ci(A[:, 9])[1]),
        "early": float(A[:, :3].mean()), "regret": float(np.mean([regret(r) for r in rs])), "regret_ci": float(ci([regret(r) for r in rs])[1]),
        "success": float((A[:, 5:] >= .85).mean()), "steps_to_85": float(np.nanmedian(first)) if np.any(~np.isnan(first)) else None,
        "by_type_v10": {t: float(np.mean([r["videos"][9]["ok"][t] / max(r["videos"][9]["n"][t], 1) for r in rs])) for t in T},
        "secs": float(np.mean([r["secs"] for r in rs])),
        "vs_proposed": {"regret": float(d.mean()), "regret_ci": float(ci(d)[1]), "v10": float(dv.mean()), "v10_ci": float(ci(dv)[1]),
                        "early": float(de.mean()), "early_ci": float(ci(de)[1])}}

raw = json.loads((D / "e" / "e11_raw.json").read_text(encoding="utf-8"))
rh = {"Fixed rule": [r for r in raw if r["algo"] == "rule"], "Linear bandit": [r for r in raw if r["algo"] == "bandit"],
      "RLSVI": [r for r in raw if r["algo"] == "rlsvi"]}
for a in ("DQN", "PPO"):
    for w, name in ((False, f"{a}, blank"), (True, f"{a}, warm start")):
        rh[name] = [r for r in deep["rhythm"] if r["algo"] == a and r["warm"] == w]
rl = sorted(rh["RLSVI"], key=key)
for name, rs in rh.items():
    rs = sorted(rs, key=key)
    a, r = np.array([[c["acc"] for c in x["curve"]] for x in rs]), np.array([[c["rating"] for c in x["curve"]] for x in rs])
    d = a[:, -1] - np.array([x["curve"][-1]["acc"] for x in rl])
    d3 = a[:, 3] - np.array([x["curve"][3]["acc"] for x in rl])
    N["rhythm"][name] = {"v0": float(a[:, 0].mean()), "v3": float(a[:, 3].mean()), "final": float(a[:, -1].mean()), "final_ci": float(ci(a[:, -1])[1]),
                         "rating": float(r[:, -1].mean()), "rating_ci": float(ci(r[:, -1])[1]), "vs_rlsvi": float(d.mean()), "vs_rlsvi_ci": float(ci(d)[1]) if d.std() else 0.0,
                         "v3_vs_rlsvi": float(d3.mean()), "v3_vs_rlsvi_ci": float(ci(d3)[1]) if d3.std() else 0.0,
                         "by_persona": {p: float(np.mean([x["curve"][-1]["acc"] for x in rs if x["persona"] == p])) for p in ("steady", "buildup", "busy", "minimal")},
                         "secs": float(np.mean([x.get("secs", 0) for x in rs]))}
(D / "deep_numbers.json").write_text(json.dumps(N, indent=1), encoding="utf-8")

STY = {"Rule": ("#888888", "s", ":", "Rule"), "LinUCB": ("#009E73", "D", "--", "LinUCB"), "LinTS": ("#0072B2", "o", "-", "LinTS, generic prior"),
       "Proposed": ("#D55E00", "*", "-", "LinTS, style prior (proposed)"), "Neural": ("#CC79A7", "v", "-.", "Neural $\\epsilon$-greedy, blank"),
       "Neural, pretrained": ("#7F3C8D", "^", "-.", "Neural $\\epsilon$-greedy, pretrained")}
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.25))
for lab, (c, m, l, name) in STY.items():
    rs = [r for r in rows if r["label"] == lab]
    mu, h = ci([acc(r) for r in rs])
    x = np.arange(1, 11)
    axes[0].plot(x, mu, color=c, marker=m, ls=l, label=name)
    axes[0].fill_between(x, mu - h, mu + h, color=c, alpha=.14, lw=0)
    mu, h = ci([np.cumsum([sum(v["n"][t] - v["ok"][t] for t in T) for v in r["videos"]]) for r in rs])
    axes[1].plot(x, mu, color=c, marker=m, ls=l, label=name)
    axes[1].fill_between(x, mu - h, mu + h, color=c, alpha=.14, lw=0)
axes[0].set(xlabel="Video", ylabel="Acceptance", title="(a) Acceptance per video")
axes[1].set(xlabel="Video", ylabel="Cumulative regret", title="(b) Cumulative regret")
h, l = axes[0].get_legend_handles_labels()
fig.tight_layout(pad=.3)
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.17), fontsize=7)
fig.savefig(F / "deep.pdf", bbox_inches="tight", pad_inches=.02)

COL = {"Fixed rule": ("#888888", "s", ":"), "Linear bandit": ("#0072B2", "o", "--"), "RLSVI": ("#D55E00", "*", "-"), "DQN, blank": ("#009E73", "v", "-."),
       "DQN, warm start": ("#009E73", "^", "-"), "PPO, blank": ("#CC79A7", "D", "-."), "PPO, warm start": ("#CC79A7", "P", "-")}
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.25))
for name, (c, m, l) in COL.items():
    rs = rh[name]
    x = np.arange(0, 9)
    for ax, k in zip(axes, ("acc", "rating"), strict=True):
        mu, h = ci([[p[k] for p in r["curve"]] for r in rs])
        ax.plot(x, mu, color=c, marker=m, ls=l, label=name)
        ax.fill_between(x, mu - h, mu + h, color=c, alpha=.12, lw=0)
axes[0].set(xlabel="Videos edited", ylabel="Share of beats matching", title="(a) Match with the creator's ideal, unseen videos")
axes[1].set(xlabel="Videos edited", ylabel="Whole-video rating", title="(b) Delayed rating, unseen videos")
h, l = axes[0].get_legend_handles_labels()
fig.tight_layout(pad=.3)
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.17), fontsize=7)
fig.savefig(F / "rhythm.pdf", bbox_inches="tight", pad_inches=.02)


def rnd(o):
    return round(o, 3) if isinstance(o, float) else {k: rnd(v) for k, v in o.items()} if isinstance(o, dict) else o


print(json.dumps(rnd({k: v for k, v in N.items() if k != "tuning"})))
