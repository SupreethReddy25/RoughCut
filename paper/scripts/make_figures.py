"""paper/data/*.json -> paper/figures/*.pdf and paper/data/numbers.json (every number quoted in the paper's tables).

    backend\\.venv\\Scripts\\python.exe paper\\scripts\\make_figures.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
D, F = ROOT / "data", ROOT / "figures"
F.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.3, "lines.markersize": 3.5, "axes.grid": True,
                     "grid.color": "#dddddd", "grid.linewidth": 0.5, "legend.frameon": False})
ONE, TWO = 3.45, 7.1
TYPES = ("phrase", "pause", "filler", "zoom", "emphasis")
TNAME = {"phrase": "Keep/cut", "pause": "Pause", "filler": "Filler", "zoom": "Zoom", "emphasis": "Emphasis"}
STY = {"Rule": ("#888888", "s", ":"), "Global": ("#E69F00", "v", ":"), "eps-greedy": ("#56B4E9", "^", "--"),
       "LinUCB": ("#009E73", "D", "--"), "LinTS": ("#0072B2", "o", "-"), "Proposed": ("#D55E00", "*", "-"),
       "Product": ("#CC79A7", "P", "-")}
LABEL = {"eps-greedy": "$\\epsilon$-greedy", "Global": "Global prior", "LinTS": "LinTS, generic prior",
         "Proposed": "LinTS, style prior (proposed)", "Product": "Proposed, product settings", "Rule": "Rule"}
PERSONAS = ("snappy", "relaxed", "storyteller", "minimalist", "focused", "blend")
N: dict = {}


def load(name):
    p = D / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def e(name):
    return json.loads((D / "e" / f"{name}_summary.json").read_text(encoding="utf-8"))


def acc(r, types=None):
    ts = types or TYPES
    return np.array([sum(v["ok"][t] for t in ts) / max(sum(v["n"][t] for t in ts), 1) for v in r["videos"]])


def acc_t(r, t):
    return np.array([v["ok"][t] / v["n"][t] if v["n"][t] else np.nan for v in r["videos"]])


def ret(r):
    return np.array([(v["acc"] - v["corr"] + 0.5 * v["imp"]) / max(sum(v["n"].values()), 1) for v in r["videos"]])


def steps(r):
    return np.cumsum([sum(v["n"].values()) for v in r["videos"]])


def regret(r):
    return np.cumsum([sum(v["n"].values()) - sum(v["ok"].values()) for v in r["videos"]])


def ci(x):
    """Mean and 95 % half-width over creators' runs."""
    x = np.asarray(x, float)
    n = np.sum(~np.isnan(x), axis=0)
    return np.nanmean(x, axis=0), stats.t.ppf(.975, np.maximum(n - 1, 1)) * np.nanstd(x, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))


def band(ax, x, m, h, label, color, marker="o", ls="-", every=1):
    ax.plot(x, m, color=color, marker=marker, ls=ls, label=label, markevery=every)
    ax.fill_between(x, m - h, m + h, color=color, alpha=.15, lw=0)


def save(fig, name):
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def by_label(rs):
    out = defaultdict(list)
    for r in rs:
        out[r["label"]].append(r)
    return out


# ---------------------------------------------------------------- main comparison
main = by_label(load("main"))
ALG = ("Rule", "Global", "eps-greedy", "LinUCB", "LinTS", "Proposed", "Product")
fig, axes = plt.subplots(1, 2, figsize=(TWO, 2.3))
for a in ALG:
    c, m, l = STY[a]
    rs = main[a]
    x = np.mean([steps(r) for r in rs], axis=0)
    mu, h = ci([ret(r) for r in rs])
    band(axes[0], x, mu, h, LABEL.get(a, a), c, m, l)
    mu, h = ci([regret(r) for r in rs])
    band(axes[1], x, mu, h, LABEL.get(a, a), c, m, l)
axes[0].set(xlabel="Environment steps (editing decisions)", ylabel="Mean return per decision", title="(a) Episodic return")
axes[1].set(xlabel="Environment steps (editing decisions)", ylabel="Cumulative regret (wrong proposals)", title="(b) Cumulative regret")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.24), fontsize=7)
save(fig, "learning")

rows = {}
for a in ALG:
    rs = main[a]
    A = np.array([acc(r) for r in rs])
    T = np.array([np.mean(r_) for r_ in A])
    first = []
    for r in rs:
        hit = np.flatnonzero(acc(r) >= .85)
        first.append(float(steps(r)[hit[0]]) if len(hit) else np.nan)
    by_seed = defaultdict(list)
    for r in rs:
        by_seed[r["seed"]].append(acc(r)[-1])
    seed_means = [np.mean(v) for v in by_seed.values()]
    mins = sum(sum(v["min"] for v in r["videos"][5:]) for r in rs)
    rows[a] = {"v1": float(A[:, 0].mean()), "v5": float(A[:, 4].mean()), "v10": float(A[:, 9].mean()),
               "v10_ci": float(ci(A[:, 9])[1]), "early": float(A[:, :3].mean()), "regret": float(np.mean([regret(r)[-1] for r in rs])),
               "regret_ci": float(ci([regret(r)[-1] for r in rs])[1]),
               "decisions": float(np.mean([steps(r)[-1] for r in rs])), "success": float((A[:, 5:] >= .85).mean()),
               "reach_rate": float(np.mean(~np.isnan(first))), "steps_to_85": float(np.nanmedian(first)) if np.any(~np.isnan(first)) else None,
               "corr_per_min": float(sum(sum(v["corr"] for v in r["videos"][5:]) for r in rs) / mins),
               "seed_sd": float(np.std(seed_means, ddof=1)), "run_sd": float(A[:, 9].std(ddof=1)), "return_v10": float(np.mean([ret(r)[-1] for r in rs])),
               "by_type_v10": {t: float(np.nanmean([acc_t(r, t)[-1] for r in rs])) for t in TYPES},
               "by_persona_v10": {p: float(np.mean([acc(r)[-1] for r in rs if r["persona"] == p])) for p in PERSONAS},
               "secs": float(np.mean([r["secs"] for r in rs]))}
N["main"] = rows

fig, axes = plt.subplots(1, 5, figsize=(TWO, 1.75), sharey=True)
for ax, t in zip(axes, TYPES):
    for a in ("Rule", "Global", "LinUCB", "Proposed"):
        c, m, l = STY[a]
        mu, h = ci([acc_t(r, t) for r in main[a]])
        band(ax, np.arange(1, 11), mu, h, LABEL.get(a, a), c, m, l)
    ax.set(title=TNAME[t], xlabel="Video", xticks=[1, 5, 10], ylim=(0, 1.02))
axes[0].set_ylabel("Acceptance")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.2), fontsize=7)
save(fig, "by_type")

fig, ax = plt.subplots(figsize=(ONE, 2.25))
pos = np.arange(len(PERSONAS))
for i, (a, off) in enumerate((("Rule", -.27), ("LinTS", 0), ("Proposed", .27))):
    data = [[acc(r)[-1] for r in main[a] if r["persona"] == p] for p in PERSONAS]
    bp = ax.boxplot(data, positions=pos + off, widths=.22, patch_artist=True, showfliers=True, flierprops={"markersize": 1.5},
                    medianprops={"color": "black", "lw": .8}, boxprops={"lw": .5}, whiskerprops={"lw": .5}, capprops={"lw": .5})
    for b in bp["boxes"]:
        b.set_facecolor(STY[a][0])
    ax.plot([], [], color=STY[a][0], lw=4, label=LABEL.get(a, a))
ax.set_xticks(pos, [p.capitalize() for p in PERSONAS], rotation=20)
ax.set(ylabel="Acceptance on video 10")
ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=3, fontsize=6)
save(fig, "seeds")

# ---------------------------------------------------------------- ablation
ab = by_label(load("ablation"))
full = {(r["persona"], r["seed"]): r for r in ab["Full"]}
rows = {}
for lab, rs in ab.items():
    d_early, d_late = [], []
    for r in rs:
        f = full[(r["persona"], r["seed"])]
        d_early.append(acc(r)[:3].mean() - acc(f)[:3].mean())
        d_late.append(acc(r)[7:].mean() - acc(f)[7:].mean())
    rows[lab] = {"early": float(np.mean([acc(r)[:3].mean() for r in rs])), "late": float(np.mean([acc(r)[7:].mean() for r in rs])),
                 "d_early": float(np.mean(d_early)), "d_early_ci": float(ci(d_early)[1]), "d_late": float(np.mean(d_late)),
                 "d_late_ci": float(ci(d_late)[1]), "regret": float(np.mean([regret(r)[-1] for r in rs]))}
N["ablation"] = rows
labs = [k for k in rows if k != "Full"]
labs.sort(key=lambda k: rows[k]["d_late"])
fig, ax = plt.subplots(figsize=(ONE, 2.25))
y = np.arange(len(labs))
ax.barh(y + .19, [rows[k]["d_early"] for k in labs], .36, xerr=[rows[k]["d_early_ci"] for k in labs], color="#56B4E9", label="Videos 1-3", error_kw={"lw": .6})
ax.barh(y - .19, [rows[k]["d_late"] for k in labs], .36, xerr=[rows[k]["d_late_ci"] for k in labs], color="#D55E00", label="Videos 8-10", error_kw={"lw": .6})
ax.set_yticks(y, labs)
ax.axvline(0, color="black", lw=.6)
ax.set(xlabel="Change in acceptance against the full method")
ax.legend(loc="lower right")
save(fig, "ablation")

# ---------------------------------------------------------------- sensitivity
se = by_label(load("sensitivity"))
groups = {"lambda": ("Prior strength $\\lambda$", True), "v": ("Posterior scale $v$", True), "gamma": ("Discount $\\gamma$", False),
          "eps": ("Exploration floor $\\epsilon$", False), "implicit": ("Implicit reward", False)}
rows = {}
fig, axes = plt.subplots(1, 5, figsize=(TWO, 1.6), sharey=True)
for ax, (g, (title, logx)) in zip(axes, groups.items()):
    keys = sorted((k for k in se if k.startswith(g + "=")), key=lambda k: float(k.split("=")[1]))
    xs = [float(k.split("=")[1]) for k in keys]
    early = [ci([acc(r)[:3].mean() for r in se[k]]) for k in keys]
    late = [ci([acc(r)[7:].mean() for r in se[k]]) for k in keys]
    rows[g] = {k.split("=")[1]: {"early": float(a[0]), "late": float(b[0]), "late_ci": float(b[1]), "regret": float(np.mean([regret(r)[-1] for r in se[k]]))}
               for k, a, b in zip(keys, early, late)}
    xp = np.arange(len(xs))
    ax.errorbar(xp, [a[0] for a in early], [a[1] for a in early], color="#56B4E9", marker="o", label="Videos 1-3", capsize=1.5)
    ax.errorbar(xp, [b[0] for b in late], [b[1] for b in late], color="#D55E00", marker="s", label="Videos 8-10", capsize=1.5)
    ax.set_xticks(xp, [f"{v:g}" for v in xs], fontsize=6)
    ax.set_title(title, fontsize=7)
axes[0].set_ylabel("Acceptance")
axes[0].legend(loc="lower right", fontsize=5.5)
save(fig, "sensitivity")
N["sensitivity"] = rows

# ---------------------------------------------------------------- robustness
ro = by_label(load("robustness"))
rows = defaultdict(dict)
fig, axes = plt.subplots(1, 3, figsize=(TWO, 1.8), sharey=True)
for ax, (g, title, xl) in zip(axes, (("noise", "(a) Feedback noise", "Share of wrong labels"), ("review", "(b) Review budget", "Share of decisions reviewed"),
                                     ("override", "(c) Willingness to correct", "Probability of correcting"))):
    for algo, c, m in (("Proposed", "#D55E00", "o"), ("Rule", "#888888", "s")):
        keys = sorted((k for k in ro if k.startswith(f"{algo}|{g}=")), key=lambda k: float(k.split("=")[1]))
        if not keys:
            continue
        xs = [float(k.split("=")[1]) for k in keys]
        vals = [ci([acc(r)[7:].mean() for r in ro[k]]) for k in keys]
        ax.errorbar(xs, [v[0] for v in vals], [v[1] for v in vals], color=c, marker=m, capsize=1.5, label=LABEL.get(algo, algo))
        rows[f"{algo}_{g}"] = {str(x): {"late": float(v[0]), "ci": float(v[1])} for x, v in zip(xs, vals)}
    ax.set(title=title, xlabel=xl, ylim=(.4, 1))
axes[0].set_ylabel("Acceptance, videos 8-10")
axes[0].legend(loc="lower left", fontsize=6)
save(fig, "robustness")
for n in (12, 30, 60):
    v = ci([acc(r)[7:].mean() for r in ro[f"Proposed|phrases={n}"]])
    rows["phrases"][str(n)] = {"late": float(v[0]), "ci": float(v[1]),
                               "decisions": float(np.mean([steps(r)[-1] / 10 for r in ro[f"Proposed|phrases={n}"]]))}
N["robustness"] = rows

# ---------------------------------------------------------------- E2..E10
s = e("e3")
N["e3"] = {"diag": s["mean_diagonal"], "off": s["mean_off_diagonal"]}
fig, ax = plt.subplots(figsize=(ONE, 2.3))
im = ax.imshow(np.array(s["matrix"]), vmin=0, vmax=1, cmap="viridis")
names = [p.capitalize() for p in s["personas"]]
ax.set_xticks(range(6), names, rotation=30, ha="right")
ax.set_yticks(range(6), names)
ax.grid(False)
for i, row in enumerate(s["matrix"]):
    for j, v in enumerate(row):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="black" if v > .6 else "white")
ax.set(xlabel="Creator the policy is applied to", ylabel="Creator the policy was trained on")
save(fig, "cross")

s = e("e4")
N["e4"] = {k: {kk: s[k][kk] for kk in ("at_switch", "before_switch_mean", "videos_to_recover")} | {"end": float(np.mean(s[k]["mean_curve"][-3:]))} for k in s}
fig, ax = plt.subplots(figsize=(ONE, 2.1))
for k, c, m in (("1.0", "#888888", "s"), ("0.995", "#D55E00", "o"), ("0.98", "#0072B2", "^")):
    y = s[k]["mean_curve"]
    ax.plot(range(1, len(y) + 1), y, color=c, marker=m, label=f"$\\gamma$ = {k}")
ax.axvline(5.5, color="black", lw=.6, ls="--")
ax.text(5.7, .96, "style changes", fontsize=6.5)
ax.set(xlabel="Video", ylabel="Acceptance")
ax.legend(loc="lower right")
save(fig, "drift")

s = e("e5")
N["e5"] = {a: {"err0": s["table"][a]["dur_err"]["mean"][0], "err": s["table"][a]["dur_err"]["mean"][-1], "err_ci": s["table"][a]["dur_err"]["ci95"][-1],
               "acc": s["table"][a]["acc"]["mean"][-1], "acc_ci": s["table"][a]["acc"]["ci95"][-1], "phrase": s["table"][a]["acc_phrase"]["mean"][-1],
               "by_target": s["table"][a]["by_target"]} for a in s["table"]}
fig, axes = plt.subplots(1, 2, figsize=(ONE, 1.95))
for a, c, m, lab in (("bandit", "#0072B2", "o", "Bandit"), ("bandit+plan", "#D55E00", "s", "Bandit + MDP planner")):
    t = s["table"][a]
    x = np.arange(len(t["acc"]["mean"]))
    band(axes[0], x, np.array(t["dur_err"]["mean"]), np.array(t["dur_err"]["ci95"]), lab, c, m)
    band(axes[1], x, np.array(t["acc"]["mean"]), np.array(t["acc"]["ci95"]), lab, c, m)
axes[0].set(yscale="log", xlabel="Videos edited", ylabel="Relative length error")
axes[1].set(xlabel="Videos edited", ylabel="Acceptance")
axes[1].legend(loc="lower right", fontsize=5.5)
fig.tight_layout(pad=.3)
save(fig, "target")

s = e("e8")
N["e8"] = s["table"]
N["e8_budgets"] = s["budgets"]
fig, ax = plt.subplots(figsize=(ONE, 2.1))
for k, c, m in (("uncertainty", "#D55E00", "o"), ("random", "#0072B2", "^"), ("timeline", "#888888", "s")):
    t = s["table"][k]["posted"]
    band(ax, 100 * np.array(s["budgets"]), np.array(t["mean"]), np.array(t["ci95"]), k.capitalize() + " order", c, m)
ax.set(xlabel="Decisions the creator reviews (%)", ylabel="Correct decisions in the export")
ax.legend(loc="lower right")
save(fig, "review")

s = e("e9")
N["e9"] = s
fig, axes = plt.subplots(1, 2, figsize=(TWO, 2.0), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
est = list(s["estimators"])
x = np.arange(len(est))
ax.bar(x - .2, [s["estimators"][k]["rmse"] for k in est], .4, color="#0072B2", label="RMSE")
ax.bar(x + .2, [abs(s["estimators"][k]["bias"]) for k in est], .4, color="#E69F00", label="Absolute bias")
ax.set_xticks(x, est)
ax.set(ylabel="Estimation error", title="(a) Accuracy of the estimators", ylim=(0, .2))
ax.legend(loc="upper left", ncol=2)
ax = axes[1]
names = list(s["targets"])
y = np.arange(len(names))
ax.barh(y + .2, [s["targets"][k]["truth"] for k in names], .38, color="#444444", label="True value")
ax.barh(y - .2, [s["targets"][k]["DR"] for k in names], .38, color="#D55E00", label="DR estimate from logs")
ax.set_yticks(y, ["Learned LinTS", "Rule", "Generic prior", "Other styles"])
ax.set(xlim=(.4, 1), xlabel="Acceptance", title="(b) Offline ranking of policies")
ax.invert_yaxis()
ax.legend(loc="lower right", framealpha=1, frameon=True, edgecolor="none")
fig.tight_layout(pad=.3)
save(fig, "ope")

s = e("e10")
N["e10"] = s["table"]
fig, axes = plt.subplots(1, 3, figsize=(TWO, 1.8), sharey=True)
for ax, p in zip(axes, s["table"]):
    for a, c, m, lab in (("rule", "#888888", "s", "Rule"), ("global", "#E69F00", "v", "Global prior"), ("lints", "#D55E00", "o", "LinTS")):
        t = s["table"][p][a]
        band(ax, np.arange(1, len(t["mean"]) + 1), np.array(t["mean"]), np.array(t["ci95"]), lab, c, m)
    ax.axhline(s["table"][p]["constant"], color="black", ls="--", lw=.7, label="Best fixed arm")
    ax.set(title=p.capitalize(), xlabel="Video")
axes[0].set_ylabel("Acceptance (keep/cut, pause, filler)")
axes[0].legend(loc="lower right", fontsize=5.5)
save(fig, "unseen")

s = e("e2")
fig, ax = plt.subplots(figsize=(ONE, 2.0))
for k, c, m, lab in (("none", "#888888", "s", "No prior"), ("generic", "#0072B2", "o", "Generic prior"), ("mixture", "#D55E00", "*", "Style mixture (proposed)")):
    y = s["cold_start"][k]["mean_curve"]
    ax.plot(range(1, len(y) + 1), y, color=c, marker=m, label=lab)
ax.set(xlabel="Video", ylabel="Acceptance")
ax.legend(loc="lower right")
save(fig, "coldstart")
N["e2"] = {k: {"early": v["early"], "curve": v["mean_curve"]} for k, v in s["cold_start"].items()}
N["e2_id"] = {k: s["identification"][k]["median_over_personas"] for k in s["identification"]}
N["e2_id_persona"] = s["identification"]["product"]["by_persona"]
for extra in ("runtime", "real_clips"):
    p = D / f"{extra}.json"
    if p.exists():
        N[extra] = json.loads(p.read_text(encoding="utf-8"))
(D / "numbers.json").write_text(json.dumps(N, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: N[k] for k in ("main", "ablation", "sensitivity", "robustness", "e2", "e2_id", "e3", "e4", "e5", "e8", "e9")}, indent=None, default=lambda x: round(float(x), 3))[:12000])
