"""Summarise paper/data/final/*.json into final_summary.json and print it.

    backend\\.venv\\Scripts\\python.exe paper\\scripts\\final_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

D = Path(__file__).resolve().parents[1] / "data" / "final"
HARD = ("phrase", "pause", "filler")


def load(name):
    p = D / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def acc(v, keys=None):
    keys = keys or v["n"].keys()
    return sum(v["correct"][k] for k in keys) / max(sum(v["n"][k] for k in keys), 1)


def ci(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def table(runs, keys=None, at=(1, 10)):
    by = {}
    for r in runs:
        by.setdefault(r["label"], []).append(r)
    out = {}
    for lab, rs in by.items():
        row = {}
        for k in at:
            if k <= len(rs[0]["videos"]):
                row[f"video_{k}"] = ci([acc(r["videos"][k - 1], keys) for r in rs])
                row[f"regret_{k}"] = ci([sum(sum(v["n"].values()) - sum(v["correct"].values()) for v in r["videos"][:k]) for r in rs])
        row["n"] = len(rs)
        out[lab] = row
    return out


def paired(runs, a, b, k, keys=None, regret=False):
    idx = {(r["label"], r["persona"], r["seed"]): r for r in runs}
    d = []
    for (lab, p, s), r in idx.items():
        if lab == a and (b, p, s) in idx:
            o = idx[(b, p, s)]
            if regret:
                d.append(sum(sum(v["n"].values()) - sum(v["correct"].values()) for v in r["videos"][:k]) - sum(sum(v["n"].values()) - sum(v["correct"].values()) for v in o["videos"][:k]))
            else:
                d.append(acc(r["videos"][k - 1], keys) - acc(o["videos"][k - 1], keys))
    m, h = ci(d)
    return {"diff": m, "ci": h, "p": float(stats.ttest_1samp(d, 0).pvalue), "n": len(d)}


out = {}
t = load("tune")
if t:
    out["tuned"] = t["best"]
long = load("long")
if long:
    out["long"] = table(long, at=(1, 3, 5, 10, 20, 30, 50))
    out["long_paired"] = {f"Proposed-{b}": {f"video_{k}": paired(long, "Proposed", b, k) for k in (10, 50)} | {f"regret_{k}": paired(long, "Proposed", b, k, regret=True) for k in (10, 50)}
                          for b in ("Neural", "Neural, pretrained", "NeuralLinear", "NeuralLinear, pretrained", "LinTS", "LinUCB", "Proposed, retuned")}
    out["retuned_paired"] = {f"retuned-{b}": {f"video_{k}": paired(long, "Proposed, retuned", b, k) for k in (1, 10, 50)} | {f"regret_{k}": paired(long, "Proposed, retuned", b, k, regret=True) for k in (10, 50)}
                             for b in ("Neural", "Neural, pretrained", "NeuralLinear", "NeuralLinear, pretrained", "Product")}
fresh = load("fresh")
if fresh:
    out["fresh"] = table(fresh, at=(1, 6, 10))
    out["fresh_paired"] = {f"video_{k}": paired(fresh, "Proposed, retuned", "Proposed", k) for k in (1, 6, 10)} | {"regret_10": paired(fresh, "Proposed, retuned", "Proposed", 10, regret=True)}
e10 = load("e10")
if e10:
    out["e10"] = {}
    for p in ("nonlinear", "noisy", "arbitrary"):
        rs = [r for r in e10["runs"] if r["persona"] == p]
        out["e10"][p] = table(rs, HARD, at=(1, 8, 30))
        out["e10"][p]["constant"] = float(np.mean([c["constant"] for c in e10["constant"] if c["persona"] == p]))
        out["e10"][p]["paired_30"] = {b: paired(rs, "LinTS", b, 30, HARD) for b in ("Neural", "NeuralLinear", "Neural, pretrained", "NeuralLinear, pretrained")}
dl = load("delayed")
if dl:
    out["delayed"] = {}
    for f in (1.0, .3, .1, 0.0):
        out["delayed"][str(f)] = {}
        for a in ("rule", "bandit", "rlsvi", "dqn"):
            rs = [r for r in dl if r["algo"] == a and r["feedback"] == f]
            out["delayed"][str(f)][a] = {"match_8": ci([r["curve"][8]["acc"] for r in rs]), "match_16": ci([r["curve"][16]["acc"] for r in rs]),
                                         "rating_16": ci([r["curve"][16]["rating"] for r in rs])}
        rl = {(r["persona"], r["seed"]): r for r in dl if r["algo"] == "rlsvi" and r["feedback"] == f}
        ba = {(r["persona"], r["seed"]): r for r in dl if r["algo"] == "bandit" and r["feedback"] == f}
        for key in ("acc", "rating"):
            d = [rl[k]["curve"][16][key] - ba[k]["curve"][16][key] for k in rl]
            m, h = ci(d)
            out["delayed"][str(f)][f"rlsvi_minus_bandit_{key}"] = {"diff": m, "ci": h, "p": float(stats.ttest_1samp(d, 0).pvalue)}
(D.parent / "final_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def show(o, ind=0):
    for k, v in o.items():
        if isinstance(v, dict):
            print(" " * ind + str(k))
            show(v, ind + 2)
        elif isinstance(v, (list, tuple)):
            print(" " * ind + f"{k}: {v[0]:.3f} +- {v[1]:.3f}")
        elif isinstance(v, float):
            print(" " * ind + f"{k}: {v:.3f}")
        else:
            print(" " * ind + f"{k}: {v}")


show(out)
