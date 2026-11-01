"""Paired comparisons quoted in the paper (same persona and seed): paper/data/paired.json.

    backend\\.venv\\Scripts\\python.exe paper\\scripts\\paired_stats.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

D = Path(__file__).resolve().parents[1] / "data"
main = json.loads((D / "main.json").read_text(encoding="utf-8"))
T = ("phrase", "pause", "filler", "zoom", "emphasis")
rows = defaultdict(dict)
for r in main:
    v = r["videos"]
    rows[r["label"]][(r["persona"], r["seed"])] = {
        "regret": sum(sum(x["n"][t] - x["ok"][t] for t in T) for x in v),
        "v10": sum(v[-1]["ok"].values()) / sum(v[-1]["n"].values()),
        "early": float(np.mean([sum(x["ok"].values()) / sum(x["n"].values()) for x in v[:3]])),
    }
out = {}
for other in ("LinTS", "LinUCB", "eps-greedy", "Rule", "Global"):
    out[other] = {}
    for m in ("regret", "v10", "early"):
        keys = sorted(rows["Proposed"])
        a = np.array([rows["Proposed"][k][m] for k in keys])
        b = np.array([rows[other][k][m] for k in keys])
        d = a - b
        out[other][m] = {"diff": float(d.mean()), "ci": float(stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))),
                         "p": float(stats.ttest_rel(a, b).pvalue), "p_wilcoxon": float(stats.wilcoxon(d).pvalue)}
ex_path = D / "exploration.json"
if ex_path.exists():
    ex = json.loads(ex_path.read_text(encoding="utf-8"))
    tab = defaultdict(dict)
    for c in sorted({r["cond"] for r in ex}):
        for v in sorted({r["variant"] for r in ex}):
            rs = [r for r in ex if r["cond"] == c and r["variant"] == v]
            tab[c][v] = {"n": len(rs), "train": float(np.mean([r["train"] for r in rs])), "held": float(np.mean([r["held"] for r in rs])),
                         "held_ci": float(stats.t.ppf(.975, len(rs) - 1) * np.std([r["held"] for r in rs], ddof=1) / np.sqrt(len(rs)))}
    out["exploration"] = tab
    base = "Thompson v=0.25"
    pair = {}
    for c in tab:
        key = {(r["persona"], r["seed"]): r["held"] for r in ex if r["cond"] == c and r["variant"] == base}
        for v in tab[c]:
            d = np.array([r["held"] - key[(r["persona"], r["seed"])] for r in ex if r["cond"] == c and r["variant"] == v])
            if v != base:
                pair[f"{c}|{v}"] = {"diff": float(d.mean()), "ci": float(stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))}
    out["exploration_paired"] = pair
(D / "paired.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, default=lambda x: round(float(x), 3))[:6000])
