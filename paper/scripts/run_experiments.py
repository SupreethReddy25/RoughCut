"""Experiments behind the paper's tables and figures. Reuses the simulator in backend/sim; changes nothing in it.

    set PYTHONPATH=backend
    backend\\.venv\\Scripts\\python.exe paper\\scripts\\run_experiments.py [main ablation sensitivity robustness]

Each group writes paper/data/<group>.json as soon as it finishes, so an interrupted run keeps what it has.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from app.appctx import (
    PRODUCT_EPS,
    PRODUCT_EPS_BY_TYPE,
    PRODUCT_LAMBDA,
    PRODUCT_LAMBDA_BY_TYPE,
    PRODUCT_V,
    STYLE_PRIOR_SIGMA,
    STYLE_SIGMA,
)
from app.policy.linear import LinTS
from app.policy.log_replay import Learner
from app.policy.reward import RewardConfig
from app.policy.style import StyleMixture
from sim.experiments.common import (
    generic_prior,
    make_learner,
    prototype,
    style_mixture,
    videos_for,
)
from sim.personas import BASE_PERSONAS, make_persona
from sim.session import DECISION_TYPES, run_user
from sim.synth import make_video

OUT = Path(__file__).resolve().parents[1] / "data"
WORKERS = 6
GROUPS = {"semantic": ("semantic_redundancy", "topic_centrality"),
          "retake": ("retake_score", "is_earlier_take", "is_incomplete"),
          "history": ("hist_output_ratio", "hist_since_last_cut", "hist_since_last_zoom", "hist_prev_zoomed")}


class StrippedLinTS(LinTS):
    """LinTS that never sees one group of context features (feature ablation)."""

    def __init__(self, *a, hide: tuple[str, ...] = (), **kw):
        super().__init__(*a, **kw)
        self.hide = hide

    def _strip(self, ctx):
        return {k: (0.0 if k in self.hide else v) for k, v in ctx.items()}

    def choose(self, decision_type, ctx, offsets=None):
        return super().choose(decision_type, self._strip(ctx), offsets)

    def update(self, decision_type, ctx, arm, reward, weight=1.0):
        super().update(decision_type, self._strip(ctx), arm, reward, weight)


def build_learner(t: dict) -> Learner:
    name, seed = t["persona"], t["seed"]
    excl = (name,)
    cfg = RewardConfig(**t.get("cfg", {}))
    prior_mode = t.get("prior", "mixture")
    kw = dict(t.get("kw", {}))
    gamma = t.get("gamma", 0.995)
    if t["algo"] == "product":
        names = [n for n in BASE_PERSONAS if n != name]
        mix = StyleMixture({n: prototype(n) for n in names}, generic_prior(excl), sigma=STYLE_SIGMA,
                           prior_sigma=STYLE_PRIOR_SIGMA)
        return Learner(LinTS(mix.mean_prior(), lam=PRODUCT_LAMBDA, v=PRODUCT_V, eps_floor=PRODUCT_EPS, seed=seed, gamma=gamma,
                             lam_by_type=PRODUCT_LAMBDA_BY_TYPE, eps_by_type=PRODUCT_EPS_BY_TYPE), mix, cfg)
    if t["algo"] == "stripped":
        mix = style_mixture(excl)
        return Learner(StrippedLinTS(mix.mean_prior(), seed=seed, gamma=gamma, hide=tuple(t["hide"])), mix, cfg)
    if prior_mode == "mixture":
        return make_learner(t["algo"], None, seed, gamma, cfg, style_mixture(excl), **kw)
    prior = generic_prior(excl) if prior_mode == "generic" else None
    return make_learner(t["algo"], prior, seed, gamma, cfg, None, **kw)


def task(t: dict) -> dict:
    t0 = time.perf_counter()
    persona = make_persona(t["persona"], **t.get("persona_kw", {}))
    learner = build_learner(t)
    if "n_phrases" in t:
        vids = [make_video(t["seed"] * 1000 + k, ("vlog", "tutorial", "opinion")[(t["seed"] + k) % 3], index=k + 1,
                           n_phrases=t["n_phrases"]) for k in range(t["videos"])]
    else:
        vids = videos_for(t["seed"], t["videos"])
    run = run_user(persona, learner, vids, t["seed"])
    rows = [{"n": {d: r.n[d] for d in DECISION_TYPES}, "ok": {d: r.correct[d] for d in DECISION_TYPES},
             "corr": r.corrections, "acc": r.accepts, "imp": r.implicits, "min": r.minutes} for r in run.videos]
    return {"label": t["label"], "persona": t["persona"], "seed": t["seed"], "videos": rows,
            "secs": time.perf_counter() - t0}


def grid(label: str, seeds: int, videos: int = 10, **kw) -> list[dict]:
    return [{"label": label, "persona": p, "seed": s, "videos": videos, **kw}
            for p in BASE_PERSONAS for s in range(1, seeds + 1)]


def main_tasks() -> list[dict]:
    out = []
    for label, kw in {"Rule": {"algo": "rule", "prior": "generic"}, "Global": {"algo": "global", "prior": "generic"},
                      "eps-greedy": {"algo": "eps", "prior": "generic"}, "LinUCB": {"algo": "linucb", "prior": "generic"},
                      "LinTS": {"algo": "lints", "prior": "generic"}, "Proposed": {"algo": "lints", "prior": "mixture"},
                      "Product": {"algo": "product"}}.items():
        out += grid(label, 10, **kw)
    return out


def ablation_tasks() -> list[dict]:
    base = {"algo": "lints"}
    out = grid("Full", 10, **base)
    out += grid("No style prior", 10, **base, prior="generic")
    out += grid("No prior at all", 10, **base, prior="none")
    out += grid("No exploration", 10, **base, kw={"v": 0.0, "eps_floor": 0.0})
    out += grid("No discounting", 10, **base, gamma=1.0)
    out += grid("No implicit feedback", 10, **base, cfg={"implicit_weight": 0.0})
    out += grid("No manual-edit mapping", 10, **base, cfg={"learn_manual": False})
    for g, keys in GROUPS.items():
        out += grid(f"No {g} features", 10, algo="stripped", hide=list(keys))
    return out


def sensitivity_tasks() -> list[dict]:
    out = []
    for lam in (0.5, 2.0, 10.0, 30.0):
        out += grid(f"lambda={lam}", 6, algo="lints", kw={"lam": lam})
    for v in (0.05, 0.25, 0.5, 1.0, 2.0):
        out += grid(f"v={v}", 6, algo="lints", kw={"v": v})
    for g in (1.0, 0.995, 0.98, 0.95):
        out += grid(f"gamma={g}", 6, algo="lints", gamma=g)
    for e in (0.0, 0.02, 0.1, 0.2):
        out += grid(f"eps={e}", 6, algo="lints", kw={"eps_floor": e})
    for r in (0.0, 0.25, 0.5, 1.0):
        out += grid(f"implicit={r}", 6, algo="lints", cfg={"implicit": r})
    return out


def robustness_tasks() -> list[dict]:
    out = []
    for algo, kw in (("Proposed", {"algo": "lints"}), ("Rule", {"algo": "rule", "prior": "generic"})):
        for n in (0.0, 0.05, 0.1, 0.2, 0.3):
            out += grid(f"{algo}|noise={n}", 6, persona_kw={"noise": n}, **kw)
    for p in (0.2, 0.4, 0.6, 1.0):
        out += grid(f"Proposed|override={p}", 6, algo="lints", persona_kw={"p_override": p})
    for b in (0.1, 0.3, 0.6, 1.0):
        out += grid(f"Proposed|review={b}", 6, algo="lints", persona_kw={"review_budget": b})
    for n in (12, 30, 60):
        out += grid(f"Proposed|phrases={n}", 6, algo="lints", n_phrases=n)
    return out


SETS = {"main": main_tasks, "ablation": ablation_tasks, "sensitivity": sensitivity_tasks, "robustness": robustness_tasks}


def run_set(name: str) -> None:
    tasks = SETS[name]()
    t0 = time.time()
    # build the shared priors once in this process so the workers read them from the disk cache
    for p in BASE_PERSONAS:
        generic_prior((p,))
        prototype(p)
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(task, tasks, chunksize=2), 1):
            results.append(r)
            if i % max(1, len(tasks) // 10) == 0:
                print(f"{name}: {i}/{len(tasks)} after {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(results), encoding="utf-8")
    print(f"{name}: done, {len(results)} runs, {time.time() - t0:.0f}s, mean run {np.mean([r['secs'] for r in results]):.1f}s",
          flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:] or list(SETS):
        run_set(s)
