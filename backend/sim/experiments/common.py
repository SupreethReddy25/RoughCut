"""Shared machinery for the simulator experiments (PLAN 10): videos, priors, policy factories, parallel runs, CIs."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np

from app.policy.base import Policy
from app.policy.linear import EpsGreedyLinear, GlobalPolicy, LinTS, LinUCB
from app.policy.log_replay import Learner
from app.policy.prior import Prior, StyleMixture, fit_prior
from app.policy.rule import RulePolicy
from app.session import EditSession
from sim.personas import BASE_PERSONAS, Persona, make_persona
from sim.session import DECISION_TYPES, UserRun, run_user
from sim.synth import SynthVideo, make_video

ALGOS = ("rule", "global", "eps", "linucb", "lints")
ALGO_LABEL = {"rule": "Rule", "global": "Global", "eps": "ε-greedy", "linucb": "LinUCB", "lints": "LinTS (ours)"}
VIDEO_TYPES = ("vlog", "tutorial", "opinion")
PRIOR_VIDEOS = 8  # training videos per persona for prototype / generic priors
PRIOR_SEED_BASE = 700_000  # priors are fit on videos disjoint from every experiment seed


def videos_for(seed: int, n: int, offset: int = 0) -> list[SynthVideo]:
    """n synthetic videos for `seed` (types cycle so every persona sees every type)."""
    return [make_video(seed * 1000 + offset + k, VIDEO_TYPES[(seed + k) % 3], index=k + 1) for k in range(n)]


# --------------------------------------------------------------------------- priors
@cache
def labeled_for(persona_name: str) -> tuple:
    """Full-information labels (decision_type, context, ideal arm) from a noise-free persona on training videos."""
    persona = make_persona(persona_name)
    out = []
    for k in range(PRIOR_VIDEOS):
        v = make_video(PRIOR_SEED_BASE + k, VIDEO_TYPES[k % 3], index=k + 1)
        p = persona.for_video(k + 1, v.video_type)
        p.prepare(v)
        sess = EditSession(v.uf, v.word_feats, v.video_type)
        for item in sess.items():
            ideal = p.ideal_action(item)
            out.append((item.decision_type, item.ctx, ideal))
            sess.commit(item, ideal)
    return tuple(out)


PRIOR_CACHE = Path(__file__).resolve().parents[3] / "data" / "experiments" / "priors"
_CODE_FILES = ("sim/synth.py", "sim/personas.py", "app/policy/context.py", "app/policy/prior.py", "app/session.py",
               "app/pipeline/features.py")


@cache
def _code_key() -> str:
    """Fingerprint of everything a fitted prior depends on, so a stale disk cache can never be used."""
    root = Path(__file__).resolve().parents[2]
    h = hashlib.sha1()
    for f in _CODE_FILES:
        h.update((root / f).read_bytes())
    h.update(f"{PRIOR_VIDEOS}:{PRIOR_SEED_BASE}".encode())
    return h.hexdigest()[:12]


def _cached_prior(name: str, fit) -> Prior:
    """Fit once, not once per worker process: priors are deterministic functions of the (fixed) training videos."""
    path = PRIOR_CACHE / f"{name}_{_code_key()}.json"
    try:
        return Prior.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    prior = fit()
    try:
        PRIOR_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(prior.to_dict()), encoding="utf-8")
        tmp.replace(path)  # atomic: concurrent workers never read a half-written file
    except OSError:
        pass  # a read-only checkout just refits
    return prior


@cache
def prototype(persona_name: str) -> Prior:
    return _cached_prior(f"proto_{persona_name}", lambda: fit_prior(labeled_for(persona_name)))


@cache
def generic_prior(exclude: tuple[str, ...] = ()) -> Prior:
    """Pooled generic prior over every base persona not in `exclude` (leave-one-persona-out when evaluating)."""
    def fit() -> Prior:
        return fit_prior([x for n in BASE_PERSONAS if n not in exclude for x in labeled_for(n)])

    return _cached_prior("generic_" + ("-".join(exclude) or "all"), fit)


def style_mixture(exclude: tuple[str, ...] = ()) -> StyleMixture:
    names = [n for n in BASE_PERSONAS if n not in exclude]
    return StyleMixture({n: prototype(n) for n in names}, generic_prior(exclude))


# --------------------------------------------------------------------------- policies
def make_policy(algo: str, prior: Prior | None, seed: int, gamma: float = 0.995, **kw) -> Policy:
    if algo == "rule":
        return RulePolicy(seed)
    if algo == "global":
        return GlobalPolicy(prior or Prior(), seed)
    if algo == "eps":
        return EpsGreedyLinear(prior, gamma=gamma, seed=seed, **kw)
    if algo == "linucb":
        return LinUCB(prior, gamma=gamma, seed=seed, **kw)
    if algo == "lints":
        return LinTS(prior, gamma=gamma, seed=seed, **kw)
    raise ValueError(algo)


def make_learner(algo: str, prior: Prior | None, seed: int, gamma: float = 0.995, cfg=None, mixture=None,
                 **kw) -> Learner:
    pol = make_policy(algo, prior, seed, gamma, **kw)
    return Learner(pol, mixture, cfg) if cfg is not None else Learner(pol, mixture)


# --------------------------------------------------------------------------- running
def summarize(run: UserRun) -> list[dict]:
    rows = []
    for r in run.videos:
        rows.append({"index": r.index, "type": r.video_type, "minutes": r.minutes,
                     "n": {dt: r.n[dt] for dt in DECISION_TYPES}, "correct": {dt: r.correct[dt] for dt in DECISION_TYPES},
                     "corrections": r.corrections, "manual_edits": r.manual_edits, "accepts": r.accepts,
                     "implicits": r.implicits})
    return rows


def run_learning_task(task: dict) -> dict:
    """task: persona (name or spec), algo, seed, n_videos, prior ('generic'|'none'), gamma, kw. Top-level => picklable."""
    persona: Persona = make_persona(task["persona"], **task.get("persona_kw", {}))
    exclude = tuple(task.get("exclude", (task["persona"],) if task["persona"] in BASE_PERSONAS else ()))
    mode = task.get("prior", "generic")
    prior = {"generic": lambda: generic_prior(exclude), "none": lambda: None}[mode]() if mode != "mixture" else None
    mixture = style_mixture(exclude) if mode == "mixture" else None
    learner = make_learner(task["algo"], prior, task["seed"], task.get("gamma", 0.995), task.get("cfg"), mixture,
                           **task.get("policy_kw", {}))
    run = run_user(persona, learner, videos_for(task["seed"], task["n_videos"]), task["seed"])
    return {**{k: task[k] for k in ("persona", "algo", "seed")}, "tag": task.get("tag"), "videos": summarize(run)}


def run_parallel(fn: Callable, tasks: list, workers: int | None = None, progress: bool = True) -> list:
    workers = workers or min(os.cpu_count() or 1, 16)
    if workers <= 1 or len(tasks) == 1:
        return [fn(t) for t in tasks]
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(fn, tasks, chunksize=1), 1):
            out.append(r)
            if progress and i % max(1, len(tasks) // 10) == 0:
                print(f"  {i}/{len(tasks)} tasks", flush=True)
    return out


# --------------------------------------------------------------------------- statistics
def ci95(x: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """mean and 95% CI half-width (t distribution) over `axis`."""
    from scipy import stats

    x = np.asarray(x, dtype=float)
    n = np.sum(~np.isnan(x), axis=axis)
    mean = np.nanmean(x, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):  # a single seed has no spread: half-width becomes nan
        se = np.nanstd(x, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))
    t = stats.t.ppf(0.975, np.maximum(n - 1, 1))
    return mean, t * se


def acc_matrix(results: list[dict], dt: str | None = None) -> np.ndarray:
    """(n_results, n_videos) acceptance (overall or for one decision type); nan where a type has no decisions."""
    rows = []
    for r in results:
        row = []
        for v in r["videos"]:
            n = sum(v["n"].values()) if dt is None else v["n"][dt]
            c = sum(v["correct"].values()) if dt is None else v["correct"][dt]
            row.append(c / n if n else np.nan)
        rows.append(row)
    return np.array(rows)


def corrections_per_minute(results: list[dict]) -> np.ndarray:
    return np.array([[v["corrections"] / v["minutes"] for v in r["videos"]] for r in results])


def regret_curve(results: list[dict]) -> np.ndarray:
    """cumulative number of proposals that differed from the persona's ideal."""
    return np.array([np.cumsum([sum(v["n"].values()) - sum(v["correct"].values()) for v in r["videos"]])
                     for r in results])
