"""Stronger tests of the deep baselines and of our own method: paper/data/final/<stage>.json.

    tune      every agent family tuned the same way, on seeds 201-203 only, by acceptance on videos 5-6: the neural bandit on
              a wider grid than deep_baselines.py (whose best epsilon was the lowest it tried), NeuralLinear, and our method
    long      50 videos on the six creators, seeds 1-10; the first 10 videos are the usual comparison
    fresh     our default against our retuned settings, 10 videos, seeds 301-320, which nothing was tuned on
    e10       the creators outside the model family (sim/experiments/e10.py), 30 videos, seeds 1-8, with the neural agents
    delayed   visual rhythm when only some beats get feedback, down to the whole-video rating alone, seeds 101-120

Each stage writes its file when it finishes, and a finished stage is skipped on a later run.

    set PYTHONPATH=backend;paper\\scripts
    backend\\.venv\\Scripts\\python.exe paper\\scripts\\final_checks.py [stage ...]
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import torch
from deep_baselines import BATCH, NeuralBandit, RhythmDQN, mlp
from run_experiments import build_learner
from torch import nn

from app.policy import context
from app.policy.base import Choice
from app.policy.log_replay import Learner
from app.rl.rhythm import ARMS, GAMMA, Episode, RhythmAgent, Step
from sim.experiments.common import generic_prior, make_learner, prototype, summarize, videos_for
from sim.experiments.e10 import PERSONAS as E10_PERSONAS
from sim.experiments.e10 import best_constant
from sim.experiments.e11 import N_TEST, NOISE, P_OVERRIDE
from sim.experiments.e11 import PERSONAS as RHYTHM_PERSONAS
from sim.experiments.e11 import _eval, _rating, _rule, _walk, ideal
from sim.personas import BASE_PERSONAS, make_persona
from sim.session import run_user

OUT = Path(__file__).resolve().parents[1] / "data" / "final"
WORKERS = 14
TUNE_SEEDS = (201, 202, 203)
TUNE_VIDEOS = 6


class TunableBandit(NeuralBandit):
    """NeuralBandit with the network width and the gradient steps per event as settings too."""

    name = "neural"

    def __init__(self, seed: int, lr: float = 1e-3, eps: float = 0.05, steps: int = 1, hidden: int = 64):
        super().__init__(seed, lr, eps)
        self.grad_steps = steps
        for dt in context.DIMS:
            k = 1 if dt == "emphasis" else len(context.ARMS[dt])
            self.nets[dt] = nn.Sequential(nn.Linear(context.dim(dt), hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, k))
            self.opts[dt] = torch.optim.Adam(self.nets[dt].parameters(), lr=lr)

    def _train(self, dt) -> None:
        buf = self.buf[dt]
        for _ in range(self.grad_steps):
            self._step(dt, [buf[int(i)] for i in self.rng.integers(len(buf), size=min(BATCH, len(buf)))])

    def update(self, dt, ctx, arm, reward, weight=1.0):
        if weight <= 0:
            return
        if dt == "emphasis":
            item = (self._x(dt, ctx, arm), 0, float(reward))
        else:
            item = (self._x(dt, ctx), context.ARMS[dt].index(arm), float(reward))
        buf = self.buf[dt]
        buf.append(item)
        del buf[:-5000]
        self._train(dt)
        self.n_updates += 1


class NeuralLinear(TunableBandit):
    """Riquelme et al. 2018: a Bayesian linear model on the network's last hidden layer, acting by Thompson sampling. The
    posterior is refit from the buffer every `refit` events, because the layer below it keeps moving. The head's current
    weights are the prior mean, which is what carries pretraining into the posterior."""

    name = "neural_linear"

    def __init__(self, seed: int, lr: float = 1e-3, v: float = 0.25, steps: int = 1, hidden: int = 64, lam: float = 1.0, refit: int = 32):
        super().__init__(seed, lr, 0.0, steps, hidden)
        self.v, self.lam, self.refit = v, lam, refit
        self.post: dict = {}
        self.since = {dt: 0 for dt in context.DIMS}

    def _embed(self, dt, x: torch.Tensor) -> torch.Tensor:
        return self.nets[dt][:-1](x)

    def _head(self, dt, k: int) -> np.ndarray:
        return self.nets[dt][-1].weight.detach()[k].numpy().astype(float)

    def _fit(self, dt) -> None:
        n_arms = self.nets[dt][-1].out_features
        buf = self.buf[dt]
        post = {}
        with torch.no_grad():
            z = self._embed(dt, torch.stack([b[0] for b in buf])).numpy().astype(float) if buf else None
        arm = np.array([b[1] for b in buf]) if buf else None
        y = np.array([b[2] for b in buf]) if buf else None
        for k in range(n_arms):
            prior = self._head(dt, k)
            A, b = self.lam * np.eye(len(prior)), self.lam * prior
            if buf:
                m = arm == k
                A = A + z[m].T @ z[m]
                b = b + z[m].T @ y[m]
            cov = np.linalg.inv(A)
            post[k] = (cov @ b, np.linalg.cholesky(self.v ** 2 * (cov + cov.T) / 2 + 1e-9 * np.eye(len(b))))
        self.post[dt] = post
        self.since[dt] = 0

    def _posterior(self, dt):
        if dt not in self.post:
            self._fit(dt)
        return self.post[dt]

    def _scores(self, dt, ctx, sample: bool) -> dict:
        post = self._posterior(dt)

        def theta(k):
            mean, chol = post[k]
            return mean + chol @ self.rng.normal(size=len(mean)) if sample else mean

        with torch.no_grad():
            if dt == "emphasis":
                t = theta(0)  # one draw for the whole decision: the arms share a model
                return {a: float(self._embed(dt, self._x(dt, ctx, a)).numpy() @ t) for a in context.arms_for(dt, ctx)}
            z = self._embed(dt, self._x(dt, ctx)).numpy()
            return {a: float(z @ theta(i)) for i, a in enumerate(context.ARMS[dt])}

    def mean_scores(self, dt, ctx):
        return self._scores(dt, ctx, False)

    def choose(self, dt, ctx, offsets=None):
        s = self._scores(dt, ctx, True)
        m = self._scores(dt, ctx, False)
        a = max(s, key=s.get)
        return Choice(a, 1.0, 1.0, a != max(m, key=m.get), {})

    def update(self, dt, ctx, arm, reward, weight=1.0):
        if weight <= 0:
            return
        super().update(dt, ctx, arm, reward, weight)
        self.since[dt] += 1
        if self.since[dt] >= self.refit:
            self._fit(dt)

    def pretrain(self, persona: str, steps: int = 3000) -> None:
        super().pretrain(persona, steps)
        self.post = {}


def neural_learner(t: dict) -> Learner:
    cls = NeuralLinear if t["algo"] == "neural_linear" else TunableBandit
    pol = cls(t["seed"], **t["kw"])
    if t.get("warm"):
        pol.pretrain(t["persona"])
    return Learner(pol)


def decision_task(t: dict) -> dict:
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    persona = E10_PERSONAS[t["persona"]]() if t["persona"] in E10_PERSONAS else make_persona(t["persona"])
    if t["algo"] in ("neural", "neural_linear"):
        learner = neural_learner(t)
    elif t["persona"] in E10_PERSONAS:  # as E10: the generic prior, gamma 0.995
        learner = make_learner(t["algo"], generic_prior(), t["seed"], 0.995, **t.get("kw", {}))
    else:
        learner = build_learner(t)
    run = run_user(persona, learner, videos_for(t["seed"], t["videos"]), t["seed"])
    return {"label": t["label"], "persona": t["persona"], "seed": t["seed"], "videos": summarize(run), "secs": time.perf_counter() - t0}


def constant_task(t: dict) -> dict:
    return {**t, "constant": best_constant(E10_PERSONAS[t["persona"]](), t["seed"], t["videos"])}


# --------------------------------------------------------------------------- visual rhythm with sparse feedback
class DelayedDQN(RhythmDQN):
    """RhythmDQN that still learns from the rating when nobody touched the last beat."""

    def add_episode(self, ep: Episode):
        if ep.steps and ep.steps[-1].reward is None and ep.rating is not None:
            ep.steps[-1].reward = 0.0
        if not any(s.reward is not None for s in ep.steps):
            return
        super().add_episode(ep)


def rhythm_task(t: dict) -> dict:
    torch.set_num_threads(1)
    name, seed, algo, rate = t["persona"], t["seed"], t["algo"], t["feedback"]
    tests = videos_for(seed, N_TEST, offset=500)
    if algo == "rule":
        return {**t, "curve": [_eval(name, tests, _rule)] * (t["videos"] + 1)}
    rng = np.random.default_rng(seed * 31 + 7)
    if algo == "dqn":
        agent = DelayedDQN(seed, True, **t["kw"])
    else:
        agent = RhythmAgent(seed=seed, gamma=0.0 if algo == "bandit" else GAMMA, n_draws=0)
        agent.begin_video()
    curve = [_eval(name, tests, agent.mean_action)]
    for video in videos_for(seed, t["videos"]):
        agent.begin_video()
        ep = Episode()
        beats = [0]

        def choose(item, ep=ep, beats=beats):
            a = agent.choose(item.ctx).action
            step = Step(dict(item.ctx), a)
            final = a
            if rng.random() < rate:
                want = ideal(name, item.ctx)
                if rng.random() < NOISE:
                    want = str(rng.choice([x for x in item.arms if x != want]))
                if want == a:
                    step.reward = 1.0
                elif rng.random() < P_OVERRIDE:
                    step.reward, step.final = -1.0, want
                    final = want
                else:
                    step.reward = 0.5
            ep.steps.append(step)
            beats[0] += final != "NONE"
            return final

        sess = _walk(video, choose)
        if algo != "bandit":  # as in E11, the bandit has no way to use the rating
            ep.rating = _rating(name, beats[0], sess._out_t)
        agent.add_episode(ep)
        agent.begin_video()
        curve.append(_eval(name, tests, agent.mean_action))
    return {**t, "curve": curve}


# --------------------------------------------------------------------------- stages
def run_map(fn, tasks: list[dict], stage: str) -> list[dict]:
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(fn, tasks, chunksize=1), 1):
            out.append(r)
            if i % max(1, len(tasks) // 20) == 0:
                print(f"{stage}: {i}/{len(tasks)} after {time.time() - t0:.0f}s", flush=True)
    print(f"{stage}: done, {len(out)} runs, {time.time() - t0:.0f}s", flush=True)
    return out


def acceptance_end(rows: list[dict]) -> float:
    return float(np.mean([sum(v["correct"].values()) / sum(v["n"].values()) for r in rows for v in r["videos"][-2:]]))


def read(stage: str) -> dict | list:
    return json.loads((OUT / f"{stage}.json").read_text(encoding="utf-8"))


def write(stage: str, data) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{stage}.json").write_text(json.dumps(data), encoding="utf-8")


def stage_tune() -> None:
    grids = {
        "neural": [{"lr": lr, "eps": e, "steps": s, "hidden": h}
                   for lr, e, s, h in product((3e-4, 1e-3, 3e-3), (0.0, 0.02, 0.05), (1, 4), (64, 128))],
        "neural_linear": [{"lr": lr, "v": v, "steps": s, "hidden": 64} for lr, v, s in product((3e-4, 1e-3, 3e-3), (0.05, 0.1, 0.25, 0.5), (1, 4))],
        "lints": [{"lam": lam, "v": v, "eps_floor": e} for lam, v, e in product((0.5, 2.0, 10.0, 30.0), (0.05, 0.1, 0.15, 0.25, 0.5), (0.0, 0.005, 0.02))],
    }
    tasks = []
    for algo, grid in grids.items():
        for kw in grid:
            tasks += [{"label": f"{algo}|{json.dumps(kw)}", "algo": algo, "kw": kw, "persona": p, "seed": s, "videos": TUNE_VIDEOS,
                       **({"prior": "mixture"} if algo == "lints" else {})} for p in BASE_PERSONAS for s in TUNE_SEEDS]
    res = run_map(decision_task, tasks, "tune")
    table: dict = {}
    for r in res:
        table.setdefault(r["label"], []).append(r)
    scores = {k: acceptance_end(v) for k, v in table.items()}
    best = {}
    for algo in grids:
        k = max((k for k in scores if k.startswith(algo + "|")), key=scores.get)
        best[algo] = json.loads(k.split("|", 1)[1])
        print("best", algo, best[algo], round(scores[k], 4), flush=True)
    write("tune", {"scores": scores, "best": best})


def stage_long() -> None:
    best = read("tune")["best"]
    agents = {"Rule": {"algo": "rule", "prior": "generic"}, "LinUCB": {"algo": "linucb", "prior": "generic"},
              "LinTS": {"algo": "lints", "prior": "generic"}, "Proposed": {"algo": "lints", "prior": "mixture"},
              "Proposed, retuned": {"algo": "lints", "prior": "mixture", "kw": best["lints"]}, "Product": {"algo": "product"},
              "Neural": {"algo": "neural", "kw": best["neural"]}, "Neural, pretrained": {"algo": "neural", "kw": best["neural"], "warm": True},
              "NeuralLinear": {"algo": "neural_linear", "kw": best["neural_linear"]},
              "NeuralLinear, pretrained": {"algo": "neural_linear", "kw": best["neural_linear"], "warm": True}}
    tasks = [{"label": lab, **kw, "persona": p, "seed": s, "videos": 50} for lab, kw in agents.items() for p in BASE_PERSONAS for s in range(1, 11)]
    write("long", run_map(decision_task, tasks, "long"))


def stage_fresh() -> None:
    best = read("tune")["best"]
    tasks = [{"label": lab, "algo": "lints", "prior": "mixture", "kw": kw, "persona": p, "seed": s, "videos": 10}
             for lab, kw in (("Proposed", {}), ("Proposed, retuned", best["lints"])) for p in BASE_PERSONAS for s in range(301, 321)]
    write("fresh", run_map(decision_task, tasks, "fresh"))


def stage_e10() -> None:
    best = read("tune")["best"]
    agents = {"Rule": {"algo": "rule"}, "LinTS": {"algo": "lints"}, "LinTS, retuned": {"algo": "lints", "kw": best["lints"]},
              "Neural": {"algo": "neural", "kw": best["neural"]}, "Neural, pretrained": {"algo": "neural", "kw": best["neural"], "warm": True},
              "NeuralLinear": {"algo": "neural_linear", "kw": best["neural_linear"]},
              "NeuralLinear, pretrained": {"algo": "neural_linear", "kw": best["neural_linear"], "warm": True}}
    tasks = [{"label": lab, **kw, "persona": p, "seed": s, "videos": 30} for lab, kw in agents.items() for p in E10_PERSONAS for s in range(1, 9)]
    runs = run_map(decision_task, tasks, "e10")
    consts = run_map(constant_task, [{"persona": p, "seed": s, "videos": 30} for p in E10_PERSONAS for s in range(1, 9)], "e10 constant")
    write("e10", {"runs": runs, "constant": consts})


def stage_delayed() -> None:
    dqn = json.loads((OUT.parent / "deep.json").read_text(encoding="utf-8"))["rhythm_tuning"]["best"]["DQN|True"]
    tasks = [{"algo": a, "persona": p, "seed": 100 + s, "videos": 16, "feedback": f, "kw": dqn if a == "dqn" else {}}
             for a in ("rule", "bandit", "rlsvi", "dqn") for f in (1.0, 0.3, 0.1, 0.0) for p in RHYTHM_PERSONAS for s in range(1, 21)]
    write("delayed", run_map(rhythm_task, tasks, "delayed"))


STAGES = {"tune": stage_tune, "long": stage_long, "fresh": stage_fresh, "e10": stage_e10, "delayed": stage_delayed}

if __name__ == "__main__":
    for p in BASE_PERSONAS:  # fit the shared priors once, so the workers read them from the disk cache
        generic_prior((p,))
        prototype(p)
    generic_prior()
    for name in sys.argv[1:] or STAGES:
        if (OUT / f"{name}.json").exists():
            print(name, "already done", flush=True)
            continue
        STAGES[name]()
