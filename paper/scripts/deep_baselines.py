"""Deep RL baselines on the same simulated creators: paper/data/deep.json.

Keep/cut, pause, filler, zoom and highlight decisions: an online neural value model with epsilon-greedy and replay, the
DQN recipe with gamma = 0 because these decisions have no state transition. Learning rate and epsilon are chosen on seeds
201-203, and results use seeds 1-10, the seeds of the other baselines. Two starts: blank, and pretrained on the other
creators' labels, which is the same knowledge the generic prior has.

Visual rhythm (the E11 setting, seeds 101-120): DQN with a target network, gamma = 0.9, and PPO with clipping and GAE, each
blank and warm-started from the rule that gives RLSVI its prior.

    set PYTHONPATH=backend
    backend\\.venv\\Scripts\\python.exe paper\\scripts\\deep_baselines.py
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from app.policy import context
from app.policy.base import Choice, Policy
from app.policy.log_replay import Learner
from app.rl.rhythm import ARMS, D, Episode, Step, default_prior, phi
from sim.experiments.common import labeled_for, videos_for
from sim.experiments.e11 import N_TEST, NOISE, P_OVERRIDE, PERSONAS, _eval, _rating, _walk, ideal
from sim.personas import BASE_PERSONAS, make_persona
from sim.session import DECISION_TYPES, run_user

OUT = Path(__file__).resolve().parents[1] / "data" / "deep.json"
HIDDEN, BATCH, BUFFER = 64, 32, 5000
WORKERS = 6


def mlp(d: int, k: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, k))


class NeuralBandit(Policy):
    """One network per decision type, trained online from a replay buffer on the reward of the arm that was tried."""

    name = "neural"

    def __init__(self, seed: int, lr: float = 1e-3, eps: float = 0.1):
        super().__init__(seed)
        torch.manual_seed(seed)
        self.eps, self.nets, self.opts, self.buf = eps, {}, {}, {}
        for dt in context.DIMS:
            k = 1 if dt == "emphasis" else len(context.ARMS[dt])
            self.nets[dt] = mlp(context.dim(dt), k)
            self.opts[dt] = torch.optim.Adam(self.nets[dt].parameters(), lr=lr)
            self.buf[dt] = []

    def _x(self, dt, ctx, arm=None):
        v = context.vectorize_arm(ctx, arm) if dt == "emphasis" else context.vectorize(dt, ctx)
        return torch.as_tensor(v, dtype=torch.float32)

    def mean_scores(self, dt, ctx):
        with torch.no_grad():
            if dt == "emphasis":
                arms = context.arms_for(dt, ctx)
                return {a: float(self.nets[dt](self._x(dt, ctx, a))[0]) for a in arms}
            q = self.nets[dt](self._x(dt, ctx))
            return dict(zip(context.ARMS[dt], q.tolist(), strict=True))

    def choose(self, dt, ctx, offsets=None):
        s = self.mean_scores(dt, ctx)
        arms = list(s)
        best = max(arms, key=lambda a: s[a])
        a = arms[int(self.rng.integers(len(arms)))] if self.rng.random() < self.eps else best
        p = {x: self.eps / len(arms) + (1 - self.eps) * (x == best) for x in arms}
        return Choice(a, p[a], p[a], a != best, p)

    def _step(self, dt, batch):
        x = torch.stack([b[0] for b in batch])
        y = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        if dt == "emphasis":
            pred = self.nets[dt](x)[:, 0]
        else:
            idx = torch.tensor([b[1] for b in batch])
            pred = self.nets[dt](x).gather(1, idx[:, None])[:, 0]
        loss = ((pred - y) ** 2).mean()
        self.opts[dt].zero_grad()
        loss.backward()
        self.opts[dt].step()

    def update(self, dt, ctx, arm, reward, weight=1.0):
        if weight <= 0:
            return
        if dt == "emphasis":
            item = (self._x(dt, ctx, arm), 0, float(reward))
        else:
            item = (self._x(dt, ctx), context.ARMS[dt].index(arm), float(reward))
        buf = self.buf[dt]
        buf.append(item)
        del buf[:-BUFFER]
        self._step(dt, [buf[int(i)] for i in self.rng.integers(len(buf), size=min(BATCH, len(buf)))])
        self.n_updates += 1

    def pretrain(self, persona: str, steps: int = 3000) -> None:
        """Fit to the labels of the other creators (+1 for the arm they wanted, -1 otherwise)."""
        rows: dict = {}
        for n in BASE_PERSONAS:
            if n == persona:
                continue
            for dt, ctx, want in labeled_for(n):
                for arm in context.arms_for(dt, ctx):
                    r = 1.0 if arm == want else -1.0
                    x = self._x(dt, ctx, arm) if dt == "emphasis" else self._x(dt, ctx)
                    rows.setdefault(dt, []).append((x, 0 if dt == "emphasis" else context.ARMS[dt].index(arm), r))
        for dt, data in rows.items():
            for _ in range(steps):
                self._step(dt, [data[int(i)] for i in self.rng.integers(len(data), size=BATCH)])


def decision_task(t: dict) -> dict:
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    pol = NeuralBandit(t["seed"], t["lr"], t["eps"])
    if t.get("warm"):
        pol.pretrain(t["persona"])
    run = run_user(make_persona(t["persona"]), Learner(pol), videos_for(t["seed"], t["videos"]), t["seed"])
    rows = [{"n": {d: r.n[d] for d in DECISION_TYPES}, "ok": {d: r.correct[d] for d in DECISION_TYPES}, "corr": r.corrections,
             "acc": r.accepts, "imp": r.implicits, "min": r.minutes} for r in run.videos]
    return {"label": t["label"], "persona": t["persona"], "seed": t["seed"], "videos": rows, "secs": time.perf_counter() - t0}


# --------------------------------------------------------------------------- visual rhythm
def _warm_targets(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """States and the prior's values for them: what RLSVI starts from."""
    rng = np.random.default_rng(seed)
    theta = np.stack(default_prior())
    xs = []
    for _ in range(800):
        xs.append(phi({"hist_since_visual": float(rng.uniform(0, 30)), "hist_last_visual": int(rng.integers(0, 4)),
                       "hist_streak": float(rng.integers(0, 4)), "loudness_z": float(rng.normal()), "duration": float(rng.uniform(.5, 8)),
                       "topic_centrality": float(rng.uniform(0, 1)), "has_emphasis_punct": float(rng.random() < .15),
                       "is_first": float(rng.random() < .05), "position": float(rng.uniform(0, 1)),
                       "hist_beats_per_min": float(rng.uniform(0, 20))}))
    x = np.array(xs, dtype=np.float32)
    return torch.as_tensor(x), torch.as_tensor(x @ theta.T, dtype=torch.float32)


class RhythmDQN:
    def __init__(self, seed: int, warm: bool, gamma: float = 0.9, lr: float = 1e-3, eps: float = 0.1):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.q, self.target = mlp(D, len(ARMS)), mlp(D, len(ARMS))
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.eps, self.buf, self.n = gamma, eps, [], 0
        if warm:
            x, y = _warm_targets(seed)
            for _ in range(800):
                i = torch.as_tensor(self.rng.integers(len(x), size=BATCH))
                loss = ((self.q(x[i]) - y[i]) ** 2).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        self.target.load_state_dict(self.q.state_dict())

    def begin_video(self):
        pass

    def _q(self, ctx):
        with torch.no_grad():
            return self.q(torch.as_tensor(phi(ctx), dtype=torch.float32))

    def mean_action(self, ctx):
        return ARMS[int(self._q(ctx).argmax())]

    def choose(self, ctx):
        best = self.mean_action(ctx)
        a = ARMS[int(self.rng.integers(len(ARMS)))] if self.rng.random() < self.eps else best
        return Choice(a, 1.0, 1.0, a != best)

    def add_episode(self, ep: Episode):
        xs = [phi(s.ctx) for s in ep.steps]
        for i, s in enumerate(ep.steps):
            nxt = xs[i + 1] if i + 1 < len(xs) else None
            bonus = 2.0 * ep.rating if ep.rating is not None and nxt is None else 0.0
            if s.reward is not None:
                self.buf.append((xs[i], ARMS.index(s.action), s.reward + bonus, nxt))
            if s.final and s.final != s.action:
                self.buf.append((xs[i], ARMS.index(s.final), 1.0 + bonus, nxt))
        del self.buf[:-BUFFER]
        for _ in range(len(ep.steps) * 4):
            b = [self.buf[int(i)] for i in self.rng.integers(len(self.buf), size=min(BATCH, len(self.buf)))]
            x = torch.as_tensor(np.array([r[0] for r in b]), dtype=torch.float32)
            a = torch.tensor([r[1] for r in b])
            r = torch.tensor([r[2] for r in b], dtype=torch.float32)
            nx = torch.as_tensor(np.array([r[3] if r[3] is not None else np.zeros(D) for r in b]), dtype=torch.float32)
            live = torch.tensor([r[3] is not None for r in b], dtype=torch.float32)
            with torch.no_grad():
                y = r + self.gamma * live * self.target(nx).max(1).values
            loss = ((self.q(x).gather(1, a[:, None])[:, 0] - y) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.n += 1
            if self.n % 50 == 0:
                self.target.load_state_dict(self.q.state_dict())


class RhythmPPO:
    def __init__(self, seed: int, warm: bool, gamma: float = 0.9, lam: float = 0.95, lr: float = 3e-4, clip: float = 0.2, epochs: int = 4):
        torch.manual_seed(seed)
        self.pi, self.v = mlp(D, len(ARMS)), mlp(D, 1)
        self.opt = torch.optim.Adam([*self.pi.parameters(), *self.v.parameters()], lr=lr)
        self.gamma, self.lam, self.clip, self.logp, self.epochs = gamma, lam, clip, [], epochs
        if warm:  # behaviour cloning of the prior's best arm
            x, y = _warm_targets(seed)
            want = y.argmax(1)
            opt = torch.optim.Adam(self.pi.parameters(), lr=1e-3)
            for _ in range(800):
                loss = nn.functional.cross_entropy(self.pi(x), want)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def begin_video(self):
        pass

    def mean_action(self, ctx):
        with torch.no_grad():
            return ARMS[int(self.pi(torch.as_tensor(phi(ctx), dtype=torch.float32)).argmax())]

    def choose(self, ctx):
        with torch.no_grad():
            d = Categorical(logits=self.pi(torch.as_tensor(phi(ctx), dtype=torch.float32)))
            a = d.sample()
        self.logp.append(float(d.log_prob(a)))
        return Choice(ARMS[int(a)], float(d.log_prob(a).exp()), float(d.log_prob(a).exp()), False)

    def add_episode(self, ep: Episode):
        x = torch.as_tensor(np.array([phi(s.ctx) for s in ep.steps]), dtype=torch.float32)
        a = torch.tensor([ARMS.index(s.action) for s in ep.steps])
        r = np.array([s.reward if s.reward is not None else 0.0 for s in ep.steps], dtype=np.float32)
        if ep.rating is not None:
            r[-1] += 2.0 * ep.rating
        old = torch.tensor(self.logp[:len(ep.steps)], dtype=torch.float32)
        self.logp = []
        with torch.no_grad():
            v = self.v(x)[:, 0].numpy()
        adv, g = np.zeros(len(r), dtype=np.float32), 0.0
        for t in range(len(r) - 1, -1, -1):
            nv = v[t + 1] if t + 1 < len(r) else 0.0
            g = r[t] + self.gamma * nv - v[t] + self.gamma * self.lam * g
            adv[t] = g
        ret = torch.as_tensor(adv + v)
        adv = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-6))
        for _ in range(self.epochs):
            d = Categorical(logits=self.pi(x))
            ratio = (d.log_prob(a) - old).exp()
            loss = (-torch.min(ratio * adv, ratio.clamp(1 - self.clip, 1 + self.clip) * adv).mean()
                    + 0.5 * ((self.v(x)[:, 0] - ret) ** 2).mean() - 0.01 * d.entropy().mean())
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()


def rhythm_task(t: dict) -> dict:
    torch.set_num_threads(1)
    name, seed = t["persona"], t["seed"]
    tests = videos_for(seed, N_TEST, offset=500)
    rng = np.random.default_rng(seed * 31 + 7)
    agent = (RhythmDQN if t["algo"] == "DQN" else RhythmPPO)(seed, t["warm"], **t.get("kw", {}))
    t0 = time.perf_counter()
    curve = [_eval(name, tests, agent.mean_action)]
    for video in videos_for(seed, t["videos"]):
        ep = Episode()
        beats = [0]

        def choose(item, ep=ep, beats=beats):
            a = agent.choose(item.ctx).action
            want = ideal(name, item.ctx)
            if rng.random() < NOISE:
                want = str(rng.choice([x for x in item.arms if x != want]))
            step = Step(dict(item.ctx), a)
            final = a
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
        ep.rating = _rating(name, beats[0], sess._out_t)
        agent.add_episode(ep)
        curve.append(_eval(name, tests, agent.mean_action))
    return {**t, "curve": curve, "secs": time.perf_counter() - t0}


def score(rows: list[dict]) -> float:
    return float(np.mean([sum(v["ok"].values()) / sum(v["n"].values()) for r in rows for v in r["videos"][-2:]]))


if __name__ == "__main__":
    t0 = time.time()
    for p in BASE_PERSONAS:
        labeled_for(p)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        grid = [(lr, eps) for lr in (1e-3, 3e-3) for eps in (0.05, 0.1)]
        tune = [{"label": f"{lr}|{eps}", "persona": p, "seed": s, "videos": 6, "lr": lr, "eps": eps}
                for lr, eps in grid for p in BASE_PERSONAS for s in (201, 202, 203)]
        res = list(ex.map(decision_task, tune, chunksize=2))
        scores = {f"{lr}|{eps}": score([r for r in res if r["label"] == f"{lr}|{eps}"]) for lr, eps in grid}
        lr, eps = max(grid, key=lambda g: scores[f"{g[0]}|{g[1]}"])
        print("tuning", scores, "chosen", lr, eps, round(time.time() - t0), flush=True)
        main = [{"label": lab, "persona": p, "seed": s, "videos": 10, "lr": lr, "eps": eps, "warm": w}
                for lab, w in (("Neural", False), ("Neural, pretrained", True)) for p in BASE_PERSONAS for s in range(1, 11)]
        decisions = list(ex.map(decision_task, main, chunksize=2))
        print("decisions done", round(time.time() - t0), flush=True)
        rh = [{"algo": a, "warm": w, "persona": p, "seed": 100 + s, "videos": 8}
              for a in ("DQN", "PPO") for w in (False, True) for p in PERSONAS for s in range(1, 21)]
        rhythm = list(ex.map(rhythm_task, rh, chunksize=2))
    OUT.write_text(json.dumps({"tuning": scores, "chosen": {"lr": lr, "eps": eps}, "decisions": decisions, "rhythm": rhythm}), encoding="utf-8")
    print("done", round(time.time() - t0), flush=True)
