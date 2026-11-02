"""Off-policy evaluation (PLAN 7.4, U7): how good would policy pi_e have been, judged only from the feedback
collected while a DIFFERENT policy (the logging policy pi_b, here LinTS) was making the suggestions?

Every suggestion logs its context x, the proposed action a ~ pi_b(.|x), the propensity p = pi_b(a|x), and the
person's explicit verdict r (accept = 1, reject / override to something else = 0). The value of a target policy
is V(pi_e) = E_x E_{a ~ pi_e(.|x)} [r(x, a)], i.e. its expected acceptance rate. Estimators:

    DM     direct method: (1/n) sum_i sum_a pi_e(a|x_i) q(x_i, a)           (biased if the reward model q is wrong)
    IPS    inverse propensity scoring: (1/n) sum_i w_i r_i, w_i = pi_e(a_i|x_i) / p_i      (unbiased, high variance)
    SNIPS  self-normalised IPS: sum_i w_i r_i / sum_i w_i                      (small bias, much lower variance)
    DR     doubly robust: DM + (1/n) sum_i w_i (r_i - q(x_i, a_i))             (unbiased if p is right OR q is)
    SNDR   self-normalised DR: DM + sum_i w_i (r_i - q_i) / sum_i w_i          (DR's correction, SNIPS's variance)

(Horvitz & Thompson 1952; Swaminathan & Joachims 2015; Dudik, Langford & Li 2011.) q is a ridge reward model per
(decision type, arm) on the same context vectors the bandit uses, CROSS-FITTED (fit on the other folds) so DR's
correction term stays unbiased. IPS stays unbiased under adaptive logging (pi_b learns between rounds) because
each round's own propensity is used. LinTS supports every arm thanks to its epsilon floor (p >= eps/K), which
bounds the weights. Confidence intervals: the spread of a cluster bootstrap over videos, turned into
estimate +- t * sd with Student's t on (videos - 1) degrees of freedom and the small-sample factor sqrt(G / (G - 1)).
A plain percentile bootstrap is too narrow with a handful of videos, which is the normal case here.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from app.policy import context

TargetPolicy = Callable[[str, dict[str, float], list[str]], dict[str, float]]  # (dt, ctx, arms) -> pi_e(.|x)
ESTIMATORS = ("DM", "IPS", "SNIPS", "DR", "SNDR")


@dataclass
class Logged:
    """One logged suggestion with explicit feedback."""

    decision_type: str
    ctx: dict[str, float]
    arms: list[str]
    action: str
    propensity: float
    reward: float
    group: int = 0  # e.g. the video: bootstrap resamples whole groups when given
    true_value: dict[str, float] = field(default_factory=dict)  # simulator only: E[r | x, a] per arm


def _x(dt: str, ctx: dict[str, float], arm: str) -> tuple[tuple[str, str], np.ndarray]:
    if dt == "emphasis":
        return (dt, "*"), context.vectorize_arm(ctx, arm)
    return (dt, arm), context.vectorize(dt, ctx)


class RewardModel:
    """Ridge q(x, a) per (decision type, arm); the shared action-feature model for emphasis. Clipped to [0, 1]."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.w: dict[tuple[str, str], np.ndarray] = {}
        self.mean: dict[str, float] = {}
        self.overall = 0.5

    def fit(self, data: Sequence[Logged]) -> RewardModel:
        xtx: dict = {}
        xty: dict = {}
        by_dt: dict[str, list[float]] = {}
        for d in data:
            by_dt.setdefault(d.decision_type, []).append(d.reward)
        self.overall = float(np.mean([d.reward for d in data])) if data else 0.5
        self.mean = {dt: float(np.mean(v)) for dt, v in by_dt.items()}
        # ridge on rewards centred at the decision type's mean: an arm seen 3 times stays close to that mean
        for d in data:
            k, x = _x(d.decision_type, d.ctx, d.action)
            xtx[k] = xtx.get(k, 0.0) + np.outer(x, x)
            xty[k] = xty.get(k, 0.0) + (d.reward - self.mean[d.decision_type]) * x
        for k, a in xtx.items():
            self.w[k] = np.linalg.solve(a + self.alpha * np.eye(a.shape[0]), xty[k])
        return self

    def q(self, dt: str, ctx: dict[str, float], arm: str) -> float:
        k, x = _x(dt, ctx, arm)
        w = self.w.get(k)
        m = self.mean.get(dt, self.overall)
        return m if w is None else float(np.clip(m + w @ x, 0.0, 1.0))


def _cross_fit_q(data: Sequence[Logged], folds: int, seed: int) -> list[RewardModel]:
    """models[i] = reward model fitted WITHOUT the fold that contains decision i."""
    rng = np.random.default_rng(seed)
    groups = sorted({d.group for d in data})
    fold_of_group = {g: int(f) for g, f in zip(groups, rng.permutation(len(groups)) % folds, strict=True)}
    fold = [fold_of_group[d.group] if len(groups) >= folds else i % folds for i, d in enumerate(data)]
    models = [RewardModel().fit([d for d, f in zip(data, fold, strict=True) if f != k]) for k in range(folds)]
    return [models[f] for f in fold]


@dataclass
class Terms:
    """Per-decision terms; every estimator is a (ratio of) mean(s) of these, which makes bootstrapping cheap."""

    w: np.ndarray  # importance weight pi_e(a_i|x_i) / p_i
    r: np.ndarray
    dm: np.ndarray  # sum_a pi_e(a|x_i) q(x_i, a)
    q_logged: np.ndarray  # q(x_i, a_i)
    truth: np.ndarray  # sum_a pi_e(a|x_i) E[r | x_i, a] (nan outside the simulator)
    group: np.ndarray


def terms(data: Sequence[Logged], target: TargetPolicy, folds: int = 2, seed: int = 0) -> Terms:
    qs = _cross_fit_q(data, folds, seed)
    w, r, dm, ql, truth, grp = [], [], [], [], [], []
    for d, qm in zip(data, qs, strict=True):
        pe = target(d.decision_type, d.ctx, d.arms)
        w.append(pe.get(d.action, 0.0) / max(d.propensity, 1e-6))
        r.append(d.reward)
        dm.append(sum(p * qm.q(d.decision_type, d.ctx, a) for a, p in pe.items()))
        ql.append(qm.q(d.decision_type, d.ctx, d.action))
        truth.append(sum(p * d.true_value[a] for a, p in pe.items()) if d.true_value else np.nan)
        grp.append(d.group)
    return Terms(*(np.asarray(v, dtype=float) for v in (w, r, dm, ql, truth)), np.asarray(grp))


def _estimates(t: Terms, idx: np.ndarray) -> dict[str, float]:
    w, r = t.w[idx], t.r[idx]
    sw = w.sum()
    return {
        "DM": float(t.dm[idx].mean()),
        "IPS": float((w * r).mean()),
        "SNIPS": float((w * r).sum() / sw) if sw > 0 else float("nan"),
        "DR": float((t.dm[idx] + w * (r - t.q_logged[idx])).mean()),
        "SNDR": float(t.dm[idx].mean() + (w * (r - t.q_logged[idx])).sum() / sw) if sw > 0 else float("nan"),
    }


@dataclass
class Result:
    estimates: dict[str, float]
    ci: dict[str, tuple[float, float]]
    n: int
    ess: float  # effective sample size of the importance weights, (sum w)^2 / sum w^2
    max_weight: float
    truth: float  # simulator ground truth on the same contexts (nan for real logs)


def evaluate(data: Sequence[Logged], target: TargetPolicy, n_boot: int = 300, folds: int = 2,
             seed: int = 0) -> Result:
    if not data:
        nan = float("nan")
        return Result(dict.fromkeys(ESTIMATORS, nan), dict.fromkeys(ESTIMATORS, (nan, nan)), 0, 0.0, nan, nan)
    t = terms(data, target, folds, seed)
    n = len(t.r)
    est = _estimates(t, np.arange(n))
    rng = np.random.default_rng(seed + 1)
    groups = np.unique(t.group)
    members = {g: np.flatnonzero(t.group == g) for g in groups}
    boots = []
    for _ in range(n_boot):  # cluster bootstrap: decisions of one video are not independent
        pick = rng.choice(groups, size=len(groups), replace=True)
        boots.append(_estimates(t, np.concatenate([members[g] for g in pick])))
    g = len(groups)
    crit = float(stats.t.ppf(0.975, g - 1)) * np.sqrt(g / (g - 1)) if g > 1 else float("inf")
    ci = {k: (est[k] - crit * float(np.nanstd([b[k] for b in boots])),
              est[k] + crit * float(np.nanstd([b[k] for b in boots]))) for k in ESTIMATORS}
    ess = float(t.w.sum() ** 2 / (t.w ** 2).sum()) if (t.w ** 2).sum() > 0 else 0.0
    return Result(est, ci, n, ess, float(t.w.max()), float(np.nanmean(t.truth)) if not np.isnan(t.truth).all() else float("nan"))


# --------------------------------------------------------------------------- target policies
def greedy_target(policy) -> TargetPolicy:
    """A deterministic policy that proposes `policy`'s posterior-mean argmax."""

    def pe(dt: str, ctx: dict[str, float], arms: list[str]) -> dict[str, float]:
        s = policy.mean_scores(dt, ctx)
        best = max((a for a in arms if a in s), key=lambda a: s[a])
        return {a: float(a == best) for a in arms}

    return pe


def stochastic_target(policy) -> TargetPolicy:
    """A stochastic policy's own action probabilities (e.g. LinTS's Thompson-sampling distribution)."""

    def pe(dt: str, ctx: dict[str, float], arms: list[str]) -> dict[str, float]:
        p = policy.choose(dt, ctx).probs
        tot = sum(p.get(a, 0.0) for a in arms) or 1.0
        return {a: p.get(a, 0.0) / tot for a in arms}

    return pe
