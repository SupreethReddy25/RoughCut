"""Linear contextual bandits (PLAN 7.1): LinTS (ours), LinUCB, epsilon-greedy, and the never-updated Global baseline.

Sufficient statistics are split into prior and data, so the prior can be swapped without touching the data:
    prior:      A0 = lam * I,  b0 = lam * theta_prior
    data:       A_d <- gamma * A_d + w x x^T ;  b_d <- gamma * b_d + w r x
    posterior:  A = A0 + A_d ;  theta_hat = A^-1 (b0 + b_d) ;  Sigma = v^2 A^-1

Disjoint form (D1-D4): one Ridge per (decision type, arm).   Action-feature form (D5): ONE Ridge whose input is
the arm's own feature vector, so what is learned about "numbers" transfers to every caption line.
"""
from __future__ import annotations

import numpy as np

from app.policy import context
from app.policy.base import Choice, Policy, softmax_confidence
from app.policy.prior import SHARED, Prior, key_of

LAMBDA = 2.0
V = 0.25  # PLAN default 0.5; tuned on the simulator (E1): same accuracy at video 5, but confidences stay informative
GAMMA = 0.995
EPS_FLOOR = 0.02
N_DRAWS = 200


class Ridge:
    """Prior + discounted data statistics for one weight vector."""

    def __init__(self, d: int, lam: float, gamma: float, theta0: np.ndarray | None = None):
        self.d, self.lam, self.gamma = d, lam, gamma
        self.theta0 = np.zeros(d) if theta0 is None else np.array(theta0, dtype=float)
        self.A_d = np.zeros((d, d))
        self.b_d = np.zeros(d)
        self.n = 0
        self._dirty = True
        self._ainv = self._theta = self._chol = None

    def set_prior(self, theta0: np.ndarray) -> None:
        self.theta0 = np.array(theta0, dtype=float)
        self._dirty = True

    def update(self, x: np.ndarray, r: float, w: float = 1.0) -> None:
        if self.gamma != 1.0:
            self.A_d *= self.gamma
            self.b_d *= self.gamma
        self.A_d += w * np.outer(x, x)
        self.b_d += w * r * x
        self.n += 1
        self._dirty = True

    def _refresh(self) -> None:
        a = self.lam * np.eye(self.d) + self.A_d
        ainv = np.linalg.inv(a)
        self._ainv = (ainv + ainv.T) / 2
        self._theta = self._ainv @ (self.lam * self.theta0 + self.b_d)
        self._chol = None
        self._dirty = False

    @property
    def theta(self) -> np.ndarray:
        if self._dirty:
            self._refresh()
        return self._theta

    @property
    def ainv(self) -> np.ndarray:
        if self._dirty:
            self._refresh()
        return self._ainv

    @property
    def chol(self) -> np.ndarray:
        """Cholesky factor L of A^-1 (A^-1 = L L^T), for drawing theta ~ N(theta_hat, v^2 A^-1)."""
        if self._dirty:
            self._refresh()
        if self._chol is None:
            self._chol = np.linalg.cholesky(self._ainv)
        return self._chol


class LinearPolicy(Policy):
    """Shared storage + posterior queries; subclasses implement `choose`."""

    def __init__(self, prior: Prior | None = None, lam: float = LAMBDA, gamma: float = GAMMA,
                 seed: int | None = None):
        super().__init__(seed)
        self.prior = prior or Prior()
        self.lam, self.gamma = lam, gamma
        self.models: dict[tuple[str, str], Ridge] = {}

    # -- storage
    def _model(self, decision_type: str, arm: str) -> Ridge:
        k = key_of(decision_type, arm)
        m = self.models.get(k)
        if m is None:
            m = self.models[k] = Ridge(context.dim(decision_type), self.lam, self.gamma,
                                       self.prior.theta(decision_type, arm))
        return m

    def set_prior(self, prior: Prior) -> None:
        """Swap the prior (e.g. the style-mixture mean); learned data statistics are kept."""
        self.prior = prior
        for (dt, arm), m in self.models.items():
            m.set_prior(prior.theta(dt, arm))

    def update(self, decision_type: str, ctx: dict[str, float], arm: str, reward: float,
               weight: float = 1.0) -> None:
        if arm not in self.arms(decision_type, ctx):
            raise ValueError(f"unknown arm {arm!r} for {decision_type}")
        if not weight >= 0.0:  # a negative weight would make A_d indefinite and break the posterior
            raise ValueError(f"weight must be >= 0, got {weight}")
        x = context.vectorize_arm(ctx, arm) if decision_type == "emphasis" else context.vectorize(decision_type, ctx)
        self._model(decision_type, arm).update(x, float(reward), float(weight))
        self.n_updates += 1

    # -- posterior queries
    def _disjoint(self, decision_type: str, ctx: dict[str, float]):
        arms = context.ARMS[decision_type]
        x = context.vectorize(decision_type, ctx)
        mu = np.empty(len(arms))
        var = np.empty(len(arms))
        for i, a in enumerate(arms):
            m = self._model(decision_type, a)
            mu[i] = m.theta @ x
            var[i] = max(float(x @ m.ainv @ x), 0.0)
        return arms, mu, var

    def _shared(self, decision_type: str, ctx: dict[str, float]):
        arms = context.arms_for(decision_type, ctx)
        X = context.arm_matrix(ctx, arms)
        return arms, X, self._model(decision_type, SHARED)

    def mean_scores(self, decision_type: str, ctx: dict[str, float]) -> dict[str, float]:
        if decision_type == "emphasis":
            arms, X, m = self._shared(decision_type, ctx)
            return dict(zip(arms, (X @ m.theta).tolist(), strict=True))
        arms, mu, _ = self._disjoint(decision_type, ctx)
        return dict(zip(arms, mu.tolist(), strict=True))

    def theta(self, decision_type: str, arm: str) -> np.ndarray:
        """Posterior-mean weight vector of an arm (the shared one for emphasis)."""
        return self._model(decision_type, arm).theta

    def uncertainty(self, decision_type: str, ctx: dict[str, float]) -> dict[str, float]:
        """sqrt(x^T A^-1 x) per arm (posterior std of the score divided by v)."""
        if decision_type == "emphasis":
            arms, X, m = self._shared(decision_type, ctx)
            return dict(zip(arms, np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", X, m.ainv, X), 0)).tolist(),
                            strict=True))
        arms, _, var = self._disjoint(decision_type, ctx)
        return dict(zip(arms, np.sqrt(var).tolist(), strict=True))

    def _offset_vec(self, arms: list[str], offsets: dict[str, float] | None) -> np.ndarray:
        return np.array([(offsets or {}).get(a, 0.0) for a in arms])

    # -- persistence
    def state_dict(self) -> dict:
        return {
            "meta": {"name": self.name, "lam": self.lam, "gamma": self.gamma, "n_updates": self.n_updates},
            "prior": {k: v.copy() for k, v in ((f"{dt}|{a}", w) for (dt, a), w in self.prior.weights.items())},
            "models": {f"{dt}|{arm}": {"A_d": m.A_d.copy(), "b_d": m.b_d.copy(), "n": m.n}
                       for (dt, arm), m in self.models.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.n_updates = int(state["meta"]["n_updates"])
        if "prior" in state:  # theta_hat depends on the prior: restore the one the statistics were built with
            self.prior = Prior.from_dict({k: np.asarray(v, dtype=float).tolist() for k, v in state["prior"].items()})
        self.models = {}
        for k, v in state["models"].items():
            dt, arm = k.split("|", 1)
            m = Ridge(context.dim(dt), self.lam, self.gamma, self.prior.theta(dt, arm))
            m.A_d, m.b_d, m.n = np.array(v["A_d"], dtype=float), np.array(v["b_d"], dtype=float), int(v["n"])
            self.models[(dt, arm)] = m


class LinTS(LinearPolicy):
    """Linear Thompson Sampling with prior + discounting (ours)."""

    name = "linTS"

    def __init__(self, prior: Prior | None = None, lam: float = LAMBDA, v: float = V, gamma: float = GAMMA,
                 eps_floor: float = EPS_FLOOR, n_draws: int = N_DRAWS, seed: int | None = None):
        super().__init__(prior, lam, gamma, seed)
        self.v, self.eps_floor, self.n_draws = v, eps_floor, n_draws

    def choose(self, decision_type: str, ctx: dict[str, float],
               offsets: dict[str, float] | None = None) -> Choice:
        M, eps = self.n_draws, self.eps_floor
        if decision_type == "emphasis":
            arms, X, m = self._shared(decision_type, ctx)
            mu = X @ m.theta + self._offset_vec(arms, offsets)
            K = len(arms)
            theta = m.theta + self.v * (self.rng.standard_normal((M, m.d)) @ m.chol.T)  # (M, d) draws
            scores = theta @ X.T + self._offset_vec(arms, offsets)
        else:
            arms, mu, var = self._disjoint(decision_type, ctx)
            mu = mu + self._offset_vec(arms, offsets)
            K = len(arms)
            scores = mu + self.v * np.sqrt(var) * self.rng.standard_normal((M, K))  # arms are independent
        best = scores.argmax(axis=1)
        freq = np.bincount(best, minlength=K) / M
        p = (1 - eps) * freq + eps / K
        chosen = int(best[0])  # one Thompson draw decides; epsilon-floor mixes in a uniform pick
        if self.rng.random() < eps:
            chosen = int(self.rng.integers(K))
        explored = chosen != int(np.argmax(mu))
        return Choice(arms[chosen], float(p[chosen]), float(p[chosen]), explored, dict(zip(arms, p.tolist(), strict=True)))


class EpsGreedyLinear(LinearPolicy):
    """Ridge posterior-mean argmax, random arm with probability epsilon."""

    name = "epsGreedy"

    def __init__(self, prior: Prior | None = None, eps: float = 0.1, lam: float = LAMBDA, gamma: float = GAMMA,
                 seed: int | None = None):
        super().__init__(prior, lam, gamma, seed)
        self.eps = eps

    def choose(self, decision_type: str, ctx: dict[str, float],
               offsets: dict[str, float] | None = None) -> Choice:
        scores = self.mean_scores(decision_type, ctx)
        arms = list(scores)
        s = np.array(list(scores.values())) + self._offset_vec(arms, offsets)
        K, greedy = len(arms), int(np.argmax(s))
        chosen = int(self.rng.integers(K)) if self.rng.random() < self.eps else greedy
        p = np.full(K, self.eps / K)
        p[greedy] += 1 - self.eps
        return Choice(arms[chosen], float(p[chosen]), softmax_confidence(s, chosen), chosen != greedy,
                      dict(zip(arms, p.tolist(), strict=True)))


class LinUCB(LinearPolicy):
    """Optimism: argmax of theta_hat.x + alpha * sqrt(x^T A^-1 x). Deterministic (propensity 1)."""

    name = "linUCB"

    def __init__(self, prior: Prior | None = None, alpha: float = 1.0, lam: float = LAMBDA, gamma: float = GAMMA,
                 seed: int | None = None):
        super().__init__(prior, lam, gamma, seed)
        self.alpha = alpha

    def choose(self, decision_type: str, ctx: dict[str, float],
               offsets: dict[str, float] | None = None) -> Choice:
        mean = self.mean_scores(decision_type, ctx)
        arms = list(mean)
        mu = np.array(list(mean.values())) + self._offset_vec(arms, offsets)
        unc = self.uncertainty(decision_type, ctx)
        sd = np.array([unc[a] for a in arms])
        s = mu + self.alpha * sd
        chosen = int(np.argmax(s))
        p = np.zeros(len(arms))
        p[chosen] = 1.0
        return Choice(arms[chosen], 1.0, softmax_confidence(s, chosen), chosen != int(np.argmax(mu)),
                      dict(zip(arms, p.tolist(), strict=True)))


class GlobalPolicy(Policy):
    """Generic prior, never updated: argmax of the prior weights' scores."""

    name = "global"

    def __init__(self, prior: Prior, seed: int | None = None):
        super().__init__(seed)
        self.prior = prior

    def mean_scores(self, decision_type: str, ctx: dict[str, float]) -> dict[str, float]:
        return {a: self.prior.score(decision_type, ctx, a) for a in self.arms(decision_type, ctx)}

    def choose(self, decision_type: str, ctx: dict[str, float],
               offsets: dict[str, float] | None = None) -> Choice:
        scores = self.mean_scores(decision_type, ctx)
        arms = list(scores)
        s = np.array(list(scores.values())) + np.array([(offsets or {}).get(a, 0.0) for a in arms])
        chosen = int(np.argmax(s))
        p = np.zeros(len(arms))
        p[chosen] = 1.0
        return Choice(arms[chosen], 1.0, softmax_confidence(s, chosen), False, dict(zip(arms, p.tolist(), strict=True)))
