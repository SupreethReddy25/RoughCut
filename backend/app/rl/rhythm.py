"""Visual rhythm agent: a real sequential decision problem, learned with randomized least-squares value iteration.

At every kept phrase the agent picks a visual beat: nothing, a punch-in (1.15x / 1.3x), a text pop-up or a sound effect.
Unlike keep/cut, the action changes the next state, and the state is what the creator reacts to: three punch-ins in a
row are tiring, one after eight quiet seconds is welcome.

    state    seconds since the last visual change, which kind that was, how many came in a row, plus the phrase
    actions  NONE | Z115 | Z130 | TEXT_POP | SFX
    reward   accept +1, override -1 (and +1 for the action chosen instead), untouched +0.5, unseen 0 at low weight,
             plus the whole-video rating at export on the last step: a delayed reward value iteration has to carry back

RLSVI (Osband, Van Roy & Wen 2016) is Thompson sampling for value functions: rewards and the prior are perturbed with
Gaussian noise, fitted value iteration is run on the perturbed data, and the agent acts greedily on that one draw for a
whole video. So it explores coherently across an episode, which single-step Thompson sampling cannot. Q is linear per
arm. With gamma = 0 the same code is a contextual bandit on the same features, the baseline of experiment E11.

The agent is rebuilt from the suggestion and event log on every use, so undo and replay stay exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

import numpy as np

from app.policy.base import Choice

ARMS = ["NONE", "Z115", "Z130", "TEXT_POP", "SFX"]
LAST_NONE, LAST_ZOOM, LAST_TEXT, LAST_SFX = 0, 1, 2, 3
LAST_OF = {"Z115": LAST_ZOOM, "Z130": LAST_ZOOM, "TEXT_POP": LAST_TEXT, "SFX": LAST_SFX}
SINCE_NEVER = 30.0
GAMMA = 0.9
LAMBDA = 1.0
SIGMA = 0.15  # reward noise of a draw: enough to try things, small enough not to scatter effects on a first video
SWEEPS = 10  # value-iteration sweeps per draw
N_DRAWS = 16  # extra draws that estimate how likely the acting draw's choice was (propensity, review confidence)
UNSEEN_WEIGHT = 0.2  # a step nobody looked at carries no reward of its own, only the value of what follows it
RATING_REWARD = 2.0  # the export rating, on the last step of the episode
MAX_ROWS = 3000
Q_CLIP = 6.0


def phi(ctx: dict[str, float]) -> np.ndarray:
    """State features of one rhythm decision."""
    since = float(ctx.get("hist_since_visual", SINCE_NEVER))
    last = int(ctx.get("hist_last_visual", LAST_NONE))
    return np.array([
        1.0,
        float(since < 2.0), float(2.0 <= since < 4.0), float(4.0 <= since < 7.0), float(7.0 <= since < 12.0), float(since >= 12.0),
        min(since, SINCE_NEVER) / SINCE_NEVER,
        float(last == LAST_ZOOM), float(last == LAST_TEXT), float(last == LAST_SFX),
        min(float(ctx.get("hist_streak", 0.0)), 4.0) / 2.0,
        float(np.clip(ctx.get("loudness_z", 0.0) / 3.0, -2.0, 2.0)),
        min(float(ctx.get("duration", 0.0)), 12.0) / 6.0,
        float(ctx.get("topic_centrality", 0.0)),
        float(ctx.get("has_emphasis_punct", 0.0)),
        float(ctx.get("is_first", 0.0)),
        float(ctx.get("position", 0.0)),
    ])


D = 17


@cache
def default_prior() -> tuple[np.ndarray, ...]:
    """Before any feedback: a calm default, one modest beat every several seconds, on the phrases that matter. Fitted to
    a small hand-written rule over random states, so it is a starting point in the same form as what is learned."""
    rng = np.random.default_rng(7)
    X, Y = [], []
    for _ in range(600):
        ctx = {"hist_since_visual": float(rng.uniform(0, 20)), "hist_last_visual": int(rng.integers(0, 4)),
               "hist_streak": float(rng.integers(0, 3)), "loudness_z": float(rng.normal()), "duration": float(rng.uniform(.5, 7)),
               "topic_centrality": float(rng.uniform(0, 1)), "has_emphasis_punct": float(rng.random() < .15),
               "is_first": float(rng.random() < .05), "position": float(rng.uniform(0, 1))}
        since, key = ctx["hist_since_visual"], ctx["topic_centrality"] + .3 * ctx["has_emphasis_punct"] + .1 * ctx["loudness_z"]
        y = {"NONE": .5, "Z115": -.6, "Z130": -.9, "TEXT_POP": -.9, "SFX": -1.0}
        if since >= 6.0:
            y.update(NONE=.1, Z115=.3 + .6 * key, Z130=-.3 + .6 * key, TEXT_POP=-.4 + .8 * key, SFX=-.6)
        X.append(phi(ctx))
        Y.append([y[a] for a in ARMS])
    x, y = np.array(X), np.array(Y)
    theta = np.linalg.solve(x.T @ x + np.eye(D), x.T @ y)  # (D, arms)
    return tuple(theta[:, k] for k in range(len(ARMS)))


@dataclass
class Step:
    ctx: dict[str, float]
    action: str  # what was proposed
    reward: float | None = None  # None: nobody looked at it
    final: str | None = None  # the action the creator switched to


@dataclass
class Episode:
    steps: list[Step] = field(default_factory=list)
    rating: float | None = None  # -1, 0, +1


class RhythmAgent:
    name = "rlsvi"

    def __init__(self, seed: int = 0, gamma: float = GAMMA, lam: float = LAMBDA, sigma: float = SIGMA, sweeps: int = SWEEPS,
                 n_draws: int = N_DRAWS, greedy: bool = False):
        self.rng = np.random.default_rng(seed)
        self.gamma, self.lam, self.sigma, self.sweeps, self.n_draws = gamma, lam, sigma, sweeps, n_draws
        self.greedy = greedy
        self.episodes: list[Episode] = []
        self._ready = False

    def add_episode(self, ep: Episode) -> None:
        if ep.steps:
            self.episodes.append(ep)
            self._ready = False

    # ---- data
    def _rows(self):
        S, A, R, W, N, H = [], [], [], [], [], []

        def row(x, arm, r, w, nxt):
            S.append(x), A.append(ARMS.index(arm)), R.append(r), W.append(w)
            N.append(nxt if nxt is not None else np.zeros(D)), H.append(float(nxt is not None))

        for ep in self.episodes:
            xs = [phi(s.ctx) for s in ep.steps]
            for i, s in enumerate(ep.steps):
                nxt = xs[i + 1] if i + 1 < len(xs) else None
                last_bonus = RATING_REWARD * ep.rating if ep.rating is not None and nxt is None else 0.0
                if s.reward is None:
                    w = 1.0 if last_bonus else UNSEEN_WEIGHT
                    row(xs[i], s.action, last_bonus, w, nxt)
                    continue
                row(xs[i], s.action, s.reward + last_bonus, 1.0, nxt)
                if s.final is not None and s.final != s.action and s.final in ARMS:
                    row(xs[i], s.final, 1.0 + last_bonus, 1.0, nxt)
        if len(S) > MAX_ROWS:
            S, A, R, W, N, H = S[-MAX_ROWS:], A[-MAX_ROWS:], R[-MAX_ROWS:], W[-MAX_ROWS:], N[-MAX_ROWS:], H[-MAX_ROWS:]
        if not S:
            return None
        return np.array(S), np.array(A), np.array(R, float), np.array(W, float), np.array(N), np.array(H)

    def _prepare(self) -> None:
        self._data = self._rows()
        self._ainv, self._mask = {}, {}
        if self._data is not None:
            S, A, _, W, _, _ = self._data
            for k in range(len(ARMS)):
                m = A == k
                x = S[m]
                self._mask[k] = m
                self._ainv[k] = np.linalg.inv((x * W[m, None]).T @ x + self.lam * np.eye(D))
        self._ready = True

    def draw(self, perturb: bool = True) -> np.ndarray:
        """One value function: (arms, D) weights. perturb=False is the plain fitted value iteration (the mean)."""
        if not self._ready:
            self._prepare()
        prior = np.stack(default_prior())
        if perturb:
            prior = prior + self.rng.standard_normal(prior.shape) * self.sigma / np.sqrt(self.lam)
        if self._data is None:
            return prior
        S, _, R, W, N, H = self._data
        r = R + (self.rng.standard_normal(len(R)) * self.sigma / np.sqrt(W) if perturb else 0.0)
        theta = prior.copy()
        for _ in range(self.sweeps if self.gamma > 0 else 1):
            y = r + self.gamma * H * np.clip((N @ theta.T).max(axis=1), -Q_CLIP, Q_CLIP)
            for k in range(len(ARMS)):
                m = self._mask[k]
                theta[k] = self._ainv[k] @ ((S[m] * W[m, None]).T @ y[m] + self.lam * prior[k])
        return theta

    # ---- acting
    def begin_video(self) -> None:
        """Fix the draws for one video: acting greedily on one draw for a whole episode is what RLSVI means."""
        self._mean = self.draw(perturb=False)
        self._act = self._mean if self.greedy else self.draw()
        self._draws = [self.draw() for _ in range(self.n_draws)]

    def mean_action(self, ctx: dict[str, float]) -> str:
        return ARMS[int(np.argmax(self._mean @ phi(ctx)))]

    def choose(self, ctx: dict[str, float]) -> Choice:
        x = phi(ctx)
        votes = np.zeros(len(ARMS))
        for t in self._draws:
            votes[int(np.argmax(t @ x))] += 1
        probs = (votes + 0.05) / (votes.sum() + 0.05 * len(ARMS))
        k = int(np.argmax(self._act @ x))
        best = int(np.argmax(self._mean @ x))
        return Choice(ARMS[k], float(probs[k]), float(probs[k]), k != best, dict(zip(ARMS, probs.tolist(), strict=True)))


def episodes_from_log(rows: list, events: list) -> list[Episode]:
    """Episodes of one creator from their rhythm suggestions (active rows, in order, any video) and non-undone events."""
    by_video: dict[int, list] = {}
    for r in rows:
        by_video.setdefault(r.video_id, []).append(r)
    last: dict[int, tuple[str, dict]] = {}
    rating: dict[int, float] = {}
    for e in events:
        if e.kind == "rating":
            rating[e.video_id] = float(e.payload_json.get("value", 0))
        elif e.suggestion_id is not None and e.kind in ("accept", "reject", "override", "implicit"):
            last[e.suggestion_id] = (e.kind, e.payload_json)
    out = []
    for vid in sorted(by_video):
        ep = Episode(rating=rating.get(vid))
        for r in sorted(by_video[vid], key=lambda r: r.id):
            s = Step(dict(r.context_raw_json), r.proposed_action)
            kind, payload = last.get(r.id, (None, {}))
            if kind == "accept":
                s.reward = 1.0
            elif kind == "implicit":
                s.reward = 0.5
            elif kind in ("reject", "override"):
                s.reward = -1.0
                s.final = payload.get("final_action") or payload.get("applied")
            ep.steps.append(s)
        out.append(ep)
    return out


def explain(ctx: dict[str, float], action: str) -> str:
    since = float(ctx.get("hist_since_visual", SINCE_NEVER))
    quiet = "No visual change yet in this video." if since >= SINCE_NEVER else f"The last visual change was {since:.1f} s ago."
    if action == "NONE":
        return f"{quiet} The agent expects the best overall video from leaving this phrase alone."
    return f"{quiet} The agent expects a {action.replace('_', ' ').lower()} here to pay off over the rest of the video, not just on this phrase."
