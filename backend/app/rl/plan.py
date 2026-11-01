"""Sequential layer (PLAN 7.6): fit an edit to a target duration by planning, not by greedy per-decision choices.

Why a bandit is not enough: "make this 30 s" is a constraint on the WHOLE edit. Whether phrase 12 should be cut
depends on everything kept before it and on what is still to come, so the per-decision bandit (which only sees its
own context) cannot respect it. That is a finite-horizon MDP:

    stages  k = 0..K-1   the phrase and pause decisions in time order (exactly the order EditSession asks them)
    state   s = (out, sc) output seconds so far (0.1 s grid), whether the previous phrase was kept (sc = 1) or cut (0)
    actions phrase: KEEP | CUT        pause: KEEP | TIGHT_40 | TIGHT_15 | REMOVE
    step    out' = out + seconds the action keeps; sc' = 0 after a cut, sc + 1 after a kept phrase
    reward  r_k(a) = theta_a . x_k: the per-user bandit's learned reward model, with the history features held at their
            values on the bandit's own trajectory. Letting them follow the MDP state looks more exact but is a trap
            (model exploitation): the model is linear and never saw states like "everything cut so far", it
            extrapolates high rewards there, and an exact solver then happily cuts the whole video.
    final   -beta * max(0, out / target - 1 - tol) when too long, -beta_under * max(0, 1 - tol - out / target) when
            short. A target is mainly a limit: going over must be fixed, but a clip whose good edit is shorter should
            not be padded with dead air or stuffed with retakes to reach it.
    ties    a phrase that continues the sentence of the phrase before it follows that phrase's fate (sc = 0 means the
            previous phrase was cut), so a length target can never leave half a sentence behind.

It is solved exactly by backward induction (Bellman optimality, V_k(s) = max_a r + V_{k+1}(s')), vectorised over the
whole state grid: a few ms per stage. The transition model is known and deterministic and the horizon is short, so
exact dynamic programming is optimal for the given reward; PPO (the PLAN's first idea) would only approximate this
solution from samples.

Exploration (posterior sampling for RL, Osband, Russo & Van Roy 2013): per video, theta is DRAWN from the bandit's
posterior and the MDP is solved for that draw. This is Thompson sampling lifted from single decisions to whole
episodes; with no target (beta = 0) it reduces to the bandit. The policy is closed loop: an action is looked up at
the state actually reached, so overrides by the creator are absorbed by the plan. Propensities (for off-policy
evaluation) are the share of `n_draws` posterior draws whose optimal plan takes the same action in that state,
mixed with the same epsilon floor as LinTS.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.edl.build import PAD, PAUSE_TARGET
from app.pipeline.crew import THRESHOLD as CREW_THRESHOLD
from app.pipeline.segment import is_sentence_end
from app.policy import context
from app.schemas import UnitsFile

GRID = 0.1  # s per output-time bucket
SC_MAX = 1  # sc: 0 = the previous phrase was cut, 1 = kept
BETA = 150.0  # weight of going over the target (reward units per 100 % over): strong, a limit should hold
BETA_UNDER = 6.0  # weight of falling short of it
TOL = 0.05  # +-5 % of the target is free
CALIBRATION_STEPS = 2  # re-plan against the real EDL length this many times
N_DRAWS = 16
EPS_FLOOR = 0.02
CREW_CUT_REWARD = 4.0


@dataclass
class Stage:
    decision_type: str  # "phrase" | "pause"
    unit_id: int
    delta: dict[str, float]  # output seconds each action keeps
    x: np.ndarray  # context vector (history features from the bandit's own trajectory)
    tied: bool = False  # phrase continuing the sentence of the previous phrase: must share its fate
    crew: bool = False  # sounds like talk to the crew: the plan should cut it (pipeline/crew.py)


def _trajectory_history(uf: UnitsFile, video_type: str, policy) -> dict[tuple[str, int], dict[str, float]]:
    """History features along the posterior-mean edit, per (decision type, unit)."""
    from app.session import EditSession

    sess = EditSession(uf, None, video_type)
    out: dict[tuple[str, int], dict[str, float]] = {}
    for item in sess.items():
        if item.decision_type == "emphasis":
            break
        out[(item.decision_type, item.target_id)] = {k: v for k, v in item.ctx.items() if k.startswith("hist_")}
        s = policy.mean_scores(item.decision_type, item.ctx)
        sess.commit(item, max(s, key=lambda a: s[a]))
    return out


def build_stages(uf: UnitsFile, video_type: str, policy) -> list[Stage]:
    """Stages in the order EditSession asks them: (pause before phrase k, if any) then phrase k."""
    hist = _trajectory_history(uf, video_type, policy)
    speech = [u for u in uf.units if u.kind == "speech"]
    pause_before, n_sp = {}, 0
    for u in uf.units:
        if u.kind == "speech":
            n_sp += 1
        elif u.kind == "pause":
            pause_before[n_sp] = u
    of = lambda dt, u: hist.get((dt, u.id), {"hist_output_ratio": 1.0})
    out: list[Stage] = []
    for k, su in enumerate(speech):
        p = pause_before.get(k)
        if p is not None and k > 0:
            gap = p.end - p.start
            delta = {a: gap if a == "KEEP" else min(gap, 2 * PAD + PAUSE_TARGET[a]) for a in context.ARMS["pause"]}
            out.append(_stage("pause", p.id, delta, p.features, of("pause", p), video_type))
        d = su.end - su.start
        st = _stage("phrase", su.id, {"KEEP": d, "CUT": 0.0}, su.features, of("phrase", su), video_type)
        st.tied = k > 0 and _continues(speech[k - 1], su)
        st.crew = su.features.get("crew_score", 0.0) >= CREW_THRESHOLD
        out.append(st)
    return out


def _continues(prev, su) -> bool:
    """`su` carries on the sentence `prev` started: no sentence end between them, and `prev` is real content rather
    than an abandoned take or a lone opener ("Okay,"), which may be cut without touching what follows."""
    f = prev.features
    return (not is_sentence_end(prev.text.rstrip()) and f.get("is_earlier_take", 0.0) < 0.5
            and su.features.get("is_earlier_take", 0.0) < 0.5 and f.get("n_content_words", 0.0) >= 2)


def _stage(dt, uid, delta, feats, hist, video_type) -> Stage:
    return Stage(dt, uid, delta, context.vectorize(dt, context.build_context(feats, hist, video_type)))


class Plan:
    """The optimal closed-loop policy of the MDP for one reward model (one posterior draw or the mean)."""

    def __init__(self, stages: list[Stage], theta: dict[tuple[str, str], np.ndarray], target: float | None,
                 offsets: dict[str, dict[str, float]] | None = None, beta: float = BETA, tol: float = TOL,
                 beta_under: float = BETA_UNDER):
        self.stages = stages
        nb = int(np.ceil(sum(max(s.delta.values()) for s in stages) / GRID)) + 2
        out = np.arange(nb) * GRID  # output seconds per bucket
        sc = np.arange(SC_MAX + 1)
        if target and target > 0:
            over = np.maximum(out / target - 1 - tol, 0.0)
            under = np.maximum(1 - tol - out / target, 0.0)
            v = np.broadcast_to(-(beta * over + beta_under * under)[:, None], (nb, SC_MAX + 1)).copy()
        else:
            v = np.zeros((nb, SC_MAX + 1))
        self.best: list[np.ndarray] = [None] * len(stages)  # type: ignore[list-item]  # (nb, sc) arm index
        self.value0 = 0.0
        for k in range(len(stages) - 1, -1, -1):
            st = stages[k]
            arms = context.ARMS[st.decision_type]
            q = np.empty((len(arms), nb, SC_MAX + 1))
            for ai, a in enumerate(arms):
                r = float(theta[(st.decision_type, a)] @ st.x) + (offsets or {}).get(st.decision_type, {}).get(a, 0.0)
                if st.crew and a == "CUT":
                    r += CREW_CUT_REWARD
                shift = int(round(st.delta[a] / GRID))
                nxt_out = np.minimum(np.arange(nb) + shift, nb - 1)
                if st.decision_type == "phrase":
                    nxt_sc = np.full_like(sc, 0 if a == "CUT" else 1)
                else:
                    nxt_sc = sc
                q[ai] = r + v[nxt_out][:, nxt_sc]
                if st.tied:  # sc = 0: the previous phrase was cut, so this one goes too; otherwise it stays
                    q[ai][:, 0 if a == "KEEP" else slice(1, None)] = -np.inf
            self.best[k] = q.argmax(axis=0).astype(np.int8)
            v = q.max(axis=0)
        self.value0 = float(v[0, 0])
        self.index = {(s.decision_type, s.unit_id): k for k, s in enumerate(stages)}

    def action(self, k: int, out_t: float, since_cut: int) -> str:
        b = min(max(int(round(out_t / GRID)), 0), self.best[k].shape[0] - 1)
        return context.ARMS[self.stages[k].decision_type][int(self.best[k][b, min(max(since_cut, 0), SC_MAX)])]

    def rollout(self) -> tuple[dict[int, str], float]:
        """Open-loop actions if nobody intervenes, and the output seconds they give (planner's model)."""
        out_t, sc, acts = 0.0, 0, {}
        for k, st in enumerate(self.stages):
            a = self.action(k, out_t, sc)
            acts[st.unit_id] = a
            out_t += st.delta[a]
            if st.decision_type == "phrase":
                sc = 0 if a == "CUT" else 1
        return acts, out_t


def mean_theta(policy) -> dict[tuple[str, str], np.ndarray]:
    return {(dt, a): np.array(policy.theta(dt, a)) for dt in ("phrase", "pause") for a in context.ARMS[dt]}


def sample_theta(policy, rng: np.random.Generator) -> dict[tuple[str, str], np.ndarray]:
    """One draw from the LinTS posterior, theta_a ~ N(theta_hat_a, v^2 A_a^-1), per arm."""
    out = {}
    v = getattr(policy, "v", 0.0)
    for dt in ("phrase", "pause"):
        for a in context.ARMS[dt]:
            m = policy._model(dt, a)
            out[(dt, a)] = m.theta + v * (m.chol @ rng.standard_normal(m.d))
    return out


def _edl_seconds(uf: UnitsFile, actions: dict[int, str], fillers: dict[int, str]) -> float:
    from app.edl.build import Decisions, build_edl_with_lines

    return build_edl_with_lines(uf, Decisions(actions={**actions, **fillers}))[0].duration


def _filler_choices(uf: UnitsFile, video_type: str, policy) -> dict[int, str]:
    """What the bandit would do with each filler (they are not planned, but they shorten the edit)."""
    out = {}
    for u in uf.units:
        if u.kind == "filler" and u.features.get("clean_gap", 1.0) >= 1.0:
            s = policy.mean_scores("filler", context.build_context(u.features, {}, video_type))
            out[u.id] = max(s, key=lambda a: s[a])
    return out


class EpisodePlanner:
    """PSRL over one video: the acting plan (one posterior draw, or the mean when greedy) plus `n_draws` more
    draws to estimate the propensity of each action taken."""

    def __init__(self, uf: UnitsFile, video_type: str, policy, target: float | None, *, greedy: bool = False,
                 offsets: dict[str, dict[str, float]] | None = None, seed: int = 0, n_draws: int = N_DRAWS,
                 beta: float = BETA, eps: float = EPS_FLOOR):
        self.rng = np.random.default_rng(seed)
        self.stages = build_stages(uf, video_type, policy)
        self.eps_of = getattr(policy, "eps_of", lambda _dt: eps)
        self.greedy = greedy
        # The MDP's step model is an approximation of the EDL (fillers come out later, pauses next to a cut phrase
        # merge), so the target it plans for is corrected until the REAL edit of the mean plan has the wanted length.
        fillers = _filler_choices(uf, video_type, policy)
        goal = target
        self.mean_plan = Plan(self.stages, mean_theta(policy), goal, offsets, beta)
        for _ in range(CALIBRATION_STEPS if target else 0):
            acts, _ = self.mean_plan.rollout()
            real = _edl_seconds(uf, acts, fillers)
            if real <= 0 or abs(real - target) / target <= TOL / 2:
                break
            goal = max(goal + target - real, 1.0)
            self.mean_plan = Plan(self.stages, mean_theta(policy), goal, offsets, beta)
        target = goal
        self.act_plan = self.mean_plan if greedy else Plan(self.stages, sample_theta(policy, self.rng), target,
                                                           offsets, beta)
        self.draws = [Plan(self.stages, sample_theta(policy, self.rng), target, offsets, beta)
                      for _ in range(n_draws)]

    def handles(self, decision_type: str, unit_id: int) -> bool:
        return (decision_type, unit_id) in self.act_plan.index

    def choose(self, decision_type: str, unit_id: int, out_t: float, since_cut: int):
        """-> (action, propensity, probs, mean-plan action) at the state actually reached."""
        k = self.act_plan.index[(decision_type, unit_id)]
        arms = context.ARMS[decision_type]
        votes = np.zeros(len(arms))
        for p in self.draws:
            votes[arms.index(p.action(k, out_t, since_cut))] += 1
        # the same exploration rule as the bandit (none on keep/cut in the product, and never against a sentence tie)
        eps = self.eps_of(decision_type)
        probs = (1 - eps) * votes / max(len(self.draws), 1) + eps / len(arms)
        a = self.act_plan.action(k, out_t, since_cut)
        if not self.greedy and not self.stages[k].tied and self.rng.random() < eps:
            a = arms[int(self.rng.integers(len(arms)))]
        return a, float(probs[arms.index(a)]), dict(zip(arms, probs.tolist(), strict=True)), \
            self.mean_plan.action(k, out_t, since_cut)
