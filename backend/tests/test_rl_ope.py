"""Off-policy evaluation estimators (app/ope) and the episode planner (app/rl/plan.py)."""
from __future__ import annotations

import numpy as np
import pytest

from app.ope.estimators import Logged, evaluate
from app.ope.logs import logged_from, true_reward
from app.policy.linear import LinTS
from app.rl.plan import Plan, build_stages, mean_theta
from app.schemas import Event, Suggestion
from app.session import PLANNER_TAG, EditSession, suggest


# ---------------------------------------------------------------- OPE
def _bandit_log(n: int, seed: int = 0) -> list[Logged]:
    """Pause decisions logged by a known stochastic policy; reward probability depends on the context and arm."""
    rng = np.random.default_rng(seed)
    arms = ["KEEP", "TIGHT_40", "TIGHT_15", "REMOVE"]
    out = []
    for i in range(n):
        gap = float(rng.uniform(0.2, 3.0))
        ctx = {"gap_seconds": gap, "bias": 1.0, "type_other": 1.0}
        # truth: long gaps want REMOVE, short ones KEEP
        p_ok = {a: 0.9 if (a == "REMOVE") == (gap > 1.0) else 0.3 for a in arms}
        logp = np.array([0.55, 0.15, 0.15, 0.15])  # logging policy: mostly KEEP, but every arm has support
        k = int(rng.choice(4, p=logp))
        out.append(Logged("pause", ctx, arms, arms[k], float(logp[k]), float(rng.random() < p_ok[arms[k]]),
                          group=i // 10, true_value=p_ok))
    return out


def test_ips_snips_dr_recover_the_value_of_another_policy():
    data = _bandit_log(6000)
    target = lambda dt, ctx, arms: {a: float(a == ("REMOVE" if ctx["gap_seconds"] > 1.0 else "KEEP")) for a in arms}  # noqa: E731
    r = evaluate(data, target, n_boot=50)
    assert abs(r.truth - 0.9) < 1e-9
    for e in ("IPS", "SNIPS", "DR", "SNDR"):
        assert abs(r.estimates[e] - r.truth) < 0.04, (e, r.estimates[e])
        assert r.ci[e][0] <= r.truth <= r.ci[e][1] or abs(r.estimates[e] - r.truth) < 0.02
    # the naive average of logged rewards is the LOGGING policy's value, far below the target's
    assert np.mean([d.reward for d in data]) < 0.75


def test_logged_from_maps_verdicts_and_skips_non_policy_rows():
    def sug(i, src="policy"):
        return Suggestion(id=i, video_id="v", target_id=i, decision_type="phrase", proposed_action="KEEP",
                          context_raw={"bias": 1.0}, propensity=0.8, confidence=0.8, source=src)
    sugs = {0: sug(0), 1: sug(1), 2: sug(2), 3: sug(3, "autopilot"), 4: sug(4)}
    evs = [Event(id=0, user_id=1, video_id="v", kind="accept", suggestion_id=0),
           Event(id=1, user_id=1, video_id="v", kind="override", suggestion_id=1, payload={"final_action": "CUT"}),
           Event(id=2, user_id=1, video_id="v", kind="implicit", suggestion_id=2),
           Event(id=3, user_id=1, video_id="v", kind="accept", suggestion_id=3),
           Event(id=4, user_id=1, video_id="v", kind="reject", suggestion_id=4, undone=True)]
    rewards = sorted(d.reward for d in logged_from(sugs, evs))
    assert rewards == [0.0, 1.0, 1.0]  # accept 1, override-to-other 0, implicit 1; autopilot + undone left out


def test_true_reward_matches_the_persona_feedback_model():
    t = true_reward(["KEEP", "CUT"], "CUT", noise=0.0, p_override=1.0)
    assert t == {"KEEP": 0.0, "CUT": 1.0}
    t = true_reward(["KEEP", "CUT"], "CUT", noise=0.0, p_override=0.6)
    assert t["KEEP"] == pytest.approx(0.4)  # wrong but left alone 40 % of the time


# ---------------------------------------------------------------- planner
@pytest.fixture(scope="module")
def video():
    from sim.experiments.common import videos_for

    return videos_for(7, 1)[0]


@pytest.fixture(scope="module")
def policy():
    from sim.experiments.common import generic_prior

    return LinTS(generic_prior(), seed=0)


def _rollout_reward(video, policy, actions: dict[int, str]) -> float:
    """Sum of posterior-mean rewards of phrase/pause actions along the trajectory they produce."""
    sess = EditSession(video.uf, video.word_feats, video.video_type)
    tot = 0.0
    for item in sess.items():
        a = actions.get(item.target_id) if item.decision_type in ("phrase", "pause") else None
        if a is None:
            s = policy.mean_scores(item.decision_type, item.ctx)
            a = max(s, key=lambda k: s[k])
        else:
            tot += policy.mean_scores(item.decision_type, item.ctx)[a]
        sess.commit(item, a)
    return tot


def test_planner_does_not_exploit_the_reward_model_by_cutting_everything(video, policy):
    """Regression: with history features following the state, the solver found "cut it all" as the optimum."""
    stages = build_stages(video.uf, video.video_type, policy)
    acts, _ = Plan(stages, mean_theta(policy), target=None).rollout()
    phrases = [st for st in stages if st.decision_type == "phrase"]
    myopic = suggest(video.uf, video.word_feats, policy, video_type=video.video_type, greedy=True).decisions.actions
    agree = sum(acts[st.unit_id] == myopic.get(st.unit_id, "KEEP") for st in phrases) / len(phrases)
    assert agree >= 0.8 and sum(acts[st.unit_id] == "KEEP" for st in phrases) >= 0.5 * len(phrases)


def test_planner_stays_within_the_target_length(video, policy):
    stages = build_stages(video.uf, video.video_type, policy)
    from app.edl.build import build_edl_with_lines

    for target in (20.0, 35.0, 60.0):
        _, out = Plan(stages, mean_theta(policy), target=target).rollout()
        assert 0.6 * target <= out <= 1.1 * target  # the raw model; the real edit is calibrated below
        res = suggest(video.uf, video.word_feats, policy, video_type=video.video_type, greedy=True, target=target)
        assert 0.6 * target <= build_edl_with_lines(video.uf, res.decisions)[0].duration <= 1.1 * target


def test_suggest_with_target_uses_the_planner(video, policy):
    res = suggest(video.uf, video.word_feats, policy, video_type=video.video_type, target=25.0, seed=3)
    planned = [s for s in res.suggestions if s.policy_name.endswith(PLANNER_TAG)]
    assert planned and all(s.decision_type in ("phrase", "pause") for s in planned)
    assert all(0.0 < s.propensity <= 1.0 for s in planned)
    plain = suggest(video.uf, video.word_feats, policy, video_type=video.video_type)
    assert not any(s.policy_name.endswith(PLANNER_TAG) for s in plain.suggestions)


def test_planner_never_leaves_half_a_sentence(video, policy):
    stages = build_stages(video.uf, video.video_type, policy)
    tied = [k for k, st in enumerate(stages) if st.tied]
    assert tied  # the synthetic videos have multi-phrase sentences
    acts, _ = Plan(stages, mean_theta(policy), target=15.0).rollout()
    phrases = [st for st in stages if st.decision_type == "phrase"]
    for prev, st in zip(phrases, phrases[1:]):
        if st.tied:
            assert acts[st.unit_id] == acts[prev.unit_id]


def test_a_target_is_a_limit_and_a_short_edit_is_not_padded(video, policy):
    natural = suggest(video.uf, video.word_feats, policy, video_type=video.video_type, greedy=True)
    long_target = suggest(video.uf, video.word_feats, policy, video_type=video.video_type, greedy=True, target=600.0)
    n_keep = lambda r: sum(a == "KEEP" for u, a in r.decisions.actions.items())  # noqa: E731
    assert n_keep(long_target) <= n_keep(natural) + 3  # room to spare does not turn into kept dead air


def test_real_edit_length_is_calibrated_to_the_target(video, policy):
    from app.edl.build import build_edl_with_lines

    res = suggest(video.uf, video.word_feats, policy, video_type=video.video_type, greedy=True, target=25.0)
    assert build_edl_with_lines(video.uf, res.decisions)[0].duration <= 25.0 * 1.1
