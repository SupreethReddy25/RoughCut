"""Policies (PLAN 7): contexts, LinTS in both forms, baselines, propensities, replay == online, store, prior, explain."""
from __future__ import annotations

import numpy as np
import pytest

from app.policy import context
from app.policy.base import Policy
from app.policy.explain import explain
from app.policy.linear import EpsGreedyLinear, GlobalPolicy, LinTS, LinUCB
from app.policy.log_replay import Learner, recompute_from_log, replay
from app.policy.prior import Prior, fit_prior
from app.policy.rule import RulePolicy
from app.policy.store import PolicyStore
from app.policy.style import StyleMixture


def rand_ctx(dt: str, rng: np.random.Generator) -> dict[str, float]:
    ctx = {}
    for _, key, _arg in context.SPECS[dt]:
        ctx[key] = float(rng.uniform(0, 1))
    ctx.update(context.type_onehot(["tutorial", "vlog", "opinion", "other"][int(rng.integers(4))]))
    ctx["bias"] = 1.0
    ctx["hist_since_last_zoom"] = float(rng.uniform(0, 12))
    ctx["gap_seconds"] = float(rng.uniform(0.15, 3))
    return ctx


def rand_emph_ctx(rng: np.random.Generator) -> dict[str, float]:
    n = int(rng.integers(1, 4))
    ctx = {"n_cand": float(n), "line_len": float(rng.integers(1, 5)), "line_has_number": float(rng.random() < 0.3),
           "bias": 1.0, **context.type_onehot("vlog")}
    for k in range(n):
        for key, _ in context.CAND_KEYS:
            ctx[f"c{k}_{key}"] = float(rng.uniform(0, 1))
    return ctx


# ---------------------------------------------------------------- context vectors
def test_dims_names_and_bins():
    for dt in ("phrase", "pause", "zoom", "filler", "emphasis"):
        assert context.dim(dt) == len(context.feature_names(dt))
    c = {"gap_seconds": 0.55}
    v = context.vectorize("pause", c)
    names = context.feature_names("pause")
    hot = [n for n, x in zip(names, v, strict=True) if n.startswith("gap_seconds") and x == 1.0]
    assert hot == ["gap_seconds[0.5,0.6)"]  # exactly one bin is hot (one-hot)
    assert context.vectorize("pause", {"gap_seconds": 5.0})[names.index("gap_seconds>=2.0")] == 1.0
    assert np.all(np.abs(context.vectorize("phrase", {"loudness_z": 99.0, "duration": 1e6})) <= 3.0)  # clipped


def test_emphasis_vectors_share_one_space_and_candidate_choice():
    rng = np.random.default_rng(0)
    ctx = rand_emph_ctx(rng)
    n = int(ctx["n_cand"])
    assert context.arms_for("emphasis", ctx) == ["NONE"] + [f"CAND_{k}" for k in range(n)]
    xn, x0 = context.vectorize_arm(ctx, "NONE"), context.vectorize_arm(ctx, "CAND_0")
    assert xn.shape == x0.shape == (context.EMPH_DIM,)
    nc = len(context.CAND_KEYS)
    assert x0[nc] == 1.0 and xn[nc] == 0.0 and not xn[:nc].any() and not x0[nc + 1:].any()


def test_build_emphasis_context_picks_non_stopword_candidates():
    from tests.test_pipeline_units import seq

    words = seq("the three timeline of 40% sales and Tokyo")
    wf = {w.id: {"is_stopword": float(w.text_original in ("the", "of", "and")), "tfidf_in_transcript": t,
                 "is_number": float("%" in w.text_original or w.text_original == "three"), "word_loudness_z": 0.0}
          for w, t in zip(words, [0, 0.9, 0.5, 0, 0.8, 0.3, 0, 0.6], strict=True)}
    ctx, ids = context.build_emphasis_context(words, wf, "vlog")
    assert ids == [words[1].id, words[4].id, words[7].id]  # top-3 by tfidf, stopwords excluded
    assert ctx["n_cand"] == 3 and ctx["line_has_number"] == 1.0 and ctx["c0_word_id"] == words[1].id
    assert context.build_emphasis_context(seq("the of and"), {}, "vlog") is None


# ---------------------------------------------------------------- LinTS converges (both forms)
def test_lints_converges_disjoint_form():
    rng = np.random.default_rng(1)
    dt = "zoom"
    d, arms = context.dim(dt), context.ARMS[dt]
    theta = {a: rng.standard_normal(d) / np.sqrt(d) * 2 for a in arms}
    pol = LinTS(lam=1.0, v=0.25, gamma=1.0, seed=2)
    for _ in range(2500):
        ctx = rand_ctx(dt, rng)
        x = context.vectorize(dt, ctx)
        a = pol.choose(dt, ctx).action
        pol.update(dt, ctx, a, float(theta[a] @ x + rng.normal(0, 0.1)))
    test = [rand_ctx(dt, rng) for _ in range(400)]
    hit = sum(max(pol.mean_scores(dt, c), key=pol.mean_scores(dt, c).get)
              == max(arms, key=lambda a: theta[a] @ context.vectorize(dt, c)) for c in test)
    assert hit / len(test) >= 0.9
    for a in arms:  # arms that are chosen often predict their reward accurately (theta itself is not
        m = pol._model(dt, a)  # identifiable: one-hot groups are collinear with the bias)
        if m.n > 300:
            err = [m.theta @ context.vectorize(dt, c) - theta[a] @ context.vectorize(dt, c) for c in test]
            assert np.sqrt(np.mean(np.square(err))) < 0.15


def test_lints_converges_action_feature_form():
    rng = np.random.default_rng(3)
    dt = "emphasis"
    theta = rng.standard_normal(context.EMPH_DIM)
    pol = LinTS(lam=1.0, v=0.25, gamma=1.0, seed=4)
    for _ in range(1500):
        ctx = rand_emph_ctx(rng)
        a = pol.choose(dt, ctx).action
        pol.update(dt, ctx, a, float(theta @ context.vectorize_arm(ctx, a) + rng.normal(0, 0.1)))
    hits = 0
    for _ in range(400):
        ctx = rand_emph_ctx(rng)
        arms = context.arms_for(dt, ctx)
        best = max(arms, key=lambda a: theta @ context.vectorize_arm(ctx, a))
        sc = pol.mean_scores(dt, ctx)
        hits += max(sc, key=sc.get) == best
    assert hits / 400 >= 0.9
    assert len(pol.models) == 1  # ONE shared theta, not one per arm


# ---------------------------------------------------------------- propensities
def test_propensities_are_valid_probabilities():
    rng = np.random.default_rng(5)
    pol = LinTS(seed=6)
    eps = pol.eps_floor
    for dt in ("phrase", "pause", "zoom", "filler"):
        for _ in range(20):
            ctx = rand_ctx(dt, rng)
            ch = pol.choose(dt, ctx)
            K = len(context.ARMS[dt])
            assert set(ch.probs) == set(context.ARMS[dt])
            assert sum(ch.probs.values()) == pytest.approx(1.0)
            assert all(p >= eps / K - 1e-12 for p in ch.probs.values())  # the epsilon floor
            assert ch.propensity == ch.probs[ch.action] == ch.confidence and 0 < ch.propensity <= 1
    for _ in range(20):
        ctx = rand_emph_ctx(rng)
        ch = pol.choose("emphasis", ctx)
        assert sum(ch.probs.values()) == pytest.approx(1.0) and 0 < ch.propensity <= 1
        assert set(ch.probs) == set(context.arms_for("emphasis", ctx))


def test_logged_propensity_matches_empirical_choice_frequency():
    """The logged probabilities are an unbiased estimate of how often each action is really chosen."""
    rng = np.random.default_rng(7)
    ctx = rand_ctx("pause", rng)
    n = 2000
    counts = dict.fromkeys(context.ARMS["pause"], 0)
    logged = dict.fromkeys(context.ARMS["pause"], 0.0)
    for s in range(n):
        ch = LinTS(seed=s + 1).choose("pause", ctx)
        counts[ch.action] += 1
        for a, p in ch.probs.items():
            logged[a] += p / n
    for a in counts:
        assert counts[a] / n == pytest.approx(logged[a], abs=0.04)


def test_baseline_policies_interface():
    rng = np.random.default_rng(8)
    ctx = rand_ctx("pause", rng)
    ctx.update({"gap_seconds": 0.9})
    for pol in (EpsGreedyLinear(seed=1), LinUCB(seed=1), GlobalPolicy(Prior(), seed=1), RulePolicy(1), LinTS(seed=1)):
        ch = pol.choose("pause", ctx)
        assert ch.action in context.ARMS["pause"] and 0 < ch.propensity <= 1 and 0 <= ch.confidence <= 1
        assert sum(ch.probs.values()) == pytest.approx(1.0)
        assert set(pol.mean_scores("pause", ctx)) == set(context.ARMS["pause"])
    eg = EpsGreedyLinear(eps=0.1, seed=3).choose("pause", ctx)
    assert eg.propensity in (pytest.approx(0.1 / 4), pytest.approx(0.9 + 0.1 / 4))
    for p in (LinUCB(seed=1), GlobalPolicy(Prior(), seed=1), RulePolicy(1)):
        assert p.choose("pause", ctx).propensity == 1.0  # deterministic policies


def test_rule_policy_rules():
    r = RulePolicy()
    assert r.choose("pause", {"gap_seconds": 0.6}).action == "REMOVE"
    assert r.choose("pause", {"gap_seconds": 0.5}).action == "KEEP"
    assert r.choose("phrase", {"is_earlier_take": 1.0}).action == "CUT" and r.choose("phrase", {}).action == "KEEP"
    assert r.choose("filler", {"tok_um": 1.0}).action == "REMOVE" and r.choose("filler", {"tok_like": 1.0}).action == "KEEP"
    assert r.choose("zoom", {}).action == "NONE"
    ctx = {"n_cand": 3.0, "c0_word_len": 4.0, "c1_word_len": 9.0, "c2_word_len": 6.0}
    assert r.choose("emphasis", ctx).action == "CAND_1"
    assert r.choose("emphasis", {"n_cand": 0.0}).action == "NONE"
    assert r.update("pause", {}, "KEEP", 1.0) is None and r.n_updates == 0  # stateless


def test_offsets_shift_choice_and_nudges():
    from app.policy.reward import nudge_offsets

    ctx = {"gap_seconds": 0.4}
    pol = RulePolicy()
    assert pol.choose("pause", ctx).action == "KEEP"
    assert pol.choose("pause", ctx, {"REMOVE": 2.0}).action == "REMOVE"
    sn = nudge_offsets("snappier")
    assert sn["pause"]["REMOVE"] > 0 and sn["phrase"]["CUT"] > 0 and nudge_offsets("calmer")["pause"]["REMOVE"] < 0
    assert nudge_offsets("more_zooms")["zoom"]["Z115"] > 0 and nudge_offsets("fewer_zooms")["zoom"]["Z130"] < 0
    ts = LinTS(seed=1)
    off = {"CUT": 50.0}
    assert all(ts.choose("phrase", {"bias": 1.0}, off).action == "CUT" for _ in range(5))
    with pytest.raises(ValueError):
        nudge_offsets("louder")


def test_update_rejects_unknown_arm_and_discounting():
    pol = LinTS(gamma=0.9, seed=0)
    with pytest.raises(ValueError):
        pol.update("pause", {}, "NOPE", 1.0)
    with pytest.raises(ValueError):
        pol.update("emphasis", {"n_cand": 1.0}, "CAND_2", 1.0)
    ctx = {"gap_seconds": 1.0, "bias": 1.0}
    pol.update("pause", ctx, "KEEP", 1.0)
    a1 = pol._model("pause", "KEEP").A_d.copy()
    pol.update("pause", ctx, "KEEP", 1.0)
    x = context.vectorize("pause", ctx)
    assert np.allclose(pol._model("pause", "KEEP").A_d, 0.9 * a1 + np.outer(x, x))  # A_d <- gamma A_d + x x^T


def test_prior_is_the_posterior_mean_without_data():
    rng = np.random.default_rng(9)
    ctx = rand_ctx("zoom", rng)
    th = {("zoom", "Z115"): np.full(context.dim("zoom"), 0.2)}
    pol = LinTS(prior=Prior(th), seed=0)
    sc = pol.mean_scores("zoom", ctx)
    assert sc["Z115"] == pytest.approx(float(th[("zoom", "Z115")] @ context.vectorize("zoom", ctx)))
    assert sc["NONE"] == 0.0
    # data pulls the estimate away from the prior; swapping the prior keeps the data statistics
    for _ in range(50):
        pol.update("zoom", ctx, "Z115", -1.0)
    low = pol.mean_scores("zoom", ctx)["Z115"]
    assert low < sc["Z115"]
    pol.set_prior(Prior())
    assert pol._model("zoom", "Z115").n == 50 and pol.mean_scores("zoom", ctx)["Z115"] < low + 1e-9


# ---------------------------------------------------------------- replay == online
def _small_run(keep_log=True, mixture=False, algo=LinTS, seed=3, n_videos=3):
    from sim.experiments.common import generic_prior, style_mixture, videos_for
    from sim.personas import make_persona
    from sim.session import run_user

    prior = generic_prior(("snappy",))
    learner = Learner(algo(prior, seed=seed), style_mixture(("snappy",)) if mixture else None)
    run = run_user(make_persona("snappy"), learner, videos_for(seed, n_videos), seed, keep_log=keep_log)
    return learner, run, prior


def _same_state(a: Policy, b: Policy):
    # scoring an arm creates its model lazily (prior only, no data), so only models that hold data must match
    assert a.n_updates == b.n_updates
    assert {k for k, m in a.models.items() if m.n} == {k for k, m in b.models.items() if m.n}
    for k, m in a.models.items():
        if not m.n:
            continue
        assert np.array_equal(m.A_d, b.models[k].A_d) and np.array_equal(m.b_d, b.models[k].b_d) and m.n == b.models[k].n
        assert np.array_equal(m.theta, b.models[k].theta)


@pytest.mark.parametrize("mixture", [False, True])
def test_replay_equals_online(mixture):
    from sim.experiments.common import generic_prior, style_mixture

    online, run, prior = _small_run(mixture=mixture)
    assert len(run.events) > 100 and any(e.kind == "implicit" for e in run.events)
    assert any(e.payload.get("via") == "manual" for e in run.events)  # manual-edit path is exercised
    sugs = {s.id: s for s in run.suggestions}
    again = recompute_from_log(
        lambda: Learner(LinTS(prior, seed=0), style_mixture(("snappy",)) if mixture else None), sugs, run.events)
    _same_state(online.policy, again.policy)
    if mixture:
        assert np.array_equal(online.mixture.logw, again.mixture.logw) and online.mixture.n_events == again.mixture.n_events
    assert generic_prior(("snappy",)) is prior  # cached object: the test did not mutate the prior


def test_undone_events_are_skipped_on_replay():
    online, run, prior = _small_run()
    sugs = {s.id: s for s in run.suggestions}
    ev = list(run.events)
    for e in ev[10:60]:
        e.undone = True  # undo = flip the flag and recompute
    full = recompute_from_log(lambda: Learner(LinTS(prior, seed=0)), sugs, ev)
    kept = [e for e in ev if not e.undone]
    for e in ev:
        e.undone = False
    subset = recompute_from_log(lambda: Learner(LinTS(prior, seed=0)), sugs, kept)
    _same_state(full.policy, subset.policy)
    assert full.policy.n_updates < online.policy.n_updates
    for e in ev[10:60]:  # redo
        e.undone = False
    redo = recompute_from_log(lambda: Learner(LinTS(prior, seed=0)), sugs, ev)
    _same_state(redo.policy, online.policy)


def test_replay_unknown_suggestion_raises():
    _, run, prior = _small_run(n_videos=1)
    with pytest.raises(KeyError):
        replay(Learner(LinTS(prior)), {}, run.events[:3])


# ---------------------------------------------------------------- store
def test_store_roundtrip_and_staleness(tmp_path):
    online, run, prior = _small_run(n_videos=1)
    store = PolicyStore(tmp_path)
    store.save(7, online.policy, n_events=len(run.events))
    fresh = LinTS(prior, seed=0)
    assert store.load_into(7, fresh, n_events=len(run.events))
    _same_state(online.policy, fresh)
    assert not store.load_into(7, LinTS(prior), n_events=len(run.events) + 1)  # stale cache -> recompute
    assert not store.load_into(8, LinTS(prior))  # missing
    assert store.meta(7, "linTS")["n_events"] == len(run.events)


# ---------------------------------------------------------------- prior / style mixture
def test_fit_prior_separates_labelled_arms():
    rng = np.random.default_rng(10)
    data = []
    for _ in range(400):
        ctx = rand_ctx("pause", rng)
        data.append(("pause", ctx, "REMOVE" if ctx["gap_seconds"] > 1.5 else "KEEP"))
    pr = fit_prior(data)
    ok = 0
    for _ in range(200):
        ctx = rand_ctx("pause", rng)
        want = "REMOVE" if ctx["gap_seconds"] > 1.5 else "KEEP"
        ok += max(context.ARMS["pause"], key=lambda a: pr.score("pause", ctx, a)) == want
    assert ok / 200 > 0.9
    rt = Prior.from_dict(pr.to_dict())
    assert all(np.allclose(rt.weights[k], v) for k, v in pr.weights.items())


def test_style_mixture_identifies_style_and_freezes():
    rng = np.random.default_rng(11)

    def labelled(thr):
        return [("pause", c, "REMOVE" if c["gap_seconds"] > thr else "KEEP") for c in (rand_ctx("pause", rng) for _ in range(300))]

    snappy, relaxed = fit_prior(labelled(0.8)), fit_prior(labelled(2.2))
    mix = StyleMixture({"snappy": snappy, "relaxed": relaxed}, Prior(), freeze_after=120)
    assert mix.weights()["snappy"] == pytest.approx(1 / 3)
    for _ in range(100):  # a user who behaves like "snappy"
        c = rand_ctx("pause", rng)
        mix.observe("pause", c, [("REMOVE" if c["gap_seconds"] > 0.8 else "KEEP", 1.0)])
    w = mix.weights()
    assert w["snappy"] > 0.8 and w["relaxed"] < w["snappy"] and sum(w.values()) == pytest.approx(1.0)
    assert not mix.frozen
    for _ in range(40):
        mix.observe("pause", rand_ctx("pause", rng), [("KEEP", -1.0)])
    assert mix.frozen
    before = mix.weights().copy()
    mix.observe("pause", rand_ctx("pause", rng), [("KEEP", 1.0)])  # frozen: no change
    assert mix.weights() == before and mix.n_events == 120
    assert isinstance(mix.mean_prior(), Prior)


# ---------------------------------------------------------------- explanations
def test_explanations():
    rng = np.random.default_rng(12)
    dt = "pause"
    pol = LinTS(gamma=1.0, seed=0)
    for _ in range(600):  # the user removes long pauses and keeps short ones
        ctx = rand_ctx(dt, rng)
        long = ctx["gap_seconds"] > 1.5
        pol.update(dt, ctx, "REMOVE", 1.0 if long else -1.0)
        pol.update(dt, ctx, "KEEP", -1.0 if long else 1.0)
    ctx = rand_ctx(dt, rng)
    ctx["gap_seconds"] = 2.6
    ex = explain(pol, dt, ctx, "REMOVE", confidence=0.82)
    assert "remove" in ex.text and "82% sure" in ex.text and "pause is 2.6 s long" in ex.text
    assert len(ex.reasons) <= 3 and all(r[1] > 0 for r in ex.reasons)
    rule = explain(RulePolicy(), dt, ctx, "REMOVE")
    assert "fixed rule" in rule.text
    # D5: "You tend to highlight numbers."
    emph = LinTS(gamma=1.0, seed=0)
    for _ in range(300):
        c = rand_emph_ctx(rng)
        for k in range(int(c["n_cand"])):
            c[f"c{k}_is_number"] = float(rng.random() < 0.4)
        for a in context.arms_for("emphasis", c):
            r = 1.0 if (a != "NONE" and c[f"c{a[-1]}_is_number"] == 1.0) else -1.0 if a != "NONE" else 0.0
            emph.update("emphasis", c, a, r)
    c = rand_emph_ctx(rng)
    c["c0_is_number"] = 1.0
    assert "numbers" in explain(emph, "emphasis", c, "CAND_0").text


# ---------------------------------------------------------------- regressions
def test_learner_with_a_style_mixture_and_a_policy_without_a_prior():
    from sim.experiments.common import style_mixture

    for pol in (RulePolicy(), GlobalPolicy(Prior())):
        lr = Learner(pol, style_mixture(()))  # used to raise AttributeError (no set_prior)
        lr.apply("pause", {"gap_seconds": 1.0, "bias": 1.0}, "KEEP", "override", "REMOVE")
        assert lr.mixture is None


def test_negative_update_weight_is_rejected():
    with pytest.raises(ValueError):
        LinTS(seed=0).update("pause", {"bias": 1.0}, "KEEP", 1.0, weight=-1.0)
    with pytest.raises(ValueError):
        LinTS(seed=0).update("pause", {"bias": 1.0}, "KEEP", 1.0, weight=float("nan"))


def test_store_restores_the_prior_the_statistics_were_built_with(tmp_path):
    from sim.experiments.common import generic_prior, style_mixture

    online, run, prior = _small_run(n_videos=1, mixture=True)
    store = PolicyStore(tmp_path)
    store.save(1, online.policy)
    fresh = LinTS(Prior(), seed=0)  # a different (empty) prior than the one used online
    assert store.load_into(1, fresh)
    _same_state(online.policy, fresh)  # theta_hat identical: the saved prior was restored too
    ctx = run.suggestions[0].context_raw
    assert fresh.mean_scores("phrase", ctx) == pytest.approx(online.policy.mean_scores("phrase", ctx))
    assert np.array_equal(fresh.theta("phrase", "KEEP"), online.policy.theta("phrase", "KEEP"))
    # the style mixture is cached separately
    mix = style_mixture(("snappy",))
    store.save_mixture(1, online.mixture)
    assert store.load_mixture(1, mix) and np.array_equal(mix.logw, online.mixture.logw)
    assert mix.n_events == online.mixture.n_events and not store.load_mixture(2, mix)
    assert generic_prior(("snappy",)) is prior


def test_style_mixture_counts_user_events_not_updates():
    from sim.experiments.common import style_mixture

    lr = Learner(LinTS(seed=0), style_mixture(()))
    lr.apply("phrase", {"bias": 1.0}, "CUT", "reject")  # binary reject = TWO updates (proposed -1, other +1)
    assert lr.policy.n_updates == 2 and lr.mixture.n_events == 1
    lr.apply("pause", {"gap_seconds": 1.0}, "KEEP", "crop")  # not a learning event
    assert lr.mixture.n_events == 1


def test_safe_exploration_never_flips_a_confident_keep_at_random():
    from app.policy.prior import Prior

    ctx = {"bias": 1.0, "duration": 2.0}
    th = {("phrase", "KEEP"): np.zeros(context.dim("phrase")), ("phrase", "CUT"): np.zeros(context.dim("phrase"))}
    i = context.feature_names("phrase").index("bias")
    th[("phrase", "KEEP")][i], th[("phrase", "CUT")][i] = 5.0, -5.0
    safe = LinTS(Prior(th), lam=1000.0, seed=1, eps_by_type={"phrase": 0.0})
    assert all(safe.choose("phrase", ctx).action == "KEEP" for _ in range(600))
    plain = LinTS(Prior(th), lam=1000.0, seed=1)
    assert any(plain.choose("phrase", ctx).action == "CUT" for _ in range(600))  # the default floor explores
