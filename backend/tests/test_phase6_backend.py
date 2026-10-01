"""Phase 6: style prior (policy/style.py), what-it-learned sentences, profile statistics, experiments E2 and E8."""
from __future__ import annotations

import numpy as np
import pytest

from app.policy.linear import LinTS
from app.policy.preferences import MIN_EVIDENCE, describe_preferences
from app.policy.prior import Prior
from app.policy.style import IDENTIFIED_ABOVE, POOLED, StyleMixture, style_label, summarize
from app.svc.stats import VideoStats, aggregate
from sim.experiments import e2, e8
from sim.experiments.common import generic_prior, make_learner, style_mixture, videos_for
from sim.personas import make_persona
from sim.review import STRATEGIES, pick, run_budgeted
from sim.session import run_user
from tests.apihelpers import new_ready_project, post_event
from tests.test_policy import rand_ctx


# --------------------------------------------------------------------------- style mixture and its sentence
def two_styles():
    rng = np.random.default_rng(5)
    from app.policy.prior import fit_prior

    def labelled(thr):
        return [("pause", c, "REMOVE" if c["gap_seconds"] > thr else "KEEP") for c in (rand_ctx("pause", rng) for _ in range(300))]

    return fit_prior(labelled(0.8)), fit_prior(labelled(2.2)), rng


def test_the_posterior_is_sharp_while_the_prior_follows_it_softly():
    snappy, relaxed, rng = two_styles()
    sharp = StyleMixture({"snappy": snappy, "relaxed": relaxed}, Prior(), sigma=0.7)
    soft = StyleMixture({"snappy": snappy, "relaxed": relaxed}, Prior(), sigma=0.7, prior_sigma=3.0)
    for _ in range(12):
        c = rand_ctx("pause", rng)
        pairs = [("REMOVE" if c["gap_seconds"] > 0.8 else "KEEP", 1.0)]
        sharp.observe("pause", c, pairs)
        soft.observe("pause", c, pairs)
    assert soft.weights() == sharp.weights()  # the displayed posterior does not depend on the tempering
    assert np.array_equal(soft.logw, sharp.logw)
    assert soft.weights()["snappy"] > 0.8
    tempered = soft.prior_weights()
    assert tempered.max() < max(soft.weights().values()) and tempered.sum() == pytest.approx(1.0)
    assert np.array_equal(sharp.prior_weights(), np.array(list(sharp.weights().values())))  # no tempering when unset
    # the prior mean is built from the tempered weights
    want = Prior.mix(soft.priors, tempered)
    c = rand_ctx("pause", rng)
    assert soft.mean_prior().score("pause", c, "REMOVE") == pytest.approx(want.score("pause", c, "REMOVE"))


def test_a_prior_sigma_below_sigma_is_refused():
    with pytest.raises(ValueError):
        StyleMixture({}, Prior(), sigma=1.0, prior_sigma=0.5)
    with pytest.raises(ValueError):
        StyleMixture({}, Prior(), sigma=0.0)


def test_summary_sentence_and_identification():
    s = summarize({POOLED: 0.05, "snappy": 0.7, "storyteller": 0.22, "relaxed": 0.03}, n_events=12)
    assert s.text == "Your style so far: 70% Snappy · 22% Storyteller"
    assert s.top == "snappy" and s.identified is False  # 0.7 is not above 0.8
    assert [n for n, _ in s.shares] == ["snappy", "storyteller"]
    sure = summarize({POOLED: 0.02, "snappy": 0.93, "relaxed": 0.05}, 30)
    assert sure.text == "Your style so far: 93% Snappy" and sure.identified is True and sure.top == "snappy"
    assert summarize({POOLED: 0.1, "snappy": 0.8}, 5).identified is False  # exactly 0.8 is not "above"
    empty = summarize({POOLED: 1.0, "snappy": 0.0}, 0)
    assert empty.text.startswith("No style yet") and empty.top is None and not empty.identified
    unclear = summarize({POOLED: 0.8, "snappy": 0.1, "relaxed": 0.1}, 4)  # the generic prior still leads
    assert unclear.top is None and "not clear yet" in unclear.text and not unclear.identified


def test_style_names_are_readable():
    assert style_label("storyteller") == "Storyteller" and style_label(POOLED) == "Generic"
    assert style_label("deadpan_comedian") == "Deadpan Comedian"


def identified_after(model, seeds=(1, 2, 3, 4, 5, 6)):
    out = []
    for seed in seeds:
        r = e2.identification_task({"persona": "snappy", "seed": seed, "videos": 3, "model": model})
        out.append(e2.decisions_to_identify(r["curve"]))
    return out


def test_snappy_is_identified_within_about_thirty_decisions_in_the_product():
    """PLAN Phase 6 acceptance, with the settings the app runs: more than 0.8 on Snappy within ~30 feedback events."""
    n = identified_after("product")
    assert None not in n and max(n) <= 30, n


def test_snappy_is_identified_within_about_thirty_decisions_in_the_plain_model():
    """The PLAN model (no tempering, lambda 2) is a little less steady: typically fast, one seed in six is slow."""
    n = identified_after("plain")
    assert None not in n and float(np.median(n)) <= 30 and sum(x <= 30 for x in n) >= 5, n


def test_decisions_to_identify_counts_the_first_crossing_or_the_last_one():
    assert e2.decisions_to_identify([0.2, 0.9, 0.95, 0.99]) == 2
    dipping = [0.2, 0.9, 0.5, 0.9, 0.95]
    assert e2.decisions_to_identify(dipping) == 2 and e2.decisions_to_identify(dipping, stable=True) == 4
    assert e2.decisions_to_identify([0.2, 0.3, 0.4]) is None and e2.decisions_to_identify([0.2, 0.3], stable=True) is None
    assert e2.decisions_to_identify([0.9, 0.5], stable=True) is None  # ends below: never identified for good
    assert e2.decisions_to_identify([]) is None
    assert e2.decisions_to_identify([0.9, 0.9]) == 1 and e2.decisions_to_identify([0.9, 0.9], stable=True) == 1
    assert IDENTIFIED_ABOVE == 0.8


# --------------------------------------------------------------------------- sentences read off the learned policy
def trained(name, videos=4, seed=1):
    mix = style_mixture(())
    learner = make_learner("lints", None, seed, 0.995, mixture=mix, lam=10.0)
    run = run_user(make_persona(name), learner, videos_for(seed, videos), seed)
    evidence: dict[str, int] = {}
    for v in run.videos:
        for dt, n in v.n.items():
            evidence[dt] = evidence.get(dt, 0) + n
    return learner.policy, evidence


def test_the_sentences_tell_snappy_from_relaxed():
    snappy, ev_s = trained("snappy")
    relaxed, ev_r = trained("relaxed")
    s = {p.decision_type: p.text for p in describe_preferences(snappy, ev_s)}
    r = {p.decision_type: p.text for p in describe_preferences(relaxed, ev_r)}
    assert s["pause"].startswith("You remove pauses longer than ~") and "tighten the shorter ones" in s["pause"]
    assert r["pause"].startswith("You tighten pauses longer than ~")  # relaxed rarely removes a pause
    assert "every filler word" in s["filler"] and "but keep" in r["filler"]
    assert "zoom" in s["zoom"] and r["zoom"] == "You rarely zoom."
    assert all(t.endswith(".") and t[0] == "Y" for t in [*s.values(), *r.values()])
    assert float(s["pause"].split("~")[1].split(" s")[0]) <= 0.8  # a Snappy creator removes pauses from well under 1 s


def test_nothing_is_said_without_evidence():
    policy, evidence = trained("snappy", videos=2)
    assert describe_preferences(policy, {}) == []
    only = describe_preferences(policy, {"pause": MIN_EVIDENCE})
    assert [p.decision_type for p in only] == ["pause"]
    assert describe_preferences(policy, {"pause": MIN_EVIDENCE - 1, "zoom": 0}) == []
    assert evidence["pause"] > MIN_EVIDENCE


def test_an_untrained_policy_still_gives_sentences_and_a_policy_without_weights_none_for_emphasis():
    fresh = LinTS(Prior(), seed=0)
    assert describe_preferences(fresh, {"emphasis": 100}) == []  # all-zero weights: no highlight preference
    texts = [p.text for p in describe_preferences(fresh, {"pause": 100, "zoom": 100})]
    assert "You mostly keep pauses as they are." in texts and "You rarely zoom." in texts


# --------------------------------------------------------------------------- statistics
def test_aggregate_statistics():
    a = VideoStats(out_seconds=60, src_seconds=100, kept_seconds=60, n_segments=6, n_cuts=5, n_zooms=3, n_fillers_removed=4)
    b = VideoStats(out_seconds=30, src_seconds=50, kept_seconds=30, n_segments=2, n_cuts=1, n_zooms=0, n_fillers_removed=2)
    s = aggregate([a, b])
    assert s["n_videos"] == 2 and s["avg_clip_seconds"] == pytest.approx(90 / 8)
    assert s["cuts_per_minute"] == pytest.approx(6 / 1.5) and s["zooms_per_minute"] == pytest.approx(2.0)
    assert s["percent_kept"] == pytest.approx(60.0) and s["fillers_removed_per_minute"] == pytest.approx(4.0)
    assert aggregate([]) is None


# --------------------------------------------------------------------------- the profile API
def test_profile_carries_style_sentences_and_statistics(client):
    uid, pk, _ = new_ready_project(client)
    empty = client.get(f"/api/users/{uid}/profile").json()
    assert empty["stats"] is None and empty["preferences"] == [] and empty["style_mixture"]["text"].startswith("No style yet")
    assert empty["style_labels"]["snappy"]["label"] == "Snappy" and empty["style_labels"]["pooled"]["label"] == "Generic"

    s = client.post(f"/api/projects/{pk}/suggest").json()
    for sug in [x for x in s["suggestions"] if x["decision_type"] in ("pause", "phrase", "zoom")][:14]:
        post_event(client, pk, "accept", suggestion_id=sug["id"])
    client.post(f"/api/projects/{pk}/seen", json={"unit_ids": [u["id"] for u in client.get(f"/api/projects/{pk}").json()["units"]]})
    assert client.post(f"/api/projects/{pk}/finalize", json={"seconds_active": 30}).status_code == 200

    prof = client.get(f"/api/users/{uid}/profile").json()
    st = prof["stats"]
    assert st["n_videos"] == 1 and st["avg_clip_seconds"] > 0 and 0 < st["percent_kept"] <= 100
    assert st["cuts_per_minute"] >= 0 and st["zooms_per_minute"] >= 0 and st["fillers_removed_per_minute"] >= 0
    assert prof["style_mixture"]["text"].startswith("Your style so far") and prof["style_mixture"]["n_events"] > 0
    assert set(prof["n_decisions_by_type"]) <= {"phrase", "pause", "filler", "zoom", "emphasis"}
    assert sum(prof["n_decisions_by_type"].values()) == prof["n_learning_events"]  # every learned event has a type
    assert all(p["text"].startswith("You ") for p in prof["preferences"])
    assert {p["decision_type"] for p in prof["preferences"]} <= set(prof["n_decisions_by_type"])


def test_project_state_carries_the_style_text(client):
    _, pk, _ = new_ready_project(client)
    s = client.post(f"/api/projects/{pk}/suggest").json()
    assert s["style_mixture"]["text"].startswith("No style yet") and s["style_mixture"]["identified"] is False
    sug = next(x for x in s["suggestions"] if x["decision_type"] == "pause")
    out = post_event(client, pk, "accept", suggestion_id=sug["id"])
    assert out["style_mixture"]["text"].startswith("Your style so far") and out["style_mixture"]["top"] is not None


# --------------------------------------------------------------------------- budgeted review (E8)
def test_review_strategies_pick_what_they_say():
    conf = [0.9, 0.2, 0.6, 0.2, 0.95, 0.4]
    rng = np.random.default_rng(0)
    assert pick("uncertainty", conf, 3, rng) == [1, 3, 5]  # the three least confident (ties: earlier first)
    assert pick("timeline", conf, 2, rng) == [0, 1]
    r = pick("random", conf, 4, np.random.default_rng(1))
    assert len(r) == 4 and len(set(r)) == 4 and r == sorted(r) and r == pick("random", conf, 4, np.random.default_rng(1))
    for s in STRATEGIES:
        assert pick(s, conf, 0, rng) == [] and len(pick(s, conf, 99, rng)) == len(conf)
    assert pick("uncertainty", [], 3, rng) == []
    with pytest.raises(ValueError):
        pick("psychic", conf, 1, rng)


def budget_run(budget, strategy, seed=2, persona="snappy", n=3):
    learner = make_learner("lints", generic_prior((persona,)), seed, 0.995)
    return run_budgeted(make_persona(persona), learner, videos_for(seed, n), seed, budget, strategy)


def test_a_budgeted_run_keeps_its_books():
    full = budget_run(1.0, "uncertainty")
    none = budget_run(0.0, "uncertainty")
    half = budget_run(0.5, "random")
    for v in [*full, *none, *half]:
        assert 0 <= v.wrong_fixed <= v.wrong and v.correct + v.wrong == v.n
        assert v.final_correct == v.correct + v.wrong_fixed and v.final_correct <= v.n
    assert all(v.reviewed == v.n for v in full) and all(v.reviewed == 0 for v in none)
    assert all(v.wrong_fixed == 0 and v.final_correct == v.correct for v in none)  # nobody looked: nothing was fixed
    assert all(v.reviewed == round(0.5 * v.n) for v in half)
    assert sum(v.wrong_fixed for v in full) > 0
    assert budget_run(0.5, "random") == half  # deterministic for a seed


def test_looking_at_the_uncertain_ones_finds_more_errors_than_a_random_look():
    """The point of the queue: for the same patience it catches more of the wrong suggestions."""
    caught = {"uncertainty": 0, "random": 0}
    wrong = {"uncertainty": 0, "random": 0}
    for seed in (1, 2, 3):
        for strat in caught:
            for v in budget_run(0.2, strat, seed=seed, persona="relaxed", n=2):
                caught[strat] += v.wrong_fixed
                wrong[strat] += v.wrong
    assert caught["uncertainty"] / wrong["uncertainty"] > caught["random"] / wrong["random"]


# --------------------------------------------------------------------------- experiments produce their plots
def small(monkeypatch, module):
    monkeypatch.setattr(module, "BASE_PERSONAS", ("snappy", "relaxed"))


def test_e2_runs_and_writes_its_figures(tmp_path, monkeypatch):
    small(monkeypatch, e2)
    res = e2.run(seeds=1, n_videos=3, id_videos=1, workers=1)
    t = e2.report(res, tmp_path)
    for f in ("e2_cold_start.png", "e2_identification.png", "e2_summary.json"):
        assert (tmp_path / f).stat().st_size > 1000
    assert set(t["cold_start"]) == set(e2.PRIORS) and all(0 < v["early"] <= 1 for v in t["cold_start"].values())
    assert set(t["identification"]) == {"plain", "product"} and t["threshold"] == 0.8
    assert set(t["identification"]["plain"]["by_persona"]) == {"snappy", "relaxed"}


def test_e8_runs_and_writes_its_figure(tmp_path, monkeypatch):
    small(monkeypatch, e8)
    res = e8.run(seeds=1, n_videos=3, workers=1, budgets=(0.2, 1.0))
    s = e8.report(res, tmp_path, budgets=(0.2, 1.0))
    assert (tmp_path / "e8_review_budget.png").stat().st_size > 1000 and (tmp_path / "e8_summary.json").exists()
    assert set(s["table"]) == set(STRATEGIES) and len(s["table"]["random"]["posted"]["mean"]) == 2
    # reviewing everything makes the three strategies the same review (same set of decisions)
    full = [s["table"][st]["posted"]["mean"][1] for st in STRATEGIES]
    assert max(full) - min(full) < 0.1
