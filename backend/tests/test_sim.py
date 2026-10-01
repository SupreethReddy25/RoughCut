"""Simulator: synthetic videos, personas, simulated sessions, experiment plumbing (E1/E3/E4)."""
from __future__ import annotations

import numpy as np
import pytest

from app.policy.linear import LinTS
from app.policy.log_replay import Learner
from app.policy.rule import RulePolicy
from app.schemas import UnitsFile
from app.session import EditSession
from sim.experiments import e1, e3, e4
from sim.experiments.common import ALGOS, acc_matrix, ci95, generic_prior, run_learning_task, videos_for
from sim.personas import BASE_PERSONAS, Drift, Minimalist, TypeConditional, make_persona
from sim.session import GreedyView, evaluate, run_user
from sim.synth import make_video


# ---------------------------------------------------------------- synthetic videos
def test_synth_video_is_a_valid_units_file_with_the_real_features():
    v = make_video(3, "vlog")
    UnitsFile.model_validate(v.uf.model_dump())
    kinds = {u.kind for u in v.uf.units}
    assert kinds == {"speech", "pause", "filler"} and 20 < sum(u.kind == "speech" for u in v.uf.units) < 60
    assert [u.id for u in v.uf.units] == list(range(len(v.uf.units)))
    sp = [u for u in v.uf.units if u.kind == "speech"]
    for key in ("retake_score", "is_earlier_take", "topic_centrality", "loudness_z", "face_present", "filler_ratio",
                "semantic_redundancy"):
        assert key in sp[0].features
    assert any(u.features["is_earlier_take"] == 1.0 for u in sp)  # retakes are planted
    assert all(np.isfinite(x) for u in v.uf.units for x in u.features.values())
    assert v.uf.video_type == "vlog" and v.minutes == pytest.approx(v.uf.duration / 60)


def test_synth_is_deterministic_and_varied():
    a, b, c = make_video(5, "vlog"), make_video(5, "vlog"), make_video(6, "vlog")
    assert a.uf.model_dump() == b.uf.model_dump() and a.uf.model_dump() != c.uf.model_dump()
    vs = videos_for(1, 6)
    assert {v.video_type for v in vs} == {"vlog", "tutorial", "opinion"}
    takes = sum(any(u.features.get("is_take_boundary") == 1.0 for u in make_video(s).uf.units if u.kind == "pause")
                for s in range(40))
    assert 3 < takes < 25  # some videos are joined from several takes


def test_synth_decisions_flow_through_a_session():
    v = make_video(8, "tutorial")
    sess = EditSession(v.uf, v.word_feats, v.video_type)
    n = {}
    for item in sess.items():
        n[item.decision_type] = n.get(item.decision_type, 0) + 1
        sess.commit(item, item.arms[0])
    assert all(n[k] > 5 for k in ("phrase", "pause", "zoom", "emphasis")) and 90 < sum(n.values()) < 250


# ---------------------------------------------------------------- personas
def _items(video):
    sess = EditSession(video.uf, video.word_feats, video.video_type)
    for item in sess.items():
        yield item
        sess.commit(item, item.arms[0])


def test_persona_rules_on_known_contexts():
    from app.session import Item

    def item(dt, ctx, **kw):
        return Item(dt, kw.get("target", 0), ctx, [], None)

    sn, rl, st, fo = (make_persona(n) for n in ("snappy", "relaxed", "storyteller", "focused"))
    assert sn.ideal_action(item("pause", {"gap_seconds": 0.6})) == "REMOVE"
    assert sn.ideal_action(item("pause", {"gap_seconds": 0.4})) == "TIGHT_15"
    assert rl.ideal_action(item("pause", {"gap_seconds": 0.79})) == "KEEP" and rl.ideal_action(item("pause", {"gap_seconds": 0.8})) == "TIGHT_40"
    assert st.ideal_action(item("pause", {"after_sentence_end": 1.0})) == "TIGHT_40"
    assert st.ideal_action(item("pause", {"after_sentence_end": 0.0})) == "REMOVE"
    for p in (sn, rl, st, fo):  # every persona removes take boundaries
        assert p.ideal_action(item("pause", {"gap_seconds": 0.1, "is_take_boundary": 1.0})) == "REMOVE"
    assert sn.ideal_action(item("filler", {"tok_like": 1.0})) == "REMOVE" and rl.ideal_action(item("filler", {"tok_like": 1.0})) == "KEEP"
    assert st.ideal_action(item("filler", {"tok_you_know": 1.0})) == "REMOVE" and fo.ideal_action(item("filler", {"tok_like": 1.0})) == "REMOVE"
    assert sn.ideal_action(item("phrase", {"is_earlier_take": 1.0})) == "CUT" and sn.ideal_action(item("phrase", {"avg_word_conf": 0.4})) == "CUT"
    assert fo.ideal_action(item("phrase", {"topic_centrality": 0.2})) == "CUT" and rl.ideal_action(item("phrase", {"topic_centrality": 0.2})) == "KEEP"
    assert sn.ideal_action(item("zoom", {"loudness_z": 1.0, "hist_since_last_zoom": 5.0})) == "Z115"
    assert sn.ideal_action(item("zoom", {"loudness_z": 1.0, "hist_since_last_zoom": 3.0})) == "NONE"  # needs >= 4 s apart
    assert st.ideal_action(item("zoom", {"is_first": 1.0})) == "Z130" and rl.ideal_action(item("zoom", {"is_first": 1.0})) == "NONE"
    cand = {"n_cand": 3.0, "c0_is_number": 0.0, "c1_is_number": 1.0, "c2_is_number": 0.0, "c0_word_loudness_z": 2.0,
            "c1_word_loudness_z": 0.0, "c2_word_loudness_z": 0.5, "c0_tfidf_in_transcript": 0.2, "c1_tfidf_in_transcript": 0.1,
            "c2_tfidf_in_transcript": 0.9, "c0_is_capitalized_mid_sentence": 0.0, "c1_is_capitalized_mid_sentence": 0.0,
            "c2_is_capitalized_mid_sentence": 1.0}
    assert sn.ideal_action(item("emphasis", cand)) == "CAND_1"  # numbers first
    cand["c1_is_number"] = 0.0
    assert sn.ideal_action(item("emphasis", cand)) == "CAND_0"  # else the loudest
    assert st.ideal_action(item("emphasis", cand)) == "CAND_2" and fo.ideal_action(item("emphasis", cand)) == "CAND_2"
    assert rl.ideal_action(item("emphasis", cand)) == "NONE"


def test_blend_mixes_snappy_and_storyteller_per_decision_type():
    b, sn, st = make_persona("blend"), make_persona("snappy"), make_persona("storyteller")
    for item in _items(make_video(2, "vlog")):
        want = {"pause": sn, "filler": sn, "emphasis": sn, "phrase": st, "zoom": st}[item.decision_type]
        assert b.ideal_action(item) == want.ideal_action(item)


def test_minimalist_keeps_the_most_central_phrases_up_to_the_target():
    v = make_video(4, "opinion")
    m = Minimalist()
    m.prepare(v)
    speech = {u.id: u for u in v.uf.units if u.kind == "speech"}
    assert m._keep and m._keep <= set(speech)
    kept = sum(speech[i].end - speech[i].start for i in m._keep)
    assert 30 <= kept < 30 + max(u.end - u.start for u in speech.values())
    worst_kept = min(speech[i].features["topic_centrality"] for i in m._keep)
    assert all(u.features["topic_centrality"] <= worst_kept + 1e-9 or u.id in m._keep or u.features["is_earlier_take"] >= 0.5
               for u in speech.values())


def test_variants_switch_behaviour():
    d = Drift(at=6)
    assert d.for_video(5, "vlog").name == "relaxed" and d.for_video(6, "vlog").name == "snappy"
    t = TypeConditional()
    assert t.for_video(1, "vlog").name == "snappy" and t.for_video(1, "tutorial").name == "relaxed"
    assert t.for_video(1, "opinion").name == "relaxed"


def test_prefers_uses_clip_length():
    sn = make_persona("snappy")
    assert sn.prefers({"avg_clip_len": 2.4}, {"avg_clip_len": 6.0}) and not sn.prefers({"avg_clip_len": 6.0}, {"avg_clip_len": 2.4})


# ---------------------------------------------------------------- simulated sessions
def test_run_user_bookkeeping():
    prior = generic_prior(("snappy",))
    lr = Learner(LinTS(prior, seed=1))
    run = run_user(make_persona("snappy"), lr, videos_for(1, 3), 1, keep_log=True)
    assert len(run.videos) == 3
    for r in run.videos:
        assert 0.0 <= r.acceptance <= 1.0 and sum(r.n.values()) > 90
        assert r.accepts + r.corrections + r.implicits == sum(r.n.values())  # review_budget 1: every decision reviewed
        assert 0 <= r.manual_edits <= r.corrections
    assert len(run.suggestions) == sum(sum(r.n.values()) for r in run.videos)
    assert len(run.events) == sum(r.accepts + r.corrections + r.implicits for r in run.videos)
    assert [s.id for s in run.suggestions] == list(range(len(run.suggestions)))
    corrected = [e for e in run.events if e.kind == "override"]
    assert corrected and all("final_action" in e.payload for e in corrected)


def test_review_budget_limits_feedback():
    p = make_persona("snappy", review_budget=0.2)
    run = run_user(p, Learner(RulePolicy()), videos_for(1, 2), 1)
    for r in run.videos:
        reviewed = r.accepts + r.corrections + r.implicits
        assert 0.1 * sum(r.n.values()) < reviewed < 0.3 * sum(r.n.values())


def test_noise_free_always_overriding_persona_only_accepts_matches():
    p = make_persona("snappy", noise=0.0, p_override=1.0)
    run = run_user(p, Learner(RulePolicy()), videos_for(2, 2), 2)
    for r in run.videos:
        assert r.accepts == sum(r.correct.values()) and r.implicits == 0 and r.accepts + r.corrections == sum(r.n.values())


def test_rule_run_is_deterministic_and_matches_hand_count():
    a = run_user(make_persona("relaxed"), Learner(RulePolicy()), videos_for(3, 2), 3)
    b = run_user(make_persona("relaxed"), Learner(RulePolicy()), videos_for(3, 2), 3)
    assert [r.correct for r in a.videos] == [r.correct for r in b.videos]
    # Relaxed never zooms and neither does the Rule: those proposals match exactly
    assert all(r.acceptance_of("zoom") == 1.0 for r in a.videos)


def test_learning_improves_over_the_first_videos():
    accs = []
    for seed in range(1, 4):
        run = run_user(make_persona("focused"), Learner(LinTS(generic_prior(("focused",)), seed=seed)), videos_for(seed, 4), seed)
        accs.append([r.acceptance for r in run.videos])
    m = np.mean(accs, axis=0)
    assert m[-1] > m[0] > 0.6


def test_greedy_evaluation():
    learner = Learner(LinTS(generic_prior(), seed=0))
    run_user(make_persona("snappy"), learner, videos_for(1, 4), 1)
    view = GreedyView(learner.policy)
    res = evaluate(view.choose, make_persona("snappy"), videos_for(1, 2, offset=500))
    assert 0.5 < res["overall"] <= 1.0 and set(res) >= {"overall", "phrase", "pause"}
    assert view.choose("pause", {"gap_seconds": 1.0, "bias": 1.0}) in ("KEEP", "TIGHT_40", "TIGHT_15", "REMOVE")


# ---------------------------------------------------------------- experiment plumbing
def test_run_learning_task_and_ci():
    r = run_learning_task({"persona": "snappy", "algo": "lints", "seed": 1, "n_videos": 2})
    assert len(r["videos"]) == 2 and set(r["videos"][0]["n"]) == {"phrase", "pause", "filler", "zoom", "emphasis"}
    m = acc_matrix([r])
    assert m.shape == (1, 2) and (m <= 1).all()
    mean, half = ci95(np.array([[0.8, 0.9], [0.9, 0.7], [0.85, 0.8]]))
    assert mean == pytest.approx([0.85, 0.8]) and (half > 0).all()
    for algo in ALGOS:
        assert run_learning_task({"persona": "relaxed", "algo": algo, "seed": 2, "n_videos": 1})["algo"] == algo
    for mode in ("none", "mixture"):
        assert run_learning_task({"persona": "relaxed", "algo": "lints", "seed": 2, "n_videos": 1, "prior": mode})


def test_e1_check_logic():
    def video(n, c):
        return {"n": dict.fromkeys(("phrase", "pause", "filler", "zoom", "emphasis"), n),
                "correct": {k: int(n * v) for k, v in c.items()}, "corrections": 1, "minutes": 1.0}

    def runs(persona, algo, accs):  # accs = per-video per-type accuracy dict
        return [{"persona": persona, "algo": algo, "seed": 1, "videos": [video(100, a) for a in accs]}]

    full = lambda x: dict.fromkeys(("phrase", "pause", "filler", "zoom", "emphasis"), x)
    good = (runs("snappy", "lints", [full(0.9)] * 6) + runs("snappy", "rule", [full(0.7)] * 6)
            + runs("snappy", "global", [full(0.6)] * 6))
    v = e1.check(good)
    assert v["overall_ok"] and v["by_type_ok"] and not v["details"] and v["cells"]["ahead"] == 10
    # a baseline that is the persona's exact rule on one type: LinTS must converge (>= 0.90 by the last video)
    rule_ceiling = [{**full(0.7), "zoom": 1.0}] * 6
    lints_conv = [{**full(0.9), "zoom": 0.8}] * 5 + [{**full(0.9), "zoom": 0.93}]
    v = e1.check(runs("snappy", "lints", lints_conv) + runs("snappy", "rule", rule_ceiling) + runs("snappy", "global", [full(0.6)] * 6))
    assert v["by_type_ok"] and v["cells"]["ceiling"] == 1
    v = e1.check(runs("snappy", "lints", [{**full(0.9), "zoom": 0.8}] * 6) + runs("snappy", "rule", rule_ceiling)
                 + runs("snappy", "global", [full(0.6)] * 6))
    assert not v["by_type_ok"]
    # a non-ceiling cell where LinTS is behind fails
    v = e1.check(runs("snappy", "lints", [{**full(0.9), "pause": 0.5}] * 6) + runs("snappy", "rule", [full(0.7)] * 6)
                 + runs("snappy", "global", [full(0.6)] * 6))
    assert not v["by_type_ok"] and v["overall_ok"]


def test_e3_dominance_and_e4_recovery_helpers():
    M = np.eye(6) * 0.5 + 0.4
    assert e3.dominance(M)["column_dominant"] and e3.dominance(M)["row_dominant"]
    M[0, 0] = 0.3
    assert not e3.dominance(M)["column_dominant"] and not e3.dominance(M)["row_dominant"]
    curve = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.5, 0.6, 0.7, 0.82, 0.9])
    assert e4.recovery(curve, switch=6, thr=0.8) == 3 and e4.recovery(curve, switch=6, thr=0.99) is None
    assert e4.recovery(np.array([0.9] * 8), switch=6, thr=0.8) == 0


# ---------------------------------------------------------------- Phase 2 acceptance (slow): smaller E1 / E3
@pytest.mark.slow
def test_e1_lints_beats_rule_and_global_overall_by_video_5():
    res = e1.run(seeds=4, n_videos=6, workers=1, algos=("rule", "global", "lints"))
    table = e1.summary_table(res, at=5)
    for p in BASE_PERSONAS:
        row = table[p]
        assert row["lints"]["overall"] > row["rule"]["overall"] + 0.05, p
        assert row["lints"]["overall"] > row["global"]["overall"] + 0.05, p
        assert row["lints"]["overall"] > 0.8, p


@pytest.mark.slow
def test_e3_matrix_is_diagonal_dominant():
    res = e3.run(seeds=4, train_videos=10, test_videos=3, workers=1)
    M = e3.matrix(res)
    d = e3.dominance(M)
    assert d["column_dominant"] and d["row_dominant"], np.round(M, 2)
    assert np.mean(np.diag(M)) > np.mean(M[~np.eye(len(M), dtype=bool)]) + 0.25
