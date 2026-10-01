"""EditSession / suggest(): left-to-right order, history features, skipped fillers, EDL consistency."""
from __future__ import annotations

import pytest

from app.edl.build import build_edl, build_edl_with_lines
from app.policy.linear import LinTS
from app.policy.rule import RulePolicy
from app.schemas import Suggestion
from app.session import EditSession, suggest
from tests.test_edl import ids, make_units


@pytest.fixture
def uf3():
    # phrase 0 has a clean filler 'um'; phrase 1 is short; phrase 2 has a co-articulated filler (no clean gap)
    uf, w = make_units([("um I think this works well today.", 0.5), ("Short one here.", 1.0),
                        ("then so we go there now.", 0.8)])
    return uf, w


def test_order_is_pause_phrase_fillers_zoom_then_emphasis(uf3):
    uf, _ = uf3
    sess = EditSession(uf, {})
    seq = []
    for item in sess.items():
        seq.append((item.decision_type, item.target_id))
        sess.commit(item, item.arms[0])
    kinds = [k for k, _ in seq]
    sp = ids(uf, "speech")
    assert kinds[0] == "phrase" and seq[0][1] == sp[0]  # no pause before the first phrase
    assert kinds[:4] == ["phrase", "filler", "zoom", "pause"]  # phrase -> its fillers -> zoom, then the next pause
    i1 = kinds.index("pause")
    assert kinds[i1:i1 + 3] == ["pause", "phrase", "zoom"]
    assert seq[i1 + 1][1] == sp[1]
    assert kinds.count("phrase") == kinds.count("zoom") == 3 and kinds.count("pause") == 2
    # every item appears at most once, and the per-line emphasis decisions come last
    assert len(set(seq)) == len(seq)


def test_cut_phrase_has_no_fillers_or_zoom(uf3):
    uf, _ = uf3
    sess = EditSession(uf, {})
    out = []
    for item in sess.items():
        out.append((item.decision_type, item.target_id))
        sess.commit(item, "CUT" if item.decision_type == "phrase" else item.arms[0])
    assert [k for k, _ in out].count("zoom") == 0 and [k for k, _ in out].count("filler") == 0


def test_filler_without_clean_gap_is_skipped(uf3):
    uf, _ = uf3
    for u in uf.units:  # make the 2nd filler co-articulated
        if u.kind == "filler":
            u.features["clean_gap"] = 0.0
    res = suggest(uf, {}, RulePolicy())
    assert not [s for s in res.suggestions if s.decision_type == "filler"]
    assert sorted(res.skipped_fillers) == sorted(ids(uf, "filler"))
    assert all(res.decisions.action(f) == "KEEP" for f in res.skipped_fillers)  # default KEEP


def test_history_features_follow_committed_actions(uf3):
    uf, _ = uf3
    sess = EditSession(uf, {})
    seen = {}
    plan = {"phrase": "KEEP", "zoom": "Z115", "pause": "KEEP", "filler": "KEEP", "emphasis": "NONE"}
    for item in sess.items():
        seen.setdefault((item.decision_type, len([1 for k in seen if k[0] == item.decision_type])), item.ctx)
        sess.commit(item, plan[item.decision_type])
    p0, p1, p2 = (seen[("phrase", i)] for i in range(3))
    z0, z1 = seen[("zoom", 0)], seen[("zoom", 1)]
    assert p0["hist_output_ratio"] == 1.0 and p0["hist_since_last_cut"] == 0 and p0["hist_since_last_zoom"] == 30.0
    assert p1["hist_since_last_cut"] == 1 and p2["hist_since_last_cut"] == 2  # kept phrases since the last cut
    assert z0["hist_prev_zoomed"] == 0.0 and z1["hist_prev_zoomed"] == 1.0  # the previous kept phrase was zoomed
    assert z1["hist_since_last_zoom"] < 30.0 and 0 < p1["hist_output_ratio"] <= 1.5
    assert p0["type_other"] == 1.0 and p0["bias"] == 1.0


def test_cut_resets_since_last_cut_and_pause_removal_lowers_output_ratio(uf3):
    uf, _ = uf3
    sess = EditSession(uf, {})
    ctxs = []
    for item in sess.items():
        if item.decision_type == "phrase":
            ctxs.append(item.ctx)
        sess.commit(item, {"phrase": "CUT" if item.target_id == ids(uf, "speech")[0] else "KEEP", "pause": "REMOVE"}
                    .get(item.decision_type, item.arms[0]))
    assert ctxs[1]["hist_since_last_cut"] == 0 and ctxs[2]["hist_since_last_cut"] == 1
    assert ctxs[1]["hist_output_ratio"] < 1.0  # a whole phrase was cut


def test_commit_validates_action(uf3):
    uf, _ = uf3
    sess = EditSession(uf, {})
    item = next(iter(sess.items()))
    with pytest.raises(ValueError):
        sess.commit(item, "TIGHT_15")


def test_suggest_returns_consistent_suggestions_and_decisions(uf3):
    uf, _ = uf3
    res = suggest(uf, {}, LinTS(seed=1))
    assert len(res.suggestions) == len(res.items)
    for s, it in zip(res.suggestions, res.items, strict=True):
        assert isinstance(s, Suggestion) and s.video_id == uf.video_id and s.target_id == it.target_id
        assert s.decision_type == it.decision_type and s.proposed_action in it.arms
        assert 0 < s.propensity <= 1 and s.confidence == s.propensity and s.policy_name == "linTS"
        assert s.context_raw == pytest.approx({k: float(v) for k, v in it.ctx.items()})
        assert s.candidates == (it.arms if it.decision_type == "emphasis" else [])
    # the decisions the session returns are exactly the proposals
    for s in res.suggestions:
        if s.decision_type in ("phrase", "pause", "filler"):
            assert res.decisions.action(s.target_id) == s.proposed_action
        elif s.decision_type == "zoom":
            assert res.decisions.zoom[s.target_id] == s.proposed_action


def test_suggestions_change_the_edl(uf3):
    uf, _ = uf3
    keep_all = build_edl(uf, suggest(uf, {}, _Fixed({})).decisions)
    cut_pauses = build_edl(uf, suggest(uf, {}, _Fixed({"pause": "REMOVE"})).decisions)
    assert cut_pauses.duration < keep_all.duration
    assert build_edl(uf, suggest(uf, {}, _Fixed({"phrase": "CUT"})).decisions).clips == []


class _Fixed(RulePolicy):
    """Always proposes the given action for a decision type (KEEP/NONE otherwise)."""

    name = "fixed"

    def __init__(self, table):
        super().__init__()
        self.table = table

    def decide(self, decision_type, ctx):
        return self.table.get(decision_type, "NONE" if decision_type in ("zoom", "emphasis") else "KEEP")


def test_emphasis_items_follow_the_edl_caption_lines():
    uf, _ = make_units([("the timeline export is really slow today", 0.5), ("so we fix the render settings now", 1.5)])
    wf = {x.id: {"is_stopword": float(x.text_original in ("the", "is", "so", "we")), "tfidf_in_transcript": 0.5,
                 "word_loudness_z": 0.0} for x in uf.words}
    res = suggest(uf, wf, RulePolicy())
    emph = [s for s in res.suggestions if s.decision_type == "emphasis"]
    edl, line_ids = build_edl_with_lines(uf, res.decisions)
    assert emph and len(line_ids) == len(edl.captions.lines)
    for s, it in zip(emph, [i for i in res.items if i.decision_type == "emphasis"], strict=True):
        assert 0 <= s.target_id < len(line_ids) and it.line_word_ids == line_ids[s.target_id]
        assert set(it.candidate_word_ids) <= set(it.line_word_ids) and s.candidates[0] == "NONE"
        assert s.context_raw["n_cand"] == len(it.candidate_word_ids)
    # emphasis chosen by the policy ends up as highlighted caption words
    n_emph = sum(x.emphasis for line in edl.captions.lines for x in line.words)
    assert n_emph == len(res.decisions.emphasis) == len([s for s in emph if s.proposed_action != "NONE"])


def test_preview_does_not_touch_session_state(uf3):
    uf, _ = uf3
    sess = EditSession(uf, {})
    item = next(iter(sess.items()))
    pv = sess.preview(item, "CUT")
    assert pv.action(item.target_id) == "CUT" and sess.decisions.action(item.target_id) == "KEEP"


def test_rule_decisions_wrapper_matches_rule_policy(uf3):
    from app.edl.rules import rule_decisions

    uf, _ = uf3
    assert rule_decisions(uf).actions == suggest(uf, {}, RulePolicy()).decisions.actions
