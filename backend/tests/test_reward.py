"""PLAN 6.4: every row of the reward table and every row of the manual-edit -> feedback mapping."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.edl.build import Decisions, build_edl
from app.policy.linear import LinTS
from app.policy.log_replay import Learner
from app.policy.reward import (
    DEFAULT,
    ManualEdit,
    RewardConfig,
    apply_mapping,
    classify_manual_edit,
    snap_pause_option,
    updates_for,
)
from tests.test_edl import ids, make_units


def ups(*a, **k):
    return [(u.arm, u.reward, u.weight) for u in updates_for(*a, **k)]


# ---------------------------------------------------------------- suggestion feedback rows
def test_reward_table_rows():
    pause = ["KEEP", "TIGHT_40", "TIGHT_15", "REMOVE"]
    assert ups("accept", "pause", "REMOVE", pause) == [("REMOVE", 1.0, 1.0)]  # accept: proposed +1
    # reject on binary (D1, D4): proposed -1, the other arm +1
    assert ups("reject", "phrase", "CUT", ["KEEP", "CUT"]) == [("CUT", -1.0, 1.0), ("KEEP", 1.0, 1.0)]
    assert ups("reject", "filler", "REMOVE", ["KEEP", "REMOVE"]) == [("REMOVE", -1.0, 1.0), ("KEEP", 1.0, 1.0)]
    # reject on multi-arm (D2, D3, D5) without an alternative: proposed -1 only
    assert ups("reject", "pause", "REMOVE", pause) == [("REMOVE", -1.0, 1.0)]
    assert ups("reject", "zoom", "Z115", ["NONE", "Z115", "Z130"]) == [("Z115", -1.0, 1.0)]
    assert ups("reject", "emphasis", "CAND_0", ["NONE", "CAND_0", "CAND_1"]) == [("CAND_0", -1.0, 1.0)]
    # override to a': proposed -1, a' +1
    assert ups("override", "pause", "REMOVE", pause, "TIGHT_40") == [("REMOVE", -1.0, 1.0), ("TIGHT_40", 1.0, 1.0)]
    assert ups("override", "phrase", "KEEP", ["KEEP", "CUT"], "CUT") == [("KEEP", -1.0, 1.0), ("CUT", 1.0, 1.0)]
    assert ups("override", "pause", "KEEP", pause, "KEEP") == [("KEEP", 1.0, 1.0)]  # override to itself == accept
    # untouched at finalize and seen: +0.5 (implicit); never seen: the caller emits nothing
    assert ups("implicit", "pause", "REMOVE", pause) == [("REMOVE", 0.5, 1.0)]
    assert ups("implicit", "pause", "REMOVE", pause, cfg=RewardConfig(implicit_weight=0.3)) == [("REMOVE", 0.5, 0.3)]
    # events that are not decisions never learn
    for kind in ("fine_cut", "crop", "caption_text", "text_overlay", "music", "speed", "nudge", "setting"):
        assert ups(kind, "pause", "REMOVE", pause) == []
    with pytest.raises(ValueError):
        updates_for("override", "pause", "REMOVE", pause, "WARP")


def test_manual_edit_learning_can_be_ablated():
    pause = ["KEEP", "TIGHT_40", "TIGHT_15", "REMOVE"]
    off = RewardConfig(learn_manual=False)
    assert updates_for("override", "pause", "REMOVE", pause, "KEEP", off, manual=True) == []
    assert len(updates_for("override", "pause", "REMOVE", pause, "KEEP", off, manual=False)) == 2
    assert len(updates_for("override", "pause", "REMOVE", pause, "KEEP", DEFAULT, manual=True)) == 2


def test_learner_applies_updates_and_logs_nothing_for_edit_only():
    pol = LinTS(seed=0)
    lr = Learner(pol)
    ctx = {"gap_seconds": 1.0, "bias": 1.0}
    got = lr.apply("pause", ctx, "REMOVE", "override", "KEEP")
    assert [(u.arm, u.reward) for u in got] == [("REMOVE", -1.0), ("KEEP", 1.0)] and pol.n_updates == 2
    assert lr.apply("pause", ctx, "REMOVE", "crop") == [] and pol.n_updates == 2


# ---------------------------------------------------------------- manual edit mapping rows
@pytest.fixture
def scene():
    # phrase 0: "um I think we should go now" (7 words, 'um' is a filler), phrase 1: "and then we stop here today"
    uf, w = make_units([("um I think we should go now", 0.5), ("and then we stop here today", 1.5)])
    return SimpleNamespace(uf=uf, w=w, sp=ids(uf, "speech"), fu=ids(uf, "filler")[0], p=ids(uf, "pause")[0],
                           dec=Decisions())


def word_ids(uf, uid):
    return [x.id for x in uf.words if x.unit_id == uid]


def test_delete_whole_phrase_is_a_phrase_override_and_is_applied(scene):
    uf, dec = scene.uf, scene.dec
    sp = scene.sp
    (m,) = classify_manual_edit(ManualEdit("delete_words", word_ids=word_ids(uf, sp[1])), uf, dec)
    assert (m.kind, m.decision_type, m.target_id, m.final_action) == ("override", "phrase", sp[1], "CUT")
    apply_mapping(dec, m, uf)
    assert dec.action(sp[1]) == "CUT"
    assert "stop" not in [x.text for line in build_edl(uf, dec).captions.lines for x in line.words]
    # restoring all of it is the opposite override
    (r,) = classify_manual_edit(ManualEdit("restore_words", word_ids=word_ids(uf, sp[1])), uf, dec)
    assert (r.kind, r.decision_type, r.final_action) == ("override", "phrase", "KEEP")
    apply_mapping(dec, r, uf)
    assert dec.action(sp[1]) == "KEEP"


def test_deleting_70_percent_of_a_phrase_cuts_the_rest_too(scene):
    uf, dec = scene.uf, scene.dec
    sp = scene.sp
    ids0 = word_ids(uf, sp[0])  # 7 equal words: 5/7 = 71% >= 70%, 4/7 = 57% < 70%
    (m,) = classify_manual_edit(ManualEdit("delete_words", word_ids=ids0[2:7]), uf, dec)
    assert (m.kind, m.decision_type, m.final_action) == ("override", "phrase", "CUT")
    apply_mapping(dec, m, uf)
    assert not [x for line in build_edl(uf, dec).captions.lines for x in line.words if x.text in ("um", "I", "think")]
    (n,) = classify_manual_edit(ManualEdit("delete_words", word_ids=ids0[3:7]), uf, Decisions())
    assert n.kind == "fine_cut"


def test_delete_and_restore_a_single_filler_token_is_a_filler_override(scene):
    uf, dec = scene.uf, scene.dec
    fu = scene.fu
    um = next(x.id for x in uf.words if x.text_original == "um")
    (m,) = classify_manual_edit(ManualEdit("delete_words", word_ids=[um]), uf, dec)
    assert (m.kind, m.decision_type, m.target_id, m.final_action) == ("override", "filler", fu, "REMOVE")
    apply_mapping(dec, m, uf)
    assert dec.action(fu) == "REMOVE" and "um" not in [x.text for line in build_edl(uf, dec).captions.lines for x in line.words]
    (r,) = classify_manual_edit(ManualEdit("restore_words", word_ids=[um]), uf, dec)
    assert (r.kind, r.decision_type, r.target_id, r.final_action) == ("override", "filler", fu, "KEEP")
    apply_mapping(dec, r, uf)
    assert dec.action(fu) == "KEEP"


def test_small_non_filler_deletion_is_a_fine_cut_not_learned(scene):
    uf, dec = scene.uf, scene.dec
    think = next(x.id for x in uf.words if x.text_original == "think")
    (m,) = classify_manual_edit(ManualEdit("delete_words", word_ids=[think]), uf, dec)
    assert m.kind == "fine_cut" and m.op == "cut" and m.word_ids == [think] and m.decision_type is None
    apply_mapping(dec, m, uf)
    assert think in dec.fine_cuts
    words = [x.text for line in build_edl(uf, dec).captions.lines for x in line.words]
    assert "think" not in words and "should" in words
    # logged but never learned: no bandit update comes out of it
    assert updates_for("fine_cut", "phrase", "KEEP", ["KEEP", "CUT"]) == []
    (r,) = classify_manual_edit(ManualEdit("restore_words", word_ids=[think]), uf, dec)
    assert r.kind == "fine_cut" and r.op == "restore"
    apply_mapping(dec, r, uf)
    assert think not in dec.fine_cuts
    assert "think" in [x.text for line in build_edl(uf, dec).captions.lines for x in line.words]


def test_partial_restore_inside_a_cut_phrase_is_a_fine_cut(scene):
    uf, dec = scene.uf, scene.dec
    sp = scene.sp
    dec.actions[sp[1]] = "CUT"
    some = word_ids(uf, sp[1])[:2]
    (m,) = classify_manual_edit(ManualEdit("restore_words", word_ids=some), uf, dec)
    assert m.kind == "fine_cut" and m.op == "restore"
    apply_mapping(dec, m, uf)
    assert set(some) <= dec.restored
    words = [x.text for line in build_edl(uf, dec).captions.lines for x in line.words]
    assert "and" in words and "then" in words and "stop" not in words


def test_deleting_already_cut_words_changes_nothing(scene):
    uf, dec = scene.uf, scene.dec
    sp = scene.sp
    dec.actions[sp[1]] = "CUT"
    (m,) = classify_manual_edit(ManualEdit("delete_words", word_ids=word_ids(uf, sp[1])), uf, dec)
    assert m.kind == "edit_only"


def test_pause_edits_snap_to_the_nearest_option(scene):
    uf, dec = scene.uf, scene.dec
    sp = scene.sp
    p = scene.p
    (m,) = classify_manual_edit(ManualEdit("set_pause", unit_id=p, option="TIGHT_15"), uf, dec)
    assert (m.kind, m.decision_type, m.target_id, m.final_action) == ("override", "pause", p, "TIGHT_15")
    for out_gap, want in [(5.0, "KEEP"), (0.5, "TIGHT_40"), (0.25, "TIGHT_15"), (0.12, "REMOVE"), (0.0, "REMOVE")]:
        (d,) = classify_manual_edit(ManualEdit("drag_pause", unit_id=p, out_gap=out_gap), uf, dec)
        assert d.final_action == want, (out_gap, d.final_action)
    assert snap_pause_option(0.3, 0.3) == "KEEP"  # a short pause: every option leaves the same silence -> KEEP
    apply_mapping(dec, m, uf)
    assert dec.action(p) == "TIGHT_15"
    with pytest.raises(ValueError):
        classify_manual_edit(ManualEdit("set_pause", unit_id=p, option="WARP"), uf, dec)
    with pytest.raises(ValueError):
        classify_manual_edit(ManualEdit("drag_pause", unit_id=sp[0], out_gap=0.2), uf, dec)  # not a pause unit


def test_zoom_change_is_a_zoom_override(scene):
    uf, dec = scene.uf, scene.dec
    sp = scene.sp
    (m,) = classify_manual_edit(ManualEdit("set_zoom", unit_id=sp[0], option="Z130"), uf, dec)
    assert (m.kind, m.decision_type, m.target_id, m.final_action) == ("override", "zoom", sp[0], "Z130")
    apply_mapping(dec, m, uf)
    assert dec.zoom[sp[0]] == "Z130"
    with pytest.raises(ValueError):
        classify_manual_edit(ManualEdit("set_zoom", unit_id=sp[0], option="Z999"), uf, dec)


def test_clicking_a_caption_word_is_an_emphasis_override(scene):
    uf, dec = scene.uf, scene.dec
    line = [x.id for x in uf.words][:4]
    cands = [line[1], line[3]]
    (m,) = classify_manual_edit(ManualEdit("set_emphasis", line_idx=0, word_id=line[3], line_word_ids=line,
                                           candidate_word_ids=cands), uf, dec)
    assert (m.kind, m.decision_type, m.target_id, m.final_action) == ("override", "emphasis", 0, "CAND_1")
    apply_mapping(dec, m, uf)
    assert dec.emphasis == {line[3]}
    (clear,) = classify_manual_edit(ManualEdit("set_emphasis", line_idx=0, word_id=None, line_word_ids=line,
                                               candidate_word_ids=cands), uf, dec)
    assert clear.final_action == "NONE"
    apply_mapping(dec, clear, uf)
    assert dec.emphasis == set()
    (other,) = classify_manual_edit(ManualEdit("set_emphasis", line_idx=0, word_id=line[0], line_word_ids=line,
                                               candidate_word_ids=cands), uf, dec)
    assert other.kind == "edit_only"  # a word that is not one of the 3 candidates cannot be learned as an arm
    # and the override feeds the shared-theta bandit like any other
    assert ups("override", "emphasis", "CAND_0", ["NONE", "CAND_0", "CAND_1"], "CAND_1") == [("CAND_0", -1.0, 1.0), ("CAND_1", 1.0, 1.0)]


@pytest.mark.parametrize("kind", ["boundary_nudge", "crop", "caption_text", "text_overlay", "music", "speed"])
def test_edit_only_kinds_are_not_learned(scene, kind):
    uf, dec = scene.uf, scene.dec
    (m,) = classify_manual_edit(ManualEdit(kind), uf, dec)
    assert m.kind == "edit_only"
    before = (dict(dec.actions), dict(dec.zoom), set(dec.fine_cuts), set(dec.restored), set(dec.emphasis))
    apply_mapping(dec, m, uf)
    assert before == (dict(dec.actions), dict(dec.zoom), set(dec.fine_cuts), set(dec.restored), set(dec.emphasis))
    assert updates_for(kind, "phrase", "KEEP", ["KEEP", "CUT"]) == []


def test_unknown_manual_edit_raises(scene):
    uf, dec = scene.uf, scene.dec
    with pytest.raises(ValueError):
        classify_manual_edit(ManualEdit("teleport"), uf, dec)
