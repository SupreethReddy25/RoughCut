"""API: the event pipeline (accept / reject / override / manual edits / edit-only events / nudges)."""
from __future__ import annotations

import pytest
from sqlmodel import select

from app.models import EventRow, SuggestionRow, TextOverlayRow
from tests.apihelpers import (
    assert_policy_consistent,
    other_action,
    pick,
    post_event,
    upload,
    wait_ready,
)


def _suitable(s: dict, p: dict) -> bool:
    """A synthetic video that offers every situation the tests below need (the proposals depend on the prior)."""
    sug = s["suggestions"]
    n_words = {u["id"]: sum(1 for w in p["words"] if w["unit_id"] == u["id"]) for u in p["units"] if u["kind"] == "speech"}
    kept_long = [x for x in sug if x["decision_type"] == "phrase" and x["current_action"] == "KEEP"
                 and n_words[x["target_id"]] >= 8]
    return (sum(x["decision_type"] == "pause" and x["current_action"] == "REMOVE" for x in sug) >= 12
            and any(x["decision_type"] == "filler" and x["current_action"] == "KEEP" for x in sug)
            and any(x["decision_type"] == "zoom" and x["current_action"] == "NONE" for x in sug)
            and any(x["decision_type"] == "zoom" and x["current_action"] != "NONE" for x in sug)
            and any(x["decision_type"] == "emphasis" and x["current_action"] == "CAND_0" and len(x["candidates"]) >= 3
                    for x in sug)
            and bool(kept_long))


@pytest.fixture
def proj(client, ctx):
    for i in range(80):
        uid = client.post("/api/users", json={"name": f"U{i}"}).json()["id"]
        pk = upload(client, uid, (f"clip{i}.mp4",))["id"]
        p = wait_ready(client, pk)
        s = client.post(f"/api/projects/{pk}/suggest").json()
        if _suitable(s, p):
            return type("P", (), {"client": client, "ctx": ctx, "uid": uid, "pk": pk, "units": p["units"],
                                  "words": p["words"], "state": s})()
    pytest.fail("no synthetic video with the required situations in 80 tries")


def find(state: dict, s: dict) -> dict:
    """The active suggestion for the same decision as `s` (live re-scoring may have replaced the row)."""
    return next(x for x in state["suggestions"] if x["decision_type"] == s["decision_type"] and x["target_id"] == s["target_id"])


def words(state: dict) -> list[str]:
    return [w["text"] for line in state["edl"]["captions"]["lines"] for w in line["words"]]


def n_updates(proj) -> int:
    with proj.ctx.session() as s:
        return proj.ctx.policies.get(s, proj.uid).policy.n_updates


def events(proj) -> list[EventRow]:
    with proj.ctx.session() as s:
        return list(s.exec(select(EventRow).where(EventRow.video_id == proj.pk).order_by(EventRow.id)).all())


def unit_words(proj, unit_id: int) -> list[dict]:
    return [w for w in proj.words if w["unit_id"] == unit_id]


def refresh(proj) -> dict:
    return proj.client.get(f"/api/projects/{proj.pk}").json()


# ---------------------------------------------------------------- explicit feedback
def test_accept_trains_the_policy_and_marks_accepted(proj):
    s = pick(proj.state, "pause")
    out = post_event(proj.client, proj.pk, "accept", suggestion_id=s["id"])
    assert out["status"] == "edited" and out["n_events"] == 1 and out["can_undo"] and not out["can_redo"]
    got = find(out, s)
    assert got["status"] == "accepted" and got["current_action"] == s["proposed_action"]
    assert s["id"] not in out["queue"]["needs_review"] + out["queue"]["confident"]  # reviewed items leave the queue
    (e,) = events(proj)
    assert (e.kind, e.reward, e.weight, e.undone) == ("accept", 1.0, 1.0, False)
    assert n_updates(proj) == 1 and out["style_mixture"]["n_events"] == 1
    assert_policy_consistent(proj.ctx, proj.uid)
    if not out["changes"]:  # accepting changes the edit only through live re-scoring of OTHER suggestions
        assert out["edl"] == proj.state["edl"]


def test_override_changes_the_edit_and_learns_both_arms(proj):
    s = next(x for x in proj.state["suggestions"] if x["decision_type"] == "pause" and x["current_action"] == "REMOVE")
    before = proj.state["edl"]["duration"]
    out = post_event(proj.client, proj.pk, "override", suggestion_id=s["id"], final_action="KEEP")
    got = find(out, s)
    assert got["status"] == "corrected" and got["current_action"] == "KEEP" and got["proposed_action"] == "REMOVE"
    assert out["edl"]["duration"] > before  # the pause is back
    (e,) = events(proj)
    assert e.kind == "override" and e.reward == -1.0 and e.payload_json["final_action"] == "KEEP"
    assert e.payload_json["via"] == "panel" and e.payload_json["mapping"]["decision_type"] == "pause"
    assert n_updates(proj) == 2  # proposed -1, new arm +1
    assert_policy_consistent(proj.ctx, proj.uid)


@pytest.mark.parametrize("dt,final", [("phrase", "CUT"), ("filler", "REMOVE"), ("zoom", "Z130")])
def test_override_other_decision_types(proj, dt, final):
    s = next(x for x in proj.state["suggestions"] if x["decision_type"] == dt and x["current_action"] != final)
    out = post_event(proj.client, proj.pk, "override", suggestion_id=s["id"], final_action=final)
    assert find(out, s)["current_action"] == final
    assert_policy_consistent(proj.ctx, proj.uid)


def test_emphasis_override_moves_the_highlight(proj):
    s = pick(proj.state, "emphasis", current_action="CAND_0")
    out = post_event(proj.client, proj.pk, "override", suggestion_id=s["id"], final_action="NONE")
    assert find(out, s)["current_action"] == "NONE" and find(out, s)["status"] == "corrected"

    def lit(state):
        return sum(w["emphasis"] for line in state["edl"]["captions"]["lines"] for w in line["words"])

    if not out["changes"]:  # (live re-scoring may also move OTHER lines' highlights)
        assert lit(out) == lit(proj.state) - 1
    back = post_event(proj.client, proj.pk, "override", suggestion_id=find(out, s)["id"], final_action="CAND_1")
    assert find(back, s)["current_action"] == "CAND_1"
    if not back["changes"]:
        assert lit(back) == lit(out) + 1
    assert_policy_consistent(proj.ctx, proj.uid)


def test_reject_binary_applies_the_other_arm_and_multi_arm_applies_the_neutral_one(proj):
    ph = next(x for x in proj.state["suggestions"] if x["decision_type"] == "phrase" and x["current_action"] == "KEEP")
    out = post_event(proj.client, proj.pk, "reject", suggestion_id=ph["id"])
    assert find(out, ph)["current_action"] == "CUT"
    e = events(proj)[-1]
    assert e.kind == "reject" and e.reward == -1.0 and e.payload_json["applied"] == "CUT"
    assert n_updates(proj) == 2  # binary: proposed -1 and the other arm +1
    pa = next(x for x in proj.state["suggestions"] if x["decision_type"] == "pause" and x["current_action"] == "REMOVE")
    out = post_event(proj.client, proj.pk, "reject", suggestion_id=pa["id"])
    assert find(out, pa)["current_action"] == "KEEP"  # neutral
    assert n_updates(proj) == 3  # multi-arm without alternative: proposed -1 only
    zo = next(x for x in proj.state["suggestions"] if x["decision_type"] == "zoom" and x["current_action"] != "NONE")
    out = post_event(proj.client, proj.pk, "reject", suggestion_id=zo["id"], final_action="NONE")
    assert find(out, zo)["current_action"] == "NONE"
    assert n_updates(proj) == 5  # with an alternative a multi-arm reject is an override
    assert_policy_consistent(proj.ctx, proj.uid)


def test_validation_errors(proj):
    c, pk = proj.client, proj.pk
    s = pick(proj.state, "pause")
    assert post_event(c, pk, "accept", 422)["detail"]  # no suggestion_id
    assert post_event(c, pk, "accept", 404, suggestion_id=99999)["detail"]
    assert post_event(c, pk, "override", 422, suggestion_id=s["id"])["detail"]  # no final_action
    assert post_event(c, pk, "override", 422, suggestion_id=s["id"], final_action="WARP")["detail"]
    assert post_event(c, pk, "override", 409, suggestion_id=s["id"], final_action=s["current_action"])["detail"]
    assert post_event(c, pk, "reject", 422, suggestion_id=s["id"], final_action="Z115")["detail"]
    assert c.post(f"/api/projects/{pk}/events", json={"kind": "dance"}).status_code == 422
    assert post_event(c, pk, "music", 501)["detail"]
    other = upload(c, proj.uid, ("other.mp4",))["id"]
    wait_ready(c, other)
    c.post(f"/api/projects/{other}/suggest")
    assert post_event(c, other, "accept", 404, suggestion_id=s["id"])["detail"]  # a suggestion of another project
    assert events(proj) == []  # nothing was written by any rejected request
    assert n_updates(proj) == 0


def test_changing_your_mind_moves_from_the_current_action(proj):
    s = next(x for x in proj.state["suggestions"] if x["decision_type"] == "pause" and x["current_action"] == "REMOVE")
    post_event(proj.client, proj.pk, "override", suggestion_id=s["id"], final_action="KEEP")
    out = post_event(proj.client, proj.pk, "override", suggestion_id=s["id"], final_action="TIGHT_40")
    assert find(out, s)["current_action"] == "TIGHT_40"
    e2 = events(proj)[-1]
    assert e2.payload_json["from_action"] == "KEEP"  # KEEP -1 and TIGHT_40 +1, not REMOVE -1 a second time
    acc = post_event(proj.client, proj.pk, "accept", suggestion_id=s["id"])
    e3 = events(proj)[-1]
    assert e3.payload_json["from_action"] == "TIGHT_40"
    if not acc["changes"]:
        assert acc["edl"] == out["edl"]
    assert n_updates(proj) == 2 + 2 + 1
    assert_policy_consistent(proj.ctx, proj.uid)


# ---------------------------------------------------------------- manual (transcript) edits
def test_deleting_a_whole_phrase_is_a_phrase_override(proj):
    s = next(x for x in proj.state["suggestions"] if x["decision_type"] == "phrase" and x["current_action"] == "KEEP")
    ids = [w["id"] for w in unit_words(proj, s["target_id"])]
    before = proj.state["edl"]["duration"]
    out = post_event(proj.client, proj.pk, "manual", edit={"kind": "delete_words", "word_ids": ids})
    got = find(out, s)
    assert got["current_action"] == "CUT" and got["status"] == "corrected" and out["edl"]["duration"] < before
    e = events(proj)[-1]
    assert e.kind == "override" and e.payload_json["via"] == "manual" and n_updates(proj) == 2
    assert_policy_consistent(proj.ctx, proj.uid)
    # restoring the words brings the phrase back
    out2 = post_event(proj.client, proj.pk, "manual", edit={"kind": "restore_words", "word_ids": ids})
    assert find(out2, s)["current_action"] == "KEEP"
    if not out["changes"] and not out2["changes"]:  # otherwise live re-scoring also moved other suggestions
        assert out2["edl"]["duration"] == pytest.approx(before, abs=0.01)


def test_deleting_a_filler_token_is_a_filler_override(proj):
    s = next(x for x in proj.state["suggestions"] if x["decision_type"] == "filler" and x["current_action"] == "KEEP")
    fu = next(u for u in proj.units if u["id"] == s["target_id"])
    ids = [w["id"] for w in proj.words if w["unit_id"] == fu["parent_unit_id"] and w["is_filler"]
           and w["start"] >= fu["start"] - 1e-3 and w["end"] <= fu["end"] + 1e-3]
    out = post_event(proj.client, proj.pk, "manual", edit={"kind": "delete_words", "word_ids": ids})
    got = find(out, s)
    assert got["current_action"] == "REMOVE" and got["status"] == "corrected"
    assert events(proj)[-1].kind == "override" and n_updates(proj) == 2


def test_a_small_word_deletion_is_a_fine_cut_that_is_not_learned(proj):
    unit = next(u for u in proj.units if u["kind"] == "speech" and len(unit_words(proj, u["id"])) >= 8
                and next(x for x in proj.state["suggestions"] if x["decision_type"] == "phrase"
                         and x["target_id"] == u["id"])["current_action"] == "KEEP")
    w = unit_words(proj, unit["id"])[3]
    out = post_event(proj.client, proj.pk, "manual", edit={"kind": "delete_words", "word_ids": [w["id"]]})
    e = events(proj)[-1]
    assert e.kind == "fine_cut" and e.reward is None and n_updates(proj) == 0
    assert out["can_undo"] and out["n_events"] == 1
    assert out["edl"]["duration"] < proj.state["edl"]["duration"]
    undone = proj.client.post(f"/api/projects/{proj.pk}/undo").json()
    assert undone["edl"] == proj.state["edl"] and not undone["can_undo"] and undone["can_redo"]


def test_manual_pause_zoom_and_emphasis_edits(proj):
    c, pk = proj.client, proj.pk
    pa = next(x for x in proj.state["suggestions"] if x["decision_type"] == "pause" and x["current_action"] == "REMOVE")
    out = post_event(c, pk, "manual", edit={"kind": "set_pause", "unit_id": pa["target_id"], "option": "TIGHT_40"})
    assert find(out, pa)["current_action"] == "TIGHT_40"
    unit = next(u for u in proj.units if u["id"] == pa["target_id"])
    gap = unit["end"] - unit["start"]
    out = post_event(c, pk, "manual", edit={"kind": "drag_pause", "unit_id": pa["target_id"], "out_gap": gap})
    assert find(out, pa)["current_action"] == "KEEP"
    zo = next(x for x in proj.state["suggestions"] if x["decision_type"] == "zoom" and x["current_action"] == "NONE")
    out = post_event(c, pk, "manual", edit={"kind": "set_zoom", "unit_id": zo["target_id"], "option": "Z115"})
    assert find(out, zo)["current_action"] == "Z115"
    assert any(cl["zoom"] == 1.15 for cl in out["edl"]["clips"])
    em = pick(proj.state, "emphasis", current_action="CAND_0")
    line = [w for w in proj.words if w["id"] in em["candidate_word_ids"]]
    cand = em["candidate_word_ids"]
    line_ids = [w["id"] for w in proj.words if w["unit_id"] == line[0]["unit_id"]]
    out = post_event(c, pk, "manual", edit={"kind": "set_emphasis", "word_id": cand[-1] if len(cand) > 1 else cand[0],
                                            "line_word_ids": line_ids})
    got = find(out, em)
    assert got["current_action"] == f"CAND_{len(cand) - 1}" if len(cand) > 1 else "CAND_0"
    cleared = post_event(c, pk, "manual", edit={"kind": "set_emphasis", "word_id": None, "line_word_ids": line_ids})
    assert find(cleared, em)["current_action"] == "NONE"
    assert all(e.kind == "override" for e in events(proj))
    assert_policy_consistent(proj.ctx, proj.uid)


def test_emphasising_a_non_candidate_word_is_applied_but_not_learned(proj):
    em = pick(proj.state, "emphasis", current_action="CAND_0")
    unit = next(w for w in proj.words if w["id"] in em["candidate_word_ids"])["unit_id"]
    line_ids = [w["id"] for w in proj.words if w["unit_id"] == unit]
    other = next(i for i in line_ids if i not in em["candidate_word_ids"])
    out = post_event(proj.client, proj.pk, "manual", edit={"kind": "set_emphasis", "word_id": other, "line_word_ids": line_ids})
    e = events(proj)[-1]
    assert e.kind == "fine_cut" and n_updates(proj) == 0
    assert sum(w["emphasis"] for ln in out["edl"]["captions"]["lines"] for w in ln["words"]) >= 1


def test_manual_edits_that_change_nothing_create_no_event(proj):
    pa = next(x for x in proj.state["suggestions"] if x["decision_type"] == "pause")
    out = post_event(proj.client, proj.pk, "manual", edit={"kind": "set_pause", "unit_id": pa["target_id"],
                                                           "option": pa["current_action"]})
    assert out["message"].startswith("No change") and out["n_events"] == 0 and events(proj) == []
    out = post_event(proj.client, proj.pk, "manual", edit={"kind": "set_emphasis", "word_id": None, "line_word_ids": [0]})
    assert out["n_events"] == 0 and events(proj) == []


def test_manual_edit_validation(proj):
    c, pk = proj.client, proj.pk
    assert c.post(f"/api/projects/{pk}/events", json={"kind": "manual"}).status_code == 422
    bad_ids = {"kind": "delete_words", "word_ids": [10**6]}
    assert c.post(f"/api/projects/{pk}/events", json={"kind": "manual", "edit": bad_ids}).status_code == 422
    assert c.post(f"/api/projects/{pk}/events", json={"kind": "manual", "edit": {"kind": "set_pause", "unit_id": 0, "option": "WARP"}}).status_code == 422
    assert c.post(f"/api/projects/{pk}/events", json={"kind": "manual", "edit": {"kind": "set_zoom", "unit_id": 0, "option": "Z9"}}).status_code == 422
    assert c.post(f"/api/projects/{pk}/events", json={"kind": "manual", "edit": {"kind": "drag_pause", "unit_id": 0, "out_gap": 0.5}}).status_code == 422  # unit 0 is speech
    assert c.post(f"/api/projects/{pk}/events", json={"kind": "manual", "edit": {"kind": "teleport"}}).status_code == 422
    assert events(proj) == []


def test_a_transcript_edit_that_maps_to_several_events_undoes_as_one(proj):
    s = next(x for x in proj.state["suggestions"] if x["decision_type"] == "filler" and x["current_action"] == "KEEP")
    fu = next(u for u in proj.units if u["id"] == s["target_id"])
    fillers = [w["id"] for w in proj.words if w["unit_id"] == fu["parent_unit_id"] and w["is_filler"]
               and w["start"] >= fu["start"] - 1e-3 and w["end"] <= fu["end"] + 1e-3]
    phrase_words = unit_words(proj, fu["parent_unit_id"])
    extra = next(w["id"] for w in phrase_words if not w["is_filler"])
    out = post_event(proj.client, proj.pk, "manual", edit={"kind": "delete_words", "word_ids": [*fillers, extra]})
    kinds = sorted(e.kind for e in events(proj))
    assert kinds == ["fine_cut", "override"] and out["n_events"] == 2
    assert len({e.payload_json["group"] for e in events(proj)}) == 1
    undone = proj.client.post(f"/api/projects/{proj.pk}/undo").json()
    assert undone["n_events"] == 0 and not undone["can_undo"] and undone["edl"] == proj.state["edl"]
    assert all(e.undone for e in events(proj))
    redone = proj.client.post(f"/api/projects/{proj.pk}/redo").json()
    assert redone["n_events"] == 2 and redone["edl"] == out["edl"]
    assert_policy_consistent(proj.ctx, proj.uid)


# ---------------------------------------------------------------- edit-only events (never learned)
def test_speed_and_settings(proj):
    c, pk = proj.client, proj.pk
    base = proj.state["edl"]["duration"]
    out = post_event(c, pk, "speed", payload={"value": 1.25})
    assert out["settings"]["speed"] == 1.25 and out["edl"]["output"]["speed"] == 1.25
    assert out["edl"]["duration"] == pytest.approx(base / 1.25, abs=0.05)
    assert post_event(c, pk, "speed", 422, payload={"value": 3})["detail"]
    assert post_event(c, pk, "speed", 422, payload={})["detail"]
    out = post_event(c, pk, "setting", payload={"key": "caption_style", "value": "clean"})
    assert out["edl"]["captions"]["style_id"] == "clean"
    out = post_event(c, pk, "setting", payload={"key": "caption_position", "value": "top"})
    assert out["edl"]["captions"]["position"] == "top"
    out = post_event(c, pk, "setting", payload={"key": "captions_enabled", "value": False})
    assert out["edl"]["captions"]["enabled"] is False
    out = post_event(c, pk, "setting", payload={"key": "audio_enhance", "value": False})
    assert out["edl"]["audio"]["enhance"] is False
    for bad in ({"key": "caption_style", "value": "neon"}, {"key": "caption_position", "value": "left"},
                {"key": "captions_enabled", "value": "yes"}, {"key": "volume", "value": 11}):
        assert post_event(c, pk, "setting", 422, payload=bad)["detail"]
    assert n_updates(proj) == 0  # edit-only: nothing is learned from settings
    assert all(e.reward is None for e in events(proj))


def test_caption_text_fix(proj):
    wid = next(w["id"] for w in proj.words if not w["is_filler"] and w["text_display"].strip(".,?!") in words(proj.state))
    orig = next(w for w in proj.words if w["id"] == wid)["text_display"].strip(".,?!")
    out = post_event(proj.client, proj.pk, "caption_text", payload={"word_id": wid, "text": "FIXED"})
    assert "FIXED" in words(out) and len(words(out)) == len(words(proj.state))
    assert orig in words(proj.state)
    assert post_event(proj.client, proj.pk, "caption_text", 422, payload={"word_id": wid, "text": "  "})["detail"]
    assert post_event(proj.client, proj.pk, "caption_text", 422, payload={"word_id": 10**6, "text": "x"})["detail"]
    assert post_event(proj.client, proj.pk, "caption_text", 422, payload={"text": "x"})["detail"]
    assert post_event(proj.client, proj.pk, "caption_text", 422, payload={"word_id": wid, "text": "x" * 81})["detail"]
    assert n_updates(proj) == 0


def test_crop_override(proj):
    clip = proj.state["edl"]["clips"][0]
    first_unit = next(i for i in clip["unit_ids"] if proj.units[i]["kind"] == "speech")
    out = post_event(proj.client, proj.pk, "crop", payload={"unit_id": first_unit, "x": 0.31, "y": 0.62})
    assert out["edl"]["clips"][0]["crop_center"] == {"x": 0.31, "y": 0.62}
    for bad in ({"unit_id": first_unit, "x": 1.5, "y": 0.5}, {"unit_id": 10**6, "x": 0.5, "y": 0.5}, {"x": 0.5}):
        assert post_event(proj.client, proj.pk, "crop", 422, payload=bad)["detail"]
    pause = next(u["id"] for u in proj.units if u["kind"] == "pause")
    assert post_event(proj.client, proj.pk, "crop", 422, payload={"unit_id": pause, "x": 0.5, "y": 0.5})["detail"]


def test_text_overlays_add_update_delete(proj, ctx):
    c, pk = proj.client, proj.pk
    out = post_event(c, pk, "text_overlay", payload={"op": "add", "text": "3 mistakes", "t0": 0.0, "t1": 3.0})
    (ov,) = out["edl"]["texts"]
    assert ov["text"] == "3 mistakes" and ov["position"] == "top" and ov["style_id"] == "title" and len(ov["id"]) == 8
    out = post_event(c, pk, "text_overlay", payload={"op": "update", "id": ov["id"], "text": "Four!", "position": "custom", "y": 0.4})
    assert out["edl"]["texts"][0]["text"] == "Four!" and out["edl"]["texts"][0]["y"] == 0.4
    out = post_event(c, pk, "text_overlay", payload={"op": "add", "text": "second", "t0": 1.0, "t1": 2.0, "style_id": "label"})
    assert len(out["edl"]["texts"]) == 2
    out = post_event(c, pk, "text_overlay", payload={"op": "delete", "id": ov["id"]})
    assert [t["text"] for t in out["edl"]["texts"]] == ["second"]
    bad = [{"op": "add", "text": "x", "t0": 2.0, "t1": 1.0}, {"op": "add", "text": "", "t0": 0, "t1": 1},
           {"op": "add", "text": "x", "t0": 0, "t1": 1, "style_id": "neon"}, {"op": "add", "text": "x", "t0": 0, "t1": 1, "position": "custom"},
           {"op": "add", "text": "y" * 121, "t0": 0, "t1": 1}, {"op": "move"}]
    for p in bad:
        assert post_event(c, pk, "text_overlay", 422, payload=p)["detail"], p
    assert post_event(c, pk, "text_overlay", 404, payload={"op": "delete", "id": "nope"})["detail"]
    assert post_event(c, pk, "text_overlay", 404, payload={"op": "update", "id": "nope", "text": "x"})["detail"]
    assert n_updates(proj) == 0
    # undo of the delete brings it back
    undone = c.post(f"/api/projects/{pk}/undo").json()
    assert len(undone["edl"]["texts"]) == 2


# ---------------------------------------------------------------- nudges and live re-scoring
def test_nudge_rescores_untouched_suggestions_and_is_undoable(proj):
    c, pk = proj.client, proj.pk
    before = proj.state
    out = post_event(c, pk, "nudge", payload={"name": "snappier", "magnitude": 3.0})
    assert out["changes"] and out["message"].startswith("Adjusted")
    assert out["edl"]["duration"] < before["edl"]["duration"]
    ids = {x["id"] for x in out["suggestions"]}
    for ch in out["changes"]:
        assert ch["old_id"] not in ids and ch["new_id"] in ids and ch["from_action"] != ch["to_action"]
    assert {x["source"] for x in out["suggestions"] if x["id"] in {ch["new_id"] for ch in out["changes"]}} == {"nudge"}
    assert {ch["decision_type"] for ch in out["changes"]} <= {"pause", "phrase", "filler"}  # the only nudged types
    assert all(ch["to_action"] in ("TIGHT_15", "REMOVE", "CUT") for ch in out["changes"])
    assert n_updates(proj) == 0
    with ctx_session(proj) as s:
        old = s.get(SuggestionRow, out["changes"][0]["old_id"])
        new = s.get(SuggestionRow, out["changes"][0]["new_id"])
        assert old.superseded and old.superseded_by_event == new.caused_by_event and new.rescored_from == old.id
        assert new.propensity == 1.0 and new.policy_name.endswith(":nudge")
    undone = c.post(f"/api/projects/{pk}/undo").json()
    assert undone["edl"] == before["edl"] and [x["id"] for x in undone["suggestions"]] == [x["id"] for x in before["suggestions"]]
    redone = c.post(f"/api/projects/{pk}/redo").json()
    assert redone["edl"] == out["edl"] and {x["id"] for x in redone["suggestions"]} == ids
    calmer = post_event(c, pk, "nudge", payload={"name": "calmer", "magnitude": 3.0})
    assert calmer["edl"]["duration"] > out["edl"]["duration"]
    for bad in ({"name": "louder"}, {"name": "snappier", "magnitude": 0}, {"name": "snappier", "magnitude": 9}):
        assert post_event(c, pk, "nudge", 422, payload=bad)["detail"]


def ctx_session(proj):
    return proj.ctx.session()


def test_live_rescoring_after_feedback_replaces_only_untouched_unseen_suggestions(proj):
    c, pk = proj.client, proj.pk
    n_remove = sum(x["decision_type"] == "pause" and x["current_action"] == "REMOVE" for x in proj.state["suggestions"])
    assert n_remove >= 12
    # the user keeps correcting REMOVE -> KEEP: the model should start to doubt its other REMOVE pauses
    changed, touched = [], []
    state = proj.state
    for _ in range(12):
        x = next((x for x in state["suggestions"] if x["decision_type"] == "pause" and x["current_action"] == "REMOVE"
                  and x["status"] == "pending"), None)
        if x is None:
            break
        out = post_event(c, pk, "override", suggestion_id=x["id"], final_action="KEEP")
        touched.append((x["decision_type"], x["target_id"]))
        changed += out["changes"]
        if out["changes"]:
            n = len(out["changes"])
            assert out["message"] == f"Adjusted {n} suggestion{'s' if n != 1 else ''} based on your last change"
        state = out
    assert changed, "the posterior moved but no unreviewed suggestion was re-scored"
    by_id = {x["id"]: x for x in state["suggestions"]}
    for ch in changed:  # replaced rows are gone from the active set, their replacements are pending
        assert ch["old_id"] not in by_id or by_id[ch["old_id"]]["status"] != "pending"
    for dt, tid in touched:  # touched suggestions were never replaced
        assert next(x for x in state["suggestions"] if x["decision_type"] == dt and x["target_id"] == tid)["status"] == "corrected"
    assert_policy_consistent(proj.ctx, proj.uid)
    # seen suggestions are not moved under the user's eyes
    unit_ids = [u["id"] for u in proj.units]
    assert c.post(f"/api/projects/{pk}/seen", json={"unit_ids": unit_ids}).json()["marked"] == len(unit_ids)
    rest = next(x for x in state["suggestions"] if x["decision_type"] == "pause" and x["status"] == "pending")
    out = post_event(c, pk, "override", suggestion_id=rest["id"], final_action=other_action(rest))
    assert out["changes"] == []


def test_acting_on_a_replaced_suggestion_is_a_conflict_that_names_the_replacement(proj):
    c, pk = proj.client, proj.pk
    out = post_event(c, pk, "nudge", payload={"name": "snappier", "magnitude": 3.0})
    ch = out["changes"][0]
    r = c.post(f"/api/projects/{pk}/events", json={"kind": "accept", "suggestion_id": ch["old_id"]})
    assert r.status_code == 409 and r.json()["replaced_by"] == ch["new_id"]
    assert post_event(c, pk, "accept", suggestion_id=ch["new_id"])["n_events"] == 2


def test_events_mark_the_project_edited(proj):
    assert refresh(proj)["status"] == "ready"
    post_event(proj.client, proj.pk, "accept", suggestion_id=pick(proj.state, "pause")["id"])
    assert refresh(proj)["status"] == "edited"


def test_text_overlay_table_is_synced_on_render(proj, ctx):
    post_event(proj.client, proj.pk, "text_overlay", payload={"op": "add", "text": "hello", "t0": 0, "t1": 2})
    from app.svc.bundle import load_bundle
    from app.svc.projects import sync_text_overlays

    with ctx.session() as s:
        sync_text_overlays(s, load_bundle(ctx, s, proj.pk))
        s.commit()
        rows = s.exec(select(TextOverlayRow).where(TextOverlayRow.video_id == proj.pk)).all()
        assert [(r.text, r.t1) for r in rows] == [("hello", 2.0)]
