"""Phase 5 (backend): suggested title, remembered defaults, background music, autopilot / trust, caption word ids."""
from __future__ import annotations

import io
import subprocess
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from app import config
from app import music as music_lib
from app.appctx import AppContext
from app.main import create_app
from app.models import Metric, SuggestionRow, Video
from app.render import graph
from app.state import hook_overlay, sanitize_defaults, validate_music
from tests.apihelpers import fake_ingest, new_ready_project, post_event, upload, wait_ready

LIB_TRACK = "lib_sunrise"


def get_state(client, pk):
    return client.get(f"/api/projects/{pk}").json()


def suggested(client, pk):
    return client.post(f"/api/projects/{pk}/suggest").json()


# --------------------------------------------------------------------------- suggested title (PLAN 8.7)
def test_a_new_project_starts_with_a_suggested_title_the_user_can_delete(client):
    _, pk, p = new_ready_project(client)
    s = suggested(client, pk)
    (hook,) = [t for t in s["edl"]["texts"] if t["id"] == "hook"]
    assert 0 < len(hook["text"].split()) <= 8 and hook["t0"] == 0 and hook["t1"] == 3.0 and hook["style_id"] == "title"
    first = next(u for u in p["units"] if u["kind"] == "speech")
    assert hook["text"].split()[0].lower().strip(".,") == first["text"].split()[0].lower().strip(".,")

    gone = post_event(client, pk, "text_overlay", payload={"op": "delete", "id": "hook"})
    assert not [t for t in gone["edl"]["texts"] if t["id"] == "hook"]
    assert gone["events"][0]["kind"] == "text_overlay"  # an edit-only event: nothing is learned from it
    assert client.post(f"/api/projects/{pk}/undo").json()["edl"]["texts"][0]["id"] == "hook"

    edited = post_event(client, pk, "text_overlay", payload={"op": "update", "id": "hook", "text": "My own title"})
    assert next(t for t in edited["edl"]["texts"] if t["id"] == "hook")["text"] == "My own title"


def test_hook_overlay_is_the_first_sentence_capped_at_eight_words(client):
    _, _, p = new_ready_project(client)
    from app.schemas import Unit, UnitsFile, Word

    def uf(texts):
        words = [Word(id=i, idx=i, unit_id=0, start=i * 0.3, end=i * 0.3 + 0.25, text_original=t, text_display=t, conf=0.9)
                 for i, t in enumerate(texts)]
        unit = Unit(id=0, idx=0, kind="speech", start=0, end=len(texts) * 0.3, text=" ".join(texts))
        return UnitsFile(video_id="v", duration=10, video_type="vlog", source_map=[], units=[unit], words=words)

    assert hook_overlay(uf(["so", "today", "we", "are", "going", "to", "talk", "about", "microphones", "in", "depth"])).text == \
        "so today we are going to talk about"
    assert hook_overlay(uf(["Hello", "there.", "And", "more"])).text == "Hello there"  # stops at the sentence end
    assert hook_overlay(uf(["Wait,"])).text == "Wait"
    # the first sentence runs across phrases (a short "Okay," phrase on its own would make a poor title)
    two = uf(["Okay,", "so", "this", "is", "a", "test.", "Next"])
    two.units.append(Unit(id=1, idx=1, kind="speech", start=1, end=2, text="so this is a test."))
    for w in two.words[1:]:
        w.unit_id = 1
    assert hook_overlay(two).text == "Okay, so this is a test"
    assert hook_overlay(UnitsFile(video_id="v", duration=1, video_type="vlog", source_map=[], units=[], words=[])) is None
    assert p["status"] == "ready"


# --------------------------------------------------------------------------- music library + upload
def test_the_library_has_the_bundled_tracks_and_serves_them(client):
    uid, _, _ = new_ready_project(client)
    tracks = client.get(f"/api/music?user_id={uid}").json()
    lib = [t for t in tracks if t["source"] == "library"]
    assert len(lib) >= 8 and {t["id"] for t in lib} >= {LIB_TRACK}
    assert all(t["duration"] > 10 for t in lib)
    r = client.get(f"/api/music/{LIB_TRACK}/audio?user_id={uid}")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg" and len(r.content) > 10_000
    assert client.get(f"/api/music/lib_nope/audio?user_id={uid}").status_code == 404
    assert client.get("/api/music?user_id=999").status_code == 404
    licenses = (music_lib.LIB_DIR / "LICENSES.md").read_text(encoding="utf-8")
    assert all(t["id"].removeprefix("lib_") in licenses for t in lib)  # every bundled track is accounted for


def mp3_bytes() -> bytes:
    return (music_lib.LIB_DIR / "coffee_break.mp3").read_bytes()


def test_a_user_can_upload_their_own_track_and_only_they_can_use_it(client):
    uid, pk, _ = new_ready_project(client)
    r = client.post(f"/api/users/{uid}/music", files={"file": ("my song.mp3", io.BytesIO(mp3_bytes()), "audio/mpeg")})
    assert r.status_code == 201, r.text
    t = r.json()
    assert t["source"] == "upload" and t["name"] == "my song" and t["id"].startswith(f"up_{uid}_")
    assert t in client.get(f"/api/music?user_id={uid}").json()
    assert client.get(f"/api/music/{t['id']}/audio?user_id={uid}").status_code == 200
    # the same bytes again is the same track, not a duplicate
    again = client.post(f"/api/users/{uid}/music", files={"file": ("copy.mp3", io.BytesIO(mp3_bytes()), "audio/mpeg")}).json()
    assert again["id"] == t["id"]

    other = client.post("/api/users", json={"name": "Bob"}).json()["id"]
    assert t["id"] not in {x["id"] for x in client.get(f"/api/music?user_id={other}").json()}
    assert client.get(f"/api/music/{t['id']}/audio?user_id={other}").status_code == 404
    opk = upload(client, other)["id"]
    wait_ready(client, opk)
    post_event(client, opk, "music", expect=404, payload={"track_id": t["id"]})  # someone else's upload

    suggested(client, pk)
    on = post_event(client, pk, "music", payload={"track_id": t["id"]})
    assert on["edl"]["audio"]["music"]["track_id"] == t["id"]


@pytest.mark.parametrize("name,size,why", [
    ("song.ogg", 100, "only"),
    ("song.mp3", 0, "empty"),
    ("song.mp3", 2400, "readable"),
    ("song.mp3", 11 * 2**20, "10 MB"),
])
def test_bad_music_uploads_are_refused(client, name, size, why):
    data = (b"this is not audio at all" * 200)[:size] if size < 4800 else b"0" * size
    uid, _, _ = new_ready_project(client)
    r = client.post(f"/api/users/{uid}/music", files={"file": (name, io.BytesIO(data), "audio/mpeg")})
    assert r.status_code == 422 and why in r.json()["detail"]
    assert not [t for t in client.get(f"/api/music?user_id={uid}").json() if t["source"] == "upload"]


def test_track_ids_cannot_escape_the_music_folders(ctx):
    for bad in ["../x", "lib_../../etc", "up_1_zzzzzzzz", "up_1_../../x", "", "lib_", "UP_1_abcdef12"]:
        assert music_lib.track_path(ctx.data_dir, 1, bad) is None


# --------------------------------------------------------------------------- music events
def test_music_events_set_change_remove_and_undo(client):
    uid, pk, _ = new_ready_project(client)
    suggested(client, pk)
    assert get_state(client, pk)["edl"]["audio"]["music"] is None
    s = post_event(client, pk, "music", payload={"track_id": LIB_TRACK, "gain_db": -20, "duck_db": -12})
    assert s["edl"]["audio"]["music"] == {"track_id": LIB_TRACK, "gain_db": -20.0, "duck_db": -12.0}
    assert s["events"][0]["kind"] == "music" and s["events"][0]["reward"] is None  # edit-only: nothing learned
    same = post_event(client, pk, "music", payload={"track_id": LIB_TRACK, "gain_db": -20, "duck_db": -12})
    assert same["events"] == [] and "No change" in same["message"]
    louder = post_event(client, pk, "music", payload={"track_id": LIB_TRACK, "gain_db": -12})
    assert louder["edl"]["audio"]["music"]["gain_db"] == -12.0 and louder["edl"]["audio"]["music"]["duck_db"] == -10.0
    gone = post_event(client, pk, "music", payload={"track_id": None})
    assert gone["edl"]["audio"]["music"] is None
    assert client.post(f"/api/projects/{pk}/undo").json()["edl"]["audio"]["music"]["gain_db"] == -12.0
    assert client.post(f"/api/projects/{pk}/undo").json()["edl"]["audio"]["music"]["gain_db"] == -20.0
    assert client.post(f"/api/projects/{pk}/redo").json()["edl"]["audio"]["music"]["gain_db"] == -12.0
    assert uid


@pytest.mark.parametrize("payload,code", [
    ({"track_id": LIB_TRACK, "gain_db": 0}, 422),
    ({"track_id": LIB_TRACK, "gain_db": -60}, 422),
    ({"track_id": LIB_TRACK, "duck_db": 5}, 422),
    ({"track_id": LIB_TRACK, "gain_db": "loud"}, 422),
    ({"track_id": LIB_TRACK, "gain_db": True}, 422),
    ({"track_id": "lib_missing"}, 404),
    ({"track_id": "../../x"}, 404),
])
def test_bad_music_events_are_refused_and_change_nothing(client, payload, code):
    _, pk, _ = new_ready_project(client)
    suggested(client, pk)
    before = get_state(client, pk)["n_events"]
    post_event(client, pk, "music", expect=code, payload=payload)
    assert get_state(client, pk)["n_events"] == before


def test_music_values_are_validated_and_rounded():
    assert validate_music(None) is None
    assert validate_music({"track_id": "lib_a", "gain_db": -18.04}) == {"track_id": "lib_a", "gain_db": -18.0, "duck_db": -10.0}
    with pytest.raises(ValueError):
        validate_music({"gain_db": -18})


def test_render_refuses_when_the_chosen_track_has_gone(client, ctx):
    uid, pk, _ = new_ready_project(client)
    suggested(client, pk)
    t = client.post(f"/api/users/{uid}/music", files={"file": ("a.mp3", io.BytesIO(mp3_bytes()), "audio/mpeg")}).json()
    post_event(client, pk, "music", payload={"track_id": t["id"]})
    for f in music_lib.user_dir(ctx.data_dir, uid).glob("up_*.mp3"):
        f.unlink()
    r = client.post(f"/api/projects/{pk}/render", json={"preset": "reels"})
    assert r.status_code == 409 and "not available" in r.json()["detail"]


# --------------------------------------------------------------------------- remembered defaults (PLAN 8.2 / 8.6 / 8.10)
def test_choices_become_the_defaults_of_the_next_project(client):
    uid, pk, _ = new_ready_project(client)
    suggested(client, pk)
    post_event(client, pk, "setting", payload={"key": "caption_style", "value": "karaoke"})
    post_event(client, pk, "setting", payload={"key": "captions_enabled", "value": False})
    post_event(client, pk, "speed", payload={"value": 1.15})
    post_event(client, pk, "music", payload={"track_id": LIB_TRACK, "gain_db": -22})

    pk2 = upload(client, uid, ("second.mp4",))["id"]
    wait_ready(client, pk2)
    p2 = get_state(client, pk2)
    assert p2["settings"]["caption_style"] == "karaoke" and p2["settings"]["captions_enabled"] is False
    assert p2["settings"]["speed"] == 1.15
    assert p2["edl"]["audio"]["music"]["track_id"] == LIB_TRACK and p2["edl"]["audio"]["music"]["gain_db"] == -22.0
    assert p2["n_events"] == 0 and p2["can_undo"] is False  # defaults are a starting point, not events

    # an event on the new project overrides the default; the first project is not touched
    post_event(client, pk2, "setting", payload={"key": "caption_style", "value": "clean"})
    assert get_state(client, pk2)["settings"]["caption_style"] == "clean"
    assert get_state(client, pk)["settings"]["caption_style"] == "karaoke"

    bob = client.post("/api/users", json={"name": "Bob"}).json()["id"]
    bpk = upload(client, bob, ("b.mp4",))["id"]
    wait_ready(client, bpk)
    assert get_state(client, bpk)["settings"]["caption_style"] == "bold_pop"  # defaults are per user


def test_stale_defaults_are_dropped_not_fatal():
    d = sanitize_defaults({"caption_style": "gone", "speed": 1.15, "bogus": 1, "music": {"track_id": "lib_x", "gain_db": 5},
                           "captions_enabled": "yes", "audio_enhance": False})
    assert d == {"speed": 1.15, "audio_enhance": False}
    assert sanitize_defaults(None) == {}


# --------------------------------------------------------------------------- caption word ids (emphasis clicks)
def test_caption_word_ids_line_up_with_the_caption_words(client):
    _, pk, _ = new_ready_project(client)
    s = suggested(client, pk)
    lines, ids = s["edl"]["captions"]["lines"], s["caption_word_ids"]
    assert len(ids) == len(lines) > 0
    assert all(len(i) == len(line["words"]) for i, line in zip(ids, lines, strict=True))
    flat = [w for line in ids for w in line]
    assert flat == sorted(flat) and len(set(flat)) == len(flat)


def test_clicking_a_caption_word_sets_and_clears_the_emphasis(client):
    _, pk, _ = new_ready_project(client)
    s = suggested(client, pk)
    emph = next(x for x in s["suggestions"] if x["decision_type"] == "emphasis" and x["candidate_word_ids"])
    cand = emph["candidate_word_ids"][0]
    line = next(i for i in s["caption_word_ids"] if cand in i)
    li = s["caption_word_ids"].index(line)
    if emph["current_action"] != "NONE":  # start from a line without emphasis
        post_event(client, pk, "manual", edit={"kind": "set_emphasis", "word_id": None, "line_word_ids": line})
    out = post_event(client, pk, "manual", edit={"kind": "set_emphasis", "word_id": cand, "line_word_ids": line})
    assert [e["kind"] for e in out["events"]] == ["override"] and out["events"][0]["payload"]["via"] == "manual"
    assert out["edl"]["captions"]["lines"][li]["words"][line.index(cand)]["emphasis"] is True
    cleared = post_event(client, pk, "manual", edit={"kind": "set_emphasis", "word_id": None, "line_word_ids": line})
    assert [e["kind"] for e in cleared["events"]] == ["override"]
    assert not any(w["emphasis"] for w in cleared["edl"]["captions"]["lines"][li]["words"])
    # a word that is not one of the line's candidates is applied but not learned
    other = next(w for w in line if w not in emph["candidate_word_ids"])
    free = post_event(client, pk, "manual", edit={"kind": "set_emphasis", "word_id": other, "line_word_ids": line})
    assert [e["kind"] for e in free["events"]] == ["fine_cut"]
    assert free["edl"]["captions"]["lines"][li]["words"][line.index(other)]["emphasis"] is True


# --------------------------------------------------------------------------- global nudges
def test_a_nudge_is_one_edit_only_event_that_re_scores_untouched_suggestions(client):
    _, pk, _ = new_ready_project(client)
    s = suggested(client, pk)
    out = post_event(client, pk, "nudge", payload={"name": "snappier", "magnitude": 3.0})
    assert [e["kind"] for e in out["events"]] == ["nudge"] and out["events"][0]["reward"] is None
    # a gentle push: one click moves at most a quarter of each kind of decision (and never leaves a video empty)
    import math
    from collections import Counter
    per_type = Counter(x["decision_type"] for x in s["suggestions"])
    moved = Counter(c["decision_type"] for c in out["changes"])
    assert all(n <= max(1, math.ceil(0.25 * per_type[dt])) for dt, n in moved.items()), (moved, per_type)
    assert out["edl"]["clips"]
    assert out["changes"] and all(c["decision_type"] in ("pause", "phrase", "filler") for c in out["changes"])
    assert {x["source"] for x in out["suggestions"] if x["id"] in {c["new_id"] for c in out["changes"]}} == {"nudge"}
    post_event(client, pk, "nudge", expect=422, payload={"name": "wilder"})
    post_event(client, pk, "nudge", expect=422, payload={"name": "snappier", "magnitude": 9})
    undone = client.post(f"/api/projects/{pk}/undo").json()
    assert {x["id"] for x in undone["suggestions"]} == {x["id"] for x in s["suggestions"]}


# --------------------------------------------------------------------------- autopilot / trust (PLAN 8.8)
def autopilot_project(client, uid, name="a.mp4"):
    """Upload a batch video and wait until it is analysed AND auto-edited (the edit follows the analysis)."""
    pk = upload(client, uid, (name,), autopilot="true")["id"]
    wait_ready(client, pk)
    for _ in range(300):
        p = get_state(client, pk)
        if p["suggestions"]:
            return pk, p
        time.sleep(0.05)
    raise TimeoutError("no autopilot suggestions")


def test_a_batch_video_is_auto_edited_greedily_and_labelled(client, ctx):
    uid = client.post("/api/users", json={"name": "Ana"}).json()["id"]
    _pk, p = autopilot_project(client, uid)
    assert p["autopilot"] is True
    assert p["suggestions"], "suggestions are made as soon as the analysis is done"
    assert {s["source"] for s in p["suggestions"]} == {"autopilot"}
    assert all(s["propensity"] == 1.0 and s["explored"] is False for s in p["suggestions"])
    # no trust yet (nothing finalized): a batch video can only "need review"
    assert p["badge"] == "review" and p["n_review"] == len(p["queue"]["needs_review"])
    assert p["expected_corrections"] == pytest.approx(sum(1 - s["confidence"] for s in p["suggestions"]), abs=0.01)
    assert client.get(f"/api/projects?user_id={uid}").json()[0]["badge"] == "review"
    # a normal project of the same user is not an autopilot one
    npk = upload(client, uid, ("n.mp4",))["id"]
    n = wait_ready(client, npk)
    assert n["autopilot"] is False and n["badge"] is None and n["suggestions"] == []


def set_confidence(ctx, pk, conf):
    with ctx.session() as s:
        for r in s.exec(select(SuggestionRow).where(SuggestionRow.video_id == pk)).all():
            r.confidence = conf
            s.add(r)
        s.commit()


def add_metric(ctx, uid, index, reviewed, accepted, via_autopilot=False):
    with ctx.session() as s:
        v = Video(user_id=uid, title=f"m{index}", status="edited")
        s.add(v)
        s.commit()
        s.add(Metric(user_id=uid, video_id=v.id, video_index=index, n_reviewed=reviewed, n_accepted=accepted,
                     exported_via_autopilot=via_autopilot))
        s.commit()


def test_trust_is_the_acceptance_rate_of_the_last_two_reviewed_videos(client, ctx):
    uid = client.post("/api/users", json={"name": "Ana"}).json()["id"]
    assert client.get(f"/api/users/{uid}/profile").json()["trust"] is None
    add_metric(ctx, uid, 1, 10, 2)  # an old, bad video: outside the window once two newer ones exist
    add_metric(ctx, uid, 2, 20, 18)
    add_metric(ctx, uid, 3, 10, 9)
    add_metric(ctx, uid, 4, 0, 0)  # finalized without looking at anything: says nothing
    add_metric(ctx, uid, 5, 50, 0, via_autopilot=True)  # exported by autopilot: ignored
    prof = client.get(f"/api/users/{uid}/profile").json()
    assert prof["trust"] == pytest.approx(27 / 30) and prof["trust_threshold"] == 0.85


def test_ready_to_post_needs_few_expected_corrections_and_trust(client, ctx):
    uid = client.post("/api/users", json={"name": "Ana"}).json()["id"]
    pk, _ = autopilot_project(client, uid)
    set_confidence(ctx, pk, 0.999)
    add_metric(ctx, uid, 1, 20, 19)  # trust 0.95
    p = get_state(client, pk)
    assert p["badge"] == "ready" and p["expected_corrections"] < 3
    # a lower trust, or more uncertainty, means review
    add_metric(ctx, uid, 2, 20, 10)  # last two videos: 29/40 = 0.725
    assert get_state(client, pk)["badge"] == "review"
    add_metric(ctx, uid, 3, 20, 20)
    add_metric(ctx, uid, 4, 20, 20)
    assert get_state(client, pk)["badge"] == "ready"
    set_confidence(ctx, pk, 0.5)
    q = get_state(client, pk)
    assert q["badge"] == "review" and q["n_review"] > 0 and q["expected_corrections"] > 3


def test_the_badge_goes_away_once_the_video_has_been_worked_on(client, ctx):
    uid = client.post("/api/users", json={"name": "Ana"}).json()["id"]
    pk, p = autopilot_project(client, uid)
    sug = next(s for s in p["suggestions"] if s["decision_type"] == "phrase")
    post_event(client, pk, "accept", suggestion_id=sug["id"])
    after = get_state(client, pk)
    assert after["status"] == "edited" and after["badge"] is None  # worked on: the normal status badge takes over
    assert after["n_review"] == 0


def test_an_unreviewed_autopilot_video_leaves_no_learning_signal(client, ctx):
    uid = client.post("/api/users", json={"name": "Ana"}).json()["id"]
    pk, _ = autopilot_project(client, uid)
    n_before = client.get(f"/api/users/{uid}/profile").json()["n_learning_events"]
    fin = client.post(f"/api/projects/{pk}/finalize", json={"seconds_active": 0}).json()
    assert fin["implicit_events"] == 0  # nothing was seen, so nothing is "untouched and seen"
    prof = client.get(f"/api/users/{uid}/profile").json()
    assert prof["n_learning_events"] == n_before == 0
    assert prof["trust"] is None  # n_reviewed == 0: not part of the trust score


def test_at_most_five_batch_videos_can_wait_at_once(tmp_path):
    gate = threading.Event()

    def slow_ingest(inputs, video_id, video_type, out_dir):
        gate.wait(20)
        return fake_ingest(inputs, video_id, video_type, out_dir)

    ctx = AppContext(tmp_path / "data", ingest_fn=slow_ingest)
    try:
        with TestClient(create_app(ctx)) as client:
            uid = client.post("/api/users", json={"name": "Ana"}).json()["id"]
            for i in range(5):
                upload(client, uid, (f"b{i}.mp4",), autopilot="true")
            files = [("files", ("six.mp4", io.BytesIO(b"x video"), "video/mp4"))]
            r = client.post("/api/projects", files=files, data={"user_id": str(uid), "autopilot": "true"})
            assert r.status_code == 429 and "at most 5" in r.json()["detail"]
            # an ordinary upload is not limited by the batch queue
            upload(client, uid, ("normal.mp4",))
            gate.set()
    finally:
        gate.set()
        ctx.close()


# --------------------------------------------------------------------------- the ffmpeg graph
def _mini_edl():
    from app.schemas import EDL, Clip, Point

    clips = [Clip(src_start=0, src_end=1, zoom=1.0, crop_center=Point(x=0.5, y=0.5), unit_ids=[0]),
             Clip(src_start=2, src_end=3.5, zoom=1.0, crop_center=Point(x=0.5, y=0.5), unit_ids=[1])]
    return EDL(video_id="v", clips=clips, duration=2.5)


def test_music_graph_structure():
    edl, src = _mini_edl(), graph.SourceInfo(1920, 1080)
    plain, _, _ = graph.build_filter_complex(edl, src, audio_chain=["highpass=f=80"], post_chain=["loudnorm=I=-14"],
                                             video=False)
    assert plain.splitlines()[-1] == "[ac]highpass=f=80,loudnorm=I=-14[ao]"  # post_chain continues the chain
    mix = graph.MusicMix(gain_db=-18, duck_db=-10, duration=2.5)
    g, _, label = graph.build_filter_complex(edl, src, audio_chain=["highpass=f=80"], post_chain=["loudnorm=I=-14"],
                                             video=False, music=mix)
    assert label == "ao" and "[ac]highpass=f=80[vox]" in g
    assert "[1:a]" in g and "atrim=0:2.5000" in g and "volume=-18.0dB" in g
    assert "afade=t=in:st=0:d=1.000" in g and "afade=t=out:st=1.5000:d=1.000" in g
    assert "sidechaincompress=threshold=0.0301:ratio=8.0:attack=20:release=300" in g
    assert "amix=inputs=2:duration=first:normalize=0[mix]" in g
    assert g.splitlines()[-1] == "[mix]loudnorm=I=-14[ao]"
    nod, _, _ = graph.build_filter_complex(edl, src, video=False, music=graph.MusicMix(-18, 0, 2.5))
    assert "sidechaincompress" not in nod and "anullsink" in nod  # no ducking: the voice key is just dropped


def test_short_clips_get_short_fades_and_ducking_depth_is_monotone():
    g = graph.music_chain(graph.MusicMix(-18, -10, 1.0))
    assert any("afade=t=in:st=0:d=0.500" in line and "afade=t=out:st=0.5000:d=0.500" in line for line in g)
    ratios = [graph.duck_ratio(d) for d in (-2, -6, -10, -16, -24, -40)]
    assert ratios == sorted(ratios) and ratios[2] == 8.0 and ratios[0] >= 1.5 and ratios[-1] <= 20
    thresholds = [graph.duck_threshold(d) for d in (-2, -6, -10, -16, -24)]
    assert thresholds == sorted(thresholds, reverse=True) and thresholds[2] == 0.0301  # deeper duck, lower threshold
    assert 0.001 <= min(thresholds) and max(thresholds) <= 1.0  # inside what ffmpeg accepts


@pytest.mark.skipif(not config.ffmpeg_path(), reason="ffmpeg missing")
@pytest.mark.parametrize("duck,min_drop", [(-6, 1.5), (-12, 6.0), (-20, 10.0)])
def test_the_music_really_ducks_under_the_voice(duck, min_drop, tmp_path):
    """Run the music part of the graph on a synthetic voice (1 s bursts every 2 s) and measure the music level."""
    lines = graph.music_chain(graph.MusicMix(gain_db=-12, duck_db=duck, duration=8))
    key = "aevalsrc='if(lt(mod(t,2),1),0.1*sin(2*PI*220*t),0)':s=48000:d=8,aformat=channel_layouts=stereo[vkey]"
    g = ";".join([key, *lines[1:-1]])  # the music part only (lines[0] splits the voice, lines[-1] mixes): out = [musd]
    (tmp_path / "g.txt").write_text(g, encoding="utf-8")
    p = subprocess.run([config.ffmpeg_path(), "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:d=1",
                        "-f", "lavfi", "-i", "sine=f=440:d=8:r=48000", "-/filter_complex", str(tmp_path / "g.txt"),
                        "-map", "[musd]", "-f", "s16le", "-ac", "1", "-ar", "48000", "-"], capture_output=True, check=False)
    assert p.returncode == 0, p.stderr.decode()
    x = np.frombuffer(p.stdout, dtype="<i2").astype(float) / 32768
    rms = lambda a, b: 20 * np.log10(np.sqrt(np.mean(x[int(a * 48000):int(b * 48000)] ** 2)) + 1e-9)
    speaking = np.mean([rms(0.4 + 2 * k, 0.9 + 2 * k) for k in range(1, 4)])  # after the attack, inside a burst
    silent = np.mean([rms(1.5 + 2 * k, 1.95 + 2 * k) for k in range(1, 3)])  # after the release, in a gap
    assert silent - speaking >= min_drop, (speaking, silent)
