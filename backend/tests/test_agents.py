"""The product's full decision set: rhythm, pace, hook, music timing, crew chatter, rating, queue caps, progress."""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from app.edl.build import Decisions, build_edl
from app.edl.remap import Remap
from app.pipeline import crew
from app.render.graph import SourceInfo, build_filter_complex
from app.rl.rhythm import ARMS, Episode, RhythmAgent, Step, episodes_from_log, phi
from app.schemas import EDL, AudioSpec, Clip, MusicSpec, SfxCue, Unit
from sim.synth import make_video


def test_remap_with_clip_paces():
    r = Remap([(0.0, 2.0), (4.0, 6.0)], 1.0, [1.0, 2.0])
    assert r.duration == pytest.approx(3.0)
    assert r.src_to_out(5.0) == pytest.approx(2.5)
    assert r.out_to_src(2.5) == pytest.approx(5.0)
    assert r.src_to_out(3.0, "prev") == pytest.approx(2.0)


def _ctx(since=30.0, last=0, central=0.5):
    return {"hist_since_visual": since, "hist_last_visual": float(last), "hist_streak": 0.0, "topic_centrality": central,
            "duration": 3.0, "loudness_z": 0.0}


def test_rhythm_agent_learns_a_state_dependent_preference():
    agent = RhythmAgent(seed=1, n_draws=2)
    for _ in range(25):
        ep = Episode()
        for since in (1.0, 2.0, 9.0, 11.0):
            want = "Z130" if since > 5 else "NONE"
            ep.steps.append(Step(_ctx(since), "NONE", 1.0 if want == "NONE" else -1.0, want))
        agent.add_episode(ep)
    agent.begin_video()
    assert agent.mean_action(_ctx(10.0)) == "Z130" and agent.mean_action(_ctx(1.5)) == "NONE"
    c = agent.choose(_ctx(10.0))
    assert c.action in ARMS and 0 < c.propensity <= 1 and abs(sum(c.probs.values()) - 1) < 1e-6


def test_rating_is_carried_back_as_a_delayed_reward():
    def trained(rating, gamma):
        agent = RhythmAgent(seed=1, gamma=gamma, n_draws=0, rating_share=0.0)  # only on the last step
        for _ in range(20):
            agent.add_episode(Episode([Step(_ctx(9.0), "Z115"), Step(_ctx(1.0), "NONE")], rating=rating))
        return agent.draw(perturb=False)[ARMS.index("Z115")] @ phi(_ctx(9.0))

    assert trained(1.0, 0.9) > trained(-1.0, 0.9) + 0.5  # the first step's value depends on the rating of the last
    assert trained(1.0, 0.0) == pytest.approx(trained(-1.0, 0.0))  # a bandit cannot see it


def test_episodes_from_log_reads_feedback_and_rating():
    row = lambda i, v, a: SimpleNamespace(id=i, video_id=v, proposed_action=a, context_raw_json=_ctx())
    ev = lambda k, sid, p=None, v=1: SimpleNamespace(kind=k, suggestion_id=sid, payload_json=p or {}, video_id=v)
    eps = episodes_from_log([row(1, 1, "NONE"), row(2, 1, "Z115"), row(3, 1, "NONE")],
                            [ev("accept", 1), ev("override", 2, {"final_action": "SFX"}), ev("rating", None, {"value": -1})])
    s = eps[0].steps
    assert (s[0].reward, s[1].reward, s[1].final, s[2].reward, eps[0].rating) == (1.0, -1.0, "SFX", None, -1.0)


def _u(i, text, start, end, z=0.0):
    return Unit(id=i, idx=i, kind="speech", start=start, end=end, text=text, features={"loudness_z": z})


def test_crew_chatter_scores():
    us = [_u(0, "This is what editing looks like.", 0, 3, .5),
          _u(1, "Most of it will be voiceover because we'll create it later, okay?", 13, 17, -1.4),
          _u(2, "Client wanted a clean edit.", 25, 28, 0),
          _u(3, "You can add a voiceover to make your video better, check it out.", 29, 33, 0),
          _u(4, "This is for YouTube Shorts.", 40, 42, -.3)]
    crew.annotate(us, [])
    got = [u.features["crew_score"] >= crew.THRESHOLD for u in us]
    assert got == [False, True, False, False, True]


def test_like_with_meaning_is_not_a_filler():
    from app.pipeline import fillers as fl
    from app.schemas import Word

    def find(text):
        ws = [Word(id=i, idx=i, unit_id=-1, start=i * .3, end=i * .3 + .2, text_original=t, text_display=t)
              for i, t in enumerate(text.split())]
        return fl.find_fillers(ws, [(0, len(ws))])

    assert find("this is what it looks like to edit") == []
    assert find("we create it like the other") == []
    assert [g.token for g in find("what you, like, want to see")] == ["like"]


def test_graph_with_pace_sfx_and_delayed_music():
    edl = EDL(video_id="v", clips=[Clip(src_start=0, src_end=2), Clip(src_start=3, src_end=5, speed=1.25)], duration=3.6,
              audio=AudioSpec(music=MusicSpec(track_id="x", start_s=1.5), sfx=[SfxCue(t0=2.0)]))
    from app.render.graph import MusicMix

    g, _, _ = build_filter_complex(edl, SourceInfo(1920, 1080), music=MusicMix(-18, -10, 3.6, 1.5))
    assert "atempo=1.250000" in g and "/1.250000" in g and "anoisesrc" in g and "adelay=1500" in g and "[vox2]" in g


def test_api_full_flow_with_agents(tmp_path):
    from fastapi.testclient import TestClient
    from sqlmodel import select  # noqa: F401

    from app.appctx import AppContext
    from app.main import create_app
    from tests.apihelpers import fake_ingest, upload, wait_ready

    app = create_app(AppContext(tmp_path / "data", ingest_fn=fake_ingest, agents=True))
    with TestClient(app) as c:
        uid = c.post("/api/users", json={"name": "A"}).json()["id"]
        pk = upload(c, uid)["id"]
        wait_ready(c, pk)
        st = c.post(f"/api/projects/{pk}/suggest").json()
        kinds = Counter(s["decision_type"] for s in st["suggestions"])
        assert kinds["rhythm"] > 0 and kinds["pace"] > 0 and kinds["hook"] == 1 and kinds["music_timing"] == 1 and kinds["zoom"] == 0
        by_id = {s["id"]: s for s in st["suggestions"]}
        needs = Counter(by_id[i]["decision_type"] for i in st["queue"]["needs_review"])
        assert needs["emphasis"] <= 2 and needs["pace"] <= 2 and needs["rhythm"] <= 4 and needs["music_timing"] == 0

        hook = next(s for s in st["suggestions"] if s["decision_type"] == "hook")
        assert hook["candidate_texts"]
        out = c.post(f"/api/projects/{pk}/events", json={"kind": "reject", "suggestion_id": hook["id"]}).json()
        assert not [t for t in out["edl"]["texts"] if t["id"] == "hook"]
        assert c.post(f"/api/projects/{pk}/undo").json()["edl"]["texts"][0]["id"] == "hook"

        beat = next(s for s in st["suggestions"] if s["decision_type"] == "rhythm")
        out = c.post(f"/api/projects/{pk}/events", json={"kind": "override", "suggestion_id": beat["id"], "final_action": "TEXT_POP"}).json()
        assert any(t["id"] == f"pop_{beat['target_id']}" for t in out["edl"]["texts"])
        out = c.post(f"/api/projects/{pk}/events", json={"kind": "override", "suggestion_id": beat["id"], "final_action": "SFX"}).json()
        assert out["edl"]["audio"]["sfx"]

        pace = next(s for s in st["suggestions"] if s["decision_type"] == "pace")
        before = out["edl"]["duration"]
        out = c.post(f"/api/projects/{pk}/events", json={"kind": "override", "suggestion_id": pace["id"], "final_action": "S125"}).json()
        assert out["edl"]["duration"] < before and any(cl["speed"] == 1.25 for cl in out["edl"]["clips"])

        ex = c.get(f"/api/projects/{pk}/suggestions/{beat['id']}/explain").json()
        assert ex["text"]
        assert c.post(f"/api/projects/{pk}/finalize", json={}).status_code == 200
        assert c.post(f"/api/projects/{pk}/events", json={"kind": "rating", "payload": {"value": 1}}).status_code == 200
        assert c.get(f"/api/projects/{pk}").json()["rating"] == 1
        assert c.post(f"/api/projects/{pk}/events", json={"kind": "rating", "payload": {"value": 5}}).status_code == 422

        # the next video learns from the first without errors
        pk2 = upload(c, uid)["id"]
        wait_ready(c, pk2)
        assert c.post(f"/api/projects/{pk2}/suggest").status_code == 200
        assert c.post(f"/api/projects/{pk}/events", json={"kind": "nudge", "payload": {"name": "more_zooms", "magnitude": 2}}).status_code == 200


def test_decisions_default_build_unchanged():
    uf = make_video(3, "vlog", index=1).uf
    edl = build_edl(uf, Decisions())
    assert all(c.speed == 1.0 for c in edl.clips) and edl.audio.sfx == [] and np.isfinite(edl.duration)
