"""API: undo / redo, seen + finalize, persistence across restarts, concurrency, failure atomicity, render guards."""
from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from app.appctx import AppContext
from app.main import create_app
from app.models import EventRow, Metric, SuggestionRow, Video
from app.svc import events as svc_events
from tests.apihelpers import (
    assert_policy_consistent,
    fake_ingest,
    new_ready_project,
    pick,
    post_event,
    upload,
    wait_ready,
)
from tests.test_api_events import _suitable, find, n_updates, refresh


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
    pytest.fail("no suitable synthetic video")


def digest(state: dict) -> dict:
    """Everything a user can see of the edit: the EDL, the active suggestions with their status, the queue, the style."""
    return {"edl": state["edl"], "settings": state["settings"],
            "suggestions": sorted((x["id"], x["decision_type"], x["target_id"], x["current_action"], x["status"])
                                  for x in state["suggestions"]),
            "queue": state["queue"], "style": state["style_mixture"], "n": state["n_events"]}


def db_events(ctx, pk):
    with ctx.session() as s:
        return list(s.exec(select(EventRow).where(EventRow.video_id == pk).order_by(EventRow.id)).all())


# ---------------------------------------------------------------- undo / redo
def run_session(proj) -> list[dict]:
    """A realistic mixed bag of actions; returns the state after each request (index 0 = right after suggest)."""
    c, pk, st = proj.client, proj.pk, proj.state
    states = [st]

    def do(kind, **body):
        out = post_event(c, pk, kind, **body)
        states.append(out)
        return out

    s = pick(st, "pause", current_action="REMOVE")
    do("accept", suggestion_id=s["id"])
    cur = refresh(proj)
    s = pick(cur, "pause", current_action="REMOVE")
    do("override", suggestion_id=s["id"], final_action="TIGHT_40")
    cur = states[-1]
    s = next(x for x in cur["suggestions"] if x["decision_type"] == "phrase" and x["current_action"] == "KEEP")
    ids = [w["id"] for w in proj.words if w["unit_id"] == s["target_id"]]
    do("manual", edit={"kind": "delete_words", "word_ids": ids})
    cur = states[-1]
    long_unit = next(u for u in proj.units if u["kind"] == "speech" and len([w for w in proj.words if w["unit_id"] == u["id"]]) >= 8
                     and find(cur, {"decision_type": "phrase", "target_id": u["id"]})["current_action"] == "KEEP")
    w = [w for w in proj.words if w["unit_id"] == long_unit["id"]][3]
    do("manual", edit={"kind": "delete_words", "word_ids": [w["id"]]})
    do("speed", payload={"value": 1.15})
    do("setting", payload={"key": "caption_style", "value": "karaoke"})
    do("text_overlay", payload={"op": "add", "text": "hook", "t0": 0, "t1": 2})
    do("caption_text", payload={"word_id": ids[0] if False else proj.words[5]["id"], "text": "Edited"})
    do("nudge", payload={"name": "snappier", "magnitude": 3.0})
    cur = states[-1]
    s = next(x for x in cur["suggestions"] if x["decision_type"] == "zoom" and x["status"] == "pending")
    do("reject", suggestion_id=s["id"])
    cur = states[-1]
    s = next(x for x in cur["suggestions"] if x["decision_type"] == "phrase" and x["status"] == "pending")
    do("reject", suggestion_id=s["id"])
    return states


def test_undo_and_redo_walk_the_whole_history_exactly(proj):
    c, pk = proj.client, proj.pk
    states = run_session(proj)
    n = len(states) - 1
    assert n >= 10 and n_updates(proj) > 0
    assert_policy_consistent(proj.ctx, proj.uid)
    # undo one request at a time: after k undos the state is exactly the state after (n - k) requests
    for k in range(1, n + 1):
        out = c.post(f"/api/projects/{pk}/undo")
        assert out.status_code == 200, out.text
        out = out.json()
        assert digest(out) == digest(states[n - k]), f"undo step {k}"
        assert out["can_redo"] and out["can_undo"] == (k < n)
        assert_policy_consistent(proj.ctx, proj.uid)
    assert n_updates(proj) == 0 and refresh(proj)["style_mixture"]["n_events"] == 0
    assert c.post(f"/api/projects/{pk}/undo").status_code == 409
    for k in range(1, n + 1):
        out = c.post(f"/api/projects/{pk}/redo")
        assert out.status_code == 200, out.text
        assert digest(out.json()) == digest(states[k]), f"redo step {k}"
        assert_policy_consistent(proj.ctx, proj.uid)
    assert c.post(f"/api/projects/{pk}/redo").status_code == 409
    assert digest(refresh(proj)) == digest(states[n])  # GET agrees with what undo/redo returned


def test_a_new_event_after_undo_drops_the_redo_branch(proj):
    c, pk, st = proj.client, proj.pk, proj.state
    a, b = pick(st, "pause", nth=0), pick(st, "pause", nth=1)
    post_event(c, pk, "accept", suggestion_id=a["id"])
    cur = post_event(c, pk, "accept", suggestion_id=find(refresh(proj), b)["id"])
    undone = c.post(f"/api/projects/{pk}/undo").json()
    assert undone["can_redo"]
    post_event(c, pk, "speed", payload={"value": 1.1})
    assert not refresh(proj)["can_redo"] and c.post(f"/api/projects/{pk}/redo").status_code == 409
    rows = db_events(proj.ctx, pk)
    assert [(e.kind, e.undone) for e in rows] == [("accept", False), ("accept", True), ("speed", False)]
    assert cur["n_events"] == 2
    assert_policy_consistent(proj.ctx, proj.uid)


def test_undo_never_touches_the_implicit_events_of_finalize(proj):
    c, pk, st = proj.client, proj.pk, proj.state
    post_event(c, pk, "accept", suggestion_id=pick(st, "pause")["id"])
    c.post(f"/api/projects/{pk}/seen", json={"unit_ids": [u["id"] for u in proj.units][:30]})
    fin = c.post(f"/api/projects/{pk}/finalize").json()
    assert fin["implicit_events"] > 0
    undone = c.post(f"/api/projects/{pk}/undo").json()  # undoes the accept, not the implicit events
    assert not undone["can_undo"]
    kinds = [(e.kind, e.undone) for e in db_events(proj.ctx, pk)]
    assert kinds[0] == ("accept", True) and all(k == ("implicit", False) for k in kinds[1:])
    assert_policy_consistent(proj.ctx, proj.uid)


def test_undo_restores_suggestions_replaced_by_rescoring(proj):
    c, pk = proj.client, proj.pk
    base = digest(proj.state)
    state = proj.state
    n = 0
    for _ in range(12):
        x = next((x for x in state["suggestions"] if x["decision_type"] == "pause" and x["current_action"] == "REMOVE"
                  and x["status"] == "pending"), None)
        if x is None:
            break
        state = post_event(c, pk, "override", suggestion_id=x["id"], final_action="KEEP")
        n += 1
    assert any(e.payload_json for e in db_events(proj.ctx, pk))
    for _ in range(n):
        out = c.post(f"/api/projects/{pk}/undo").json()
    assert digest(out) == base  # the very same suggestion rows are active again


# ---------------------------------------------------------------- seen + finalize
def test_seen_validation(proj):
    c, pk = proj.client, proj.pk
    assert c.post(f"/api/projects/{pk}/seen", json={"unit_ids": [10**6]}).status_code == 422
    assert c.post(f"/api/projects/{pk}/seen", json={"unit_ids": []}).status_code == 422
    assert c.post("/api/projects/999/seen", json={"unit_ids": [0]}).status_code == 404
    r = c.post(f"/api/projects/{pk}/seen", json={"unit_ids": [0, 0, 1]}).json()
    assert r == {"marked": 2}
    assert refresh(proj)["seen_unit_ids"] == [0, 1]
    assert c.post(f"/api/projects/{pk}/seen", json={"unit_ids": [1]}).json() == {"marked": 1}  # idempotent


def test_finalize_makes_implicit_accepts_only_for_seen_untouched_suggestions(proj):
    c, pk, st = proj.client, proj.pk, proj.state
    touched = pick(st, "pause")
    post_event(c, pk, "override", suggestion_id=touched["id"], final_action="KEEP" if touched["current_action"] != "KEEP" else "REMOVE")
    cur = refresh(proj)
    seen_units = [u["id"] for u in proj.units][: len(proj.units) // 2]
    c.post(f"/api/projects/{pk}/seen", json={"unit_ids": seen_units})
    cur = refresh(proj)
    expect = [x for x in cur["suggestions"] if x["status"] == "pending" and x["seen"]]
    assert expect and len(expect) < len([x for x in cur["suggestions"] if x["status"] == "pending"])
    before = n_updates(proj)
    fin = c.post(f"/api/projects/{pk}/finalize", json={"seconds_active": 321.5})
    assert fin.status_code == 200, fin.text
    fin = fin.json()
    assert fin["implicit_events"] == len(expect)
    rows = [e for e in db_events(proj.ctx, pk) if e.kind == "implicit"]
    assert sorted(e.suggestion_id for e in rows) == sorted(x["id"] for x in expect)
    assert all(e.reward == 0.5 and e.weight == 1.0 and e.payload_json == {} for e in rows)
    assert n_updates(proj) == before + len(expect)
    m = fin["metrics"]
    assert m["video_index"] == 1 and m["n_decisions"] == len(cur["suggestions"]) and m["seconds_active"] == 321.5
    assert m["n_corrected"] == 1 and m["n_accepted"] == len(expect)
    assert m["n_reviewed"] == len([x for x in cur["suggestions"] if x["status"] != "pending" or x["seen"]])
    assert m["acceptance"] == pytest.approx(m["n_accepted"] / m["n_reviewed"])
    with proj.ctx.session() as s:
        row = s.exec(select(Metric).where(Metric.video_id == pk)).one()
        assert row.n_accepted == m["n_accepted"] and row.exported_via_autopilot is False
    assert all(x["status"] == "accepted" for x in fin["suggestions"] if x["id"] in {e.suggestion_id for e in rows})
    assert_policy_consistent(proj.ctx, proj.uid)
    # finalizing again changes nothing (idempotent) but refreshes the metric row
    again = c.post(f"/api/projects/{pk}/finalize", json={"seconds_active": 400}).json()
    assert again["implicit_events"] == 0 and again["metrics"]["seconds_active"] == 400
    assert n_updates(proj) == before + len(expect)
    with proj.ctx.session() as s:
        assert len(s.exec(select(Metric).where(Metric.video_id == pk)).all()) == 1


def test_unseen_suggestions_give_no_signal_at_finalize(proj):
    fin = proj.client.post(f"/api/projects/{proj.pk}/finalize").json()
    assert fin["implicit_events"] == 0 and fin["metrics"]["n_reviewed"] == 0 and fin["metrics"]["acceptance"] == 0.0
    assert n_updates(proj) == 0 and fin["status"] == "edited"


def test_an_explicit_decision_replaces_the_implicit_one(proj):
    c, pk = proj.client, proj.pk
    c.post(f"/api/projects/{pk}/seen", json={"unit_ids": [u["id"] for u in proj.units]})
    fin = c.post(f"/api/projects/{pk}/finalize").json()
    assert fin["implicit_events"] > 10
    s = pick(refresh(proj), "pause", status="accepted")
    n0 = n_updates(proj)
    post_event(c, pk, "override", suggestion_id=s["id"], final_action="KEEP" if s["current_action"] != "KEEP" else "REMOVE")
    imp = [e for e in db_events(proj.ctx, pk) if e.kind == "implicit" and e.suggestion_id == s["id"]]
    assert len(imp) == 1 and imp[0].undone  # the 'untouched' evidence no longer holds
    assert n_updates(proj) == n0 - 1 + 2
    assert_policy_consistent(proj.ctx, proj.uid)
    assert c.post(f"/api/projects/{pk}/finalize").json()["implicit_events"] == 0  # corrected now, not untouched


def test_video_index_counts_a_users_finalized_videos(client, ctx):
    uid, p1, _ = new_ready_project(client)
    p2 = upload(client, uid, ("second.mp4",))["id"]
    wait_ready(client, p2)
    for pk in (p1, p2):
        client.post(f"/api/projects/{pk}/suggest")
    assert client.post(f"/api/projects/{p2}/finalize").json()["metrics"]["video_index"] == 1
    assert client.post(f"/api/projects/{p1}/finalize").json()["metrics"]["video_index"] == 2
    assert client.post(f"/api/projects/{p2}/finalize").json()["metrics"]["video_index"] == 1  # stays


# ---------------------------------------------------------------- persistence
def test_state_survives_an_application_restart(tmp_path):
    data = tmp_path / "data"
    ctx1 = AppContext(data, ingest_fn=fake_ingest)
    with TestClient(create_app(ctx1)) as c1:
        uid, pk = None, None
        for i in range(80):
            uid = c1.post("/api/users", json={"name": f"U{i}"}).json()["id"]
            pk = upload(c1, uid, (f"clip{i}.mp4",))["id"]
            p = wait_ready(c1, pk)
            s = c1.post(f"/api/projects/{pk}/suggest").json()
            if _suitable(s, p):
                break
        proj = type("P", (), {"client": c1, "ctx": ctx1, "uid": uid, "pk": pk, "units": p["units"], "words": p["words"],
                              "state": s})()
        run_session(proj)
        before = digest(refresh(proj))
        scores_before = {}
        with ctx1.session() as se:
            learner = ctx1.policies.get(se, uid)
            row = se.exec(select(SuggestionRow)).first()
            scores_before = learner.policy.mean_scores(row.decision_type, row.context_raw_json)
    ctx1.close()

    ctx2 = AppContext(data, ingest_fn=fake_ingest)
    try:
        rebuilt = []
        orig = ctx2.policies._rebuild
        ctx2.policies._rebuild = lambda *a, **k: (rebuilt.append(1), orig(*a, **k))[1]  # type: ignore[method-assign]
        with TestClient(create_app(ctx2)) as c2:
            assert digest(c2.get(f"/api/projects/{pk}").json()) == before
            with ctx2.session() as se:
                row = se.exec(select(SuggestionRow)).first()
                learner = ctx2.policies.get(se, uid)
                assert learner.policy.mean_scores(row.decision_type, row.context_raw_json) == pytest.approx(scores_before)
            assert rebuilt == []  # restored from the policy cache, not replayed
            assert_policy_consistent(ctx2, uid)
            # a stale cache (events changed behind its back) is detected and rebuilt from the log
            with ctx2.session() as se:  # tamper with the log behind the cache's back
                e = se.exec(select(EventRow).where(EventRow.kind == "override")).first()
                e.undone = True
                se.add(e)
                se.commit()
            n_before = len(rebuilt)
            with ctx2.session() as se:
                fixed = ctx2.policies.get(se, uid)  # fingerprint mismatch => rebuilt from the log
            assert len(rebuilt) == n_before + 1
            with ctx2.session() as se:
                assert fixed.policy.n_updates == orig(se, uid).policy.n_updates
    finally:
        ctx2.close()


# ---------------------------------------------------------------- concurrency and failure atomicity
def test_concurrent_events_keep_the_policy_and_log_consistent(proj):
    ctx, pk = proj.ctx, proj.pk
    pauses = [x for x in proj.state["suggestions"] if x["decision_type"] == "pause"][:12]
    errors: list[BaseException] = []

    def work(chunk):
        try:
            with ctx.session() as session:
                for x in chunk:
                    from app.api.io import EventIn

                    svc_events.handle_event(ctx, session, pk, EventIn(kind="override", suggestion_id=x["id"],
                                                                      final_action="KEEP" if x["current_action"] != "KEEP" else "REMOVE"))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(pauses[i::4],)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    # a suggestion may legitimately have been replaced by a re-score while another thread was working: those are 409s
    real = [e for e in errors if not (isinstance(e, svc_events.ApiError) and e.status == 409)]
    assert real == []
    done = db_events(ctx, pk)
    assert len(done) >= 3 and len({e.suggestion_id for e in done}) == len(done)
    assert_policy_consistent(ctx, proj.uid)


def test_a_failure_halfway_leaves_neither_a_partial_event_nor_a_drifted_policy(proj, monkeypatch):
    c, pk, st = proj.client, proj.pk, proj.state
    post_event(c, pk, "accept", suggestion_id=pick(st, "pause")["id"])
    n0 = n_updates(proj)

    def boom(*_a, **_k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(svc_events, "rescore", boom)
    s2 = pick(refresh(proj), "pause", nth=1)
    with pytest.raises(RuntimeError):
        c.post(f"/api/projects/{pk}/events", json={"kind": "override", "suggestion_id": s2["id"],
                                                    "final_action": "KEEP" if s2["current_action"] != "KEEP" else "REMOVE"})
    monkeypatch.undo()
    assert len(db_events(proj.ctx, pk)) == 1  # the half-made event was rolled back
    assert n_updates(proj) == n0  # ... and the learner that had already been updated was discarded and rebuilt
    assert_policy_consistent(proj.ctx, proj.uid)
    ok = post_event(c, pk, "override", suggestion_id=s2["id"], final_action="KEEP" if s2["current_action"] != "KEEP" else "REMOVE")
    assert ok["n_events"] == 2


# ---------------------------------------------------------------- render guards (the real render is in the slow test)
def test_render_and_download_guards(proj):
    c, pk = proj.client, proj.pk
    assert c.post(f"/api/projects/{pk}/render", json={"preset": "vhs"}).status_code == 422
    assert c.get(f"/api/projects/{pk}/download").status_code == 404  # nothing rendered yet
    assert c.get(f"/api/projects/{pk}/download?fmt=png").status_code == 422
    assert c.get(f"/api/projects/{pk}/download?fmt=gif").status_code == 422
    for x in [x for x in proj.state["suggestions"] if x["decision_type"] == "phrase" and x["current_action"] == "KEEP"]:
        cur = find(refresh(proj), x)
        if cur["current_action"] == "KEEP":
            post_event(c, pk, "override", suggestion_id=cur["id"], final_action="CUT")
    state = refresh(proj)
    if state["edl"]["clips"]:  # re-scoring may have put a few phrases back: cut whatever is left
        for x in [x for x in state["suggestions"] if x["decision_type"] == "phrase" and x["current_action"] == "KEEP"]:
            post_event(c, pk, "override", suggestion_id=find(refresh(proj), x)["id"], final_action="CUT")
    if not refresh(proj)["edl"]["clips"]:
        r = c.post(f"/api/projects/{pk}/render")
        assert r.status_code == 409 and "nothing is kept" in r.json()["detail"]


def test_a_failing_render_job_is_reported_and_leaves_no_download(proj):
    c, pk = proj.client, proj.pk
    r = c.post(f"/api/projects/{pk}/render", json={"preset": "reels"})  # the fake pipeline made no source.mp4
    assert r.status_code == 202
    job = r.json()["job"]
    proj.ctx.jobs.wait(job["id"], 30)
    j = c.get(f"/api/jobs/{job['id']}").json()
    assert j["status"] == "failed" and j["kind"] == "render" and j["error"]
    assert c.get(f"/api/projects/{pk}/download").status_code == 404
    assert refresh(proj)["status"] == "ready" and refresh(proj)["exports"] == {}


def test_status_after_video_row_updates(proj):
    with proj.ctx.session() as s:
        assert s.get(Video, proj.pk).status == "ready"


# ---------------------------------------------------------------- robustness
def test_accepting_twice_is_a_conflict_but_accepting_again_after_a_change_is_fine(proj):
    c, pk, st = proj.client, proj.pk, proj.state
    s = pick(st, "pause")
    post_event(c, pk, "accept", suggestion_id=s["id"])
    r = c.post(f"/api/projects/{pk}/events", json={"kind": "accept", "suggestion_id": s["id"]})
    assert r.status_code == 409 and "already accepted" in r.json()["detail"]
    assert n_updates(proj) == 1  # no double counting
    new = "KEEP" if s["current_action"] != "KEEP" else "REMOVE"
    post_event(c, pk, "override", suggestion_id=s["id"], final_action=new)
    post_event(c, pk, "accept", suggestion_id=s["id"])  # accepting the NEW action is a real decision
    assert_policy_consistent(proj.ctx, proj.uid)


def test_a_torn_policy_cache_file_is_ignored_and_rebuilt(proj):
    c, pk, st = proj.client, proj.pk, proj.state
    post_event(c, pk, "accept", suggestion_id=pick(st, "pause")["id"])
    n0 = n_updates(proj)
    proj.ctx.policies.invalidate(proj.uid)
    for f in (proj.ctx.data_dir / "policies").glob("*.npz"):
        f.write_bytes(b"PK not a zip")  # a crash mid-write
    assert n_updates(proj) == n0
    assert_policy_consistent(proj.ctx, proj.uid)


def test_projects_interrupted_by_a_restart_are_resumed(tmp_path):
    data = tmp_path / "data"
    ctx1 = AppContext(data, ingest_fn=fake_ingest, workers=1)
    gate = threading.Event()
    ctx1.ingest_fn = lambda *a: (gate.wait(5), fake_ingest(*a))[1]  # keeps the first project 'processing'
    with TestClient(create_app(ctx1)) as c1:
        uid = c1.post("/api/users", json={"name": "Ana"}).json()["id"]
        pk = upload(c1, uid, ("a.mp4",))["id"]
        stuck = upload(c1, uid, ("b.mp4",))["id"]  # queued behind the first (one worker)
        import time

        time.sleep(0.3)
        assert c1.get(f"/api/projects/{stuck}").json()["status"] in ("uploaded", "processing")
        gate.set()
        ctx1.jobs.wait_idle(30)
        with ctx1.session() as s:  # pretend the process died while the second project was mid-ingest
            v = s.get(Video, stuck)
            v.status = "processing"
            s.add(v)
            s.commit()
    ctx1.close()
    ctx2 = AppContext(data, ingest_fn=fake_ingest)
    try:
        with TestClient(create_app(ctx2)) as c2:
            assert wait_ready(c2, stuck)["status"] == "ready" and wait_ready(c2, pk)["status"] == "ready"
            assert len(c2.get(f"/api/projects/{stuck}").json()["units"]) > 50  # no duplicated rows from the re-run
        with ctx2.session() as s:
            v = s.get(Video, stuck)
            v.status = "processing"
            v.sources_json = [str(tmp_path / "gone.mp4")]
            s.add(v)
            s.commit()
        with TestClient(create_app(ctx2)) as c3:
            p = c3.get(f"/api/projects/{stuck}").json()
            assert p["status"] == "failed" and "files are gone" in p["error"]
    finally:
        ctx2.close()


def test_a_cache_written_under_other_settings_is_not_trusted(proj, monkeypatch):
    from app import appctx

    c, pk, st = proj.client, proj.pk, proj.state
    post_event(c, pk, "accept", suggestion_id=pick(st, "pause")["id"])
    proj.ctx.policies.invalidate(proj.uid)
    rebuilt = []
    orig = proj.ctx.policies._rebuild
    monkeypatch.setattr(proj.ctx.policies, "_rebuild", lambda *a, **k: (rebuilt.append(1), orig(*a, **k))[1])
    with proj.ctx.session() as se:
        proj.ctx.policies.get(se, proj.uid)
    assert rebuilt == []  # same settings: restored from the cache
    proj.ctx.policies.invalidate(proj.uid)
    monkeypatch.setattr(appctx, "PRODUCT_LAMBDA", 3.0)
    proj.ctx.policies.config_key = appctx._config_key(proj.ctx.shipped_priors)  # as a new process would compute it
    with proj.ctx.session() as se:
        proj.ctx.policies.get(se, proj.uid)
    assert rebuilt == [1]  # different lambda: the stale cache is ignored and rebuilt from the log
