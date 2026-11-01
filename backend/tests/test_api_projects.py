"""API: users, project creation / ingest jobs, listing, proxy, suggestions."""
from __future__ import annotations

import io

import pytest
from sqlmodel import select

from app.models import SuggestionRow, UnitRow, Video, WordRow
from app.policy import context
from tests.apihelpers import make_client, new_ready_project, upload, wait_ready


def _name(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def test_health_still_works_without_a_context():
    from fastapi.testclient import TestClient

    from app.main import app

    assert TestClient(app).get("/api/health").json()["status"] == "ok"


def test_users(client):
    assert client.get("/api/users").json() == []
    u = client.post("/api/users", json={"name": "  Ana  "})
    assert u.status_code == 201 and u.json()["name"] == "Ana"
    client.post("/api/users", json={"name": "Bo"})
    assert [x["name"] for x in client.get("/api/users").json()] == ["Ana", "Bo"]
    assert client.post("/api/users", json={"name": "   "}).status_code == 422
    assert client.post("/api/users", json={"name": ""}).status_code == 422
    assert client.post("/api/users", json={}).status_code == 422


def test_create_project_runs_the_ingest_job_and_stores_units(client, ctx):
    uid = client.post("/api/users", json={"name": "Ana"}).json()["id"]
    created = upload(client, uid, ("one.mp4", "two.mov"), title="My reel", target="45")
    assert created["status"] in ("uploaded", "processing", "ready") and created["title"] == "My reel"
    assert created["video_id"] == f"v_{created['id']}" and created["target_duration"] == 45.0
    pk = created["id"]
    p = wait_ready(client, pk)
    assert p["status"] == "ready" and p["error"] is None and p["duration"] > 30 and p["video_type"] == "vlog"
    assert p["job"]["kind"] == "ingest" and p["job"]["status"] == "done"
    assert len(p["units"]) > 50 and len(p["words"]) > 100 and p["edl"]["video_id"] == p["video_id"]
    assert p["suggestions"] == [] and p["can_undo"] is False and p["settings"]["speed"] == 1.0
    with ctx.session() as s:  # rows mirror the analysis (PLAN 16.1)
        assert len(s.exec(select(UnitRow).where(UnitRow.video_id == pk)).all()) == len(p["units"])
        assert len(s.exec(select(WordRow).where(WordRow.video_id == pk)).all()) == len(p["words"])
        assert [_name(x) for x in s.get(Video, pk).sources_json] == ["00_one.mp4", "01_two.mov"]
    # uploads were moved next to the project; no temp dirs are left behind
    assert [d.name for d in ctx.uploads_dir.iterdir()] == [f"v_{pk}"]


def test_upload_order_is_respected(client, ctx):
    uid = client.post("/api/users", json={"name": "Ana"}).json()["id"]
    created = upload(client, uid, ("a.mp4", "b.mp4", "c.mp4"), order="[2, 0, 1]")
    wait_ready(client, created["id"])
    with ctx.session() as s:
        names = [_name(x) for x in s.get(Video, created["id"]).sources_json]
    assert names == ["00_c.mp4", "01_a.mp4", "02_b.mp4"]


@pytest.mark.parametrize("kwargs", [
    {"order": "[0, 0]"}, {"order": "nope"}, {"order": "[1]"}, {"type": "podcast"}, {"target": "2"}, {"target": "999"},
])
def test_create_project_validation(client, kwargs):
    uid = client.post("/api/users", json={"name": "Ana"}).json()["id"]
    files = [("files", ("a.mp4", io.BytesIO(b"x" * 10), "video/mp4")),
             ("files", ("b.mp4", io.BytesIO(b"y" * 10), "video/mp4"))]
    data = {"user_id": str(uid), "type": "vlog", **kwargs}
    assert client.post("/api/projects", files=files, data=data).status_code == 422


def test_too_long_a_video_is_refused_at_upload_with_a_clear_message(client, ctx, monkeypatch):
    from types import SimpleNamespace

    from app.models import Video as V
    from app.svc import projects as pj

    uid = client.post("/api/users", json={"name": "Ana"}).json()["id"]
    monkeypatch.setattr(pj, "probe", lambda _p: SimpleNamespace(duration=400.0))
    r = client.post("/api/projects", files=[("files", ("long.mp4", io.BytesIO(b"x"), "video/mp4"))],
                    data={"user_id": str(uid), "type": "vlog"})
    assert r.status_code == 422 and "6:40" in r.json()["detail"] and "5 minutes" in r.json()["detail"]
    with ctx.session() as s:
        assert s.exec(select(V)).all() == []  # no failed project left behind


def test_create_project_rejects_bad_uploads(client, ctx):
    uid = client.post("/api/users", json={"name": "Ana"}).json()["id"]
    ok = ("files", ("a.mp4", io.BytesIO(b"x"), "video/mp4"))
    assert client.post("/api/projects", data={"user_id": str(uid)}).status_code == 422  # no files
    r = client.post("/api/projects", files=[("files", ("notes.txt", io.BytesIO(b"x"), "text/plain"))],
                    data={"user_id": str(uid)})
    assert r.status_code == 422
    r = client.post("/api/projects", files=[("files", ("a.mp4", io.BytesIO(b""), "video/mp4"))], data={"user_id": str(uid)})
    assert r.status_code == 422  # empty file
    assert client.post("/api/projects", files=[ok], data={"user_id": "999"}).status_code == 404
    assert client.post("/api/projects", files=[ok]).status_code == 422  # no user_id
    assert not any(ctx.uploads_dir.iterdir())  # failed uploads leave nothing behind


def test_upload_size_limit(client, ctx, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "MAX_TOTAL_BYTES", 1000)
    uid = client.post("/api/users", json={"name": "Ana"}).json()["id"]
    r = client.post("/api/projects", files=[("files", ("a.mp4", io.BytesIO(b"x" * 5000), "video/mp4"))],
                    data={"user_id": str(uid)})
    assert r.status_code == 413 and not any(ctx.uploads_dir.iterdir())


def test_unsafe_filenames_stay_inside_the_project_folder(client, ctx):
    uid = client.post("/api/users", json={"name": "Ana"}).json()["id"]
    r = client.post("/api/projects", files=[("files", ("../../evil.mp4", io.BytesIO(b"x"), "video/mp4"))],
                    data={"user_id": str(uid)})
    assert r.status_code == 201
    pk = r.json()["id"]
    wait_ready(client, pk)
    assert [p.name for p in (ctx.uploads_dir / f"v_{pk}").iterdir()] == ["00_evil.mp4"]
    assert not (ctx.data_dir.parent / "evil.mp4").exists()


def test_failed_ingest_is_reported(tmp_path):
    def boom(*_a):
        raise RuntimeError("ffprobe could not read the file")

    with make_client(tmp_path, ingest_fn=boom) as c:
        uid = c.post("/api/users", json={"name": "Ana"}).json()["id"]
        pk = upload(c, uid)["id"]
        p = wait_ready(c, pk)
        assert p["status"] == "failed" and "ffprobe could not read" in p["error"] and p["edl"] is None
        assert p["job"]["status"] == "failed"
        setting = {"kind": "setting", "payload": {"key": "audio_enhance", "value": False}}
        assert c.post(f"/api/projects/{pk}/suggest").status_code == 409  # not ready
        assert c.post(f"/api/projects/{pk}/events", json=setting).status_code == 409
        assert c.post(f"/api/projects/{pk}/undo").status_code == 409
        assert c.post(f"/api/projects/{pk}/finalize").status_code == 409
        assert c.post(f"/api/projects/{pk}/render").status_code == 409
        assert c.get(f"/api/projects/{pk}/proxy").status_code == 404
        c.app.state.ctx.close()


def test_video_without_speech_fails_cleanly(tmp_path):
    def silent(inputs, video_id, video_type, out_dir):
        from app.schemas import UnitsFile

        out_dir.mkdir(parents=True, exist_ok=True)
        return UnitsFile(video_id=video_id, duration=3.0)

    with make_client(tmp_path, ingest_fn=silent) as c:
        uid = c.post("/api/users", json={"name": "Ana"}).json()["id"]
        p = wait_ready(c, upload(c, uid)["id"])
        assert p["status"] == "failed" and "no speech" in p["error"]
        c.app.state.ctx.close()


def test_listing_and_not_found(client):
    uid, pk, _ = new_ready_project(client)
    other = client.post("/api/users", json={"name": "Bo"}).json()["id"]
    assert [p["id"] for p in client.get(f"/api/projects?user_id={uid}").json()] == [pk]
    assert client.get(f"/api/projects?user_id={other}").json() == []
    assert client.get("/api/projects?user_id=999").status_code == 404
    assert client.get("/api/projects").status_code == 422
    assert client.get("/api/projects/999").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get(f"/api/projects/{pk}/thumbnail").status_code == 404  # the fake pipeline makes no media
    pk2 = upload(client, uid, ("z.mp4",))["id"]
    wait_ready(client, pk2)
    ids = [p["id"] for p in client.get(f"/api/projects?user_id={uid}").json()]
    assert ids == [pk2, pk]  # most recently updated first


def test_proxy_serves_http_range_requests(client, ctx):
    _, pk, _ = new_ready_project(client)
    (ctx.video_dir(pk) / "proxy.mp4").write_bytes(bytes(range(256)) * 40)  # 10 240 bytes
    full = client.get(f"/api/projects/{pk}/proxy")
    assert full.status_code == 200 and len(full.content) == 10240 and full.headers["accept-ranges"] == "bytes"
    part = client.get(f"/api/projects/{pk}/proxy", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and part.content == full.content[100:200]
    assert part.headers["content-range"] == "bytes 100-199/10240"
    tail = client.get(f"/api/projects/{pk}/proxy", headers={"Range": "bytes=10000-"})
    assert tail.status_code == 206 and tail.content == full.content[10000:]
    assert client.get(f"/api/projects/{pk}/proxy", headers={"Range": "bytes=99999-"}).status_code == 416


def test_suggest_creates_decisions_for_every_type_and_is_idempotent(client, ctx):
    _, pk, p = new_ready_project(client)
    base_duration = p["edl"]["duration"]
    s = client.post(f"/api/projects/{pk}/suggest")
    assert s.status_code == 200
    s = s.json()
    kinds = {x["decision_type"] for x in s["suggestions"]}
    assert kinds == {"phrase", "pause", "filler", "zoom", "emphasis"} and 100 < len(s["suggestions"]) < 400
    assert all(x["status"] == "pending" and x["current_action"] == x["proposed_action"] for x in s["suggestions"])
    assert all(0 < x["confidence"] <= 1 and 0 < x["propensity"] <= 1 for x in s["suggestions"])
    assert s["edl"]["duration"] < base_duration  # pauses / retakes are cut by default
    ids = sorted(x["id"] for x in s["suggestions"])
    assert sorted(s["queue"]["needs_review"] + s["queue"]["confident"]) == ids
    conf = {x["id"]: x["confidence"] for x in s["suggestions"]}
    nr = [conf[i] for i in s["queue"]["needs_review"]]
    assert all(c < 0.7 for c in nr) and nr == sorted(nr) and all(conf[i] >= 0.7 for i in s["queue"]["confident"])
    assert s["style_mixture"]["n_events"] == 0 and abs(sum(s["style_mixture"]["weights"].values()) - 1) < 1e-9
    again = client.post(f"/api/projects/{pk}/suggest").json()
    assert sorted(x["id"] for x in again["suggestions"]) == ids  # not regenerated
    with ctx.session() as sess:
        rows = sess.exec(select(SuggestionRow).where(SuggestionRow.video_id == pk)).all()
        assert len(rows) == len(ids) and all(r.mean_best for r in rows)
        for r in rows:  # the stored context is complete enough to rebuild a policy from the log
            assert r.context_raw_json["bias"] == 1.0 and r.feature_version == 1
            if r.decision_type == "emphasis":
                assert r.candidates_json[0] == "NONE"
            else:
                assert r.candidates_json == [] and r.proposed_action in context.ARMS[r.decision_type]


def test_get_project_matches_the_state_returned_by_suggest(client):
    _, pk, _ = new_ready_project(client)
    s = client.post(f"/api/projects/{pk}/suggest").json()
    g = client.get(f"/api/projects/{pk}").json()
    assert g["edl"] == s["edl"] and g["suggestions"] == s["suggestions"] and g["queue"] == s["queue"]


def test_explain_endpoint(client):
    _, pk, _ = new_ready_project(client)
    s = client.post(f"/api/projects/{pk}/suggest").json()
    sid = s["suggestions"][0]["id"]
    r = client.get(f"/api/projects/{pk}/suggestions/{sid}/explain")
    assert r.status_code == 200 and r.json()["suggestion_id"] == sid and "usually" in r.json()["text"]
    assert client.get(f"/api/projects/{pk}/suggestions/99999/explain").status_code == 404


def test_projects_of_different_users_have_separate_policies(client):
    u1, p1, _ = new_ready_project(client, "Ana")
    u2 = client.post("/api/users", json={"name": "Bo"}).json()["id"]
    p2 = upload(client, u2, ("b.mp4",))["id"]
    wait_ready(client, p2)
    for pk in (p1, p2):
        client.post(f"/api/projects/{pk}/suggest")
    s1 = client.get(f"/api/projects/{p1}").json()
    for x in [x for x in s1["suggestions"] if x["decision_type"] == "pause"][:8]:
        new = "KEEP" if x["current_action"] != "KEEP" else "REMOVE"
        client.post(f"/api/projects/{p1}/events", json={"kind": "override", "suggestion_id": x["id"], "final_action": new})
    assert client.get(f"/api/projects/{p2}").json()["style_mixture"]["n_events"] == 0
    assert client.get(f"/api/projects/{p1}").json()["style_mixture"]["n_events"] > 0
    assert u1 != u2


@pytest.mark.parametrize("raw,expected", [
    ("a?b*c.mp4", "a_b_c.mp4"), ("../../x.mp4", "x.mp4"), (r"C:\clips\take 1.MOV", "take 1.MOV"),
    ("CON.mp4", "CON.mp4"), ("  .. ", "clip0.mp4"), ("", "clip0.mp4"), ("naïve vidéo.mp4", "naïve vidéo.mp4"),
    ("a" * 200 + ".mp4", "a" * 60 + ".mp4"), ("bad<>|\":name.mp4", "bad_____name.mp4"),
])
def test_safe_filename(raw, expected):
    from app.api.routes import safe_filename

    assert safe_filename(raw, 0) == expected


def test_awkward_file_names_do_not_strand_a_project(client, ctx):
    uid = client.post("/api/users", json={"name": "Ana"}).json()["id"]
    r = client.post("/api/projects", files=[("files", ("what?.mp4", io.BytesIO(b"x"), "video/mp4"))],
                    data={"user_id": str(uid)})
    assert r.status_code == 201
    assert wait_ready(client, r.json()["id"])["status"] == "ready"


def test_order_must_be_integers_not_booleans(client):
    uid = client.post("/api/users", json={"name": "Ana"}).json()["id"]
    files = [("files", ("a.mp4", io.BytesIO(b"x"), "video/mp4")), ("files", ("b.mp4", io.BytesIO(b"y"), "video/mp4"))]
    r = client.post("/api/projects", files=files, data={"user_id": str(uid), "order": "[true, false]"})
    assert r.status_code == 422


def test_renders_are_not_starved_by_queued_ingests(tmp_path):
    import threading

    from app.jobs import JobRunner

    runner = JobRunner(workers=1, ingest_workers=1)
    gate = threading.Event()
    runner.submit(lambda: gate.wait(10), kind="ingest")
    runner.submit(lambda: gate.wait(10), kind="ingest")  # queued behind the first
    done = runner.submit(lambda: "rendered", kind="render")
    assert runner.wait(done.id, 5).result == "rendered"  # runs while both ingests are still pending
    gate.set()
    runner.wait_idle(10)
    runner.shutdown()
