"""Phase 3 acceptance (real pipeline, real ffmpeg): create project (2 clips) -> ready -> suggest -> 10 feedbacks +
3 transcript edits -> undo/redo (state matches) -> finalize -> render -> download MP4 + SRT."""
from __future__ import annotations

import io
import json
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from app import config
from app.appctx import AppContext
from app.main import create_app
from app.render import audio
from tests.test_api_state import digest

CLIPS = [config.RAW_DIR / "clip_02_take_a.mp4", config.RAW_DIR / "clip_03_take_b.mp4"]


def wait_until(client, url: str, done, timeout: float, what: str):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(url).json()
        if done(r):
            return r
        time.sleep(0.5)
    raise TimeoutError(what)


def find(state, dt, target):
    return next(x for x in state["suggestions"] if x["decision_type"] == dt and x["target_id"] == target)


@pytest.mark.slow
@pytest.mark.skipif(not all(p.exists() for p in CLIPS), reason="run scripts/make_test_clips.py first")
def test_phase3_acceptance_flow(tmp_path):
    ctx = AppContext(tmp_path / "data")  # the REAL ingest pipeline
    try:
        with TestClient(create_app(ctx)) as c:
            uid = c.post("/api/users", json={"name": "Ana"}).json()["id"]
            files = [("files", (p.name, io.BytesIO(p.read_bytes()), "video/mp4")) for p in CLIPS]
            r = c.post("/api/projects", files=files, data={"user_id": str(uid), "type": "vlog", "title": "Two takes"})
            assert r.status_code == 201, r.text
            pk = r.json()["id"]

            # ---- ready
            p = wait_until(c, f"/api/projects/{pk}", lambda x: x["status"] in ("ready", "failed"), 600, "ingest")
            assert p["status"] == "ready", p["error"]
            assert len(p["source_map"]) == 2 and abs(p["duration"] - 29.0) < 0.2
            assert any(u["features"].get("is_take_boundary") == 1 for u in p["units"] if u["kind"] == "pause")
            assert c.get(f"/api/projects/{pk}/thumbnail").status_code == 200
            part = c.get(f"/api/projects/{pk}/proxy", headers={"Range": "bytes=0-1023"})
            assert part.status_code == 206 and len(part.content) == 1024

            # ---- suggest
            s = c.post(f"/api/projects/{pk}/suggest").json()
            assert len(s["suggestions"]) > 40 and s["edl"]["duration"] < p["edl"]["duration"]
            states = [s]

            def post(kind, **body):
                out = c.post(f"/api/projects/{pk}/events", json={"kind": kind, **body})
                assert out.status_code == 200, out.text
                states.append(out.json())
                return states[-1]

            # ---- 10 feedbacks (accept / reject / override, each resolved against the CURRENT suggestions)
            def pending(dt):
                return [x for x in states[-1]["suggestions"] if x["decision_type"] == dt and x["status"] == "pending"]

            n = 0
            for dt, kind in [("pause", "accept"), ("pause", "accept"), ("phrase", "accept"), ("pause", "override"),
                             ("zoom", "override"), ("pause", "reject"), ("filler", "accept"), ("emphasis", "accept"),
                             ("phrase", "reject"), ("pause", "accept")]:
                items = pending(dt)
                if not items:
                    continue
                x = items[0]
                if kind == "override":
                    arms = x["candidates"] or {"pause": ["KEEP", "TIGHT_40", "TIGHT_15", "REMOVE"], "zoom": ["NONE", "Z115", "Z130"]}[dt]
                    post("override", suggestion_id=x["id"], final_action=next(a for a in arms if a != x["current_action"]))
                else:
                    post(kind, suggestion_id=x["id"])
                n += 1
            assert n >= 9

            # ---- 3 transcript edits
            words = p["words"]
            kept = [x for x in states[-1]["suggestions"] if x["decision_type"] == "phrase" and x["current_action"] == "KEEP"]
            victim = kept[0]
            ids = [w["id"] for w in words if w["unit_id"] == victim["target_id"]]
            edit1 = post("manual", edit={"kind": "delete_words", "word_ids": ids})
            assert find(edit1, "phrase", victim["target_id"])["current_action"] == "CUT"
            edit2 = post("manual", edit={"kind": "restore_words", "word_ids": ids})
            assert find(edit2, "phrase", victim["target_id"])["current_action"] == "KEEP"
            big = next(u for u in p["units"] if u["kind"] == "speech" and sum(w["unit_id"] == u["id"] for w in words) >= 4
                       and find(edit2, "phrase", u["id"])["current_action"] == "KEEP")
            one = [w for w in words if w["unit_id"] == big["id"]][1]
            before_words = [w["text"] for ln in edit2["edl"]["captions"]["lines"] for w in ln["words"]]
            edit3 = post("manual", edit={"kind": "delete_words", "word_ids": [one["id"]]})  # a fine cut
            after_words = [w["text"] for ln in edit3["edl"]["captions"]["lines"] for w in ln["words"]]
            assert len(after_words) == len(before_words) - 1 and edit3["can_undo"]

            # ---- undo / redo: the state matches exactly
            final = states[-1]
            undone = [c.post(f"/api/projects/{pk}/undo").json() for _ in range(3)]
            for k, out in enumerate(undone, 1):
                assert digest(out) == digest(states[-1 - k]), f"undo {k}"
            for k in range(3):
                out = c.post(f"/api/projects/{pk}/redo").json()
                assert digest(out) == digest(states[len(states) - 3 + k]), f"redo {k}"
            assert digest(c.get(f"/api/projects/{pk}").json()) == digest(final)

            # ---- finalize
            c.post(f"/api/projects/{pk}/seen", json={"unit_ids": [u["id"] for u in p["units"]]})
            fin = c.post(f"/api/projects/{pk}/finalize", json={"seconds_active": 95}).json()
            assert fin["metrics"]["n_reviewed"] == fin["metrics"]["n_decisions"] and fin["metrics"]["video_index"] == 1
            edl = fin["edl"]
            assert edl["clips"] and edl["duration"] > 3

            # ---- render + download
            r = c.post(f"/api/projects/{pk}/render", json={"preset": "reels"})
            assert r.status_code == 202, r.text
            job = wait_until(c, f"/api/jobs/{r.json()['job']['id']}", lambda j: j["status"] in ("done", "failed"), 600, "render")
            assert job["status"] == "done", job["error"]
            mp4 = c.get(f"/api/projects/{pk}/download?fmt=mp4")
            srt = c.get(f"/api/projects/{pk}/download?fmt=srt")
            assert mp4.status_code == 200 and srt.status_code == 200 and mp4.headers["x-render-stale"] == "false"
            assert mp4.headers["content-type"] == "video/mp4" and len(mp4.content) > 50_000
            out = tmp_path / "dl.mp4"
            out.write_bytes(mp4.content)
            probe = json.loads(subprocess.run([config.ffprobe_path(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                                               "stream=width,height,codec_name:format=duration", "-of", "json", str(out)],
                                              capture_output=True, text=True, check=True).stdout)
            assert (probe["streams"][0]["width"], probe["streams"][0]["height"]) == (1080, 1920)
            assert abs(float(probe["format"]["duration"]) - edl["duration"]) <= 0.1
            assert abs(audio.measure_loudness(out)["I"] + 14.0) <= 1.0
            text = srt.text
            assert text.startswith("1\n") and " --> " in text and len(text.strip().split("\n\n")) == len(edl["captions"]["lines"])
            done_state = c.get(f"/api/projects/{pk}").json()
            assert done_state["status"] == "exported" and done_state["exports"]["reels"]["duration"] > 3
            # editing after the export marks the download stale
            post("speed", payload={"value": 1.1})
            stale = c.get(f"/api/projects/{pk}/download?fmt=mp4")
            assert stale.headers["x-render-stale"] == "true"
            assert c.get(f"/api/projects/{pk}").json()["status"] == "edited"
    finally:
        ctx.close()
