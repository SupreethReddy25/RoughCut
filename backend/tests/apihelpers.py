"""Helpers for the API tests: a fake (instant) analysis pipeline built on the simulator's synthetic videos."""
from __future__ import annotations

import io
import json
import time
import zlib
from pathlib import Path

from fastapi.testclient import TestClient

from app.appctx import AppContext
from app.main import create_app
from app.schemas import UnitsFile
from sim.synth import make_video


def fake_ingest(inputs: list[Path], video_id: str, video_type: str, out_dir: Path) -> UnitsFile:
    """Writes the same files the real pipeline does (minus the media) from a synthetic video seeded by the uploads."""
    seed = zlib.crc32("|".join(p.name for p in inputs).encode()) % 100_000
    v = make_video(seed, video_type if video_type != "other" else "vlog", video_id=video_id)
    uf = v.uf.model_copy(update={"video_type": video_type})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "units.json").write_text(uf.model_dump_json(), encoding="utf-8")
    (out_dir / "word_features.json").write_text(json.dumps({str(k): f for k, f in v.word_feats.items()}), encoding="utf-8")
    (out_dir / "face_track.json").write_text(json.dumps({"fps": 5, "samples": []}), encoding="utf-8")
    return uf


def make_client(tmp_path: Path, ingest_fn=fake_ingest, **kw) -> TestClient:
    ctx = AppContext(tmp_path / "data", ingest_fn=ingest_fn, **kw)
    return TestClient(create_app(ctx))


def upload(client: TestClient, user_id: int, names=("a.mp4",), **form) -> dict:
    files = [("files", (n, io.BytesIO(b"not really a video " + n.encode()), "video/mp4")) for n in names]
    r = client.post("/api/projects", files=files, data={"user_id": str(user_id), "type": "vlog", **form})
    assert r.status_code == 201, r.text
    return r.json()


def wait_ready(client: TestClient, pk: int, timeout: float = 60.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = client.get(f"/api/projects/{pk}").json()
        if p["status"] in ("ready", "failed"):
            return p
        time.sleep(0.05)
    raise TimeoutError(f"project {pk} still {p['status']}")


def new_ready_project(client: TestClient, name="Ana", names=("a.mp4",)) -> tuple[int, int, dict]:
    uid = client.post("/api/users", json={"name": name}).json()["id"]
    pk = upload(client, uid, names)["id"]
    p = wait_ready(client, pk)
    assert p["status"] == "ready", p["error"]
    return uid, pk, p


def post_event(client: TestClient, pk: int, kind: str, expect: int = 200, **body) -> dict:
    r = client.post(f"/api/projects/{pk}/events", json={"kind": kind, **body})
    assert r.status_code == expect, r.text
    return r.json()


def pick(state: dict, decision_type: str, status: str = "pending", nth: int = 0, **where) -> dict:
    items = [s for s in state["suggestions"] if s["decision_type"] == decision_type and s["status"] == status
             and all(s[k] == v for k, v in where.items())]
    return items[nth]


def other_action(s: dict, avoid: set[str] | None = None) -> str:
    from app.policy import context

    arms = s["candidates"] or context.ARMS[s["decision_type"]]
    return next(a for a in arms if a != s["current_action"] and a not in (avoid or set()))


def assert_policy_consistent(ctx: AppContext, user_id: int) -> None:
    """The online-updated (cached) learner must equal a fresh replay of the event log, exactly."""
    import numpy as np

    with ctx.session() as s:
        cached = ctx.policies.get(s, user_id)
        rebuilt = ctx.policies._rebuild(s, user_id)
    a, b = cached.policy, rebuilt.policy
    assert a.n_updates == b.n_updates
    keys = {k for k, m in {**a.models, **b.models}.items() if (a.models.get(k) or b.models.get(k)).n or
            (b.models.get(k) and b.models[k].n)}  # models never updated only exist because choose() looked at them
    assert keys == {k for k, m in a.models.items() if m.n} | {k for k, m in b.models.items() if m.n}
    for k in keys:
        ma, mb = a.models[k], b.models[k]
        assert ma.n == mb.n, k
        assert np.allclose(ma.A_d, mb.A_d, atol=1e-12) and np.allclose(ma.b_d, mb.b_d, atol=1e-12), k
        assert np.allclose(ma.theta, mb.theta, atol=1e-10), k
    if cached.mixture is not None:
        assert np.allclose(cached.mixture.logw, rebuilt.mixture.logw)
        assert cached.mixture.n_events == rebuilt.mixture.n_events
