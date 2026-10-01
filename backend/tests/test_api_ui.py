"""Backend support for the editor UI: caption line breaks (preview == export), styles/fonts, profile, SPA hosting."""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from app import config
from app.appctx import AppContext
from app.main import create_app
from app.render import ass as ass_mod
from app.schemas import EDL, CaptionLine, Captions, CaptionWord, TextOverlay
from tests.apihelpers import fake_ingest, new_ready_project, pick, post_event, upload, wait_ready


def _edl(style: str, texts: list[list[str]], overlay: str | None = None) -> EDL:
    lines = []
    t = 0.0
    for ws in texts:
        words = [CaptionWord(t0=t + i * 0.3, t1=t + i * 0.3 + 0.3, text=w) for i, w in enumerate(ws)]
        lines.append(CaptionLine(t0=t, t1=t + 0.3 * len(ws), words=words))
        t += 0.3 * len(ws) + 0.5
    edl = EDL(video_id="v", captions=Captions(style_id=style, lines=lines), duration=t + 1)
    if overlay:
        edl.texts = [TextOverlay(id="t1", text=overlay, t0=0, t1=2)]
    return edl


@pytest.mark.parametrize("style", ["bold_pop", "clean", "karaoke", "minimal"])
def test_caption_breaks_equal_the_breaks_in_the_exported_ass(style):
    """PLAN 18: 'same line breaks' - the preview breaks where the ASS export breaks, from the same function."""
    edl = _edl(style, [["Entrepreneurship", "revolutionizes", "communication", "everywhere"], ["Hi", "there"],
                       ["Three", "mistakes", "beginners", "make"], ["supercalifragilistic", "words"]])
    breaks = ass_mod.caption_breaks(edl)
    assert len(breaks) == 4 and breaks[1] is None and breaks[0] is not None
    ass = ass_mod.build_ass(edl)
    events = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:") and ",Cap," in ln]
    cap = ass_mod.load_styles()[0]["styles"][style]
    for li, line in enumerate(edl.captions.lines):
        start = ass_mod._fmt_time(line.t0)
        mine = [e for e in events if f",{start}," in e or e.startswith((f"Dialogue: 0,{start}", f"Dialogue: 1,{start}"))]
        assert mine, (style, li)
        for e in mine:  # every event of the line (one per word in pop mode) breaks at the same word
            text = e.split(",,", 1)[1]
            text = re.sub(r"\{[^}]*\}", "", text.split("}", 1)[1] if text.startswith("{\\an5") else text)
            before = text.split("\\N")
            got = None if len(before) == 1 else len(before[0].split())
            assert got == breaks[li], (style, li, text)
    assert cap["mode"] in ("pop", "line", "karaoke")


def test_text_overlay_breaks_use_the_text_style():
    edl = _edl("bold_pop", [["a"]], overlay="Three editing mistakes that every single beginner makes")
    tb = ass_mod.text_breaks(edl)
    assert tb["t1"] is not None
    ass = ass_mod.build_ass(edl)
    line = next(ln for ln in ass.splitlines() if "Txt_title" in ln and ln.startswith("Dialogue"))
    assert "\\N" in line and len(line.split("\\N")[0].split("}")[-1].split()) == tb["t1"]
    short = _edl("bold_pop", [["a"]], overlay="Hi")
    assert ass_mod.text_breaks(short) == {"t1": None}


def test_state_carries_the_breaks_and_the_source_size(client):
    _, pk, p = new_ready_project(client)
    assert p["source"] == {"width": 1920, "height": 1080}
    assert len(p["caption_breaks"]) == len(p["edl"]["captions"]["lines"])
    s = client.post(f"/api/projects/{pk}/suggest").json()
    assert len(s["caption_breaks"]) == len(s["edl"]["captions"]["lines"]) > 5
    ev = post_event(client, pk, "setting", payload={"key": "caption_style", "value": "karaoke"})
    assert len(ev["caption_breaks"]) == len(ev["edl"]["captions"]["lines"])
    ov = post_event(client, pk, "text_overlay", payload={"op": "add", "text": "Three editing mistakes that every single beginner makes", "t0": 0, "t1": 2})
    oid = ov["edl"]["texts"][0]["id"]
    assert ov["text_breaks"][oid] is not None


def test_styles_and_fonts_endpoints(client):
    r = client.get("/api/styles").json()
    assert set(r) == {"captions", "texts", "presets"}
    assert set(r["captions"]["styles"]) == {"bold_pop", "clean", "karaoke", "minimal"}
    assert r["captions"]["positions"]["lower_third"] == 0.68 and set(r["presets"]) == {"reels", "shorts", "tiktok"}
    for st in [*r["captions"]["styles"].values(), *r["texts"]["styles"].values()]:
        f = client.get(f"/api/fonts/{st['font_file']}")
        assert f.status_code == 200 and f.headers["content-type"] == "font/ttf" and len(f.content) > 100_000
    assert client.get("/api/fonts/nope.ttf").status_code == 404
    assert client.get("/api/fonts/..%2Fpyproject.toml").status_code == 404


def test_profile(client):
    uid, pk, _ = new_ready_project(client)
    prof = client.get(f"/api/users/{uid}/profile").json()
    assert prof["user"]["name"] == "Ana" and prof["n_projects"] == 1 and prof["n_finalized"] == 0 and prof["metrics"] == []
    s = client.post(f"/api/projects/{pk}/suggest").json()
    post_event(client, pk, "accept", suggestion_id=pick(s, "pause")["id"])
    client.post(f"/api/projects/{pk}/seen", json={"unit_ids": [0, 1, 2]})
    client.post(f"/api/projects/{pk}/finalize", json={"seconds_active": 60})
    prof = client.get(f"/api/users/{uid}/profile").json()
    assert prof["n_finalized"] == 1 and prof["n_learning_events"] >= 1 and prof["style_mixture"]["n_events"] >= 1
    (m,) = prof["metrics"]
    assert m["video_index"] == 1 and m["project_id"] == pk and m["manual_per_minute"] >= 0 and m["seconds_active"] == 60
    assert client.get("/api/users/999/profile").status_code == 404


def test_the_built_frontend_is_served_with_spa_fallback(tmp_path, monkeypatch):
    dist = tmp_path / "frontend" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>app</html>", encoding="utf-8")
    (dist / "assets" / "x.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    ctx = AppContext(tmp_path / "data", ingest_fn=fake_ingest)
    try:
        with TestClient(create_app(ctx)) as c:
            assert c.get("/").text == "<html>app</html>"
            assert c.get("/projects/12").text == "<html>app</html>"  # a client-side route
            assert c.get("/assets/x.js").text == "console.log(1)"
            assert c.get("/api/nope").status_code == 404 and c.get("/api/health").json()["status"] == "ok"
            assert c.get("/../secret.txt").text == "<html>app</html>"  # never outside dist
            assert c.get("/%2e%2e/secret.txt").text == "<html>app</html>"
    finally:
        ctx.close()


def test_upload_page_flow_unchanged_by_the_ui_additions(client):
    uid = client.post("/api/users", json={"name": "Z"}).json()["id"]
    pk = upload(client, uid, ("q.mp4",))["id"]
    assert wait_ready(client, pk)["source"]["width"] == 1920
