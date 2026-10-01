import re
from pathlib import Path

from fastapi.testclient import TestClient

from app import schemas
from app.main import app
from app.schemas import EDL, CaptionLine, CaptionWord, Clip, Event, Suggestion, UnitsFile

TS = Path(__file__).resolve().parents[2] / "frontend" / "src" / "types.ts"


def test_edl_roundtrip():
    edl = EDL(
        video_id="v_1",
        clips=[Clip(src_start=0.84, src_end=3.12, zoom=1.15, unit_ids=[0])],
        duration=2.28,
    )
    edl.captions.lines.append(
        CaptionLine(t0=0, t1=1, words=[CaptionWord(t0=0, t1=1, text="Hi", emphasis=True)])
    )
    again = EDL.model_validate_json(edl.model_dump_json())
    assert again == edl
    assert again.edl_version == 3 and again.output.width == 1080


def test_other_models_roundtrip():
    u = UnitsFile(video_id="v_1", duration=10.0)
    assert UnitsFile.model_validate_json(u.model_dump_json()) == u
    s = Suggestion(video_id="v_1", target_id=3, decision_type="pause", proposed_action="REMOVE")
    assert Suggestion.model_validate_json(s.model_dump_json()) == s
    e = Event(user_id=1, video_id="v_1", kind="override", payload={"final_action": "KEEP"})
    assert Event.model_validate_json(e.model_dump_json()) == e


def test_ts_types_in_sync():
    """Every pydantic model and its fields must appear in types.ts."""
    ts = TS.read_text(encoding="utf-8")
    models = {
        "Word": schemas.Word, "Unit": schemas.Unit, "UnitsFile": schemas.UnitsFile,
        "Point": schemas.Point, "OutputSpec": schemas.OutputSpec, "Clip": schemas.Clip,
        "CaptionWord": schemas.CaptionWord, "CaptionLine": schemas.CaptionLine,
        "Captions": schemas.Captions, "TextOverlay": schemas.TextOverlay,
        "MusicSpec": schemas.MusicSpec, "AudioSpec": schemas.AudioSpec, "EDL": schemas.EDL,
        "Suggestion": schemas.Suggestion, "AppEvent": schemas.Event,
    }
    assert f"SCHEMA_VERSION = {schemas.SCHEMA_VERSION}" in ts
    for name, model in models.items():
        m = re.search(rf"export interface {name} \{{(.*?)\n\}}", ts, re.DOTALL)
        assert m, f"interface {name} missing in types.ts"
        ts_fields = set(re.findall(r"^\s*(\w+)\??:", m.group(1), re.MULTILINE))
        assert ts_fields == set(model.model_fields), f"{name}: {ts_fields ^ set(model.model_fields)}"


def test_health():
    r = TestClient(app).get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["ffmpeg"]["ok"], body
    assert body["ffmpeg"]["missing_filters"] == []
