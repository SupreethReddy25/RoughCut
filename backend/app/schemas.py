"""FROZEN CONTRACT (PLAN.md sections 5, 6, 16.1). Do not change without team agreement.

If you change anything here, update frontend/src/types.ts to match
(tests/test_schemas.py checks the mirror) and bump SCHEMA_VERSION.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 2

VideoType = Literal["tutorial", "vlog", "opinion", "other"]
DecisionType = Literal["phrase", "pause", "zoom", "filler", "emphasis", "hook", "rhythm", "pace", "music_timing"]
UnitKind = Literal["speech", "pause", "filler"]

# Action vocabularies per decision type (PLAN 6.2)
PHRASE_ACTIONS = ["KEEP", "CUT"]
PAUSE_ACTIONS = ["KEEP", "TIGHT_40", "TIGHT_15", "REMOVE"]
ZOOM_ACTIONS = ["NONE", "Z115", "Z130"]
FILLER_ACTIONS = ["KEEP", "REMOVE"]
RHYTHM_ACTIONS = ["NONE", "Z115", "Z130", "TEXT_POP", "SFX"]
PACE_ACTIONS = ["S100", "S110", "S125"]
MUSIC_TIMING_ACTIONS = ["FROM_START", "AFTER_INTRO"]
# emphasis / hook actions are "NONE" or "CAND_<k>" (k = 0..2)


# ---------------------------------------------------------------- units.json
class Word(BaseModel):
    id: int
    idx: int
    unit_id: int
    start: float
    end: float
    text_original: str
    text_display: str
    conf: float = 1.0
    is_filler: bool = False


class Unit(BaseModel):
    id: int
    idx: int
    kind: UnitKind
    parent_unit_id: int | None = None
    start: float
    end: float
    text: str = ""
    features: dict[str, float] = Field(default_factory=dict)


class UnitsFile(BaseModel):
    schema_version: int = SCHEMA_VERSION
    feature_version: int = 1
    video_id: str
    duration: float
    video_type: VideoType = "other"
    source_map: list[dict] = Field(default_factory=list)  # multi-clip join: [{file, src_start, src_end, out_start}]
    units: list[Unit] = Field(default_factory=list)
    words: list[Word] = Field(default_factory=list)


# ---------------------------------------------------------------- EDL v3
class Point(BaseModel):
    x: float
    y: float


class OutputSpec(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: int = 30
    speed: float = 1.0


class Clip(BaseModel):
    src_start: float
    src_end: float
    zoom: float = 1.0
    crop_center: Point = Point(x=0.5, y=0.5)
    unit_ids: list[int] = Field(default_factory=list)
    speed: float = 1.0  # this clip's own pace, on top of OutputSpec.speed


class CaptionWord(BaseModel):
    t0: float  # output time
    t1: float
    text: str
    emphasis: bool = False


class CaptionLine(BaseModel):
    t0: float
    t1: float
    words: list[CaptionWord]


class Captions(BaseModel):
    enabled: bool = True
    style_id: str = "bold_pop"
    position: Literal["lower_third", "center", "top"] = "lower_third"
    lines: list[CaptionLine] = Field(default_factory=list)


class TextOverlay(BaseModel):
    id: str
    text: str
    t0: float  # output time
    t1: float
    position: Literal["top", "center", "custom"] = "top"
    y: float | None = None  # 0..1, used when position == "custom"
    style_id: str = "title"


class MusicSpec(BaseModel):
    track_id: str
    gain_db: float = -18.0
    duck_db: float = -10.0
    start_s: float = 0.0  # output time the music comes in


class SfxCue(BaseModel):
    t0: float  # output time
    name: str = "whoosh"


class AudioSpec(BaseModel):
    enhance: bool = True
    target_lufs: float = -14.0
    music: MusicSpec | None = None
    sfx: list[SfxCue] = Field(default_factory=list)


class EDL(BaseModel):
    video_id: str
    edl_version: int = 3
    output: OutputSpec = Field(default_factory=OutputSpec)
    clips: list[Clip] = Field(default_factory=list)
    captions: Captions = Field(default_factory=Captions)
    texts: list[TextOverlay] = Field(default_factory=list)
    audio: AudioSpec = Field(default_factory=AudioSpec)
    duration: float = 0.0  # output duration (after cuts and speed)


# ---------------------------------------------------------------- suggestions / events
class Suggestion(BaseModel):
    id: int | None = None
    video_id: str
    target_id: int  # unit id (phrase/pause/zoom/filler) or caption line index (emphasis) or 0 (hook)
    decision_type: DecisionType
    proposed_action: str
    candidates: list[str] = Field(default_factory=list)  # for emphasis/hook slates
    context_raw: dict[str, float] = Field(default_factory=dict)
    propensity: float = 1.0
    confidence: float = 1.0
    explored: bool = False
    source: Literal["policy", "nudge", "autopilot"] = "policy"
    policy_name: str = "linTS"
    policy_version: int = 0
    feature_version: int = 1
    superseded: bool = False


EventKind = Literal[
    "accept", "reject", "override", "implicit",
    "fine_cut", "crop", "caption_text", "text_overlay", "music", "speed", "nudge", "setting", "rating",
]


class Event(BaseModel):
    id: int | None = None
    user_id: int
    video_id: str
    kind: EventKind
    suggestion_id: int | None = None
    payload: dict = Field(default_factory=dict)  # e.g. {"final_action": "REMOVE"}
    reward: float | None = None
    weight: float | None = None
    undone: bool = False
    ts: float | None = None
