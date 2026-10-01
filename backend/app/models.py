"""SQLModel tables (PLAN 16.1). The event log is the source of truth; everything derived (policy state, EDL) is a cache.

Ids exposed to the outside as `video_id` are strings "v_<pk>" (schemas.py / EDL), tables keep the integer pk.
Suggestion / Event carry a few extra columns beyond the frozen pydantic contract (they are DB-only bookkeeping):
`mean_best`, `caused_by_event`, `superseded_by_event`, `rescored_from` make live re-scoring undoable.
"""
from __future__ import annotations

import time

from sqlalchemy import JSON, Column, Index
from sqlmodel import Field, SQLModel

VIDEO_STATUSES = ("uploaded", "processing", "ready", "edited", "exported", "failed")


def now() -> float:
    return time.time()


def vid_str(pk: int) -> str:
    return f"v_{pk}"


def vid_pk(video_id: str) -> int:
    return int(video_id.split("_", 1)[1])


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    created_at: float = Field(default_factory=now)
    defaults_json: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))


class Video(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    title: str = ""
    sources_json: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    source_map_json: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    duration: float = 0.0
    video_type: str = "other"
    target_duration: float | None = None
    status: str = "uploaded"
    feature_version: int = 1
    settings_json: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    autopilot: bool = False
    created_at: float = Field(default_factory=now)
    updated_at: float = Field(default_factory=now)
    error: str | None = None


class UnitRow(SQLModel, table=True):
    __tablename__ = "unit"
    __table_args__ = (Index("ix_unit_video_unit", "video_id", "unit_id", unique=True),)
    pk: int | None = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    unit_id: int  # id inside the video's units.json (== idx)
    idx: int
    kind: str
    parent_unit_id: int | None = None
    start: float
    end: float
    text: str = ""
    features_json: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    seen: bool = False


class WordRow(SQLModel, table=True):
    __tablename__ = "word"
    __table_args__ = (Index("ix_word_video_word", "video_id", "word_id", unique=True),)
    pk: int | None = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    word_id: int
    idx: int
    unit_id: int
    start: float
    end: float
    text_original: str
    text_display: str
    conf: float = 1.0
    is_filler: bool = False


class SuggestionRow(SQLModel, table=True):
    __tablename__ = "suggestion"
    id: int | None = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    target_id: int
    decision_type: str
    proposed_action: str
    candidates_json: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    context_raw_json: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    propensity: float = 1.0
    confidence: float = 1.0
    explored: bool = False
    source: str = "policy"  # policy | nudge | autopilot
    policy_name: str = "linTS"
    policy_version: int = 0
    feature_version: int = 1
    superseded: bool = False
    created_at: float = Field(default_factory=now)
    # DB-only bookkeeping for live re-scoring (see service.rescore)
    mean_best: str | None = None  # posterior-mean argmax when the row was created
    caused_by_event: int | None = None  # event whose re-scoring created this row
    superseded_by_event: int | None = None  # event whose re-scoring replaced this row
    rescored_from: int | None = None  # the suggestion this row replaced


class EventRow(SQLModel, table=True):
    __tablename__ = "event"
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    kind: str
    suggestion_id: int | None = Field(default=None, foreign_key="suggestion.id")
    payload_json: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    reward: float | None = None
    weight: float | None = None
    undone: bool = False
    ts: float = Field(default_factory=now)


class TextOverlayRow(SQLModel, table=True):
    __tablename__ = "textoverlay"
    pk: int | None = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    overlay_id: str
    text: str
    t0: float
    t1: float
    position: str = "top"
    y: float | None = None
    style_id: str = "title"


class Comparison(SQLModel, table=True):  # U9 pairwise "which feels better?" (Phase 9)
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    video_id: int = Field(foreign_key="video.id")
    stats_a_json: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    stats_b_json: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    choice: str = ""
    ts: float = Field(default_factory=now)


class PolicyState(SQLModel, table=True):  # cache only: rebuilt from the event log when stale
    pk: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    algo: str
    blob_path: str
    n_events: int = 0
    version: int = 0


class StylePosterior(SQLModel, table=True):
    user_id: int = Field(foreign_key="user.id", primary_key=True)
    weights_json: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    frozen: bool = False
    updated_at: float = Field(default_factory=now)


class Metric(SQLModel, table=True):
    pk: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    video_id: int = Field(foreign_key="video.id", unique=True)
    video_index: int = 0
    n_decisions: int = 0
    n_reviewed: int = 0
    n_accepted: int = 0
    n_corrected: int = 0
    n_manual_actions: int = 0
    acceptance: float = 0.0
    seconds_active: float = 0.0
    exported_via_autopilot: bool = False
    ts: float = Field(default_factory=now)


class StudySession(SQLModel, table=True):  # E11 timing entries (Phase 8)
    id: int | None = Field(default=None, primary_key=True)
    participant: str
    condition: str  # roughcut | simple_editor
    clip_id: str
    seconds: float
    n_actions: int = 0
    ts: float = Field(default_factory=now)
