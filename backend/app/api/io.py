"""Request / response models of the REST API (PLAN 16.2). The frozen contracts stay in schemas.py (EDL, Unit, Word)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas import EDL, Unit, Word


class UserIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class UserOut(BaseModel):
    id: int
    name: str
    created_at: float


class ManualEditIn(BaseModel):
    """A transcript / timeline / panel edit (PLAN 6.4 mapping table)."""

    kind: Literal["delete_words", "restore_words", "set_pause", "drag_pause", "set_zoom", "set_emphasis"]
    word_ids: list[int] = Field(default_factory=list)
    unit_id: int | None = None
    option: str | None = None
    out_gap: float | None = Field(default=None, ge=0)
    word_id: int | None = None  # set_emphasis: the clicked word (null clears the line's emphasis)
    line_word_ids: list[int] = Field(default_factory=list)  # set_emphasis: the words of that caption line


class EventIn(BaseModel):
    kind: Literal["accept", "reject", "override", "manual", "nudge", "crop", "caption_text", "text_overlay", "speed",
                  "setting", "music"]
    suggestion_id: int | None = None
    final_action: str | None = None
    edit: ManualEditIn | None = None
    payload: dict = Field(default_factory=dict)


class SeenIn(BaseModel):
    unit_ids: list[int] = Field(min_length=1, max_length=5000)


class FinalizeIn(BaseModel):
    seconds_active: float = Field(default=0.0, ge=0)


class RenderIn(BaseModel):
    preset: str = "reels"


class SuggestionOut(BaseModel):
    id: int
    decision_type: str
    target_id: int
    proposed_action: str
    current_action: str
    candidates: list[str]
    candidate_word_ids: list[int]
    confidence: float
    propensity: float
    explored: bool
    source: str
    status: Literal["pending", "accepted", "corrected"]
    unit_id: int | None
    seen: bool


class QueueOut(BaseModel):
    needs_review: list[int]  # suggestion ids with confidence < 0.7, least confident first
    confident: list[int]  # the rest ("spot-check")


class StyleOut(BaseModel):
    weights: dict[str, float]
    n_events: int
    frozen: bool


class ChangeOut(BaseModel):
    old_id: int
    new_id: int
    decision_type: str
    target_id: int
    from_action: str
    to_action: str


class StateOut(BaseModel):
    status: str
    settings: dict
    suggestions: list[SuggestionOut]
    queue: QueueOut
    edl: EDL | None
    style_mixture: StyleOut | None
    can_undo: bool
    can_redo: bool
    n_events: int
    changes: list[ChangeOut] = Field(default_factory=list)
    message: str | None = None
    events: list[dict] = Field(default_factory=list)  # the event rows this request created / flipped
    # where each caption line / text overlay breaks into two visual lines: computed by the export's own wrapping code
    caption_breaks: list[int | None] = Field(default_factory=list)
    text_breaks: dict[str, int | None] = Field(default_factory=dict)
    # source word ids of every caption line (the browser maps a clicked caption word to the word it emphasises)
    caption_word_ids: list[list[int]] = Field(default_factory=list)


class FinalizeOut(StateOut):
    metrics: dict
    implicit_events: int


class ProjectSummary(BaseModel):
    id: int
    video_id: str
    user_id: int
    title: str
    status: str
    error: str | None
    duration: float
    video_type: str
    target_duration: float | None
    created_at: float
    updated_at: float
    has_thumbnail: bool = False
    # autopilot (PLAN 8.8): "ready" = can be posted as it is, "review" = needs a look (n_review uncertain decisions)
    autopilot: bool = False
    badge: Literal["ready", "review"] | None = None
    n_review: int = 0
    expected_corrections: float | None = None


class ProjectOut(ProjectSummary, StateOut):
    source_map: list[dict]
    units: list[Unit]
    words: list[Word]
    seen_unit_ids: list[int]
    exports: dict
    job: dict | None = None
    source: dict  # {"width", "height"} of source.mp4 (for the preview's crop maths)


class ProfileOut(BaseModel):
    user: UserOut
    n_projects: int
    n_finalized: int
    n_learning_events: int
    style_mixture: StyleOut | None
    metrics: list[dict]
    trust: float | None = None  # explicit-acceptance rate of the last 2 reviewed videos (PLAN 8.8); None = not enough yet
    trust_threshold: float = 0.85


class TrackOut(BaseModel):
    id: str
    name: str
    mood: str = ""
    duration: float
    source: Literal["library", "upload"]
