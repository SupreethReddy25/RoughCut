"""Edit state = a pure function of (active suggestions, non-undone events)  (PLAN 5, principle 2).

Nothing here is stored: the current decisions, settings and EDL are recomputed from the log on every request, which is
what makes undo / redo exact ("flip `undone` and recompute"). Every state-changing event carries everything needed to
replay it in its payload (`mapping`, `value`, ...), so the replay never depends on heuristics that could change later.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from app.edl.build import BuildOptions, Decisions
from app.policy.reward import Mapping, apply_mapping
from app.render.ass import load_styles
from app.schemas import AudioSpec, OutputSpec, TextOverlay, UnitsFile

SPEEDS = (1.0, 1.1, 1.15, 1.25)
CAPTION_POSITIONS = ("lower_third", "center", "top")
NEUTRAL = {"pause": "KEEP", "zoom": "NONE", "emphasis": "NONE"}  # what "reject without an alternative" leaves


@dataclass
class EditState:
    decisions: Decisions = field(default_factory=Decisions)
    speed: float = 1.0
    caption_style: str = "bold_pop"
    caption_position: str = "lower_third"
    captions_enabled: bool = True
    audio_enhance: bool = True
    texts: dict[str, TextOverlay] = field(default_factory=dict)
    nudges: list[dict] = field(default_factory=list)  # active nudge events, oldest first

    def settings(self) -> dict:
        return {"speed": self.speed, "caption_style": self.caption_style, "caption_position": self.caption_position,
                "captions_enabled": self.captions_enabled, "audio_enhance": self.audio_enhance}

    def build_options(self, output: OutputSpec | None = None, target_lufs: float = -14.0) -> BuildOptions:
        return BuildOptions(speed=self.speed, captions_enabled=self.captions_enabled, caption_style=self.caption_style,
                            caption_position=self.caption_position, texts=list(self.texts.values()),
                            audio=AudioSpec(enhance=self.audio_enhance, target_lufs=target_lufs),
                            output=output or OutputSpec())


def mapping_to_payload(m: Mapping) -> dict:
    return asdict(m)


def mapping_from_payload(d: dict) -> Mapping:
    return Mapping(**d)


def validate_setting(key: str, value) -> object:
    """Normalised value of a `setting` event, or ValueError."""
    if key == "caption_style":
        if value not in load_styles()[0]["styles"]:
            raise ValueError(f"unknown caption style {value!r}")
        return value
    if key == "caption_position":
        if value not in CAPTION_POSITIONS:
            raise ValueError(f"caption_position must be one of {CAPTION_POSITIONS}")
        return value
    if key in ("captions_enabled", "audio_enhance"):
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false")
        return value
    raise ValueError(f"unknown setting {key!r}")


def validate_speed(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError("speed must be a number") from None
    if v not in SPEEDS:
        raise ValueError(f"speed must be one of {SPEEDS}")
    return v


def base_decisions(active: list) -> Decisions:
    """The decisions the policy proposed (every proposal is pre-applied; PLAN 5)."""
    dec = Decisions()
    for s in active:
        if s.decision_type == "zoom":
            dec.zoom[s.target_id] = s.proposed_action
        elif s.decision_type == "emphasis":
            if s.proposed_action != "NONE":
                dec.emphasis.add(int(s.context_raw_json[f"c{s.proposed_action.split('_')[1]}_word_id"]))
        else:
            dec.actions[s.target_id] = s.proposed_action
    return dec


def apply_event(st: EditState, kind: str, payload: dict, uf: UnitsFile) -> None:
    """Apply one (non-undone) event to the state."""
    if kind in ("override", "reject", "fine_cut"):
        if "mapping" in payload:
            apply_mapping(st.decisions, mapping_from_payload(payload["mapping"]), uf)
    elif kind == "speed":
        st.speed = float(payload["value"])
    elif kind == "setting":
        setattr(st, payload["key"], payload["value"])
    elif kind == "caption_text":
        st.decisions.caption_text[int(payload["word_id"])] = payload["text"]
    elif kind == "crop":
        st.decisions.crop[int(payload["unit_id"])] = (float(payload["x"]), float(payload["y"]))
    elif kind == "text_overlay":
        op = payload["op"]
        if op == "delete":
            st.texts.pop(payload["overlay"]["id"], None)
        else:  # add / update
            ov = TextOverlay(**payload["overlay"])
            st.texts[ov.id] = ov
    elif kind == "nudge":
        st.nudges.append(payload)
    # accept / implicit / (other kinds): no effect on the edit


def rebuild(uf: UnitsFile, active_suggestions: list, events: list) -> EditState:
    """`events` = non-undone events in log order; `active_suggestions` = rows that are not superseded."""
    st = EditState(decisions=base_decisions(active_suggestions))
    for ev in events:
        apply_event(st, ev.kind, ev.payload_json, uf)
    return st


def current_action(st: EditState, decision_type: str, target_id: int, ctx: dict) -> str:
    """The action that is in the edit right now for a suggestion's target."""
    if decision_type == "zoom":
        return st.decisions.zoom.get(target_id, "NONE")
    if decision_type == "emphasis":
        for k in range(int(ctx.get("n_cand", 0))):
            if int(ctx[f"c{k}_word_id"]) in st.decisions.emphasis:
                return f"CAND_{k}"
        return "NONE"
    return st.decisions.action(target_id)
