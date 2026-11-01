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
from app.schemas import AudioSpec, MusicSpec, OutputSpec, TextOverlay, UnitsFile

SPEEDS = (1.0, 1.1, 1.15, 1.25)
CAPTION_POSITIONS = ("lower_third", "center", "top")
NEUTRAL = {"pause": "KEEP", "zoom": "NONE", "emphasis": "NONE"}  # what "reject without an alternative" leaves
GAIN_RANGE = (-40.0, -6.0)  # background music level under the voice, dB
DUCK_RANGE = (-24.0, 0.0)  # how far the music dips while the voice speaks, dB (0 = no ducking)
HOOK_ID = "hook"  # the suggested title (PLAN 8.7); a normal overlay once it is in the state
HOOK_MAX_WORDS = 8
HOOK_SECONDS = 3.0
# the settings a user's earlier choices carry over to a new project (PLAN 8.2 / 8.6 / 8.10 "remembered default")
DEFAULT_KEYS = ("speed", "caption_style", "caption_position", "captions_enabled", "audio_enhance", "music")


@dataclass
class EditState:
    decisions: Decisions = field(default_factory=Decisions)
    speed: float = 1.0
    caption_style: str = "bold_pop"
    caption_position: str = "lower_third"
    captions_enabled: bool = True
    audio_enhance: bool = True
    music: MusicSpec | None = None
    texts: dict[str, TextOverlay] = field(default_factory=dict)
    nudges: list[dict] = field(default_factory=list)  # active nudge events, oldest first

    def settings(self) -> dict:
        return {"speed": self.speed, "caption_style": self.caption_style, "caption_position": self.caption_position,
                "captions_enabled": self.captions_enabled, "audio_enhance": self.audio_enhance}

    def build_options(self, output: OutputSpec | None = None, target_lufs: float = -14.0) -> BuildOptions:
        return BuildOptions(speed=self.speed, captions_enabled=self.captions_enabled, caption_style=self.caption_style,
                            caption_position=self.caption_position, texts=list(self.texts.values()),
                            audio=AudioSpec(enhance=self.audio_enhance, target_lufs=target_lufs, music=self.music),
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


def validate_music(value) -> dict | None:
    """The normalised payload of a `music` event: None (no music) or {track_id, gain_db, duck_db}; ValueError if bad."""
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("track_id"), str) or not value["track_id"]:
        raise ValueError("music needs a track_id (or null to remove the music)")
    out = {"track_id": value["track_id"]}
    for key, (lo, hi), default in (("gain_db", GAIN_RANGE, -18.0), ("duck_db", DUCK_RANGE, -10.0)):
        raw = value.get(key, default)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not lo <= float(raw) <= hi:
            raise ValueError(f"{key} must be a number from {lo:g} to {hi:g}")
        out[key] = round(float(raw), 1)
    return out


def sanitize_defaults(d: dict | None) -> dict:
    """The part of a stored defaults dict that is still valid (a style may have been renamed since it was saved)."""
    out: dict = {}
    for key, val in (d or {}).items():
        if key not in DEFAULT_KEYS:
            continue
        try:
            if key == "speed":
                out[key] = validate_speed(val)
            elif key == "music":
                out[key] = validate_music(val)
            else:
                out[key] = validate_setting(key, val)
        except ValueError:
            continue
    return out


def hook_overlay(uf: UnitsFile) -> TextOverlay | None:
    """The suggested title (PLAN 8.7): the first sentence of the talk, at most 8 words, shown for the first 3 s.
    Fillers ("So,") and abandoned takes are left out, so the title reads like what remains after the edit."""
    skip = {u.id for u in uf.units if u.kind == "speech" and u.features.get("is_earlier_take", 0.0) >= 0.5}
    words: list[str] = []
    for w in sorted(uf.words, key=lambda w: w.idx):  # across phrases: a sentence
        if w.is_filler or w.unit_id in skip:
            continue
        words.append(w.text_display.strip())
        if len(words) >= HOOK_MAX_WORDS or words[-1].endswith((".", "?", "!", "…")):
            break
    text = " ".join(x for x in words if x).rstrip(",;:.!?…- ").strip()
    if not text:
        return None
    text = text[0].upper() + text[1:]
    return TextOverlay(id=HOOK_ID, text=text, t0=0.0, t1=HOOK_SECONDS, position="top", style_id="title")


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
    elif kind == "music":
        st.music = MusicSpec(**payload["music"]) if payload.get("music") else None
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


def rebuild(uf: UnitsFile, active_suggestions: list, events: list, defaults: dict | None = None) -> EditState:
    """`events` = non-undone events in log order; `active_suggestions` = rows that are not superseded.
    `defaults` = the user's remembered settings, as they were when the project was created: the starting point
    the events then change. A new project also starts with the suggested title (a deletion of it is an event)."""
    st = EditState(decisions=base_decisions(active_suggestions))
    d = sanitize_defaults(defaults)
    for key in ("speed", "caption_style", "caption_position", "captions_enabled", "audio_enhance"):
        if key in d:
            setattr(st, key, d[key])
    if d.get("music"):
        st.music = MusicSpec(**d["music"])
    hook = hook_overlay(uf)
    if hook is not None:
        st.texts[hook.id] = hook
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
