"""Simulated creators (PLAN 9): each persona holds a hidden ground-truth policy over every decision type.

`ideal_action(item)` is a deterministic function of the decision's context (static unit features + history), so it
is learnable in principle by a linear model with the PLAN's features; noise / override behaviour is applied by the
simulator session, not here. Take boundaries are removed by every persona (creators almost always do).

Interpretation notes where PLAN 9 is loose:
- "low-conf" phrase = low average word confidence (garbled speech);
- Blend (50/50 Snappy/Storyteller) takes Snappy for pauses, fillers and emphasis and Storyteller for phrases and zooms;
- Minimalist keeps the most central phrases until the target duration is reached (decided per video in `prepare`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.session import Item


def _cand(item: Item, key: str) -> list[float]:
    return [item.ctx.get(f"c{k}_{key}", 0.0) for k in range(int(item.ctx.get("n_cand", 0)))]


def _argmax_arm(vals: list[float]) -> str:
    return f"CAND_{max(range(len(vals)), key=lambda k: (vals[k], -k))}"


@dataclass
class Persona:
    name: str = "persona"
    target_duration: float | None = None
    p_override: float = 0.6  # chance a reviewed disagreement is actually corrected
    noise: float = 0.05  # chance the persona's feedback label is wrong
    review_budget: float = 1.0  # fraction of decisions the persona looks at
    preferred_clip_len: float = 3.5  # seconds; used by `prefers` (Bradley-Terry comparisons, U9)
    _keep: set[int] = field(default_factory=set, repr=False)

    # -- hooks
    def for_video(self, index: int, video_type: str) -> Persona:
        """The behaviour in force for video `index` (1-based) of `video_type` (drift / type-conditional variants)."""
        return self

    def prepare(self, video) -> None:
        """Per-video setup (Minimalist decides which phrases survive)."""

    # -- decisions
    def ideal_action(self, item: Item) -> str:
        c = item.ctx
        dt = item.decision_type
        if dt == "pause":
            return "REMOVE" if c.get("is_take_boundary") == 1.0 else self.pause(c)
        if dt == "phrase":
            return self.phrase(c, item)
        if dt == "filler":
            return self.filler(c)
        if dt == "zoom":
            return self.zoom(c)
        return self.emphasis(item)

    def pause(self, c: dict[str, float]) -> str:
        raise NotImplementedError

    def phrase(self, c: dict[str, float], item: Item) -> str:
        return "CUT" if c.get("is_earlier_take", 0.0) >= 0.5 else "KEEP"

    def filler(self, c: dict[str, float]) -> str:
        return "REMOVE"

    def zoom(self, c: dict[str, float]) -> str:
        return "NONE"

    def emphasis(self, item: Item) -> str:
        return "NONE"

    def prefers(self, a: dict[str, float], b: dict[str, float]) -> bool:
        """True if window A feels better than window B (U9): closer to the persona's preferred clip length."""
        return abs(a["avg_clip_len"] - self.preferred_clip_len) < abs(b["avg_clip_len"] - self.preferred_clip_len)


@dataclass
class Snappy(Persona):
    name: str = "snappy"
    preferred_clip_len: float = 2.5

    def pause(self, c):
        return "REMOVE" if c.get("gap_seconds", 0.0) > 0.5 else "TIGHT_15"

    def phrase(self, c, item):
        fillery = c.get("filler_ratio", 0.0) >= 0.4
        return "CUT" if (c.get("is_earlier_take", 0.0) >= 0.5 or fillery or c.get("avg_word_conf", 1.0) < 0.5) else "KEEP"

    def zoom(self, c):
        return "Z115" if (c.get("loudness_z", 0.0) > 0.5 and c.get("hist_since_last_zoom", 30.0) >= 4.0) else "NONE"

    def emphasis(self, item):
        nums = _cand(item, "is_number")
        if any(nums):
            return _argmax_arm(nums)
        return _argmax_arm(_cand(item, "word_loudness_z"))


@dataclass
class Relaxed(Persona):
    name: str = "relaxed"
    preferred_clip_len: float = 6.0

    def pause(self, c):
        return "KEEP" if c.get("gap_seconds", 0.0) < 0.8 else "TIGHT_40"

    def filler(self, c):
        return "REMOVE" if (c.get("tok_um") == 1.0 or c.get("tok_uh") == 1.0) else "KEEP"


@dataclass
class Storyteller(Persona):
    name: str = "storyteller"
    preferred_clip_len: float = 4.5

    def pause(self, c):
        return "TIGHT_40" if c.get("after_sentence_end") == 1.0 else "REMOVE"

    def phrase(self, c, item):
        only_filler = c.get("filler_ratio", 0.0) >= 0.5 and c.get("n_words", 99.0) <= 2
        return "CUT" if (c.get("is_earlier_take", 0.0) >= 0.5 or only_filler) else "KEEP"

    def filler(self, c):
        keep = c.get("tok_um") == 1.0 or c.get("tok_uh") == 1.0 or c.get("tok_you_know") == 1.0
        return "REMOVE" if keep else "KEEP"

    def zoom(self, c):
        return "Z130" if (c.get("has_emphasis_punct") == 1.0 or c.get("is_first") == 1.0) else "NONE"

    def emphasis(self, item):
        proper = _cand(item, "is_capitalized_mid_sentence")
        if any(proper):
            return _argmax_arm(proper)
        tf = _cand(item, "tfidf_in_transcript")
        return _argmax_arm(tf) if tf and max(tf) >= 0.6 else "NONE"


@dataclass
class Minimalist(Persona):
    name: str = "minimalist"
    target_duration: float | None = 30.0
    preferred_clip_len: float = 2.0

    def prepare(self, video) -> None:
        speech = [u for u in video.uf.units if u.kind == "speech" and u.features.get("is_earlier_take", 0.0) < 0.5]
        speech.sort(key=lambda u: (-u.features.get("topic_centrality", 0.0), u.id))
        total, self._keep = 0.0, set()
        for u in speech:
            if total >= (self.target_duration or 1e9):
                break
            self._keep.add(u.id)
            total += u.end - u.start

    def pause(self, c):
        return "REMOVE"

    def phrase(self, c, item):
        return "KEEP" if item.target_id in self._keep else "CUT"


@dataclass
class Focused(Persona):
    name: str = "focused"
    preferred_clip_len: float = 3.5

    def pause(self, c):
        return "TIGHT_15"

    def phrase(self, c, item):
        return "CUT" if (c.get("topic_centrality", 1.0) < 0.3 or c.get("is_earlier_take", 0.0) >= 0.5) else "KEEP"

    def filler(self, c):
        return "REMOVE" if (c.get("tok_um") == 1.0 or c.get("tok_uh") == 1.0 or c.get("tok_like") == 1.0) else "KEEP"

    def zoom(self, c):
        return "Z115" if c.get("topic_centrality", 0.0) > 0.6 else "NONE"

    def emphasis(self, item):
        tf = _cand(item, "tfidf_in_transcript")
        return _argmax_arm(tf) if tf else "NONE"


@dataclass
class Blend(Persona):
    name: str = "blend"
    preferred_clip_len: float = 3.5

    def __post_init__(self):
        self._snappy, self._story = Snappy(), Storyteller()

    def pause(self, c):
        return self._snappy.pause(c)

    def phrase(self, c, item):
        return self._story.phrase(c, item)

    def filler(self, c):
        return self._snappy.filler(c)

    def zoom(self, c):
        return self._story.zoom(c)

    def emphasis(self, item):
        return self._snappy.emphasis(item)


@dataclass
class Drift(Persona):
    """Behaves as `before` until video `at` (1-based), then as `after` (PLAN E4: Relaxed -> Snappy at video 6)."""

    name: str = "drift"
    before: Persona = field(default_factory=Relaxed)
    after: Persona = field(default_factory=Snappy)
    at: int = 6

    def for_video(self, index: int, video_type: str) -> Persona:
        return self.before if index < self.at else self.after


@dataclass
class TypeConditional(Persona):
    """Snappy on vlogs, Relaxed on tutorials (and everything else)."""

    name: str = "type_conditional"
    by_type: dict[str, Persona] = field(default_factory=lambda: {"vlog": Snappy(), "tutorial": Relaxed()})
    default: Persona = field(default_factory=Relaxed)

    def for_video(self, index: int, video_type: str) -> Persona:
        return self.by_type.get(video_type, self.default)


BASE_PERSONAS = ("snappy", "relaxed", "storyteller", "minimalist", "focused", "blend")


def make_persona(name: str, **kw) -> Persona:
    table = {"snappy": Snappy, "relaxed": Relaxed, "storyteller": Storyteller, "minimalist": Minimalist,
             "focused": Focused, "blend": Blend, "drift": Drift, "type_conditional": TypeConditional}
    return table[name](**kw)
