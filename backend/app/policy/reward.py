"""Reward mapping (PLAN 6.4): suggestions *and* manual edits -> bandit updates.  Weights live here so they can be ablated.

| Event                                              | Update                                      |
| Accept (explicit)                                  | proposed arm r = +1                         |
| Reject on binary (D1, D4)                          | proposed r = -1; the other arm r = +1       |
| Reject on multi-arm (D2, D3, D5), no alternative   | proposed r = -1 only                        |
| Override to a'                                     | proposed r = -1; a' r = +1                  |
| Untouched at Finalize, unit was seen               | proposed r = +0.5 (implicit)                |
| Untouched, never seen                              | no update                                   |

Manual edits (transcript / timeline / panels) are first *classified* (`classify_manual_edit`) into either an
override of a learnable decision, a `fine_cut` (applied to the EDL, logged, NOT learned) or an edit-only event;
overrides then go through the same math. `apply_mapping` applies the same classification to the decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.edl.build import PAD, PAUSE_TARGET, Decisions, removed_word_ids
from app.policy import context
from app.schemas import PAUSE_ACTIONS, UnitsFile

BINARY = ("phrase", "filler")
LEARNED_KINDS = ("accept", "reject", "override", "implicit")
PHRASE_DELETE_FRACTION = 0.7  # deleting >= 70% of a phrase's speech is a phrase cut (D1 override)


@dataclass(frozen=True)
class RewardConfig:
    accept: float = 1.0
    reject: float = -1.0
    override_new: float = 1.0  # reward of the arm the user switched to
    implicit: float = 0.5  # reward of an untouched-but-seen proposal
    implicit_weight: float = 1.0  # E7 ablation knob
    learn_manual: bool = True  # E7 ablation: map manual edits to feedback or ignore them


DEFAULT = RewardConfig()


@dataclass
class Update:
    arm: str
    reward: float
    weight: float = 1.0


def updates_for(kind: str, decision_type: str, proposed: str, arms: list[str], final: str | None = None,
                cfg: RewardConfig = DEFAULT, manual: bool = False) -> list[Update]:
    """Bandit updates implied by one event on a suggestion. Non-learning kinds return []."""
    if kind not in LEARNED_KINDS or (manual and not cfg.learn_manual):
        return []
    if kind == "implicit":
        return [Update(proposed, cfg.implicit, cfg.implicit_weight)]
    if kind == "accept":
        return [Update(proposed, cfg.accept)]
    if kind == "override" and final is not None and final != proposed:
        if final not in arms:
            raise ValueError(f"override target {final!r} is not an arm of {decision_type}")
        return [Update(proposed, cfg.reject), Update(final, cfg.override_new)]
    if kind == "override":  # "override" to what was already proposed == accept
        return [Update(proposed, cfg.accept)]
    # reject
    if decision_type in BINARY:
        other = [a for a in arms if a != proposed]
        return [Update(proposed, cfg.reject)] + [Update(a, cfg.override_new) for a in other]
    if final is not None and final != proposed:
        return [Update(proposed, cfg.reject), Update(final, cfg.override_new)]
    return [Update(proposed, cfg.reject)]


# --------------------------------------------------------------------------- global nudges (PLAN 8.10)
def nudge_offsets(name: str, magnitude: float = 0.5) -> dict[str, dict[str, float]]:
    """Per-decision-type arm score offsets for a global nudge button (applied for this video only)."""
    snappy = {"pause": {"TIGHT_15": 1.0, "REMOVE": 1.0}, "phrase": {"CUT": 1.0}, "filler": {"REMOVE": 1.0}}
    zooms = {"zoom": {"Z115": 1.0, "Z130": 1.0}}
    table = {"snappier": (snappy, 1), "calmer": (snappy, -1), "more_zooms": (zooms, 1), "fewer_zooms": (zooms, -1)}
    if name not in table:
        raise ValueError(f"unknown nudge {name!r}")
    base, sign = table[name]
    return {dt: {a: sign * magnitude * v for a, v in arms.items()} for dt, arms in base.items()}


# --------------------------------------------------------------------------- manual edits
@dataclass
class ManualEdit:
    kind: str  # delete_words | restore_words | set_pause | drag_pause | set_zoom | set_emphasis | <edit-only kinds>
    word_ids: list[int] = field(default_factory=list)
    unit_id: int | None = None  # pause / speech unit id
    option: str | None = None  # new action name (set_pause / set_zoom)
    out_gap: float | None = None  # drag_pause: the silence the user dragged to (seconds)
    line_idx: int | None = None  # set_emphasis
    word_id: int | None = None  # set_emphasis: the clicked word (None = clear)
    line_word_ids: list[int] = field(default_factory=list)  # set_emphasis: words of that caption line
    candidate_word_ids: list[int] = field(default_factory=list)  # set_emphasis: the line's D5 candidates in order


@dataclass
class Mapping:
    kind: str  # "override" | "fine_cut" | "edit_only"
    decision_type: str | None = None
    target_id: int | None = None
    final_action: str | None = None
    word_ids: list[int] = field(default_factory=list)
    word_id: int | None = None  # emphasis: the word that gets highlighted (None = clear)
    op: str = "cut"  # fine_cut: "cut" | "restore"
    note: str = ""


EDIT_ONLY = {"boundary_nudge", "crop", "caption_text", "text_overlay", "music", "speed"}


def _units(uf: UnitsFile):
    speech = {u.id: u for u in uf.units if u.kind == "speech"}
    fillers: dict[int, list] = {}
    for u in uf.units:
        if u.kind == "filler":
            fillers.setdefault(u.parent_unit_id, []).append(u)
    words_of: dict[int, list] = {}
    for w in uf.words:
        words_of.setdefault(w.unit_id, []).append(w)
    return speech, fillers, words_of


def _dur(words) -> float:
    return sum(w.end - w.start for w in words)


def filler_word_ids(uf: UnitsFile, fu) -> set[int]:
    return {w.id for w in uf.words if w.unit_id == fu.parent_unit_id and w.is_filler
            and w.start >= fu.start - 1e-3 and w.end <= fu.end + 1e-3}


def snap_pause_option(gap: float, out_gap: float) -> str:
    """Nearest pause option to a dragged-to silence (compared on the silence each option really leaves)."""
    best, best_d = "KEEP", float("inf")
    for opt in PAUSE_ACTIONS:
        left = gap if opt == "KEEP" else min(gap, 2 * PAD + PAUSE_TARGET[opt])
        if abs(left - out_gap) < best_d - 1e-12:
            best, best_d = opt, abs(left - out_gap)
    return best


def classify_manual_edit(edit: ManualEdit, uf: UnitsFile, dec: Decisions) -> list[Mapping]:
    """Map one manual edit to feedback rows of PLAN 6.4 (see module docstring)."""
    if edit.kind in EDIT_ONLY:
        return [Mapping("edit_only", note=edit.kind)]
    speech, fillers, words_of = _units(uf)
    removed_ids = removed_word_ids(uf, dec)
    by_id = {w.id: w for w in uf.words}

    if edit.kind in ("delete_words", "restore_words"):
        deleting = edit.kind == "delete_words"
        ids = {i for i in edit.word_ids if i in by_id}
        changed = ids - removed_ids if deleting else ids & removed_ids
        out: list[Mapping] = []
        for uid in speech:
            mine = [w for w in words_of.get(uid, []) if w.id in changed]
            if not mine:
                continue
            all_words = words_of[uid]
            cut = dec.action(uid) == "CUT"
            now = removed_ids & {w.id for w in all_words}
            after = (now | changed) if deleting else (now - changed)
            gone_frac = _dur([w for w in all_words if w.id in after]) / max(_dur(all_words), 1e-9)
            handled: set[int] = set()
            if deleting and not cut and gone_frac >= PHRASE_DELETE_FRACTION:
                out.append(Mapping("override", "phrase", uid, "CUT", [w.id for w in mine], note="delete >=70%"))
                continue
            if not deleting and cut and (1 - gone_frac) >= PHRASE_DELETE_FRACTION:
                out.append(Mapping("override", "phrase", uid, "KEEP", [w.id for w in mine], note="restore >=70%"))
                continue
            for fu in fillers.get(uid, []):
                fw = filler_word_ids(uf, fu)
                if not fw & changed:
                    continue
                if deleting and fw <= (now | changed) and not fw <= now and dec.action(fu.id) != "REMOVE" and not cut:
                    out.append(Mapping("override", "filler", fu.id, "REMOVE", sorted(fw), note="delete filler"))
                    handled |= fw
                elif not deleting and fw <= changed and dec.action(fu.id) == "REMOVE" and not cut:
                    out.append(Mapping("override", "filler", fu.id, "KEEP", sorted(fw), note="restore filler"))
                    handled |= fw
            rest = sorted(w.id for w in mine if w.id not in handled)
            if rest:
                out.append(Mapping("fine_cut", None, uid, None, rest, op="cut" if deleting else "restore",
                                   note="small manual edit: applied, not learned"))
        return out or [Mapping("edit_only", note="no effective change")]

    if edit.kind == "set_pause":
        if edit.unit_id is None or edit.option not in PAUSE_ACTIONS:
            raise ValueError("set_pause needs unit_id and a pause option")
        return [Mapping("override", "pause", edit.unit_id, edit.option)]
    if edit.kind == "drag_pause":
        p = next((u for u in uf.units if u.id == edit.unit_id and u.kind == "pause"), None)
        if p is None or edit.out_gap is None:
            raise ValueError("drag_pause needs a pause unit_id and out_gap")
        return [Mapping("override", "pause", p.id, snap_pause_option(p.end - p.start, edit.out_gap))]
    if edit.kind == "set_zoom":
        if edit.unit_id is None or edit.option not in context.ARMS["zoom"]:
            raise ValueError("set_zoom needs unit_id and a zoom option")
        return [Mapping("override", "zoom", edit.unit_id, edit.option)]
    if edit.kind == "set_emphasis":
        if edit.line_idx is None:
            raise ValueError("set_emphasis needs line_idx")
        if edit.word_id is None:
            return [Mapping("override", "emphasis", edit.line_idx, "NONE", list(edit.line_word_ids))]
        if edit.word_id in edit.candidate_word_ids:
            k = edit.candidate_word_ids.index(edit.word_id)
            return [Mapping("override", "emphasis", edit.line_idx, f"CAND_{k}", list(edit.line_word_ids),
                            word_id=edit.word_id)]
        return [Mapping("edit_only", "emphasis", edit.line_idx, None, list(edit.line_word_ids), word_id=edit.word_id,
                        note="emphasised word is not one of the line's candidates: applied, not learned")]
    raise ValueError(f"unknown manual edit kind {edit.kind!r}")


def apply_mapping(dec: Decisions, m: Mapping, uf: UnitsFile) -> None:
    """Apply a classified edit to the decisions (in place)."""
    if m.kind == "override":
        if m.decision_type == "zoom":
            dec.zoom[m.target_id] = m.final_action
        elif m.decision_type == "emphasis":
            dec.emphasis -= set(m.word_ids)
            if m.word_id is not None:
                dec.emphasis.add(m.word_id)
        else:
            dec.actions[m.target_id] = m.final_action
            if m.decision_type == "phrase":  # the whole phrase follows the decision: drop word-level leftovers
                ids = {w.id for w in uf.words if w.unit_id == m.target_id}
                dec.fine_cuts -= ids
                dec.restored -= ids
            elif m.decision_type == "filler":
                dec.fine_cuts -= set(m.word_ids)
                dec.restored -= set(m.word_ids)
    elif m.kind == "edit_only" and m.decision_type == "emphasis":  # applied to the captions, not learned
        dec.emphasis -= set(m.word_ids)
        if m.word_id is not None:
            dec.emphasis.add(m.word_id)
    elif m.kind == "fine_cut":
        ids = set(m.word_ids)
        if m.op == "cut":
            dec.fine_cuts |= ids
            dec.restored -= ids
        else:
            dec.fine_cuts -= ids
            dec.restored |= ids
