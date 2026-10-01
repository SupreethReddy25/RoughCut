"""EDL builder (PLAN 5, 6.2, 8.2, 8.4): decisions + units.json -> schemas.EDL.   The EDL is built ONLY here.

Timing rules
- Speech edges are padded by PAD (40 ms) on both sides, so a cut never clips a word.
- A pause action keeps *T* seconds of silence between the padded speech edges (TIGHT_40 -> 0.40, TIGHT_15 ->
  0.15, REMOVE -> 0.03; KEEP leaves the pause untouched). The silence is taken half from the end of the pause's
  head and half from its tail, so room tone on both sides of the cut stays natural.
- When a cut phrase / filler sits between two kept words, the join keeps min(target T of the pauses in it).
- A filler cut runs from previous word end + 20 ms to next word start - 20 ms (PLAN 12).
- Clip edges are snapped to the 30 fps frame grid (so rendered length == EDL length).
- Clips are maximal continuous source ranges; they are also split where the zoom changes between phrases
  (a zoom is a per-clip property, so a zoom change is a "punch-in" cut at the midpoint of the gap).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from app.edl.remap import Remap, subtract_ranges
from app.pipeline.vision import FaceSample
from app.schemas import (
    EDL,
    AudioSpec,
    CaptionLine,
    Captions,
    CaptionWord,
    Clip,
    OutputSpec,
    Point,
    TextOverlay,
    UnitsFile,
    Word,
)

PAD = 0.04
FILLER_PAD = 0.02
PAUSE_TARGET = {"TIGHT_40": 0.40, "TIGHT_15": 0.15, "REMOVE": 0.03}
ZOOM_VALUE = {"NONE": 1.0, "Z115": 1.15, "Z130": 1.30}

LINE_MAX_WORDS = 4
LINE_MAX_SECONDS = 1.4
LINE_BREAK_GAP = 0.25  # output-time silence between two words that always starts a new caption line
_SENTENCE_END = (".", "?", "!", "…")
_CLAUSE_END = (",", ";", ":", "—")


@dataclass
class Decisions:
    """Everything the policy (or a person) decided. Missing entries default to KEEP / NONE."""

    actions: dict[int, str] = field(default_factory=dict)  # unit id -> KEEP|CUT / KEEP|TIGHT_*|REMOVE / KEEP|REMOVE
    zoom: dict[int, str] = field(default_factory=dict)  # speech unit id -> NONE|Z115|Z130
    emphasis: set[int] = field(default_factory=set)  # word ids highlighted in captions (D5)
    # manual transcript edits that are not decisions (PLAN 6.4): extra word-level cuts / restores.
    fine_cuts: set[int] = field(default_factory=set)  # word ids cut by a small manual deletion (not learned)
    restored: set[int] = field(default_factory=set)  # word ids kept although their phrase/filler is cut

    def action(self, unit_id: int) -> str:
        return self.actions.get(unit_id, "KEEP")


@dataclass
class BuildOptions:
    speed: float = 1.0
    captions_enabled: bool = True
    caption_style: str = "bold_pop"
    caption_position: str = "lower_third"
    texts: list[TextOverlay] = field(default_factory=list)
    audio: AudioSpec = field(default_factory=AudioSpec)
    output: OutputSpec = field(default_factory=OutputSpec)
    pad: float = PAD


def _r(x: float) -> float:
    return round(x, 3)


# --------------------------------------------------------------------------- cut geometry
def _removed_words(uf: UnitsFile, dec: Decisions) -> tuple[list[bool], set[int]]:
    """removed[i] per word, plus the set of word idxs removed *as fillers* (not as part of a cut phrase)."""
    removed = [False] * len(uf.words)
    cut_speech = {u.id for u in uf.units if u.kind == "speech" and dec.action(u.id) == "CUT"}
    by_unit: dict[int, list[int]] = {}
    for i, w in enumerate(uf.words):
        by_unit.setdefault(w.unit_id, []).append(i)
        if w.unit_id in cut_speech:
            removed[i] = True
    filler_removed: set[int] = set()
    for u in uf.units:
        if u.kind != "filler" or dec.action(u.id) != "REMOVE" or u.parent_unit_id in cut_speech:
            continue
        for i in by_unit.get(u.parent_unit_id, []):
            w = uf.words[i]
            if w.is_filler and w.start >= u.start - 1e-3 and w.end <= u.end + 1e-3:
                removed[i] = True
                filler_removed.add(i)
    for i in dec.fine_cuts:
        if 0 <= i < len(removed):
            removed[i] = True
    for i in dec.restored:
        if 0 <= i < len(removed):
            removed[i] = False
            filler_removed.discard(i)
    return removed, filler_removed


def removed_word_ids(uf: UnitsFile, dec: Decisions) -> set[int]:
    """Ids of the words that are currently cut (by phrase / filler decisions and manual word-level edits)."""
    removed, _ = _removed_words(uf, dec)
    return {uf.words[i].id for i, r in enumerate(removed) if r}


def _pause_target(p, dec: Decisions) -> float:
    act = dec.action(p.id)
    return p.end - p.start if act == "KEEP" else PAUSE_TARGET[act]


def _removals(uf: UnitsFile, dec: Decisions, removed: list[bool], filler_removed: set[int],
              pad: float) -> tuple[list[tuple[float, float]], list[int]]:
    words = uf.words
    kept = [i for i, r in enumerate(removed) if not r]
    if not kept:
        return [(0.0, uf.duration)], kept
    pauses = [u for u in uf.units if u.kind == "pause"]
    cuts: list[tuple[float, float]] = []
    first, last = words[kept[0]], words[kept[-1]]
    cuts.append((0.0, max(0.0, first.start - pad)))
    cuts.append((min(uf.duration, last.end + pad), uf.duration))
    for a, b in pairwise(kept):
        wa, wb = words[a], words[b]
        gone = list(range(a + 1, b))
        between = [p for p in pauses if p.start >= wa.end - 1e-3 and p.end <= wb.start + 1e-3]
        if not gone:
            if not between or all(dec.action(p.id) == "KEEP" for p in between):
                continue  # continuous
            T = min(_pause_target(p, dec) for p in between)
            lo, hi = wa.end + pad + T / 2, wb.start - pad - T / 2
        elif not between and all(i in filler_removed for i in gone):
            lo = min(wa.end + FILLER_PAD, words[gone[0]].start)
            hi = max(wb.start - FILLER_PAD, words[gone[-1]].end)
        else:
            T = min((_pause_target(p, dec) for p in between), default=0.0)
            lo, hi = wa.end + pad + T / 2, wb.start - pad - T / 2
            lo, hi = min(lo, words[gone[0]].start), max(hi, words[gone[-1]].end)
        if hi > lo:
            cuts.append((lo, hi))
    return cuts, kept


# --------------------------------------------------------------------------- crop / zoom helpers
def _median_center(samples: list[FaceSample], t0: float, t1: float) -> tuple[float, float] | None:
    hit = [s for s in samples if t0 - 1e-6 <= s.t <= t1 + 1e-6]
    if not hit:
        return None
    return statistics.median(s.cx for s in hit), statistics.median(s.cy for s in hit)


def _split_on_zoom(ranges: list[tuple[float, float]], kept_words: list[Word], zoom_of_unit: dict[int, float]
                   ) -> list[tuple[float, float]]:
    """Split kept ranges where consecutive phrases (both inside the same range) use a different zoom."""
    splits: list[float] = []
    prev: Word | None = None
    for w in kept_words:
        if prev is not None and w.unit_id != prev.unit_id and \
                zoom_of_unit.get(w.unit_id, 1.0) != zoom_of_unit.get(prev.unit_id, 1.0):
            splits.append((prev.end + w.start) / 2)
        prev = w
    out = []
    for a, b in ranges:
        pts = [a] + [s for s in splits if a + 1e-6 < s < b - 1e-6] + [b]
        out += list(pairwise(pts))
    return out


# --------------------------------------------------------------------------- captions
def _display(text: str) -> str:
    return text.strip().strip('"“”').rstrip(".,;:…—-").strip()


def group_caption_lines(items: list[dict]) -> list[list[dict]]:
    """items: [{text, t0, t1, ...}] in output time -> lines of <= 4 words / <= 1.4 s, broken at pauses
    (>= 0.25 s of silence), sentence ends, and (once a line has 2+ words) commas/clause ends."""
    lines: list[list[dict]] = []
    cur: list[dict] = []
    for it in items:
        if cur:
            prev = cur[-1]
            raw = prev["raw"]
            brk = (it["t0"] - prev["t1"] >= LINE_BREAK_GAP or len(cur) >= LINE_MAX_WORDS
                   or it["t1"] - cur[0]["t0"] > LINE_MAX_SECONDS or raw.endswith(_SENTENCE_END)
                   or (len(cur) >= 2 and raw.endswith(_CLAUSE_END)))
            if brk:
                lines.append(cur)
                cur = []
        cur.append(it)
    if cur:
        lines.append(cur)
    return lines


# --------------------------------------------------------------------------- main entry
def build_edl(uf: UnitsFile, dec: Decisions, face_samples: list[FaceSample] | None = None,
              options: BuildOptions | None = None) -> EDL:
    return build_edl_with_lines(uf, dec, face_samples, options)[0]


def build_edl_with_lines(uf: UnitsFile, dec: Decisions, face_samples: list[FaceSample] | None = None,
                         options: BuildOptions | None = None) -> tuple[EDL, list[list[int]]]:
    """Like build_edl, plus the word ids of every caption line (line i of the EDL <-> line_word_ids[i])."""
    opt = options or BuildOptions()
    face_samples = face_samples or []
    removed, filler_removed = _removed_words(uf, dec)
    cuts, kept_idx = _removals(uf, dec, removed, filler_removed, opt.pad)
    kept_words = [uf.words[i] for i in kept_idx]
    total = uf.duration  # never past the end of the media (a clip beyond the last frame would render short)
    ranges = subtract_ranges((0.0, total), cuts)

    zoom_of_unit = {u.id: ZOOM_VALUE[dec.zoom.get(u.id, "NONE")] for u in uf.units if u.kind == "speech"}
    ranges = _split_on_zoom(ranges, kept_words, zoom_of_unit)
    # snap clip edges to the output frame grid: ffmpeg's trim works in whole frames, so off-grid edges would
    # make every clip up to one frame longer than the EDL says
    fps = opt.output.fps
    last_frame = int(total * fps + 1e-6)
    snapped = [(round(round(a * fps) / fps, 5), round(min(round(b * fps), last_frame) / fps, 5)) for a, b in ranges]
    ranges = [(a, b) for a, b in snapped if b > a]

    units_by_kind = {k: [u for u in uf.units if u.kind == k] for k in ("speech", "pause", "filler")}
    clips: list[Clip] = []
    last_center = (0.5, 0.5)
    for a, b in ranges:
        inside = [w for w in kept_words if a - 1e-6 <= (w.start + w.end) / 2 <= b + 1e-6]
        sp_ids = list(dict.fromkeys(w.unit_id for w in inside))
        ids = list(sp_ids)
        ids += [u.id for u in units_by_kind["pause"] if min(u.end, b) - max(u.start, a) > 1e-6]
        ids += [u.id for u in units_by_kind["filler"]
                if dec.action(u.id) == "KEEP" and u.start >= a - 1e-6 and u.end <= b + 1e-6]
        c = _median_center(face_samples, a, b) or last_center
        last_center = c
        clips.append(Clip(src_start=a, src_end=b, zoom=zoom_of_unit.get(sp_ids[0], 1.0) if sp_ids else 1.0,
                          crop_center=Point(x=round(c[0], 4), y=round(c[1], 4)), unit_ids=sorted(ids)))

    remap = Remap(ranges, opt.speed)
    items = []
    for w in kept_words:
        o0 = remap.src_to_out(w.start, "next")
        o1 = max(remap.src_to_out(w.end, "prev"), o0 + 0.02)
        items.append({"id": w.id, "raw": w.text_display, "text": _display(w.text_display) or w.text_display,
                      "t0": _r(o0), "t1": _r(o1), "emph": w.id in dec.emphasis})
    lines, line_ids = [], []
    for grp in group_caption_lines(items):
        line_ids.append([g["id"] for g in grp])
        lines.append(CaptionLine(
            t0=grp[0]["t0"], t1=grp[-1]["t1"],
            words=[CaptionWord(t0=g["t0"], t1=g["t1"], text=g["text"], emphasis=g["emph"]) for g in grp]))

    out = opt.output.model_copy(update={"speed": opt.speed})
    edl = EDL(
        video_id=uf.video_id, output=out, clips=clips,
        captions=Captions(enabled=opt.captions_enabled, style_id=opt.caption_style,
                          position=opt.caption_position, lines=lines),  # type: ignore[arg-type]
        texts=list(opt.texts), audio=opt.audio, duration=_r(remap.duration),
    )
    return edl, line_ids


# --------------------------------------------------------------------------- loading processed data
def load_processed(video_dir: str | Path):
    """-> (UnitsFile, face_samples, word_features) from a data/processed/{id} folder."""
    import json

    d = Path(video_dir)
    uf = UnitsFile.model_validate_json((d / "units.json").read_text(encoding="utf-8"))
    samples: list[FaceSample] = []
    ft = d / "face_track.json"
    if ft.exists():
        samples = [FaceSample(**s) for s in json.loads(ft.read_text(encoding="utf-8"))["samples"]]
    wf: dict[int, dict[str, float]] = {}
    wp = d / "word_features.json"
    if wp.exists():
        wf = {int(k): v for k, v in json.loads(wp.read_text(encoding="utf-8")).items()}
    return uf, samples, wf
