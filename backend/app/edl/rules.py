"""Rule-based decisions, for testing the EDL/render path. Becomes RulePolicy in Phase 2.

- pause > 0.5 s -> REMOVE, otherwise TIGHT_15 (a take boundary is always REMOVE)
- phrase flagged is_earlier_take -> CUT
- um / uh fillers -> REMOVE (only when the cut is clean: both neighbour gaps >= 30 ms)
- zoom: Z115 on every third kept phrase of >= 1.5 s (just enough variety to exercise the zoom/crop path)
- caption emphasis: one keyword per phrase (number, or a high TF-IDF non-stopword), if word features are given
"""
from __future__ import annotations

from app.edl.build import Decisions
from app.schemas import UnitsFile

PAUSE_REMOVE_OVER = 0.5
EARLIER_TAKE_CUT = 0.5
EMPHASIS_MIN_TFIDF = 0.6


def rule_decisions(uf: UnitsFile, word_features: dict[int, dict[str, float]] | None = None) -> Decisions:
    d = Decisions()
    cut = set()
    for u in uf.units:
        f = u.features
        if u.kind == "pause":
            gap = u.end - u.start
            d.actions[u.id] = "REMOVE" if (gap > PAUSE_REMOVE_OVER or f.get("is_take_boundary") == 1) else "TIGHT_15"
        elif u.kind == "speech" and f.get("is_earlier_take", 0) >= EARLIER_TAKE_CUT:
            d.actions[u.id] = "CUT"
            cut.add(u.id)
    for u in uf.units:
        if u.kind == "filler" and u.parent_unit_id not in cut:
            f = u.features
            if (f.get("tok_um") == 1 or f.get("tok_uh") == 1) and f.get("clean_gap", 0) == 1:
                d.actions[u.id] = "REMOVE"
    n_kept = 0
    for u in uf.units:
        if u.kind == "speech" and u.id not in cut:
            if n_kept % 3 == 1 and u.end - u.start >= 1.5:
                d.zoom[u.id] = "Z115"
            n_kept += 1
    if word_features:
        best: dict[int, tuple[float, int]] = {}
        for w in uf.words:
            if w.is_filler or w.unit_id in cut:
                continue
            wf = word_features.get(w.id, {})
            if wf.get("is_stopword", 0):
                continue
            score = 1.0 if wf.get("is_number", 0) else wf.get("tfidf_in_transcript", 0.0)
            if score >= EMPHASIS_MIN_TFIDF and score > best.get(w.unit_id, (0.0, -1))[0]:
                best[w.unit_id] = (score, w.id)
        d.emphasis = {wid for _, wid in best.values()}
    return d
