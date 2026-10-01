"""Editing statistics of a creator's finished videos (PLAN 16.3): average clip length, cuts/min, zooms/min, % kept,
fillers removed/min. Computed from the current edit of every finalized video, per minute of OUTPUT video."""
from __future__ import annotations

from dataclasses import dataclass

from app.svc.bundle import Bundle, build_edl_for

JOIN_GAP = 0.01  # seconds of source skipped between two clips for it to count as a cut


@dataclass
class VideoStats:
    out_seconds: float
    src_seconds: float
    kept_seconds: float  # source seconds that are in the edit
    n_segments: int  # stretches of video between cuts (what a creator calls a clip)
    n_cuts: int
    n_zooms: int
    n_fillers_removed: int


def video_stats(b: Bundle) -> VideoStats | None:
    """Statistics of the current edit of one video, or None when it has nothing to measure (no analysis / no clips)."""
    if b.analysis is None:
        return None
    edl, _ = build_edl_for(b)
    if edl is None or not edl.clips or edl.duration <= 0:
        return None
    cuts = zooms = 0
    prev = None
    for c in edl.clips:
        joined = prev is not None and c.src_start - prev.src_end <= JOIN_GAP
        if prev is not None and not joined:
            cuts += 1
        if c.zoom > 1.0 and not (joined and prev.zoom > 1.0):
            zooms += 1
        prev = c
    uf, dec = b.analysis.uf, b.state.decisions
    phrase_cut = {u.id for u in uf.units if u.kind == "speech" and dec.action(u.id) == "CUT"}
    fillers = sum(1 for u in uf.units if u.kind == "filler" and dec.action(u.id) == "REMOVE"
                  and u.parent_unit_id not in phrase_cut)
    kept = sum(c.src_end - c.src_start for c in edl.clips)
    return VideoStats(edl.duration, uf.duration, kept, cuts + 1, cuts, zooms, fillers)


def aggregate(videos: list[VideoStats]) -> dict | None:
    """Totals over several videos (a long video counts for more than a short one). None for no videos."""
    if not videos:
        return None
    out = sum(v.out_seconds for v in videos)
    minutes = out / 60.0
    segments = sum(v.n_segments for v in videos)
    src = sum(v.src_seconds for v in videos)
    return {
        "n_videos": len(videos),
        "avg_clip_seconds": sum(v.kept_seconds for v in videos) / max(segments, 1),
        "cuts_per_minute": sum(v.n_cuts for v in videos) / minutes,
        "zooms_per_minute": sum(v.n_zooms for v in videos) / minutes,
        "percent_kept": 100.0 * sum(v.kept_seconds for v in videos) / src if src > 0 else 0.0,
        "fillers_removed_per_minute": sum(v.n_fillers_removed for v in videos) / minutes,
    }
