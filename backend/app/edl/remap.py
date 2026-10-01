"""Source-time <-> output-time math for an EDL (PLAN 5: "the timing math lives in one place").

An EDL is an ordered list of kept source ranges plus a global speed. Output time of a source instant t that
falls in kept range k is  (sum of earlier range lengths + (t - range_k.start)) / speed.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from app.schemas import EDL

EPS = 1e-9


def subtract_ranges(total: tuple[float, float], removals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Kept ranges = total minus the union of `removals` (zero-length leftovers are dropped)."""
    lo, hi = total
    kept, cur = [], lo
    for a, b in sorted(removals):
        a, b = max(a, lo), min(b, hi)
        if b <= a:
            continue
        if a > cur + EPS:
            kept.append((cur, a))
        cur = max(cur, b)
    if hi > cur + EPS:
        kept.append((cur, hi))
    return kept


@dataclass
class Remap:
    ranges: list[tuple[float, float]]  # kept source ranges, in order, non-overlapping
    speed: float = 1.0
    _starts: list[float] = field(init=False, repr=False)
    _offsets: list[float] = field(init=False, repr=False)  # pre-speed output offset of each range

    def __post_init__(self) -> None:
        self._starts = [r[0] for r in self.ranges]
        acc, self._offsets = 0.0, []
        for a, b in self.ranges:
            self._offsets.append(acc)
            acc += b - a
        self._total = acc

    @classmethod
    def from_edl(cls, edl: EDL) -> Remap:
        return cls([(c.src_start, c.src_end) for c in edl.clips], edl.output.speed)

    @property
    def duration(self) -> float:
        """Output duration (after cuts and speed)."""
        return self._total / self.speed

    def locate(self, t: float) -> int | None:
        """Index of the kept range containing source time t, else None (t is in a cut)."""
        k = bisect.bisect_right(self._starts, t + EPS) - 1
        if k >= 0 and t <= self.ranges[k][1] + EPS:
            return k
        return None

    def is_kept(self, t: float) -> bool:
        return self.locate(t) is not None

    def src_to_out(self, t: float, snap: str = "nearest") -> float:
        """Output time of source time t. If t is inside a cut: snap='next' -> start of the next kept range,
        'prev' -> end of the previous one, 'nearest' -> whichever is closer in source time."""
        k = self.locate(t)
        if k is not None:
            a, _ = self.ranges[k]
            return (self._offsets[k] + (min(max(t, a), self.ranges[k][1]) - a)) / self.speed
        j = bisect.bisect_right(self._starts, t)  # first range starting after t
        prev_end = self._offsets[j - 1] + (self.ranges[j - 1][1] - self.ranges[j - 1][0]) if j > 0 else 0.0
        next_start = self._offsets[j] if j < len(self.ranges) else self._total
        if snap == "next":
            out = next_start
        elif snap == "prev":
            out = prev_end
        else:
            d_prev = t - self.ranges[j - 1][1] if j > 0 else float("inf")
            d_next = self.ranges[j][0] - t if j < len(self.ranges) else float("inf")
            out = prev_end if d_prev <= d_next else next_start
        return out / self.speed

    def out_to_src(self, o: float) -> float:
        """Source time of output time o (clamped to the program)."""
        x = min(max(o * self.speed, 0.0), self._total)
        k = max(0, bisect.bisect_right(self._offsets, x + EPS) - 1)
        a, b = self.ranges[k]
        return min(a + (x - self._offsets[k]), b)
