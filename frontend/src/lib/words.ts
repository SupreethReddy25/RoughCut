import type { Clip, Unit, Word } from "../types";

/** Words that are in the programme: the midpoint lies in a kept clip (same rule as the EDL builder). */
export function keptWordIds(words: Word[], clips: Clip[]): Set<number> {
  const kept = new Set<number>();
  if (clips.length === 0) return kept;
  const sorted = [...clips].sort((a, b) => a.src_start - b.src_start);
  let k = 0;
  for (const w of [...words].sort((a, b) => a.start + a.end - (b.start + b.end))) {
    const mid = (w.start + w.end) / 2;
    while (k < sorted.length && sorted[k].src_end + 1e-6 < mid) k++;
    if (k < sorted.length && sorted[k].src_start - 1e-6 <= mid) kept.add(w.id);
  }
  return kept;
}

export function groupWordsByUnit(words: Word[]): Map<number, Word[]> {
  const m = new Map<number, Word[]>();
  for (const w of words) {
    const list = m.get(w.unit_id);
    if (list) list.push(w);
    else m.set(w.unit_id, [w]);
  }
  return m;
}

export function indexUnits(units: Unit[]): Map<number, Unit> {
  return new Map(units.map((u) => [u.id, u]));
}

/** The word being spoken at source time t (the last word that started), or null in a pause. */
export function wordAt(words: Word[], t: number): Word | null {
  let lo = 0;
  let hi = words.length - 1;
  let best: Word | null = null;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (words[mid].start <= t) {
      best = words[mid];
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return best && t < best.end + 0.05 ? best : null;
}

/** Ids of the units (any kind) that overlap the source interval [a, b]. */
export function unitsBetween(units: Unit[], a: number, b: number): number[] {
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  return units.filter((u) => u.start < hi && u.end > lo).map((u) => u.id);
}
