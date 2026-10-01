// Which caption line / word is on screen at an output time. Mirrors the timing the ASS export uses
// (backend/app/render/ass.py): a line is shown from its first word until `lineEnd` (a short hold, but never into the
// next line), and in the pop style each word of the line is "active" from its own start until the next word starts.
import type { CaptionLine, EDL } from "../types";

export const HOLD = 0.12;

export function lineEnd(lines: CaptionLine[], i: number, duration: number): number {
  const l = lines[i];
  const next = i + 1 < lines.length ? lines[i + 1].t0 : duration;
  return Math.max(l.t1, Math.min(l.t1 + HOLD, next, duration));
}

/** Index of the line visible at output time t, or -1. */
export function activeLine(edl: EDL, t: number): number {
  const lines = edl.captions.lines;
  let found = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].t0 > t) break;
    if (t < lineEnd(lines, i, edl.duration)) found = i;
  }
  return found;
}

/** The word whose window contains t (window of word 0 starts at the line start); -1 before the line starts. */
export function activeWord(line: CaptionLine, t: number): number {
  let j = -1;
  for (let i = 0; i < line.words.length; i++) {
    const start = i === 0 ? line.t0 : line.words[i].t0;
    if (start <= t) j = i;
    else break;
  }
  return j;
}

/** Karaoke fill of word j at time t: 0 before it is spoken, 1 after, linear in between (until the next word). */
export function wordFill(line: CaptionLine, j: number, t: number, end: number): number {
  const w = line.words[j];
  const next = j + 1 < line.words.length ? line.words[j + 1].t0 : end;
  const span = Math.max(next - w.t0, 1e-6);
  return Math.min(Math.max((t - w.t0) / span, 0), 1);
}

export function applyCase(text: string, mode: "upper" | "lower" | "none"): string {
  return mode === "upper" ? text.toUpperCase() : mode === "lower" ? text.toLowerCase() : text;
}
