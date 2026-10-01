// The browser's approximation of the export's background music (PLAN 8.6): the track loops under the voice and dips
// while someone is speaking (the export uses a sidechain compressor; here the speech regions come from the EDL).
import type { EDL } from "../types";

export const GAIN_MIN = -40;
export const GAIN_MAX = -6;
export const DUCK_MIN = -24;
export const DUCK_MAX = 0;
export const PREVIEW_SECONDS = 10;
const DUCK_LEAD = 0.05; // start dipping slightly before the word, like the compressor's attack
const DUCK_TAIL = 0.3; // and recover slowly after it (the release)

export const dbToGain = (db: number) => Math.pow(10, db / 20);

/** Is a caption word being spoken at output time t? */
export function isSpeaking(edl: EDL, t: number): boolean {
  for (const line of edl.captions.lines) {
    if (t < line.t0 - DUCK_LEAD) break;
    if (t > line.t1 + DUCK_TAIL) continue;
    for (const w of line.words) if (t >= w.t0 - DUCK_LEAD && t <= w.t1 + DUCK_TAIL) return true;
  }
  return false;
}

/** The volume (0..1) the music element should have: base level, dipped by the duck depth while speaking. */
export function targetVolume(gainDb: number, duckDb: number, speaking: boolean): number {
  return Math.min(1, dbToGain(gainDb + (speaking ? duckDb : 0))); // relative to the track file, like the export
}

/** One smoothing step towards the target (attack faster than release), so the dip does not click. */
export function smoothVolume(current: number, target: number): number {
  const k = target < current ? 0.5 : 0.15;
  return Math.min(1, Math.max(0, current + (target - current) * k));
}

/** Where in a looping track of `length` seconds the music is when the programme is at output time t. */
export function musicPosition(t: number, length: number): number {
  return length > 0 ? ((t % length) + length) % length : 0;
}

/** Circular distance between two positions in a loop (so 0.1 vs 33.9 of a 34 s loop is 0.2, not 33.8). */
export function loopDistance(a: number, b: number, length: number): number {
  const d = Math.abs(a - b) % Math.max(length, 1e-9);
  return Math.min(d, length - d);
}
