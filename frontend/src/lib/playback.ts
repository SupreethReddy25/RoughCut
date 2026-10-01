// Preview playback over an EDL (PLAN 12): ONE <video> (the proxy) plays the source; whenever the playhead reaches the
// end of a kept range we jump to the start of the next one, and if it ever lands inside a cut we skip forward.
// `Playback` has no DOM dependency (it only needs a few members of the video element), so it is unit-tested.
import { Remap } from "./remap";

export interface VideoLike {
  currentTime: number;
  paused: boolean;
  playbackRate: number;
  play(): Promise<void> | void;
  pause(): void;
}

export interface TickResult {
  /** index of the clip the playhead is in (-1 when there are no clips) */
  clip: number;
  /** playback reached the end of the programme and was paused */
  ended: boolean;
  /** the video was repositioned (a jump over a cut) */
  jumped: boolean;
}

// Jump this early (about half a 30 fps frame) so a frame beyond the clip end is never shown.
export const END_MARGIN = 1 / 60;
// Two ranges closer than this are one continuous stretch of video: no seek needed (it would stutter).
const CONTIGUOUS = 0.002;

export class Playback {
  constructor(public remap: Remap) {}

  setRemap(remap: Remap): void {
    this.remap = remap;
  }

  /** Output time of the video's current position. */
  outTime(v: VideoLike): number {
    return this.remap.srcToOut(v.currentTime, "next");
  }

  tick(v: VideoLike): TickResult {
    const r = this.remap;
    const n = r.ranges.length;
    if (n === 0) return { clip: -1, ended: false, jumped: false };
    let t = v.currentTime;
    let k = r.locate(t);
    let jumped = false;
    if (k === null) {
      // inside a cut (or before the first / after the last clip): skip to the next kept range
      const next = r.ranges.findIndex(([a]) => a > t);
      if (next < 0) return this.finish(v, n - 1);
      v.currentTime = r.ranges[next][0];
      return { clip: next, ended: false, jumped: true };
    }
    if (!v.paused && t >= r.ranges[k][1] - END_MARGIN) {
      if (k + 1 >= n) return this.finish(v, k);
      const nextStart = r.ranges[k + 1][0];
      if (Math.abs(nextStart - r.ranges[k][1]) > CONTIGUOUS) {
        v.currentTime = nextStart;
        jumped = true;
      }
      k += 1;
      t = nextStart;
    }
    return { clip: k, ended: false, jumped };
  }

  private finish(v: VideoLike, lastClip: number): TickResult {
    v.pause();
    v.currentTime = this.remap.ranges[lastClip][1];
    return { clip: lastClip, ended: true, jumped: true };
  }

  /** Move the playhead to output time o. Returns the clip index it lands in. */
  seekOut(v: VideoLike, o: number): number {
    if (this.remap.ranges.length === 0) return -1;
    const src = this.remap.outToSrc(Math.min(Math.max(o, 0), this.remap.duration));
    v.currentTime = src;
    return this.remap.locate(src) ?? 0;
  }

  /** Move to a source time; inside a cut it lands on the start of the next kept range. */
  seekSrc(v: VideoLike, t: number): number {
    const r = this.remap;
    if (r.ranges.length === 0) return -1;
    const k = r.locate(t);
    if (k !== null) {
      v.currentTime = t;
      return k;
    }
    const next = r.ranges.findIndex(([a]) => a > t);
    const idx = next < 0 ? r.ranges.length - 1 : next;
    v.currentTime = next < 0 ? r.ranges[idx][1] : r.ranges[idx][0];
    return idx;
  }
}
