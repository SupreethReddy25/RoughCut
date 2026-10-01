// Source-time <-> output-time maths of an EDL: a line-by-line port of backend/app/edl/remap.py (same results, tested
// against golden fixtures generated from the Python code). The browser preview and the ffmpeg export therefore agree.
export type Snap = "nearest" | "next" | "prev";
export type Range = [number, number];
const EPS = 1e-9;

function bisectRight(a: number[], x: number): number {
  let lo = 0;
  let hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (x < a[mid]) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

export class Remap {
  readonly starts: number[];
  readonly offsets: number[]; // pre-speed output offset of each range
  readonly total: number;

  constructor(
    readonly ranges: Range[],
    readonly speed = 1,
  ) {
    this.starts = ranges.map((r) => r[0]);
    let acc = 0;
    this.offsets = [];
    for (const [a, b] of ranges) {
      this.offsets.push(acc);
      acc += b - a;
    }
    this.total = acc;
  }

  /** Output duration (after cuts and speed). */
  get duration(): number {
    return this.total / this.speed;
  }

  /** Index of the kept range containing source time t, or null when t is inside a cut. */
  locate(t: number): number | null {
    const k = bisectRight(this.starts, t + EPS) - 1;
    if (k >= 0 && t <= this.ranges[k][1] + EPS) return k;
    return null;
  }

  isKept(t: number): boolean {
    return this.locate(t) !== null;
  }

  srcToOut(t: number, snap: Snap = "nearest"): number {
    const k = this.locate(t);
    if (k !== null) {
      const [a, b] = this.ranges[k];
      return (this.offsets[k] + (Math.min(Math.max(t, a), b) - a)) / this.speed;
    }
    const j = bisectRight(this.starts, t); // first range starting after t
    const prevEnd = j > 0 ? this.offsets[j - 1] + (this.ranges[j - 1][1] - this.ranges[j - 1][0]) : 0;
    const nextStart = j < this.ranges.length ? this.offsets[j] : this.total;
    let out: number;
    if (snap === "next") out = nextStart;
    else if (snap === "prev") out = prevEnd;
    else {
      const dPrev = j > 0 ? t - this.ranges[j - 1][1] : Infinity;
      const dNext = j < this.ranges.length ? this.ranges[j][0] - t : Infinity;
      out = dPrev <= dNext ? prevEnd : nextStart;
    }
    return out / this.speed;
  }

  /** Source time of output time o (clamped to the program). */
  outToSrc(o: number): number {
    if (this.ranges.length === 0) throw new Error("empty program: no kept ranges");
    const x = Math.min(Math.max(o * this.speed, 0), this.total);
    const k = Math.max(0, bisectRight(this.offsets, x + EPS) - 1);
    const [a, b] = this.ranges[k];
    return Math.min(a + (x - this.offsets[k]), b);
  }
}
