// Dragging a text bar on the timeline's text track (PLAN 8.7): move it, or pull one of its ends.
export type DragMode = "move" | "start" | "end";

export const MIN_TEXT_SECONDS = 0.3;
const round = (x: number) => Math.round(x * 10) / 10;

/**
 * New [t0, t1] (output seconds, 0.1 s grid) after dragging by `dx` seconds. A bar never leaves [0, duration], never
 * gets shorter than 0.3 s, and keeps its length when moved.
 */
export function dragOverlay(mode: DragMode, t0: number, t1: number, dx: number, duration: number): [number, number] {
  const end = Math.max(duration, t1, MIN_TEXT_SECONDS);
  if (mode === "move") {
    const len = t1 - t0;
    const a = Math.min(Math.max(t0 + dx, 0), Math.max(end - len, 0));
    return [round(a), round(a + len)];
  }
  if (mode === "start") {
    const a = Math.min(Math.max(t0 + dx, 0), t1 - MIN_TEXT_SECONDS);
    return [round(a), t1];
  }
  const b = Math.min(Math.max(t1 + dx, t0 + MIN_TEXT_SECONDS), end);
  return [t0, round(b)];
}
