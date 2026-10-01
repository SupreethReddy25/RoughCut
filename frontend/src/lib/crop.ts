// The 9:16 crop window of a clip: a port of backend/app/render/graph.py crop_rect (tested against golden fixtures),
// plus the CSS that makes the <video> show exactly that window inside the preview frame.
export type Rect = [x: number, y: number, w: number, h: number];

/** Python's round(): half to even (JS Math.round rounds halves up). */
export function pyRound(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

const even = (x: number, lo = 2): number => Math.max(lo, pyRound(x / 2) * 2);

export function cropRect(zoom: number, cx: number, cy: number, srcW: number, srcH: number, outW = 1080, outH = 1920): Rect {
  const ar = outW / outH;
  let bw = srcH * ar; // fit by height ...
  let bh = srcH;
  if (bw > srcW) {
    // ... unless the source is narrower than the output aspect: fit by width
    bw = srcW;
    bh = srcW / ar;
  }
  const z = Math.max(zoom, 1);
  const cw = Math.min(even(bw / z), srcW);
  const ch = Math.min(even(bh / z), srcH);
  const x = Math.min(Math.max(cx * srcW - cw / 2, 0), srcW - cw);
  const y = Math.min(Math.max(cy * srcH - ch / 2, 0), srcH - ch);
  return [Math.floor(Math.trunc(x) / 2) * 2, Math.floor(Math.trunc(y) / 2) * 2, cw, ch];
}

export interface VideoBox {
  width: number;
  height: number;
  transform: string;
}

/** Size/offset of the full source video inside a frame of frameW x frameH px so that `rect` fills the frame. */
export function videoBox(rect: Rect, srcW: number, srcH: number, frameW: number, frameH: number): VideoBox {
  const [x, y, w, h] = rect;
  const sx = frameW / w;
  const sy = frameH / h;
  return { width: srcW * sx, height: srcH * sy, transform: `translate(${-x * sx}px, ${-y * sy}px)` };
}

/** New crop centre after dragging the picture by (dx, dy) frame pixels (the picture follows the pointer). */
export function dragCenter(
  cx: number, cy: number, dx: number, dy: number, rect: Rect, srcW: number, srcH: number, frameW: number, frameH: number,
): [number, number] {
  const nx = cx - (dx / (frameW / rect[2])) / srcW;
  const ny = cy - (dy / (frameH / rect[3])) / srcH;
  return [Math.min(Math.max(nx, 0), 1), Math.min(Math.max(ny, 0), 1)];
}
