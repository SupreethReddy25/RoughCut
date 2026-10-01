export function fmtTime(t: number): string {
  if (!isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

export function fmtDuration(t: number): string {
  if (!isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60);
  return m > 0 ? `${m}m ${Math.round(t - m * 60)}s` : `${t.toFixed(1)}s`;
}

export function fmtGap(t: number): string {
  return `${t.toFixed(t < 1 ? 2 : 1)}s`;
}

export function pct(x: number): string {
  return `${Math.round(x * 100)}%`;
}
