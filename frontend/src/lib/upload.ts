// Pure helpers of the upload form (PLAN 16.2: multi-file with reorder, type, target length).
export const VIDEO_EXTS = [".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"];
export const MAX_TOTAL_BYTES = 500 * 1024 * 1024;
export const BATCH_MAX = 5; // videos in one batch upload (PLAN 8.8)
export const TARGET_MIN = 5;
export const TARGET_MAX = 180;

/** Move the item at `i` by `delta` places (clamped); returns a new list. */
export function moveItem<T>(list: T[], i: number, delta: number): T[] {
  const j = Math.min(Math.max(i + delta, 0), list.length - 1);
  if (i < 0 || i >= list.length || i === j) return list;
  const out = [...list];
  const [x] = out.splice(i, 1);
  out.splice(j, 0, x);
  return out;
}

export const isVideoName = (name: string) => VIDEO_EXTS.some((e) => name.toLowerCase().endsWith(e));

/** "" means no target; otherwise a number of seconds in [5, 180]. Returns the value or an error text. */
export function parseTarget(raw: string): { value: number | null } | { error: string } {
  const t = raw.trim();
  if (t === "") return { value: null };
  const n = Number(t);
  if (!isFinite(n) || n < TARGET_MIN || n > TARGET_MAX) return { error: `Target length must be ${TARGET_MIN} to ${TARGET_MAX} seconds.` };
  return { value: n };
}

/** The reason the file list cannot be uploaded, or null. */
export function validateFiles(files: { name: string; size: number }[], maxFiles?: number): string | null {
  if (files.length === 0) return "Choose at least one video file.";
  if (maxFiles !== undefined && files.length > maxFiles) return `A batch can have at most ${maxFiles} videos.`;
  const bad = files.find((f) => !isVideoName(f.name));
  if (bad) return `"${bad.name}" is not a supported video file (${VIDEO_EXTS.join(", ")}).`;
  if (files.some((f) => f.size === 0)) return "One of the files is empty.";
  if (files.reduce((n, f) => n + f.size, 0) > MAX_TOTAL_BYTES) return "The files are larger than 500 MB in total.";
  return null;
}
