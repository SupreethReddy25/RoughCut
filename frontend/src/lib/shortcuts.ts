// Keyboard map of the editor (PLAN Phase 4): A accept, R reject, 1-4 choose action, J next in queue, Space play,
// arrows seek, Del delete selection, Ctrl+Z / Ctrl+Shift+Z undo / redo. Pure, so it can be tested without a DOM.
export type Shortcut =
  | { type: "play" }
  | { type: "accept" }
  | { type: "reject" }
  | { type: "choose"; index: number } // 0-based
  | { type: "next" }
  | { type: "seek"; delta: number }
  | { type: "delete" }
  | { type: "undo" }
  | { type: "redo" };

export interface KeyInfo {
  key: string;
  ctrlKey: boolean;
  metaKey: boolean;
  shiftKey: boolean;
  altKey: boolean;
  repeat?: boolean;
  /** focus is in a text field / select / contenteditable: the key belongs to it */
  typing: boolean;
  /** focus is on a button / link / summary: Space and Enter belong to it */
  interactive?: boolean;
}

export const SEEK_STEP = 0.5;
export const SEEK_BIG_STEP = 5;

export function resolveShortcut(k: KeyInfo): Shortcut | null {
  if (k.altKey) return null;
  const mod = k.ctrlKey || k.metaKey;
  if (mod) {
    if (k.typing) return null; // let the text field undo its own typing
    const key = k.key.toLowerCase();
    if (key === "z") return k.shiftKey ? { type: "redo" } : { type: "undo" };
    if (key === "y") return { type: "redo" };
    return null;
  }
  if (k.typing) return null;
  switch (k.key) {
    case " ":
    case "Spacebar":
      return k.interactive ? null : { type: "play" };
    case "ArrowLeft":
      return { type: "seek", delta: -(k.shiftKey ? SEEK_BIG_STEP : SEEK_STEP) };
    case "ArrowRight":
      return { type: "seek", delta: k.shiftKey ? SEEK_BIG_STEP : SEEK_STEP };
    case "Delete":
    case "Backspace":
      return k.repeat ? null : { type: "delete" };
  }
  if (k.shiftKey || k.repeat) return null;
  switch (k.key.toLowerCase()) {
    case "a":
      return { type: "accept" };
    case "r":
      return { type: "reject" };
    case "j":
      return { type: "next" };
  }
  if (/^[1-4]$/.test(k.key)) return { type: "choose", index: Number(k.key) - 1 };
  return null;
}

export function isTypingTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable === true || el.getAttribute("contenteditable") === "true";
}

export function isInteractiveTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  return ["BUTTON", "A", "SUMMARY"].includes(el.tagName);
}
