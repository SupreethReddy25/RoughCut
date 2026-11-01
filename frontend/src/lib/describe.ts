// Human wording for suggestions and their actions (review queue, timeline, why box).
import type { SuggestionOut } from "../api-types";
import type { Unit, Word } from "../types";
import { fmtGap } from "./format";

export const ARMS: Record<string, string[]> = {
  phrase: ["KEEP", "CUT"],
  pause: ["KEEP", "TIGHT_40", "TIGHT_15", "REMOVE"],
  zoom: ["NONE", "Z115", "Z130"],
  filler: ["KEEP", "REMOVE"],
  rhythm: ["NONE", "Z115", "Z130", "TEXT_POP", "SFX"],
  pace: ["S100", "S110", "S125"],
  music_timing: ["FROM_START", "AFTER_INTRO"],
};

/** The actions the user can pick for a suggestion, in the order of the 1-4 keys. */
export function actionsFor(s: SuggestionOut): string[] {
  return s.decision_type === "emphasis" || s.decision_type === "hook" ? s.candidates : (ARMS[s.decision_type] ?? []);
}

const PLAIN: Record<string, string> = {
  KEEP: "Keep",
  CUT: "Cut",
  REMOVE: "Remove",
  TIGHT_40: "Tighten to 0.4 s",
  TIGHT_15: "Tighten to 0.15 s",
  Z115: "Zoom 1.15×",
  Z130: "Zoom 1.3×",
  TEXT_POP: "Text pop-up",
  SFX: "Sound effect",
  S100: "Normal speed",
  S110: "Speed up 1.1×",
  S125: "Speed up 1.25×",
  FROM_START: "Music from the start",
  AFTER_INTRO: "Music after the first line",
};

export const DECISION_LABEL: Record<string, string> = {
  phrase: "Phrase",
  pause: "Pause",
  zoom: "Zoom",
  filler: "Filler",
  emphasis: "Highlight",
  rhythm: "Visual beat",
  pace: "Speed",
  hook: "Opening title",
  music_timing: "Music",
};

export function actionLabel(s: SuggestionOut, action: string, wordsById: Map<number, Word>): string {
  if (s.decision_type === "zoom" && action === "NONE") return "No zoom";
  if (s.decision_type === "rhythm" && action === "NONE") return "No effect";
  if (s.decision_type === "hook") {
    if (action === "NONE") return "No title";
    const t = s.candidate_texts?.[Number(action.split("_")[1])];
    return t ? `“${t}”` : action;
  }
  if (s.decision_type === "emphasis") {
    if (action === "NONE") return "No highlight";
    const k = Number(action.split("_")[1]);
    const w = wordsById.get(s.candidate_word_ids[k]);
    return w ? `Highlight “${w.text_display.replace(/[.,;:!?]+$/, "")}”` : action;
  }
  return PLAIN[action] ?? action;
}

const clip = (t: string, n = 56) => (t.length > n ? `${t.slice(0, n - 1)}…` : t);

/** A one-line description of what a suggestion is about. */
export function describeTarget(s: SuggestionOut, units: Map<number, Unit>, wordsById: Map<number, Word>): string {
  const u = units.get(s.target_id);
  switch (s.decision_type) {
    case "pause":
      return u ? `Pause of ${fmtGap(u.end - u.start)}` : "Pause";
    case "hook":
      return "Which line opens the video";
    case "music_timing":
      return "When the music comes in";
    case "phrase":
    case "zoom":
    case "rhythm":
    case "pace":
      return u ? `“${clip(u.text)}”` : "Phrase";
    case "filler":
      return u ? `“${u.text.replace(/[.,;:!?]+$/, "")}” (filler)` : "Filler";
    case "emphasis": {
      const ws = s.candidate_word_ids.map((id) => wordsById.get(id)?.text_display.replace(/[.,;:!?]+$/, "")).filter(Boolean);
      return ws.length ? `Caption word: ${ws.join(" / ")}` : "Caption word";
    }
  }
}

/** Source time to jump to when a suggestion is selected. */
export function targetStart(s: SuggestionOut, units: Map<number, Unit>, wordsById: Map<number, Word>): number | null {
  if (s.decision_type === "hook" || s.decision_type === "music_timing") return 0;
  if (s.decision_type === "emphasis") {
    const w = wordsById.get(s.candidate_word_ids[0]);
    return w ? w.start : null;
  }
  const u = units.get(s.target_id);
  return u ? u.start : null;
}

export function confidenceColor(c: number): string {
  return c < 0.5 ? "bg-red-500" : c < 0.7 ? "bg-amber-400" : "bg-emerald-500";
}
