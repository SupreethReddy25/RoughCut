// Caption-word emphasis (PLAN D5 / 6.4): click a caption word to highlight it, click the highlighted word to clear it.
import type { ManualEditIn, SuggestionOut } from "../api-types";
import type { EDL } from "../types";

export interface Chip {
  wordId: number;
  text: string;
  emphasis: boolean;
  /** one of the words the agent considered for this line (its D5 candidates) */
  candidate: boolean;
}

/** The caption line on screen at output time t; between lines the one that just ended, before the first the first. */
export function lineAt(edl: EDL, t: number): number {
  const lines = edl.captions.lines;
  if (lines.length === 0) return -1;
  let best = 0;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].t0 <= t + 1e-6) best = i;
    else break;
  }
  return best;
}

/** The words of caption line `idx` as clickable chips. Returns [] if the ids do not line up with the EDL. */
export function chipsOf(edl: EDL, wordIds: number[][], suggestions: SuggestionOut[], idx: number): Chip[] {
  const line = edl.captions.lines[idx];
  const ids = wordIds[idx];
  if (!line || !ids || ids.length !== line.words.length) return [];
  const cands = new Set<number>();
  for (const s of suggestions) {
    if (s.decision_type === "emphasis" && s.candidate_word_ids.some((w) => ids.includes(w))) s.candidate_word_ids.forEach((w) => cands.add(w));
  }
  return line.words.map((w, i) => ({ wordId: ids[i], text: w.text, emphasis: w.emphasis, candidate: cands.has(ids[i]) }));
}

/** What a click on `chip` means: clear when it is the highlighted word, else highlight it (one word per line). */
export function emphasisEdit(chips: Chip[], chip: Chip): ManualEditIn {
  return { kind: "set_emphasis", word_id: chip.emphasis ? null : chip.wordId, line_word_ids: chips.map((c) => c.wordId) };
}
