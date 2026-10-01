// Display helpers of the profile page (PLAN 16.3): preference sentences grouped by decision, the statistics cards.
import type { ProfileOut } from "../api-types";

const ORDER = ["pause", "phrase", "filler", "zoom", "emphasis"];
const LABEL: Record<string, string> = { pause: "Pauses", phrase: "Phrases", filler: "Fillers", zoom: "Zooms", emphasis: "Captions" };

export interface PrefGroup {
  type: string;
  label: string;
  texts: string[];
}

/** The sentences grouped by what they are about, in a fixed order (unknown types last). */
export function groupPreferences(prefs: ProfileOut["preferences"]): PrefGroup[] {
  const by = new Map<string, string[]>();
  for (const p of prefs) by.set(p.decision_type, [...(by.get(p.decision_type) ?? []), p.text]);
  const rank = (t: string) => (ORDER.includes(t) ? ORDER.indexOf(t) : ORDER.length);
  return [...by.entries()].sort((a, b) => rank(a[0]) - rank(b[0])).map(([type, texts]) => ({ type, label: LABEL[type] ?? type, texts }));
}

const one = (x: number) => (Math.round(x * 10) / 10).toString();

/** The statistics as label / value cards; none before a first finished video. */
export function statCards(stats: ProfileOut["stats"]): { label: string; value: string }[] {
  if (!stats) return [];
  return [
    { label: "average clip length", value: `${one(stats.avg_clip_seconds)} s` },
    { label: "cuts per minute", value: one(stats.cuts_per_minute) },
    { label: "zooms per minute", value: one(stats.zooms_per_minute) },
    { label: "of the footage kept", value: `${Math.round(stats.percent_kept)}%` },
    { label: "fillers removed per minute", value: one(stats.fillers_removed_per_minute) },
  ];
}
