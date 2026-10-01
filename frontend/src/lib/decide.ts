// What a keypress / click on a suggestion means as an event (PLAN 6.4). Pure, so the rules are unit-tested.
import type { EventIn, QueueOut, SuggestionOut } from "../api-types";
import { actionsFor } from "./describe";

export interface Key {
  decision_type: string;
  target_id: number;
}

export const keyOf = (s: SuggestionOut): Key => ({ decision_type: s.decision_type, target_id: s.target_id });
export const sameKey = (s: SuggestionOut, k: Key | null) => !!k && s.decision_type === k.decision_type && s.target_id === k.target_id;

export function acceptEvent(s: SuggestionOut | null): EventIn | null {
  return s && s.status === "pending" ? { kind: "accept", suggestion_id: s.id } : null;
}

export function rejectEvent(s: SuggestionOut | null): EventIn | null {
  return s && s.status === "pending" ? { kind: "reject", suggestion_id: s.id } : null;
}

/** Picking the action that is already in the edit is an accept (the server refuses a no-op override). */
export function chooseEvent(s: SuggestionOut | null, action: string | undefined): EventIn | null {
  if (!s || action === undefined || !actionsFor(s).includes(action)) return null;
  if (action === s.current_action) return acceptEvent(s);
  return { kind: "override", suggestion_id: s.id, final_action: action };
}

/** The pending suggestion after `from` in the review order (needs-review first, then the confident rest); wraps. */
export function nextInQueue(suggestions: SuggestionOut[], queue: QueueOut, from: Key | null): SuggestionOut | null {
  const byId = new Map(suggestions.map((s) => [s.id, s]));
  const order = [...queue.needs_review, ...queue.confident].map((id) => byId.get(id)).filter((s): s is SuggestionOut => !!s && s.status === "pending");
  if (order.length === 0) return null;
  const i = order.findIndex((s) => sameKey(s, from));
  return order[(i + 1) % order.length];
}
