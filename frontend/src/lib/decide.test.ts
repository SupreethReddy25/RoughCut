import { describe, expect, it } from "vitest";
import { makeProject, suggestion } from "../test/factory";
import { acceptEvent, chooseEvent, keyOf, nextInQueue, rejectEvent, sameKey } from "./decide";
import { isVideoName, moveItem, parseTarget, validateFiles } from "./upload";

describe("what a decision means as an event", () => {
  const pending = suggestion(5, { decision_type: "pause", candidates: ["KEEP", "TIGHT_40", "TIGHT_15", "REMOVE"], proposed_action: "REMOVE", current_action: "REMOVE" });

  it("accepts and rejects only pending suggestions", () => {
    expect(acceptEvent(pending)).toEqual({ kind: "accept", suggestion_id: 5 });
    expect(rejectEvent(pending)).toEqual({ kind: "reject", suggestion_id: 5 });
    expect(acceptEvent({ ...pending, status: "accepted" })).toBeNull();
    expect(rejectEvent({ ...pending, status: "corrected" })).toBeNull();
    expect(acceptEvent(null)).toBeNull();
  });

  it("turns a chosen action into an override, and the current action into an accept", () => {
    expect(chooseEvent(pending, "TIGHT_15")).toEqual({ kind: "override", suggestion_id: 5, final_action: "TIGHT_15" });
    expect(chooseEvent(pending, "REMOVE")).toEqual({ kind: "accept", suggestion_id: 5 });
    // already decided and the same action again: nothing to send (the server would refuse an empty override)
    expect(chooseEvent({ ...pending, status: "accepted" }, "REMOVE")).toBeNull();
  });

  it("ignores keys that are not an action of the decision", () => {
    expect(chooseEvent(pending, "Z115")).toBeNull();
    expect(chooseEvent(pending, undefined)).toBeNull();
    expect(chooseEvent(null, "KEEP")).toBeNull();
  });

  it("uses the candidate list for highlight decisions", () => {
    const em = suggestion(6, { decision_type: "emphasis", candidates: ["NONE", "CAND_0", "CAND_1"], proposed_action: "NONE", current_action: "NONE" });
    expect(chooseEvent(em, "CAND_1")).toEqual({ kind: "override", suggestion_id: 6, final_action: "CAND_1" });
  });
});

describe("the review order", () => {
  const p = makeProject();

  it("starts at the least confident and wraps around", () => {
    const first = nextInQueue(p.suggestions, p.queue, null)!;
    expect(first.id).toBe(1);
    const second = nextInQueue(p.suggestions, p.queue, keyOf(first))!;
    expect(second.id).toBe(2);
    const third = nextInQueue(p.suggestions, p.queue, keyOf(second))!;
    expect(third.id).toBe(3);
    expect(nextInQueue(p.suggestions, p.queue, keyOf(third))!.id).toBe(1);
  });

  it("finds the selection again after live re-scoring gave the suggestion a new id", () => {
    const replaced = p.suggestions.map((s) => (s.id === 1 ? { ...s, id: 99 } : s));
    const queue = { needs_review: [99, 2], confident: [3] };
    expect(sameKey(replaced[0], keyOf(p.suggestions[0]))).toBe(true);
    expect(nextInQueue(replaced, queue, keyOf(p.suggestions[0]))!.id).toBe(2);
  });

  it("skips decided suggestions and copes with an empty queue", () => {
    const done = p.suggestions.map((s) => ({ ...s, status: "accepted" as const }));
    expect(nextInQueue(done, p.queue, null)).toBeNull();
    expect(nextInQueue([], { needs_review: [], confident: [] }, null)).toBeNull();
  });
});

describe("upload form rules", () => {
  it("reorders without mutating and clamps at the ends", () => {
    const l = ["a", "b", "c"];
    expect(moveItem(l, 0, 1)).toEqual(["b", "a", "c"]);
    expect(moveItem(l, 2, -1)).toEqual(["a", "c", "b"]);
    expect(moveItem(l, 0, -1)).toBe(l);
    expect(moveItem(l, 2, 1)).toBe(l);
    expect(moveItem(l, 5, 1)).toBe(l);
    expect(l).toEqual(["a", "b", "c"]);
  });

  it("validates the target length", () => {
    expect(parseTarget("")).toEqual({ value: null });
    expect(parseTarget(" 30 ")).toEqual({ value: 30 });
    expect(parseTarget("4")).toHaveProperty("error");
    expect(parseTarget("181")).toHaveProperty("error");
    expect(parseTarget("abc")).toHaveProperty("error");
  });

  it("validates the files", () => {
    expect(validateFiles([])).toMatch(/at least one/);
    expect(validateFiles([{ name: "a.txt", size: 10 }])).toMatch(/not a supported/);
    expect(validateFiles([{ name: "a.mp4", size: 0 }])).toMatch(/empty/);
    expect(validateFiles([{ name: "a.mp4", size: 400 * 2 ** 20 }, { name: "b.mov", size: 200 * 2 ** 20 }])).toMatch(/500 MB/);
    expect(validateFiles([{ name: "A.MP4", size: 10 }, { name: "b.mov", size: 10 }])).toBeNull();
    expect(isVideoName("clip.WEBM")).toBe(true);
    expect(isVideoName("clip")).toBe(false);
  });
});
