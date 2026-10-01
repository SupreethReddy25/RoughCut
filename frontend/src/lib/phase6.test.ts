import { describe, expect, it } from "vitest";
import { makeProject, suggestion } from "../test/factory";
import { nextUncertain, reviewProgress } from "./decide";
import { groupPreferences, statCards } from "./profile";

describe("review helpers", () => {
  const p = makeProject();

  it("goes to the top of the uncertain queue, skipping what is decided", () => {
    expect(nextUncertain(p.suggestions, p.queue)!.id).toBe(1);
    const decided = p.suggestions.map((s) => (s.id === 1 ? { ...s, status: "accepted" as const } : s));
    expect(nextUncertain(decided, p.queue)!.id).toBe(2); // 1 is stale in the queue ids: skipped
    expect(nextUncertain(p.suggestions, { needs_review: [], confident: [3] })).toBeNull();
    expect(nextUncertain(p.suggestions, { needs_review: [99], confident: [] })).toBeNull(); // unknown id
  });

  it("counts decided suggestions and the uncertain ones still open", () => {
    expect(reviewProgress(p.suggestions, p.queue)).toEqual({ done: 0, total: 3, uncertain: 2 });
    const some = [suggestion(1, { status: "accepted" }), suggestion(2), suggestion(3, { status: "corrected" })];
    expect(reviewProgress(some, { needs_review: [2], confident: [] })).toEqual({ done: 2, total: 3, uncertain: 1 });
    expect(reviewProgress([], { needs_review: [], confident: [] })).toEqual({ done: 0, total: 0, uncertain: 0 });
  });
});

describe("profile helpers", () => {
  it("groups sentences by decision type in a fixed order, unknown types last", () => {
    const g = groupPreferences([
      { decision_type: "zoom", text: "z" }, { decision_type: "mystery", text: "m" }, { decision_type: "pause", text: "p1" },
      { decision_type: "emphasis", text: "e" }, { decision_type: "pause", text: "p2" },
    ]);
    expect(g.map((x) => x.type)).toEqual(["pause", "zoom", "emphasis", "mystery"]);
    expect(g[0]).toEqual({ type: "pause", label: "Pauses", texts: ["p1", "p2"] });
    expect(g[3].label).toBe("mystery");
    expect(groupPreferences([])).toEqual([]);
  });

  it("formats the statistics, and shows nothing before a first video", () => {
    expect(statCards(null)).toEqual([]);
    const cards = statCards({ n_videos: 1, avg_clip_seconds: 3.04, cuts_per_minute: 12, zooms_per_minute: 0, percent_kept: 49.5, fillers_removed_per_minute: 0.25 });
    expect(cards.map((c) => c.value)).toEqual(["3 s", "12", "0", "50%", "0.3"]);
    expect(cards).toHaveLength(5);
  });
});
