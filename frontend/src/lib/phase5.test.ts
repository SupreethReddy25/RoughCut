import { describe, expect, it } from "vitest";
import type { EDL } from "../types";
import { makeProject, suggestion } from "../test/factory";
import { chipsOf, emphasisEdit, lineAt } from "./emphasis";
import { dbToGain, isSpeaking, loopDistance, musicPosition, smoothVolume, targetVolume } from "./music";
import { dragOverlay, MIN_TEXT_SECONDS } from "./overlay";

const edlWithLines = (): EDL => {
  const p = makeProject();
  p.edl!.captions.lines = [
    { t0: 0.5, t1: 1.5, words: [{ t0: 0.5, t1: 0.9, text: "hello", emphasis: false }, { t0: 0.9, t1: 1.5, text: "there", emphasis: true }] },
    { t0: 2, t1: 3, words: [{ t0: 2, t1: 2.5, text: "second", emphasis: false }, { t0: 2.5, t1: 3, text: "part", emphasis: false }] },
  ];
  return p.edl!;
};

describe("caption emphasis chips", () => {
  const edl = edlWithLines();

  it("finds the line under the playhead, and the nearest one before / between lines", () => {
    expect(lineAt(edl, 0)).toBe(0); // before the first line
    expect(lineAt(edl, 1.0)).toBe(0);
    expect(lineAt(edl, 1.8)).toBe(0); // the gap after line 0
    expect(lineAt(edl, 2.0)).toBe(1);
    expect(lineAt(edl, 99)).toBe(1);
    expect(lineAt({ ...edl, captions: { ...edl.captions, lines: [] } }, 1)).toBe(-1);
  });

  it("makes chips from the line's words and the server's word ids, marking the agent's candidates", () => {
    const sug = [suggestion(9, { decision_type: "emphasis", target_id: 0, candidate_word_ids: [11, 12], candidates: ["NONE", "CAND_0", "CAND_1"] })];
    const chips = chipsOf(edl, [[10, 11], [12, 13]], sug, 0);
    expect(chips).toEqual([
      { wordId: 10, text: "hello", emphasis: false, candidate: false },
      { wordId: 11, text: "there", emphasis: true, candidate: true },
    ]);
    expect(chipsOf(edl, [[10, 11], [12, 13]], sug, 1)[0].candidate).toBe(true); // 12 is a candidate of that suggestion
  });

  it("returns nothing when the ids do not line up with the line (never guess a word)", () => {
    expect(chipsOf(edl, [[10]], [], 0)).toEqual([]);
    expect(chipsOf(edl, [], [], 0)).toEqual([]);
    expect(chipsOf(edl, [[10, 11]], [], 5)).toEqual([]);
  });

  it("highlights the clicked word, and clears when the highlighted word is clicked", () => {
    const chips = chipsOf(edl, [[10, 11], [12, 13]], [], 0);
    expect(emphasisEdit(chips, chips[0])).toEqual({ kind: "set_emphasis", word_id: 10, line_word_ids: [10, 11] });
    expect(emphasisEdit(chips, chips[1])).toEqual({ kind: "set_emphasis", word_id: null, line_word_ids: [10, 11] });
  });
});

describe("dragging a text bar", () => {
  it("moves it and keeps its length, inside the programme", () => {
    expect(dragOverlay("move", 1, 3, 0.52, 10)).toEqual([1.5, 3.5]);
    expect(dragOverlay("move", 1, 3, -5, 10)).toEqual([0, 2]);
    expect(dragOverlay("move", 1, 3, 50, 10)).toEqual([8, 10]);
  });

  it("pulls the start or the end, never shorter than the minimum or past the ends", () => {
    expect(dragOverlay("start", 1, 3, 0.5, 10)).toEqual([1.5, 3]);
    expect(dragOverlay("start", 1, 3, 9, 10)).toEqual([3 - MIN_TEXT_SECONDS, 3]);
    expect(dragOverlay("start", 1, 3, -9, 10)).toEqual([0, 3]);
    expect(dragOverlay("end", 1, 3, 1.04, 10)).toEqual([1, 4]);
    expect(dragOverlay("end", 1, 3, -9, 10)).toEqual([1, 1 + MIN_TEXT_SECONDS]);
    expect(dragOverlay("end", 1, 3, 99, 10)).toEqual([1, 10]);
  });

  it("copes with an overlay that already runs past a short programme", () => {
    expect(dragOverlay("move", 0, 3, 1, 2)).toEqual([0, 3]); // programme 2 s, bar 3 s: nowhere to go
    expect(dragOverlay("end", 0, 3, 5, 2)).toEqual([0, 3]);
  });
});

describe("music preview maths", () => {
  const edl = edlWithLines();

  it("knows when someone is speaking, with a short lead and a tail", () => {
    expect(isSpeaking(edl, 0.7)).toBe(true);
    expect(isSpeaking(edl, 0.47)).toBe(true); // lead
    expect(isSpeaking(edl, 1.7)).toBe(true); // tail after "there" (ends at 1.5)
    expect(isSpeaking(edl, 1.9)).toBe(false);
    expect(isSpeaking(edl, 4)).toBe(false);
    expect(isSpeaking(edl, 0)).toBe(false);
  });

  it("dips the volume while speaking and never exceeds 1", () => {
    expect(targetVolume(-20, -10, false)).toBeCloseTo(dbToGain(-20));
    expect(targetVolume(-20, -10, true)).toBeCloseTo(dbToGain(-30));
    expect(targetVolume(-20, 0, true)).toBeCloseTo(targetVolume(-20, 0, false));
    expect(targetVolume(6, 0, false)).toBe(1);
  });

  it("dips quickly and recovers slowly", () => {
    const down = smoothVolume(0.2, 0.05) - 0.2;
    const up = smoothVolume(0.05, 0.2) - 0.05;
    expect(down).toBeLessThan(0);
    expect(Math.abs(down)).toBeGreaterThan(Math.abs(up) * 0.5); // 0.5 * 0.15 vs 0.15 * 0.15: faster per unit of distance
    let v = 0;
    for (let i = 0; i < 100; i++) v = smoothVolume(v, 0.3);
    expect(v).toBeCloseTo(0.3, 3);
  });

  it("finds the place in a looping track", () => {
    expect(musicPosition(10, 34)).toBe(10);
    expect(musicPosition(40, 34)).toBe(6);
    expect(musicPosition(-1, 34)).toBe(33);
    expect(musicPosition(5, 0)).toBe(0);
    expect(loopDistance(0.1, 33.9, 34)).toBeCloseTo(0.2);
    expect(loopDistance(10, 12, 34)).toBe(2);
  });
});
