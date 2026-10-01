import { describe, expect, it } from "vitest";
import type { SuggestionOut } from "../api-types";
import type { CaptionLine, Clip, EDL, Unit, Word } from "../types";
import { activeLine, activeWord, applyCase, HOLD, lineEnd, wordFill } from "./captions";
import { actionLabel, actionsFor, describeTarget, targetStart } from "./describe";
import { Playback, type VideoLike } from "./playback";
import { Remap } from "./remap";
import { isTypingTarget, resolveShortcut, type KeyInfo } from "./shortcuts";
import { keptWordIds, unitsBetween, wordAt } from "./words";

const line = (t0: number, words: [number, number][]): CaptionLine => ({
  t0,
  t1: words[words.length - 1][1],
  words: words.map(([a, b], i) => ({ t0: a, t1: b, text: `w${i}`, emphasis: false })),
});
const edlOf = (lines: CaptionLine[], duration = 10): EDL => ({
  video_id: "v", edl_version: 3, output: { width: 1080, height: 1920, fps: 30, speed: 1 }, clips: [],
  captions: { enabled: true, style_id: "bold_pop", position: "lower_third", lines }, texts: [],
  audio: { enhance: true, target_lufs: -14, music: null }, duration,
});

describe("captions timing (same rules as the ASS export)", () => {
  const lines = [line(1, [[1, 1.3], [1.3, 1.6], [1.6, 2]]), line(2.05, [[2.05, 2.4]]), line(5, [[5, 5.4]])];

  it("holds a line briefly after its last word but never into the next line", () => {
    expect(lineEnd(lines, 0, 10)).toBeCloseTo(2.05, 9); // 2 + 0.12 capped by the next line start
    expect(lineEnd(lines, 2, 10)).toBeCloseTo(5.4 + HOLD, 9);
    expect(lineEnd(lines, 2, 5.45)).toBeCloseTo(5.45, 9); // ... and never past the end of the video
  });

  it("finds the visible line", () => {
    const e = edlOf(lines);
    expect(activeLine(e, 0.5)).toBe(-1);
    expect(activeLine(e, 1)).toBe(0);
    expect(activeLine(e, 2.04)).toBe(0);
    expect(activeLine(e, 2.05)).toBe(1);
    expect(activeLine(e, 3)).toBe(-1);
    expect(activeLine(e, 5.2)).toBe(2);
    expect(activeLine(e, 5.6)).toBe(-1);
  });

  it("finds the active word (word 0 starts with the line) and the karaoke fill", () => {
    const l = lines[0];
    expect(activeWord(l, 0.9)).toBe(-1);
    expect(activeWord(l, 1.0)).toBe(0);
    expect(activeWord(l, 1.3)).toBe(1);
    expect(activeWord(l, 1.59)).toBe(1);
    expect(activeWord(l, 1.9)).toBe(2);
    expect(wordFill(l, 1, 1.3, 2.05)).toBe(0);
    expect(wordFill(l, 1, 1.45, 2.05)).toBeCloseTo(0.5, 9);
    expect(wordFill(l, 1, 3, 2.05)).toBe(1);
    expect(wordFill(l, 2, 1.0, 2.05)).toBe(0);
  });

  it("changes case like the style", () => {
    expect(applyCase("Hi There", "upper")).toBe("HI THERE");
    expect(applyCase("Hi There", "lower")).toBe("hi there");
    expect(applyCase("Hi There", "none")).toBe("Hi There");
  });
});

describe("words", () => {
  const w = (id: number, start: number, end: number, unit = 0): Word => ({
    id, idx: id, unit_id: unit, start, end, text_original: `t${id}`, text_display: `t${id}`, conf: 1, is_filler: false,
  });
  const clip = (a: number, b: number): Clip => ({ src_start: a, src_end: b, zoom: 1, crop_center: { x: 0.5, y: 0.5 }, unit_ids: [] });

  it("keeps words whose midpoint is inside a clip", () => {
    const words = [w(0, 0, 0.4), w(1, 0.5, 1.5), w(2, 2.0, 2.2), w(3, 3, 3.4), w(4, 5, 5.2)];
    const kept = keptWordIds(words, [clip(0.45, 2.1), clip(3.1, 6)]);
    expect([...kept].sort()).toEqual([1, 2, 3, 4]);
    expect(kept.has(0)).toBe(false); // midpoint 0.2 is before the first clip
    expect(kept.has(2)).toBe(true); // midpoint 2.1 is on the clip edge
    expect(kept.has(3)).toBe(true); // midpoint 3.2
    expect(keptWordIds(words, []).size).toBe(0);
  });

  it("finds the spoken word and the units overlapping an interval", () => {
    const words = [w(0, 0, 0.4), w(1, 0.5, 1.5), w(2, 2.0, 2.2)];
    expect(wordAt(words, 0.1)?.id).toBe(0);
    expect(wordAt(words, 1.7)).toBeNull(); // a pause
    expect(wordAt(words, 2.1)?.id).toBe(2);
    expect(wordAt(words, -1)).toBeNull();
    const units = [
      { id: 0, start: 0, end: 1 }, { id: 1, start: 1, end: 2 }, { id: 2, start: 2, end: 3 },
    ] as Unit[];
    expect(unitsBetween(units, 0.5, 1.5)).toEqual([0, 1]);
    expect(unitsBetween(units, 1.0, 1.0)).toEqual([]);
    expect(unitsBetween(units, 2.5, 0.9)).toEqual([0, 1, 2]);
  });
});

describe("describe", () => {
  const sug = (over: Partial<SuggestionOut>): SuggestionOut => ({
    id: 1, decision_type: "pause", target_id: 3, proposed_action: "REMOVE", current_action: "REMOVE", candidates: [],
    candidate_word_ids: [], confidence: 0.9, propensity: 0.9, explored: false, source: "policy", status: "pending",
    unit_id: 3, seen: false, ...over,
  });
  const units = new Map<number, Unit>([
    [3, { id: 3, idx: 3, kind: "pause", parent_unit_id: null, start: 1, end: 1.8, text: "", features: {} }],
    [4, { id: 4, idx: 4, kind: "speech", parent_unit_id: null, start: 2, end: 4, text: "Hello there, everyone.", features: {} }],
  ]);
  const words = new Map<number, Word>([[7, { id: 7, idx: 7, unit_id: 4, start: 2, end: 2.3, text_original: "Hello,", text_display: "Hello,", conf: 1, is_filler: false }]]);

  it("lists the arms in key order and labels them", () => {
    expect(actionsFor(sug({}))).toEqual(["KEEP", "TIGHT_40", "TIGHT_15", "REMOVE"]);
    expect(actionsFor(sug({ decision_type: "zoom" }))).toEqual(["NONE", "Z115", "Z130"]);
    expect(actionsFor(sug({ decision_type: "emphasis", candidates: ["NONE", "CAND_0"] }))).toEqual(["NONE", "CAND_0"]);
    expect(actionLabel(sug({ decision_type: "zoom" }), "NONE", words)).toBe("No zoom");
    expect(actionLabel(sug({}), "TIGHT_15", words)).toBe("Tighten to 0.15 s");
    const e = sug({ decision_type: "emphasis", candidate_word_ids: [7], candidates: ["NONE", "CAND_0"] });
    expect(actionLabel(e, "CAND_0", words)).toBe("Highlight “Hello”");
    expect(actionLabel(e, "NONE", words)).toBe("No highlight");
  });

  it("describes targets and finds where to seek", () => {
    expect(describeTarget(sug({}), units, words)).toBe("Pause of 0.80s");
    expect(describeTarget(sug({ decision_type: "phrase", target_id: 4 }), units, words)).toBe("“Hello there, everyone.”");
    expect(describeTarget(sug({ decision_type: "emphasis", candidate_word_ids: [7] }), units, words)).toBe("Caption word: Hello");
    expect(targetStart(sug({}), units, words)).toBe(1);
    expect(targetStart(sug({ decision_type: "emphasis", candidate_word_ids: [7] }), units, words)).toBe(2);
    expect(targetStart(sug({ target_id: 99 }), units, words)).toBeNull();
  });
});

describe("shortcuts", () => {
  const key = (k: string, o: Partial<KeyInfo> = {}): KeyInfo => ({
    key: k, ctrlKey: false, metaKey: false, shiftKey: false, altKey: false, typing: false, ...o,
  });

  it("maps the editor keys", () => {
    expect(resolveShortcut(key("a"))).toEqual({ type: "accept" });
    expect(resolveShortcut(key("A", { shiftKey: true }))).toBeNull(); // shift+a is not accept
    expect(resolveShortcut(key("A"))).toEqual({ type: "accept" }); // caps lock
    expect(resolveShortcut(key("r"))).toEqual({ type: "reject" });
    expect(resolveShortcut(key("j"))).toEqual({ type: "next" });
    expect(resolveShortcut(key(" "))).toEqual({ type: "play" });
    expect(resolveShortcut(key("1"))).toEqual({ type: "choose", index: 0 });
    expect(resolveShortcut(key("4"))).toEqual({ type: "choose", index: 3 });
    expect(resolveShortcut(key("5"))).toBeNull();
    expect(resolveShortcut(key("ArrowLeft"))).toEqual({ type: "seek", delta: -0.5 });
    expect(resolveShortcut(key("ArrowRight", { shiftKey: true }))).toEqual({ type: "seek", delta: 5 });
    expect(resolveShortcut(key("Delete"))).toEqual({ type: "delete" });
    expect(resolveShortcut(key("Backspace"))).toEqual({ type: "delete" });
    expect(resolveShortcut(key("z", { ctrlKey: true }))).toEqual({ type: "undo" });
    expect(resolveShortcut(key("Z", { ctrlKey: true, shiftKey: true }))).toEqual({ type: "redo" });
    expect(resolveShortcut(key("z", { metaKey: true, shiftKey: true }))).toEqual({ type: "redo" });
    expect(resolveShortcut(key("y", { ctrlKey: true }))).toEqual({ type: "redo" });
  });

  it("leaves keys alone while typing, on buttons, with alt, or when held down", () => {
    for (const k of ["a", "r", "j", " ", "1", "ArrowLeft", "Delete"]) expect(resolveShortcut(key(k, { typing: true })), k).toBeNull();
    expect(resolveShortcut(key("z", { ctrlKey: true, typing: true }))).toBeNull();
    expect(resolveShortcut(key(" ", { interactive: true }))).toBeNull(); // Space presses a focused button
    expect(resolveShortcut(key("a", { altKey: true }))).toBeNull();
    expect(resolveShortcut(key("a", { repeat: true }))).toBeNull();
    expect(resolveShortcut(key("Delete", { repeat: true }))).toBeNull();
    expect(resolveShortcut(key("ArrowRight", { repeat: true }))).toEqual({ type: "seek", delta: 0.5 }); // seeking may repeat
    expect(resolveShortcut(key("c", { ctrlKey: true }))).toBeNull();
  });

  it("recognises text fields", () => {
    expect(isTypingTarget(document.createElement("input"))).toBe(true);
    expect(isTypingTarget(document.createElement("textarea"))).toBe(true);
    expect(isTypingTarget(document.createElement("select"))).toBe(true);
    expect(isTypingTarget(document.createElement("div"))).toBe(false);
    expect(isTypingTarget(null)).toBe(false);
  });
});

describe("Playback over cuts", () => {
  class FakeVideo implements VideoLike {
    currentTime = 0;
    paused = true;
    playbackRate = 1;
    seeks: number[] = [];
    play() {
      this.paused = false;
    }
    pause() {
      this.paused = true;
    }
  }
  // set currentTime through the Playback API so jumps are recorded
  const spy = (v: FakeVideo) => {
    let t = v.currentTime;
    Object.defineProperty(v, "currentTime", {
      get: () => t,
      set: (x: number) => {
        t = x;
        v.seeks.push(x);
      },
    });
  };
  const make = () => {
    const v = new FakeVideo();
    spy(v);
    return { v, p: new Playback(new Remap([[1, 3], [5, 6], [8, 10]])) };
  };

  it("jumps over each cut when playing and stops at the end", () => {
    const { v, p } = make();
    v.paused = false;
    (v as { currentTime: number }).currentTime = 2.99;
    v.seeks.length = 0;
    expect(p.tick(v)).toMatchObject({ clip: 1, jumped: true, ended: false });
    expect(v.seeks).toEqual([5]);
    (v as { currentTime: number }).currentTime = 5.99;
    v.seeks.length = 0;
    expect(p.tick(v)).toMatchObject({ clip: 2, jumped: true });
    expect(v.seeks).toEqual([8]);
    (v as { currentTime: number }).currentTime = 9.995;
    expect(p.tick(v)).toMatchObject({ clip: 2, ended: true });
    expect(v.paused).toBe(true);
    expect(v.currentTime).toBe(10);
  });

  it("does not seek between contiguous clips (a zoom change inside continuous video)", () => {
    const v = new FakeVideo();
    spy(v);
    const p = new Playback(new Remap([[1, 3], [3, 5]]));
    v.paused = false;
    (v as { currentTime: number }).currentTime = 2.99;
    v.seeks.length = 0;
    expect(p.tick(v)).toMatchObject({ clip: 1, jumped: false });
    expect(v.seeks).toEqual([]);
  });

  it("skips forward when the playhead is inside a cut, also while paused", () => {
    const { v, p } = make();
    (v as { currentTime: number }).currentTime = 4;
    v.seeks.length = 0;
    expect(p.tick(v)).toMatchObject({ clip: 1, jumped: true });
    expect(v.seeks).toEqual([5]);
    (v as { currentTime: number }).currentTime = 0.2; // before the first clip
    expect(p.tick(v).clip).toBe(0);
    expect(v.currentTime).toBe(1);
    (v as { currentTime: number }).currentTime = 11; // after the last clip
    expect(p.tick(v)).toMatchObject({ clip: 2, ended: true });
  });

  it("seeks by output time and by source time", () => {
    const { v, p } = make();
    expect(p.seekOut(v, 2.5)).toBe(1);
    expect(v.currentTime).toBeCloseTo(5.5, 9);
    expect(p.seekOut(v, -3)).toBe(0);
    expect(v.currentTime).toBe(1);
    expect(p.seekOut(v, 99)).toBe(2);
    expect(v.currentTime).toBe(10);
    expect(p.seekSrc(v, 2)).toBe(0);
    expect(v.currentTime).toBe(2);
    expect(p.seekSrc(v, 4)).toBe(1); // inside a cut: the start of the next kept range
    expect(v.currentTime).toBe(5);
    expect(p.seekSrc(v, 99)).toBe(2);
    expect(v.currentTime).toBe(10);
    expect(p.outTime(v)).toBe(5);
  });

  it("copes with an empty programme", () => {
    const v = new FakeVideo();
    const p = new Playback(new Remap([]));
    expect(p.tick(v)).toEqual({ clip: -1, ended: false, jumped: false });
    expect(p.seekOut(v, 1)).toBe(-1);
    expect(p.seekSrc(v, 1)).toBe(-1);
  });
});
