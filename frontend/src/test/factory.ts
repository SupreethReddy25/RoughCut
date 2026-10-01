// Small hand-made projects for component tests (no server needed).
import type { ProfileOut, ProjectOut, SuggestionOut } from "../api-types";
import type { Clip, Unit, Word } from "../types";

export function word(id: number, unit: number, start: number, end: number, text: string): Word {
  return { id, idx: id, unit_id: unit, start, end, text_original: text, text_display: text, conf: 0.9, is_filler: false };
}

export function unit(id: number, kind: Unit["kind"], start: number, end: number, text = ""): Unit {
  return { id, idx: id, kind, parent_unit_id: null, start, end, text, features: {} };
}

export function suggestion(id: number, over: Partial<SuggestionOut> = {}): SuggestionOut {
  return {
    id, decision_type: "phrase", target_id: id, proposed_action: "KEEP", current_action: "KEEP", candidates: ["KEEP", "CUT"],
    candidate_word_ids: [], confidence: 0.6, propensity: 0.9, explored: false, source: "policy", status: "pending",
    unit_id: id, seen: false, ...over,
  };
}

/** Three speech units (0-1 s, 1.4-2.4 s, 3-4 s) with a pause between each, kept as three clips. */
export function makeProject(over: Partial<ProjectOut> = {}): ProjectOut {
  const units = [
    unit(0, "speech", 0, 1, "hello there"),
    unit(1, "pause", 1, 1.4),
    unit(2, "speech", 1.4, 2.4, "second part"),
    unit(3, "pause", 2.4, 3),
    unit(4, "speech", 3, 4, "third part"),
  ];
  const words = [
    word(0, 0, 0, 0.5, "hello"), word(1, 0, 0.5, 1, "there"),
    word(2, 2, 1.4, 1.9, "second"), word(3, 2, 1.9, 2.4, "part"),
    word(4, 4, 3, 3.5, "third"), word(5, 4, 3.5, 4, "part"),
  ];
  const clips: Clip[] = [
    { src_start: 0, src_end: 1, zoom: 1, crop_center: { x: 0.5, y: 0.5 }, unit_ids: [0] },
    { src_start: 1.4, src_end: 2.4, zoom: 1.15, crop_center: { x: 0.5, y: 0.5 }, unit_ids: [2] },
    { src_start: 3, src_end: 4, zoom: 1, crop_center: { x: 0.5, y: 0.5 }, unit_ids: [4] },
  ];
  const suggestions = [
    suggestion(1, { decision_type: "phrase", target_id: 2, unit_id: 2, confidence: 0.4 }),
    suggestion(2, { decision_type: "pause", target_id: 3, unit_id: 3, candidates: ["KEEP", "TIGHT_40", "TIGHT_15", "REMOVE"], proposed_action: "REMOVE", current_action: "REMOVE", confidence: 0.55 }),
    suggestion(3, { decision_type: "zoom", target_id: 4, unit_id: 4, candidates: ["NONE", "Z115", "Z130"], proposed_action: "NONE", current_action: "NONE", confidence: 0.9 }),
  ];
  return {
    id: 7, video_id: "v_7", user_id: 1, title: "Test", status: "ready", error: null, duration: 4, video_type: "other",
    target_duration: null, created_at: 0, updated_at: 0, has_thumbnail: false,
    autopilot: false, badge: null, n_review: 0, expected_corrections: null,
    settings: { speed: 1, caption_style: "bold_pop", caption_position: "lower_third", captions_enabled: true, audio_enhance: true },
    suggestions, queue: { needs_review: [1, 2], confident: [3] },
    edl: {
      video_id: "v_7", edl_version: 3, output: { width: 1080, height: 1920, fps: 30, speed: 1 }, clips,
      captions: { enabled: true, style_id: "bold_pop", position: "lower_third", lines: [] }, texts: [],
      audio: { enhance: true, target_lufs: -14, music: null }, duration: 3,
    },
    style_mixture: null, can_undo: false, can_redo: false, n_events: 0, changes: [], message: null, events: [],
    caption_breaks: [], text_breaks: {}, caption_word_ids: [[0, 1], [2, 3], [4, 5]], source_map: [], units, words, seen_unit_ids: [], exports: {}, job: null,
    source: { width: 1920, height: 1080 },
    ...over,
  };
}

export function makeProfile(over: Partial<ProfileOut> = {}): ProfileOut {
  return {
    user: { id: 1, name: "Tester", created_at: 0 }, n_projects: 3, n_finalized: 2, n_learning_events: 10,
    style_mixture: null, trust: null, trust_threshold: 0.85, preferences: [], stats: null, style_labels: {},
    n_decisions_by_type: {}, metrics: [], ...over,
  };
}
