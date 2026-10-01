// Request / response shapes of the REST API (backend/app/api/io.py). The frozen contracts live in types.ts.
import type { EDL, Unit, Word, VideoType } from "./types";

export type SuggestionStatus = "pending" | "accepted" | "corrected";
export type DecisionKind = "phrase" | "pause" | "zoom" | "filler" | "emphasis";

export interface UserOut {
  id: number;
  name: string;
  created_at: number;
}

export interface SuggestionOut {
  id: number;
  decision_type: DecisionKind;
  target_id: number;
  proposed_action: string;
  current_action: string;
  candidates: string[];
  candidate_word_ids: number[];
  confidence: number;
  propensity: number;
  explored: boolean;
  source: string;
  status: SuggestionStatus;
  unit_id: number | null;
  seen: boolean;
}

export interface QueueOut {
  needs_review: number[];
  confident: number[];
}

export interface StyleOut {
  weights: Record<string, number>;
  n_events: number;
  frozen: boolean;
  /** "Your style so far: 70% Snappy · 22% Storyteller" */
  text: string;
  /** the most likely named style */
  top: string | null;
  /** its weight is above 0.8 */
  identified: boolean;
}

export interface ChangeOut {
  old_id: number;
  new_id: number;
  decision_type: string;
  target_id: number;
  from_action: string;
  to_action: string;
}

export interface Settings {
  speed: number;
  caption_style: string;
  caption_position: "lower_third" | "center" | "top";
  captions_enabled: boolean;
  audio_enhance: boolean;
}

export interface StateOut {
  status: string;
  settings: Settings;
  suggestions: SuggestionOut[];
  queue: QueueOut;
  edl: EDL | null;
  style_mixture: StyleOut | null;
  can_undo: boolean;
  can_redo: boolean;
  n_events: number;
  changes: ChangeOut[];
  message: string | null;
  events: Record<string, unknown>[];
  caption_breaks: (number | null)[];
  /** the source word ids of every caption line (line i of edl.captions.lines) */
  caption_word_ids: number[][];
  text_breaks: Record<string, number | null>;
}

export interface JobOut {
  id: string;
  kind: string;
  status: "queued" | "running" | "done" | "failed";
  error: string | null;
  meta: Record<string, unknown>;
}

export interface ExportInfo {
  event_id: number;
  duration: number;
  loudness: number;
  ts: number;
  over_platform_limit: boolean;
}

export interface ProjectSummary {
  id: number;
  video_id: string;
  user_id: number;
  title: string;
  status: "uploaded" | "processing" | "ready" | "edited" | "exported" | "failed";
  error: string | null;
  duration: number;
  video_type: VideoType;
  target_duration: number | null;
  created_at: number;
  updated_at: number;
  has_thumbnail: boolean;
  /** batch video (PLAN 8.8): "ready" = can be posted as it is, "review" = needs a look */
  autopilot: boolean;
  badge: "ready" | "review" | null;
  n_review: number;
  expected_corrections: number | null;
}

export interface ProjectOut extends ProjectSummary, Omit<StateOut, "status"> {
  source_map: Record<string, unknown>[];
  units: Unit[];
  words: Word[];
  seen_unit_ids: number[];
  exports: Record<string, ExportInfo>;
  job: JobOut | null;
  source: { width: number; height: number };
}

export interface FinalizeOut extends StateOut {
  metrics: Record<string, number>;
  implicit_events: number;
}

export interface TrackOut {
  id: string;
  name: string;
  mood: string;
  duration: number;
  source: "library" | "upload";
}

export interface ProfileOut {
  user: UserOut;
  n_projects: number;
  n_finalized: number;
  n_learning_events: number;
  style_mixture: StyleOut | null;
  /** acceptance rate of the last 2 reviewed videos, null until there is one */
  trust: number | null;
  trust_threshold: number;
  /** what the agent has learned, as sentences ("You remove pauses longer than ~0.5 s.") */
  preferences: { decision_type: string; text: string }[];
  /** editing statistics of the finished videos; null before the first */
  stats: {
    n_videos: number;
    avg_clip_seconds: number;
    cuts_per_minute: number;
    zooms_per_minute: number;
    percent_kept: number;
    fillers_removed_per_minute: number;
  } | null;
  style_labels: Record<string, { label: string; blurb: string }>;
  n_decisions_by_type: Record<string, number>;
  metrics: {
    video_index: number;
    project_id: number;
    title: string;
    acceptance: number;
    n_decisions: number;
    n_reviewed: number;
    n_accepted: number;
    n_corrected: number;
    n_manual_actions: number;
    seconds_active: number;
    duration: number;
    manual_per_minute: number;
  }[];
}

export interface Box {
  color: string;
  opacity: number;
  padding: number;
}

export interface CaptionStyle {
  label: string;
  mode: "pop" | "line" | "karaoke";
  font: string;
  font_file: string;
  size: number;
  case: "upper" | "lower" | "none";
  color: string;
  emphasis_color: string;
  fill_color?: string;
  outline_color: string;
  outline: number;
  shadow: number;
  active_scale?: number;
  max_width: number;
  box?: Box;
}

export interface TextStyle {
  font: string;
  font_file: string;
  size: number;
  case: "upper" | "lower" | "none";
  color: string;
  outline_color: string;
  outline: number;
  shadow: number;
  max_width: number;
  box?: Box;
}

export interface Preset {
  label: string;
  width: number;
  height: number;
  fps: number;
  crf: number;
  audio_kbps: number;
  max_duration_s: number;
  safe_zones: { top: number; bottom: number; right: number };
}

export interface StylesOut {
  captions: { positions: Record<string, number>; styles: Record<string, CaptionStyle> };
  texts: { positions: Record<string, number>; styles: Record<string, TextStyle> };
  presets: Record<string, Preset>;
}

export interface ExplainOut {
  suggestion_id: number;
  text: string;
  confidence: number;
  reasons: { feature: string; contribution: number; sentence: string }[];
}

// ---- event bodies
export interface ManualEditIn {
  kind: "delete_words" | "restore_words" | "set_pause" | "drag_pause" | "set_zoom" | "set_emphasis";
  word_ids?: number[];
  unit_id?: number;
  option?: string;
  out_gap?: number;
  word_id?: number | null;
  line_word_ids?: number[];
}

export interface EventIn {
  kind:
    | "accept"
    | "reject"
    | "override"
    | "manual"
    | "nudge"
    | "crop"
    | "caption_text"
    | "text_overlay"
    | "speed"
    | "setting"
    | "music";
  suggestion_id?: number;
  final_action?: string;
  edit?: ManualEditIn;
  payload?: Record<string, unknown>;
}
