// Mirror of backend/app/schemas.py (FROZEN CONTRACT). Keep in sync; checked by backend/tests/test_schema_sync.py.
export const SCHEMA_VERSION = 2;

export type VideoType = "tutorial" | "vlog" | "opinion" | "other";
export type DecisionType = "phrase" | "pause" | "zoom" | "filler" | "emphasis" | "hook" | "rhythm" | "pace" | "music_timing";
export type UnitKind = "speech" | "pause" | "filler";

export const PHRASE_ACTIONS = ["KEEP", "CUT"] as const;
export const PAUSE_ACTIONS = ["KEEP", "TIGHT_40", "TIGHT_15", "REMOVE"] as const;
export const ZOOM_ACTIONS = ["NONE", "Z115", "Z130"] as const;
export const FILLER_ACTIONS = ["KEEP", "REMOVE"] as const;
export const RHYTHM_ACTIONS = ["NONE", "Z115", "Z130", "TEXT_POP", "SFX"] as const;
export const PACE_ACTIONS = ["S100", "S110", "S125"] as const;
export const MUSIC_TIMING_ACTIONS = ["FROM_START", "AFTER_INTRO"] as const;

export interface Word {
  id: number;
  idx: number;
  unit_id: number;
  start: number;
  end: number;
  text_original: string;
  text_display: string;
  conf: number;
  is_filler: boolean;
}

export interface Unit {
  id: number;
  idx: number;
  kind: UnitKind;
  parent_unit_id: number | null;
  start: number;
  end: number;
  text: string;
  features: Record<string, number>;
}

export interface UnitsFile {
  schema_version: number;
  feature_version: number;
  video_id: string;
  duration: number;
  video_type: VideoType;
  source_map: Record<string, unknown>[];
  units: Unit[];
  words: Word[];
}

export interface Point {
  x: number;
  y: number;
}

export interface OutputSpec {
  width: number;
  height: number;
  fps: number;
  speed: number;
}

export interface Clip {
  src_start: number;
  src_end: number;
  zoom: number;
  crop_center: Point;
  unit_ids: number[];
  speed?: number;
}

export interface CaptionWord {
  t0: number; // output time
  t1: number;
  text: string;
  emphasis: boolean;
}

export interface CaptionLine {
  t0: number;
  t1: number;
  words: CaptionWord[];
}

export interface Captions {
  enabled: boolean;
  style_id: string;
  position: "lower_third" | "center" | "top";
  lines: CaptionLine[];
}

export interface TextOverlay {
  id: string;
  text: string;
  t0: number; // output time
  t1: number;
  position: "top" | "center" | "custom";
  y: number | null;
  style_id: string;
}

export interface MusicSpec {
  track_id: string;
  gain_db: number;
  duck_db: number;
  start_s?: number;
}

export interface SfxCue {
  t0: number;
  name: string;
}

export interface AudioSpec {
  enhance: boolean;
  target_lufs: number;
  music: MusicSpec | null;
  sfx?: SfxCue[];
}

export interface EDL {
  video_id: string;
  edl_version: number;
  output: OutputSpec;
  clips: Clip[];
  captions: Captions;
  texts: TextOverlay[];
  audio: AudioSpec;
  duration: number;
}

export interface Suggestion {
  id: number | null;
  video_id: string;
  target_id: number;
  decision_type: DecisionType;
  proposed_action: string;
  candidates: string[];
  context_raw: Record<string, number>;
  propensity: number;
  confidence: number;
  explored: boolean;
  source: "policy" | "nudge" | "autopilot";
  policy_name: string;
  policy_version: number;
  feature_version: number;
  superseded: boolean;
}

export type EventKind =
  | "accept" | "reject" | "override" | "implicit"
  | "fine_cut" | "crop" | "caption_text" | "text_overlay" | "music" | "speed" | "nudge" | "setting" | "rating";

export interface AppEvent {
  id: number | null;
  user_id: number;
  video_id: string;
  kind: EventKind;
  suggestion_id: number | null;
  payload: Record<string, unknown>;
  reward: number | null;
  weight: number | null;
  undone: boolean;
  ts: number | null;
}
