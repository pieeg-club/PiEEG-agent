// TypeScript mirrors of the JSON the Python backend emits. These match the
// tool serializers in pieeg_agent/agent/tools.py and decode_tools.py and the
// WebEngine.snapshot() assembly — keep them in sync if those change.

export interface Info {
  stream?: string;
  channels?: number;
  rate?: number;
  provider?: string;
  model?: string;
  control?: boolean;
}

export interface NeuralState {
  status?: string; // "no_data" when the cascade hasn't produced a state yet
  detail?: string;
  timestamp?: number;
  rel_bands?: Record<string, number>;
  dominant_band?: string;
  focus?: number;
  relax?: number;
  engagement?: number;
  signal_quality?: number;
  n_channels?: number;
  bad_channels?: string[];
  warming_up?: boolean;
}

export interface PerChannelBands {
  channel: string;
  [band: string]: number | string;
}

export interface Bands {
  status?: string;
  timestamp?: number;
  relative?: Record<string, number>;
  dominant?: string;
  n_channels?: number;
  units?: string;
  per_channel?: PerChannelBands[];
}

export interface ChannelQuality {
  index: number;
  label: string;
  status: string; // good | flat | rail | noisy | line
  score: number;
  rms_uv: number;
  line_ratio: number;
  rail_frac: number;
}

export interface Quality {
  status?: string;
  timestamp?: number;
  overall?: number;
  worst?: string | null;
  channels?: ChannelQuality[];
}

export interface NeuralEvent {
  timestamp: number;
  type: string;
  value: number;
  detail: string;
  severity: string;
}

export interface Events {
  count: number;
  events: NeuralEvent[];
}

export interface Artifact {
  timestamp: number;
  type: string; // blink | blink_double | jaw_clench | motion
  channel: string;
  duration_ms: number;
  confidence: number;
  detail: string;
  severity: string;
}

export interface Artifacts {
  count: number;
  artifacts: Artifact[];
}

export interface PatternState {
  name: string;
  probability: number;
  active: boolean;
}

export interface Patterns {
  patterns: PatternState[];
  active: string[];
}

export interface ConnectivityPair {
  a: string;
  b: string;
  r: number;
}

export interface PerChannelConn {
  channel: string;
  mean_abs_r: number;
  flat: boolean;
}

export interface Connectivity {
  status?: string;  // "insufficient" or "no_data" on failure
  detail?: string;
  band?: string;
  n_channels?: number;
  n_frames?: number;
  mean_connectivity?: number;
  strongest_pairs?: ConnectivityPair[];
  per_channel?: PerChannelConn[];
  most_connected?: string;
  least_connected?: string;
  method?: string;
  caveat?: string;
  labels?: string[];
  matrix?: number[][];
}

export interface Snapshot {
  state: NeuralState;
  bands: Bands;
  quality: Quality;
  events: Events;
  artifacts: Artifacts;
  patterns: Patterns;
  connectivity: Connectivity;
}

export interface PatternMeta {
  name: string;
  balanced_accuracy: number | null;
  metric: string | null;
  loaded: boolean;
}

export interface PatternList {
  count: number;
  patterns: PatternMeta[];
}

export interface ChannelImportance {
  channel: string;
  importance: number;
}

export interface PatternExplain {
  error?: string;
  known?: string[];
  name?: string;
  cross_validation?: Record<string, unknown>;
  channel_importance?: ChannelImportance[] | number[];
  ranking?: {
    top_features?: { feature: string; cohens_d: number }[] | string[];
    caveat?: string;
    [k: string]: unknown;
  };
  threshold?: number;
}

// ── chat wire events (/ws/chat) ──────────────────────────────────────────
export type ChatWireEvent =
  | { type: "token"; text: string }
  | { type: "tool_start"; name: string; arguments: Record<string, unknown> }
  | { type: "tool_result"; name: string; result: unknown }
  | {
      type: "done";
      text: string;
      tool_calls: string[];
      usage: { input_tokens: number; output_tokens: number };
      iterations: number;
    }
  | { type: "reset" }
  | { type: "error"; detail: string };

// ── training wire events (/ws/train) ─────────────────────────────────────
export interface TrainResult {
  status?: string;
  error?: string;
  name?: string;
  detail?: string;
  next?: string;
  label?: string;
  captured_frames?: number;
  totals?: { rest: number; active: number; reps: number };
  warning?: string;
  hint?: string;
  saved_to?: string;
  balanced_accuracy?: number;
  n_folds?: number;
  channel_importance?: ChannelImportance[];
  top_features?: { feature: string; cohens_d: number }[];
  caveat?: string;
}

export interface TrainReply {
  action: string;
  result: TrainResult;
}
