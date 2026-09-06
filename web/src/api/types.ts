export type PredictionStatus =
  | "starting"
  | "queued"
  | "processing"
  | "succeeded"
  | "failed"
  | "canceled"
  | string;

export interface Prediction {
  id: string;
  status: PredictionStatus;
  prompt?: string;
  created_at?: string;
  completed_at?: string | null;
  output?: Record<string, string> | null;
  error?: { error_code?: string; message?: string } | null;
  metrics?: Record<string, number>;
  urls?: Record<string, string>;
}

export interface Envelope<T> {
  code: number;
  message: string;
  data: T;
}

export interface UploadAck {
  url: string;
  asset_id?: string;
  expires_at?: string;
  content_type?: string;
  size_bytes?: number;
  probe?: Record<string, unknown> | null;
}

export interface UsageSnapshot {
  seconds_generated?: number;
  predictions?: number;
  [key: string]: unknown;
}

export type AudioMode = "replace" | "mix" | "duck";
export type VideoHandling = "copy" | "reencode";

export interface SfxRequestOptions {
  prompt: string;
  negative_prompt?: string;
  seed?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  duration?: number;
  start_time?: number;
  audio_mode?: AudioMode;
  sfx_gain_db?: number;
  original_audio_gain_db?: number;
  duck_threshold_db?: number;
  duck_ratio?: number;
  duck_attack_ms?: number;
  duck_release_ms?: number;
  target_loudness_lufs?: number;
  true_peak_db?: number;
  video_handling?: VideoHandling;
  return_audio_only?: boolean;
  enable_safety_checker?: boolean;
  webhook_url?: string;
  metadata?: Record<string, unknown>;
}
