/**
 * Request/response models for the vsfx TypeScript SDK.
 * @module
 */

/** Terminal prediction statuses. */
export const TERMINAL_STATUSES: ReadonlySet<string> = new Set(["succeeded", "failed", "canceled"]);

/** In-flight prediction statuses. */
export const ACTIVE_STATUSES: ReadonlySet<string> = new Set(["starting", "queued", "processing"]);

/** Generation parameters for a video-to-video SFX prediction. */
export interface SfxParams {
  prompt: string;
  negativePrompt?: string;
  seed?: number;
  numInferenceSteps?: number;
  guidanceScale?: number;
  duration?: number;
  startTime?: number;
  audioMode?: "replace" | "mix" | "duck";
  sfxGainDb?: number;
  originalAudioGainDb?: number;
  duckThresholdDb?: number;
  duckRatio?: number;
  duckAttackMs?: number;
  duckReleaseMs?: number;
  targetLoudnessLufs?: number;
  truePeakDb?: number;
  videoHandling?: "copy" | "reencode";
  returnAudioOnly?: boolean;
  enableSafetyChecker?: boolean;
  webhookUrl?: string;
  metadata?: Record<string, unknown>;
}

/** Convert camelCase SDK params to the snake_case wire format. */
export function paramsToWire(params: SfxParams): Record<string, unknown> {
  const wire: Record<string, unknown> = { prompt: params.prompt };
  const optional: Array<[keyof SfxParams, string]> = [
    ["negativePrompt", "negative_prompt"],
    ["seed", "seed"],
    ["numInferenceSteps", "num_inference_steps"],
    ["guidanceScale", "guidance_scale"],
    ["duration", "duration"],
    ["startTime", "start_time"],
    ["audioMode", "audio_mode"],
    ["sfxGainDb", "sfx_gain_db"],
    ["originalAudioGainDb", "original_audio_gain_db"],
    ["duckThresholdDb", "duck_threshold_db"],
    ["duckRatio", "duck_ratio"],
    ["duckAttackMs", "duck_attack_ms"],
    ["duckReleaseMs", "duck_release_ms"],
    ["targetLoudnessLufs", "target_loudness_lufs"],
    ["truePeakDb", "true_peak_db"],
    ["videoHandling", "video_handling"],
    ["returnAudioOnly", "return_audio_only"],
    ["enableSafetyChecker", "enable_safety_checker"],
    ["webhookUrl", "webhook_url"],
    ["metadata", "metadata"],
  ];
  for (const [field, wireName] of optional) {
    const value = params[field];
    if (value !== undefined && value !== null) wire[wireName] = value;
  }
  return wire;
}

/** A prediction resource. */
export interface Prediction {
  id: string;
  status: "starting" | "queued" | "processing" | "succeeded" | "failed" | "canceled" | string;
  prompt?: string;
  created_at?: string;
  started_at?: string | null;
  completed_at?: string | null;
  output?: Record<string, unknown> | null;
  error?: Record<string, unknown> | null;
  metrics?: Record<string, unknown>;
  urls?: Record<string, string>;
  [key: string]: unknown;
}

/** Whether the prediction reached a final state. */
export function isTerminal(prediction: Prediction): boolean {
  return TERMINAL_STATUSES.has(prediction.status);
}

/** Whether the prediction is still progressing. */
export function isActive(prediction: Prediction): boolean {
  return ACTIVE_STATUSES.has(prediction.status);
}

/** Result envelope wrapper used by every endpoint. */
export interface Envelope<T = unknown> {
  code: number;
  message: string;
  data: T;
}

/** POST /uploads response payload. */
export interface UploadAck {
  url: string;
  asset_id?: string;
  path?: string;
  expires_at?: string;
  content_type?: string;
  size_bytes?: number;
  probe?: Record<string, unknown> | null;
}

/** POST /uploads/presign response payload. */
export interface UploadPresign {
  upload_url: string;
  url: string;
  path?: string;
  expires_at?: string;
}

/** GET /models/video-to-video-sfx payload. */
export interface ModelInfo extends Record<string, unknown> {
  id?: string;
}

/** GET /account/usage payload. */
export interface AccountUsage extends Record<string, unknown> {
  seconds_generated?: number;
  predictions?: number;
}

/** POST /webhooks/test payload. */
export interface WebhookTestResult {
  accepted: boolean;
  status_code: number;
  delivery?: Record<string, unknown>;
  [key: string]: unknown;
}
