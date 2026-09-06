/**
 * vsfx-client: TypeScript SDK for the Video-to-Video SFX service.
 *
 * ```ts
 * import { VSFXClient } from "vsfx-client";
 *
 * const client = new VSFXClient("vsfx_key", { baseUrl: "https://api.example.com" });
 * const prediction = await client.create("https://cdn/clip.mp4", {
 *   prompt: "cinematic whoosh with sub-bass impact",
 *   audioMode: "duck",
 * });
 * const done = await client.wait(prediction.id);
 * if (done.status === "succeeded") await client.download(done, "out.mp4");
 * ```
 *
 * @module
 */

export {
  AuthenticationError,
  ClientError,
  ConcurrencyLimitedError,
  IdempotencyConflictError,
  QuotaExceededError,
  RateLimitedError,
  ServerError,
  ServiceUnavailableError,
  TransportError,
  ValidationError,
  VSFXError,
  errorFromPayload,
} from "./errors.js";
export type { ErrorCode } from "./errors.js";

export {
  ACTIVE_STATUSES,
  TERMINAL_STATUSES,
  isActive,
  isTerminal,
  paramsToWire,
} from "./models.js";
export type {
  AccountUsage,
  Envelope,
  ModelInfo,
  Prediction,
  SfxParams,
  UploadAck,
  UploadPresign,
  WebhookTestResult,
} from "./models.js";

export { API_PREFIX, Transport, backoffDelay, parsePrediction, sleep } from "./transport.js";
export type { FetchLike, RequestOptions } from "./transport.js";

export { VSFXClient } from "./client.js";
export type { ClientOptions } from "./client.js";

export const VERSION = "1.0.0";
