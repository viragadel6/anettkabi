/**
 * Typed errors raised by the vsfx TypeScript SDK.
 * @module
 */

/** Machine-readable error codes from the service taxonomy. */
export type ErrorCode =
  | "invalid_request"
  | "missing_video"
  | "invalid_video_url"
  | "unsupported_media_type"
  | "video_too_large"
  | "video_too_long"
  | "video_too_short"
  | "no_video_stream"
  | "corrupt_media"
  | "download_failed"
  | "download_forbidden_host"
  | "prompt_too_long"
  | "prompt_blocked"
  | "seed_out_of_range"
  | "parameter_out_of_range"
  | "unauthorized"
  | "forbidden_scope"
  | "rate_limited"
  | "concurrency_limited"
  | "quota_exceeded"
  | "idempotency_key_conflict"
  | "prediction_not_found"
  | "prediction_not_cancelable"
  | "weights_unavailable"
  | "inference_failed"
  | "inference_timeout"
  | "gpu_out_of_memory"
  | "mux_failed"
  | "storage_failed"
  | "internal_error"
  | "transport_error";

/** Base class for every SDK failure. */
export class VSFXError extends Error {
  readonly errorCode: ErrorCode;
  readonly httpStatus: number;
  readonly requestId: string;
  readonly details: Record<string, unknown>;
  readonly retryAfter: number | null;

  constructor(
    errorCode: ErrorCode,
    httpStatus: number,
    message: string,
    requestId = "",
    details: Record<string, unknown> = {},
    retryAfter: number | null = null,
  ) {
    super(message);
    this.name = "VSFXError";
    this.errorCode = errorCode;
    this.httpStatus = httpStatus;
    this.requestId = requestId;
    this.details = details;
    this.retryAfter = retryAfter;
  }

  /** Whether an exponential-backoff retry is appropriate. */
  get retryable(): boolean {
    if (this instanceof RateLimitedError || this instanceof ConcurrencyLimitedError) return true;
    if (this instanceof ServerError) return !(this instanceof QuotaExceededError);
    return this.httpStatus >= 500;
  }
}

/** 4xx family. */
export class ClientError extends VSFXError {
  constructor(errorCode: ErrorCode, httpStatus: number, message: string, requestId = "", details: Record<string, unknown> = {}, retryAfter: number | null = null) {
    super(errorCode, httpStatus, message, requestId, details, retryAfter);
    this.name = "ClientError";
  }
}

/** Request rejected by validation. */
export class ValidationError extends ClientError {
  constructor(errorCode: ErrorCode, httpStatus: number, message: string, requestId = "", details: Record<string, unknown> = {}, retryAfter: number | null = null) {
    super(errorCode, httpStatus, message, requestId, details, retryAfter);
    this.name = "ValidationError";
  }
}

/** Missing/invalid API key. */
export class AuthenticationError extends ClientError {
  constructor(errorCode: ErrorCode, httpStatus: number, message: string, requestId = "", details: Record<string, unknown> = {}, retryAfter: number | null = null) {
    super(errorCode, httpStatus, message, requestId, details, retryAfter);
    this.name = "AuthenticationError";
  }
}

/** Rate budget exhausted (429). */
export class RateLimitedError extends ClientError {
  constructor(errorCode: ErrorCode, httpStatus: number, message: string, requestId = "", details: Record<string, unknown> = {}, retryAfter: number | null = null) {
    super(errorCode, httpStatus, message, requestId, details, retryAfter);
    this.name = "RateLimitedError";
  }
}

/** Too many concurrent predictions (429). */
export class ConcurrencyLimitedError extends ClientError {
  constructor(errorCode: ErrorCode, httpStatus: number, message: string, requestId = "", details: Record<string, unknown> = {}, retryAfter: number | null = null) {
    super(errorCode, httpStatus, message, requestId, details, retryAfter);
    this.name = "ConcurrencyLimitedError";
  }
}

/** Monthly quota exhausted (402). */
export class QuotaExceededError extends ClientError {
  constructor(errorCode: ErrorCode, httpStatus: number, message: string, requestId = "", details: Record<string, unknown> = {}, retryAfter: number | null = null) {
    super(errorCode, httpStatus, message, requestId, details, retryAfter);
    this.name = "QuotaExceededError";
  }
}

/** Idempotency key reused with a different payload (409). */
export class IdempotencyConflictError extends ClientError {
  constructor(errorCode: ErrorCode, httpStatus: number, message: string, requestId = "", details: Record<string, unknown> = {}, retryAfter: number | null = null) {
    super(errorCode, httpStatus, message, requestId, details, retryAfter);
    this.name = "IdempotencyConflictError";
  }
}

/** 5xx family. */
export class ServerError extends VSFXError {
  constructor(errorCode: ErrorCode, httpStatus: number, message: string, requestId = "", details: Record<string, unknown> = {}, retryAfter: number | null = null) {
    super(errorCode, httpStatus, message, requestId, details, retryAfter);
    this.name = "ServerError";
  }
}

/** Weights unavailable / GPU saturation (503). */
export class ServiceUnavailableError extends ServerError {
  constructor(errorCode: ErrorCode, httpStatus: number, message: string, requestId = "", details: Record<string, unknown> = {}, retryAfter: number | null = null) {
    super(errorCode, httpStatus, message, requestId, details, retryAfter);
    this.name = "ServiceUnavailableError";
  }
}

/** Network-level failure (DNS, refused, timeout). */
export class TransportError extends VSFXError {
  constructor(message: string) {
    super("transport_error", 0, message);
    this.name = "TransportError";
  }
}

const CODE_TO_CLASS: Partial<Record<ErrorCode, new (errorCode: ErrorCode, httpStatus: number, message: string, requestId?: string, details?: Record<string, unknown>, retryAfter?: number | null) => VSFXError>> = {
  unauthorized: AuthenticationError,
  forbidden_scope: AuthenticationError,
  rate_limited: RateLimitedError,
  concurrency_limited: ConcurrencyLimitedError,
  quota_exceeded: QuotaExceededError,
  idempotency_key_conflict: IdempotencyConflictError,
  weights_unavailable: ServiceUnavailableError,
  gpu_out_of_memory: ServiceUnavailableError,
  invalid_request: ValidationError,
  missing_video: ValidationError,
  invalid_video_url: ValidationError,
  unsupported_media_type: ValidationError,
  video_too_large: ValidationError,
  video_too_long: ValidationError,
  video_too_short: ValidationError,
  no_video_stream: ValidationError,
  corrupt_media: ValidationError,
  download_forbidden_host: ValidationError,
  prompt_too_long: ValidationError,
  prompt_blocked: ValidationError,
  seed_out_of_range: ValidationError,
  parameter_out_of_range: ValidationError,
  prediction_not_found: ClientError,
  prediction_not_cancelable: ClientError,
  inference_failed: ServerError,
  inference_timeout: ServerError,
  mux_failed: ServerError,
  storage_failed: ServerError,
  download_failed: ServerError,
  internal_error: ServerError,
};

/** Build the most specific error instance for an error payload. */
export function errorFromPayload(
  errorCode: string,
  httpStatus: number,
  message: string,
  requestId = "",
  details: Record<string, unknown> = {},
  retryAfter: number | null = null,
): VSFXError {
  const cls = CODE_TO_CLASS[errorCode as ErrorCode];
  if (cls !== undefined) return new cls(errorCode as ErrorCode, httpStatus, message, requestId, details, retryAfter);
  if (httpStatus >= 500) return new ServerError("internal_error", httpStatus, message, requestId, details, retryAfter);
  return new ClientError("invalid_request", httpStatus, message, requestId, details, retryAfter);
}
