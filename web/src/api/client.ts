import type { Envelope, Prediction, SfxRequestOptions, UploadAck, UsageSnapshot } from "./types";

const API_PREFIX = "/api/v1";

export class ApiError extends Error {
  readonly errorCode: string;
  readonly httpStatus: number;
  readonly retryAfter: number | null;

  constructor(errorCode: string, httpStatus: number, message: string, retryAfter: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.errorCode = errorCode;
    this.httpStatus = httpStatus;
    this.retryAfter = retryAfter;
  }
}

export interface ApiKeyStore {
  get(): string;
  set(key: string): void;
  clear(): void;
}

const STORAGE_KEY = "vsfx.api_key";

export const localStorageKeyStore: ApiKeyStore = {
  get: () => window.localStorage.getItem(STORAGE_KEY) ?? "",
  set: (key: string) => window.localStorage.setItem(STORAGE_KEY, key),
  clear: () => window.localStorage.removeItem(STORAGE_KEY),
};

export class SfxApiClient {
  private apiKey: string;

  constructor(key: string) {
    this.apiKey = key;
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers);
    headers.set("Accept", "application/json");
    if (this.apiKey) headers.set("Authorization", `Bearer ${this.apiKey}`);
    if (init.body !== undefined && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }
    let response: Response;
    try {
      response = await fetch(`${API_PREFIX}/${path.replace(/^\/+/, "")}`, { ...init, headers });
    } catch (cause) {
      throw new ApiError("transport_error", 0, `network failure: ${String(cause)}`);
    }
    if (!response.ok) await this.raise(response);
    const payload = (await response.json()) as Envelope<T>;
    return payload.data;
  }

  private async raise(response: Response): Promise<never> {
    let errorCode = "internal_error";
    let message = response.statusText || "request failed";
    try {
      const body = (await response.json()) as Record<string, unknown>;
      if (typeof body.error_code === "string") errorCode = body.error_code;
      if (typeof body.message === "string") message = body.message;
    } catch {
      void 0;
    }
    const retryHeader = response.headers.get("retry-after");
    const retryAfter = retryHeader !== null ? Number.parseFloat(retryHeader) : null;
    throw new ApiError(errorCode, response.status, message, retryAfter !== null && !Number.isNaN(retryAfter) ? retryAfter : null);
  }

  createFromUrl(videoUrl: string, options: SfxRequestOptions, idempotencyKey?: string): Promise<Prediction> {
    const headers: Record<string, string> = {};
    if (idempotencyKey) headers["Idempotency-Key"] = idempotencyKey;
    return this.request<Prediction>("predictions/video-to-video-sfx", {
      method: "POST",
      headers,
      body: JSON.stringify({ video: videoUrl, ...options }),
    });
  }

  createMultipart(file: File, options: SfxRequestOptions, idempotencyKey?: string): Promise<Prediction> {
    const form = new FormData();
    form.set("video", file, file.name);
    for (const [key, value] of Object.entries(options)) {
      if (value === undefined) continue;
      form.set(key, typeof value === "object" ? JSON.stringify(value) : String(value));
    }
    const headers: Record<string, string> = {};
    if (idempotencyKey) headers["Idempotency-Key"] = idempotencyKey;
    return this.request<Prediction>("predictions/video-to-video-sfx", { method: "POST", headers, body: form });
  }

  get(predictionId: string): Promise<Prediction> {
    return this.request<Prediction>(`predictions/${predictionId}`);
  }

  list(params: { status?: string; limit?: number; after?: string } = {}): Promise<Prediction[]> {
    const search = new URLSearchParams();
    if (params.status) search.set("status", params.status);
    if (params.limit) search.set("limit", String(params.limit));
    if (params.after) search.set("after", params.after);
    const query = search.toString();
    return this.request<Prediction[]>(`predictions${query ? `?${query}` : ""}`);
  }

  cancel(predictionId: string): Promise<Prediction> {
    return this.request<Prediction>(`predictions/${predictionId}/cancel`, { method: "POST" });
  }

  remove(predictionId: string): Promise<void> {
    return this.request<void>(`predictions/${predictionId}`, { method: "DELETE" });
  }

  upload(file: File): Promise<UploadAck> {
    const form = new FormData();
    form.set("file", file, file.name);
    return this.request<UploadAck>("uploads", { method: "POST", body: form });
  }

  usage(): Promise<UsageSnapshot> {
    return this.request<UsageSnapshot>("account/usage");
  }

  health(): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>("healthz");
  }
}

export function resolveOutputUrl(prediction: Prediction): string | null {
  const output = prediction.output ?? {};
  const candidates = [output.video, output.audio, output.mp4, output.url];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate) {
      return candidate.startsWith("http") ? candidate : candidate;
    }
  }
  return null;
}
