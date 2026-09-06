/**
 * The vsfx TypeScript client.
 * @module
 */

import { readFile } from "node:fs/promises";
import type { AccountUsage, Envelope, ModelInfo, Prediction, SfxParams, UploadAck, UploadPresign, WebhookTestResult } from "./models.js";
import { isActive, isTerminal, paramsToWire } from "./models.js";
import { Transport } from "./transport.js";

/** Client construction options. */
export interface ClientOptions {
  baseUrl?: string;
  timeoutMs?: number;
  fetchFn?: typeof fetch;
  userAgent?: string;
}

/** Video-to-Video SFX API client (all methods async). */
export class VSFXClient {
  private readonly transport: Transport;
  readonly baseUrl: string;

  constructor(apiKey: string, options: ClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "http://localhost:8000").replace(/\/+$/, "");
    this.transport = new Transport(
      this.baseUrl,
      {
        Authorization: `Bearer ${apiKey}`,
        Accept: "application/json",
        "User-Agent": options.userAgent ?? "vsfx-ts/1.0",
      },
      options.fetchFn,
    );
  }

  /** POST /predictions/video-to-video-sfx from a URL or local file path. */
  async create(
    video: string,
    params: SfxParams,
    options: { idempotencyKey?: string | undefined; signal?: AbortSignal | undefined } = {},
  ): Promise<Prediction> {
    const videoValue = /^https?:\/\//.test(video) ? video : await this.uploadLocal(video).then((ack) => ack.url);
    const headers: Record<string, string> = {};
    if (options.idempotencyKey) headers["Idempotency-Key"] = options.idempotencyKey;
    return this.transport.request<Prediction>("POST", "predictions/video-to-video-sfx", {
      jsonBody: { video: videoValue, ...paramsToWire(params) },
      headers,
      signal: options.signal,
    });
  }

  /** POST multipart (direct upload + create in one call). */
  async createMultipart(
    videoPath: string,
    params: SfxParams,
    options: { idempotencyKey?: string | undefined; signal?: AbortSignal | undefined } = {},
  ): Promise<Prediction> {
    const data = new Uint8Array(await readFile(videoPath));
    const name = videoPath.split("/").pop() ?? "upload.mp4";
    const type = name.endsWith(".mov") ? "video/quicktime" : name.endsWith(".webm") ? "video/webm" : name.endsWith(".mkv") ? "video/x-matroska" : "video/mp4";
    const fields: Record<string, string> = {};
    const wire = paramsToWire(params);
    for (const [key, value] of Object.entries(wire)) {
      if (key === "metadata") fields[key] = JSON.stringify(value);
      else if (typeof value === "object") fields[key] = JSON.stringify(value);
      else fields[key] = String(value);
    }
    const headers: Record<string, string> = {};
    if (options.idempotencyKey) headers["Idempotency-Key"] = options.idempotencyKey;
    return this.transport.postMultipart<Prediction>(
      "predictions/video-to-video-sfx",
      fields,
      "video",
      { name, type, data },
      headers,
      options.signal,
    );
  }

  /** GET /predictions/{id}. */
  async get(predictionId: string, signal?: AbortSignal): Promise<Prediction> {
    return this.transport.request<Prediction>("GET", `predictions/${predictionId}`, { signal });
  }

  /** GET /predictions/{id}/result. */
  async result(predictionId: string, signal?: AbortSignal): Promise<Prediction> {
    return this.transport.request<Prediction>("GET", `predictions/${predictionId}/result`, { signal });
  }

  /** GET /predictions with filters. */
  async list(
    options: {
      status?: string | undefined;
      limit?: number | undefined;
      after?: string | undefined;
      signal?: AbortSignal | undefined;
    } = {},
  ): Promise<Prediction[]> {
    const data = await this.transport.request<unknown>("GET", "predictions", {
      params: { status: options.status, limit: options.limit, after: options.after },
      signal: options.signal,
    });
    if (Array.isArray(data)) return data as Prediction[];
    if (data !== null && typeof data === "object" && Array.isArray((data as Record<string, unknown>).results)) {
      return (data as { results: Prediction[] }).results;
    }
    return [];
  }

  /** POST /predictions/{id}/cancel. */
  async cancel(predictionId: string, signal?: AbortSignal): Promise<Prediction> {
    return this.transport.request<Prediction>("POST", `predictions/${predictionId}/cancel`, { signal });
  }

  /** DELETE /predictions/{id}. */
  async delete(predictionId: string, signal?: AbortSignal): Promise<void> {
    await this.transport.request<unknown>("DELETE", `predictions/${predictionId}`, { signal });
  }

  /** Poll until a terminal status. */
  async wait(
    predictionId: string,
    options: {
      pollIntervalMs?: number | undefined;
      timeoutMs?: number | undefined;
      onPoll?: ((prediction: Prediction) => void) | undefined;
      signal?: AbortSignal | undefined;
    } = {},
  ): Promise<Prediction> {
    const interval = options.pollIntervalMs ?? 2000;
    const deadline = options.timeoutMs !== undefined ? Date.now() + options.timeoutMs : Number.POSITIVE_INFINITY;
    let attempt = 0;
    for (;;) {
      const prediction = await this.get(predictionId, options.signal);
      options.onPoll?.(prediction);
      if (isTerminal(prediction)) return prediction;
      if (isActive(prediction) && Date.now() >= deadline) {
        throw new (await import("./errors.js")).VSFXError("inference_timeout", 504, `prediction ${predictionId} timed out`);
      }
      const jitter = Math.random() * Math.min(10_000, interval * 2 ** attempt);
      await new Promise((resolve) => setTimeout(resolve, Math.min(Math.max(interval, jitter), 30_000)));
      attempt += 1;
    }
  }

  /** Create then wait for completion. */
  async createAndWait(
    video: string,
    params: SfxParams,
    options: {
      idempotencyKey?: string | undefined;
      pollIntervalMs?: number | undefined;
      timeoutMs?: number | undefined;
      signal?: AbortSignal | undefined;
    } = {},
  ): Promise<Prediction> {
    const prediction = await this.create(video, params, { idempotencyKey: options.idempotencyKey, signal: options.signal });
    return this.wait(prediction.id, options);
  }

  /** Resolve the primary output artifact URL. */
  outputUrl(prediction: Prediction): string {
    const output = (prediction.output ?? {}) as Record<string, unknown>;
    const relative = output.video ?? output.audio ?? output.mp4 ?? output.url;
    if (typeof relative !== "string" || !relative) {
      throw new Error(`prediction ${prediction.id} has no output URL`);
    }
    return relative.startsWith("http://") || relative.startsWith("https://") ? relative : `${this.baseUrl}${relative}`;
  }

  /** Download the output artifact to a local file. */
  async download(prediction: Prediction, destination: string): Promise<number> {
    return this.transport.download(this.outputUrl(prediction), destination);
  }

  /** POST /uploads (multipart). */
  async upload(videoPath: string, signal?: AbortSignal): Promise<UploadAck> {
    const data = new Uint8Array(await readFile(videoPath));
    const name = videoPath.split("/").pop() ?? "upload.mp4";
    const type = name.endsWith(".mov") ? "video/quicktime" : name.endsWith(".webm") ? "video/webm" : name.endsWith(".mkv") ? "video/x-matroska" : "video/mp4";
    return this.transport.postMultipart<UploadAck>("uploads", {}, "file", { name, type, data }, {}, signal);
  }

  private async uploadLocal(videoPath: string): Promise<UploadAck> {
    return this.upload(videoPath);
  }

  /** POST /uploads/presign. */
  async presign(
    request: { filename: string; contentType?: string | undefined; sizeBytes?: number | undefined },
    signal?: AbortSignal | undefined,
  ): Promise<UploadPresign> {
    return this.transport.request<UploadPresign>("POST", "uploads/presign", {
      jsonBody: { filename: request.filename, content_type: request.contentType ?? "video/mp4", size_bytes: request.sizeBytes },
      signal,
    });
  }

  /** Presign then PUT the file directly to storage. */
  async presignedUpload(videoPath: string, signal?: AbortSignal): Promise<UploadAck> {
    const name = videoPath.split("/").pop() ?? "upload.mp4";
    const contentType = name.endsWith(".mov") ? "video/quicktime" : name.endsWith(".webm") ? "video/webm" : name.endsWith(".mkv") ? "video/x-matroska" : "video/mp4";
    const data = new Uint8Array(await readFile(videoPath));
    const presign = await this.presign({ filename: name, contentType, sizeBytes: data.byteLength }, signal);
    await this.transport.put(presign.upload_url, data, contentType, signal);
    const ack: UploadAck = { url: presign.url };
    if (presign.path !== undefined) ack.path = presign.path;
    if (presign.expires_at !== undefined) ack.expires_at = presign.expires_at;
    return ack;
  }

  /** GET /models/video-to-video-sfx. */
  async modelInfo(signal?: AbortSignal): Promise<ModelInfo> {
    return this.transport.request<ModelInfo>("GET", "models/video-to-video-sfx", { signal });
  }

  /** GET /account/usage. */
  async usage(signal?: AbortSignal): Promise<AccountUsage> {
    return this.transport.request<AccountUsage>("GET", "account/usage", { signal });
  }

  /** POST /webhooks/test. */
  async testWebhook(url: string, signal?: AbortSignal): Promise<WebhookTestResult> {
    return this.transport.request<WebhookTestResult>("POST", "webhooks/test", { jsonBody: { url }, signal });
  }

  /** GET /healthz. */
  async health(): Promise<Record<string, unknown>> {
    const response = await this.transport.rawGet("healthz");
    const payload = (await response.json()) as Envelope<Record<string, unknown>>;
    return payload.data;
  }

  /** GET /readyz. */
  async ready(): Promise<Record<string, unknown>> {
    const response = await this.transport.rawGet("readyz");
    const payload = (await response.json()) as Envelope<Record<string, unknown>>;
    return payload.data;
  }

  /** GET /version. */
  async version(): Promise<Record<string, unknown>> {
    const response = await this.transport.rawGet("version");
    const payload = (await response.json()) as Envelope<Record<string, unknown>>;
    return payload.data;
  }

  /** Await many predictions concurrently. */
  static async gather(
    client: VSFXClient,
    predictionIds: string[],
    options: { pollIntervalMs?: number | undefined; timeoutMs?: number | undefined } = {},
  ): Promise<Prediction[]> {
    return Promise.all(predictionIds.map((id) => client.wait(id, options)));
  }
}
