/**
 * HTTP core: fetch wrapper, envelope unwrapping, error mapping.
 * @module
 */

import { errorFromPayload, TransportError, VSFXError } from "./errors.js";
import type { Envelope, Prediction } from "./models.js";

/** API prefix used by the service. */
export const API_PREFIX = "/api/v1";

/** Default fetch options provider (injectable for tests). */
export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

/** Shared request options. */
export interface RequestOptions {
  jsonBody?: unknown | undefined;
  params?: Record<string, string | number | boolean | undefined> | undefined;
  headers?: Record<string, string> | undefined;
  signal?: AbortSignal | undefined;
}

function buildQuery(params: Record<string, string | number | boolean | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) search.set(key, String(value));
  }
  const encoded = search.toString();
  return encoded ? `?${encoded}` : "";
}

function unwrap<T>(payload: unknown): T {
  if (payload !== null && typeof payload === "object" && "data" in payload) {
    const envelope = payload as Envelope<T>;
    return envelope.data;
  }
  return payload as T;
}

async function raiseForResponse(response: Response): Promise<never> {
  let errorCode = "internal_error";
  let message = response.statusText || "request failed";
  let requestId = response.headers.get("x-request-id") ?? "";
  let details: Record<string, unknown> = {};
  let retryAfter: number | null = null;
  try {
    const body: unknown = await response.json();
    if (body !== null && typeof body === "object") {
      const record = body as Record<string, unknown>;
      if (typeof record.error_code === "string") errorCode = record.error_code;
      if (typeof record.message === "string") message = record.message;
      if (typeof record.request_id === "string") requestId = record.request_id;
      if (record.details !== null && typeof record.details === "object") {
        details = record.details as Record<string, unknown>;
      }
    }
  } catch {
    const text = await response.text().catch(() => "");
    if (text) message = text.slice(0, 300);
  }
  const headerRetry = response.headers.get("retry-after");
  if (headerRetry !== null) {
    const parsed = Number.parseFloat(headerRetry);
    retryAfter = Number.isNaN(parsed) ? null : parsed;
  }
  throw errorFromPayload(errorCode, response.status, message, requestId, details, retryAfter);
}

/** Transport implementing request/download against the service. */
export class Transport {
  private readonly baseUrl: string;
  private readonly baseHeaders: Record<string, string>;
  private readonly fetchFn: FetchLike;

  constructor(baseUrl: string, baseHeaders: Record<string, string>, fetchFn?: FetchLike) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.baseHeaders = baseHeaders;
    this.fetchFn = fetchFn ?? ((input, init) => fetch(input, init));
  }

  private url(
    path: string,
    params?: Record<string, string | number | boolean | undefined> | undefined,
  ): string {
    const target = path.startsWith("http://") || path.startsWith("https://")
      ? path
      : `${this.baseUrl}${API_PREFIX}/${path.replace(/^\/+/, "")}`;
    return params ? `${target}${buildQuery(params)}` : target;
  }

  async request<T>(method: string, path: string, options: RequestOptions = {}): Promise<T> {
    const headers: Record<string, string> = { ...this.baseHeaders, ...options.headers };
    let body: string | undefined;
    if (options.jsonBody !== undefined) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(options.jsonBody);
    }
    let response: Response;
    try {
      const init: RequestInit = { method, headers };
      if (body !== undefined) init.body = body;
      if (options.signal !== undefined) init.signal = options.signal;
      response = await this.fetchFn(this.url(path, options.params), init);
    } catch (cause) {
      throw new TransportError(`${method} ${path} failed: ${String(cause)}`);
    }
    if (!response.ok) await raiseForResponse(response);
    const payload: unknown = await response.json();
    return unwrap<T>(payload);
  }

  async postMultipart<T>(
    path: string,
    fields: Record<string, string>,
    fileField: string,
    file: { name: string; type: string; data: Uint8Array | Blob },
    headers: Record<string, string> = {},
    signal?: AbortSignal,
  ): Promise<T> {
    const form = new FormData();
    for (const [key, value] of Object.entries(fields)) form.set(key, value);
    const blob = file.data instanceof Blob ? file.data : new Blob([file.data], { type: file.type });
    form.set(fileField, blob, file.name);
    let response: Response;
    try {
      const init: RequestInit = { method: "POST", headers: { ...this.baseHeaders, ...headers }, body: form };
      if (signal !== undefined) init.signal = signal;
      response = await this.fetchFn(this.url(path), init);
    } catch (cause) {
      throw new TransportError(`POST ${path} failed: ${String(cause)}`);
    }
    if (!response.ok) await raiseForResponse(response);
    const payload: unknown = await response.json();
    return unwrap<T>(payload);
  }

  async put(url: string, data: Uint8Array, contentType: string, signal?: AbortSignal): Promise<void> {
    let response: Response;
    try {
      const init: RequestInit = { method: "PUT", headers: { "Content-Type": contentType }, body: data as BodyInit };
      if (signal !== undefined) init.signal = signal;
      response = await this.fetchFn(url, init);
    } catch (cause) {
      throw new TransportError(`PUT upload failed: ${String(cause)}`);
    }
    if (!response.ok) await raiseForResponse(response);
  }

  async download(url: string, destination: string): Promise<number> {
    const { writeFile } = await import("node:fs/promises");
    let response: Response;
    try {
      response = await this.fetchFn(url.startsWith("http") ? url : this.url(url), { method: "GET" });
    } catch (cause) {
      throw new TransportError(`download failed: ${String(cause)}`);
    }
    if (!response.ok) await raiseForResponse(response);
    const buffer = new Uint8Array(await response.arrayBuffer());
    await writeFile(destination, buffer);
    return buffer.byteLength;
  }

  async rawGet(
    path: string,
    params?: Record<string, string | number | boolean | undefined> | undefined,
  ): Promise<Response> {
    try {
      return await this.fetchFn(this.url(path, params), { method: "GET", headers: this.baseHeaders });
    } catch (cause) {
      throw new TransportError(`GET ${path} failed: ${String(cause)}`);
    }
  }
}

/** Full-jitter exponential backoff helper. */
export function backoffDelay(attempt: number, base = 1.0, cap = 10.0): number {
  const ceiling = Math.min(cap, base * 2 ** attempt);
  return Math.random() * ceiling;
}

/** Sleep helper. */
export function sleep(seconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, seconds * 1000));
}

/** Narrow an unknown payload into a Prediction. */
export function parsePrediction(payload: unknown): Prediction {
  if (payload !== null && typeof payload === "object") return payload as Prediction;
  throw new VSFXError("invalid_request", 0, "prediction payload was not an object");
}
