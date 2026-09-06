# Architecture

## Process topology

* **API** (`app/main.py` → `app/asgi.py`): FastAPI + Uvicorn. Owns
  authentication (argon2 bearer keys), rate limiting (Redis token bucket,
  fail-open), idempotency (sha256 of normalized input + key), quota checks,
  uploads/presigning, prediction CRUD, webhooks registry, static web UI.
* **Worker** (`app/worker/main.py`): Redis Streams consumer group. One
  process per GPU; an in-process semaphore + Redis token (`GpuLock`) guard
  VRAM. Stages: download → probe → validate → moderate → generate → decode →
  post → mix → mux → upload → webhook/usage. Heartbeats (XAUTOCLAIM idle
  reclaim), retry with backoff (`attempt−1`, base/max, priority +1), DLQ
  `{stream}:dlq`, stale-job reaper, hourly retention sweep.
* **Postgres** (asyncpg + SQLAlchemy 2, Alembic migrations 0001–0003):
  `api_keys`, `assets`, `predictions`, `usage_events`, `webhook_deliveries`.
* **Redis**: job stream `vsfx:jobs` (+ DLQ), rate buckets, GPU cluster lock.
* **S3/MinIO**: `{prefix}/uploads/yyyy/mm/dd/{sha256}{ext}` sources and
  `{prefix}/outputs/{id}/{output.mp4|audio.wav}` results.
* **Web UI** (`web/`, React 18 + Vite): dropzone/URL input, full parameter
  form, live polling list, inline MP4/WAV playback.

## Request lifecycle

1. `POST /predictions/video-to-video-sfx` (JSON or multipart).
2. Middleware: request id → access log → body limit → auth (argon2 verify +
  scopes) → rate limit.
3. Normalize + validate (`app/api/schemas`, `app/utils/validators.py`);
   moderation (`text_blocklist` + pluggable backends) unless disabled.
4. Idempotency: same `Idempotency-Key` + same payload sha256 ⇒ return the
   existing row; different payload ⇒ `409`.
5. Quota/concurrency gates ⇒ insert `queued` prediction, `XADD` job.
6. Worker claims, executes the stage chain (metrics per stage), bills usage
   on success **and** failure (billed seconds = plan duration), uploads the
   muxed faststart MP4, signs + delivers the webhook.
7. Client polls `GET /predictions/{id}` (or awaits the webhook).

## Failure semantics

* Worker crash mid-job → stream entry idles → another worker `XAUTOCLAIM`s.
* Poison job → retries with backoff until `max_attempts` → DLQ + `failed`.
* Weights missing/mismatched → `weights_unavailable` (fail fast, no mock
  audio) at pipeline build, surfaced per-request as 503.
* Redis rate-limit errors fail **open** (availability over strictness);
  queue/DLQ writes fail closed with `internal_error`.
* Cancellation is cooperative: checked between stages and inside long loops;
  cancels map to `canceled` (not billed beyond work done).

## Envelope & ids

ULID-style 26-char Crockford ids (`app/utils/ids.py`, monotonic within ms);
every response `{"code","message","data"}`; errors carry the taxonomy codes
listed in `docs/api.md`. Request ids flow through structlog + OTel-compatible
trace ids into `X-Request-Id`.

## Inference pipeline (per window)

CLIP-visual tokens (8 fps) + sync tokens (25 fps) + CLIP text tokens (77)
+ optional memory latents → MMDiT velocity field → integrate (euler/midpoint,
  CFG with φ-rescale) → VAE decode → vocoder → DC removal → LUFS normalize →
  true-peak limit → equal-power crossfade across windows → FFmpeg mix/duck →
  `-c:v copy -c:a aac -movflags +faststart` mux. Audio/video drift is bounded
  by construction to ±10 ms (`±1 output sample` at container start).
