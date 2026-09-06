# Video-to-Video SFX — HTTP API Reference

Base URL: `https://<host>/api/v1` · All responses use the standard envelope
`{"code", "message", "data"}`. Errors additionally carry `error_code`,
`request_id`, and where relevant `details` / `retry_after`.

## Authentication

Every endpoint except the probes requires a bearer API key:

```http
Authorization: Bearer vsfx_XXXXXXXX_yyyy...yyyy
```

Keys are created out-of-band (`python scripts/create_api_key.py --name prod`).
Keys carry scopes (`predictions:write`, `predictions:read`, `admin`, …),
a per-key requests/minute budget, a concurrency limit, and a monthly
seconds-generated quota.

## Content types

Two mutually interchangeable request shapes for prediction creation:

* `application/json` — `{"video": "https://...", "prompt": "...", ...}` for a
  previously uploaded or public URL (upload first via `POST /uploads`).
* `multipart/form-data` — a `video` file part plus one form field per
  parameter (same names as the JSON keys).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/predictions/video-to-video-sfx` | Create a prediction (200, async) |
| GET | `/predictions/{id}` | Fetch status/payload |
| GET | `/predictions/{id}/result` | Result envelope (identical body) |
| GET | `/predictions?status=&limit=&after=` | Page the caller's predictions |
| POST | `/predictions/{id}/cancel` | Best-effort cancellation |
| DELETE | `/predictions/{id}` | Soft-delete (output purged by retention) |
| POST | `/uploads` | Direct multipart upload → retrievable URL |
| POST | `/uploads/presign` | Presigned PUT URL for large files |
| GET | `/models/video-to-video-sfx` | Model card + JSON Schemas |
| GET | `/account/usage` | Caller's usage snapshot |
| POST | `/webhooks/test` | Signed webhook probe |
| GET | `/healthz` `/readyz` `/version` `/metrics` | Probes (no auth) |

## Prediction request fields

| Field | Type | Default | Bounds / values |
|---|---|---|---|
| `video` | string | — | https URL of an uploaded/public video, or a `data:` URI |
| `prompt` | string | required | ≤ 2000 chars, NFC-normalized |
| `negative_prompt` | string | `""` | ≤ 2000 chars |
| `seed` | int | `-1` | `-1` (random) or `[0, 2³¹−1]` |
| `num_inference_steps` | int | 25 | 1..100 |
| `guidance_scale` | float | 4.5 | 0..15 (0 disables CFG → single pass) |
| `duration` | float | full | > 0 s, capped by `MAX_VIDEO_DURATION_S` |
| `start_time` | float | 0 | ≥ 0 |
| `audio_mode` | enum | `replace` | `replace` / `mix` / `duck` |
| `sfx_gain_db` | float | 0 | −30..12 |
| `original_audio_gain_db` | float | −6 | −60..12 |
| `duck_threshold_db` | float | −24 | −60..0 |
| `duck_ratio` | float | 4 | 1..20 |
| `duck_attack_ms` | float | 15 | 1..500 |
| `duck_release_ms` | float | 250 | 10..2000 |
| `target_loudness_lufs` | float | −14 | −40..0 or `null` |
| `true_peak_db` | float | −1 | −12..0 |
| `video_handling` | enum | `copy` | `copy` / `reencode` |
| `return_audio_only` | bool | false | WAV-only output |
| `enable_safety_checker` | bool | true | prompt moderation toggle |
| `webhook_url` | string | — | https, SSRF-guarded |
| `metadata` | object | `{}` | ≤ 4 KiB serialized |

Headers: `Idempotency-Key` (recommended; up to 255 chars). Replaying the same
key with an identical payload returns the original prediction; a different
payload yields `409 idempotency_key_conflict`.

## Statuses

`starting → queued → processing → succeeded | failed | canceled`
(webhook fires on every terminal transition). Terminal payloads include
`output` URLs (faststart MP4 and/or WAV), `metrics`
(`total_s`, `generate_s`, `mux_s`, …), and on failure an `error_code` from
the taxonomy below.

## Error taxonomy

| HTTP | `error_code` | Meaning |
|---|---|---|
| 400 | `invalid_request` | malformed envelope/fields |
| 400 | `missing_video` | no usable `video` input |
| 400 | `invalid_video_url` | unparseable/unsafe URL |
| 400 | `download_forbidden_host` | private/forbidden host (SSRF guard) |
| 400 | `prompt_too_long` | prompt exceeds 2000 chars |
| 400 | `seed_out_of_range` | seed outside `[-1, 2³¹−1]` |
| 400 | `parameter_out_of_range` | any other bound violation |
| 401 | `unauthorized` | missing/invalid key |
| 402 | `quota_exceeded` | monthly seconds quota exhausted |
| 403 | `forbidden_scope` | key lacks the required scope |
| 404 | `prediction_not_found` | unknown id (or not owner) |
| 409 | `idempotency_key_conflict` | key reused with different payload |
| 409 | `prediction_not_cancelable` | already terminal |
| 413 | `video_too_large` | upload exceeds size cap |
| 415 | `unsupported_media_type` | container/codec sniffing failed |
| 422 | `video_too_long` / `video_too_short` / `no_video_stream` / `corrupt_media` / `prompt_blocked` | media validation & moderation |
| 429 | `rate_limited` | per-key RPM budget (includes `Retry-After`) |
| 429 | `concurrency_limited` | too many in-flight jobs (retry in ~30 s) |
| 500 | `inference_failed` / `mux_failed` / `internal_error` | worker-side failures |
| 502 | `download_failed` / `storage_failed` | upstream fetch/store failures |
| 503 | `weights_unavailable` | verified weights missing — train/export first |
| 503 | `gpu_out_of_memory` | GPU saturation (auto-retry w/ smaller batch) |
| 504 | `inference_timeout` | exceeded `INFERENCE_TIMEOUT_S` |

## Webhooks

Deliveries are signed; verify before trusting:

```
X-VSFX-Signature: sha256=<hex hmac>
X-VSFX-Timestamp: <unix seconds>
```

`signature = HMAC-SHA256(webhook_signing_secret, timestamp + "." + raw_body)`.
Reject deliveries older than 5 minutes, then retry-verify with a fresh clock
read. Failed deliveries retry with exponential backoff (≤ 5 attempts);
every attempt is persisted in `webhook_deliveries`.

## SDK & CLI examples

Python:

```python
from vsfx_client import Client, SfxParams

client = Client("vsfx_...", base_url="https://sfx.example.com")
p = client.create("https://cdn/clip.mp4",
                  SfxParams(prompt="rain on tin roof", audio_mode="duck"))
p = client.wait(p.id)
client.download(p, "out.mp4")
```

TypeScript: `VSFXClient` (see `sdk/typescript/README.md`).

CLI:

```console
$ vsfx config set --api-key vsfx_... --base-url https://sfx.example.com
$ vsfx predict create --file clip.mp4 --prompt "deep impact whoosh" --wait --out hit.mp4
```
