# Troubleshooting

## Quick triage by error code

| `error_code` | You see | Do this |
|---|---|---|
| `weights_unavailable` | 503 immediately after submit | `python scripts/verify_weights.py`; train/export (`docs/training.md`) or fix `WEIGHTS_DIR` mount/manifest |
| `unauthorized` | 401 | key header is `Authorization: Bearer vsfx_…`; check copy-paste whitespace |
| `forbidden_scope` | 403 | key lacks scope — recreate with `scripts/create_api_key.py --scopes` |
| `quota_exceeded` | 402 | monthly seconds used up; raise `monthly_seconds_quota` |
| `rate_limited` | 429 + `Retry-After` | honor the header; batch or raise `rate_limit_rpm` |
| `concurrency_limited` | 429 | retry after ~30 s or raise `concurrent_jobs_per_key` |
| `idempotency_key_conflict` | 409 | same `Idempotency-Key` with a different payload — use a new key |
| `unsupported_media_type` | 415 | container sniffed as non-video; re-encode (`ffmpeg -i in.avi -c:v libx264 -c:a aac out.mp4`) |
| `no_video_stream` | 422 | audio-only or broken file; probe locally with `ffprobe` |
| `video_too_long` | 422 | trim to ≤ `MAX_VIDEO_DURATION_S` (300 s) or request `duration`/`start_time` |
| `prompt_blocked` | 422 | prompt matched the safety blocklist (`config/safety_blocklist.txt`) |
| `download_failed` / `download_forbidden_host` | 502/400 | URL must be public https; private CIDRs refused (SSRF guard) |
| `gpu_out_of_memory` | 503 | transient; auto-retried; if persistent reduce concurrency or variant |
| `inference_timeout` | 504 | raise `VSFX_JOBS__…` timeout or shorten `duration` |
| `mux_failed` | 500 | worker ffmpeg logs; check ffmpeg present + disk in `work_dir` |

## Local setup issues

* **`make dev` exits on config errors** — startup validation lists the exact
  offending `VSFX_*` variable; copy `.env.example` and fill it.
* **ffmpeg not found** — install ffmpeg/ffprobe (apt/brew); both are checked
  at startup.
* **`make web` shows the fallback JSON page** — the API serves `web/dist`
  when it exists; run `cd web && npm install && npm run build` (or
  `make docker-build`, web profile).
* **Migrations fail to connect** — `make docker-up` first; verify
  `VSFX_DATABASE__DATABASE_URL` uses `postgresql+asyncpg://`.
* **MinIO 403 on outputs** — bucket policy; rerun `minio-init` (compose) or
  the `vsfx-minio-init` job (k8s).

## Worker/queue issues

* **Stuck `processing`** — worker died mid-stage; the reaper fails it after
  `job_stale_after_s`; confirm heartbeats in logs.
* **Jobs skipped** — check DLQ: `redis-cli XRANGE vsfx:jobs:dlq - + COUNT 5`
  shows the payload + last error.
* **Duplicate processing** — expected during `XAUTOCLAIM` handoff; stages are
  idempotent (outputs overwrite by prediction id).

## SDK/CLI issues

* **Python `ImportError: vsfx_client`** — `pip install -e sdk/python` (or run
  from the repo root; the CLI bootstraps the path itself).
* **TS `fetch is not defined`** — Node ≥ 18 required.
* **`vsfx` not found** — `pip install -e .` installs the console script.
* **Download 403 from MinIO** — presigned GET expiry; re-fetch the prediction
  for a fresh URL.

## Still stuck?

Reproduce minimally with `scripts/make_test_media.py`, submit via
`vsfx predict create --file test.mp4 --prompt ping --wait --json`, and open an
issue containing: the failing `error_code`, `request_id`, worker log lines
with that `prediction_id`, `weights/manifest.json` revision, and versions
(`/api/v1/version`, ffmpeg, GPU driver).
