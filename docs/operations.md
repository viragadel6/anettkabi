# Operations Runbook

## Probes & dashboards

* `GET /api/v1/healthz` — liveness (process up).
* `GET /api/v1/readyz` — readiness: DB ping, Redis ping, pipeline build OK.
* `GET /api/v1/metrics` — Prometheus text format (API); worker metrics on
  `:9090/metrics` (k8s annotation scrapes both).

Key series (all prefixed per exporter config):

| Metric | Labels | Alert baseline |
|---|---|---|
| `vsfx_predictions_total` | `status` | failed ratio > 5% over 15 m |
| `vsfx_inference_seconds` | — | p95 > SLA (default 90 s window) |
| `vsfx_queue_depth` | `stream` | > 2× workers for 10 m (scale workers) |
| `vsfx_dlq_size` | `stream` | > 0 growing (inspect payloads) |
| `vsfx_gpu_memory_bytes` | — | > 90% of budget sustained |
| `vsfx_rate_limited_total` / `vsfx_quota_blocked_total` | — | spikes = abuse or bad quotas |
| `vsfx_webhook_deliveries_total` | `outcome` | `failure` ratio > 1% |
| `vsfx_weights_revision` | `revision` | changes → expect rollout |

## Logs

structlog JSON, one event object per line: `event`, `request_id`,
`prediction_id`, `api_key_id` (first 8 chars), stage timings, `error_code`.
PII policy: prompts are logged only as `prompt_hash` + length; blocklist
matches log the category, never the prompt itself. Ship with any OTel-aware
agent; `VSFX_TELEMETRY__LOG_FORMAT=console` for local dev.

## Capacity & scaling

* One worker per GPU; VRAM budget enforced by `GpuLock` (semaphore + Redis
  token, fail-open). A10G ≈ 2 concurrent `small_16k` windows.
* API is CPU-bound (auth + uploads): HPA on 70% CPU.
* Postgres: `predictions` grows ~1 KB/row + JSONB input; expression index on
  `created_at DESC` keeps listing flat. Partition by month when > 10⁷ rows.
* S3 lifecycle: expire `outputs/` after `RESULT_RETENTION_DAYS` (default 30);
  uploads TTL 24 h; the hourly retention sweep deletes expired rows/objects.

## Backups

* Postgres: nightly `pg_dump` (or PITR); the only durable state together
  with S3. Redis is a queue — reconstructable; DLQ worth snapshotting.
* Weights: `weights/manifest.json` + artifacts are immutable per revision;
  archive each exported revision (they are the model).

## Common tasks

```console
python scripts/create_api_key.py --name ci --scopes predictions:write
python scripts/verify_weights.py                 ; echo $?
python scripts/benchmark.py --duration 60 --concurrency 4
kubectl -n vsfx logs deploy/vsfx-worker -f | jq 'select(.event=="stage_finished")'
redis-cli XLEN vsfx:jobs && redis-cli XLEN vsfx:jobs:dlq
```

## Incident playbook (short)

| Symptom | First checks | Likely fix |
|---|---|---|
| 503 `weights_unavailable` | `verify_weights.py`, PVC mount | re-export weights / fix mount |
| Queue grows, no drains | worker logs `gpu_out_of_memory` | scale workers / reduce batch |
| 429 storm | `rate_limited_total` by key | quotas vs. abusive key rotation |
| Webhooks failing | `webhook_deliveries.outcome` | peer TLS/clock; rotate secret |
| A/V drift reports | `mux_seconds`, ffmpeg version | pin ffmpeg; check `-faststart` |
