# Deployment

## Docker Compose (single host)

```console
cp .env.example .env
make docker-build
make docker-up            # postgres redis minio minio-init migrate api worker
docker compose -f docker/docker-compose.yml --profile full up -d web
```

* API on `:8000`, MinIO `:9000` (console `:9001`), Postgres `:5432`,
  Redis `:6379`, web (nginx) on `:8080` proxies `/api` → `api:8000`.
* `make migrate` (or the one-shot `migrate` service) applies Alembic head.
* Mount trained weights at `./weights:/srv/weights:ro`
  (`WEIGHTS_DIR` env for compose). Without them the API is healthy but
  predictions return `weights_unavailable` until you export weights.
* `WORKER_REPLICAS` scales workers (one GPU each).

## Kubernetes (`deploy/k8s/`)

```console
kubectl apply -f deploy/k8s/00-namespace.yaml
cp deploy/k8s/20-secrets.example.yaml > /tmp/secrets.yaml   # edit values
kubectl apply -f /tmp/secrets.yaml -f deploy/k8s/10-config.yaml
kubectl apply -f deploy/k8s/30-postgres.yaml -f deploy/k8s/31-redis.yaml -f deploy/k8s/32-minio.yaml
kubectl -n vsfx rollout status statefulset/vsfx-minio
kubectl delete job vsfx-minio-init --ignore-not-found && kubectl apply -f deploy/k8s/32-minio.yaml
kubectl apply -f deploy/k8s/40-migrate-job.yaml
kubectl -n vsfx wait --for=condition=complete job/vsfx-migrate --timeout=300s
kubectl apply -f deploy/k8s/41-api.yaml -f deploy/k8s/42-worker.yaml \
              -f deploy/k8s/43-ingress.yaml -f deploy/k8s/44-hpa.yaml
```

* Replace `ghcr.io/example/vsfx-*` with your registry refs (CI builds them
  via `docker/Dockerfile.{api,worker,web}`).
* Workers request `nvidia.com/gpu: 1` — install the NVIDIA device plugin and
  node selector/taints to taste.
* `vsfx-weights` is a RWX PVC holding the verified weights + manifest;
  workers and API mount it read-only.
* HPA scales the API on CPU/memory 2→12; workers scale manually or via
  KEDA on stream depth (XLEN `vsfx:jobs`).

## Environment

Every knob is `VSFX_SECTION__FIELD` (see `.env.example` for the annotated
table and `app/config.py` for validation). Critical ones:

| Variable | Default | Notes |
|---|---|---|
| `VSFX_DATABASE__DATABASE_URL` | postgres `+asyncpg` | required |
| `VSFX_REDIS__REDIS_URL` | `redis://localhost:6379/0` | stream + buckets |
| `VSFX_STORAGE__S3_*` | MinIO local | endpoint/bucket/keys/path-style |
| `VSFX_MODEL__MODEL_VARIANT` | `small_16k` | must match trained weights |
| `VSFX_SERVER__PUBLIC_BASE_URL` | `http://localhost:8000` | used for output URLs |
| `VSFX_SAFETY__SAFETY_BLOCKLIST_PATH` | `config/safety_blocklist.txt` | shipped blocklist |
| `VSFX_WEBHOOK__WEBHOOK_SIGNING_SECRET` | `""` | HMAC key for deliveries |

Startup validation is fail-fast: missing ffmpeg/ffprobe, unreachable schema
config, or variant/sample-rate mismatches abort with actionable messages.

## Zero-downtime rollouts

1. Build/push images; bump `image:` in 41-api/42-worker.
2. `kubectl apply` — API rolls with `maxUnavailable: 0`; workers drain by
   finishing in-flight jobs before exiting (SIGTERM → finish stage → exit).
3. Migrations are additive-only per policy (0001→0003 pattern); run
   `40-migrate-job.yaml` before the rollout.
