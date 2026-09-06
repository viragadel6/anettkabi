Video-to-Video SFX — Generative Sound-Design Service

Production-ready asynchronous web service that accepts an input video plus an optional
natural-language sound-design prompt, generates sound effects temporally aligned to the
video's visual content, mixes or replaces the audio track, and returns a finished
faststart MP4 as an HTTP URL inside a standard prediction response envelope.

The generative model (video-to-video-sfx) is implemented in this repository, from
scratch: a conditional latent flow-matching multimodal diffusion transformer (MMDiT)
over audio VAE latents, conditioned on CLIP visual features (8 fps), a dedicated
high-rate sync encoder (25 fps), and CLIP text features with classifier-free guidance.
Training runs on Modal GPUs via modal/sfx_training.py
(see docs/training.md); inference weights are verified by SHA-256 against a manifest.




Architecture


┌──────────────────────────────────────────────────────────┐
│                      Nginx (edge)                        │
│   TLS, body-size limit, gzip, byte-range for MP4, static  │
└───────────────┬───────────────────────────┬──────────────┘
                │ /api                      │ / (web UI)
┌───────────────▼───────────────┐   ┌───────▼────────┐
│  FastAPI API  (Uvicorn, N×)   │   │  Vite build    │
│  auth, rate limit, idempotency │   │  (React 18)    │
│  uploads → S3 (MinIO/AWS)     │   └────────────────┘
└──────┬───────────────┬────────┘
Postgres  │               │  Redis Streams
(SQLAlchemy asyncpg,     │  (queue, rate limits,
Alembic migrations)     │   GPU cluster lock)
       │               │
┌──────▼───────────────▼────────┐
│         GPU Worker            │
│ download → probe → validate   │
│ → features (CLIP + sync)      │
│ → flow-matching MMDiT windows │
│ → VAE decode → vocoder        │
│ → loudness/limit/mix/duck     │
│ → FFmpeg mux → S3 upload      │
│ → webhook + metrics           │
└──────┬───────────────┬────────┘
  S3 / MinIO       Prometheus / structlog
(sources, outputs)  (OTel-compatible ids)


Quickstart (docker compose, CPU dev profile)

cp .env.example .env
docker compose -f docker/docker-compose.yml up -d postgres redis minio
make migrate
make weights downloads/exports verified weight artifacts
python scripts/create_api_key.py --name dev
make dev api + worker + web ui

curl -s -H "Authorization: Bearer vsfx_XXXXXXXX_yyyy...yyyy"
-F video=@clip.mp4
-F "prompt=gritty realistic footsteps on gravel, distant wind"
http://localhost:8000/api/v1/predictions/video-to-video-sfx

Poll data.urls.get until data.status == "completed"; data.outputs[0] is the MP4.

Until the generator/V AE/vocoder/sync checkpoints exist in WEIGHTS_DIR (produced by
make train-generator MODAL_APP=modal/sfx_training.py or exported from a finished
Modal run), requests fail fast with error code weights_unavailable — by design; the
service never fabricates audio.

Repository map

Path	Purpose
app/	API, config, middleware, DB, services, media, ML inference, worker, telemetry
training/	dataset pipelines (FSD50K, VGGSound), trainers (VAE, vocoder, sync, MMDiT), export
modal/	Modal GPU training app
migrations/	Alembic schema history
scripts/	weights, keys, dev seed, test media, benchmark
sdk/python, sdk/typescript	typed clients
cli/	vsfx command line
web/	React 18 + Vite UI
tests/	unit, integration, load
docker/, deploy/k8s/	packaging and orchestration
docs/	API, architecture, model, training, deployment, operations, troubleshooting
Documentation

docs/api.md — every endpoint, field, error code, SDK/CLI examples, webhook signing
docs/model.md — exact model pipeline, variant table, window math, reproducibility
docs/training.md — Modal training runs, datasets, artifact export
docs/architecture.md, docs/deployment.md, docs/operations.md, docs/troubleshooting.md

Environment variables

See the fully documented table in .env.example; every setting lives under the VSFX_
prefix and is validated at startup by app/config.py (fail-fast with actionable errors).

License

MIT — see LICENSE. Third-party weight licences are recorded per artifact in
weights/manifest.json (generated at export time by training/export_weights.py).
