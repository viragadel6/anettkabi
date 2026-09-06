"""Modal GPU training pipeline for the Video-to-Video SFX generator.

Stages (each a Modal Function over a shared Volume):

  1. download_weights  - export CLIP towers + BPE vocab into the weights volume
  2. fetch_fsd50k      - download + index FSD50K into the datasets volume
  3. fetch_vggsound    - download + index VGGSound clips into the datasets volume
  4. train_vae         - mel-VAE on FSD50K
  5. train_vocoder     - BigVGAN-style vocoder on FSD50K
  6. train_sync        - sync encoder on VGGSound envelope grids
  7. build_shards      - paired feature shards (must run AFTER train_sync)
  8. train_generator   - flow-matching MMDiT over the shards
  9. export_weights    - package artifacts + manifest for the service

Run everything with:

    python -m modal.sfx_training            (or: modal run modal/sfx_training.py)

Stage selection:

    modal run modal/sfx_training.py --stages vae,vocoder
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

APP_NAME = "vsfx-training"
IMAGE_NAME = "vsfx-train"
REMOTE_ROOT = Path("/root/vsfx")
DATASETS_REMOTE = Path("/root/vsfx/datasets")
RUNS_REMOTE = Path("/root/vsfx/runs")
WEIGHTS_REMOTE = Path("/root/vsfx/weights")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "unzip", "zip", "p7zip-full", "ffmpeg", "sox")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "numpy==1.26.4",
        "soundfile==0.12.1",
        "soxr==0.5.0",
        "librosa==0.10.2.post1",
        "pydantic==2.9.2",
        "pydantic-settings==2.6.1",
        "structlog==24.4.0",
        "safetensors==0.4.5",
        "open-clip-torch==2.29.0",
        "yt-dlp==2024.12.13",
        "alembic==1.14.0",
        "asyncpg==0.30.0",
        "redis==5.2.1",
        "httpx==0.28.1",
        "python-multipart==0.0.20",
        "uvicorn==0.34.0",
    )
    .add_local_dir(
        local_path=Path(__file__).resolve().parent.parent / "app",
        remote_path=REMOTE_ROOT / "app",
        copy=False,
    )
    .add_local_dir(
        local_path=Path(__file__).resolve().parent.parent / "training",
        remote_path=REMOTE_ROOT / "training",
        copy=False,
    )
    .add_local_dir(
        local_path=Path(__file__).resolve().parent.parent / "scripts",
        remote_path=REMOTE_ROOT / "scripts",
        copy=False,
    )
    .add_local_dir(
        local_path=Path(__file__).resolve().parent.parent / "config",
        remote_path=REMOTE_ROOT / "config",
        copy=False,
    )
    .workdir(REMOTE_ROOT)
    .env(
        {
            "VSFX_TRAIN_DATA_DIR": str(DATASETS_REMOTE),
            "VSFX_TRAIN_OUTPUT_DIR": str(RUNS_REMOTE),
            "VSFX_TRAIN_WEIGHTS_DIR": str(WEIGHTS_REMOTE),
            "PYTHONPATH": str(REMOTE_ROOT),
        }
    )
)

datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)
runs_volume = modal.Volume.from_name(f"{APP_NAME}-runs", create_if_missing=True)
weights_volume = modal.Volume.from_name(f"{APP_NAME}-weights", create_if_missing=True)

app = modal.App(name=APP_NAME, image=train_image)

GPU_CONFIG = os.environ.get("VSFX_MODAL_GPU", "a10g")
VARIANT = os.environ.get("VSFX_MODAL_VARIANT", "small_16k")


def _train_settings_env(epochs_override: str) -> dict[str, str]:
    """Build the stage environment from CLI knobs.

    Parameters:
        epochs_override: Optional epoch override value.

    Returns:
        Env overrides consumed by training.config.get_train_settings.
    """
    env = {
        "VSFX_TRAIN_VARIANT": VARIANT,
        "VSFX_TRAIN_DEVICE": "cuda",
    }
    if epochs_override:
        env["VSFX_TRAIN_EPOCHS_OVERRIDE"] = epochs_override
    return env


@app.function(
    image=train_image,
    gpu=GPU_CONFIG,
    timeout=60 * 60 * 24,
    volumes={
        str(DATASETS_REMOTE): datasets_volume,
        str(RUNS_REMOTE): runs_volume,
        str(WEIGHTS_REMOTE): weights_volume,
    },
)
def run_stage(stage: str, epochs_override: str = "") -> str:
    """Execute one training stage on a Modal GPU.

    Parameters:
        stage: One of weights, fsd50k, vggsound, vae, vocoder, sync, shards,
            generator, export.
        epochs_override: Optional VSFX_TRAIN_EPOCHS_OVERRIDE value.

    Returns:
        Short human-readable result line.

    Raises:
        ValueError: On unknown stages.
    """
    os.environ.update(_train_settings_env(epochs_override))
    import subprocess
    import sys

    commands = {
        "weights": [sys.executable, "scripts/download_weights.py"],
        "fsd50k": [sys.executable, "-m", "training.data.fsd50k"],
        "vggsound": [sys.executable, "-m", "training.data.vggsound"],
        "vae": [sys.executable, "-m", "training.train_vae"],
        "vocoder": [sys.executable, "-m", "training.train_vocoder"],
        "sync": [sys.executable, "-m", "training.train_sync"],
        "shards": [sys.executable, "-m", "training.data.shards"],
        "generator": [sys.executable, "-m", "training.train_gen"],
        "export": [sys.executable, "-m", "training.export_weights"],
    }
    if stage not in commands:
        raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(commands)}")
    result = subprocess.run(commands[stage], check=False, cwd=str(REMOTE_ROOT))
    for volume in (datasets_volume, runs_volume, weights_volume):
        volume.commit()
    if result.returncode != 0:
        raise RuntimeError(f"stage {stage} failed with exit code {result.returncode}")
    return f"stage {stage} completed"


@app.function(image=train_image, timeout=60 * 60 * 2)
def stage_plan() -> str:
    """Return the ordered pipeline plan as JSON (no GPU required).

    Returns:
        JSON list of (stage, requires_gpu) pairs.
    """
    plan = [
        ("weights", False),
        ("fsd50k", False),
        ("vggsound", False),
        ("vae", True),
        ("vocoder", True),
        ("sync", True),
        ("shards", True),
        ("generator", True),
        ("export", False),
    ]
    return json.dumps(plan)


@app.local_entrypoint()
def main(stages: str = "all", epochs_override: str = "") -> None:
    """Run the training pipeline on Modal.

    GPU and variant are fixed at import time through VSFX_MODAL_GPU and
    VSFX_MODAL_VARIANT (e.g. `VSFX_MODAL_GPU=a100-40g modal run ...`).

    Parameters:
        stages: Comma-separated stage list, or `all` for the full pipeline.
        epochs_override: Optional epoch override (per-stage settings keep
            their defaults when empty).
    """
    full_order = [
        "weights",
        "fsd50k",
        "vggsound",
        "vae",
        "vocoder",
        "sync",
        "shards",
        "generator",
        "export",
    ]
    selected = full_order if stages.strip().lower() in ("", "all") else [
        item.strip() for item in stages.split(",") if item.strip()
    ]
    for stage in selected:
        print(f"=== vsfx-training stage: {stage} (gpu={GPU_CONFIG}) ===", flush=True)
        print(run_stage.remote(stage, epochs_override=epochs_override), flush=True)
    print("all requested stages finished")
