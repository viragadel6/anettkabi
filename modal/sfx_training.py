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
REMOTE_ROOT_POSIX = REMOTE_ROOT.as_posix()
DATASETS_REMOTE_POSIX = DATASETS_REMOTE.as_posix()
RUNS_REMOTE_POSIX = RUNS_REMOTE.as_posix()
WEIGHTS_REMOTE_POSIX = WEIGHTS_REMOTE.as_posix()
LOCAL_ROOT = Path(__file__).resolve().parent.parent

STAGE_ORDER: tuple[str, ...] = (
    "weights",
    "fsd50k",
    "vggsound",
    "vae",
    "vocoder",
    "sync",
    "shards",
    "generator",
    "export",
)
GPU_STAGES: tuple[str, ...] = (
    "vae",
    "vocoder",
    "sync",
    "shards",
    "generator",
)
CPU_STAGES: tuple[str, ...] = tuple(
    stage for stage in STAGE_ORDER if stage not in GPU_STAGES
)
STAGE_REQUIRES_GPU: dict[str, bool] = {
    stage: stage in GPU_STAGES for stage in STAGE_ORDER
}
STAGE_COMMANDS: dict[str, tuple[str, ...]] = {
    "weights": ("scripts/download_weights.py",),
    "fsd50k": ("-m", "training.data.fsd50k"),
    "vggsound": ("-m", "training.data.vggsound"),
    "vae": ("-m", "training.train_vae"),
    "vocoder": ("-m", "training.train_vocoder"),
    "sync": ("-m", "training.train_sync"),
    "shards": ("-m", "training.data.shards"),
    "generator": ("-m", "training.train_gen"),
    "export": ("-m", "training.export_weights"),
}


def _normalize_gpu_config(value: str | None) -> str:
    raw_value = (value or "").strip()
    if not raw_value:
        raw_value = "A10G"
    base_value, separator, count_value = raw_value.partition(":")
    base_value = base_value.strip() or "A10G"
    normalized_key = base_value.lower().replace("_", "-")
    aliases = {
        "t4": "T4",
        "l4": "L4",
        "a10g": "A10G",
        "a100": "A100",
        "a100-40g": "A100-40GB",
        "a100-40gb": "A100-40GB",
        "a10040g": "A100-40GB",
        "a10040gb": "A100-40GB",
        "a100-80g": "A100-80GB",
        "a100-80gb": "A100-80GB",
        "a10080g": "A100-80GB",
        "a10080gb": "A100-80GB",
        "h100": "H100",
        "h100-80g": "H100",
        "h100-80gb": "H100",
        "any": "any",
    }
    normalized_base = aliases.get(normalized_key, base_value.upper())
    if separator:
        normalized_count = count_value.strip()
        if normalized_count:
            return f"{normalized_base}:{normalized_count}"
    return normalized_base


def _normalize_variant(value: str | None) -> str:
    normalized_value = (value or "").strip()
    return normalized_value or "small_16k"


GPU_CONFIG = _normalize_gpu_config(os.environ.get("VSFX_MODAL_GPU"))
VARIANT = _normalize_variant(os.environ.get("VSFX_MODAL_VARIANT"))

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ca-certificates",
        "git",
        "unzip",
        "zip",
        "p7zip-full",
        "ffmpeg",
        "sox",
        "libsox-fmt-all",
        "libsndfile1",
        "libgomp1",
    )
    .pip_install(
        "torch>=2.4,<3",
        "torchaudio>=2.4,<3",
        "torchvision>=0.19,<1",
        "numpy>=1.26,<3",
        "soundfile>=0.12,<1",
        "soxr>=0.3,<2",
        "librosa>=0.10.2,<2",
        "pydantic>=2,<3",
        "pydantic-settings>=2,<3",
        "structlog>=24,<27",
        "safetensors>=0.4,<1",
        "open-clip-torch>=2.26,<4",
        "yt-dlp>=2024.8.6",
        "alembic>=1.13,<2",
        "asyncpg>=0.29,<1",
        "redis>=5,<9",
        "httpx>=0.27,<1",
        "python-multipart>=0.0.9,<1",
        "uvicorn>=0.30,<1",
    )
    .add_local_dir(
        local_path=LOCAL_ROOT / "app",
        remote_path=(REMOTE_ROOT / "app").as_posix(),
        copy=False,
    )
    .add_local_dir(
        local_path=LOCAL_ROOT / "training",
        remote_path=(REMOTE_ROOT / "training").as_posix(),
        copy=False,
    )
    .add_local_dir(
        local_path=LOCAL_ROOT / "scripts",
        remote_path=(REMOTE_ROOT / "scripts").as_posix(),
        copy=False,
    )
    .add_local_dir(
        local_path=LOCAL_ROOT / "config",
        remote_path=(REMOTE_ROOT / "config").as_posix(),
        copy=False,
    )
    .workdir(REMOTE_ROOT_POSIX)
    .env(
        {
            "VSFX_TRAIN_DATA_DIR": DATASETS_REMOTE_POSIX,
            "VSFX_TRAIN_OUTPUT_DIR": RUNS_REMOTE_POSIX,
            "VSFX_TRAIN_WEIGHTS_DIR": WEIGHTS_REMOTE_POSIX,
            "VSFX_TRAIN_VARIANT": VARIANT,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": REMOTE_ROOT_POSIX,
        }
    )
)

datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)
runs_volume = modal.Volume.from_name(f"{APP_NAME}-runs", create_if_missing=True)
weights_volume = modal.Volume.from_name(f"{APP_NAME}-weights", create_if_missing=True)

VOLUME_MOUNTS = {
    DATASETS_REMOTE_POSIX: datasets_volume,
    RUNS_REMOTE_POSIX: runs_volume,
    WEIGHTS_REMOTE_POSIX: weights_volume,
}
TRAINING_VOLUMES = (datasets_volume, runs_volume, weights_volume)

app = modal.App(name=APP_NAME, image=train_image)


def _validate_stage(stage: str) -> str:
    normalized_stage = stage.strip().lower()
    if normalized_stage not in STAGE_COMMANDS:
        expected_stages = ", ".join(sorted(STAGE_COMMANDS))
        raise ValueError(
            f"unknown stage {stage!r}; expected one of {expected_stages}"
        )
    return normalized_stage


def _selected_stages(stages: str) -> list[str]:
    requested_stages = (stages or "").strip()
    if requested_stages.lower() in ("", "all"):
        return list(STAGE_ORDER)
    selected_stages = [
        _validate_stage(stage_name.strip())
        for stage_name in requested_stages.split(",")
        if stage_name.strip()
    ]
    if not selected_stages:
        raise ValueError(
            "no stages selected; use 'all' or one or more comma-separated stage names"
        )
    return selected_stages


def _command_for_stage(stage: str) -> list[str]:
    import sys

    return [sys.executable, *STAGE_COMMANDS[stage]]


def _train_settings_env(
    epochs_override: str,
    stage: str = "generator",
    variant: str | None = None,
) -> dict[str, str]:
    normalized_stage = _validate_stage(stage)
    selected_variant = VARIANT if variant is None else variant
    env = {
        "VSFX_TRAIN_VARIANT": _normalize_variant(selected_variant),
        "VSFX_TRAIN_DEVICE": (
            "cuda" if STAGE_REQUIRES_GPU[normalized_stage] else "cpu"
        ),
    }
    normalized_epochs_override = (epochs_override or "").strip()
    if normalized_epochs_override:
        env["VSFX_TRAIN_EPOCHS_OVERRIDE"] = normalized_epochs_override
    return env


def _apply_train_settings_env(
    stage: str,
    epochs_override: str,
    variant: str,
) -> None:
    os.environ.update(
        _train_settings_env(
            epochs_override=epochs_override,
            stage=stage,
            variant=variant,
        )
    )
    if not (epochs_override or "").strip():
        os.environ.pop("VSFX_TRAIN_EPOCHS_OVERRIDE", None)


def _reload_volumes() -> None:
    for volume in TRAINING_VOLUMES:
        volume.reload()


def _commit_volumes() -> None:
    for volume in TRAINING_VOLUMES:
        volume.commit()


def _ensure_remote_directories() -> None:
    for path in (REMOTE_ROOT, DATASETS_REMOTE, RUNS_REMOTE, WEIGHTS_REMOTE):
        path.mkdir(parents=True, exist_ok=True)


def _execute_stage(stage: str, epochs_override: str, variant: str) -> str:
    normalized_stage = _validate_stage(stage)
    _reload_volumes()
    _ensure_remote_directories()
    _apply_train_settings_env(
        stage=normalized_stage,
        epochs_override=epochs_override,
        variant=variant,
    )

    import subprocess

    command = _command_for_stage(normalized_stage)
    result = None
    try:
        result = subprocess.run(command, check=False, cwd=REMOTE_ROOT_POSIX)
    finally:
        _commit_volumes()

    if result is None:
        raise RuntimeError(f"stage {normalized_stage} did not start")
    if result.returncode != 0:
        raise RuntimeError(
            f"stage {normalized_stage} failed with exit code {result.returncode}"
        )
    return f"stage {normalized_stage} completed"


@app.function(
    image=train_image,
    gpu=GPU_CONFIG,
    timeout=60 * 60 * 24,
    volumes=VOLUME_MOUNTS,
)
def run_stage(stage: str, epochs_override: str = "", variant: str = VARIANT) -> str:
    return _execute_stage(
        stage=stage,
        epochs_override=epochs_override,
        variant=variant,
    )


@app.function(
    image=train_image,
    timeout=60 * 60 * 24,
    volumes=VOLUME_MOUNTS,
)
def run_cpu_stage(
    stage: str,
    epochs_override: str = "",
    variant: str = VARIANT,
) -> str:
    normalized_stage = _validate_stage(stage)
    if normalized_stage not in CPU_STAGES:
        cpu_stage_list = ", ".join(CPU_STAGES)
        raise ValueError(
            f"stage {normalized_stage!r} requires a GPU; CPU stages are {cpu_stage_list}"
        )
    return _execute_stage(
        stage=normalized_stage,
        epochs_override=epochs_override,
        variant=variant,
    )


@app.function(image=train_image, timeout=60 * 60 * 2)
def stage_plan() -> str:
    plan = [(stage, STAGE_REQUIRES_GPU[stage]) for stage in STAGE_ORDER]
    return json.dumps(plan)


@app.local_entrypoint()
def main(stages: str = "all", epochs_override: str = "") -> None:
    selected_stages = _selected_stages(stages)
    for stage in selected_stages:
        requires_gpu = STAGE_REQUIRES_GPU[stage]
        gpu_label = GPU_CONFIG if requires_gpu else "none"
        print(
            f"=== {APP_NAME} stage: {stage} (image={IMAGE_NAME}, gpu={gpu_label}) ===",
            flush=True,
        )
        if requires_gpu:
            stage_result = run_stage.remote(
                stage,
                epochs_override=epochs_override,
                variant=VARIANT,
            )
        else:
            stage_result = run_cpu_stage.remote(
                stage,
                epochs_override=epochs_override,
                variant=VARIANT,
            )
        print(stage_result, flush=True)
    print("all requested stages finished")
