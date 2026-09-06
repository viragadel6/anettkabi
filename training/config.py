"""Training settings (VSFX_TRAIN_ prefix), validated."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.ml.registry import VariantSpec, get_variant

__all__ = ["TrainSettings", "get_train_settings"]


class TrainSettings(BaseSettings):
    """All knobs controlling dataset preparation and training runs.

    Attributes:
        variant: Registry variant to train.
        data_dir: Dataset root (FSD50K/VGGSound downloads).
        output_dir: Checkpoint/run directory.
        weights_dir: Export target (must equal service WEIGHTS_DIR).
        batch_size_vae / batch_size_vocoder / batch_size_sync / batch_size_gen: Stage batch sizes.
        lr_*: Per-stage learning rates.
        epochs_*: Per-stage epoch budgets.
        num_workers: DataLoader workers.
        val_fraction: Held-out fraction.
        seed: Global training seed.
        device: cuda|cpu.
        ema_decay: EMA coefficient for generator weights.
        accumulation_steps: Grad accumulation for large variants.
        vggsound_max_clips: Cap on scraped VGGSound clips.
        clip_length_s: Audio clip length per training example.
        modal_gpu: GPU string for Modal (a10g, a100-40g, ...).
        hf_repo: Optional HF repo id for exported artifacts.
        dataset_url_fsd50k_eval: Zenodo record carrying the eval audio (the
            official FSD50K release keeps dev and eval on the same record
            4060432; the previously documented 4273845 is an unrelated paper).
    """

    model_config = SettingsConfigDict(
        env_prefix="VSFX_TRAIN_",
        extra="forbid",
        case_sensitive=False,
        env_file=".env",
    )

    variant: str = "small_16k"
    data_dir: Path = Path("./datasets")
    output_dir: Path = Path("./runs")
    weights_dir: Path = Path("./weights")
    batch_size_vae: int = Field(default=16, ge=1)
    batch_size_vocoder: int = Field(default=8, ge=1)
    batch_size_sync: int = Field(default=12, ge=1)
    batch_size_gen: int = Field(default=32, ge=1)
    lr_vae: float = Field(default=2e-4, gt=0)
    lr_vocoder: float = Field(default=1e-4, gt=0)
    lr_sync: float = Field(default=3e-4, gt=0)
    lr_gen: float = Field(default=1e-4, gt=0)
    epochs_vae: int = Field(default=40, ge=1)
    epochs_vocoder: int = Field(default=400, ge=1)
    epochs_sync: int = Field(default=20, ge=1)
    epochs_gen: int = Field(default=100, ge=1)
    num_workers: int = Field(default=4, ge=0)
    val_fraction: float = Field(default=0.02, gt=0, le=0.5)
    seed: int = Field(default=4242, ge=0)
    device: str = "cuda"
    ema_decay: float = Field(default=0.999, gt=0, le=1)
    accumulation_steps: int = Field(default=1, ge=1)
    vggsound_max_clips: int = Field(default=20000, ge=100)
    clip_length_s: float = Field(default=10.0, gt=0)
    epochs_override: int = Field(default=0, ge=0)
    modal_gpu: str = "a10g"
    hf_repo: str = ""
    dataset_url_fsd50k_dev: str = "https://zenodo.org/records/4060432"
    dataset_url_fsd50k_eval: str = "https://zenodo.org/records/4060432"
    dataset_vggsound_csv: str = (
        "https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound.csv"
    )

    def effective_epochs(self, stage_epochs: int) -> int:
        """Resolve an epoch override for a stage.

        Parameters:
            stage_epochs: Stage default.

        Returns:
            The override when set (> 0), else the stage default.
        """
        return self.epochs_override if self.epochs_override > 0 else stage_epochs

    @property
    def spec(self) -> VariantSpec:
        """Return the registry spec for the configured variant.

        Returns:
            VariantSpec.

        Raises:
            KeyError: For unknown variants.
        """
        return get_variant(self.variant)


def get_train_settings() -> TrainSettings:
    """Construct training settings from the environment.

    Returns:
        Validated TrainSettings.

    Raises:
        ValueError: With an actionable message when invalid.
    """
    try:
        settings = TrainSettings()
        _ = settings.spec
    except Exception as exc:
        raise ValueError(f"invalid training configuration: {exc}") from exc
    return settings
