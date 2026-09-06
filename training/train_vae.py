"""Train the mel-spectrogram VAE on FSD50K."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import nn

from app.ml.audio_codec.vae import AudioVAE
from training.config import TrainSettings, get_train_settings
from training.data.fsd50k import iter_fsd50k
from training.engine import fit_loop
from training.torch_data import AudioClipDataset


def _split(entries: list[dict[str, object]], fraction: float, seed: int) -> tuple[list, list]:
    """Split entries deterministically.

    Parameters:
        entries: Full list.
        fraction: Validation fraction.
        seed: Shuffle seed.

    Returns:
        (train_entries, val_entries).
    """
    rng = random.Random(seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * fraction))
    return shuffled[cut:], shuffled[:cut]


def train(settings: TrainSettings) -> Path:
    """Run VAE training.

    Parameters:
        settings: Training settings.

    Returns:
        The checkpoint directory.

    Raises:
        RuntimeError: When no FSD50K clips are indexed.
    """
    spec = settings.spec
    entries = list(iter_fsd50k(settings))
    if not entries:
        raise RuntimeError(
            "no FSD50K clips with audio on disk; run `python -m training.data.fsd50k` "
            "to download the dev audio archive first"
        )
    train_entries, val_entries = _split(entries, settings.val_fraction, settings.seed)
    train_set = AudioClipDataset(settings, spec, train_entries)
    val_set = AudioClipDataset(settings, spec, val_entries)
    model = AudioVAE(spec.vae, spec.n_mels)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.lr_vae, weight_decay=1e-5, betas=(0.9, 0.99)
    )
    checkpoint_dir = settings.output_dir / settings.variant / "vae"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loss_fn(model: nn.Module, batch: dict[str, object]) -> tuple[torch.Tensor, dict[str, float]]:
        """Reconstruction + KL loss on mel batches.

        Parameters:
            model: The VAE.
            batch: Batch dict with mel.

        Returns:
            (loss, parts).
        """
        mel = batch["mel"]
        reconstruction, mean, logvar = model(mel)
        loss, parts = AudioVAE.vae_loss(reconstruction, mel, mean, logvar)
        return loss, parts

    def evaluate_fn(model: nn.Module, batch: dict[str, object]) -> dict[str, float]:
        """Validation pass.

        Parameters:
            model: The VAE (EMA candidate).
            batch: Batch dict.

        Returns:
            Metrics with a `loss` key.
        """
        mel = batch["mel"]
        reconstruction, mean, logvar = model(mel)
        loss, _parts = AudioVAE.vae_loss(reconstruction, mel, mean, logvar)
        return {"loss": float(loss)}

    fit_loop(
        stage="vae",
        model=model,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=val_set,
        epochs=settings.effective_epochs(settings.epochs_vae),
        batch_size=settings.batch_size_vae,
        device=device,
        num_workers=settings.num_workers,
        checkpoint_dir=checkpoint_dir,
        ema_decay=settings.ema_decay,
        accumulation_steps=settings.accumulation_steps,
        loss_fn=loss_fn,
        evaluate_fn=evaluate_fn,
    )
    _update_latent_statistics(checkpoint_dir, model, val_set, device)
    return checkpoint_dir


def _update_latent_statistics(checkpoint_dir: Path, model: AudioVAE, dataset: AudioClipDataset, device: torch.device) -> None:
    """Compute posterior latent statistics and store them as buffers.

    Parameters:
        checkpoint_dir: Checkpoint root.
        model: Trained VAE.
        dataset: Validation dataset.
        device: Compute device.
    """
    from torch.utils.data import DataLoader

    model.to(device).eval()
    means: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=8):
            mel = batch["mel"].to(device)
            mean, _logvar = model.encode(mel)
            means.append(mean.flatten(1).mean(dim=0).cpu())
    if means:
        stacked = torch.stack(means).mean(dim=0)
        model.latent_mean.copy_(stacked.mean().reshape(1).to(model.latent_mean.device))
        model.latent_scale.copy_(
            stacked.std().clamp(min=1e-4).reshape(1).to(model.latent_scale.device)
        )
    torch.save(model.state_dict(), checkpoint_dir / "vae-final.pt")


def main() -> int:
    """CLI entry for `python -m training.train_vae`.

    Returns:
        0 on success.
    """
    train(get_train_settings())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
