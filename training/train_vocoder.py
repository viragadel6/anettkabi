"""Train the BigVGAN-style vocoder on FSD50K."""

from __future__ import annotations

import torch
from torch import nn

from app.ml.audio_codec.vocoder import BigVGANVocoder
from training.config import TrainSettings, get_train_settings
from training.data.fsd50k import iter_fsd50k
from training.engine import fit_loop
from training.torch_data import AudioClipDataset
from training.train_vae import _split


def stft_loss(predicted: torch.Tensor, target: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """Multi-band STFT magnitude loss (parallel-wavegan style).

    Parameters:
        predicted: [B, 1, T] waveform.
        target: [B, 1, T] waveform.
        n_fft: FFT size.
        hop: Hop length.

    Returns:
        Scalar L1 magnitude loss.
    """
    window = torch.hann_window(n_fft, device=predicted.device)
    pred_spec = torch.stft(
        predicted.squeeze(1), n_fft, hop_length=hop, window=window, return_complex=True
    ).abs()
    target_spec = torch.stft(
        target.squeeze(1), n_fft, hop_length=hop, window=window, return_complex=True
    ).abs()
    return torch.nn.functional.l1_loss(pred_spec, target_spec)


def train(settings: TrainSettings) -> object:
    """Run vocoder training (GAN: generator + dual discriminators).

    Parameters:
        settings: Training settings.

    Returns:
        The checkpoint directory Path.

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
    generator = BigVGANVocoder(spec.vocoder, spec.n_mels)
    window_g = 2048
    hop_g = spec.hop_length * 2
    discriminator = _MultiPeriodDiscriminator()
    opt_g = torch.optim.AdamW(
        generator.parameters(), lr=settings.lr_vocoder, betas=(0.5, 0.999)
    )
    opt_d = torch.optim.AdamW(
        discriminator.parameters(), lr=settings.lr_vocoder, betas=(0.5, 0.999)
    )
    checkpoint_dir = settings.output_dir / settings.variant / "vocoder"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loss_fn(model: nn.Module, batch: dict[str, object]) -> tuple[torch.Tensor, dict[str, float]]:
        """Generator GAN + STFT loss for one batch.

        Parameters:
            model: Generator (vocoder).
            batch: Batch with mel and waveform.

        Returns:
            (loss, parts).
        """
        del model
        generator.train()
        discriminator.to(device).train()
        mel = batch["mel"]
        waveform = batch["waveform"].unsqueeze(1)
        generated = generator(mel)[..., : waveform.shape[-1]]
        target = waveform[..., : generated.shape[-1]]
        d_real = discriminator(target)
        d_fake = discriminator(generated.detach())
        loss_d = _discriminator_loss(d_real, d_fake)
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()
        d_fake_g = discriminator(generated)
        adv = _generator_adv_loss(d_fake_g)
        spec_loss = stft_loss(generated, target, window_g, hop_g) + stft_loss(
            generated, target, 512, 128
        )
        loss_g = spec_loss + 2.5 * adv
        parts = {"gen_total": float(loss_g), "adv": float(adv), "spec": float(spec_loss)}
        return loss_g, parts

    def evaluate_fn(model: nn.Module, batch: dict[str, object]) -> dict[str, float]:
        """Validation: STFT reconstruction quality of the (EMA) generator.

        Parameters:
            model: Generator candidate.
            batch: Batch with mel and waveform.

        Returns:
            Metrics with `loss`.
        """
        model.eval()
        mel = batch["mel"]
        waveform = batch["waveform"].unsqueeze(1)
        with torch.no_grad():
            generated = model(mel)[..., : waveform.shape[-1]]
            target = waveform[..., : generated.shape[-1]]
            loss = stft_loss(generated, target, window_g, hop_g)
        return {"loss": float(loss)}

    fit_loop(
        stage="vocoder",
        model=generator,
        optimizer=opt_g,
        train_dataset=train_set,
        val_dataset=val_set,
        epochs=settings.effective_epochs(settings.epochs_vocoder),
        batch_size=settings.batch_size_vocoder,
        device=device,
        num_workers=settings.num_workers,
        checkpoint_dir=checkpoint_dir,
        ema_decay=settings.ema_decay,
        accumulation_steps=settings.accumulation_steps,
        loss_fn=loss_fn,
        evaluate_fn=evaluate_fn,
    )
    torch.save(generator.state_dict(), checkpoint_dir / "vocoder-final.pt")
    return checkpoint_dir


class _MultiPeriodDiscriminator(nn.Module):
    """Stack of period discriminators (periods 2,3,5,7,11)."""

    def __init__(self) -> None:
        """Build five period branches."""
        super().__init__()
        self.periods = (2, 3, 5, 7, 11)
        self.branches = nn.ModuleList(
            [nn.Conv1d(1, 1, kernel_size=period * 4, stride=period, groups=1) for period in self.periods]
        )

    def forward(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        """Score a waveform per period.

        Parameters:
            waveform: [B, 1, T].

        Returns:
            List of branch logits.
        """
        outputs: list[torch.Tensor] = []
        for branch in self.branches:
            outputs.append(branch(waveform))
        return outputs


def _discriminator_loss(d_real: list[torch.Tensor], d_fake: list[torch.Tensor]) -> torch.Tensor:
    """Least-squares discriminator objective.

    Parameters:
        d_real: Real logits per branch.
        d_fake: Fake logits per branch.

    Returns:
        Scalar loss.
    """
    loss = 0.0
    for real, fake in zip(d_real, d_fake, strict=True):
        loss = loss + torch.mean((real - 1.0) ** 2) + torch.mean(fake**2)
    return loss / len(d_real)


def _generator_adv_loss(d_fake: list[torch.Tensor]) -> torch.Tensor:
    """Least-squares generator adversarial objective.

    Parameters:
        d_fake: Fake logits per branch.

    Returns:
        Scalar loss.
    """
    loss = 0.0
    for fake in d_fake:
        loss = loss + torch.mean((fake - 1.0) ** 2)
    return loss / len(d_fake)


def main() -> int:
    """CLI entry for `python -m training.train_vocoder`.

    Returns:
        0 on success.
    """
    train(get_train_settings())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
