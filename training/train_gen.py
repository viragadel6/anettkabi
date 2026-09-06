"""Train the flow-matching MMDiT generator over paired feature shards."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import Dataset, Subset

from app.ml.audio_codec.vae import AudioVAE
from app.ml.encoders.clip_text import CLIPTextEncoder
from app.ml.encoders.tokenizer import CliPTokenizer
from app.ml.generator.flow_matching import flow_matching_loss, interpolate, sample_t
from app.ml.generator.mmdit import MMDiTGenerator
from training.config import TrainSettings, get_train_settings
from training.engine import fit_loop
from training.torch_data import PairedFeatureDataset

PROMPT_DROP_P = 0.1


def _load_vae(settings: TrainSettings, spec: Any, device: torch.device) -> AudioVAE:
    """Load the trained VAE for latent supervision.

    Parameters:
        settings: Training settings.
        spec: Variant spec.
        device: Target device.

    Returns:
        Frozen eval AudioVAE.

    Raises:
        FileNotFoundError: When no trained VAE checkpoint exists.
    """
    vae_dir = settings.output_dir / settings.variant / "vae"
    for name in ("vae-final.pt", "vae-best.pt", "vae-last.pt"):
        path = vae_dir / name
        if path.is_file():
            vae = AudioVAE(spec.vae, spec.n_mels)
            vae.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            return vae.to(device).eval()
    raise FileNotFoundError(
        f"no trained VAE under {vae_dir}; run python -m training.train_vae first"
    )


def _load_text_stack(settings: TrainSettings, device: torch.device) -> tuple[CliPTokenizer, CLIPTextEncoder]:
    """Load the exported CLIP text tower and tokenizer.

    Parameters:
        settings: Training settings.
        device: Target device.

    Returns:
        (tokenizer, frozen text encoder).

    Raises:
        FileNotFoundError: When exports are missing.
    """
    vocab = settings.weights_dir / "shared" / "clip_bpe_vocab.json"
    merges = settings.weights_dir / "shared" / "clip_bpe_merges.txt"
    tower = settings.weights_dir / settings.variant / "clip_text.safetensors"
    for path in (vocab, merges, tower):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} missing; run python scripts/download_weights.py first"
            )
    tokenizer = CliPTokenizer(vocab, merges)
    encoder = CLIPTextEncoder(vocab_size=tokenizer.vocab_size)
    encoder.load_state_dict(load_file(str(tower)))
    return tokenizer, encoder.to(device).eval()


class _GeneratorTrainWrapper(nn.Module):
    """Bundles the generator with its frozen conditioning stack."""

    def __init__(
        self,
        generator: MMDiTGenerator,
        vae: AudioVAE,
        tokenizer: CliPTokenizer,
        text_encoder: CLIPTextEncoder,
        spec: Any,
        seed: int,
    ) -> None:
        """Store members.

        Parameters:
            generator: Trainable generator.
            vae: Frozen VAE.
            tokenizer: Prompt tokenizer.
            text_encoder: Frozen text tower.
            spec: Variant spec.
            seed: RNG seed for prompt dropout.
        """
        super().__init__()
        self.generator = generator
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.spec = spec
        self.seed = seed
        self._drop_counter = 0

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """One flow-matching step.

        Parameters:
            batch: With mel, visual, sync, prompt.

        Returns:
            (loss, parts).
        """
        device = next(self.generator.parameters()).device
        mel = batch["mel"]
        visual = batch["visual"]
        sync = batch["sync"]
        prompts = list(batch["prompt"])
        self._drop_counter += 1
        rng = random.Random(self.seed * 7919 + self._drop_counter)
        dropped = ["" if rng.random() < PROMPT_DROP_P else text for text in prompts]
        tokens = self.tokenizer.encode_batch(dropped).to(device)
        with torch.no_grad():
            text_tokens, pooled_text = self.text_encoder.encode_tokens(tokens)
            mean, logvar = self.vae.encode(mel)
            x1 = self.vae.normalize(self.vae.reparameterize(mean, logvar)).permute(0, 2, 1)
        x0 = torch.randn_like(x1)
        t = sample_t(x1.shape[0], device)
        x_t = interpolate(x0, x1, t)
        visual_seconds = torch.arange(visual.shape[1], device=device, dtype=torch.float32) / 8.0
        sync_seconds = torch.arange(sync.shape[1], device=device, dtype=torch.float32) / 25.0
        velocity = self.generator(
            x_t,
            t,
            text_tokens,
            pooled_text,
            visual,
            visual_seconds,
            sync,
            sync_seconds,
            8.0,
            25.0,
            visual.mean(dim=1),
        )
        loss = flow_matching_loss(velocity, x0, x1)
        return loss, {"flow": float(loss)}


def train(settings: TrainSettings) -> Path:
    """Run generator training.

    Parameters:
        settings: Training settings.

    Returns:
        The checkpoint directory.

    Raises:
        FileNotFoundError: When shards or prerequisite checkpoints are missing.
    """
    spec = settings.spec
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shard_dir = settings.data_dir / "shards" / settings.variant
    dataset = PairedFeatureDataset(shard_dir)
    total = len(dataset)
    cut = max(1, int(total * (1 - settings.val_fraction)))
    train_set: Dataset = Subset(dataset, list(range(cut)))
    val_set: Dataset = Subset(dataset, list(range(cut, total)))
    tokenizer, text_encoder = _load_text_stack(settings, device)
    vae = _load_vae(settings, spec, device)
    generator = MMDiTGenerator(
        spec.mmdit,
        latent_channels=spec.vae.latent_channels,
        latent_fps=spec.latent_fps,
    )
    wrapper = _GeneratorTrainWrapper(generator, vae, tokenizer, text_encoder, spec, settings.seed)
    optimizer = torch.optim.AdamW(
        list(generator.parameters()),
        lr=settings.lr_gen,
        weight_decay=1e-4,
        betas=(0.9, 0.95),
    )
    checkpoint_dir = settings.output_dir / settings.variant / "generator"

    def loss_fn(model: nn.Module, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """Delegate to the wrapper with the live generator.

        Parameters:
            model: Live generator from the fit loop.
            batch: Batch.

        Returns:
            (loss, parts).
        """
        previous = wrapper.generator
        wrapper.generator = model
        try:
            return wrapper(batch)
        finally:
            wrapper.generator = previous

    def evaluate_fn(model: nn.Module, batch: dict[str, Any]) -> dict[str, float]:
        """Validation flow-matching loss of the candidate generator.

        Parameters:
            model: Live or EMA candidate generator.
            batch: Batch.

        Returns:
            Metrics with `loss`.
        """
        previous = wrapper.generator
        wrapper.generator = model
        model.eval()
        try:
            with torch.no_grad():
                loss, _parts = wrapper(batch)
        finally:
            wrapper.generator = previous
        return {"loss": float(loss)}

    fit_loop(
        stage="generator",
        model=generator,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=val_set,
        epochs=settings.effective_epochs(settings.epochs_gen),
        batch_size=settings.batch_size_gen,
        device=device,
        num_workers=settings.num_workers,
        checkpoint_dir=checkpoint_dir,
        ema_decay=settings.ema_decay,
        accumulation_steps=settings.accumulation_steps,
        loss_fn=loss_fn,
        evaluate_fn=evaluate_fn,
    )
    torch.save(generator.state_dict(), checkpoint_dir / "generator-final.pt")
    return checkpoint_dir


def main() -> int:
    """CLI entry for `python -m training.train_gen`.

    Returns:
        0 on success.
    """
    train(get_train_settings())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
