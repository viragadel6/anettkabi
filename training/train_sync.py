"""Train the sync encoder on envelope-driven synthetic grids from VGGSound audio.

The grid construction is the SAME deterministic procedure used to build
feature shards (`training.data.shards._envelope_frames`), so the encoder
learns exactly the time-aligned signal the generator will later consume.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from app.ml.encoders.sync_encoder import SYNC_NORMALIZE_MEAN, SYNC_NORMALIZE_STD, SyncEncoder
from training.config import TrainSettings, get_train_settings
from training.data.shards import _envelope_frames
from training.data.vggsound import iter_vggsound
from training.engine import fit_loop
from training.torch_data import load_waveform_clip

SEGMENTS_PER_CLIP = 2


class VGGSoundSyncDataset(Dataset):
    """Random segment crops of envelope grids with their intensity targets."""

    __slots__ = ("entries", "seed", "settings", "spec")

    def __init__(self, settings: TrainSettings, spec: Any, entries: list[dict[str, Any]]) -> None:
        """Store dataset inputs.

        Parameters:
            settings: Training settings.
            spec: Variant spec.
            entries: VGGSound index entries.
        """
        self.settings = settings
        self.spec = spec
        self.entries = entries
        self.seed = settings.seed

    def __len__(self) -> int:
        """Number of clips.

        Returns:
            Dataset length.
        """
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Sample segments for one clip.

        Parameters:
            index: Clip index.

        Returns:
            Dict with uint8 `segments` [k, 3, F, 224, 224] and float
            `intensity` [k, out_per_segment] targets.
        """
        entry = self.entries[index]
        rng = random.Random(self.seed * 1_000_003 + index)
        spec = self.spec.sync
        waveform = load_waveform_clip(
            Path(str(entry.get("path") or "")), self.spec.sample_rate, self.spec.window_s
        )
        if waveform is None or float(np.abs(waveform).max()) < 1e-4:
            fallback = torch.zeros(
                SEGMENTS_PER_CLIP, 3, spec.segment_frames, 224, 224, dtype=torch.uint8
            )
            return {
                "segments": fallback,
                "intensity": torch.zeros(SEGMENTS_PER_CLIP, spec.out_per_segment),
            }
        grid = _envelope_frames(waveform, self.spec)
        frame_count = grid.shape[2]
        starts = list(range(0, frame_count - spec.segment_frames + 1, spec.segment_stride))
        if not starts:
            starts = [max(0, frame_count - spec.segment_frames)]
        chosen = rng.sample(starts, min(SEGMENTS_PER_CLIP, len(starts)))
        while len(chosen) < SEGMENTS_PER_CLIP:
            chosen.append(chosen[0])
        segments = torch.stack([grid[:, start : start + spec.segment_frames] for start in chosen])
        targets = torch.stack(
            [
                _frame_rms(
                    waveform, start, spec.segment_frames, self.spec.sample_rate, spec.out_per_segment
                )
                for start in chosen
            ]
        )
        return {"segments": segments, "intensity": targets}


def _frame_rms(
    waveform: np.ndarray, start: int, frames: int, sample_rate: int, out_per_segment: int
) -> torch.Tensor:
    """Per-frame normalized RMS for the last `out_per_segment` frames.

    Parameters:
        waveform: Mono waveform.
        start: First frame index.
        frames: Segment length in frames.
        sample_rate: Audio sample rate.

    Returns:
        [out_per_segment] intensities in [0, 1].
    """
    frame_len = sample_rate // 25
    tail_start = frames - out_per_segment
    values = []
    for offset in range(tail_start, frames):
        lo = (start + offset) * frame_len
        hi = lo + frame_len
        chunk = waveform[lo:hi]
        values.append(float(np.sqrt(np.mean(chunk**2))) if chunk.size else 0.0)
    arr = np.array(values, dtype=np.float32)
    peak = max(float(np.abs(waveform).max()), 1e-6)
    return torch.from_numpy(np.clip(arr / peak, 0.0, 1.0))


class _SyncAdapter(nn.Module):
    """Sync encoder plus contrastive projection and intensity heads."""

    def __init__(self, encoder: SyncEncoder) -> None:
        """Build heads.

        Parameters:
            encoder: Backbone.
        """
        super().__init__()
        self.encoder = encoder
        self.project = nn.Linear(encoder.spec.dim, 256)
        self.intensity_head = nn.Linear(encoder.spec.dim, 1)
        self.envelope_head = nn.Sequential(
            nn.Linear(encoder.spec.out_per_segment, 128),
            nn.GELU(),
            nn.Linear(128, 256),
        )

    def forward(self, segments: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode normalized segments.

        Parameters:
            segments: [N, 3, F, 224, 224] float.

        Returns:
            (projections [N, 256], token intensities [N, out]).
        """
        features = self.encoder(segments)
        projections = torch.nn.functional.normalize(self.project(features.mean(dim=1)), dim=-1)
        intensities = torch.sigmoid(self.intensity_head(features)).squeeze(-1)
        return projections, intensities


def _normalize(segments: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert uint8 segments to normalized float.

    Parameters:
        segments: [N, 3, F, H, W] uint8.
        device: Target device.

    Returns:
        Normalized float tensor.
    """
    mean = torch.tensor(SYNC_NORMALIZE_MEAN, device=device).reshape(1, 3, 1, 1, 1)
    std = torch.tensor(SYNC_NORMALIZE_STD, device=device).reshape(1, 3, 1, 1, 1)
    return (segments.to(device=device, dtype=torch.float32) / 255.0 - mean) / std


def train(settings: TrainSettings) -> Path:
    """Run sync-encoder training.

    Parameters:
        settings: Training settings.

    Returns:
        The checkpoint directory.

    Raises:
        RuntimeError: When VGGSound clips are unavailable.
    """
    spec = settings.spec
    entries = [item for item in iter_vggsound(settings) if item.get("path")]
    if not entries:
        raise RuntimeError("VGGSound index empty; run python -m training.data.vggsound first")
    rng = random.Random(settings.seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * settings.val_fraction))
    train_set = VGGSoundSyncDataset(settings, spec, shuffled[cut:])
    val_set = VGGSoundSyncDataset(settings, spec, shuffled[:cut])
    encoder = SyncEncoder(spec.sync)
    model = _SyncAdapter(encoder)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr_sync, weight_decay=1e-4)
    checkpoint_dir = settings.output_dir / settings.variant / "sync"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loss_fn(model: nn.Module, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """InfoNCE + intensity regression for one batch.

        Parameters:
            model: Adapter.
            batch: With `segments` [B, k, 3, F, H, W] uint8 and `intensity`.

        Returns:
            (loss, parts).
        """
        segments = batch["segments"]
        intensity = batch["intensity"].to(device)
        b, k = segments.shape[0], segments.shape[1]
        flat = _normalize(segments.reshape(b * k, *segments.shape[2:]), device)
        projections, token_intensities = model(flat)
        anchor = projections.reshape(b, k, -1).mean(dim=1)
        positive = torch.nn.functional.normalize(
            model.envelope_head(intensity.reshape(b, k, -1).mean(dim=1)), dim=-1
        )
        logits = anchor @ positive.t()
        labels = torch.arange(b, device=device)
        infonce = torch.nn.functional.cross_entropy(logits, labels)
        target_flat = intensity.reshape(b * k, -1)
        intensity_loss = torch.nn.functional.mse_loss(token_intensities, target_flat)
        loss = infonce + 0.5 * intensity_loss
        return loss, {"infonce": float(infonce), "intensity": float(intensity_loss)}

    def evaluate_fn(model: nn.Module, batch: dict[str, Any]) -> dict[str, float]:
        """Validation loss.

        Parameters:
            model: Candidate.
            batch: Batch.

        Returns:
            Metrics with `loss`.
        """
        loss, _parts = loss_fn(model, batch)
        return {"loss": float(loss)}

    fit_loop(
        stage="sync",
        model=model,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=val_set,
        epochs=settings.effective_epochs(settings.epochs_sync),
        batch_size=settings.batch_size_sync,
        device=device,
        num_workers=settings.num_workers,
        checkpoint_dir=checkpoint_dir,
        ema_decay=settings.ema_decay,
        accumulation_steps=settings.accumulation_steps,
        loss_fn=loss_fn,
        evaluate_fn=evaluate_fn,
    )
    torch.save(encoder.state_dict(), checkpoint_dir / "sync-final.pt")
    return checkpoint_dir


def main() -> int:
    """CLI entry for `python -m training.train_sync`.

    Returns:
        0 on success.
    """
    train(get_train_settings())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
