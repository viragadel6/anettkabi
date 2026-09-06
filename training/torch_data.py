"""Torch datasets for VAE/vocoder/sync/generator training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from app.ml.audio_codec.mel import MelSpectrogram
from app.ml.registry import VariantSpec

__all__ = [
    "AudioClipDataset",
    "PairedFeatureDataset",
    "load_waveform_clip",
    "silent_waveform",
]


def load_waveform_clip(path: Path, sample_rate: int, length_s: float) -> np.ndarray | None:
    """Load a fixed-length mono float32 waveform.

    Parameters:
        path: Audio file.
        sample_rate: Required rate (resampled via soxr when needed).
        length_s: Target length (shorter files are zero-padded).

    Returns:
        [samples] float32 array or None when unreadable.
    """
    import soxr

    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, OSError):
        return None
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    if rate != sample_rate:
        data = soxr.resample(data[:, 0], rate, sample_rate, quality="HQ")[:, None]
    target = round(length_s * sample_rate)
    if data.shape[0] < target:
        data = np.pad(data, ((0, target - data.shape[0]), (0, 0)))
    return np.ascontiguousarray(data[:target, 0])


def silent_waveform(sample_rate: int, length_s: float) -> np.ndarray:
    """Return a zero waveform used for unreadable entries.

    Parameters:
        sample_rate: Rate in Hz.
        length_s: Length seconds.

    Returns:
        Zero-filled float32 array.
    """
    return np.zeros(round(length_s * sample_rate), dtype=np.float32)


class AudioClipDataset(Dataset):
    """Fixed-length waveform dataset over FSD50K (VAE/vocoder stages)."""

    def __init__(self, settings: Any, spec: VariantSpec, entries: list[dict[str, object]]) -> None:
        """Bind entries and transform config.

        Parameters:
            settings: TrainSettings.
            spec: Variant spec.
            entries: Index entries (clip_id/split or path).
        """
        self._settings = settings
        self._spec = spec
        self._entries = entries
        self._mel = MelSpectrogram(spec)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self._entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one waveform and its mel.

        Parameters:
            index: Entry index.

        Returns:
            Dict with waveform, mel, and prompt tensors/strings.
        """
        entry = self._entries[index]
        path: Path | None
        if "path" in entry:
            path = Path(str(entry["path"]))
        else:
            from training.data.fsd50k import clip_paths

            path = clip_paths(self._settings, str(entry["clip_id"]), str(entry["split"]))
        waveform = (
            load_waveform_clip(path, self._spec.sample_rate, self._settings.clip_length_s)
            if path is not None
            else None
        )
        if waveform is None:
            waveform = silent_waveform(self._spec.sample_rate, self._settings.clip_length_s)
        wave = torch.from_numpy(waveform)
        with torch.no_grad():
            mel = self._mel(wave.unsqueeze(0))[0]
        return {
            "waveform": wave,
            "mel": mel,
            "prompt": str(entry.get("prompt") or ""),
        }


class PairedFeatureDataset(Dataset):
    """Dataset over precomputed feature shards for generator/sync training.

    Shards are produced by `training.data.shards.build_feature_shards`; the
    sidecar `index.json` maps shard files to row counts so the dataset can
    enumerate examples without loading every shard.
    """

    def __init__(self, shard_dir: Path) -> None:
        """Bind a shard directory.

        Parameters:
            shard_dir: Directory of *.npz shards plus index.json.

        Raises:
            FileNotFoundError: When index.json is missing.
        """
        self._shard_dir = shard_dir
        index_path = shard_dir / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"{index_path} missing; run python -m training.data.shards first"
            )
        manifest = json.loads(index_path.read_text(encoding="utf-8"))
        self._shards: list[dict[str, Any]] = manifest["shards"]
        self._lengths: list[int] = [int(shard["rows"]) for shard in self._shards]
        self._total = sum(self._lengths)

    def __len__(self) -> int:
        """Return dataset size."""
        return self._total

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one example.

        Parameters:
            index: Flat index into the concatenated shard rows.

        Returns:
            Dict with mel, sync, visual, prompt tensors/strings.

        Raises:
            IndexError: On out-of-range indices.
        """
        if index < 0 or index >= self._total:
            raise IndexError(index)
        remaining = index
        for shard, length in zip(self._shards, self._lengths, strict=True):
            if remaining < length:
                payload = np.load(self._shard_dir / shard["file"], allow_pickle=False)
                row = remaining
                return {
                    "mel": torch.from_numpy(payload["mel"][row].astype(np.float32)),
                    "visual": torch.from_numpy(payload["visual"][row].astype(np.float32)),
                    "sync": torch.from_numpy(payload["sync"][row].astype(np.float32)),
                    "prompt": str(payload["prompts"][row]),
                }
            remaining -= length
        raise IndexError(index)
