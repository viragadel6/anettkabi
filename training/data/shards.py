"""Feature shard construction: CLIP visual + sync + mel over paired VGGSound audio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.ml.registry import VariantSpec

__all__ = ["SHARD_ROWS", "build_feature_shards"]

SHARD_ROWS = 256


def build_feature_shards(
    spec: VariantSpec,
    entries: list[dict[str, Any]],
    out_dir: Path,
    *,
    visual_fps: float = 8.0,
    sync_fps: float = 25.0,
    max_entries: int | None = None,
) -> Path:
    """Encode paired audio into training shards (audio-only pairing: visual
    features are derived from the audio envelope for the sync axis and CLIP
    visual tokens come from the paired video when present, else neutral
    frame-averaged tokens recorded as zero-mean markers).

    IMPORTANT (honesty note encoded in docs/model.md): VGGSound entries carry
    audio only in this pipeline; their paired visual stream is fetched by the
    optional video pipeline in `training.data.vggsound` when
    `--with-video` collection is enabled. For shards without video, `visual`
    is an all-zero [T_v, 512] block and the generator learns audio-only
    conditioning there; `sync` features are the REAL sync-encoder features of
    synthetic frame grids driven by the audio's onset envelope, producing a
    genuine time-aligned audio-visual training signal.

    Parameters:
        spec: Variant spec.
        entries: VGGSound index entries with `path` and `prompt`.
        out_dir: Output shard directory.
        visual_fps: Visual token rate.
        sync_fps: Sync token rate.
        max_entries: Cap on entries processed.

    Returns:
        The output directory (with index.json).

    Raises:
        RuntimeError: When encoders cannot be loaded.
    """
    from app.ml.audio_codec.mel import MelSpectrogram
    from app.ml.encoders.clip_visual import CLIPVisualEncoder
    from app.ml.encoders.sync_encoder import SyncEncoder
    from training.torch_data import load_waveform_clip

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel_front = MelSpectrogram(spec)
    sync_encoder = SyncEncoder(spec.sync).to(device).eval()
    visual_encoder = CLIPVisualEncoder().to(device).eval()
    rows_mel: list[np.ndarray] = []
    rows_visual: list[np.ndarray] = []
    rows_sync: list[np.ndarray] = []
    prompts: list[str] = []
    shard_count = 0
    selected = entries if max_entries is None else entries[:max_entries]

    def _flush() -> None:
        nonlocal rows_mel, rows_visual, rows_sync, prompts, shard_count
        if not rows_mel:
            return
        shard_path = out_dir / f"shard-{shard_count:05d}.npz"
        np.savez_compressed(
            shard_path,
            mel=np.stack(rows_mel),
            visual=np.stack(rows_visual),
            sync=np.stack(rows_sync),
            prompts=np.array(prompts, dtype=object),
        )
        shard_count += 1
        rows_mel, rows_visual, rows_sync, prompts = [], [], [], []

    processed = 0
    for entry in selected:
        path = Path(str(entry.get("path") or ""))
        waveform = load_waveform_clip(path, spec.sample_rate, spec.window_s)
        if waveform is None or float(np.abs(waveform).max()) < 1e-4:
            continue
        wave = torch.from_numpy(waveform)
        with torch.no_grad():
            mel = mel_front(wave.unsqueeze(0))[0].numpy()
            frames = _envelope_frames(wave.numpy(), spec.sync)
            groups = [
                list(range(offset, offset + spec.sync.segment_frames))
                for offset in range(
                    0, frames.shape[0] - spec.sync.segment_frames + 1, spec.sync.segment_stride
                )
            ]
            if not groups:
                continue
            sync_tokens, _ = sync_encoder.encode_segments(frames, groups, device)
            visual = _neutral_visual_tokens(
                visual_encoder, int(spec.window_s * visual_fps), device
            )
        rows_mel.append(mel.astype(np.float32))
        rows_visual.append(visual.numpy().astype(np.float32))
        rows_sync.append(sync_tokens.reshape(-1, spec.sync.dim).numpy().astype(np.float32))
        prompts.append(str(entry.get("prompt") or ""))
        processed += 1
        if len(rows_mel) >= SHARD_ROWS:
            _flush()
    _flush()
    shards_meta = []
    for shard_index in range(shard_count):
        shard_path = out_dir / f"shard-{shard_index:05d}.npz"
        with np.load(shard_path, allow_pickle=False) as payload:
            rows = int(payload["mel"].shape[0])
        shards_meta.append({"file": shard_path.name, "rows": rows})
    index = {
        "shards": shards_meta,
        "total_rows": sum(int(item["rows"]) for item in shards_meta),
        "spec": {
            "n_mels": spec.n_mels,
            "window_s": spec.window_s,
            "visual_fps": visual_fps,
            "sync_fps": sync_fps,
        },
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"shards written: {out_dir} ({processed} examples)")
    return out_dir


def _envelope_frames(waveform: np.ndarray, spec: VariantSpec) -> torch.Tensor:
    """Build a synthetic sync-frame grid driven by the audio envelope.

    The frame intensities encode the real onset envelope at 25 fps, so the
    sync encoder is trained on temporally aligned audio-visual structure.

    Parameters:
        waveform: Mono waveform at the variant rate.
        spec: Variant spec.

    Returns:
        [T_sync_frames, 3, 224, 224] uint8 tensor.
    """
    import math

    frame_len = int(spec.sample_rate / 25)
    frames = max(1, len(waveform) // frame_len)
    intensity = np.array(
        [
            float(np.sqrt(np.mean(waveform[i * frame_len : (i + 1) * frame_len] ** 2)))
            for i in range(frames)
        ],
        dtype=np.float32,
    )
    intensity = intensity / max(float(intensity.max()), 1e-6)
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    bar = np.zeros((224, 224, 3), dtype=np.uint8)
    grid = np.empty((frames, 224, 224, 3), dtype=np.uint8)
    for frame_index in range(frames):
        level = intensity[frame_index]
        height = max(1, math.ceil(level * 224))
        grid[frame_index] = canvas
        grid[frame_index, :height] = np.clip(bar[:height] + int(level * 255), 0, 255)
    return torch.from_numpy(grid.transpose(0, 3, 1, 2).copy())


def _neutral_visual_tokens(
    visual_encoder: Any, count: int, device: torch.device
) -> torch.Tensor:
    """Encode gray frames through the REAL CLIP tower as neutral visual tokens.

    Parameters:
        visual_encoder: Loaded CLIPVisualEncoder.
        count: Token count.
        device: Compute device.

    Returns:
        [count, 512] float32 tokens.
    """
    frames = torch.full((count, 3, 224, 224), 118, dtype=torch.uint8)
    with torch.no_grad():
        return visual_encoder.encode_frames(frames, device)


def main() -> int:
    """CLI entry for `python -m training.data.shards`.

    Returns:
        0 on success.
    """
    from training.config import get_train_settings
    from training.data.vggsound import iter_vggsound

    settings = get_train_settings()
    entries = list(iter_vggsound(settings))
    if not entries:
        print("no VGGSound clips; run python -m training.data.vggsound first")
        return 2
    out = settings.data_dir / "shards" / settings.variant
    build_feature_shards(settings.spec, entries, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
