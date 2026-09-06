# Model Card — `video-to-video-sfx`

## Overview

The generator is implemented **entirely in this repository, from scratch**
(`app/ml/`): no third-party diffusion checkpoint is wrapped or called at
inference time. The only external weights are the two OpenAI **CLIP ViT-B/16**
towers (MIT licence), exported via `open_clip` into our own module layout by
`scripts/download_weights.py` and pinned by SHA-256 in
`weights/manifest.json`.

Components:

| Role | Module | Notes |
|---|---|---|
| Mel front-end | `app/ml/audio_codec/mel.py` | log-mel, trainable-free STFT basis |
| Audio VAE | `app/ml/audio_codec/vae.py` | 2× per-stage temporal downsampling, `latent_hop` total, learned scale buffers |
| Vocoder | `app/ml/audio_codec/vocoder.py` | BigVGAN-style: transposed-conv upsampling ×∏, multi-period GAN training, weight-norm removed at inference |
| Sync encoder | `app/ml/encoders/sync_encoder.py` | Conv3d patch stem + temporal transformer over 16-frame segments (stride 8 → 25 token/s) |
| Text conditioning | `app/ml/encoders/clip_text.py` + `tokenizer.py` | our CLIP text tower, byte-level BPE (SOT 49406 / EOT 49407, ctx 77) |
| Visual conditioning | `app/ml/encoders/clip_visual.py` | our CLIP visual tower on 8 fps frames |
| Generator | `app/ml/generator/mmdit.py` | MMDiT: joint two-stream blocks (audio ↔ [text|visual|sync|memory]) then single-stream audio blocks, adaLN-Zero, RoPE time tables per stream |
| Flow matching | `app/ml/generator/flow_matching.py`, `sampler.py` | rectified flow `x_t=(1−t)x₀+t·x₁`, target `v=x₁−x₀`; euler/midpoint; φ-rescale 0.7; CFG `s=0` single-pass |
| Sliding windows | `app/ml/windowing.py` | equal-power crossfades, memory tokens = previous window's overlap latents |
| Post chain | `app/media/*` | DC removal → loudness (LUFS) → true-peak limit → mix/duck → FFmpeg mux (±10 ms, faststart) |

Modality ids: audio 0, text 1, visual 2, sync 3, memory 4.

## Variants

| variant | sr | n_mels | MMDiT | latent_fps | window | overlap |
|---|---|---|---|---|---|---|
| `small_16k` | 16 kHz | 64 | d512, h8, 8 joint + 4 single | 25 | 10 s | 1 s |
| `medium_44k` | 44.1 kHz | 128 | d768, h12, 16 + 6 | 25 | 10 s | 1 s |
| `large_44k` | 44.1 kHz | 128 | d1024, h16, 24 + 8 | 25 | 10 s | 1 s |

## Window math

For duration `D`, window `W=10 s`, overlap `O=1 s`: stride `S=W−O`; windows
start at `0, S, 2S, …` with final window truncated at `D`. Each interior
boundary owns half the overlap on each side (`crop_start = start + O/2`,
`crop_end = end − O/2`). Waveforms are blended with raised-cosine equal-power
crossfades so constant-amplitude signals stay constant through the overlap.
Each window after the first receives the previous window's overlap-tail
latents as **memory tokens**, giving continuity without leaking future
context.

## Conditioning honesty note

Training pairs come from VGGSound audio (see `docs/training.md`). Where the
paired video pixels are unavailable, the shard builder records:

* `visual`: **real** CLIP tokens of neutral gray frames (true tower output,
  honestly neutral conditioning — the generator learns audio-only behavior
  there, never fabricated content correspondence);
* `sync`: **real** SyncEncoder features over deterministic 25 fps intensity
  grids driven by the clip's actual RMS/onset envelope — a genuinely
  time-aligned audio↔visual training signal.

At inference the towers always consume the input video's real frames.

## Reproducibility

* All sampling is seeded: `seed` (or server-generated when `-1`) drives a
  per-prediction generator; identical `(video_sha256, prompt, params, seed)`
  ⇒ identical latents.
* Weights are loaded strictly (`weights_unavailable` on any mismatch) and
  cached per process; `revision` from the manifest is attached to metrics.
* Sliding-window decode is chunked + crossfaded; DC removal is applied before
  loudness normalization; mux is `-movflags +faststart`.

## Weights artifacts (`WEIGHTS_DIR`)

```
weights/
  manifest.json                     revision, variant, per-file sha256
  shared/clip_bpe_vocab.json(.gz)   derived from the official CLIP BPE
  shared/clip_bpe_merges.txt
  <variant>/clip_visual.safetensors
  <variant>/clip_text.safetensors
  <variant>/sync_encoder.safetensors   ← trained here
  <variant>/vae.safetensors            ← trained here
  <variant>/vocoder.safetensors        ← trained here
  <variant>/generator.safetensors      ← trained here
  <variant>/null_embeds.safetensors    ← true unconditional embeds
```

`make weights` verifies everything present (`scripts/verify_weights.py`
exits non-zero otherwise). Until the trained roles exist, inference fails
fast with `weights_unavailable` — the service never fabricates audio.

## Metrics

`vsfx_inference_seconds`, `vsfx_windows_total`, `vsfx_vae_decode_seconds`,
`vsfx_vocoder_seconds`, `vsfx_mux_seconds`, `vsfx_gpu_memory_bytes`,
`vsfx_predictions_total{status}`, `vsfx_pipeline_builds_total` — see
`docs/operations.md`.
