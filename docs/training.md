# Training Pipeline

Everything trains from scratch in this repository, on Modal GPUs
(`modal/sfx_training.py`) or locally on any CUDA box.

## Stages (strict order)

| # | Stage | Module | Data | Depends on |
|---|---|---|---|---|
| 1 | `weights` | `scripts/download_weights.py` | OpenAI CLIP ViT-B/16 | — |
| 2 | `fsd50k` | `training/data/fsd50k.py` | Zenodo FSD50K | — |
| 3 | `vggsound` | `training/data/vggsound.py` | VGGSound CSV + yt-dlp | — |
| 4 | `vae` | `training/train_vae.py` | FSD50K dev | 1, 2 |
| 5 | `vocoder` | `training/train_vocoder.py` | FSD50K dev | 1, 2 |
| 6 | `sync` | `training/train_sync.py` | VGGSound audio → envelope grids | 1, 3 |
| 7 | `shards` | `training/data/shards.py` | VGGSound audio (+ trained sync enc in service later) | 1, 3, 6* |
| 8 | `generator` | `training/train_gen.py` | feature shards | 1, 4, 7 |
| 9 | `export` | `training/export_weights.py` | run checkpoints | 4, 5, 6, 8 |

*Shards encode sync features with the freshly trained sync encoder when its
checkpoint exists (preferred); stage 6 therefore precedes 7.

## Modal (recommended)

```console
pip install modal
modal setup
modal run modal/sfx_training.py                       # everything
modal run modal/sfx_training.py --stages vae,vocoder  # subset
VSFX_MODAL_GPU=a100-40g modal run modal/sfx_training.py --epochs-override 3
VSFX_MODAL_VARIANT=medium_44k modal run modal/sfx_training.py
```

State lives in three Modal Volumes (`vsfx-training-datasets|-runs|-weights`);
every stage commits them, so runs resume from `{stage}-last.pt` checkpoints.
The stage list and GPU string are printed before execution; `stage_plan`
returns the plan as JSON for CI orchestration.

## Local

```console
make train-data          # FSD50K + VGGSound download & index
make train-vae train-vocoder train-sync
python -m training.data.shards
make train-generator
make export-weights VARIANT=small_16k
make weights             # verify service-visible artifacts
```

Datasets land under `datasets/` (`fsd50k/{dev,eval}`, `vggsound/wav`,
`shards/<variant>/shard-*.npz + index.json`); checkpoints under
`runs/<variant>/<stage>/`.

## Datasets (verified against upstream, September 2026)

* **FSD50K** (CC-BY-4.0, Zenodo record **4060432** — dev *and* eval live on
  the same record; the often-cited 4273845 is an unrelated paper):
  * dev audio `FSD50K.dev_audio.z01..z05` + `.zip` (~18.4 GB, 40,966 clips),
  * eval audio `FSD50K.eval_audio.z01` + `.zip` (~6.3 GB, 10,231 clips),
  * ground truth `FSD50K.ground_truth.zip` → `dev.csv`/`eval.csv` with
    `fname,labels,mids[,split]` columns (51,197 rows total).
  The downloader enumerates the archive list from the Zenodo record API at
  runtime (hardcoded fallbacks encode the layout above); split zips are
  merged with `zip -s 0` + `unzip` (or 7z); the CSVs become `index.json`
  (`clip_id`, `split`, `labels`, `prompt`, plus `subsplit` train/valid from
  the dev CSV). Use `--dev-only` to skip the 6.3 GB eval archive.
* **VGGSound** (CC-BY-4.0 csv, HF `Loie/VGGSound`): header-less
  `youtube_id,start_s,label,split` (199,666 rows). The label filter uses
  **word-prefix** matching over SFX-rich hints (car/engine/guitar/drum/rain/
  wind/thunder/dog/door/hammer/chainsaw/typing/footstep/clapping), which
  keeps `chainsawing trees` while never matching the `train`/`test` split
  column — **27,724 candidate clips** on the current CSV. Fetches are 10 s
  `yt-dlp --download-sections` segments trimmed to 48 kHz stereo WAV;
  seeded deterministic shuffle; `vggsound_max_clips` caps the crawl.
* Shards (`training/data/shards.py`): 256-row npz shards of
  mel / CLIP-visual / sync features + prompts, described in
  `docs/model.md` (conditioning honesty note).

## Hyperparameters

All under `VSFX_TRAIN_*` (`training/config.py`): per-stage batch/lr/epochs,
`clip_length_s=10`, `val_fraction=0.02`, `ema_decay=0.999`,
`accumulation_steps`, `seed=4242`. Cosine LR with linear warmup
(`training/engine.py`); grad-norm clip 1.0; best checkpoints tracked by
validation loss; history written to `{stage}-history.json`.

## Export & verification

`training/export_weights.py` copies the best checkpoint of each trained role
into `WEIGHTS_DIR/<variant>/*.safetensors`, derives true unconditional
`null_embeds` with the frozen towers, rewrites `weights/manifest.json` with
fresh SHA-256 digests and a `trained-<variant>-<hash>` revision, then
`scripts/verify_weights.py` fails on any mismatch. The service hot-verifies
the manifest on pipeline build and refuses to serve `weights_unavailable`
otherwise.
