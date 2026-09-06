"""Export trained checkpoints into the service WEIGHTS_DIR layout with a manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from app.config import get_settings
from app.ml.registry import VariantSpec
from app.utils.hashing import sha256_file
from training.config import TrainSettings, get_train_settings

__all__ = ["export_variant"]


def _pick(directory: Path, *names: str) -> Path:
    """Return the first existing checkpoint among candidate names.

    Parameters:
        directory: Stage checkpoint directory.
        names: Candidate filenames in priority order.

    Returns:
        The found checkpoint path.

    Raises:
        FileNotFoundError: When none exist.
    """
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"no checkpoint in {directory} (looked for {', '.join(names)})")


def _tensors_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Load a checkpoint's raw state dict.

    Parameters:
        path: .pt checkpoint path.

    Returns:
        state_dict of tensors.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]
    return payload


def _build_null_embeds(weights_dir: Path, spec: VariantSpec) -> dict[str, torch.Tensor]:
    """Encode the true unconditional conditioning with the frozen towers.

    Parameters:
        weights_dir: Exported weights root.
        spec: Variant spec.

    Returns:
        Dict with text_tokens [1, 77, D], pooled_text [1, D], pooled_visual [1, Dv].

    Raises:
        FileNotFoundError: When CLIP exports are missing.
    """
    from app.ml.encoders.clip_text import CLIPTextEncoder
    from app.ml.encoders.clip_visual import CLIPVisualEncoder
    from app.ml.encoders.tokenizer import CliPTokenizer

    vocab = weights_dir / "shared" / "clip_bpe_vocab.json"
    merges = weights_dir / "shared" / "clip_bpe_merges.txt"
    text_path = weights_dir / spec.files["clip_text"]
    visual_path = weights_dir / spec.files["clip_visual"]
    for path in (vocab, merges, text_path, visual_path):
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; run scripts/download_weights.py first")
    tokenizer = CliPTokenizer(vocab, merges)
    text_encoder = CLIPTextEncoder(vocab_size=tokenizer.vocab_size)
    text_encoder.load_state_dict(load_file(str(text_path)))
    text_encoder.eval()
    visual_encoder = CLIPVisualEncoder()
    visual_encoder.load_state_dict(load_file(str(visual_path)))
    visual_encoder.eval()
    tokens = tokenizer.encode_batch([""])
    with torch.no_grad():
        text_tokens, pooled_text = text_encoder.encode_tokens(tokens)
        gray = torch.full((1, 3, 224, 224), 118, dtype=torch.uint8)
        visual_tokens = visual_encoder.encode_frames(gray, torch.device("cpu"))
        pooled_visual = visual_tokens.mean(dim=0, keepdim=True)
    return {
        "text_tokens": text_tokens.contiguous(),
        "pooled_text": pooled_text.contiguous(),
        "pooled_visual": pooled_visual.contiguous(),
    }


def export_variant(settings: TrainSettings) -> Path:
    """Copy trained artifacts into WEIGHTS_DIR and rewrite the manifest.

    Parameters:
        settings: Training settings (variant, output_dir, weights_dir).

    Returns:
        The weights directory.

    Raises:
        FileNotFoundError: When a stage has not been trained.
    """
    spec = settings.spec
    variant = settings.variant
    weights_dir = settings.weights_dir
    run_root = settings.output_dir / variant
    sources = {
        "sync_encoder": _pick(run_root / "sync", "sync-final.pt", "sync-best.pt"),
        "vae": _pick(run_root / "vae", "vae-final.pt", "vae-best.pt"),
        "vocoder": _pick(run_root / "vocoder", "vocoder-final.pt", "vocoder-best.pt"),
        "generator": _pick(run_root / "generator", "generator-final.pt", "generator-best.pt"),
    }
    variant_dir = weights_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "shared").mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for role, checkpoint in sources.items():
        target = weights_dir / spec.files[role]
        target.parent.mkdir(parents=True, exist_ok=True)
        tensors = {key: value.contiguous() for key, value in _tensors_from_checkpoint(checkpoint).items()}
        if not tensors:
            raise FileNotFoundError(f"checkpoint {checkpoint} contains no tensors")
        save_file(tensors, str(target))
        saved[role] = target
    null_target = weights_dir / spec.files["null_embeds"]
    save_file(_build_null_embeds(weights_dir, spec), str(null_target))
    saved["null_embeds"] = null_target
    _rewrite_manifest(weights_dir, variant, spec, saved, run_root)
    return weights_dir


def _rewrite_manifest(
    weights_dir: Path, variant: str, spec: VariantSpec, saved: dict[str, Path], run_root: Path
) -> None:
    """Update manifest entries for freshly exported artifacts.

    Parameters:
        weights_dir: Weights root.
        variant: Variant name.
        spec: Variant spec.
        saved: role -> exported path.
        run_root: Training run root (recorded as provenance).
    """
    manifest_path = weights_dir / "manifest.json"
    try:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        manifest = {}
    entries = manifest.get("files") if isinstance(manifest, dict) else None
    files: dict[str, Any] = dict(entries) if isinstance(entries, dict) else {}
    for role, path in saved.items():
        relative = path.relative_to(weights_dir).as_posix()
        files[relative] = {
            "url": f"local:{path.resolve()}",
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "note": f"trained in-repo ({run_root.resolve()}); role {role}",
        }
    digest = hashlib.sha256()
    for relative in sorted(files):
        entry = files[relative]
        digest.update(relative.encode("utf-8"))
        digest.update(str(entry.get("sha256", "")).encode("utf-8"))
    manifest = {
        "revision": f"trained-{variant}-{digest.hexdigest()[:12]}",
        "variant": variant,
        "files": files,
    }
    del spec
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest rewritten: {manifest_path} revision {manifest['revision']}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry for `python -m training.export_weights`.

    Parameters:
        argv: Optional argument vector (defaults to sys.argv[1:]).

    Returns:
        0 on success, 2 when prerequisites are missing or args are invalid.
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(description="export trained weights for the service")
    parser.add_argument("--variant", default=None, help="registry variant (default: $VSFX_TRAIN_VARIANT)")
    parser.add_argument("--output-dir", default=None, help="training run root (default: $VSFX_TRAIN_OUTPUT_DIR)")
    parser.add_argument("--weights-dir", default=None, help="export target (default: $VSFX_TRAIN_WEIGHTS_DIR)")
    args = parser.parse_args(argv)
    if args.variant:
        os.environ["VSFX_TRAIN_VARIANT"] = args.variant
    if args.output_dir:
        os.environ["VSFX_TRAIN_OUTPUT_DIR"] = args.output_dir
    if args.weights_dir:
        os.environ["VSFX_TRAIN_WEIGHTS_DIR"] = args.weights_dir
    try:
        settings = get_train_settings()
    except ValueError as exc:
        print(f"export failed: {exc}")
        return 2
    try:
        service_weights = Path(get_settings().weights_dir)
    except Exception:
        service_weights = settings.weights_dir
    if service_weights.resolve() != settings.weights_dir.resolve():
        print(
            f"warning: TrainSettings.weights_dir ({settings.weights_dir}) differs from "
            f"service weights_dir ({service_weights}); exporting to the training path"
        )
    try:
        export_variant(settings)
    except FileNotFoundError as exc:
        print(f"export failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
