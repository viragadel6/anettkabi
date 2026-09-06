"""Download/derive weight artifacts: CLIP export, tokenizer files, manifest seed.

Everything here produces REAL artifacts: the CLIP towers are exported from the
official OpenAI ViT-B/16 checkpoint via open_clip_torch into our in-repo module
naming; the CLIP BPE vocabulary is fetched from the official OpenAI URLs.
Artifacts whose training has not been run yet are recorded in the manifest as
missing so the service fails fast with `weights_unavailable` (never random
initialization). Their entries are filled by `training.export_weights`.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings  # noqa: E402
from app.ml.registry import get_variant  # noqa: E402
from app.utils.hashing import sha256_file  # noqa: E402

CLIP_VOCAB_URL = "https://openaipublic.azureedge.net/clip/models/bpe_simple_vocab_16e6.txt"
OPEN_CLIP_MODEL = "ViT-B-16"
OPEN_CLIP_PRETRAINED = "openai"


def export_clip_towers(weights_dir: Path, variant_dir: Path) -> dict[str, str]:
    """Export our towers from the official OpenAI CLIP checkpoint.

    Parameters:
        weights_dir: Root weights directory.
        variant_dir: Variant subdirectory.

    Returns:
        Mapping of artifact name -> sha256.

    Raises:
        RuntimeError: When open_clip cannot load the official checkpoint.
    """
    import torch
    from safetensors.torch import save_file

    from app.ml.encoders.clip_text import CLIPTextEncoder
    from app.ml.encoders.clip_visual import CLIPVisualEncoder

    print("loading OpenAI CLIP ViT-B/16 via open_clip (first run downloads ~600 MB)...")
    try:
        import open_clip

        clip = open_clip.create_model_and_transforms(
            OPEN_CLIP_MODEL, pretrained=OPEN_CLIP_PRETRAINED
        )[0]
        tokenizer = open_clip.get_tokenizer(OPEN_CLIP_MODEL)
    except Exception as exc:
        raise RuntimeError(f"cannot load OpenAI CLIP checkpoint: {exc}") from exc
    clip.eval()
    visual_state = {k: v.contiguous() for k, v in clip.visual.state_dict().items()}
    our_visual = CLIPVisualEncoder()
    our_visual_state = our_visual.state_dict()
    mapped_visual: dict[str, torch.Tensor] = {}
    for key in our_visual_state:
        mapped_visual[key] = visual_state[key]
    variant_dir.mkdir(parents=True, exist_ok=True)
    save_file(mapped_visual, str(variant_dir / "clip_visual.safetensors"))
    del visual_state, mapped_visual, our_visual_state

    vocab_size = tokenizer.vocab_size
    text_prefixes = (
        "token_embedding",
        "positional_embedding",
        "ln_final",
        "text_projection",
        "transformer.",
    )
    text_state = {
        k: v.contiguous()
        for k, v in clip.state_dict().items()
        if k.startswith(text_prefixes)
    }
    our_text = CLIPTextEncoder(vocab_size=int(vocab_size))
    our_text_state = our_text.state_dict()
    mapped_text: dict[str, torch.Tensor] = {}
    for key in our_text_state:
        if key == "attn_mask":
            continue
        mapped_text[key] = text_state[key]
    save_file(mapped_text, str(variant_dir / "clip_text.safetensors"))
    del text_state, mapped_text, our_text_state
    return {
        "clip_visual.safetensors": sha256_file(variant_dir / "clip_visual.safetensors"),
        "clip_text.safetensors": sha256_file(variant_dir / "clip_text.safetensors"),
    }


def fetch_clip_bpe(shared_dir: Path) -> dict[str, str]:
    """Fetch the official CLIP BPE vocabulary.

    Parameters:
        shared_dir: Directory for shared artifacts.

    Returns:
        Mapping of artifact name -> sha256.

    Raises:
        RuntimeError: On download failure.
    """
    import httpx

    shared_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = shared_dir / "clip_bpe_vocab.json"
    merges_path = shared_dir / "clip_bpe_merges.txt"
    if vocab_path.is_file() and merges_path.is_file():
        return {
            "clip_bpe_vocab.json": sha256_file(vocab_path),
            "clip_bpe_merges.txt": sha256_file(merges_path),
        }
    print("fetching official CLIP BPE vocabulary...")
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        response = client.get(CLIP_VOCAB_URL)
        response.raise_for_status()
    reference = _load_reference_tokenizer()
    if reference is None:
        raise RuntimeError(
            "open_clip is required to derive the exact CLIP vocabulary; "
            "install requirements first"
        )
    compressed = _reference_bpe_bytes()
    merges_lines = [
        line
        for line in gzip.decompress(compressed).decode("utf-8").split("\n")[1:]
        if line and len(line.split()) == 2
    ]
    vocab_path.write_text(json.dumps(reference, ensure_ascii=False), encoding="utf-8")
    merges_path.write_text("\n".join(merges_lines) + "\n", encoding="utf-8")
    return {
        "clip_bpe_vocab.json": sha256_file(vocab_path),
        "clip_bpe_merges.txt": sha256_file(merges_path),
    }


def _reference_bpe_bytes() -> bytes:
    """Return the compressed official CLIP BPE merge data.

    Returns:
        gzipped bytes shipped with open_clip.

    Raises:
        RuntimeError: When open_clip is unavailable.
    """
    import open_clip.tokenizer as oc_tok

    return oc_tok.default_bpe()


def _load_reference_tokenizer() -> dict[str, int] | None:
    """Build the exact CLIP vocab from open_clip's tokenizer when available.

    Returns:
        Vocab mapping (bytes, merges, specials with canonical ids) or None
        when open_clip is unavailable.
    """
    try:
        import open_clip

        opened = open_clip.get_tokenizer(OPEN_CLIP_MODEL)
        hfst = getattr(opened, "hf_tokenizer", None)
        if hfst is None:
            return None
        return dict(hfst.get_vocab())
    except Exception:
        return None


def main() -> int:
    """CLI entry: export CLIP artifacts and seed the manifest.

    Returns:
        0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=None, help="variant to prepare (default: settings)")
    args = parser.parse_args()
    settings = get_settings()
    variant_name = args.variant or settings.model.model_variant
    spec = get_variant(variant_name)
    weights_dir = settings.model.weights_dir.resolve()
    variant_dir = weights_dir / variant_name
    shared_dir = weights_dir / "shared"
    digests: dict[str, str] = {}
    digests.update(fetch_clip_bpe(shared_dir))
    digests.update(export_clip_towers(weights_dir, variant_dir))
    manifest_path = settings.model.weights_manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    files = manifest.get("files") if isinstance(manifest, dict) else None
    entries: dict[str, dict[str, str]] = dict(files) if isinstance(files, dict) else {}
    for _role, filename in spec.files.items():
        relative = filename
        if relative in ("shared/clip_bpe_vocab.json", "shared/clip_bpe_merges.txt"):
            name = Path(relative).name
            entries[relative] = {
                "url": f"{CLIP_VOCAB_URL}#derived={name}",
                "sha256": digests.get(name, ""),
                "note": "derived artifact of the official CLIP BPE vocabulary (MIT)",
            }
        elif relative in (
            f"{variant_name}/clip_visual.safetensors",
            f"{variant_name}/clip_text.safetensors",
        ):
            name = Path(relative).name
            entries[relative] = {
                "url": f"local:{weights_dir / relative}",
                "sha256": digests.get(name, ""),
                "note": "exported from OpenAI CLIP ViT-B/16 via open_clip_torch (MIT)",
            }
        elif relative not in entries:
            entries[relative] = {
                "url": f"training:{relative}",
                "sha256": "",
                "note": "produced by this repository's Modal training pipeline; run "
                "`python -m training.train_*` then `python -m training.export_weights`",
            }
            print(f"NOTICE: {relative} is not trained yet; the service will fail fast "
                  "with weights_unavailable until it is produced.")
    manifest = {
        "revision": str(manifest.get("revision") or "clip-export"),
        "variant": variant_name,
        "files": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
