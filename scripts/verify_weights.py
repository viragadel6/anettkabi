"""Verify all required weight artifacts against the manifest checksums."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings  # noqa: E402
from app.ml.registry import get_variant  # noqa: E402
from app.utils.hashing import sha256_file  # noqa: E402


def main() -> int:
    """CLI entry verifying presence + checksums of every variant artifact.

    Returns:
        0 when everything verifies; 2 when artifacts are missing/invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=None)
    args = parser.parse_args()
    settings = get_settings()
    variant_name = args.variant or settings.model.model_variant
    spec = get_variant(variant_name)
    weights_dir = settings.model.weights_dir.resolve()
    manifest_path = settings.model.weights_manifest_path.resolve()
    if not manifest_path.is_file():
        print(f"FAIL: manifest missing at {manifest_path}")
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files", {})
    failures: list[str] = []
    for _role, relative in spec.files.items():
        path = weights_dir / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        entry = entries.get(relative, {})
        expected = entry.get("sha256") or ""
        if expected:
            actual = sha256_file(path)
            if actual != expected:
                failures.append(f"checksum mismatch: {relative} (expected {expected}, got {actual})")
        else:
            print(f"WARN: {relative} has no pinned sha256 in the manifest")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 2
    print(f"OK: all {len(spec.files)} artifacts for variant {variant_name} verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
