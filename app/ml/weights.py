"""Weight resolution: manifest verification, downloads, strict module loading, cache."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import structlog
from safetensors.torch import load_file
from torch import nn

from app.config import get_settings
from app.errors import ErrorCode, ServiceError
from app.ml.registry import VariantSpec
from app.utils.hashing import sha256_file

__all__ = [
    "ManifestEntry",
    "WeightsManifest",
    "clear_module_cache",
    "ensure_weight_file",
    "load_manifest",
    "load_module_weights",
    "load_tensor_dict",
    "module_cache_get",
    "module_cache_put",
]

_logger = structlog.get_logger("vsfx.weights")
_lock = threading.Lock()
_modules: dict[str, nn.Module] = {}
_tensors: dict[str, dict[str, Any]] = {}
_file_locks: dict[str, threading.Lock] = {}


class ManifestEntry:
    """One weight file record inside the manifest.

    Attributes:
        url: Source (`https://...`, `hf://repo/file`, or `local:relative/path`).
        sha256: Expected digest (None = trust-on-first-use).
        size: Optional byte size.
        note: Provenance/licence note.
    """

    __slots__ = ("note", "sha256", "size", "url")

    def __init__(self, url: str, sha256: str | None, size: int | None, note: str) -> None:
        """Store entry fields.

        Parameters:
            url: Source URL.
            sha256: Expected digest or None.
            size: Expected size or None.
            note: Provenance note.
        """
        self.url = url
        self.sha256 = sha256
        self.size = size
        self.note = note

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ManifestEntry:
        """Parse an entry from manifest JSON.

        Parameters:
            payload: Raw dict.

        Returns:
            The ManifestEntry.

        Raises:
            ValueError: On missing url.
        """
        url = payload.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("manifest entry missing 'url'")
        digest = payload.get("sha256")
        size = payload.get("size")
        return cls(
            url=url,
            sha256=digest if isinstance(digest, str) and digest else None,
            size=int(size) if isinstance(size, int) else None,
            note=str(payload.get("note", "")),
        )


class WeightsManifest:
    """Parsed manifest with lookup helpers."""

    __slots__ = ("files", "revision", "variant")

    def __init__(self, revision: str, variant: str, files: dict[str, ManifestEntry]) -> None:
        """Store manifest fields.

        Parameters:
            revision: Weight revision hash.
            variant: Variant the manifest was exported for.
            files: filename -> entry.
        """
        self.revision = revision
        self.variant = variant
        self.files = files

    def entry_for(self, filename: str) -> ManifestEntry:
        """Fetch the manifest entry for a file.

        Parameters:
            filename: Relative filename.

        Returns:
            The entry.

        Raises:
            ServiceError: weights_unavailable when the file is absent.
        """
        entry = self.files.get(filename)
        if entry is None:
            raise ServiceError(
                ErrorCode.WEIGHTS_UNAVAILABLE,
                f"weight artifact {filename!r} is not present in the manifest; "
                "run scripts/download_weights.py or a training export",
            )
        return entry


def load_manifest(path: Path) -> WeightsManifest:
    """Parse and validate the weights manifest file.

    Parameters:
        path: Manifest JSON path.

    Returns:
        WeightsManifest.

    Raises:
        ServiceError: weights_unavailable for unreadable/invalid manifests.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ServiceError(
            ErrorCode.WEIGHTS_UNAVAILABLE,
            f"weights manifest not found at {path}; run scripts/download_weights.py",
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ServiceError(
            ErrorCode.WEIGHTS_UNAVAILABLE,
            f"weights manifest at {path} is unreadable: {exc}",
        ) from exc
    files: dict[str, ManifestEntry] = {}
    raw_files = payload.get("files")
    if not isinstance(raw_files, dict):
        raise ServiceError(
            ErrorCode.WEIGHTS_UNAVAILABLE, "weights manifest is missing the 'files' object"
        )
    for filename, raw in raw_files.items():
        if not isinstance(raw, dict):
            continue
        try:
            files[str(filename)] = ManifestEntry.from_json(raw)
        except ValueError as exc:
            raise ServiceError(
                ErrorCode.WEIGHTS_UNAVAILABLE,
                f"invalid manifest entry for {filename}: {exc}",
            ) from exc
    return WeightsManifest(
        revision=str(payload.get("revision", "unknown")),
        variant=str(payload.get("variant", "unknown")),
        files=files,
    )


def _download(url: str, destination: Path) -> None:
    """Fetch one artifact honoring its URL scheme.

    Parameters:
        url: https://, hf://repo/file, or local: path.
        destination: Target file.

    Raises:
        ServiceError: weights_unavailable on failure.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".part")
    try:
        if url.startswith("local:"):
            source = Path(url[len("local:") :])
            if not source.is_file():
                raise FileNotFoundError(source)
            destination.write_bytes(source.read_bytes())
            return
        if url.startswith("hf://"):
            import shutil

            from huggingface_hub import hf_hub_download

            repo_id, _, filename = url[len("hf://") :].partition("/")
            resolved = hf_hub_download(repo_id=repo_id, filename=filename)
            shutil.copyfile(resolved, destination)
            return
        import httpx

        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(600.0)) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in response.iter_bytes(1024 * 512):
                        handle.write(chunk)
        tmp.replace(destination)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise ServiceError(
            ErrorCode.WEIGHTS_UNAVAILABLE,
            f"failed to fetch weight artifact {destination.name} from {url}: {exc}",
        ) from exc


def _lock_for(filename: str) -> threading.Lock:
    """Return (creating if needed) the per-file download lock.

    Parameters:
        filename: Relative filename.

    Returns:
        A threading.Lock for that file.
    """
    with _lock:
        return _file_locks.setdefault(filename, threading.Lock())


def ensure_weight_file(spec: VariantSpec, role: str) -> Path:
    """Ensure a role's artifact exists, verified, downloading when permitted.

    Parameters:
        spec: Variant spec.
        role: Weight role name.

    Returns:
        Absolute path of the verified file.

    Raises:
        ServiceError: weights_unavailable for missing/invalid artifacts.
    """
    settings = get_settings()
    filename = spec.files[role]
    path = (settings.model.weights_dir / filename).resolve()
    manifest = load_manifest(settings.model.weights_manifest_path)
    entry = manifest.entry_for(filename)
    with _lock_for(filename):
        if path.is_file() and path.stat().st_size > 0:
            if entry.sha256:
                digest = sha256_file(path)
                if digest != entry.sha256:
                    raise ServiceError(
                        ErrorCode.WEIGHTS_UNAVAILABLE,
                        f"checksum mismatch for {filename}: expected {entry.sha256}, got {digest}",
                    )
            return path
        _download(entry.url, path)
        digest = sha256_file(path)
        if entry.sha256 and digest != entry.sha256:
            path.unlink(missing_ok=True)
            raise ServiceError(
                ErrorCode.WEIGHTS_UNAVAILABLE,
                f"downloaded artifact {filename} failed checksum verification",
            )
        if not entry.sha256:
            _logger.warning(
                "weights_trust_on_first_use",
                file=filename,
                sha256=digest,
                hint="pin this digest in the manifest",
            )
        return path


def load_tensor_dict(spec: VariantSpec, role: str) -> dict[str, Any]:
    """Load and cache a raw tensor dict from a safetensors artifact.

    Parameters:
        spec: Variant spec.
        role: Weight role.

    Returns:
        The state dict.

    Raises:
        ServiceError: weights_unavailable on load failure.
    """
    cache_key = f"{spec.name}:{role}"
    if cache_key in _tensors:
        return _tensors[cache_key]
    path = ensure_weight_file(spec, role)
    try:
        tensors = load_file(str(path))
    except Exception as exc:
        raise ServiceError(
            ErrorCode.WEIGHTS_UNAVAILABLE,
            f"cannot load safetensors artifact {path.name}: {exc}",
        ) from exc
    _tensors[cache_key] = tensors
    return tensors


def load_module_weights(
    module: nn.Module,
    spec: VariantSpec,
    role: str,
    *,
    prefix_map: tuple[tuple[str, str], ...] = (),
) -> nn.Module:
    """Strictly load a role's weights into a module with a key report.

    Parameters:
        module: Target module.
        spec: Variant spec.
        role: Weight role.
        prefix_map: Optional (ckpt_prefix, model_prefix) rewrites.

    Returns:
        The module, now loaded (and cached).

    Raises:
        ServiceError: weights_unavailable on any key mismatch, with the
            explicit missing/unexpected key lists in the message.
    """
    cache_key = f"{spec.name}:{role}"
    raw = load_tensor_dict(spec, role)
    target = module.state_dict()
    remapped: dict[str, Any] = {}
    for key, value in raw.items():
        new_key = key
        for ckpt_prefix, model_prefix in prefix_map:
            if new_key.startswith(ckpt_prefix):
                new_key = model_prefix + new_key[len(ckpt_prefix) :]
                break
        remapped[new_key] = value
    expected = set(target)
    provided = set(remapped)
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    if missing or unexpected:
        raise ServiceError(
            ErrorCode.WEIGHTS_UNAVAILABLE,
            f"strict load failed for {role}: missing keys {missing[:8]} "
            f"({len(missing)} total), unexpected keys {unexpected[:8]} "
            f"({len(unexpected)} total)",
        )
    for key in target:
        if tuple(remapped[key].shape) != tuple(target[key].shape):
            raise ServiceError(
                ErrorCode.WEIGHTS_UNAVAILABLE,
                f"shape mismatch for {role} key {key}: checkpoint "
                f"{tuple(remapped[key].shape)} vs model {tuple(target[key].shape)}",
            )
    module.load_state_dict(remapped, strict=True)
    module_cache_put(cache_key, module)
    return module


def module_cache_put(key: str, module: nn.Module) -> None:
    """Insert a loaded module into the process cache.

    Parameters:
        key: Cache key.
        module: The loaded module.
    """
    with _lock:
        _modules[key] = module


def module_cache_get(key: str) -> nn.Module | None:
    """Fetch a cached module.

    Parameters:
        key: Cache key.

    Returns:
        The module or None.
    """
    with _lock:
        return _modules.get(key)


def clear_module_cache() -> None:
    """Drop all cached modules and tensor dicts (tests)."""
    with _lock:
        _modules.clear()
        _tensors.clear()
        _file_locks.clear()
