"""Filesystem helpers: atomic writes, temp dirs, disk checks, safe extension parsing."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

from app.errors import ErrorCode, ServiceError

__all__ = [
    "assert_disk_headroom",
    "atomic_write_bytes",
    "disk_headroom_bytes",
    "ensure_dir",
    "file_size",
    "remove_quietly",
    "safe_suffix",
    "temp_path",
]


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if missing and return it.

    Parameters:
        path: Directory to create.

    Returns:
        The same path, now guaranteed to exist.

    Raises:
        ServiceError: Mapped to storage_failed when creation fails.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ServiceError(ErrorCode.STORAGE_FAILED, f"cannot create directory {path}") from exc
    return path


@contextlib.contextmanager
def temp_path(work_dir: Path, suffix: str) -> Iterator[Path]:
    """Yield a unique temp file path under work_dir, removed on context exit.

    Parameters:
        work_dir: Scratch directory (created when missing).
        suffix: File suffix including the leading dot.

    Yields:
        The reserved (not yet existing) file path.
    """
    ensure_dir(work_dir)
    handle, name = tempfile.mkstemp(dir=str(work_dir), suffix=suffix)
    os.close(handle)
    path = Path(name)
    try:
        yield path
    finally:
        remove_quietly(path)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes to path atomically via a temp file and rename.

    Parameters:
        path: Destination file.
        payload: Bytes to persist.

    Raises:
        ServiceError: storage_failed when any I/O error occurs.
    """
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    except OSError as exc:
        remove_quietly(tmp)
        raise ServiceError(ErrorCode.STORAGE_FAILED, f"cannot write {path}") from exc


def file_size(path: Path) -> int:
    """Return file size in bytes.

    Parameters:
        path: File to inspect.

    Returns:
        Size in bytes, or 0 when missing.
    """
    try:
        return path.stat().st_size
    except OSError:
        return 0


def disk_headroom_bytes(path: Path) -> int:
    """Return free space under the filesystem containing path.

    Parameters:
        path: Any path on the filesystem to check.

    Returns:
        Free bytes, or 0 on error.
    """
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return 0


def assert_disk_headroom(path: Path, required_bytes: int) -> None:
    """Raise when the filesystem under path lacks headroom.

    Parameters:
        path: Filesystem anchor path.
        required_bytes: Minimum free bytes required.

    Raises:
        ServiceError: storage_failed when headroom is insufficient.
    """
    free = disk_headroom_bytes(path)
    if free < required_bytes:
        raise ServiceError(
            ErrorCode.STORAGE_FAILED,
            f"insufficient disk space under {path}: {free} bytes free, {required_bytes} required",
        )


def safe_suffix(name: str, default: str = ".bin") -> str:
    """Extract a normalized, safe file suffix.

    Parameters:
        name: Original filename or URL basename.
        default: Suffix returned when none is present.

    Returns:
        Lowercase suffix with a leading dot, max 8 chars, or the default.
    """
    suffix = Path(name).suffix.lower()
    if not suffix or len(suffix) > 8 or not suffix.startswith("."):
        return default
    return suffix


def remove_quietly(path: Path) -> None:
    """Delete a file or directory tree, swallowing all errors.

    Parameters:
        path: Path to remove when it exists.
    """
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass
