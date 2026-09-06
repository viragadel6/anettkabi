"""FFmpeg/FFprobe subprocess helpers: argument arrays, timeouts, typed errors."""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.errors import ErrorCode, ServiceError

__all__ = ["FfmpegError", "FfmpegResult", "ffmpeg_bin", "ffprobe_bin", "run_ffmpeg", "run_ffprobe"]

_STDERR_TAIL = 4096


class FfmpegError(ServiceError):
    """A failed FFmpeg/FFprobe invocation with captured stderr context."""

    def __init__(self, operation: str, args: list[str], stderr: str, returncode: int) -> None:
        """Capture the failure.

        Parameters:
            operation: Logical operation name (e.g. 'mux').
            args: The argument array passed to the binary.
            stderr: Captured stderr (kept for server-side logging).
            returncode: Process exit code.
        """
        tail = stderr[-_STDERR_TAIL:].strip()
        self.stderr_tail = tail
        self.returncode = returncode
        super().__init__(
            ErrorCode.MUX_FAILED if operation != "probe" else ErrorCode.CORRUPT_MEDIA,
            f"ffmpeg operation '{operation}' failed with exit code {returncode}: {tail[:400]}",
            details={"operation": operation, "args_tail": args[-8:]},
        )


@dataclass(slots=True)
class FfmpegResult:
    """Result of a successful subprocess run.

    Attributes:
        returncode: Exit code (always 0 for success path).
        stdout: Captured stdout text.
        stderr: Captured stderr text.
        elapsed_s: Wall time in seconds.
    """

    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float


def ffmpeg_bin() -> str:
    """Locate the ffmpeg binary.

    Returns:
        Absolute path or name of the ffmpeg executable.

    Raises:
        ServiceError: internal_error when missing.
    """
    path = shutil.which("ffmpeg")
    if not path:
        raise ServiceError(ErrorCode.INTERNAL_ERROR, "ffmpeg binary not found on PATH")
    return path


def ffprobe_bin() -> str:
    """Locate the ffprobe binary.

    Returns:
        Absolute path or name of the ffprobe executable.

    Raises:
        ServiceError: internal_error when missing.
    """
    path = shutil.which("ffprobe")
    if not path:
        raise ServiceError(ErrorCode.INTERNAL_ERROR, "ffprobe binary not found on PATH")
    return path


async def _run(bin_path: str, args: list[str], timeout_s: float, operation: str) -> FfmpegResult:
    """Run one subprocess with timeout and full capture.

    Parameters:
        bin_path: Binary to execute.
        args: Argument array (without the binary).
        timeout_s: Kill threshold.
        operation: Logical name for errors.

    Returns:
        FfmpegResult with captured output.

    Raises:
        FfmpegError: On non-zero exit or timeout.
    """
    loop = asyncio.get_running_loop()
    started = loop.time()
    process = await asyncio.create_subprocess_exec(
        bin_path,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise FfmpegError(operation, args, f"timeout after {timeout_s}s", -9) from None
    elapsed = loop.time() - started
    if process.returncode != 0:
        raise FfmpegError(
            operation,
            args,
            stderr.decode("utf-8", errors="replace"),
            process.returncode or -1,
        )
    return FfmpegResult(
        returncode=process.returncode or 0,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
        elapsed_s=elapsed,
    )


async def run_ffmpeg(
    args: list[str],
    *,
    timeout_s: float = 600.0,
    operation: str = "ffmpeg",
) -> FfmpegResult:
    """Execute ffmpeg with an explicit argument array.

    Parameters:
        args: Arguments following the binary.
        timeout_s: Hard timeout.
        operation: Logical operation name for error mapping.

    Returns:
        FfmpegResult with stdout/stderr.

    Raises:
        FfmpegError: On failure or timeout.
    """
    return await _run(ffmpeg_bin(), args, timeout_s, operation)


async def run_ffprobe(args: list[str], *, timeout_s: float = 60.0) -> str:
    """Execute ffprobe returning raw JSON stdout.

    Parameters:
        args: Arguments following the binary.
        timeout_s: Hard timeout.

    Returns:
        The raw stdout string.

    Raises:
        FfmpegError: On failure (maps to corrupt_media) or timeout.
    """
    full = ["-v", "error", "-print_format", "json", *args]
    result = await _run(ffprobe_bin(), full, timeout_s, "probe")
    return result.stdout


async def ffmpeg_version() -> str:
    """Return the ffmpeg version banner (readiness check).

    Returns:
        First line of `ffmpeg -version`.

    Raises:
        ServiceError: internal_error when ffmpeg cannot run.
    """
    result = await _run(ffmpeg_bin(), ["-version"], 15.0, "version")
    return result.stdout.splitlines()[0] if result.stdout else "unknown"


async def ffprobe_frames(path: Path) -> int:
    """Count decoded frames via ffprobe (fallback duration path).

    Parameters:
        path: Media file.

    Returns:
        Number of video frames reported.

    Raises:
        FfmpegError: When probing fails.
    """
    stdout = await run_ffprobe(
        [
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            str(path),
        ]
    )
    import json

    payload = json.loads(stdout or "{}")
    streams = payload.get("streams") or [{}]
    value = streams[0].get("nb_read_frames")
    return int(value) if value not in (None, "N/A") else 0
