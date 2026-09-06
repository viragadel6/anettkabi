"""Torch device selection, dtype policy, autocast contexts, OOM classification."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

from app.config import get_settings
from app.errors import ErrorCode, ServiceError

__all__ = ["DeviceContext", "autocast_ctx", "is_cuda_oom", "resolve_device", "resolve_dtype"]


@dataclass(slots=True)
class DeviceContext:
    """Resolved inference device configuration.

    Attributes:
        device: The torch device to use.
        dtype: Compute dtype for the generator.
        is_cuda: Whether CUDA is active.
        device_index: CUDA ordinal or -1.
    """

    device: torch.device
    dtype: torch.dtype
    is_cuda: bool
    device_index: int


def resolve_device() -> DeviceContext:
    """Resolve the effective torch device and dtype from settings.

    Device strings: auto (cuda when available), cuda, cuda:N, cpu.
    bf16 requires CUDA support; fp16 is CUDA-only; CPU forces fp32.

    Returns:
        DeviceContext with the resolved values.

    Raises:
        ServiceError: internal_error for invalid device strings.
    """
    settings = get_settings()
    requested = settings.model.device
    if requested == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(requested)
    is_cuda = device.type == "cuda"
    if is_cuda and not torch.cuda.is_available():  # pragma: no cover - driver dependent
        device = torch.device("cpu")
        is_cuda = False
    dtype_name = settings.model.dtype
    if not is_cuda:
        dtype = torch.float32
    elif dtype_name == "bf16":
        dtype = torch.bfloat16
    elif dtype_name == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    index = device.index if device.index is not None and is_cuda else (-1 if not is_cuda else 0)
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        fraction = min(1.0, max(0.1, settings.model.max_concurrent_inference * 0.5 + 0.5))
        with contextlib.suppress(RuntimeError, ValueError):
            torch.cuda.set_per_process_memory_fraction(fraction, device=index or 0)
    else:
        torch.set_num_threads(max(1, torch.get_num_threads()))
    return DeviceContext(device=device, dtype=dtype, is_cuda=is_cuda, device_index=index)


def resolve_dtype() -> torch.dtype:
    """Return only the compute dtype.

    Returns:
        The torch dtype for inference.
    """
    return resolve_device().dtype


@contextlib.contextmanager
def autocast_ctx(device_context: DeviceContext):
    """Provide the correct autocast context (no-op on fp32 CPU).

    Parameters:
        device_context: Resolved device context.

    Yields:
        None inside the autocast scope.
    """
    if device_context.is_cuda and device_context.dtype in (torch.bfloat16, torch.float16):
        with torch.autocast(device_type="cuda", dtype=device_context.dtype, enabled=True):
            yield
    else:
        yield


def is_cuda_oom(error: BaseException) -> bool:
    """Classify an exception as CUDA out-of-memory.

    Parameters:
        error: The raised exception.

    Returns:
        True when the error is an OOM condition.
    """
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    text = str(error).lower()
    markers = (
        "out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "cudnn status alloc failed",
        "failed to allocate",
    )
    return any(marker in text for marker in markers)


def assert_cuda_oom_or_reraise(error: BaseException) -> None:
    """Raise gpu_out_of_memory when the error is OOM, else re-raise.

    Parameters:
        error: The exception to classify.

    Raises:
        ServiceError: gpu_out_of_memory for OOM; otherwise the original error.
    """
    if is_cuda_oom(error):
        raise ServiceError(
            ErrorCode.GPU_OUT_OF_MEMORY,
            "GPU ran out of memory during inference; reduce duration or variant size",
        ) from error
    raise error
