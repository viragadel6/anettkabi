"""HTTP client core shared by the sync and async vsfx SDKs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from .errors import TransportError, error_from_payload
from .models import Prediction

__all__ = [
    "API_PREFIX",
    "DEFAULT_TIMEOUT",
    "AsyncTransport",
    "SyncTransport",
    "asleep",
    "backoff_delay",
    "encode_list_params",
    "parse_prediction",
    "sleep",
]

API_PREFIX = "/api/v1"
DEFAULT_TIMEOUT = httpx.Timeout(30.0, read=300.0)


def _envelope_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Unwrap an API envelope.

    Parameters:
        payload: Response JSON.

    Returns:
        The envelope `data` object (or the payload itself when bare).
    """
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], dict):
        return payload["data"]
    return payload if isinstance(payload, dict) else {}


def _raise_for_response(response: httpx.Response) -> None:
    """Translate a non-2xx response into a typed SDK error.

    Parameters:
        response: The httpx response.

    Raises:
        APIError: The mapped error.
    """
    if response.is_success:
        return
    error_code = "internal_error"
    message = response.reason_phrase or "request failed"
    request_id = response.headers.get("x-request-id", "")
    retry_after: float | None = None
    details: dict | None = None
    try:
        body = response.json()
        if isinstance(body, dict):
            error_code = str(body.get("error_code", error_code))
            message = str(body.get("message", message))
            request_id = str(body.get("request_id", request_id))
            details = body.get("details")
    except (ValueError, httpx.DecodingError):
        text = response.text
        if text:
            message = text[:300]
    header_retry = response.headers.get("retry-after")
    if header_retry:
        try:
            retry_after = float(header_retry)
        except ValueError:
            retry_after = None
    raise error_from_payload(
        error_code, response.status_code, message, request_id, details, retry_after
    )


class SyncTransport:
    """Synchronous transport: request, unwrap, error-map."""

    def __init__(self, client: httpx.Client) -> None:
        """Bind an httpx client.

        Parameters:
            client: Configured httpx.Client.
        """
        self._client = client

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Perform a request and return unwrapped envelope data.

        Parameters:
            method: HTTP verb.
            path: Path under API_PREFIX (or absolute when starting with http).
            json_body: Optional JSON body.
            params: Query parameters.
            files: Multipart files.
            data: Form fields.
            headers: Extra headers.

        Returns:
            Envelope data dict.

        Raises:
            APIError: On non-2xx.
            ConnectionError: On transport failures.
        """
        url = path if path.startswith(("http://", "https://")) else f"{API_PREFIX}/{path.lstrip('/')}"
        try:
            response = self._client.request(
                method,
                url,
                json=json_body,
                params=params,
                files=files,
                data=data,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise TransportError(f"{method} {url} failed: {exc}") from exc
        _raise_for_response(response)
        return _envelope_data(response.json())

    def stream_to_file(self, url: str, destination: str, chunk_bytes: int = 1 << 20) -> int:
        """Download a URL to a local file.

        Parameters:
            url: Absolute URL.
            destination: Local path.
            chunk_bytes: Stream chunk size.

        Returns:
            Bytes written.

        Raises:
            ConnectionError: On transport failures.
        """
        try:
            with self._client.stream("GET", url) as response:
                _raise_for_response(response)
                written = 0
                with Path(destination).open("wb") as handle:
                    for chunk in response.iter_bytes(chunk_bytes):
                        handle.write(chunk)
                        written += len(chunk)
                return written
        except httpx.HTTPError as exc:
            raise TransportError(f"download {url} failed: {exc}") from exc


class AsyncTransport:
    """Async transport mirroring SyncTransport."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        """Bind an httpx client.

        Parameters:
            client: Configured httpx.AsyncClient.
        """
        self._client = client

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Perform an async request and return unwrapped envelope data.

        Parameters:
            method: HTTP verb.
            path: Path under API_PREFIX.
            json_body: JSON body.
            params: Query parameters.
            files: Multipart files.
            data: Form fields.
            headers: Extra headers.

        Returns:
            Envelope data dict.

        Raises:
            APIError: On non-2xx.
            ConnectionError: On transport failures.
        """
        url = path if path.startswith(("http://", "https://")) else f"{API_PREFIX}/{path.lstrip('/')}"
        try:
            response = await self._client.request(
                method,
                url,
                json=json_body,
                params=params,
                files=files,
                data=data,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise TransportError(f"{method} {url} failed: {exc}") from exc
        _raise_for_response(response)
        return _envelope_data(response.json())

    async def stream_to_file(self, url: str, destination: str, chunk_bytes: int = 1 << 20) -> int:
        """Download a URL to a local file.

        Parameters:
            url: Absolute URL.
            destination: Local path.
            chunk_bytes: Chunk size.

        Returns:
            Bytes written.

        Raises:
            ConnectionError: On transport failures.
        """
        try:
            async with self._client.stream("GET", url) as response:
                _raise_for_response(response)
                written = 0
                with Path(destination).open("wb") as handle:
                    async for chunk in response.aiter_bytes(chunk_bytes):
                        handle.write(chunk)
                        written += len(chunk)
                return written
        except httpx.HTTPError as exc:
            raise TransportError(f"download {url} failed: {exc}") from exc


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 10.0) -> float:
    """Full-jitter exponential backoff delay.

    Parameters:
        attempt: Zero-based retry attempt.
        base: Base delay seconds.
        cap: Maximum delay seconds.

    Returns:
        Delay in seconds.
    """
    ceiling = min(cap, base * (2**attempt))
    import random

    return random.uniform(0.0, ceiling)


def sleep(seconds: float) -> None:
    """Sleep helper (patchable in tests).

    Parameters:
        seconds: Delay duration.
    """
    time.sleep(seconds)


async def asleep(seconds: float) -> None:
    """Async sleep helper (patchable in tests).

    Parameters:
        seconds: Delay duration.
    """
    import asyncio

    await asyncio.sleep(seconds)


def parse_prediction(payload: dict[str, Any]) -> Prediction:
    """Parse a prediction payload defensively.

    Parameters:
        payload: Envelope data.

    Returns:
        Parsed Prediction.
    """
    return Prediction.from_payload(payload)


def encode_list_params(params: dict[str, Any]) -> dict[str, Any]:
    """Quote list values into repeatable query strings.

    Parameters:
        params: Raw parameters.

    Returns:
        httpx-friendly param dict.
    """
    encoded: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            encoded[f"{key}[]"] = [quote(str(item), safe="") for item in value]
        else:
            encoded[key] = value
    return encoded

