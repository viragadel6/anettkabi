"""Synchronous vsfx client."""

from __future__ import annotations

import mimetypes
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from .errors import APIError
from .models import (
    AccountUsage,
    ModelInfo,
    Prediction,
    SfxParams,
    UploadAck,
    UploadPresign,
    WebhookTestResult,
)
from .transport import DEFAULT_TIMEOUT, SyncTransport, backoff_delay, sleep

__all__ = ["Client"]


class Client:
    """Synchronous Video-to-Video SFX API client.

    Attributes:
        base_url: Service origin (no trailing slash).
        api_key: Bearer credential.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        *,
        timeout: httpx.Timeout | float | None = None,
        transport: httpx.BaseTransport | None = None,
        user_agent: str = "vsfx-python/1.0",
    ) -> None:
        """Construct the client.

        Parameters:
            api_key: Service API key (`vsfx_...`).
            base_url: Service origin.
            timeout: Request timeout (default 30s connect / 300s read).
            transport: Optional custom httpx transport (tests).
            user_agent: User-Agent header value.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": user_agent,
            "Accept": "application/json",
        }
        client_kwargs: dict[str, Any] = {
            "base_url": self.base_url,
            "headers": headers,
            "timeout": timeout or DEFAULT_TIMEOUT,
        }
        if transport is not None:
            client_kwargs["transport"] = transport
        self._http = httpx.Client(**client_kwargs)
        self._api = SyncTransport(self._http)

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._http.close()

    def __enter__(self) -> Client:
        """Context manager entry.

        Returns:
            Self.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Context manager exit; closes the pool.

        Parameters:
            exc_info: Exception triple if raised.
        """
        self.close()

    def create(
        self,
        video: str | Path,
        params: SfxParams | None = None,
        *,
        idempotency_key: str | None = None,
        video_url: str | None = None,
    ) -> Prediction:
        """Create a prediction.

        Parameters:
            video: Local file path (multipart upload) or remote https URL.
            params: Generation parameters (prompt required).
            idempotency_key: Optional idempotency token.
            video_url: Explicit video URL (overrides `video`).

        Returns:
            The created Prediction (status queued/starting).

        Raises:
            APIError: On API errors.
            ValueError: When neither video nor video_url is provided.
        """
        resolved = video_url or (str(video) if video else None)
        if not resolved:
            raise ValueError("provide a local video path or video_url")
        body: dict[str, Any] = {"video": resolved}
        if params is not None:
            body.update(params.to_payload())
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        data = self._api.request(
            "POST", "predictions/video-to-video-sfx", json_body=body, headers=headers
        )
        return Prediction.from_payload(data)

    def create_multipart(
        self,
        video_path: str | Path,
        params: SfxParams,
        *,
        idempotency_key: str | None = None,
    ) -> Prediction:
        """Create a prediction by multipart upload.

        Parameters:
            video_path: Local video file.
            params: Generation parameters.
            idempotency_key: Optional idempotency token.

        Returns:
            The created Prediction.

        Raises:
            FileNotFoundError: When video_path does not exist.
            APIError: On API errors.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        content_type = mimetypes.guess_type(str(path))[0] or "video/mp4"
        handle = path.open("rb")
        files = {"video": (path.name, handle, content_type)}
        data = {
            key: str(value)
            for key, value in params.to_payload().items()
            if value is not None and key != "metadata"
        }
        if params.metadata is not None:
            data["metadata"] = str(params.metadata)
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        try:
            payload = self._api.request(
                "POST",
                "predictions/video-to-video-sfx",
                files=files,
                data=data,
                headers=headers,
            )
        finally:
            handle.close()
        return Prediction.from_payload(payload)

    def get(self, prediction_id: str) -> Prediction:
        """Fetch a prediction by id.

        Parameters:
            prediction_id: Prediction identifier.

        Returns:
            Current Prediction state.
        """
        data = self._api.request("GET", f"predictions/{prediction_id}")
        return Prediction.from_payload(data)

    def result(self, prediction_id: str) -> Prediction:
        """Fetch the result envelope (identical to get for this API).

        Parameters:
            prediction_id: Prediction identifier.

        Returns:
            Current Prediction state.
        """
        data = self._api.request("GET", f"predictions/{prediction_id}/result")
        return Prediction.from_payload(data)

    def list(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
        after: str | None = None,
    ) -> list[Prediction]:
        """List the caller's predictions.

        Parameters:
            status: Optional status filter.
            limit: Page size.
            after: Pagination cursor (prediction id).

        Returns:
            Page of Predictions.
        """
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if after is not None:
            params["after"] = after
        data = self._api.request("GET", "predictions", params=params)
        rows = data.get("results") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            rows = data if isinstance(data, list) else []
        return [Prediction.from_payload(row) for row in rows if isinstance(row, dict)]

    def cancel(self, prediction_id: str) -> Prediction:
        """Request cancellation.

        Parameters:
            prediction_id: Prediction identifier.

        Returns:
            Updated Prediction.
        """
        data = self._api.request("POST", f"predictions/{prediction_id}/cancel")
        return Prediction.from_payload(data)

    def delete(self, prediction_id: str) -> None:
        """Soft-delete a prediction.

        Parameters:
            prediction_id: Prediction identifier.
        """
        self._api.request("DELETE", f"predictions/{prediction_id}")

    def wait(
        self,
        prediction_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float | None = None,
        on_poll: Callable[[Prediction], None] | None = None,
    ) -> Prediction:
        """Poll until a prediction reaches a terminal state.

        Parameters:
            prediction_id: Prediction identifier.
            poll_interval: Seconds between polls.
            timeout: Optional wall-clock budget.
            on_poll: Optional progress callback.

        Returns:
            Terminal Prediction.

        Raises:
            APIError: On API errors or timeout (inference_timeout semantics).
        """
        import time as _time

        deadline = None if timeout is None else _time.monotonic() + timeout
        attempt = 0
        while True:
            prediction = self.get(prediction_id)
            if on_poll is not None:
                on_poll(prediction)
            if prediction.is_terminal:
                return prediction
            if deadline is not None and _time.monotonic() >= deadline:
                raise APIError("inference_timeout", 504, f"prediction {prediction_id} timed out")
            delay = backoff_delay(attempt, base=poll_interval, cap=max(poll_interval, 10.0))
            sleep(min(delay, poll_interval if attempt < 1 else delay))
            attempt += 1

    def create_and_wait(
        self,
        video: str | Path,
        params: SfxParams,
        *,
        idempotency_key: str | None = None,
        poll_interval: float = 2.0,
        timeout: float | None = None,
    ) -> Prediction:
        """Create a prediction and wait for completion.

        Parameters:
            video: Video path or URL.
            params: Generation parameters.
            idempotency_key: Idempotency token.
            poll_interval: Poll cadence.
            timeout: Optional wait budget.

        Returns:
            Terminal Prediction.
        """
        prediction = self.create(video, params, idempotency_key=idempotency_key)
        return self.wait(prediction.id, poll_interval=poll_interval, timeout=timeout)

    def download(self, prediction: Prediction, destination: str | Path) -> Path:
        """Download the output MP4 (or WAV) of a succeeded prediction.

        Parameters:
            prediction: Terminal prediction.
            destination: Local target path.

        Returns:
            The written path.

        Raises:
            APIError: When the prediction has no output yet.
        """
        url = self.output_url(prediction)
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._api.stream_to_file(url, str(target))
        return target

    def output_url(self, prediction: Prediction) -> str:
        """Resolve the primary output URL of a prediction.

        Parameters:
            prediction: Prediction with output.

        Returns:
            Absolute URL.

        Raises:
            APIError: When output is absent.
        """
        output = prediction.output or {}
        url = output.get("video") or output.get("audio") or output.get("mp4") or output.get("url")
        if not isinstance(url, str) or not url:
            raise APIError(
                "invalid_request", 400, f"prediction {prediction.id} has no output URL"
            )
        if url.startswith(("http://", "https://")):
            return url
        return f"{self.base_url}{url}"

    def upload(self, video_path: str | Path) -> UploadAck:
        """Upload a video through POST /uploads.

        Parameters:
            video_path: Local video file.

        Returns:
            UploadAck with the retrievable URL.

        Raises:
            FileNotFoundError: When the path is missing.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        content_type = mimetypes.guess_type(str(path))[0] or "video/mp4"
        with path.open("rb") as handle:
            files = {"file": (path.name, handle, content_type)}
            data = self._api.request("POST", "uploads", files=files)
        return UploadAck(
            url=str(data.get("url", "")),
            path=str(data.get("path", data.get("asset_id", ""))),
            expires_at=str(data.get("expires_at", "")),
        )

    def presign(
        self,
        *,
        filename: str,
        content_type: str = "video/mp4",
        size_bytes: int | None = None,
    ) -> UploadPresign:
        """Request a presigned PUT URL.

        Parameters:
            filename: Target filename (extension drives validation).
            content_type: Intended content type.
            size_bytes: Optional size hint.

        Returns:
            UploadPresign with the PUT target.
        """
        body: dict[str, Any] = {"filename": filename, "content_type": content_type}
        if size_bytes is not None:
            body["size_bytes"] = size_bytes
        data = self._api.request("POST", "uploads/presign", json_body=body)
        return UploadPresign(
            upload_url=str(data.get("upload_url", "")),
            url=str(data.get("url", "")),
            path=str(data.get("path", "")),
            expires_at=str(data.get("expires_at", "")),
        )

    def presigned_upload(self, video_path: str | Path) -> UploadAck:
        """Presign, PUT the file, and verify via storage.

        Parameters:
            video_path: Local video file.

        Returns:
            UploadAck describing the stored object.

        Raises:
            FileNotFoundError: When the path is missing.
            APIError: When the PUT rejects.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        content_type = mimetypes.guess_type(str(path))[0] or "video/mp4"
        presign = self.presign(filename=path.name, content_type=content_type, size_bytes=path.stat().st_size)
        with path.open("rb") as handle:
            put_response = self._http.put(
                presign.upload_url,
                content=handle,
                headers={"Content-Type": content_type},
            )
        if put_response.status_code >= 400:
            raise APIError(
                "storage_failed",
                put_response.status_code,
                f"presigned PUT rejected: {put_response.status_code}",
            )
        return UploadAck(url=presign.url, path=presign.path, expires_at=presign.expires_at)

    def model_info(self) -> ModelInfo:
        """Fetch the model card.

        Returns:
            ModelInfo.
        """
        data = self._api.request("GET", "models/video-to-video-sfx")
        return ModelInfo(raw=data)

    def usage(self) -> AccountUsage:
        """Fetch the caller's usage snapshot.

        Returns:
            AccountUsage.
        """
        data = self._api.request("GET", "account/usage")
        return AccountUsage(raw=data)

    def test_webhook(self, url: str) -> WebhookTestResult:
        """Send a signed webhook probe.

        Parameters:
            url: https target.

        Returns:
            WebhookTestResult.
        """
        data = self._api.request("POST", "webhooks/test", json_body={"url": url})
        delivery = data.get("delivery") if isinstance(data, dict) else None
        accepted = bool(delivery.get("accepted")) if isinstance(delivery, dict) else bool(data.get("accepted"))
        status_code = int(delivery.get("status_code", 0)) if isinstance(delivery, dict) else int(data.get("status_code", 0))
        return WebhookTestResult(accepted=accepted, status_code=status_code, raw=data)

    def health(self) -> dict[str, Any]:
        """Liveness probe (no auth required).

        Returns:
            Health payload.
        """
        return self._api.request("GET", "healthz")

    def ready(self) -> dict[str, Any]:
        """Readiness probe.

        Returns:
            Readiness payload.
        """
        return self._api.request("GET", "readyz")

    def version(self) -> dict[str, Any]:
        """Version payload.

        Returns:
            Version info.
        """
        return self._api.request("GET", "version")

    def file_size(self, path: str | Path) -> int:
        """Local helper: byte size of a file.

        Parameters:
            path: File path.

        Returns:
            Size in bytes.
        """
        return Path(path).stat().st_size
