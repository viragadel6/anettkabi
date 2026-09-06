"""Async vsfx client."""

from __future__ import annotations

import asyncio
import mimetypes
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import httpx

from .client import Client
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
from .transport import DEFAULT_TIMEOUT, AsyncTransport, asleep, backoff_delay

__all__ = ["AsyncClient"]


class AsyncClient:
    """Async Video-to-Video SFX API client."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        *,
        timeout: httpx.Timeout | float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        user_agent: str = "vsfx-python-async/1.0",
    ) -> None:
        """Construct the async client.

        Parameters:
            api_key: Service API key.
            base_url: Service origin.
            timeout: Request timeout.
            transport: Custom httpx async transport (tests).
            user_agent: User-Agent header.
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
        self._http = httpx.AsyncClient(**client_kwargs)
        self._api = AsyncTransport(self._http)

    async def close(self) -> None:
        """Close the underlying pool."""
        await self._http.aclose()

    async def __aenter__(self) -> AsyncClient:
        """Context manager entry.

        Returns:
            Self.
        """
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Context manager exit.

        Parameters:
            exc_info: Exception triple if raised.
        """
        await self.close()

    async def create(
        self,
        video: str | Path,
        params: SfxParams | None = None,
        *,
        idempotency_key: str | None = None,
        video_url: str | None = None,
    ) -> Prediction:
        """Create a prediction from a URL or local path.

        Parameters:
            video: Video URL or local path.
            params: Generation parameters.
            idempotency_key: Idempotency token.
            video_url: Explicit URL override.

        Returns:
            Created Prediction.

        Raises:
            ValueError: Without video input.
        """
        resolved = video_url or (str(video) if video else None)
        if not resolved:
            raise ValueError("provide a local video path or video_url")
        body: dict[str, Any] = {"video": resolved}
        if params is not None:
            body.update(params.to_payload())
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        data = await self._api.request(
            "POST", "predictions/video-to-video-sfx", json_body=body, headers=headers
        )
        return Prediction.from_payload(data)

    async def create_multipart(
        self,
        video_path: str | Path,
        params: SfxParams,
        *,
        idempotency_key: str | None = None,
    ) -> Prediction:
        """Create a prediction via multipart upload.

        Parameters:
            video_path: Local file.
            params: Generation parameters.
            idempotency_key: Idempotency token.

        Returns:
            Created Prediction.

        Raises:
            FileNotFoundError: Missing file.
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
            payload = await self._api.request(
                "POST", "predictions/video-to-video-sfx", files=files, data=data, headers=headers
            )
        finally:
            handle.close()
        return Prediction.from_payload(payload)

    async def get(self, prediction_id: str) -> Prediction:
        """Fetch a prediction.

        Parameters:
            prediction_id: Prediction id.

        Returns:
            Current state.
        """
        data = await self._api.request("GET", f"predictions/{prediction_id}")
        return Prediction.from_payload(data)

    async def result(self, prediction_id: str) -> Prediction:
        """Fetch the result envelope.

        Parameters:
            prediction_id: Prediction id.

        Returns:
            Current state.
        """
        data = await self._api.request("GET", f"predictions/{prediction_id}/result")
        return Prediction.from_payload(data)

    async def list(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
        after: str | None = None,
    ) -> list[Prediction]:
        """List predictions.

        Parameters:
            status: Status filter.
            limit: Page size.
            after: Cursor.

        Returns:
            Page of predictions.
        """
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if after is not None:
            params["after"] = after
        data = await self._api.request("GET", "predictions", params=params)
        rows = data.get("results") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            rows = data if isinstance(data, list) else []
        return [Prediction.from_payload(row) for row in rows if isinstance(row, dict)]

    async def cancel(self, prediction_id: str) -> Prediction:
        """Request cancellation.

        Parameters:
            prediction_id: Prediction id.

        Returns:
            Updated state.
        """
        data = await self._api.request("POST", f"predictions/{prediction_id}/cancel")
        return Prediction.from_payload(data)

    async def delete(self, prediction_id: str) -> None:
        """Soft-delete a prediction.

        Parameters:
            prediction_id: Prediction id.
        """
        await self._api.request("DELETE", f"predictions/{prediction_id}")

    async def wait(
        self,
        prediction_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float | None = None,
        on_poll: Callable[[Prediction], None] | None = None,
    ) -> Prediction:
        """Poll until terminal.

        Parameters:
            prediction_id: Prediction id.
            poll_interval: Poll cadence.
            timeout: Optional budget seconds.
            on_poll: Progress callback.

        Returns:
            Terminal Prediction.

        Raises:
            APIError: On timeout.
        """
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        attempt = 0
        while True:
            prediction = await self.get(prediction_id)
            if on_poll is not None:
                on_poll(prediction)
            if prediction.is_terminal:
                return prediction
            if deadline is not None and loop.time() >= deadline:
                raise APIError("inference_timeout", 504, f"prediction {prediction_id} timed out")
            delay = backoff_delay(attempt, base=poll_interval, cap=max(poll_interval, 10.0))
            await asleep(min(delay, max(poll_interval, 10.0)))
            attempt += 1

    async def create_and_wait(
        self,
        video: str | Path,
        params: SfxParams,
        *,
        idempotency_key: str | None = None,
        poll_interval: float = 2.0,
        timeout: float | None = None,
    ) -> Prediction:
        """Create then await completion.

        Parameters:
            video: Video source.
            params: Parameters.
            idempotency_key: Idempotency token.
            poll_interval: Poll cadence.
            timeout: Budget.

        Returns:
            Terminal Prediction.
        """
        prediction = await self.create(video, params, idempotency_key=idempotency_key)
        return await self.wait(prediction.id, poll_interval=poll_interval, timeout=timeout)

    async def download(self, prediction: Prediction, destination: str | Path) -> Path:
        """Download the primary output artifact.

        Parameters:
            prediction: Succeeded prediction.
            destination: Local path.

        Returns:
            Written path.
        """
        url = self.output_url(prediction)
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        await self._api.stream_to_file(url, str(target))
        return target

    def output_url(self, prediction: Prediction) -> str:
        """Resolve the primary output URL.

        Parameters:
            prediction: Prediction with output.

        Returns:
            Absolute URL.

        Raises:
            APIError: Without output.
        """
        output = prediction.output or {}
        url = output.get("video") or output.get("audio") or output.get("mp4") or output.get("url")
        if not isinstance(url, str) or not url:
            raise APIError("invalid_request", 400, f"prediction {prediction.id} has no output URL")
        if url.startswith(("http://", "https://")):
            return url
        return f"{self.base_url}{url}"

    async def upload(self, video_path: str | Path) -> UploadAck:
        """Upload through POST /uploads.

        Parameters:
            video_path: Local file.

        Returns:
            UploadAck.

        Raises:
            FileNotFoundError: Missing file.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        content_type = mimetypes.guess_type(str(path))[0] or "video/mp4"
        async with await asyncio.to_thread(open, path, "rb") as handle:
            files = {"file": (path.name, handle, content_type)}
            data = await self._api.request("POST", "uploads", files=files)
        return UploadAck(
            url=str(data.get("url", "")),
            path=str(data.get("path", data.get("asset_id", ""))),
            expires_at=str(data.get("expires_at", "")),
        )

    async def presign(
        self,
        *,
        filename: str,
        content_type: str = "video/mp4",
        size_bytes: int | None = None,
    ) -> UploadPresign:
        """Get a presigned PUT URL.

        Parameters:
            filename: Filename.
            content_type: Content type.
            size_bytes: Optional size hint.

        Returns:
            UploadPresign.
        """
        body: dict[str, Any] = {"filename": filename, "content_type": content_type}
        if size_bytes is not None:
            body["size_bytes"] = size_bytes
        data = await self._api.request("POST", "uploads/presign", json_body=body)
        return UploadPresign(
            upload_url=str(data.get("upload_url", "")),
            url=str(data.get("url", "")),
            path=str(data.get("path", "")),
            expires_at=str(data.get("expires_at", "")),
        )

    async def presigned_upload(self, video_path: str | Path) -> UploadAck:
        """Presign and PUT a file.

        Parameters:
            video_path: Local file.

        Returns:
            UploadAck.

        Raises:
            FileNotFoundError: Missing file.
            APIError: Rejected PUT.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        content_type = mimetypes.guess_type(str(path))[0] or "video/mp4"
        presign = await self.presign(
            filename=path.name, content_type=content_type, size_bytes=path.stat().st_size
        )
        content = await asyncio.to_thread(path.read_bytes)
        put_response = await self._http.put(
            presign.upload_url, content=content, headers={"Content-Type": content_type}
        )
        if put_response.status_code >= 400:
            raise APIError(
                "storage_failed",
                put_response.status_code,
                f"presigned PUT rejected: {put_response.status_code}",
            )
        return UploadAck(url=presign.url, path=presign.path, expires_at=presign.expires_at)

    async def model_info(self) -> ModelInfo:
        """Model card.

        Returns:
            ModelInfo.
        """
        data = await self._api.request("GET", "models/video-to-video-sfx")
        return ModelInfo(raw=data)

    async def usage(self) -> AccountUsage:
        """Usage snapshot.

        Returns:
            AccountUsage.
        """
        data = await self._api.request("GET", "account/usage")
        return AccountUsage(raw=data)

    async def test_webhook(self, url: str) -> WebhookTestResult:
        """Signed webhook probe.

        Parameters:
            url: https target.

        Returns:
            WebhookTestResult.
        """
        data = await self._api.request("POST", "webhooks/test", json_body={"url": url})
        delivery = data.get("delivery") if isinstance(data, dict) else None
        accepted = bool(delivery.get("accepted")) if isinstance(delivery, dict) else bool(data.get("accepted"))
        status_code = int(delivery.get("status_code", 0)) if isinstance(delivery, dict) else int(data.get("status_code", 0))
        return WebhookTestResult(accepted=accepted, status_code=status_code, raw=data)

    async def health(self) -> dict[str, Any]:
        """Liveness.

        Returns:
            Payload.
        """
        return await self._api.request("GET", "healthz")

    async def ready(self) -> dict[str, Any]:
        """Readiness.

        Returns:
            Payload.
        """
        return await self._api.request("GET", "readyz")

    async def version(self) -> dict[str, Any]:
        """Version.

        Returns:
            Payload.
        """
        return await self._api.request("GET", "version")


def sync_wrapper(client: Client) -> Client:
    """Return the provided sync client unchanged (interop helper).

    Parameters:
        client: A sync Client.

    Returns:
        The same client.
    """
    return client


async def gather_predictions(
    client: AsyncClient, prediction_ids: list[str]
) -> list[Prediction]:
    """Await many predictions concurrently.

    Parameters:
        client: Async client.
        prediction_ids: Ids to await.

    Returns:
        Terminal predictions in input order.
    """
    tasks: list[Coroutine[Any, Any, Prediction]] = [
        client.wait(identifier) for identifier in prediction_ids
    ]
    return list(await asyncio.gather(*tasks))
