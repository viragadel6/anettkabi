"""Python SDK behavior against a mock httpx transport."""

from __future__ import annotations

import json

import httpx
import pytest
import vsfx_client
from vsfx_client import Client, SfxParams
from vsfx_client.errors import (
    APIError,
    AuthenticationError,
    QuotaExceededError,
    RateLimitedError,
    TransportError,
)


def envelope(data: dict, code: int = 200) -> httpx.Response:
    return httpx.Response(code, json={"code": code, "message": "success", "data": data})


@pytest.fixture
def router() -> dict:
    return {}


def make_client(handler) -> Client:
    return Client("vsfx_test", base_url="http://svc", transport=httpx.MockTransport(handler))


def test_create_sends_bearer_and_body() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return envelope({"id": "01X", "status": "queued"})

    client = make_client(handler)
    prediction = client.create("https://cdn/x.mp4", SfxParams(prompt="boom", seed=3))
    assert captured["auth"] == "Bearer vsfx_test"
    assert captured["body"]["video"] == "https://cdn/x.mp4"
    assert captured["body"]["seed"] == 3
    assert prediction.id == "01X"
    assert prediction.is_active


def test_params_to_payload_drops_none() -> None:
    payload = SfxParams(prompt="p", negative_prompt=None, seed=None).to_payload()
    assert payload == {"prompt": "p"}


def test_error_mapping_rate_limited() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"code": 429, "message": "slow", "error_code": "rate_limited", "request_id": "r1"},
            headers={"Retry-After": "2.5"},
        )

    client = make_client(handler)
    with pytest.raises(RateLimitedError) as excinfo:
        client.get("01X")
    assert excinfo.value.retry_after == 2.5
    assert excinfo.value.retryable


def test_error_mapping_quota_and_auth() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/q"):
            return httpx.Response(402, json={"code": 402, "message": "pay", "error_code": "quota_exceeded"})
        return httpx.Response(401, json={"code": 401, "message": "who", "error_code": "unauthorized"})

    client = make_client(handler)
    with pytest.raises(QuotaExceededError) as quota:
        client._api.request("GET", "q")
    assert quota.value.retryable is False
    with pytest.raises(AuthenticationError):
        client.get("nope")


def test_wait_polls_until_terminal() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        status = "processing" if calls["n"] < 3 else "succeeded"
        return envelope({"id": "01W", "status": status, "output": {"video": "/o/01W.mp4"}})

    client = make_client(handler)
    terminal = client.wait("01W", poll_interval=0.0, timeout=5.0)
    assert terminal.status == "succeeded"
    assert client.output_url(terminal) == "http://svc/o/01W.mp4"


def test_output_url_requires_output() -> None:
    prediction = vsfx_client.Prediction.from_payload({"id": "01N", "status": "failed"})
    client = make_client(lambda request: envelope({}))
    with pytest.raises(APIError):
        client.output_url(prediction)


def test_transport_error_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = make_client(handler)
    with pytest.raises(TransportError):
        client.health()


def test_multipart_fields_flattened(tmp_path) -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers.get("content-type", "")
        return envelope({"id": "01M", "status": "queued"})

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 1024)
    client = make_client(handler)
    prediction = client.create_multipart(video, SfxParams(prompt="p", seed=1))
    assert prediction.id == "01M"
    assert "multipart/form-data" in captured["content_type"]


def test_download_streams_to_file(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "/api/v1/" in request.url.path:
            return envelope({"id": "01D", "status": "succeeded", "output": {"video": "/files/out.mp4"}})
        return httpx.Response(200, content=b"MP4BYTES" * 512)

    client = make_client(handler)
    prediction = client.get("01D")
    target = tmp_path / "out.mp4"
    client.download(prediction, target)
    assert target.stat().st_size == len(b"MP4BYTES") * 512
