# vsfx-client (Python)

Async and sync Python SDK for the **Video-to-Video SFX** service: generate
fully designed sound effects for any video and receive a muxed MP4 (video +
new audio) or standalone WAV, synced within ±10 ms.

## Install

```bash
pip install vsfx-client
```

Requires Python 3.10+; only runtime dependency is `httpx`.

## Quickstart

```python
from vsfx_client import Client, SfxParams

client = Client("vsfx_your_api_key", base_url="https://api.example.com")

prediction = client.create(
    video="https://cdn.example.com/clip.mp4",
    params=SfxParams(
        prompt="cinematic whoosh with deep sub-bass impact",
        audio_mode="duck",
        num_inference_steps=32,
        guidance_scale=3.0,
        seed=1234,
    ),
)
prediction = client.wait(prediction.id, poll_interval=2.0)
if prediction.status == "succeeded":
    client.download(prediction, "out.mp4")
```

## Multipart upload (no public URL needed)

```python
prediction = client.create_multipart(
    "local/clip.mp4",
    SfxParams(prompt="gentle rain on a tin roof", duration=12.0),
)
```

## Presigned upload (direct-to-storage for large files)

```python
ack = client.presigned_upload("local/big_clip.mov")
prediction = client.create(ack.url, SfxParams(prompt="thunder rumble"))
```

## Async usage

```python
import asyncio
from vsfx_client import AsyncClient, SfxParams

async def main() -> None:
    async with AsyncClient("vsfx_your_api_key") as client:
        prediction = await client.create(
            "https://cdn.example.com/clip.mp4",
            SfxParams(prompt="retro synth stab"),
        )
        prediction = await client.wait(prediction.id)
        await client.download(prediction, "out.mp4")

asyncio.run(main())
```

## Error handling

Every non-2xx response maps to a typed exception carrying `error_code`,
`http_status`, `request_id`, and `retry_after` when the server provides it:

```python
from vsfx_client import RateLimitedError, QuotaExceededError, TransportError, APIError

try:
    prediction = client.create(url, params)
except RateLimitedError as exc:
    time.sleep(exc.retry_after or 5.0)
except QuotaExceededError:
    upgrade_plan()
except TransportError as exc:
    log(exc.message)
```

`APIError.retryable` reports whether an exponential-backoff retry is
appropriate (rate/concurrency limits and 5xx family, never quota/auth).

## API surface

| Method | Endpoint |
| --- | --- |
| `create` / `create_multipart` | `POST /api/v1/predictions/video-to-video-sfx` |
| `get` / `result` | `GET /api/v1/predictions/{id}` (+`/result`) |
| `list` | `GET /api/v1/predictions` |
| `cancel` | `POST /api/v1/predictions/{id}/cancel` |
| `delete` | `DELETE /api/v1/predictions/{id}` |
| `upload` | `POST /api/v1/uploads` |
| `presign` / `presigned_upload` | `POST /api/v1/uploads/presign` + PUT |
| `model_info` | `GET /api/v1/models/video-to-video-sfx` |
| `usage` | `GET /api/v1/account/usage` |
| `test_webhook` | `POST /api/v1/webhooks/test` |
| `health` / `ready` / `version` | Probes |

## Webhook signature verification (server side of your app)

Webhooks are signed with HMAC-SHA256 over the raw body using your API key
secret; see the service docs (`docs/api.md`) for header names and the exact
canonicalization.

## License

MIT.
