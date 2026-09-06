# vsfx-client (TypeScript)

TypeScript/Node SDK for the **Video-to-Video SFX** service. Zero runtime
dependencies (uses global `fetch`, Node 18+), fully typed, ESM.

## Install

```bash
npm install vsfx-client
```

## Quickstart

```ts
import { VSFXClient } from "vsfx-client";

const client = new VSFXClient("vsfx_your_api_key", {
  baseUrl: "https://api.example.com",
});

const prediction = await client.create(
  "https://cdn.example.com/clip.mp4",
  {
    prompt: "cinematic whoosh with deep sub-bass impact",
    audioMode: "duck",
    numInferenceSteps: 32,
    guidanceScale: 3.0,
    seed: 1234,
  },
);

const done = await client.wait(prediction.id, {
  pollIntervalMs: 2000,
  timeoutMs: 600_000,
  onPoll: (p) => console.log(p.status),
});

if (done.status === "succeeded") {
  await client.download(done, "out.mp4");
}
```

## Local files

`create` transparently uploads local paths (Node only); or be explicit:

```ts
const ack = await client.upload("clip.mov");
const prediction = await client.create(ack.url, { prompt: "thunder rumble" });

const presigned = await client.presignedUpload("huge_clip.mov");
```

`createMultipart` streams the file as a single multipart request:

```ts
const prediction = await client.createMultipart("clip.mp4", {
  prompt: "gentle rain on a tin roof",
});
```

## Error handling

```ts
import { RateLimitedError, QuotaExceededError, TransportError } from "vsfx-client";

try {
  await client.create(url, { prompt: "boom" });
} catch (error) {
  if (error instanceof RateLimitedError) {
    await new Promise((r) => setTimeout(r, (error.retryAfter ?? 5) * 1000));
  } else if (error instanceof QuotaExceededError) {
    upgradePlan();
  } else if (error instanceof TransportError) {
    console.error(error.message);
  }
}
```

Every error carries `errorCode`, `httpStatus`, `requestId`, `details`, and
`retryAfter` when provided; `error.retryable` reflects whether backoff-retry
is appropriate.

## API surface

| Method | Endpoint |
| --- | --- |
| `create` / `createMultipart` | `POST /api/v1/predictions/video-to-video-sfx` |
| `get` / `result` | `GET /api/v1/predictions/{id}` (+`/result`) |
| `list` | `GET /api/v1/predictions` |
| `cancel` | `POST /api/v1/predictions/{id}/cancel` |
| `delete` | `DELETE /api/v1/predictions/{id}` |
| `wait` / `VSFXClient.gather` | polling helpers |
| `upload` | `POST /api/v1/uploads` |
| `presign` / `presignedUpload` | `POST /api/v1/uploads/presign` + PUT |
| `modelInfo` | `GET /api/v1/models/video-to-video-sfx` |
| `usage` | `GET /api/v1/account/usage` |
| `testWebhook` | `POST /api/v1/webhooks/test` |
| `health` / `ready` / `version` | probes |

## Build & test

```bash
npm install
npm run build
npm test
```

## License

MIT.
