"""Service-wide constants shared across modules."""

from __future__ import annotations

API_VERSION = "v1"
API_PREFIX = "/api/v1"
PREDICTION_ID_LEN = 26
PREDICTION_ENDPOINT_PATH = "/predictions/video-to-video-sfx"
BEARER_SCHEME = "Bearer"
API_KEY_PREFIX = "vsfx"
API_KEY_RANDOM_LEN = 8
API_KEY_SECRET_LEN = 32
API_KEY_TOTAL_LEN = len(API_KEY_PREFIX) + 1 + API_KEY_RANDOM_LEN + 1 + API_KEY_SECRET_LEN
MAX_PROMPT_CHARS = 2000
MAX_SEED = 2**31 - 1
SEED_RANDOM = -1
MAX_METADATA_BYTES = 4096
IDEMPOTENCY_WINDOW_S = 24 * 3600
UPLOAD_ASSET_TTL_S = 24 * 3600
MULTIPART_THRESHOLD_BYTES = 8 * 1024 * 1024
MULTIPART_CHUNK_BYTES = 16 * 1024 * 1024
DOWNLOAD_REDIRECT_LIMIT = 5
METADATA_HASH_PREFIX_LEN = 16
CLIP_TOKENIZER_CONTEXT_LEN = 77
SYNC_SEGMENT_FRAMES = 16
SYNC_SEGMENT_STRIDE = 8
SYNC_SHORTER_SIDE = 256
CLIP_SHORTER_SIDE = 224
CROP_SIZE = 224
FADE_SECONDS = 0.01
DURATION_TOLERANCE_MS = 10.0
LOUDNESS_BLOCK_SECONDS = 0.4
SILENCE_RMS_THRESHOLD = 1e-5
STAGE_PROGRESS_BASES: dict[str, tuple[int, int]] = {
    "download": (0, 5),
    "probe": (5, 8),
    "validate": (8, 10),
    "moderate": (10, 11),
    "extract_features": (11, 25),
    "generate": (25, 75),
    "decode_audio": (75, 80),
    "post_process": (80, 84),
    "mix": (84, 88),
    "mux": (88, 93),
    "verify": (93, 96),
    "upload": (96, 99),
    "finalize": (99, 100),
}
STAGE_ORDER: tuple[str, ...] = (
    "download",
    "probe",
    "validate",
    "moderate",
    "extract_features",
    "generate",
    "decode_audio",
    "post_process",
    "mix",
    "mux",
    "verify",
    "upload",
    "finalize",
)
STAGES_AFTER_ENQUEUE = STAGE_ORDER
TIMING_KEYS: tuple[str, ...] = (
    "queue_ms",
    "download_ms",
    "probe_ms",
    "feature_extraction_ms",
    "inference_ms",
    "vocoder_ms",
    "post_ms",
    "mux_ms",
    "upload_ms",
    "total_ms",
)
TRANSIENT_ERROR_CODES_FOR_RETRY = frozenset(
    {"download_failed", "storage_failed", "gpu_out_of_memory", "mux_failed", "inference_timeout"}
)
NEVER_RETRY_ERROR_CODES = frozenset(
    {
        "invalid_request",
        "missing_video",
        "invalid_video_url",
        "unsupported_media_type",
        "video_too_large",
        "video_too_long",
        "video_too_short",
        "no_video_stream",
        "corrupt_media",
        "download_forbidden_host",
        "prompt_too_long",
        "prompt_blocked",
        "seed_out_of_range",
        "parameter_out_of_range",
        "prediction_not_cancelable",
        "idempotency_key_conflict",
    }
)
MP4_COPY_SAFE_CODECS = frozenset({"h264", "hevc", "av1"})
FALLBACK_ATTEMPTS = 2
FFMPEG_PROGRESS_INTERVAL_S = 5.0
GPU_CLUSTER_LOCK_KEY = "vsfx:gpu-cluster-lock"
GPU_LOCK_TOKEN_TTL_S = 30
DISK_HEADROOM_BYTES = 2 * 1024 * 1024 * 1024
HTTP_USER_AGENT = "vsfx-service/1.0 (+https://example.invalid/video-sfx)"
DISPOSITION_INLINE = "inline"
CACHE_CONTROL_IMMUTABLE = "public, max-age=31536000, immutable"
