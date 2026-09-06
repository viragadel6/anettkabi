"""Redis token-bucket rate limiting per API key and per IP."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import get_settings
from app.errors import ErrorCode, ServiceError, error_envelope
from app.middleware.request_id import get_request_id
from app.services.queue import get_redis_client

__all__ = ["RateLimitMiddleware", "token_bucket_allow"]

_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local refill = tonumber(ARGV[4])
local requested = tonumber(ARGV[5])
local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil then
    tokens = burst
    ts = now
end
local delta = math.max(0, now - ts)
tokens = math.min(burst, tokens + delta * (rate / refill))
local allowed = 0
local retry_after = 0
if tokens >= requested then
    tokens = tokens - requested
    allowed = 1
else
    retry_after = math.ceil((requested - tokens) / (rate / refill))
end
redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, math.ceil(refill * 2))
return {allowed, retry_after, math.floor(tokens)}
"""


class _Clock:
    """Wall clock abstraction so tests can simulate time travel."""

    @staticmethod
    def now_s() -> float:
        """Return current time in seconds.

        Returns:
            Wall-clock seconds as float.
        """
        return time.time()


async def token_bucket_allow(
    identifier: str,
    *,
    rate_per_minute: int,
    burst: int,
    refill_window_s: float = 60.0,
    requested: float = 1.0,
    now_s: float | None = None,
) -> tuple[bool, float, float]:
    """Consume from a Redis-backed token bucket.

    Parameters:
        identifier: Bucket identity (api key id or ip).
        rate_per_minute: Sustained requests per minute.
        burst: Bucket capacity.
        refill_window_s: Seconds over which `rate_per_minute` accrues.
        requested: Token cost of this request.
        now_s: Override current time (tests).

    Returns:
        (allowed, retry_after_s, remaining_tokens).

    Raises:
        ServiceError: storage_failed only surfaces from callers; network errors
            here fail open and return (True, 0, burst).
    """
    settings = get_settings()
    client = get_redis_client(settings.redis.redis_url)
    now = now_s if now_s is not None else _Clock.now_s()
    try:
        result = await client.eval(
            _TOKEN_BUCKET_LUA,
            1,
            f"vsfx:rl:{identifier}",
            rate_per_minute,
            burst,
            f"{now:.6f}",
            refill_window_s,
            requested,
        )
        allowed, retry_after, remaining = int(result[0]), float(result[1]), float(result[2])
        return bool(allowed), retry_after, remaining
    except Exception:
        return True, 0.0, float(burst)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Applies token-bucket limits and X-RateLimit response headers."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Rate-limit authenticated /api/v1 traffic per key, per IP otherwise.

        Parameters:
            request: Incoming request.
            call_next: Next middleware/route callable.

        Returns:
            Upstream response with rate headers, or 429 envelope.
        """
        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)
        settings = get_settings()
        api_key_id = getattr(request.state, "api_key_id", None)
        rpm = getattr(request.state, "api_key_rate_limit_rpm", settings.auth.rate_limit_rpm)
        if api_key_id:
            identity = f"key:{api_key_id}"
        else:
            identity = f"ip:{request.client.host if request.client else 'unknown'}"
        burst = max(settings.auth.rate_limit_burst, rpm)
        allowed, retry_after, remaining = await token_bucket_allow(
            identity, rate_per_minute=rpm, burst=burst
        )
        response: Response
        if not allowed:
            error = ServiceError(
                ErrorCode.RATE_LIMITED,
                f"rate limit exceeded for {identity}; retry after {retry_after:.0f}s",
                retry_after_s=max(retry_after, 1.0),
            )
            response = JSONResponse(
                status_code=error.http_status,
                content=error_envelope(error, get_request_id(request)),
            )
        else:
            response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rpm)
        response.headers["X-RateLimit-Remaining"] = str(int(remaining))
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))
        if not allowed:
            response.headers["Retry-After"] = str(int(max(retry_after, 1.0)))
        return response
