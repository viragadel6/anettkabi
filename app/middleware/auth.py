"""Authentication middleware: Bearer API keys verified with argon2."""

from __future__ import annotations

import asyncio
import hmac
from typing import Any, ClassVar

import structlog
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import get_settings
from app.constants import API_KEY_PREFIX, API_KEY_TOTAL_LEN, BEARER_SCHEME
from app.db.repositories.api_keys import ApiKeyRepository
from app.db.session import session_scope
from app.errors import ErrorCode, ServiceError, error_envelope
from app.middleware.request_id import get_request_id

__all__ = ["AuthMiddleware", "hash_api_key", "parse_api_key", "verify_api_key"]

_logger = structlog.get_logger("vsfx.auth")


class BackgroundTasksLocal:
    """Holds references to fire-and-forget tasks so they are not garbage-collected."""

    _tasks: ClassVar[set[asyncio.Task[None]]] = set()

    @classmethod
    def add(cls, task: asyncio.Task[None]) -> None:
        """Track a background task until completion.

        Parameters:
            task: The created task.
        """
        cls._tasks.add(task)
        task.add_done_callback(cls._tasks.discard)



_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)

PUBLIC_PATHS = frozenset(
    {
        "/",
        "/healthz",
        "/readyz",
        "/version",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
    }
)


def parse_api_key(header_value: str | None) -> tuple[str, str] | None:
    """Split an Authorization header into (prefix, secret).

    Parameters:
        header_value: Raw header value, e.g. `Bearer vsfx_abcd1234_...`.

    Returns:
        (key_prefix, secret) when well-formed, None otherwise.
    """
    if not header_value:
        return None
    scheme, _, token = header_value.partition(" ")
    if scheme.lower() != BEARER_SCHEME.lower() or not token:
        return None
    token = token.strip()
    if not token.startswith(API_KEY_PREFIX + "_") or len(token) != API_KEY_TOTAL_LEN:
        return None
    _, prefix, secret = token.split("_", 2)
    if len(prefix) != 8 or len(secret) != 32:
        return None
    return prefix, secret


def hash_api_key(secret: str) -> str:
    """Hash an API key secret with argon2id.

    Parameters:
        secret: The 32-char secret segment.

    Returns:
        The argon2 encoded hash string.
    """
    return _hasher.hash(secret)


def verify_api_key(secret: str, key_hash: str) -> bool:
    """Verify a secret against its stored argon2 hash.

    Parameters:
        secret: Presented secret segment.
        key_hash: Stored argon2 hash.

    Returns:
        True when the secret matches.
    """
    try:
        return _hasher.verify(key_hash, secret)
    except VerifyMismatchError:
        return False
    except Exception:
        return False


class AuthMiddleware(BaseHTTPMiddleware):
    """Authenticates /api/v1 requests; public paths pass through."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Authenticate the request when required.

        Parameters:
            request: Incoming request.
            call_next: Next middleware/route callable.

        Returns:
            The upstream response, or a 401 envelope.
        """
        settings = get_settings()
        path = request.url.path
        if not settings.auth.auth_required or path in PUBLIC_PATHS or not path.startswith("/api/"):
            return await call_next(request)
        parsed = parse_api_key(request.headers.get(settings.auth.api_key_header))
        request_id = get_request_id(request)
        if parsed is None:
            return self._unauthorized(request_id, "missing or malformed API key credentials")
        prefix, secret = parsed
        try:
            api_key = await self._load_key(prefix)
        except Exception as exc:
            _logger.exception("auth_lookup_failed", prefix=prefix[:2] + "**", error=str(exc))
            return self._unauthorized(request_id, "credential lookup failed")
        if api_key is None:
            return self._unauthorized(request_id, "unknown API key")
        if not hmac.compare_digest(prefix, api_key["key_prefix"]):
            return self._unauthorized(request_id, "credential mismatch")
        if api_key["disabled_at"] is not None:
            return self._unauthorized(request_id, "API key is disabled")
        if not verify_api_key(secret, api_key["key_hash"]):
            return self._unauthorized(request_id, "invalid API key secret")
        request.state.api_key_id = str(api_key["id"])
        request.state.api_key_scopes = list(api_key["scopes"])
        request.state.api_key_rate_limit_rpm = int(api_key["rate_limit_rpm"])
        request.state.api_key_concurrency_limit = int(api_key["concurrency_limit"])
        request.state.api_key_monthly_quota = float(api_key["monthly_seconds_quota"])
        self._schedule_touch(api_key["id"])
        return await call_next(request)

    async def _load_key(self, prefix: str) -> dict[str, Any] | None:
        """Fetch key columns needed for authentication.

        Parameters:
            prefix: The 8-char key prefix.

        Returns:
            Dict of key attributes or None when unknown.
        """
        async with session_scope() as session:
            repo = ApiKeyRepository(session)
            row = await repo.get_by_prefix(prefix)
            if row is None:
                return None
            return {
                "id": row.id,
                "key_prefix": row.key_prefix,
                "key_hash": row.key_hash,
                "scopes": row.scopes,
                "disabled_at": row.disabled_at,
                "rate_limit_rpm": row.rate_limit_rpm,
                "concurrency_limit": row.concurrency_limit,
                "monthly_seconds_quota": float(row.monthly_seconds_quota),
            }

    def _schedule_touch(self, api_key_id: Any) -> None:
        """Fire-and-forget last_used_at update via a background task.

        Parameters:
            api_key_id: Key UUID.
        """
        import asyncio

        async def _touch() -> None:
            try:
                async with session_scope() as session:
                    await ApiKeyRepository(session).touch_last_used(api_key_id)
            except Exception as exc:
                _logger.debug("last_used_update_failed", error=str(exc))

        try:
            loop = asyncio.get_running_loop()
            BackgroundTasksLocal.add(loop.create_task(_touch()))
        except RuntimeError:
            pass

    def _unauthorized(self, request_id: str, reason: str) -> JSONResponse:
        """Build a 401 envelope response.

        Parameters:
            request_id: Current request id.
            reason: Safe human reason.

        Returns:
            JSONResponse with the standard envelope.
        """
        error = ServiceError(ErrorCode.UNAUTHORIZED, reason)
        return JSONResponse(status_code=error.http_status, content=error_envelope(error, request_id))
