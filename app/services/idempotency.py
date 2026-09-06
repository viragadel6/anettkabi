"""Idempotency: replay and conflict detection over normalized input hashes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.constants import IDEMPOTENCY_WINDOW_S
from app.db.models import Prediction
from app.db.repositories.predictions import PredictionRepository
from app.db.session import session_scope
from app.errors import ErrorCode, ServiceError
from app.utils.hashing import sha256_bytes

__all__ = ["IdempotencyDecision", "check_idempotency"]


@dataclass(slots=True)
class IdempotencyDecision:
    """Outcome of an idempotency check.

    Attributes:
        replay: The existing prediction to return (replay case).
        conflict: True when the key exists with a different payload (409).
    """

    replay: Prediction | None
    conflict: bool


async def check_idempotency(
    api_key_id: uuid.UUID,
    idempotency_key: str,
    input_hash: str,
    window_s: int = IDEMPOTENCY_WINDOW_S,
) -> IdempotencyDecision:
    """Check an Idempotency-Key against recent predictions.

    Parameters:
        api_key_id: Caller key.
        idempotency_key: Client header value.
        input_hash: SHA-256 of the normalized current input.
        window_s: Replay window.

    Returns:
        IdempotencyDecision (both flags empty/False for a fresh create).

    Raises:
        ServiceError: idempotency_key_conflict when the key maps to a
            different normalized payload within the window.
    """
    async with session_scope() as session:
        row = await PredictionRepository(session).find_idempotent(
            api_key_id, idempotency_key, window_s
        )
    if row is None:
        return IdempotencyDecision(replay=None, conflict=False)
    if row.input_hash != input_hash:
        raise ServiceError(
            ErrorCode.IDEMPOTENCY_KEY_CONFLICT,
            "Idempotency-Key was already used with a different request payload",
        )
    return IdempotencyDecision(replay=row, conflict=False)


def hash_normalized_input(payload: dict[str, object]) -> str:
    """Hash a normalized input dict deterministically.

    Parameters:
        payload: The normalized input (no secrets, sorted serialization).

    Returns:
        SHA-256 hex digest.
    """
    from app.db.repositories.predictions import PredictionRepository

    canonical = PredictionRepository.input_to_json(payload)
    return sha256_bytes(canonical.encode("utf-8"))
