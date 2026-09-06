"""Generate an API key: print it once, store only the argon2 hash."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings  # noqa: E402
from app.constants import API_KEY_PREFIX, API_KEY_RANDOM_LEN, API_KEY_SECRET_LEN  # noqa: E402
from app.db.repositories.api_keys import ApiKeyRepository  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.middleware.auth import hash_api_key  # noqa: E402
from app.utils.ids import new_api_secret  # noqa: E402


async def create_key(
    name: str,
    scopes: list[str],
    rate_limit_rpm: int,
    concurrency_limit: int,
    monthly_seconds_quota: float,
) -> tuple[str, str]:
    """Create one API key row.

    Parameters:
        name: Human label.
        scopes: Granted scopes.
        rate_limit_rpm: Requests per minute.
        concurrency_limit: Concurrent jobs.
        monthly_seconds_quota: Monthly seconds.

    Returns:
        (plaintext key, key id).

    Raises:
        Exception: Database errors propagate.
    """
    prefix = new_api_secret(API_KEY_RANDOM_LEN)
    secret = new_api_secret(API_KEY_SECRET_LEN)
    plaintext = f"{API_KEY_PREFIX}_{prefix}_{secret}"
    async with session_scope() as session:
        row = await ApiKeyRepository(session).create(
            name=name,
            key_prefix=prefix,
            key_hash=hash_api_key(secret),
            scopes=scopes,
            rate_limit_rpm=rate_limit_rpm,
            concurrency_limit=concurrency_limit,
            monthly_seconds_quota=monthly_seconds_quota,
        )
        return plaintext, str(row.id)


def main() -> int:
    """CLI entry creating one API key.

    Returns:
        0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="key label")
    parser.add_argument(
        "--scopes",
        default="predictions:write",
        help="comma-separated scopes",
    )
    parser.add_argument("--rpm", type=int, default=None, help="rate limit override")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--monthly-seconds", type=float, default=None)
    args = parser.parse_args()
    settings = get_settings()
    plaintext, key_id = asyncio.run(
        create_key(
            args.name,
            [scope.strip() for scope in args.scopes.split(",") if scope.strip()],
            args.rpm or settings.auth.rate_limit_rpm,
            args.concurrency or settings.auth.concurrent_jobs_per_key,
            args.monthly_seconds or settings.auth.monthly_seconds_quota,
        )
    )
    print(f"API key created (id={key_id}).")
    print("Store it now; the secret will never be shown again:")
    print(plaintext)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
