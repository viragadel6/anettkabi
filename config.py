from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "DatabaseSettings",
    "RedisSettings",
    "Settings",
    "WorkerSettings",
    "get_settings",
    "reset_settings",
]

ASYNC_POSTGRES_SCHEME = "postgresql+asyncpg://"


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=True, extra="ignore")

    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://task_engine:task_engine@localhost:5432/task_engine"
    )
    POOL_SIZE: int = Field(default=20, ge=1, le=500)
    MAX_OVERFLOW: int = Field(default=10, ge=0, le=500)
    POOL_TIMEOUT: int = Field(default=30, ge=1, le=600)
    POOL_RECYCLE: int = Field(default=1800, ge=60, le=86400)

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("DATABASE_URL must not be empty")
        if normalized.startswith("postgres://"):
            normalized = ASYNC_POSTGRES_SCHEME + normalized[len("postgres://") :]
        if not normalized.startswith(ASYNC_POSTGRES_SCHEME):
            raise ValueError(
                "DATABASE_URL must use the postgresql+asyncpg:// scheme for the async driver"
            )
        return normalized


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=True, extra="ignore")

    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_MAX_CONNECTIONS: int = Field(default=50, ge=1, le=1000)
    REDIS_SOCKET_TIMEOUT: float = Field(default=5.0, gt=0.0, le=300.0)
    REDIS_SOCKET_CONNECT_TIMEOUT: float = Field(default=5.0, gt=0.0, le=300.0)

    @field_validator("REDIS_URL")
    @classmethod
    def validate_redis_url(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized.startswith(("redis://", "rediss://", "unix://")):
            raise ValueError("REDIS_URL must use the redis://, rediss://, or unix:// scheme")
        return normalized


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WORKER_", case_sensitive=True, extra="ignore")

    CONCURRENCY: int = Field(default=16, ge=1, le=512)
    HEARTBEAT_INTERVAL_SECONDS: float = Field(default=5.0, gt=0.0, le=300.0)
    LOCK_TTL_SECONDS: float = Field(default=30.0, gt=0.0, le=3600.0)
    POLL_INTERVAL_MILLISECONDS: int = Field(default=200, ge=10, le=60000)
    DRAIN_TIMEOUT_SECONDS: float = Field(default=20.0, gt=0.0, le=600.0)
    BATCH_SIZE: int = Field(default=8, ge=1, le=512)

    @model_validator(mode="after")
    def validate_lease_relationships(self) -> WorkerSettings:
        if self.HEARTBEAT_INTERVAL_SECONDS * 2.0 > self.LOCK_TTL_SECONDS:
            raise ValueError(
                "WORKER_LOCK_TTL_SECONDS must be at least twice "
                "WORKER_HEARTBEAT_INTERVAL_SECONDS so a missed heartbeat never "
                "expires a healthy lease"
            )
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=True, extra="ignore")

    ENVIRONMENT: str = Field(default="production")
    OTEL_SERVICE_NAME_API: str = Field(default="task-engine-api")
    OTEL_SERVICE_NAME_WORKER: str = Field(default="task-engine-worker")
    OUTBOX_STREAM_NAME: str = Field(default="task_engine:outbox_events")

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"production", "staging", "development", "test"}:
            raise ValueError(
                "ENVIRONMENT must be one of production, staging, development, test"
            )
        return normalized


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings() -> None:
    get_settings.cache_clear()
