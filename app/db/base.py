"""Declarative base and shared column types."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import DateTime, MetaData
from sqlalchemy.orm import DeclarativeBase

__all__ = ["Base", "utc_now_column"]

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 declarative base with a naming convention."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def utc_now_column() -> dt.datetime:
    """Default callable producing timezone-aware UTC now.

    Returns:
        Current UTC datetime used for created_at/updated_at defaults.
    """
    return dt.datetime.now(tz=dt.UTC)


UTC_TIMESTAMP = DateTime(timezone=True)
