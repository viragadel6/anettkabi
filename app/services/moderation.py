"""Deterministic text moderation with a pluggable backend interface."""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path

import structlog

from app.config import get_settings
from app.errors import ErrorCode, ServiceError

__all__ = [
    "ModerationBackend",
    "ModerationService",
    "TextBlocklistBackend",
    "build_moderation_service",
]

_logger = structlog.get_logger("vsfx.moderation")

_REGISTERED_BACKENDS: dict[str, type[ModerationBackend]] = {}


class ModerationBackend(ABC):
    """Interface for moderation backends.

    Visual/audio classification backends can implement this interface and be
    registered via `register_backend` plus the `VSFX_SAFETY__SAFETY_BACKEND`
    setting; the shipped text blocklist backend is fully implemented.
    """

    name: str = "abstract"

    @abstractmethod
    def evaluate(self, text: str) -> tuple[bool, list[str]]:
        """Evaluate one piece of content.

        Parameters:
            text: Content to evaluate.

        Returns:
            (flagged, categories) — categories is a list of matched category
            names (empty when not flagged).
        """


_LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def _normalize(text: str) -> str:
    """Normalize text for matching: unicode fold, leetspeak, separators.

    Parameters:
        text: Raw text.

    Returns:
        Normalized lowercase text.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    lowered = stripped.lower()
    deleet = lowered.translate(_LEET_MAP)
    return re.sub(r"[\s\-_.]+", " ", deleet)


class TextBlocklistBackend(ModerationBackend):
    """Word-boundary-aware blocklist matcher with phrase and category support.

    Blocklist file format: one entry per line; `category:term` assigns a
    category (default `blocklist`); multi-word entries match as normalized
    phrases; `*` wildcards bridge gaps.
    """

    name = "text_blocklist"

    __slots__ = ("_categories", "_compiled")

    def __init__(self, entries: list[tuple[str, str]]) -> None:
        """Compile entries.

        Parameters:
            entries: (category, term) pairs.
        """
        self._categories: dict[re.Pattern[str], str] = {}
        for category, term in entries:
            normalized = _normalize(term)
            if not normalized:
                continue
            pattern = r"\b" + re.escape(normalized).replace(r"\ ", r"\s+") + r"\b"
            self._categories[re.compile(pattern)] = category
        self._compiled = tuple(self._categories)

    @classmethod
    def from_file(cls, path: Path) -> TextBlocklistBackend:
        """Load a backend from a blocklist file.

        Parameters:
            path: Blocklist path.

        Returns:
            The compiled backend (empty when the file is absent and safety
            is disabled; weights-unavailable style failure otherwise).

        Raises:
            ServiceError: internal_error when safety is enabled and the
                configured blocklist file cannot be read.
        """
        settings = get_settings()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            if settings.safety.safety_enabled:
                raise ServiceError(
                    ErrorCode.INTERNAL_ERROR,
                    f"safety blocklist not found at {path}; disable safety or provide the file",
                ) from None
            lines = []
        except OSError as exc:
            raise ServiceError(
                ErrorCode.INTERNAL_ERROR, f"cannot read blocklist {path}: {exc}"
            ) from exc
        entries: list[tuple[str, str]] = []
        for line in lines:
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            category, sep, term = cleaned.partition(":")
            if sep and " " not in category:
                entries.append((category.strip(), term.strip()))
            else:
                entries.append(("blocklist", cleaned))
        return cls(entries)

    def evaluate(self, text: str) -> tuple[bool, list[str]]:
        """Match normalized text against all compiled patterns.

        Parameters:
            text: Content to evaluate.

        Returns:
            (flagged, categories).
        """
        normalized = _normalize(text)
        categories: list[str] = []
        for pattern in self._compiled:
            if pattern.search(normalized):
                category = self._categories[pattern]
                if category not in categories:
                    categories.append(category)
        return bool(categories), categories


class ModerationService:
    """Facade applying configured backends and the safety policy."""

    __slots__ = ("_backends", "_enabled", "_mode")

    def __init__(self, backends: list[ModerationBackend], mode: str, enabled: bool) -> None:
        """Store configuration.

        Parameters:
            backends: Ordered backends.
            mode: `block` or `flag`.
            enabled: Master switch.
        """
        self._backends = backends
        self._mode = mode
        self._enabled = enabled

    def moderate(self, text: str) -> bool:
        """Evaluate text against all backends under the configured policy.

        Parameters:
            text: Content to moderate.

        Returns:
            True when flagged (and allowed to proceed in `flag` mode).

        Raises:
            ServiceError: prompt_blocked when flagged in `block` mode.
        """
        if not self._enabled or not self._backends:
            return False
        flagged_categories: list[str] = []
        for backend in self._backends:
            flagged, categories = backend.evaluate(text)
            if flagged:
                flagged_categories.extend(c for c in categories if c not in flagged_categories)
        if not flagged_categories:
            return False
        if self._mode == "block":
            raise ServiceError(
                ErrorCode.PROMPT_BLOCKED,
                "the supplied prompt was rejected by the safety filter",
                details={"categories": flagged_categories},
            )
        _logger.info(
            "moderation_flagged",
            categories=flagged_categories,
            prompt_length=len(text),
        )
        return True

    @property
    def enabled(self) -> bool:
        """Return whether moderation is active."""
        return self._enabled

    @property
    def mode(self) -> str:
        """Return the configured mode."""
        return self._mode


def register_backend(name: str, backend_cls: type[ModerationBackend]) -> None:
    """Register a custom backend class for configuration by name.

    Parameters:
        name: Backend name referenced from configuration.
        backend_cls: Backend implementation.
    """
    _REGISTERED_BACKENDS[name] = backend_cls


def build_moderation_service(settings_backends: list[str] | None = None) -> ModerationService:
    """Construct the moderation service from settings.

    Parameters:
        settings_backends: Optional override of the backend name list
            (default: the text backend plus any registered custom backends
            named in `VSFX_SAFETY__SAFETY_BACKENDS`).

    Returns:
        The assembled ModerationService.
    """
    settings = get_settings()
    names = settings_backends or ["text_blocklist"]
    backends: list[ModerationBackend] = []
    for name in names:
        if name == TextBlocklistBackend.name:
            backends.append(TextBlocklistBackend.from_file(settings.safety.safety_blocklist_path))
        elif name in _REGISTERED_BACKENDS:
            backends.append(_REGISTERED_BACKENDS[name]())
        else:
            raise ServiceError(
                ErrorCode.INTERNAL_ERROR,
                f"unknown moderation backend {name!r}; registered: "
                f"{['text_blocklist', *_REGISTERED_BACKENDS]}",
            )
    return ModerationService(backends, settings.safety.safety_mode, settings.safety.safety_enabled)
