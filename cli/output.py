"""Human/JSON output rendering for the vsfx CLI."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

__all__ = ["emit_error", "emit_json", "emit_table", "print_line"]


def emit_json(payload: Any) -> None:
    """Print a payload as pretty JSON.

    Parameters:
        payload: JSON-serializable object.
    """
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def print_line(text: str, *, quiet: bool = False) -> None:
    """Print a status line unless quiet.

    Parameters:
        text: Message.
        quiet: Suppress when True.
    """
    if not quiet:
        print(text)


def emit_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Render rows as an aligned text table.

    Parameters:
        columns: Header labels.
        rows: Row cells.
    """
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(label)) for label in columns]
    for row in rendered:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    header = "  ".join(str(label).ljust(widths[index]) for index, label in enumerate(columns))
    separator = "  ".join("-" * width for width in widths)
    print(header)
    print(separator)
    for row in rendered:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def emit_error(message: str, *, quiet: bool = False) -> None:
    """Print an error to stderr.

    Parameters:
        message: Error text.
        quiet: When False prefixes `error:`.
    """
    import sys

    text = f"error: {message}" if not quiet else message
    sys.stderr.write(text + "\n")
