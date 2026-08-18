"""Guards for the shape an answer states a section as.

The services state the same section as a mapping, a list, a bare string or
null depending on which endpoint answered, and nothing in the CLI wraps a
reader or a formatter in exception handling. So every helper here turns
"not the shape this caller iterates" into "nothing" rather than raising.

They sit beside the payload readers rather than in ``render/formatters``
because the payload layer reads a malformed section first, and may not
import the presentation side to do it.
"""

from __future__ import annotations

from math import isfinite
from typing import Any


def dict_rows(rows: list[Any]) -> list[dict[str, Any]]:
    """Keep only mapping rows, since a result array may mix in bare strings."""
    return [row for row in rows if isinstance(row, dict)]


def mapping(value: Any) -> dict[str, Any]:
    """``value`` when it is a mapping, an empty one otherwise.

    A section stated as a list or a string reads as nothing rather than
    raising ``AttributeError`` on ``.get``.
    """
    return value if isinstance(value, dict) else {}


def count(value: Any) -> int:
    """A count field as an integer, or zero when it is not a finite number.

    ``OverflowError`` counts as "not a number" alongside the type errors,
    because ``.json()`` accepts a bare ``Infinity`` and ``int`` raises on it.
    """
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def stated_count(value: Any) -> int | None:
    """A count field as an integer, or ``None`` when it states no number.

    Strict where :func:`count` is lenient, for the caller that reports the
    number back rather than working with it: ``""``, a word, ``Infinity``
    and ``NaN`` all state no count, and reading any of them as zero invents
    a number we were never told.

    ``OverflowError`` counts as "no number" for the reason :func:`count`
    gives: ``.json()`` decodes an arbitrary-precision integer, and one past
    ``sys.float_info.max`` (a ~309-digit scanner count) makes ``float``
    raise rather than answer -- left uncaught it took the whole report down.
    """
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return int(number) if isfinite(number) else None


def list_section(source: dict[str, Any], key: str) -> list[Any] | None:
    """A section, but only when it is the list its readers iterate.

    The bare TitaniumCore document states some sections as objects.
    """
    value = source.get(key)
    return value if isinstance(value, list) else None
