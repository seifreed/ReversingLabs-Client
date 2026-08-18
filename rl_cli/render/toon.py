"""Token-Oriented Object Notation (TOON) encoder.

Implements the encoding side of https://github.com/toon-format/toon
(spec v1.2): a compact, indentation-based serialization aimed at
LLM-friendly output. Only encoding is provided — the CLI emits TOON,
it never needs to parse it. The official ``toon-format`` PyPI package
(0.1.0) ships a ``NotImplementedError`` placeholder encoder, so we
implement the format here instead of depending on it.

Encoding rules covered:
- objects as ``key: value`` lines, nested via 2-space indentation
- arrays of primitives inline: ``key[N]: a,b,c``
- arrays of uniform flat objects as tables: ``key[N]{f1,f2}:`` + rows
- mixed/nested arrays as ``-`` list items
- strings quoted only when required (structural characters, leading or
  trailing whitespace, or values that would parse as numbers/booleans)
"""

from __future__ import annotations

import math
import re
from typing import Any

_INDENT = "  "
_IDENTIFIER_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")
_NUMERIC_LOOKALIKE = re.compile(r"^-?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"}

# A lone surrogate is valid JSON — ``.json()`` decodes ``"\ud800"`` to one —
# but not encodable UTF-8, so an encoded string carrying one raises
# ``UnicodeEncodeError`` when it is written out. The CLI strips these
# upstream, but the encoder answers a string and must not answer one that
# cannot be written; the replacement character is the honest stand-in for a
# character that is not one.
_SURROGATE = re.compile(r"[\ud800-\udfff]")


def encode(value: Any) -> str:
    """Serialize ``value`` to a TOON string (no trailing newline)."""
    if isinstance(value, dict):
        return "\n".join(_object_lines(value, depth=0))
    if isinstance(value, list):
        return "\n".join(_array_lines(value, key=None, depth=0))
    return _scalar(value)


def _object_lines(obj: dict[str, Any], depth: int) -> list[str]:
    pad = _INDENT * depth
    lines: list[str] = []
    for key, value in obj.items():
        encoded_key = _key(key)
        if isinstance(value, dict):
            if value:
                lines.append(f"{pad}{encoded_key}:")
                lines.extend(_object_lines(value, depth + 1))
            else:
                lines.append(f"{pad}{encoded_key}: {{}}")
        elif isinstance(value, list):
            lines.extend(_array_lines(value, key=encoded_key, depth=depth))
        else:
            lines.append(f"{pad}{encoded_key}: {_scalar(value)}")
    return lines


def _array_lines(items: list[Any], key: str | None, depth: int) -> list[str]:
    pad = _INDENT * depth
    prefix = f"{pad}{key}" if key is not None else pad
    header = f"{prefix}[{len(items)}]"

    if all(_is_primitive(item) for item in items):
        joined = ",".join(_scalar(item) for item in items)
        return [f"{header}: {joined}".rstrip()]

    fields = _tabular_fields(items)
    if fields:
        row_pad = _INDENT * (depth + 1)
        lines = [f"{header}{{{','.join(_key(f) for f in fields)}}}:"]
        for item in items:
            lines.append(row_pad + ",".join(_scalar(item[f]) for f in fields))
        return lines

    lines = [f"{header}:"]
    item_pad = _INDENT * (depth + 1)
    for item in items:
        if _is_primitive(item):
            lines.append(f"{item_pad}- {_scalar(item)}")
        elif isinstance(item, dict) and item:
            nested = _object_lines(item, depth + 2)
            first = nested[0].lstrip()
            lines.append(f"{item_pad}- {first}")
            lines.extend(nested[1:])
        elif isinstance(item, dict):
            lines.append(f"{item_pad}- {{}}")
        else:
            nested = _array_lines(item, key=None, depth=depth + 2)
            first = nested[0].lstrip()
            lines.append(f"{item_pad}- {first}")
            lines.extend(nested[1:])
    return lines


def _tabular_fields(items: list[Any]) -> list[str] | None:
    """Return the shared field list if ``items`` qualify for tabular form."""
    if not items or not all(isinstance(item, dict) and item for item in items):
        return None
    fields = list(items[0].keys())
    for item in items:
        if list(item.keys()) != fields:
            return None
        if not all(_is_primitive(value) for value in item.values()):
            return None
    return fields


def _is_primitive(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "null"
        return repr(value)
    return _string(str(value))


def _string(text: str) -> str:
    text = _SURROGATE.sub("�", text)
    if _needs_quotes(text):
        escaped = "".join(_ESCAPES.get(ch, ch) for ch in text)
        return f'"{escaped}"'
    return text


def _needs_quotes(text: str) -> bool:
    if not text or text != text.strip():
        return True
    if text in ("true", "false", "null") or _NUMERIC_LOOKALIKE.match(text):
        return True
    if text.startswith(("-", "#", '"')):
        return True
    return any(ch in text for ch in ',:"{}[]\\\n\r\t')


def _key(key: Any) -> str:
    text = _SURROGATE.sub("�", str(key))
    if _IDENTIFIER_KEY.match(text):
        return text
    escaped = "".join(_ESCAPES.get(ch, ch) for ch in text)
    return f'"{escaped}"'
