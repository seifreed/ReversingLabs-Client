"""Make attacker-authored strings safe to put on a terminal.

Every interesting string this tool prints - a sample's file name, a threat
name, a YARA rule, an extracted path, a TitaniumCore tag - was written by
whoever built the malware. Rich strips BEL, backspace and carriage return
but not ESC, so ``\\x1b[2J`` in a file name clears the analyst's screen
mid-report, and a right-to-left override makes ``invoice\\u202egpj.exe``
render as ``invoice.jpg``.

This module must stay a leaf that imports nothing from the package:
``models``, ``render`` and ``services`` all sanitize, so a home inside any
of them would aim an edge back against the layering, and
``tests/test_architecture.py`` fails when one appears. It imports no Rich
either, so a layer that never renders can still sanitize; the markup
escaping lives in ``rl_cli.render.markup``.

:func:`sanitize` and :func:`says_nothing` answer different questions and
must not be swapped. ``sanitize`` asks *is this safe to print*, and names
the characters that drive a terminal or lie about a file's extension;
``says_nothing`` asks *is there anything here at all*, over the wider set
that is merely invisible.
"""

from __future__ import annotations

import re
import unicodedata

# C0 and C1 controls except tab and newline, plus DEL, plus the surrogate
# range, plus the bidi overrides that let a name lie about its own
# extension. The surrogates are here because a lone one — ``"\ud800"`` is
# valid JSON the appliance can send, and ``.json()`` decodes it as such —
# is not encodable UTF-8, so writing it out under ``-o table`` or ``-o
# toon`` raised ``UnicodeEncodeError`` and took the whole render down. B613
# flags the bidi codepoints as trojan-source; here the pattern strips them.
_UNSAFE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\ud800-\udfff‪-‮⁦-⁩]")  # nosec B613

# Unicode general categories that put no mark of their own on a screen:
# controls, format characters, surrogates, private use, unassigned, and
# the combining marks.
_INVISIBLE_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn", "Mn", "Me"})

# The width every listing draws a digest cell at: enough to recognise a
# sample by, since the full hash is in ``-o json``.
DIGEST_CELL_WIDTH = 16

# What a clipped value ends with, and the room that costs inside the width.
_ELLIPSIS = "..."


def sanitize(value: object) -> str:
    """Render ``value`` as a string that cannot drive the terminal.

    Coerces non-strings rather than raising: a JSON field may legitimately
    be a number, a list or ``None``, and no report may die over one.
    """
    return _UNSAFE.sub("", value if isinstance(value, str) else str(value))


def clip(value: str, width: int) -> str:
    """``value`` shortened to ``width``, marked when anything was cut.

    The mark is spent out of the width, not added to it, so the result
    never exceeds ``width`` and an unmarked cell is known to be complete.
    """
    return value if len(value) <= width else value[: width - len(_ELLIPSIS)] + _ELLIPSIS


def digest_cell(value: object) -> str:
    """A hash as the fixed-width cell every listing draws it in.

    Route every digest column through this, or the truncation mark stops
    meaning one thing across the package.
    """
    return clip(sanitize(value), DIGEST_CELL_WIDTH)


def byte_count(value: object) -> str | None:
    """A byte count as grouped digits, or ``None`` when it is not one.

    ``None`` rather than a word, because the word differs by destination:
    the renderers spell an unstated value ``N/A`` in a panel field and
    ``-`` in a table cell, and a leaf module cannot see which it is
    feeding. Non-finite floats and unparseable strings are not counts
    either — ``f"{size:,}"`` prints ``inf`` for one and raises on the other.
    """
    if not isinstance(value, str | int | float):
        return None
    try:
        size = int(value)
    except (ValueError, OverflowError):
        return None
    return f"{size:,}"


def byte_size(value: object) -> str | None:
    """A byte count with its unit, or ``None`` when the value is not one.

    For wherever the column or label does not already carry the unit;
    :func:`byte_count` is the same number for the places that do.
    """
    counted = byte_count(value)
    return None if counted is None else f"{counted} bytes"


def says_nothing(value: str) -> bool:
    """Whether ``value`` would leave no readable mark on a screen.

    The wider of this module's two questions, covering everything merely
    invisible: a zero-width space, a bidi mark, the BOM, a soft hyphen, a
    lone combining accent. Written against Unicode's own categories rather
    than a list of codepoints, so that it stays in step with a standard
    that gains characters.
    """
    return not any(
        unicodedata.category(character) not in _INVISIBLE_CATEGORIES and not character.isspace()
        for character in value
    )
