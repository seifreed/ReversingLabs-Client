"""How a ``SampleVerdict`` severity is painted, in one table.

``payload.py`` decides the verdict and stays presentation-free; deciding
what it looks like belongs here. One table, and every lookup falls back to
the "no opinion" row rather than raising, so a fifth severity added to
``payload.py`` does not raise ``KeyError`` out of a renderer.

The sort rank is the one thing here that is not presentation: ``payload.py``
compares two severities to decide which verdict wins when an entry and the
record it carries disagree, so the ranks live there and ``rank_of`` reads
them rather than restating them.
"""

from __future__ import annotations

from typing import NamedTuple

from rl_cli.models.payload import severity_rank


class _Presentation(NamedTuple):
    """Everything one severity looks like, across every renderer."""

    colour: str
    icon: str


# Green means "known clean" and nothing else, so "unknown" — the severity
# of every verdict payload.py does not recognise — is neutral white rather
# than red or green.
_PRESENTATIONS: dict[str, _Presentation] = {
    "malicious": _Presentation(colour="red", icon="🔴"),
    "suspicious": _Presentation(colour="yellow", icon="🟡"),
    "unknown": _Presentation(colour="white", icon="⚪"),
    "known": _Presentation(colour="green", icon="🟢"),
}

# What a severity this module has never heard of gets: the same neutral
# treatment as a verdict nobody recognised, which is what it is.
_UNRECOGNISED = _PRESENTATIONS["unknown"]


def _presentation(severity: str) -> _Presentation:
    return _PRESENTATIONS.get(severity, _UNRECOGNISED)


def colour_of(severity: str) -> str:
    """The Rich colour a panel or table cell paints this severity in."""
    return _presentation(severity).colour


def style_of(severity: str) -> tuple[str, str]:
    """The ``(colour, icon)`` pair the TitaniumCloud panels head a report with."""
    presentation = _presentation(severity)
    return presentation.colour, presentation.icon


def rank_of(severity: str) -> int:
    """The sort key that puts the worst verdict first.

    A table sorted by it puts the payload inside a dropper at the top
    rather than wherever the archive happened to hold it.
    """
    return severity_rank(severity)
