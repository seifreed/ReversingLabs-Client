"""Escape sanitized text for the Rich markup this tool renders.

Rich reads ``[...]`` as markup wherever it renders, so a threat name
containing ``[a-z0-9]`` silently loses it, and one containing ``[/]``
raises MarkupError part-way through a report that has already printed.
This is the Rich-facing half of ``rl_cli.text``, kept separate so
that sanitizing a string does not require importing Rich.
"""

from __future__ import annotations

from rich.markup import escape

from rl_cli.text import sanitize


def safe(value: object) -> str:
    """``sanitize`` plus Rich-markup escaping, for anything Rich renders.

    Use at every point where a value is interpolated into markup or passed
    to ``Table.add_row``/``Panel``; the escape is what keeps ``[a-z0-9]``
    in a YARA-derived name and stops ``[/]`` aborting the command.
    """
    return escape(sanitize(value))
