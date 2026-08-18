"""The labelled-field panels every report in this package draws, and the
vocabulary every one of them shares.

``SampleFacts.of`` normalises the two services' spellings; the drawing
here takes an already-normalised record and knows nothing about either
service's keys.

Three decisions are made here for the whole package, because a report that
makes them twice makes them differently: what a field the payload did not
state reads as (:func:`stated_field`, :func:`stated_cell`), what a
truncated listing says it hid (:func:`more_note`), and what an AV ratio
reads as (:func:`ratio_text`).

Values arrive raw and are escaped here, at the point they meet Rich.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_cli.models.payload import SampleFacts, ScannerConsensus
from rl_cli.render.markup import safe
from rl_cli.text import byte_size, sanitize

# A sample the appliance has seen under fifty names needs the first few and
# a count, not fifty lines in the panel that is meant to identify it.
_ALIAS_LIMIT = 5

# What Rich means by "no panel border style of its own".
_NO_STYLE = "none"

# What a report says where the payload said nothing, in the one place that
# decides it. A panel field is a line of its own and has room for the
# phrase; a table cell is one of six across a line and gets the mark.
_ABSENT_IN_PANEL = "N/A"
_ABSENT_IN_CELL = "-"

# What either half of an AV ratio reads as where nothing states it: a
# number we were not told is not a zero. See :func:`ratio_text`.
_UNSTATED_HALF = "?"


def _stated(value: Any, absent: str) -> Any:
    """``value``, or ``absent`` where the payload stated nothing.

    Against ``None`` and ``""`` rather than against falsiness: a risk score
    of 0 is a score, and a zero-byte file is a size.

    ``data.get(key, ...)`` cannot stand in for this: its default fires only
    when the key is *absent*, while both APIs spell an unset field as an
    explicit JSON null or as ``""``.
    """
    return absent if value is None or value == "" else value


def stated_field(value: Any) -> Any:
    """``value``, or ``"N/A"`` for a panel field the payload did not state."""
    return _stated(value, _ABSENT_IN_PANEL)


def stated_cell(value: Any) -> Any:
    """``value``, or ``"-"`` for a table cell the payload did not state."""
    return _stated(value, _ABSENT_IN_CELL)


def more_note(hidden: int, noun: str) -> str:
    """What a listing that stopped short says it left out.

    One sentence for every cap in the package, tables and panels alike, so
    that a report states its truncation one way. The noun is not optional:
    "and 8 more" of what is the whole question.
    """
    return f"and {hidden} more {noun}"


def ratio_text(consensus: ScannerConsensus) -> str:
    """The AV ratio both reports draw, with the share where it is stated.

    Sanitized, not escaped: the caller either interpolates this into its own
    markup, escaping it there, or hands it to :func:`field_panel`, which
    escapes every value it draws.

    A record that states no share gets no parenthesis rather than a computed
    one — the tool reports what the appliance said.

    ``"?"`` for a half the record states nowhere, never a default: "0/37"
    would say no engine flagged the file and "13/13" would say every engine
    that ran did, on a record that stated neither.
    """
    detected = _UNSTATED_HALF if consensus.detected is None else sanitize(consensus.detected)
    total = _UNSTATED_HALF if consensus.total is None else sanitize(consensus.total)
    ratio = f"{detected}/{total}"
    if consensus.percent is not None:
        ratio += f" ({sanitize(consensus.percent)}%)"
    return ratio


def field_panel(
    fields: Mapping[str, object],
    *,
    title: str,
    console: Console,
    style: str = _NO_STYLE,
    colours: Mapping[str, str] | None = None,
) -> None:
    """Draw ``fields`` as one ``Label: value`` line each.

    ``colours`` paints individual values, keyed by the same labels as
    ``fields`` so that a caller cannot colour a row it did not add.
    """
    painted = colours or {}
    lines = []
    for label, value in fields.items():
        colour = painted.get(label)
        rendered = f"[{colour}]{safe(value)}[/{colour}]" if colour else safe(value)
        lines.append(f"[cyan]{label}:[/cyan] {rendered}")
    console.print(Panel("\n".join(lines), title=title, expand=False, style=style))


def _escaped_cells(row: Iterable[Any]) -> tuple[str, ...]:
    """Every cell of ``row`` escaped at the point it meets Rich."""
    return tuple(safe(cell) for cell in row)


def add_capped_rows(
    table: Table,
    rows: Sequence[Any],
    *,
    limit: int | None,
    noun: str,
    render: Callable[[Any], Iterable[str]] = _escaped_cells,
) -> None:
    """Add the first ``limit`` rows, then a tail stating what was left out.

    Every capped table in this package says how much it hid — a table that
    stops in silence reads as the whole answer. The tail marks the cut in
    the first column and states the count in the last.

    ``limit=None`` is uncapped, and is how a caller says so: a limit that
    happens to equal the row count means "no limit" only by coincidence.

    ``render`` turns one row into its cells, and is where a cell carrying
    markup of its own is built; by default each cell is escaped as it is.
    """
    for row in rows if limit is None else rows[:limit]:
        table.add_row(*render(row))
    if limit is not None and len(rows) > limit:
        columns = len(table.columns)
        tail = more_note(len(rows) - limit, noun)
        # One cell per column, always: Rich does not report extra cells as
        # an error, it appends a blank-headed column for them.
        if columns < 2:
            table.add_row(f"... {tail}")
        else:
            table.add_row("...", *[""] * (columns - 2), tail)


def print_file_information(facts: SampleFacts, *, console: Console, style: str = _NO_STYLE) -> None:
    """Draw the "File Information" panel both reports share.

    A field the service did not state is left out rather than shown as N/A —
    a TitaniumCloud reputation record carries the queried hash and nothing
    else, and four N/A rows say less than no rows at all. A record stating
    none of them gets no panel.
    """
    stated = {
        "File Name": facts.name,
        "MD5": facts.md5,
        "SHA1": facts.sha1,
        "SHA256": facts.sha256,
        "File Type": facts.file_type,
        # Every A1000 table prints a size with its unit, so this one does too.
        "File Size": "" if facts.size is None else stated_field(byte_size(facts.size)),
    }
    fields: dict[str, object] = {label: value for label, value in stated.items() if value}
    # TitaniumCloud states no aliases at all, so the row appears only where
    # the appliance answered with some.
    if facts.aliases:
        shown = ", ".join(facts.aliases[:_ALIAS_LIMIT])
        extra = len(facts.aliases) - _ALIAS_LIMIT
        fields["Aliases"] = f"{shown} (and {extra} more)" if extra > 0 else shown
    if not fields:
        return
    field_panel(fields, title="File Information", console=console, style=style)
