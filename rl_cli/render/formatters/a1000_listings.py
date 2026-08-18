"""The A1000 renderings of many samples at once: extracted files, search, listings.

Every function writes to the ``Console`` it is given, and every one that
caps what it draws returns how many rows it actually drew — the count its
command turns into "Showing N of M", so the analyst is never shown a
truncated listing that looks complete.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_cli.models.payload import SampleFacts, SampleVerdict, unwrap_envelope
from rl_cli.models.shapes import count, dict_rows
from rl_cli.render.formatters.panels import stated_cell, stated_field
from rl_cli.render.formatters.severity import colour_of, rank_of
from rl_cli.render.markup import safe
from rl_cli.text import byte_count, byte_size, clip, digest_cell, sanitize

# Column widths for the search table. A file type is a sentence
# ("PE32 executable (GUI) Intel 80386") and a threat name is dotted, so
# both need a cap that leaves room for the columns beside them.
_SEARCH_TYPE_WIDTH = 15
_SEARCH_THREAT_WIDTH = 20

# Per-sample panels are drawn at a fixed width, so their two free-text
# fields get caps of their own.
_PANEL_NAME_WIDTH = 30
_PANEL_TYPE_WIDTH = 20
_PANEL_WIDTH = 120

# How many extracted files the table draws for a caller that binds no cap:
# an installer carrying four thousand of them would otherwise push the
# sample's own report out of the scrollback.
_EXTRACTED_ROW_LIMIT = 20

# The other two defaults in this module, and each is its own number rather
# than one policy spelled three times: a screenful of panels holds a tenth
# of what a screenful of table rows does.
_SEARCH_ROW_LIMIT = 20
_SAMPLE_PANEL_LIMIT = 10


class _ExtractedFile(NamedTuple):
    """One extracted-file entry, graded once."""

    verdict: SampleVerdict
    meta: dict[str, Any]

    @classmethod
    def of(cls, entry: dict[str, Any]) -> _ExtractedFile:
        """Grade one entry, reading its shape the way every other reader does.

        The path sits at the top level and most file metadata one level
        down under "sample" — but not all of it: an entry can state the
        verdict on either half. ``unwrap_envelope`` must stay the only
        reader of that shape, or the same Conti DLL comes out "unknown" on
        screen and level "error" in the SARIF log.
        """
        meta = unwrap_envelope(entry)
        return cls(SampleVerdict.of(meta), meta)


def _graded_extracted(files: list[dict[str, Any]]) -> list[_ExtractedFile]:
    """Every extracted file with its verdict, worst first.

    The verdict is read once per row and carried from here: grading inside
    the sort key re-walks the same record on every comparison, and the row
    then walks it twice more for its colour and its text.

    This is the command for finding the payload inside a dropper, so the
    order is worst verdict first and highest risk score within a verdict
    rather than whatever puts a Conti DLL next to an icon.
    """
    rows = [_ExtractedFile.of(entry) for entry in dict_rows(files)]
    rows.sort(key=lambda row: (rank_of(row.verdict.severity), -count(row.meta.get("riskscore"))))
    return rows


def print_extracted_files_table(
    files: list[dict[str, Any]], *, max_rows: int = _EXTRACTED_ROW_LIMIT, console: Console
) -> int:
    """Render the worst ``max_rows`` extracted files, worst verdict first.

    Returns the number of rows drawn, which is how every capped listing in
    this package reports a cut: the command says "Showing N of M; use -o
    json for the full set" over it. The count is not
    ``min(max_rows, len(files))`` — ``_graded_extracted`` drops the entries
    this table cannot render.
    """
    table = Table(title="Extracted Files", show_header=True)
    table.add_column("Index", style="cyan")
    table.add_column("Filename", style="yellow")
    table.add_column("Classification")
    table.add_column("SHA256", style="white")
    table.add_column("File Type", style="green")
    table.add_column("Size", style="magenta")

    rows = _graded_extracted(files)[:max_rows]
    for idx, row in enumerate(rows, 1):
        facts = SampleFacts.of(row.meta)
        colour = colour_of(row.verdict.severity)
        table.add_row(
            str(idx),
            safe(stated_cell(facts.name)),
            f"[{colour}]{safe(row.verdict.classification or 'unknown')}[/{colour}]",
            safe(stated_cell(facts.digest)),
            # ``type_display`` is the extracted-file entry's own spelling —
            # "Text/None" where the record says "Text" — and it is the one
            # the appliance's UI shows for a carved file.
            safe(stated_cell(row.meta.get("type_display") or facts.file_type)),
            # Through ``stated_cell`` like every other cell on the line: a
            # size the entry does not state is one of six absences in this
            # row, and it read ``N/A`` while the five beside it read ``-``.
            safe(stated_cell(byte_size(facts.size))),
        )
    console.print(table)
    return len(rows)


class _SearchRow(NamedTuple):
    """One search hit, reduced to the five cells the table draws."""

    digest: str
    file_type: str
    classification: str
    threat: str
    size: str


def _graded_search(results: list[dict[str, Any]]) -> list[tuple[SampleVerdict, dict[str, Any]]]:
    """Every search hit with its verdict, worst first.

    Sorted rather than taking the appliance's order: a page of twenty
    goodware hits with one Conti sample last draws twenty green rows on
    screen while ``-o sarif`` over the same list emits an ``error`` — the
    screen saying clean and the CI gate saying error about one answer.

    Graded here and carried, not re-read inside the sort key: the row draws
    from this verdict, so a hit still costs one walk of its payload.
    """
    rows = [(SampleVerdict.of(entry), entry) for entry in dict_rows(results)]
    # Stable, so hits of equal severity keep the order the appliance
    # ranked them in.
    rows.sort(key=lambda row: rank_of(row[0].severity))
    return rows


def _search_row(verdict: SampleVerdict, result: dict[str, Any]) -> _SearchRow:
    """One search entry as table cells: one walk of the payload, five strings."""
    # Always emit the colour: an unmarked cell inherits the column style,
    # which paints an unknown verdict red.
    colour = colour_of(verdict.severity)
    facts = SampleFacts.of(result)

    # Not every search entry carries a SHA256; the digest falls back rather
    # than render a truncation-suffixed "N/A...".
    return _SearchRow(
        digest=safe(stated_cell(digest_cell(facts.digest or ""))),
        file_type=safe(clip(sanitize(stated_cell(facts.file_type)), _SEARCH_TYPE_WIDTH)),
        classification=f"[{colour}]{safe(verdict.classification or 'unknown')}[/{colour}]",
        threat=safe(clip(sanitize(stated_cell(verdict.threat_name)), _SEARCH_THREAT_WIDTH)),
        # An absent size is not a zero-byte file, and the column carries no
        # unit of its own, so a bare 0 read as one. ``byte_count`` is the
        # unit-less half, for its tolerance of the string a JSON payload
        # may hold; the unit is in the column header, and a value it cannot
        # read is an absent cell like any other.
        size=safe(stated_cell(byte_count(facts.size))),
    )


def print_search_results_table(
    results: list[dict[str, Any]], *, max_rows: int = _SEARCH_ROW_LIMIT, console: Console
) -> int:
    """Render the worst ``max_rows`` search entries, worst verdict first.

    The count is not ``min(max_rows, len(results))``: ``dict_rows`` drops
    the entries this table cannot render, so the caller's "showing N of M"
    note has to hear the drawn number rather than assume the cap.
    """
    table = Table(title="Search Results", show_header=True)
    # ``no_wrap`` only because ``_search_row`` clips this cell to
    # ``DIGEST_CELL_WIDTH``: a no_wrap column holding a full 64-char SHA256
    # claims the whole width of an 80-column terminal and squeezes its
    # siblings down to blank headers.
    table.add_column("SHA256", style="yellow", no_wrap=True)
    table.add_column("Type", style="cyan")
    table.add_column("Classification")
    table.add_column("Threat", style="magenta")
    table.add_column("Size (bytes)", style="green")

    rows = _graded_search(results)[:max_rows]
    for verdict, result in rows:
        table.add_row(*_search_row(verdict, result))
    console.print(table)
    return len(rows)


def _sample_panel_text(verdict: SampleVerdict, sample: dict[str, Any]) -> str:
    """One search entry as the body of its own panel.

    Takes a whole entry: shortening a name to what fits a panel and standing
    a word in for a field the entry does not carry is this layer's job, not
    the service's, which also feeds ``-o json`` and ``-o sarif``.

    The verdict is handed in rather than read here: the listing already
    grades every entry to sort it worst-first, and grading again inside the
    panel walks each payload twice. ``tests/test_payload.py`` meters that.
    """
    colour = colour_of(verdict.severity)
    facts = SampleFacts.of(sample)

    # Clipped, then escaped: a bare slice is the exact truncation ``clip``
    # exists to mark, and slicing after safe() could cut an escape sequence
    # in half and leave the markup it was escaping unbalanced.
    file_name = clip(sanitize(facts.name or ""), _PANEL_NAME_WIDTH)
    file_type = clip(sanitize(facts.file_type or ""), _PANEL_TYPE_WIDTH)
    return (
        f"[cyan]File:[/cyan] {safe(stated_field(file_name))}\n"
        f"[cyan]SHA256:[/cyan] [yellow]{safe(stated_field(facts.sha256))}[/yellow]\n"
        f"[cyan]SHA1:[/cyan] [yellow]{safe(stated_field(facts.sha1))}[/yellow]\n"
        f"[cyan]MD5:[/cyan] [yellow]{safe(stated_field(facts.md5))}[/yellow]\n"
        f"[cyan]Status:[/cyan] "
        f"[{colour}]{safe(stated_field(verdict.classification))}[/{colour}]\n"
        f"[cyan]Threat:[/cyan] [red]{safe(stated_field(verdict.threat_name))}[/red]\n"
        f"[cyan]Type:[/cyan] {safe(stated_field(file_type))}\n"
        f"[cyan]Size:[/cyan] {safe(stated_field(byte_size(facts.size)))}"
    )


def print_samples_panels(
    samples: list[dict[str, Any]], *, max_panels: int = _SAMPLE_PANEL_LIMIT, console: Console
) -> int:
    """Render up to ``max_panels`` samples as one panel each, worst first.

    Returns the number of panels actually rendered so the caller can decide
    whether to print a "and N more" tail.

    Sorted for the reason its two sibling listings are: a truncated listing
    has to lose the least interesting entries, not the arbitrary ones, and
    the appliance's own order puts the one malicious sample last.
    """
    graded = _graded_search(samples)
    shown = graded[:max_panels]
    for index, (verdict, sample) in enumerate(shown, 1):
        console.print(
            Panel(
                _sample_panel_text(verdict, sample),
                title=f"Sample {index}/{len(graded)}",
                expand=True,
                width=_PANEL_WIDTH,
            )
        )
        if index < len(shown):
            console.print()
    return len(shown)
