"""Rendering helpers for TitaniumCloud service results.

Same contract as the sibling A1000 modules: write to the ``Console`` passed
in, never return formatted strings. There is no default console — falling
back to a fresh one ignores ``color: false`` in every renderer the CLI did
not hand its own console to.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from rl_cli.models.payload import (
    SampleFacts,
    SampleVerdict,
    ScannerConsensus,
    unwrap_envelope,
)
from rl_cli.render.formatters.panels import (
    add_capped_rows,
    field_panel,
    print_file_information,
    ratio_text,
    stated_cell,
    stated_field,
)
from rl_cli.render.formatters.severity import colour_of, style_of
from rl_cli.text import sanitize

# How many of an ``rl.sample`` record's sightings the sources table shows;
# the rest are in the payload ``-o json`` prints.
_SOURCE_ROW_LIMIT = 10


def print_file_analysis(result: dict[str, Any], console: Console) -> None:
    """Render a file reputation/analysis payload as panels and tables."""
    # TitaniumCloud answers the reputation query wrapped in
    # {"rl": {"malware_presence": {...}}}; read through it so the fields
    # below are looked up where they actually are.
    data = unwrap_envelope(result)

    # TitaniumCloud reports MALICIOUS/KNOWN in upper case and A1000 in
    # lower; the verdict folds both, and one walk answers the icon, the
    # colour, the threat name and the level below.
    verdict = SampleVerdict.of(data)
    status_color, status_icon = style_of(verdict.severity)

    console.print(f"\n{status_icon} File Analysis Result\n", style=f"bold {status_color}")

    print_file_information(SampleFacts.of(data), console=console, style="blue")
    _print_threat_assessment(data, verdict, style=status_color, console=console)
    _print_scanner_consensus(data, console)
    _print_sources_table(data, console)


def _print_threat_assessment(
    data: dict[str, Any], verdict: SampleVerdict, *, style: str, console: Console
) -> None:
    """What the record says about the threat, when it says anything.

    Gated on the verdict rather than the literal ``threat_name`` key, which
    skips the whole panel for the payloads that spell the name
    ``classification_result`` or ``classification.family_name``.
    """
    if not verdict.threat_name and verdict.threat_level is None:
        return

    fields: dict[str, Any] = {
        "Threat": stated_field(verdict.threat_name),
        "Level": stated_field(verdict.threat_level),
    }
    # "threat_score" is not a field of malware_presence, so a row for it is
    # structurally always N/A. These are the fields the record does carry,
    # each added only when present.
    for label, key in (
        ("Reason", "reason"),
        ("Trust Factor", "trust_factor"),
        ("First Seen", "first_seen"),
        ("Last Seen", "last_seen"),
    ):
        if data.get(key) is not None:
            fields[label] = data[key]
    field_panel(fields, title="Threat Assessment", console=console, style=style)


def _print_scanner_consensus(data: dict[str, Any], console: Console) -> None:
    """The AV consensus TCA-0101 exists to report.

    "31 of 37 engines agree" is the whole value of the query, and the record
    states it in every ``malware_presence`` answer.
    """
    consensus = ScannerConsensus.of(data)
    # Zero engines is no AV panel: ``scanner_count: 0`` says none ran, and
    # a ratio about that reports nothing. But half a ratio IS a count the
    # record stated, so ``0/?`` -- no engine flagged it, out of a number we
    # were not told -- keeps its panel, and a signature ratio of 0/0 over a
    # next-gen array is engines having run.
    if consensus.unreported:
        return
    fields = {"Detections": ratio_text(consensus)}
    # The ratio above counts signature engines only, so a record that also
    # carries a next-gen array has verdicts this panel would otherwise not
    # account for anywhere -- the reader left to assume the ML engines are
    # among the engines counted, which they are not. A malware_presence
    # record states no scanner arrays, so this row is drawn for the shapes
    # that carry them and absent from the rest.
    if consensus.nextgen_total:
        fields["Next-Gen (ML)"] = f"{consensus.nextgen_detected}/{consensus.nextgen_total}"
    field_panel(
        fields,
        title="AV Scanners",
        console=console,
        style="blue",
        # The same grade its A1000 sibling paints: an unpainted ratio is this
        # report keeping to itself what the other one shouts.
        colours={"Detections": colour_of(consensus.consensus_severity)},
    )


def _print_sources_table(data: dict[str, Any], console: Console) -> None:
    """Where the sample was seen: an ``rl.sample`` record lists the download
    URL and domain under ``sources``, which are first-class IOCs.
    """
    sources = data.get("sources")
    rows = [
        (
            sanitize(stated_cell(entry.get("url") or entry.get("uri"))),
            sanitize(stated_cell(entry.get("domain"))),
            sanitize(stated_cell(entry.get("record_time") or entry.get("tasked_on"))),
        )
        for entry in (sources if isinstance(sources, list) else [])
        if isinstance(entry, dict) and (entry.get("url") or entry.get("uri") or entry.get("domain"))
    ]
    if not rows:
        return

    table = Table(title="Sources", show_header=True, header_style="bold magenta")
    table.add_column("URL", style="yellow", overflow="fold")
    table.add_column("Domain", style="cyan")
    table.add_column("Seen", style="white")
    add_capped_rows(table, rows, limit=_SOURCE_ROW_LIMIT, noun="sources")
    console.print(table)
