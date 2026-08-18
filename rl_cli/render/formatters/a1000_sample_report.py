"""The per-sample A1000 reports: status, summary, detailed report, TitaniumCore.

Every function writes to the ``Console`` it is given and returns nothing —
callers do not ``print()`` a result. Keep all Rich imports in this package
so the CLI layer stays presentation-free.
"""

from __future__ import annotations

import re
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_cli.models.payload import (
    SampleFacts,
    SampleVerdict,
    ScannerConsensus,
    ScannerDetection,
    ticore_record,
    unwrap_envelope,
)
from rl_cli.models.shapes import dict_rows, mapping
from rl_cli.render.formatters.panels import (
    add_capped_rows,
    field_panel,
    print_file_information,
    ratio_text,
    stated_cell,
    stated_field,
)
from rl_cli.render.formatters.severity import colour_of, rank_of
from rl_cli.render.formatters.ticore_panels import print_tag_panels, print_ticore_panels
from rl_cli.render.markup import safe
from rl_cli.text import sanitize

# The keys a threat-intelligence entry states its indicator under, in the
# order a report should prefer them.
_IOC_KEYS = ("ipv4", "ipv6", "ip", "url", "uri", "domain", "hostname", "host", "email")

# How much of a listing each section shows before the panel stops being a
# summary; the whole document is one ``--json`` away.
_DETECTION_ROW_LIMIT = 10
_IOC_ROW_LIMIT = 15

# "YYYY-MM-DDTHH:MM:SS" — the appliance states these timestamps with
# microseconds, which is more precision than a metadata row needs and
# enough to wrap the panel. The zone is kept: see ``_to_the_second``.
_TIMESTAMP_WIDTH = 19

# The zone a timestamp ends with, as ``Z`` or as an offset from UTC.
_TIMESTAMP_ZONE = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")

# What tells a next-gen engine's row apart from a signature engine's in the
# detections table.
_NEXTGEN_SUFFIX = " (NG)"


def print_analysis_status(result: dict[str, Any], console: Console) -> None:
    """Render the ``a1000 status`` panel for a sample classification."""
    verdict = SampleVerdict.of(result)
    field_panel(
        {
            "Classification": stated_field(verdict.classification),
            # The record carries the name in classification_result, and a
            # verdict that never names the malware is half an answer.
            "Threat Name": stated_field(verdict.threat_name),
            "Risk Score": stated_field(result.get("riskscore")),
            "First Seen": stated_field(result.get("first_seen")),
            "Last Seen": stated_field(result.get("last_seen")),
            "Data Source": stated_field(result.get("data_source")),
        },
        title="Analysis Status",
        console=console,
    )


def print_summary_panel(result: dict[str, Any], console: Console) -> None:
    """Render the compact summary panel for ``a1000 summary-report``."""
    data = unwrap_envelope(result)
    verdict = SampleVerdict.of(data)
    field_panel(
        {
            "Classification": stated_field(verdict.classification),
            "Threat Name": stated_field(verdict.threat_name),
            "Risk Score": stated_field(data.get("riskscore")),
            "File Type": stated_field(data.get("file_type")),
        },
        title="Summary Report",
        console=console,
    )


def print_report_summary(result: dict[str, Any], console: Console) -> None:
    """Render the multi-panel detailed report for ``a1000 report``."""
    data = unwrap_envelope(result)

    print_file_information(SampleFacts.of(data), console=console)
    _print_threat_assessment_panel(data, SampleVerdict.of(data), console)
    _print_av_summary(data, console)
    _print_metadata_panel(data, console)
    _print_ioc_table(data, console)
    print_ticore_panels(data, console)
    print_tag_panels(data, console)

    console.print("\n[dim]For full detailed report, use --json flag[/dim]")


def print_titanium_report(result: dict[str, Any], console: Console) -> None:
    """Render the TitaniumCore document for ``a1000 titanium-report``.

    ``/api/v2/samples/{hash}/ticore/`` answers the document itself, and
    ``ticore_record`` is what turns either of its two nestings into the flat
    record these panels draw — the shape adaptation between two endpoints is
    the payload layer's, not the renderer's.
    """
    data = ticore_record(result)

    print_file_information(SampleFacts.of(data), console=console)
    _print_threat_assessment_panel(data, SampleVerdict.of(data), console)
    print_ticore_panels(data, console)
    print_tag_panels(data, console)

    console.print("\n[dim]For the full TitaniumCore report, use --json[/dim]")


def _platform_of(data: dict[str, Any], verdict: SampleVerdict) -> str:
    """The platform the sample targets, stated or read off the threat name.

    A1000 v2 records carry the threat name as a dotted
    "Platform.Type.Family" string in ``classification_result`` and state no
    platform field of their own.
    """
    platform = mapping(data.get("classification")).get("platform")
    if not platform and verdict.threat_name and "." in verdict.threat_name:
        platform = verdict.threat_name.split(".", 1)[0]
    return str(stated_field(platform or None))


def _print_threat_assessment_panel(
    data: dict[str, Any], verdict: SampleVerdict, console: Console
) -> None:
    threat_info: dict[str, Any] = {
        "Classification": stated_field(verdict.classification),
        "Threat Name": stated_field(verdict.threat_name),
        "Platform": _platform_of(data, verdict),
        # ``SampleVerdict.of`` already falls back to the TitaniumCore
        # classification factor, so 0 arrives here as the level it is.
        "Threat Level": stated_field(verdict.threat_level),
        "Trust Factor": stated_field(data.get("trust_factor")),
    }

    # "Why is this malicious", which the report requests. Added only when
    # present so a TitaniumCore document, which states none of them, does
    # not grow three N/A rows.
    for label, field in (
        ("Reason", "classification_reason"),
        ("Origin", "classification_origin"),
        ("Source", "classification_source"),
    ):
        if data.get(field):
            threat_info[label] = data[field]

    field_panel(
        threat_info,
        title="Threat Assessment",
        console=console,
        colours={"Classification": colour_of(verdict.severity)},
    )


def _print_ratio_panel(consensus: ScannerConsensus, console: Console) -> None:
    """Draw the ratio, in the words and the colour its sibling report uses.

    ``ratio_text`` is what it reads as and ``consensus_severity`` is how it
    grades, so ``a1000 report`` and ``ticloud file-analysis`` cannot draw the
    same 13/17 red in one and white in the other.

    The words include the share the record states, which this report used to
    drop: see :func:`ratio_text`, which decides that for both of them.
    """
    ratio_colour = colour_of(consensus.consensus_severity)
    ratio = safe(ratio_text(consensus))
    lines = [
        f"[cyan]Detection Ratio:[/cyan] "
        f"[{ratio_colour}]{ratio}[/{ratio_colour}] "
        f"AV scanners flagged this file"
    ]
    # Next-gen engines answer with an ML confidence verdict, not a
    # signature hit; folding them into the ratio above overstates it.
    if consensus.nextgen_total:
        lines.append(
            f"[cyan]Next-Gen (ML):[/cyan] "
            f"{consensus.nextgen_detected}/{consensus.nextgen_total} "
            f"engines returned a malicious confidence verdict"
        )
    console.print(Panel("\n".join(lines), title="AV Scanner Summary", expand=False))


def _detection_cells(hit: ScannerDetection) -> tuple[str, ...]:
    """One detections row, marked where the engine is a next-gen one.

    The payload states no name for some engines and this is the layer that
    says so, exactly as it is the layer that escapes both fields.
    """
    name = f"{stated_cell(hit.name)}{_NEXTGEN_SUFFIX if hit.nextgen else ''}"
    return (safe(name), safe(hit.result))


def _print_detections_table(detections: tuple[ScannerDetection, ...], console: Console) -> None:
    if not detections:
        return
    # Titled for what it is: the payload order this slices is ranked by
    # nothing, so the first ten are not the "top" ten.
    table = Table(title="AV Detections", show_header=True, header_style="bold magenta")
    table.add_column("Scanner", style="cyan")
    table.add_column("Detection", style="red")
    add_capped_rows(
        table,
        detections,
        limit=_DETECTION_ROW_LIMIT,
        noun="detections",
        render=_detection_cells,
    )
    console.print(table)


def _print_av_summary(data: dict[str, Any], console: Console) -> None:
    consensus = ScannerConsensus.of(data)
    # Zero engines is no AV panel: ``scanner_count: 0`` says none ran, and
    # a ratio about that reports nothing. But half a ratio IS a count the
    # record stated, so ``0/?`` -- no engine flagged it, out of a number we
    # were not told -- keeps its panel, and a signature ratio of 0/0 over a
    # next-gen array is engines having run.
    if consensus.unreported:
        return
    _print_ratio_panel(consensus, console)
    _print_detections_table(consensus.detections, console)


def _platform_row(file_type: str) -> str | None:
    """The OS and word size a file type implies, or ``None`` for neither.

    TitaniumCore spells the Apple format "MachO", with no hyphen, so the
    hyphenated test alone leaves a macOS sample with no Platform row.
    """
    if "ELF" in file_type:
        arch = "x86-64" if "64" in file_type else ("x86" if "32" in file_type else "Unknown")
        return f"Linux/Unix ({arch})"
    if "PE" in file_type:
        return "Windows (x64)" if ("PE+" in file_type or "64" in file_type) else "Windows (x86)"
    if "Mach-O" in file_type or "MachO" in file_type:
        return "macOS"
    return None


def _to_the_second(stated: str) -> str:
    """``stated`` without its sub-second digits, keeping the zone it names.

    Cutting the string to a width alone drops the trailing ``Z`` or
    ``+05:30`` while keeping the wall clock, so a UTC timestamp reads as a
    local one — the report telling an analyst building a timeline the wrong
    hour. Nothing here parses or converts the value.
    """
    zone = _TIMESTAMP_ZONE.search(stated)
    return stated[:_TIMESTAMP_WIDTH] + (zone.group() if zone else "")


def _print_metadata_panel(data: dict[str, Any], console: Console) -> None:
    # Coerced, not trusted: a numeric file_type makes the "in" tests inside
    # _platform_row raise before any of this panel reaches the screen.
    file_type = sanitize(data.get("file_type") or "")
    metadata: dict[str, Any] = {}

    platform = _platform_row(file_type)
    if platform:
        metadata["Platform"] = platform

    metadata["Category"] = stated_field(data.get("category"))
    metadata["Risk Score"] = stated_field(data.get("riskscore"))

    if data.get("extracted_file_count") is not None:
        metadata["Extracted Files"] = data["extracted_file_count"]

    if "PE" in file_type and data.get("imphash"):
        metadata["Imphash"] = data["imphash"]

    for label, field in (("First Seen", "local_first_seen"), ("Last Seen", "local_last_seen")):
        seen = data.get(field)
        metadata[label] = _to_the_second(sanitize(seen)) if seen else stated_field(None)

    field_panel(metadata, title="File Metadata", console=console)


def _indicator_cells(row: tuple[str, str, SampleVerdict]) -> tuple[str, ...]:
    """One network-indicator row, painted by the verdict the entry carries."""
    kind, value, verdict = row
    colour = colour_of(verdict.severity)
    return (
        safe(kind),
        safe(value),
        f"[{colour}]{safe(verdict.classification or 'unknown')}[/{colour}]",
    )


def _print_ioc_table(data: dict[str, Any], console: Console) -> None:
    """The C2 addresses the two threat-intelligence sections carry.

    ``networkthreatintelligence`` and ``domainthreatintelligence`` are
    requested from the appliance on every report and hold the first-class
    IOCs an analyst pivots on. Both state their entries as lists keyed by
    indicator kind.
    """
    # One walk of the entry, not one for the colour, one for the text and
    # one for the order: an indicator carries its own verdict, and the sort
    # below and both coloured columns read the same grading.
    rows: list[tuple[str, str, SampleVerdict]] = []
    for field in ("networkthreatintelligence", "domainthreatintelligence"):
        for kind, entries in mapping(data.get(field)).items():
            if not isinstance(entries, list):
                continue
            for entry in dict_rows(entries):
                value = next((entry[key] for key in _IOC_KEYS if entry.get(key)), None)
                if value:
                    rows.append((sanitize(kind), sanitize(value), SampleVerdict.of(entry)))
    if not rows:
        return

    # Worst verdict first, exactly as the extracted-files table orders its
    # own cap: in the appliance's order twenty known CDN domains fill the
    # table and the one malicious C2 — last in the payload — ends up inside
    # "and 6 more". The sort is stable, so indicators of equal severity keep
    # the order the appliance stated them in.
    rows.sort(key=lambda row: rank_of(row[2].severity))

    table = Table(title="Network Indicators", show_header=True, header_style="bold magenta")
    table.add_column("Type", style="cyan")
    table.add_column("Indicator", style="yellow")
    table.add_column("Classification")
    add_capped_rows(table, rows, limit=_IOC_ROW_LIMIT, noun="indicators", render=_indicator_cells)
    console.print(table)
