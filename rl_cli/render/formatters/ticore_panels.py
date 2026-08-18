"""The TitaniumCore sections of an A1000 report: story, behaviour, ATT&CK, tags.

Every function writes to the ``Console`` it is given and returns nothing.
Shared by ``a1000_sample_report``'s detailed report and its TitaniumCore
report, which draw the same document from two endpoints.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rl_cli.models.shapes import count, dict_rows, list_section, mapping
from rl_cli.render.formatters.panels import add_capped_rows, more_note, stated_cell, stated_field
from rl_cli.render.markup import safe
from rl_cli.text import sanitize

# How much of each section a panel shows before it stops being a summary.
# The full document is one ``--json`` away, and these panels sit above the
# tag panels in a report an analyst reads top to bottom.
_BULLET_LIMIT = 10
_INDICATOR_LIMIT = 10
_SIGNATURE_LIMIT = 10
_ATTACK_ROW_LIMIT = 15

# The tag panels are lists of one word each, so they fit fewer lines before
# they stop earning their space — except the catch-all, which is the only
# home for tags no prefixed panel claims.
_TAG_LIMIT = 8
_OTHER_TAG_LIMIT = 10

# The tag prefixes that each get a panel of their own, and so are the ones
# the catch-all panel leaves out. A prefix with no panel — "crypto-", for
# one — must not appear here, or its tags are rendered nowhere at all.
_PANELLED_TAG_PREFIXES = ("capability-", "indicator-", "protection-")


def bullet_items(value: Any) -> list[Any]:
    """A TitaniumCore section as the list of bullets a panel draws.

    ``behaviour`` is an object in the real document, not a list, so gating
    the panel on the list spelling skips it silently every time. An object's
    list-valued members are its entries; a scalar member is rendered as
    "key: value" rather than dropped.

    This lives beside the panels rather than in ``models``: it does not read
    a shape, it decides what a section looks like as bullets.
    """
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        return []
    items: list[Any] = []
    for key, member in value.items():
        if isinstance(member, list):
            items.extend(member)
        elif member not in (None, "", {}):
            items.append(f"{key}: {member}")
    return items


def print_ticore_panels(data: dict[str, Any], console: Console) -> None:
    ticore = mapping(data.get("ticore"))
    _print_story_panel(ticore.get("story"), console)
    _print_bullet_panel(
        bullet_items(ticore.get("behaviour")),
        title="Behavioral Analysis",
        limit=_BULLET_LIMIT,
        label=_behaviour_label,
        noun="behaviours",
        console=console,
    )
    _print_indicator_panel(list_section(ticore, "indicators"), console)
    # "signatures" is not a TitaniumCore section — it is absent from the
    # SDK's ticore_fields — so it is read one level down, under
    # "certificate", rather than off ticore itself.
    certificate = mapping(ticore.get("certificate"))
    _print_bullet_panel(
        list_section(certificate, "signatures"),
        title="Matched Signatures",
        limit=_SIGNATURE_LIMIT,
        label=_signature_label,
        noun="signatures",
        console=console,
    )
    _print_attack_table(ticore.get("attack"), console)


def _print_story_panel(story: Any, console: Console) -> None:
    """TitaniumCore's plain-English narrative of what the sample is."""
    if isinstance(story, str) and story.strip():
        console.print(Panel(safe(story), title="Analysis Story", expand=False))


def _print_attack_table(attack: Any, console: Console) -> None:
    """The MITRE ATT&CK matrix TitaniumCore reports under ``attack``.

    Stated as a list of matrices, each holding tactics, each holding the
    techniques the sample was mapped to.
    """
    rows: list[tuple[str, str, str]] = []
    for matrix in dict_rows(attack if isinstance(attack, list) else []):
        for tactic in dict_rows(list_section(matrix, "tactics") or []):
            tactic_name = sanitize(stated_cell(tactic.get("name") or tactic.get("id")))
            for technique in dict_rows(list_section(tactic, "techniques") or []):
                rows.append(
                    (
                        tactic_name,
                        sanitize(stated_cell(technique.get("id"))),
                        sanitize(
                            stated_cell(technique.get("name") or technique.get("description"))
                        ),
                    )
                )
    if not rows:
        return

    table = Table(title="MITRE ATT&CK", show_header=True, header_style="bold magenta")
    table.add_column("Tactic", style="cyan")
    table.add_column("Technique", style="yellow")
    table.add_column("Name", style="white")
    add_capped_rows(table, rows, limit=_ATTACK_ROW_LIMIT, noun="techniques")
    console.print(table)


def _behaviour_label(item: Any) -> str:
    """One behaviour entry: whichever of its two names it states."""
    if not isinstance(item, dict):
        return str(item)
    return str(stated_field(item.get("name") or item.get("description")))


def _indicator_label(indicator: Any) -> str:
    """One indicator, prefixed with the priority TitaniumCore graded it."""
    if not isinstance(indicator, dict):
        return str(indicator)
    description = stated_field(indicator.get("description") or indicator.get("name"))
    priority = indicator.get("priority")
    return str(description) if priority is None else f"[{count(priority)}] {description}"


def _signature_label(signature: Any) -> str:
    """One matched signature, prefixed with the severity stated for it."""
    if not isinstance(signature, dict):
        return str(signature)
    name = stated_field(
        signature.get("name") or signature.get("signature") or signature.get("identifier")
    )
    severity = signature.get("severity", "")
    return f"[{severity}] {name}" if severity else str(name)


def _print_bullet_panel(
    items: list[Any] | None,
    *,
    title: str,
    limit: int | None,
    label: Callable[[Any], str],
    noun: str,
    console: Console,
) -> None:
    """One capped bullet panel, with the tail that says how much it hid.

    Every panel in this module goes through here, so a cut is never silent:
    a panel that stops at its limit in silence reads as the whole answer. The
    tail is ``more_note``, the same sentence the capped tables in this report
    write, and ``noun`` is required for the same reason it is there — "and 8
    more" of what is the whole question.

    ``limit=None`` is uncapped, and the only way to say so.

    The whole line is escaped, not just the label: a bracketed priority or
    severity is a prefix these panels draw, and Rich reads it as a style tag.
    """
    if not items:
        return
    shown = items if limit is None else items[:limit]
    bullets = [safe(f"• {label(item)}") for item in shown]
    if limit is not None and len(items) > limit:
        bullets.append(f"... {more_note(len(items) - limit, noun)}")
    console.print(Panel("\n".join(bullets), title=title, expand=False))


def _print_indicator_panel(indicators: list[Any] | None, console: Console) -> None:
    if not indicators:
        return
    # TitaniumCore grades an indicator with "priority" 0-10 and orders the
    # list by nothing in particular, so the first ten in payload order keep
    # priority-1 filler while pushing the priority-10 entries — "deletes
    # volume shadow copies", "encrypts files on all mounted drives" — off
    # the end. Stable, so equal priorities keep their payload order.
    ordered = sorted(
        indicators,
        key=lambda ind: count(ind.get("priority")) if isinstance(ind, dict) else 0,
        reverse=True,
    )
    _print_bullet_panel(
        ordered,
        title="Indicators",
        limit=_INDICATOR_LIMIT,
        label=_indicator_label,
        noun="indicators",
        console=console,
    )


def print_tag_panels(data: dict[str, Any], console: Console) -> None:
    # Coerced up front: every branch below tests a prefix and prints the
    # tag, and a tag is whatever TitaniumCore put in the list.
    ticore_tags = [sanitize(tag) for tag in list_section(mapping(data.get("tags")), "ticore") or []]
    if not ticore_tags:
        return

    capabilities = [t for t in ticore_tags if t.startswith("capability-")]
    indicators = [t for t in ticore_tags if t.startswith("indicator-")]
    protection = [t for t in ticore_tags if t.startswith("protection-")]
    other = [t for t in ticore_tags if not any(t.startswith(p) for p in _PANELLED_TAG_PREFIXES)]

    _print_bullet_panel(
        capabilities,
        title="Capabilities",
        limit=_TAG_LIMIT,
        label=lambda tag: tag.removeprefix("capability-"),
        noun="capabilities",
        console=console,
    )
    _print_bullet_panel(
        indicators,
        title="Indicator Types",
        limit=_TAG_LIMIT,
        label=lambda tag: tag.removeprefix("indicator-"),
        noun="indicator types",
        console=console,
    )
    _print_bullet_panel(
        protection,
        title="Security Features",
        # Uncapped: these name the hardening the binary was built with, a
        # small fixed set, and "the other 3" is not a useful thing to say
        # about ASLR, DEP and CFG.
        limit=None,
        label=lambda tag: tag.removeprefix("protection-").upper(),
        noun="security features",
        console=console,
    )
    _print_bullet_panel(
        other,
        title="Additional Tags",
        limit=_OTHER_TAG_LIMIT,
        label=str,
        noun="tags",
        console=console,
    )
