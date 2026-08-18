"""What ``rl-cli config`` draws: the availability report, and the profile list.

The profile list needs more than the ``(data, console)`` a renderer is
handed — it marks the active profile — so it takes that as an explicit
keyword argument and ``cli/commands/config.py`` binds it with
``functools.partial``. Nothing here reads a ``Settings`` or a
``click.Context``: the command resolves those and passes the facts.

The per-service lines draw the grade ``models.availability`` puts on a
probe, and do not grade one: which outcome is bad news is a decision about
the measurement, and deciding it here is deciding it twice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rich.console import Console
from rich.panel import Panel

from rl_cli.models.availability import SERVICES, ProbeGrade, grade_of
from rl_cli.render.markup import safe
from rl_cli.render.output import RichOutput


def print_availability_panel(payload: Mapping[str, Any], /, *, console: Console) -> None:
    """Head `check-access`'s report with the count and when it was measured.

    Reads the payload ``display`` hands over rather than the probe run it was
    built from, so this panel states exactly what ``-o json`` would have
    printed instead of it.
    """
    summary = payload["summary"]
    console.print(
        Panel(
            f"Services Available: {summary['services_available']}/{summary['services_total']}\n"
            f"Last Checked: {safe(payload['timestamp'])}",
            title="API Availability Status",
            style="blue",
        )
    )


def _report(output: RichOutput, status: str) -> Callable[[str], None]:
    """The line a graded probe is drawn as. One drawing per grade, and no fourth.

    ``problem`` and not ``error`` for bad news, and not because this probe
    is a special case: nothing in this package may reach the method that
    fails the run (``tests/test_formatters.py``). `check-access` is what
    the CLI tells an analyst to run when a service is unreachable, so
    saying so is the command working; ``problem`` draws the same red line
    and leaves the exit status alone.
    """
    return {
        ProbeGrade.GOOD: output.success,
        ProbeGrade.CAVEAT: output.warning,
        ProbeGrade.BAD: output.problem,
    }[grade_of(status)]


def print_service_lines(
    availability: Mapping[str, Any],
    /,
    *,
    output: RichOutput,
    detailed: bool = False,
    indent: str = "",
) -> None:
    """Say what each probe measured, one line per service, on the status console.

    Printed outside ``display`` by both commands that report a probe run, so
    these lines survive every output format: they are the human's reason, not
    the data. ``indent`` is the caller's own nesting — `config show -a` lists
    them under a heading.
    """
    for label, key in SERVICES:
        probe = availability[key]
        _report(output, probe["status"])(f"{indent}{label}: {probe['message']}")
        # Only what the probe measured: listing the API names the service
        # *has* would read as twelve working endpoints on the evidence of one
        # call.
        if detailed:
            output.info(f"  Credentials configured: {probe['credentials_configured']}")
            output.info(f"  Endpoint answered: {probe['api_accessible']}")


def print_profile_names(names: Sequence[str], /, *, output: RichOutput, current: str) -> None:
    """List the profiles on the status console, marking ``current``.

    Takes no console, and is handed to ``display`` through
    ``cli.context.without_console``: the marked names are chatter, and
    stdout carries the bare list so `-o json config list-profiles | jq -r
    '.[]'` still feeds a profile fan-out loop.
    """
    output.info("Available profiles:")
    for name in names:
        marker = " (current)" if name == current else ""
        output.info(f"  - {name}{marker}")
