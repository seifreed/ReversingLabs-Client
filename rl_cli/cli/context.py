"""Shared scaffolding for command bodies: ``ctx.obj`` accessors and the call/report step."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import click
from rich.console import Console

from rl_cli.config import Settings
from rl_cli.render.output import OutputFormat, OutputFormatter, RichOutput
from rl_cli.services.a1000 import A1000Service, A1000Session
from rl_cli.services.titanium_cloud import TitaniumCloudApi


@dataclass(frozen=True, slots=True)
class CliContext:
    """Everything the root group resolved, as the one object it hands on.

    Frozen because it is the invocation's settled answer: the profile, the
    console policy and the connection are decided once in ``cli/main.py``,
    and a command that rebound one of them here would change them for
    whatever ran next in the same process.

    ``warn_if_unavailable`` is ``None`` for a subcommand that talks to no
    service — there is nothing to probe for ``config``.

    ``verbose`` is here rather than on ``settings.output`` because it is a
    fact about this invocation, not about the configuration; ``cli/main.py``
    states why that distinction has to hold.
    """

    settings: Settings
    output: RichOutput
    formatter: OutputFormatter
    session: A1000Session
    warn_if_unavailable: Callable[[], None] | None
    verbose: bool = False


def cli_context(ctx: click.Context) -> CliContext:
    """Return ``ctx.obj`` as what ``cli/main.py`` put there.

    ``click.Context.obj`` is typed ``Any``; narrowing it in one function
    is what lets every command below read attributes mypy checks.
    """
    context: CliContext = ctx.obj
    return context


def cli_objects(ctx: click.Context) -> tuple[Settings, RichOutput, OutputFormatter]:
    """Return the ``(settings, output, formatter)`` triple every command needs."""
    context = cli_context(ctx)
    return context.settings, context.output, context.formatter


def cli_session(ctx: click.Context) -> A1000Session:
    """Return the invocation's one A1000 connection — the only way to reach it."""
    return cli_context(ctx).session


def _probed(ctx: click.Context) -> tuple[Settings, RichOutput, OutputFormatter]:
    """Warn if the service did not answer, then hand over the usual objects.

    The probe must stay in a command body: that is the first hook click
    runs after the whole command line is accepted, so ``--help``, usage
    errors and completion pay nothing for it.
    """
    context = cli_context(ctx)
    if context.warn_if_unavailable is not None:
        context.warn_if_unavailable()
    return context.settings, context.output, context.formatter


def probed_a1000[S: A1000Service](
    ctx: click.Context, service_cls: type[S]
) -> tuple[S, RichOutput, OutputFormatter]:
    """Probe, then build ``service_cls`` over the session the probe just used."""
    _, output, formatter = _probed(ctx)
    return cli_session(ctx).service(service_cls, output), output, formatter


def probed_ticloud[S: TitaniumCloudApi](
    ctx: click.Context, service_cls: type[S]
) -> tuple[S, RichOutput, OutputFormatter]:
    """Probe, then build ``service_cls``, which has no session: it handles a call at a time.

    The ``ticloud`` group spans two services, so each command asks for the
    half it uses by name.
    """
    settings, output, formatter = _probed(ctx)
    return service_cls(settings, output), output, formatter


def run_step[T](
    output: RichOutput,
    spinner: str,
    call: Callable[[], T | None],
    *,
    success: str | Callable[[T], str],
    failure: str,
    empty: str | None = None,
    render: Callable[[T], None] | None = None,
    empty_formatter: OutputFormatter | None = None,
) -> T | None:
    """Run a service call under a spinner, report the outcome, and hand the result back.

    ``None`` means the call itself failed — the service could not reach
    the endpoint or could not make sense of the answer — and is always an
    error worth exiting non-zero for, reported as ``failure``. An
    empty-but-valid result (``[]``, ``{}``, ``False``) is a different
    fact: passing ``empty`` reports it as that warning and exits 0, so a
    500 does not print "nothing matched" underneath the real error.
    Omitting ``empty`` says this command has no legitimate empty answer,
    and an empty one is treated as the failure it is.

    ``empty_formatter`` keeps ``-o json`` (and every other machine format)
    parseable when the answer is an empty collection: a consumer piping
    ``search`` into ``jq`` gets ``[]`` on stdout instead of nothing, while
    the human ``empty`` warning still goes to the status stream. It is left
    off for the console (``rich``) format, whose warning already says it,
    and for a falsey non-collection like ``False`` that has no dump to give.

    Returns ``None`` in both cases, so callers that need to do more than
    render can still branch on it.
    """
    with output.progress_spinner(spinner):
        result = call()

    if result is None:
        output.error(failure)
        return None

    if not result:
        if empty is None:
            output.error(failure)
        else:
            output.warning(empty)
            if (
                empty_formatter is not None
                and empty_formatter.format != OutputFormat.RICH
                and isinstance(result, list | dict)
            ):
                empty_formatter.display(result)
        return None

    output.success(success(result) if callable(success) else success)
    if render is not None:
        render(result)
    return result


def run_network_report(
    output: RichOutput,
    formatter: OutputFormatter,
    noun: str,
    subject: str,
    call: Callable[[], Any | None],
) -> None:
    """Run the shared domain/IP report presentation used by both services."""
    lower_noun = noun.lower()
    run_step(
        output,
        f"Fetching {lower_noun} report for {subject}...",
        call,
        success=f"{noun} report for {subject}",
        failure=f"Failed to get {lower_noun} report",
        empty=f"No network intelligence for {subject}",
        render=formatter.display,
        empty_formatter=formatter,
    )


# The console a renderer draws on, named from the module that owns the
# rendering contract. The whole of ``rl_cli/cli`` reaches for rich through
# this one line: command modules annotate their renderer closures with this
# name rather than importing rich, and ``cli/main.py`` builds the
# invocation's consoles from it.
RenderConsole = Console


class RichRenderer(Protocol):
    """A formatter that draws ``data`` on the console it is handed.

    The console is part of the contract, not a convenience: it carries the
    invocation's colour policy, and a renderer left to build its own gets
    a fresh ``Console`` that reads the terminal instead of the settings.
    """

    def __call__(self, data: Any, /, *, console: RenderConsole) -> None: ...


def without_console(draw: Callable[[Any], None]) -> RichRenderer:
    """Present a renderer that draws on the status stream to :func:`display`.

    A renderer that reports through ``RichOutput`` keeps :class:`RichRenderer`'s
    promise on the other stream, and must say so by being wrapped here rather
    than by quietly taking a console and dropping it.
    """

    def draw_on_status(data: Any, /, *, console: RenderConsole) -> None:
        del console
        draw(data)

    return draw_on_status


def display(
    formatter: OutputFormatter,
    rich_renderer: RichRenderer,
    data: Any,
    *,
    plain: bool = False,
) -> None:
    """Show ``data`` richly on the CLI's console, else as the machine-readable dump.

    ``-o json`` has to win over a command's pretty table, or a pipeline
    gets a box-drawing character where it asked for JSON. ``plain`` is a
    command's own ``--json`` flag asking for the same thing, so the two
    spellings of "defer to the dump" stay one decision.

    The console goes to the renderer from here rather than being bound at
    each ``render=`` slot, so the `color: false` and ``--no-color`` policy
    that ``cli/main.py`` built the consoles with reaches every rendering.
    """
    if plain or formatter.format != OutputFormat.RICH:
        formatter.display(data)
    else:
        rich_renderer(data, console=formatter.console)
