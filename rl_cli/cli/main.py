"""Main CLI entry point.

One rule governs the resolutions below: nothing a flag resolves is written
back onto ``settings``. That object is what ``config save`` and ``config
create-profile`` serialise, so a run's own flags written onto it end up in
the user's file. Resolved values live on a local, on the formatter, or on
:class:`~rl_cli.cli.context.CliContext`.
"""

import signal
from collections.abc import Callable
from pathlib import Path
from typing import Literal, NoReturn

import click

from rl_cli import __version__
from rl_cli.cli.commands import a1000, config, ticloud
from rl_cli.cli.context import CliContext, RenderConsole, cli_context
from rl_cli.config import ConfigFileError, Settings, get_settings
from rl_cli.render.output import OutputFormat, OutputFormatter, RichOutput
from rl_cli.services.a1000 import A1000Session
from rl_cli.services.availability import APIAvailabilityChecker, ServiceStatus


@click.group()
@click.version_option(version=__version__, prog_name="rl-cli")
@click.option(
    "--config",
    "-c",
    "config_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--profile",
    "-p",
    default=None,
    help="Configuration profile to use  [default: RL_CLI_PROFILE, else 'default']",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice([fmt.value for fmt in OutputFormat]),
    default=None,
    help="Output format (overrides the configured default)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--quiet", "-q", is_flag=True, help="Suppress progress output (warnings and errors remain)"
)
@click.pass_context
def cli(
    ctx: click.Context,
    config_file: Path | None,
    profile: str | None,
    output: str | None,
    verbose: bool,
    quiet: bool,
) -> None:
    """
    ReversingLabs CLI - A modular command-line interface for ReversingLabs services.

    This tool provides easy access to ReversingLabs TitaniumCloud and A1000 platform
    capabilities, including file analysis, reputation checking, and threat intelligence.
    """
    # Empty the slot before anything can report: it outlives the invocation,
    # so a second run in one process would otherwise report its pre-config
    # paths through the previous run's consoles.
    _remember_reporter(None)

    settings = _resolve_settings(ctx, {"profile": profile, "config_file": config_file})
    # Flags can only enable; absence must not clobber configured values.
    verbose = verbose or settings.output.verbose
    quiet = quiet or settings.output.quiet

    rich_output = _build_reporting(settings, quiet=quiet)
    # The one place that knows the colour policy, so the one place that can
    # give it to the paths reporting from outside a command.
    _remember_reporter(rich_output)
    _warn_about_a_missing_profile(settings, rich_output)
    # No fallback branch: click.Choice has already refused anything the flag
    # could not name, and a configured value is validated against the same
    # enum when the file is loaded.
    output_format = OutputFormat(output) if output else settings.output.format
    # The formatter renders through the same RichOutput as everything else:
    # a second one would be a second --quiet and a second had_error.
    formatter = OutputFormatter(output_format, rich_output)

    # The one connection this invocation gets: the availability probe and
    # the command's own services are all built from it, so an invocation
    # authenticates once however many areas of the appliance it touches.
    # Constructing it opens nothing.
    session = A1000Session(settings)

    ctx.obj = CliContext(
        settings=settings,
        output=rich_output,
        formatter=formatter,
        session=session,
        warn_if_unavailable=_availability_warning(
            ctx, APIAvailabilityChecker(session, rich_output, verbose=verbose), rich_output
        ),
        verbose=verbose,
    )


def _resolve_settings(ctx: click.Context, given: dict[str, object]) -> Settings:
    """The settings this invocation runs on, or exit 1 naming the unusable file.

    ``given`` is only what was actually named on the command line, and the
    ``None``s must keep being dropped: pydantic-settings ranks constructor
    arguments above the environment, so handing it click's own default makes
    RL_CLI_PROFILE and RL_CLI_CONFIG_FILE unreachable — and the profile
    chooses which appliance is talked to.
    """
    try:
        return get_settings(**{name: value for name, value in given.items() if value is not None})
    except ConfigFileError as error:
        # A file the user wrote states something unusable: a mistake to fix
        # rather than a fault to report under main()'s "Unexpected error".
        # Caught here rather than there because a programmatic ``cli(...)``
        # never goes through main() at all.
        _reporting().error(str(error))
        ctx.exit(1)


def _build_reporting(settings: Settings, *, quiet: bool) -> RichOutput:
    """The invocation's one reporter, built with its colour policy.

    Both consoles get the policy: warnings go to the stderr one, and
    colouring only stdout leaves half the output ignoring the choice.
    """
    no_color = not settings.output.color
    return RichOutput(
        RenderConsole(no_color=no_color),
        RenderConsole(stderr=True, no_color=no_color),
        quiet=quiet,
    )


def _warn_about_a_missing_profile(settings: Settings, output: RichOutput) -> None:
    """Say that the named profile is not in the file, rather than acting as it."""
    if not settings.missing_profile:
        return
    output.warning(
        f"Profile '{settings.missing_profile}' is not in {settings.config_file} - using defaults"
    )
    output.info("Run 'rl-cli config list-profiles' to see what is defined")


def _availability_warning(
    ctx: click.Context, checker: APIAvailabilityChecker, output: RichOutput
) -> Callable[[], None] | None:
    """The availability warning for the invoked service, or ``None``.

    Returned rather than run: this callback is reached before the
    subcommand's arguments are parsed, so probing here would make
    ``a1000 report --help`` block for the full request timeout against an
    unreachable appliance. The ``probed_*`` helpers run it later, from a
    command body.

    Quiet mode does not skip the probe: ``--quiet`` promises that warnings
    survive it, and "the appliance is unreachable" is the one line that
    explains an empty result. Only the follow-up hint is optional, and
    ``output.info`` already drops it when quiet.
    """
    service_keys: dict[str, tuple[str, Literal["titanium_cloud", "a1000"]]] = {
        "ticloud": ("TitaniumCloud", "titanium_cloud"),
        "a1000": ("A1000", "a1000"),
    }
    if ctx.invoked_subcommand not in service_keys:
        return None
    label, key = service_keys[ctx.invoked_subcommand]

    def warn() -> None:
        status = checker.check_all()[key]
        if status["status"] != ServiceStatus.AVAILABLE.value:
            output.warning(f"{label}: {status['message']}")
            output.info("Run 'rl-cli config check-access' for details")

    return warn


@cli.result_callback()
@click.pass_context
def _exit_nonzero_on_failure(ctx: click.Context, result: object, **_: object) -> None:
    """Give a failed command a non-zero exit status.

    Commands report failure by printing and returning, so without this every
    invocation exits 0 and a script chaining on success carries on with an
    empty file. Doing it here covers every command from one place.

    No ``ctx.obj`` is nothing to grade rather than a crash: nothing holds
    ``cli()`` to leaving a ``CliContext`` behind on every path, and an early
    ``return`` added there must not turn every invocation into an
    ``AttributeError`` reported as an unexpected error.
    """
    if ctx.obj is None:
        return
    if cli_context(ctx).output.status.failed:
        ctx.exit(1)


cli.add_command(ticloud.ticloud)
cli.add_command(a1000.a1000)
cli.add_command(config.config)


# The invocation's own reporter, for the two paths that have to say
# something without a command context: the SIGINT handler and main()'s
# catch-all. Both run outside click, so a module-level slot is what they
# can reach — a bare ``RichOutput()`` would read the terminal and print
# colour under `color: false`.
_reporter: RichOutput | None = None


def _remember_reporter(output: RichOutput | None) -> None:
    """Point the context-less paths at ``output``, or at nothing when ``None``."""
    global _reporter
    _reporter = output


def _reporting() -> RichOutput:
    """The configured reporter, or an uncoloured one when there is none yet.

    An interrupt can arrive before the group callback has read the config,
    or instead of it. Colour must not be guessed at there: escape sequences
    printed into a run that may have asked for none is what this avoids.
    """
    return _reporter or RichOutput(RenderConsole(no_color=True))


def _interrupted(*_: object) -> NoReturn:
    """Report Ctrl-C as an interruption and exit 130.

    Click turns a ``KeyboardInterrupt`` raised inside ``cli()`` into its own
    ``Abort`` and exits 1, making an interrupted download indistinguishable
    by exit status from a command that failed. Handling SIGINT gets in
    before that: click does not catch ``SystemExit``, and the
    ``BaseException`` cleanups that remove a half-written sample still run.
    """
    _reporting().warning("Interrupted")
    raise SystemExit(130)


def main() -> None:
    """Main entry point for the CLI.

    An unexpected exception is reported and not re-raised: a traceback of
    our internals is not a CLI's error message.
    """
    signal.signal(signal.SIGINT, _interrupted)
    try:
        cli()
    except KeyboardInterrupt:
        # Reachable when something else owns SIGINT, or the interrupt
        # arrived before the handler above was installed.
        _interrupted()
    except Exception as e:
        _reporting().error(f"Unexpected error: {e!s}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
