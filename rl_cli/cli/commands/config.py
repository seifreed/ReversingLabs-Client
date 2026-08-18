"""Configuration management CLI commands.

Thin on purpose. What a probe run reports is
:mod:`rl_cli.models.availability`'s, how it is drawn is
``render/formatters/config_report.py``'s, and the questions `config init`
puts are :mod:`rl_cli.config.wizard`'s — this module resolves the click
context, supplies the terminal, and decides what goes to stdout.
"""

from collections.abc import Sequence
from functools import partial
from pathlib import Path

import click

from rl_cli.cli.commands._shared_inputs import confirmed, yes_flag
from rl_cli.cli.context import cli_context, cli_objects, cli_session, display, without_console
from rl_cli.config.settings import Settings, read_config_file, write_private_yaml
from rl_cli.config.wizard import configure
from rl_cli.models.availability import ServiceStatus, availability_block, availability_payload
from rl_cli.render.formatters import (
    print_availability_panel,
    print_profile_names,
    print_service_lines,
)
from rl_cli.services.availability import APIAvailabilityChecker


def _active_config_file(settings: Settings) -> Path:
    """The config file the session actually loaded, or where it would live."""
    return settings.config_file or settings.config_dir / "config.yaml"


class _ClickPrompter:
    """The terminal half of :mod:`rl_cli.config.wizard`, and nothing else.

    A class rather than two module functions because the wizard takes one
    object: what it needs is an implementation of ``Prompter``, and the
    pair have to be handed over together.
    """

    def confirm(self, question: str, /, *, default: bool = False) -> bool:
        return click.confirm(question, default=default)

    def ask(
        self,
        question: str,
        /,
        *,
        default: str | None = None,
        secret: bool = False,
        choices: Sequence[str] | None = None,
    ) -> str:
        answer: str = click.prompt(
            question,
            default=default,
            hide_input=secret,
            type=click.Choice(choices) if choices is not None else None,
        )
        return answer


@click.group()
def config() -> None:
    """Manage configuration settings."""


@config.command()
@click.option("--check-access", "-a", is_flag=True, help="Also check API availability")
@click.pass_context
def show(ctx: click.Context, check_access: bool) -> None:
    """Show current configuration."""
    settings, output, formatter = cli_objects(ctx)

    # profile_dump() redacts the token and the passwords: this goes to
    # stdout, where a redirect or a pasted terminal takes it somewhere
    # far less protected than the 0600 file it was read from.
    config_dict = {
        "profile": settings.profile,
        **settings.profile_dump(),
        "cache_dir": str(settings.cache_dir),
        "config_dir": str(settings.config_dir),
    }

    availability = (
        APIAvailabilityChecker(
            cli_session(ctx), output, verbose=cli_context(ctx).verbose
        ).check_all()
        if check_access
        else None
    )
    if availability is not None:
        config_dict["api_availability"] = availability_block(availability)

    output.info(f"Current configuration (profile: {settings.profile})")
    if settings.config_file:
        output.info(f"Config file: {settings.config_file}")
    formatter.display(config_dict)

    # The block above is data on stdout; this is the human summary on
    # stderr, so `-o json | jq` keeps working while a terminal user still
    # gets the reason a service is unusable. Drawn by the same renderer
    # ``check-access`` uses, so the two cannot grade a service differently.
    if availability is not None:
        output.info("\nAPI Status Summary:")
        print_service_lines(availability, output=output, indent="  ")


@config.command()
# No ``-p`` for ``--path``: ``-p`` is ``--profile`` on the root group, and
# both take a free string, so `config save -p prod` would silently write a
# file named `prod` in the current directory holding the *active* profile's
# settings, leave config.yaml untouched, and report success. Without the
# short form that invocation is a usage error.
@click.option(
    "--path", type=click.Path(dir_okay=False, path_type=Path), help="Path to save configuration"
)
@click.pass_context
def save(ctx: click.Context, path: Path | None) -> None:
    """Save current configuration to file."""
    settings, output, _ = cli_objects(ctx)

    config_path = path or _active_config_file(settings)
    settings.save_config(config_path)

    output.success(f"Configuration saved to {config_path}")


@config.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize configuration interactively."""
    settings, output, _ = cli_objects(ctx)

    output.info("Initializing ReversingLabs CLI configuration")
    configure(settings, _ClickPrompter())

    if click.confirm("Save configuration?"):
        config_path = _active_config_file(settings)
        settings.save_config(config_path)
        output.success(f"Configuration saved to {config_path}")
    else:
        output.info("Configuration not saved")


@config.command()
@click.argument("profile")
@yes_flag
@click.pass_context
def create_profile(ctx: click.Context, profile: str, yes: bool) -> None:
    """Store this invocation's settings as a new profile, not a copy of PROFILE.

    What is written is what the CLI resolved for this run: the active
    profile overlaid with the environment and the flags given. So
    `--profile staging config create-profile staging`, on a config file
    that has no `staging` yet, stores the defaults plus whatever the
    environment supplied - not a clone of anything.
    """
    settings, output, _ = cli_objects(ctx)

    config_file = _active_config_file(settings)

    config_data = read_config_file(config_file)

    if profile in config_data and not confirmed(
        output,
        yes,
        f"Profile '{profile}' already exists. Overwrite?",
        cancelled="Profile creation cancelled",
    ):
        return

    # Unredacted: this is the 0600 file the credentials are read back from,
    # not a display.
    config_data[profile] = settings.profile_dump(redact=False)

    write_private_yaml(config_file, config_data)

    output.success(f"Profile '{profile}' created")


@config.command()
@click.option("--detailed", "-d", is_flag=True, help="Show detailed availability information")
@click.option("--force", "-f", is_flag=True, help="Force refresh, ignore cache")
@click.pass_context
def check_access(ctx: click.Context, detailed: bool, force: bool) -> None:
    """Check API access and available features based on current credentials.

    Exits 0 whenever the check was made, whatever it found. This is the
    command the CLI points an analyst at when a service is unreachable, so
    an unreachable appliance is the answer it was asked for, not a failure
    of the asking; a diagnostic that exits non-zero exactly when it has
    something to report is one that `set -e` aborts before it can be read.
    `config show -a` renders the same measurement and also exits 0.

    A non-zero exit means the check could not be made — an unusable config
    file, or the checker itself failing — and that is decided by the
    layers that own those failures rather than by what was measured.

    What was measured is on stdout, so a script branches on the answer
    rather than on the status: `rl-cli -o json config check-access | jq -e
    '.summary.services_available > 0'`.
    """
    _, output, formatter = cli_objects(ctx)

    output.info("Checking API availability...")

    checker = APIAvailabilityChecker(cli_session(ctx), output, verbose=cli_context(ctx).verbose)

    if force:
        checker.clear_cache()
        output.info("Cache cleared, performing fresh check...")

    availability = checker.check_all(force=force)
    # The panel is stdout, so ``-o json`` has to win it, ``display``'s
    # decision here as in every other command. The per-service lines stay
    # on stderr in every format: they are the human's reason, not the data.
    display(formatter, print_availability_panel, availability_payload(availability))
    print_service_lines(availability, output=output, detailed=detailed)

    if availability["summary"]["services_available"] > 0:
        output.info("\nQuick Start:")
        if availability["a1000"]["status"] == ServiceStatus.AVAILABLE.value:
            output.info("  rl-cli --config ./config.yaml a1000 list")
            output.info("  rl-cli a1000 --help")
        if availability["titanium_cloud"]["status"] == ServiceStatus.AVAILABLE.value:
            output.info("  rl-cli --config ./config.yaml ticloud reputation <hash>")
            output.info("  rl-cli ticloud --help")
    else:
        # Guidance, and no ``ctx.exit(1)`` under it: this is a measurement,
        # reported, exit 0. A script that wants to branch on the count reads
        # it off the stdout document.
        output.warning("\nNo services available. Please configure credentials:")
        output.warning("  rl-cli config init")


@config.command()
@click.pass_context
def list_profiles(ctx: click.Context) -> None:
    """List available configuration profiles."""
    settings, output, formatter = cli_objects(ctx)

    config_file = _active_config_file(settings)

    if not config_file.exists():
        output.warning("No configuration file found")
        return

    config_data = read_config_file(config_file)

    if not config_data:
        output.warning("No profiles found")
        return

    # The names are the data, so they go to stdout: `-o json config
    # list-profiles | jq -r '.[]'` feeds a profile fan-out loop. The
    # "(current)" marker is the terminal rendering.
    display(
        formatter,
        without_console(partial(print_profile_names, output=output, current=settings.profile)),
        list(config_data),
    )
