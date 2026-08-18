"""Commands about the connection itself: reaching the appliance, and how."""

import click

from rl_cli.cli.commands.a1000._shared import a1000
from rl_cli.cli.context import probed_a1000, run_step
from rl_cli.services.a1000 import A1000Service


@a1000.command()
@click.pass_context
def test(ctx: click.Context) -> None:
    """Test connection to A1000."""
    service, output, _ = probed_a1000(ctx, A1000Service)

    if run_step(
        output,
        "Testing connection...",
        service.test_connection,
        success="Connection successful",
        failure="Connection failed",
    ):
        summary = service.connection_summary()
        output.info(f"Host: {summary['host']}")
        output.info(f"Auth: {summary['auth']}")


@a1000.command()
@click.pass_context
def config_dump(ctx: click.Context) -> None:
    """Show the local A1000 connection settings (nothing is asked of the appliance)."""
    # The connection settings are all this command can show: the appliance
    # has no configuration endpoint, and the SDK's configuration_dump() is a
    # local string formatter over these same values.
    service, output, formatter = probed_a1000(ctx, A1000Service)

    # Not run through ``run_step``: that helper wraps a call to the
    # appliance in a spinner and reports what came back, and this one asks
    # the appliance nothing — the summary is five keys the settings always
    # have, so the failure line would be unreachable.
    output.success("Connection configuration")
    formatter.display(service.connection_summary())
