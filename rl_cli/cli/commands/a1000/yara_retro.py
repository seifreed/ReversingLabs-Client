"""YARA retro hunts: rescanning what the appliance already holds, in the cloud and locally."""

import click

from rl_cli.cli.commands._shared_inputs import confirmed, yes_flag
from rl_cli.cli.commands.a1000._shared import a1000
from rl_cli.cli.context import probed_a1000, run_step
from rl_cli.services.a1000 import A1000YaraService


@a1000.command()
@click.argument("ruleset_name")
@click.option(
    "--operation",
    "-o",
    type=click.Choice(["start", "stop", "clear"], case_sensitive=False),
    required=True,
    help="Operation to perform on Cloud Retro for the ruleset",
)
@yes_flag
@click.pass_context
def yara_cloud_retro(ctx: click.Context, ruleset_name: str, operation: str, yes: bool) -> None:
    """Start, stop, or clear a YARA Cloud Retro scan."""
    service, output, _ = probed_a1000(ctx, A1000YaraService)

    # ``clear`` discards the ruleset's retro-hunt history on the appliance;
    # ``start`` and ``stop`` only move a scan along. Sharing one
    # ``--operation`` choice with them is what would let the destructive one
    # inherit their silence.
    if operation.lower() == "clear" and not confirmed(
        output, yes, f"Clear Cloud Retro scan history for YARA ruleset '{ruleset_name}'?"
    ):
        return

    # The endpoint answers with a status and no document, so there is
    # nothing to render.
    run_step(
        output,
        f"{operation.upper()} Cloud Retro scan for '{ruleset_name}'...",
        lambda: service.yara_cloud_retro_scan(ruleset_name, operation),
        success=f"Cloud Retro {operation.upper()} acknowledged",
        failure=f"Failed to {operation.lower()} Cloud Retro scan",
    )


@a1000.command()
@click.argument("ruleset_name")
@click.pass_context
def yara_cloud_retro_status(ctx: click.Context, ruleset_name: str) -> None:
    """Get YARA Cloud Retro scan status for a ruleset."""
    service, output, formatter = probed_a1000(ctx, A1000YaraService)

    run_step(
        output,
        "Fetching Cloud Retro status...",
        lambda: service.get_yara_cloud_retro_scan_status(ruleset_name),
        success="Cloud Retro status retrieved",
        failure="Failed to get Cloud Retro status",
        render=formatter.display,
    )


@a1000.command()
@click.option(
    "--operation",
    "-o",
    type=click.Choice(["start", "stop"], case_sensitive=False),
    required=True,
    help="Operation to perform on Local Retro",
)
@click.pass_context
def yara_local_retro(ctx: click.Context, operation: str) -> None:
    """Start or stop the YARA Local Retro scan on the appliance."""
    service, output, _ = probed_a1000(ctx, A1000YaraService)

    run_step(
        output,
        f"{operation.upper()} Local Retro scan...",
        lambda: service.yara_local_retro_scan(operation),
        success=f"Local Retro {operation.upper()} acknowledged",
        failure=f"Failed to {operation.lower()} Local Retro scan",
    )


@a1000.command()
@click.pass_context
def yara_local_retro_status(ctx: click.Context) -> None:
    """Get YARA Local Retro scan status on the appliance."""
    service, output, formatter = probed_a1000(ctx, A1000YaraService)

    run_step(
        output,
        "Fetching Local Retro status...",
        service.get_yara_local_retro_scan_status,
        success="Local Retro status retrieved",
        failure="Failed to get Local Retro status",
        render=formatter.display,
    )
