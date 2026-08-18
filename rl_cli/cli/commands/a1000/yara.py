"""YARA rulesets: what is on the appliance, what it matches, and when it publishes."""

from functools import partial
from pathlib import Path

import click

from rl_cli.cli.commands._shared_inputs import confirmed, yes_flag
from rl_cli.cli.commands.a1000._shared import a1000, capped_listing
from rl_cli.cli.context import display, probed_a1000, run_step
from rl_cli.render.formatters import (
    print_search_results_table,
    print_yara_content,
    print_yara_rulesets_table,
)
from rl_cli.services.a1000 import A1000YaraService
from rl_cli.storage.files import read_text_lenient


@a1000.command()
@click.argument("ruleset_name")
@yes_flag
@click.pass_context
def yara_delete(ctx: click.Context, ruleset_name: str, yes: bool) -> None:
    """Delete YARA ruleset."""
    service, output, _ = probed_a1000(ctx, A1000YaraService)

    if not confirmed(output, yes, f"Delete YARA ruleset '{ruleset_name}'?"):
        return

    run_step(
        output,
        f"Deleting ruleset {ruleset_name}...",
        lambda: service.delete_yara_ruleset(ruleset_name),
        success=f"YARA ruleset '{ruleset_name}' deleted",
        failure="Failed to delete YARA ruleset",
    )


@a1000.command()
@click.argument("ruleset_name")
@click.pass_context
def yara_content(ctx: click.Context, ruleset_name: str) -> None:
    """Get YARA ruleset content."""
    service, output, formatter = probed_a1000(ctx, A1000YaraService)

    run_step(
        output,
        "Fetching ruleset content...",
        lambda: service.get_yara_content(ruleset_name),
        success=f"YARA ruleset '{ruleset_name}' content:",
        failure="Failed to get YARA content",
        # A ruleset built from an empty file has empty contents; the
        # appliance answers 200 with nothing, which is an answer.
        empty=f"YARA ruleset '{ruleset_name}' has no content",
        render=partial(display, formatter, print_yara_content),
        empty_formatter=formatter,
    )


@a1000.command()
@click.pass_context
def yara_list(ctx: click.Context) -> None:
    """List YARA rulesets on appliance."""
    service, output, formatter = probed_a1000(ctx, A1000YaraService)

    render = capped_listing(output, print_yara_rulesets_table)
    run_step(
        output,
        "Fetching YARA rulesets...",
        service.list_yara_rulesets,
        success=lambda rulesets: f"Found {len(rulesets)} YARA rulesets",
        failure="Failed to list YARA rulesets",
        empty="No YARA rulesets on this appliance",
        render=partial(display, formatter, render),
        empty_formatter=formatter,
    )


@a1000.command()
@click.argument("ruleset_name")
@click.argument("rule_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_context
def yara_create(ctx: click.Context, ruleset_name: str, rule_file: Path) -> None:
    """Create or update YARA ruleset."""
    service, output, _ = probed_a1000(ctx, A1000YaraService)

    try:
        rules_content = read_text_lenient(rule_file)
    except OSError as error:
        output.error(f"Could not read {rule_file}: {error}")
        return
    run_step(
        output,
        f"Creating ruleset {ruleset_name}...",
        lambda: service.create_yara_ruleset(ruleset_name, rules_content),
        success=f"YARA ruleset '{ruleset_name}' created/updated",
        failure="Failed to create YARA ruleset",
    )


@a1000.command()
@click.argument("ruleset_name")
@click.option("--enable/--disable", default=True, help="Enable or disable ruleset")
@click.pass_context
def yara_toggle(ctx: click.Context, ruleset_name: str, enable: bool) -> None:
    """Enable or disable YARA ruleset."""
    service, output, _ = probed_a1000(ctx, A1000YaraService)

    # Two forms of the verb, not one derived from the other: the spinner
    # needs the progressive and the failure line the infinitive.
    action, verb = ("Enabling", "enable") if enable else ("Disabling", "disable")
    run_step(
        output,
        f"{action} ruleset {ruleset_name}...",
        lambda: service.toggle_yara_ruleset(ruleset_name, enable),
        success=f"YARA ruleset '{ruleset_name}' {verb}d",
        failure=f"Failed to {verb} YARA ruleset",
    )


@a1000.command()
@click.argument("ruleset_name")
@click.argument("hash_value", required=False)
@click.pass_context
def yara_matches(ctx: click.Context, ruleset_name: str, hash_value: str | None) -> None:
    """Get samples matching a YARA ruleset, optionally one sample only."""
    service, output, formatter = probed_a1000(ctx, A1000YaraService)

    render = capped_listing(output, print_search_results_table)
    run_step(
        output,
        "Checking YARA matches...",
        lambda: service.get_yara_matches(ruleset_name, hash_value),
        success=lambda matches: f"{len(matches)} YARA match(es) for ruleset '{ruleset_name}'",
        failure=f"Failed to check matches for ruleset '{ruleset_name}'",
        # "No matches" is a legitimate answer only from a walk that read
        # every page, and ``run_step`` is handed this wording before the call
        # it wraps has run, so it cannot be the place that decides: a walk
        # that stopped early with the appliance still promising more hands
        # back ``None`` instead of an empty list, and is reported through
        # ``failure`` above (A1000YaraService.get_yara_matches).
        empty="No YARA matches found",
        render=partial(display, formatter, render),
        empty_formatter=formatter,
    )


@a1000.command()
@click.argument("ruleset_name", required=False)
@click.option("--all", "publish_all", is_flag=True, help="Publish all non-core rulesets")
@click.pass_context
def yara_publish(ctx: click.Context, ruleset_name: str | None, publish_all: bool) -> None:
    """Publish a single YARA ruleset (or --all non-core rulesets)."""
    # A wrong combination of arguments is a usage error, as in
    # yara-repo-list: reported as a failed operation it exits 1 with no
    # usage line.
    if publish_all and ruleset_name:
        raise click.UsageError("Pass either RULESET_NAME or --all, not both")
    if not publish_all and not ruleset_name:
        raise click.UsageError("Pass RULESET_NAME or --all")

    service, output, _ = probed_a1000(ctx, A1000YaraService)

    if ruleset_name:
        run_step(
            output,
            f"Publishing YARA ruleset '{ruleset_name}'...",
            lambda: service.publish_yara_ruleset(ruleset_name),
            success=f"Triggered publish for ruleset: {ruleset_name}",
            failure=f"Failed to publish YARA ruleset: {ruleset_name}",
        )
    else:
        run_step(
            output,
            "Publishing all non-core YARA rulesets...",
            service.publish_all_yara_rulesets,
            success="Triggered publish for all non-core YARA rulesets",
            failure="Failed to publish all YARA rulesets",
        )


@a1000.command()
@click.pass_context
def yara_update_now(ctx: click.Context) -> None:
    """Manually trigger the YARA update job."""
    service, output, _ = probed_a1000(ctx, A1000YaraService)

    run_step(
        output,
        "Triggering YARA update job...",
        service.run_yara_update,
        success="YARA update job triggered",
        failure="Failed to trigger YARA update job",
    )


@a1000.command()
@click.argument("seconds", type=int, required=False)
@click.option("--reset", is_flag=True, help="Reset interval to its default value")
@click.pass_context
def yara_update_interval(ctx: click.Context, seconds: int | None, reset: bool) -> None:
    """Configure or reset the YARA update job interval (in seconds)."""
    # Same class of mistake, same answer as yara-publish and
    # yara-repo-list: a usage error, with the usage.
    if reset and seconds is not None:
        raise click.UsageError("Pass either SECONDS or --reset, not both")
    if not reset and seconds is None:
        raise click.UsageError("Pass SECONDS or --reset")

    service, output, _ = probed_a1000(ctx, A1000YaraService)

    if seconds is not None:
        run_step(
            output,
            f"Setting YARA update interval to {seconds}s...",
            lambda: service.set_yara_update_interval(seconds),
            success=(
                "YARA auto-update job disabled"
                if seconds == 0
                else f"YARA update interval now {seconds}s"
            ),
            failure="Failed to set YARA update interval",
        )
    else:
        run_step(
            output,
            "Resetting YARA update interval...",
            service.reset_yara_update_interval,
            success="YARA update interval reset to default",
            failure="Failed to reset YARA update interval",
        )
