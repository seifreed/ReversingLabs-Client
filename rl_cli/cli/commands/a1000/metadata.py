"""User tags, classifications and containers attached to a sample."""

import click

from rl_cli.cli.commands._shared_inputs import confirmed, yes_flag
from rl_cli.cli.commands.a1000._shared import a1000
from rl_cli.cli.context import probed_a1000, run_step
from rl_cli.services.a1000 import A1000MetadataService


@a1000.command()
@click.argument("hash_value")
@click.option("--tag", "-t", multiple=True, required=True, help="Tags to add")
@click.pass_context
def add_tags(ctx: click.Context, hash_value: str, tag: tuple[str, ...]) -> None:
    """Add user tags to sample."""
    service, output, _ = probed_a1000(ctx, A1000MetadataService)

    tags_list = list(tag)
    run_step(
        output,
        f"Adding {len(tags_list)} tags...",
        lambda: service.add_user_tags(hash_value, tags_list),
        success=f"Added tags: {', '.join(tags_list)}",
        failure="Failed to add tags",
    )


@a1000.command()
@click.argument("hash_value")
@click.pass_context
def get_tags(ctx: click.Context, hash_value: str) -> None:
    """Get user tags for sample."""
    service, output, formatter = probed_a1000(ctx, A1000MetadataService)

    run_step(
        output,
        "Fetching tags...",
        lambda: service.get_user_tags(hash_value),
        success="User tags retrieved",
        failure="Failed to get user tags",
        empty="No user tags for this sample",
        render=formatter.display,
        empty_formatter=formatter,
    )


@a1000.command()
@click.argument("hash_value")
@click.option("--tag", "-t", multiple=True, help="Specific tags to remove (all if not specified)")
@yes_flag
@click.pass_context
def remove_tags(ctx: click.Context, hash_value: str, tag: tuple[str, ...], yes: bool) -> None:
    """Remove user tags from sample."""
    service, output, _ = probed_a1000(ctx, A1000MetadataService)

    if tag:
        tags_list = list(tag)
        run_step(
            output,
            "Removing tags...",
            lambda: service.remove_user_tags(hash_value, tags_list),
            success=f"Removed tags: {', '.join(tags_list)}",
            failure="Failed to remove tags",
        )
        return

    def confirm(tags: list[str]) -> bool:
        # No --tag is the remove-everything mode, and it is irreversible: a
        # whole triage history goes with one mistyped flag. The explicit -t
        # form says what it is removing, so it does not ask.
        return confirmed(output, yes, f"Remove all {len(tags)} tags ({', '.join(tags)})?")

    # What "all" resolves to is the service's business; the prompt is
    # this layer's. No spinner: it would be drawn over the prompt.
    removed = service.remove_all_user_tags(hash_value, confirm=confirm)
    if removed is None:
        output.error("Failed to remove tags")
    elif removed:
        output.success(f"Removed tags: {', '.join(removed)}")


@a1000.command()
@click.argument("hash_value")
@click.argument("classification", type=click.Choice(["goodware", "suspicious", "malicious"]))
@click.option("--threat-name", "-t", help="Threat name for malicious classification")
@click.option(
    "--system",
    type=click.Choice(["local", "ticloud"]),
    default="local",
    help="Where to store the classification",
)
@click.pass_context
def set_classification(
    ctx: click.Context,
    hash_value: str,
    classification: str,
    threat_name: str | None,
    system: str,
) -> None:
    """Set custom classification for sample."""
    service, output, _ = probed_a1000(ctx, A1000MetadataService)

    was_set = run_step(
        output,
        f"Setting classification to {classification}...",
        lambda: service.set_classification(hash_value, classification, threat_name, system),
        success=f"Classification set to {classification}",
        failure="Failed to set classification",
    )
    if was_set and threat_name:
        output.info(f"Threat name: {threat_name}")


@a1000.command()
@click.argument("hash_value")
@click.pass_context
def get_classification(ctx: click.Context, hash_value: str) -> None:
    """Get classification for sample."""
    service, output, formatter = probed_a1000(ctx, A1000MetadataService)

    run_step(
        output,
        "Fetching classification...",
        lambda: service.get_classification(hash_value),
        success="Classification retrieved",
        failure="Failed to get classification",
        # A sample nobody has classified is answered 200 with an empty
        # record, which is an answer about the sample: warn and exit 0, as
        # get-tags and containers below already do with theirs.
        empty="No classification for this sample",
        render=formatter.display,
        empty_formatter=formatter,
    )


@a1000.command()
@click.argument("hash_value")
@click.option(
    "--system",
    type=click.Choice(["local", "ticloud"]),
    default="local",
    help="Which stored classification to delete",
)
@yes_flag
@click.pass_context
def delete_classification(ctx: click.Context, hash_value: str, system: str, yes: bool) -> None:
    """Delete custom classification."""
    service, output, _ = probed_a1000(ctx, A1000MetadataService)

    if not confirmed(output, yes, f"Remove custom {system} classification?"):
        return

    run_step(
        output,
        f"Deleting {system} classification...",
        lambda: service.delete_classification(hash_value, system),
        success=f"Custom {system} classification deleted",
        failure="Failed to delete classification",
    )


@a1000.command()
@click.argument("hash_value")
@click.pass_context
def containers(ctx: click.Context, hash_value: str) -> None:
    """List containers (archives) for sample."""
    service, output, formatter = probed_a1000(ctx, A1000MetadataService)

    run_step(
        output,
        "Fetching containers...",
        lambda: service.list_containers(hash_value),
        success=lambda found: f"Found {len(found)} containers",
        failure="Failed to list containers",
        empty="No containers found for this sample",
        render=formatter.display,
        empty_formatter=formatter,
    )
