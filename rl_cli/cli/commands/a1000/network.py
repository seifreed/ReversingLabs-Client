"""URL submissions and network threat intelligence lookups.

These endpoints answer 200 with an empty report for a subject the appliance
holds nothing on. That is an answer and not a lookup error, so every report
command below passes ``empty``; without it, an enrichment run files every
clean URL, domain and IP as a failure.
"""

from collections.abc import Callable
from typing import Any

import click

from rl_cli.cli.commands._shared_inputs import fetch_all_flag
from rl_cli.cli.commands.a1000._shared import a1000, partial_answer_notice
from rl_cli.cli.context import probed_a1000, run_network_report, run_step
from rl_cli.services.a1000 import A1000NetworkService


@a1000.command()
@click.argument("url")
@click.option(
    "--crawler",
    type=click.Choice(["local", "cloud"]),
    default="local",
    help="Where to crawl the URL from",
)
@click.pass_context
def submit_url(ctx: click.Context, url: str, crawler: str) -> None:
    """Submit URL for analysis. Returns a task_id for polling."""
    service, output, formatter = probed_a1000(ctx, A1000NetworkService)

    output.info(f"Submitting URL: {url} (crawler={crawler})")
    run_step(
        output,
        "Submitting URL...",
        lambda: service.submit_url(url, crawler=crawler),
        success="URL submitted successfully",
        failure="Failed to submit URL",
        render=formatter.display,
    )


@a1000.command()
@click.argument("task_id")
@click.pass_context
def url_report(ctx: click.Context, task_id: str) -> None:
    """Get analysis report for a URL submission task_id."""
    service, output, formatter = probed_a1000(ctx, A1000NetworkService)

    run_step(
        output,
        "Fetching URL report...",
        lambda: service.get_url_report(task_id),
        success="URL report retrieved",
        failure="Failed to get URL report",
        empty="No URL report for this task yet",
        render=formatter.display,
        empty_formatter=formatter,
    )


@a1000.command()
@click.argument("task_id")
@click.pass_context
def url_status(ctx: click.Context, task_id: str) -> None:
    """Check URL analysis status for a submission task_id."""
    service, output, formatter = probed_a1000(ctx, A1000NetworkService)

    run_step(
        output,
        "Checking URL status...",
        lambda: service.url_status(task_id),
        success="URL status retrieved",
        failure="Failed to get URL status",
        # No ``empty``: this command exists to hand back a processing state,
        # and an answer carrying none is nothing for a poll to act on.
        render=formatter.display,
    )


# One lookup as the command calls it. The address is positional, so a
# service method is free to name its own parameter.
type _IpLookup = Callable[[str], list[dict[str, Any]] | None]


def _register_ip_lookup(
    command_name: str,
    noun: str,
    first_page: Callable[[A1000NetworkService], _IpLookup],
    every_page: Callable[[A1000NetworkService], _IpLookup],
) -> None:
    """Register one of the three "what else sits on this IP" lookups.

    The three differ only in the noun and which pair of service methods they
    call: ``get_X_from_ip`` fetches the first page,
    ``get_X_from_ip_aggregated`` walks the lot. One factory is what keeps
    ``--all`` meaning the same thing in all three.

    Each half must arrive as an accessor rather than as a method name, so
    the edge from command to service is one mypy checks rather than a
    ``getattr`` over that suffix convention. The method is bound inside the
    body — ``first_page(service)`` at the moment of the call.

    ticloud's nine pivots have their own factory, deliberately: the two
    share ``--all``, declared once in ``cli/commands/_shared_inputs``, but
    ticloud's endpoints take a ``--max-results`` bound these three have no
    aggregated form for, and the two say different words while they work.
    """

    @a1000.command(name=command_name, help=f"Find {noun} associated with IP address.")
    @click.argument("ip_address")
    @fetch_all_flag
    @click.pass_context
    def lookup(ctx: click.Context, ip_address: str, fetch_all: bool) -> None:
        service, output, formatter = probed_a1000(ctx, A1000NetworkService)
        fetch = every_page(service) if fetch_all else first_page(service)

        run_step(
            output,
            f"Finding {noun} for IP {ip_address}...",
            lambda: fetch(ip_address),
            success=lambda found: f"Found {len(found)} {noun} for {ip_address}",
            failure=f"Failed to look up {noun} for {ip_address}",
            empty=f"No {noun} found for this IP",
            render=formatter.display,
            empty_formatter=formatter,
        )
        partial_answer_notice(output, "re-run with --all", pages_left=service.pages_left_unfetched)


_register_ip_lookup(
    "ip-urls",
    "URLs",
    lambda service: service.get_urls_from_ip,
    lambda service: service.get_urls_from_ip_aggregated,
)
_register_ip_lookup(
    "ip-files",
    "files",
    lambda service: service.get_files_from_ip,
    lambda service: service.get_files_from_ip_aggregated,
)
_register_ip_lookup(
    "ip-domains",
    "domains",
    lambda service: service.get_domains_from_ip,
    lambda service: service.get_domains_from_ip_aggregated,
)


@a1000.command()
@click.argument("url")
@click.pass_context
def network_url_report(ctx: click.Context, url: str) -> None:
    """Get network intelligence for URL."""
    service, output, formatter = probed_a1000(ctx, A1000NetworkService)

    run_step(
        output,
        "Fetching network report for URL...",
        lambda: service.get_network_url_report(url),
        success="Network URL report retrieved",
        failure="Failed to get network URL report",
        empty="No network intelligence for this URL",
        render=formatter.display,
        empty_formatter=formatter,
    )


@a1000.command()
@click.argument("domain")
@click.pass_context
def domain_report(ctx: click.Context, domain: str) -> None:
    """Get threat intelligence for domain."""
    service, output, formatter = probed_a1000(ctx, A1000NetworkService)
    run_network_report(
        output, formatter, "domain", domain, lambda: service.get_domain_report(domain)
    )


@a1000.command()
@click.argument("ip_address")
@click.pass_context
def ip_report(ctx: click.Context, ip_address: str) -> None:
    """Get threat intelligence for IP address."""
    service, output, formatter = probed_a1000(ctx, A1000NetworkService)
    run_network_report(
        output, formatter, "IP", ip_address, lambda: service.get_ip_report(ip_address)
    )
