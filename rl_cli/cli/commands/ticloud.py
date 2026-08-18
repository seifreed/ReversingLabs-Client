"""TitaniumCloud CLI commands, over the two services the API's ten families split into.

The file-oriented commands build :class:`TitaniumCloudService` and the
"what else is associated with this" ones build
:class:`TitaniumCloudNetworkService`. The two are peers, and each command
asks for the half it uses.
"""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import click

from rl_cli.cli.commands._downloads import download_to
from rl_cli.cli.commands._shared_inputs import (
    MAX_LIMIT,
    fetch_all_flag,
    hash_inputs,
    limit_option,
    max_results_option,
    output_dir_option,
    partial_answer_notice,
    required_hashes,
)
from rl_cli.cli.context import RenderConsole, display, probed_ticloud, run_network_report, run_step
from rl_cli.render.formatters import print_file_analysis
from rl_cli.services.titanium_cloud import TitaniumCloudNetworkService, TitaniumCloudService

# "Additional" to the positional hashes, which is why the A1000 batch
# commands cannot use this wording: they have no positional hash.
_HASHES_HELP = "Additional hashes to look up"

# The positional argument must not be marked required, or `-f hashes.txt`
# alone — the documented way to triage a batch — cannot be invoked at all.
_HASHES_USAGE = "Pass one or more HASH arguments, -h <hash> (repeatable), or -f <file of hashes>"


@click.group()
def ticloud() -> None:
    """TitaniumCloud API commands for file reputation and threat intelligence."""


def _distinct(hashes: list[str]) -> list[str]:
    """The batch with its repeats dropped, in the order it was named.

    Two spellings of one hash are compared the way they are sent — stripped
    and lower-cased, as ``supported_hash`` normalises them — and the
    spelling named first is kept, so the batch reads as the analyst typed it.
    """
    seen: set[str] = set()
    distinct: list[str] = []
    for hash_value in hashes:
        normalised = hash_value.strip().lower()
        if normalised in seen:
            continue
        seen.add(normalised)
        distinct.append(hash_value)
    return distinct


def _print_reputations(records: list[dict[str, Any]], *, console: RenderConsole) -> None:
    """Grade every record through the one verdict path, however many there are.

    One hash and a batch must be answered at the same fidelity — verdict,
    severity colour, threat name — rather than the batch degrading to a
    table of whatever keys its entries happen to carry.
    """
    for record in records:
        print_file_analysis(record, console=console)


@ticloud.command()
@click.argument("hash_values", metavar="HASH...", nargs=-1)
@hash_inputs(_HASHES_HELP)
@click.pass_context
def reputation(
    ctx: click.Context,
    hash_values: tuple[str, ...],
    hashes: tuple[str, ...],
    hash_file: Path | None,
) -> None:
    """Get file reputation by hash (MD5, SHA1, or SHA256).

    Name several and they are graded in one bulk request rather than one
    metered request each. A batch must be all of one hash type, which is
    what the bulk endpoint queries by. A hash named twice is looked up
    once.
    """
    named = required_hashes(hashes, hash_file, _HASHES_USAGE, positional=hash_values)
    service, output, formatter = probed_ticloud(ctx, TitaniumCloudService)

    batch = _distinct(named)
    if len(batch) < len(named):
        output.info(
            f"{len(named) - len(batch)} duplicate hash(es) dropped; "
            f"{len(batch)} distinct to look up"
        )

    subject = batch[0] if len(batch) == 1 else f"{len(batch)} hashes"
    run_step(
        output,
        f"Fetching file reputation for {subject}...",
        lambda: service.get_reputation_records(batch),
        success=lambda records: (
            f"File reputation retrieved for {batch[0]}"
            if len(batch) == 1
            else f"File reputation retrieved for {len(records)} of {len(batch)} hashes"
        ),
        failure="Failed to retrieve file reputation",
        # A bulk query the endpoint answers with no records is an answer
        # about the batch, not a failed lookup.
        empty=f"TitaniumCloud holds no reputation record for {subject}",
        render=partial(display, formatter, _print_reputations),
        empty_formatter=formatter,
    )


@ticloud.command()
@click.argument("hash_value")
@click.pass_context
def av_scanners(ctx: click.Context, hash_value: str) -> None:
    """Get AV scanner results for a file hash."""
    service, output, formatter = probed_ticloud(ctx, TitaniumCloudService)

    run_step(
        output,
        "Fetching AV scanner results...",
        lambda: service.get_av_scanners(hash_value),
        success=f"AV scanner results retrieved for {hash_value}",
        failure="Failed to retrieve AV scanner results",
        # The rule every lookup in this file follows: a subject
        # TitaniumCloud holds nothing on is answered 200 with nothing, which
        # is an answer and must exit 0, or an enrichment run exits 1 on
        # every clean hash and URL it passes over.
        empty=f"TitaniumCloud holds no AV scanner results for {hash_value}",
        render=formatter.display,
        empty_formatter=formatter,
    )


@ticloud.command()
@click.argument("url")
@click.pass_context
def analyze_url(ctx: click.Context, url: str) -> None:
    """Analyze URL for threats and reputation."""
    service, output, formatter = probed_ticloud(ctx, TitaniumCloudNetworkService)

    # The URL is not checked here: the service refuses one it cannot parse
    # and says so itself. One rule, in the layer that owns the call.
    run_step(
        output,
        "Analyzing URL...",
        lambda: service.analyze_url(url),
        success=f"URL analysis complete for {url}",
        failure="Failed to analyze URL",
        empty=f"TitaniumCloud holds no analysis for {url}",
        render=formatter.display,
        empty_formatter=formatter,
    )


@ticloud.command()
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_context
def upload(ctx: click.Context, file_path: Path) -> None:
    """Upload file to TitaniumCloud for analysis."""
    service, output, formatter = probed_ticloud(ctx, TitaniumCloudService)

    output.info(f"Uploading file: {file_path.name}")
    run_step(
        output,
        "Uploading file...",
        lambda: service.upload_file(file_path),
        success=f"File uploaded successfully: {file_path.name}",
        failure="Failed to upload file",
        render=formatter.display,
    )


@ticloud.command()
@click.argument("query")
@limit_option
@click.pass_context
def search(ctx: click.Context, query: str, limit: int) -> None:
    """Search for samples in TitaniumCloud."""
    service, output, formatter = probed_ticloud(ctx, TitaniumCloudService)

    run_step(
        output,
        f"Searching for: {query}...",
        lambda: service.search_samples(query, limit),
        success=lambda results: f"Found {len(results)} samples",
        failure=f"Search failed for: {query}",
        empty=f"No samples matched: {query}",
        render=formatter.display,
        empty_formatter=formatter,
    )
    partial_answer_notice(output, _more_hits_by(limit), pages_left=service.pages_left_unfetched)


def _more_hits_by(limit: int) -> str:
    """What asks for the hits this search did not bring back.

    Raising ``--limit``, until the limit is the highest this CLI takes:
    advising a bigger one there names a command line the CLI itself refuses
    with exit 2. Past that, the remedy is a query that matches less.
    """
    if limit >= MAX_LIMIT:
        return "narrow the query and search again"
    return f"raise --limit above {limit}"


@ticloud.command()
@click.argument("hash_value")
@click.pass_context
def analysis(ctx: click.Context, hash_value: str) -> None:
    """Get detailed file analysis results."""
    service, output, formatter = probed_ticloud(ctx, TitaniumCloudService)

    run_step(
        output,
        "Fetching file analysis...",
        lambda: service.get_file_analysis(hash_value),
        success=f"File analysis retrieved for {hash_value}",
        failure="Failed to retrieve file analysis",
        empty=f"TitaniumCloud holds no analysis for {hash_value}",
        render=partial(display, formatter, print_file_analysis),
        empty_formatter=formatter,
    )


@ticloud.command()
@click.argument("domain")
@click.pass_context
def domain_report(ctx: click.Context, domain: str) -> None:
    """Get threat intelligence for a domain."""
    service, output, formatter = probed_ticloud(ctx, TitaniumCloudNetworkService)
    run_network_report(
        output, formatter, "domain", domain, lambda: service.get_domain_report(domain)
    )


@ticloud.command()
@click.argument("ip_address")
@click.pass_context
def ip_report(ctx: click.Context, ip_address: str) -> None:
    """Get threat intelligence for an IP address."""
    service, output, formatter = probed_ticloud(ctx, TitaniumCloudNetworkService)
    run_network_report(
        output, formatter, "IP", ip_address, lambda: service.get_ip_report(ip_address)
    )


# The two halves of a pivot, as the command calls them. The subject stays
# positional so that a service method spelling it ``ip``, ``domain``,
# ``url`` or ``uri`` satisfies the same shape; the result is a list of
# whatever the endpoint lists — records for eight of the nine, SHA-1
# strings for ``uri-index``.
type _FirstPage = Callable[[str], list[Any] | None]


class _EveryPage(Protocol):
    """The paging half, which takes the budget by keyword."""

    def __call__(self, subject: str, /, *, max_results: int | None = None) -> list[Any] | None: ...


def _register_pivot(
    command_name: str,
    noun: str,
    subject: str,
    first_page: Callable[[TitaniumCloudNetworkService], _FirstPage],
    every_page: Callable[[TitaniumCloudNetworkService], _EveryPage],
) -> None:
    """Register one of the "what else is associated with this" lookups.

    Nine of them, differing only in the noun, what they are asked about and
    which pair of service methods they call. One factory is what keeps
    ``--all`` meaning the same thing in all nine.

    Each half must arrive as an accessor rather than as a method name, so
    the edge from command to service is one mypy checks rather than a
    ``getattr`` over the ``_aggregated`` suffix convention. The method is
    bound inside the body — ``first_page(service)`` at the moment of the
    call — so stubbing a wrapper on the class still reaches the command.

    ``--max-results`` implies the paging variant: asking for 5000 results is
    asking for as many pages as that takes, and requiring ``--all
    --max-results`` would only invite the unbounded half to be typed alone.

    Either half can answer less than the corpus, so one notice covers both:
    a walk stopped by its cap, and a first page whose envelope stated a
    cursor.
    """

    @ticloud.command(name=command_name, help=f"Find {noun} associated with {subject}.")
    @click.argument("target", metavar=subject.upper().replace(" ", "_"))
    @fetch_all_flag
    @max_results_option("Stop after this many results (implies --all)")
    @click.pass_context
    def lookup(ctx: click.Context, target: str, fetch_all: bool, max_results: int | None) -> None:
        service, output, formatter = probed_ticloud(ctx, TitaniumCloudNetworkService)
        call = (
            partial(every_page(service), target, max_results=max_results)
            if fetch_all or max_results is not None
            else partial(first_page(service), target)
        )

        results = run_step(
            output,
            f"Finding {noun} for {target}...",
            call,
            success=lambda found: f"Found {len(found)} {noun} for {target}",
            failure=f"Failed to look up {noun} for {target}",
            empty=f"No {noun} found for {target}",
            render=formatter.display,
            empty_formatter=formatter,
        )
        partial_answer_notice(
            output,
            "re-run with --all or raise --max-results",
            pages_left=service.pages_left_unfetched,
            collected=results,
            max_results=max_results,
        )


_register_pivot(
    "ip-files",
    "files",
    "IP address",
    lambda service: service.get_files_from_ip,
    lambda service: service.get_files_from_ip_aggregated,
)
_register_pivot(
    "ip-urls",
    "URLs",
    "IP address",
    lambda service: service.get_urls_from_ip,
    lambda service: service.get_urls_from_ip_aggregated,
)
_register_pivot(
    "ip-domains",
    "domains",
    "IP address",
    lambda service: service.get_domains_from_ip,
    lambda service: service.get_domains_from_ip_aggregated,
)
_register_pivot(
    "domain-files",
    "files",
    "domain",
    lambda service: service.get_files_from_domain,
    lambda service: service.get_files_from_domain_aggregated,
)
_register_pivot(
    "domain-urls",
    "URLs",
    "domain",
    lambda service: service.get_urls_from_domain,
    lambda service: service.get_urls_from_domain_aggregated,
)
_register_pivot(
    "domain-ips",
    "IP addresses",
    "domain",
    lambda service: service.get_ips_from_domain,
    lambda service: service.get_ips_from_domain_aggregated,
)
_register_pivot(
    "domain-related",
    "related domains",
    "domain",
    lambda service: service.get_related_domains,
    lambda service: service.get_related_domains_aggregated,
)
_register_pivot(
    "url-files",
    "files",
    "URL",
    lambda service: service.get_files_from_url,
    lambda service: service.get_files_from_url_aggregated,
)
_register_pivot(
    "uri-index",
    "sample hashes",
    "URI",
    lambda service: service.get_uri_index,
    lambda service: service.get_uri_index_aggregated,
)


@ticloud.command()
@click.argument("hash_value")
@click.pass_context
def download_status(ctx: click.Context, hash_value: str) -> None:
    """Check whether a sample is available for download."""
    service, output, formatter = probed_ticloud(ctx, TitaniumCloudService)

    run_step(
        output,
        "Checking download availability...",
        lambda: service.get_download_status(hash_value),
        success=f"Download status for {hash_value}",
        failure="Failed to check download status",
        empty=f"TitaniumCloud holds no download status for {hash_value}",
        render=formatter.display,
        empty_formatter=formatter,
    )


@ticloud.command()
@click.argument("hash_value")
@output_dir_option
@click.pass_context
def download(ctx: click.Context, hash_value: str, output_dir: Path) -> None:
    """Download a sample from TitaniumCloud."""
    service, output, _ = probed_ticloud(ctx, TitaniumCloudService)

    # The service owns the rule about which hashes these endpoints take —
    # fewer types than the A1000's — and names the refusal itself.
    download_to(
        output,
        hash_value,
        output_dir,
        validate=service.supported_hash,
        fetch=service.download_sample,
    )
