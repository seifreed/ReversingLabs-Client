"""Sample submission, retrieval, listing and bulk operations."""

from functools import partial
from pathlib import Path
from typing import Any

import click

from rl_cli.cli.commands._downloads import download_to, fetch_into
from rl_cli.cli.commands._shared_inputs import (
    MAX_LIMIT,
    confirmed,
    hash_inputs,
    limit_option,
    output_dir_option,
    required_hashes,
    yes_flag,
)
from rl_cli.cli.commands.a1000._shared import (
    a1000,
    capped_listing,
    partial_answer_notice,
)
from rl_cli.cli.context import cli_objects, display, probed_a1000, run_step
from rl_cli.models.validators import normalize_hash, validate_hash
from rl_cli.render.formatters import (
    print_analysis_status,
    print_extracted_files_table,
    print_reanalyze_results_table,
    print_samples_panels,
    print_search_results_table,
)
from rl_cli.render.output import RichOutput
from rl_cli.services.a1000 import (
    A1000MetadataService,
    A1000SampleService,
    wait_for_uploaded_analysis,
)
from rl_cli.services.a1000.samples import ReanalysisBatch, unused_search_inputs


def _validated_hash(output: RichOutput, hash_value: str) -> str | None:
    """The hash to work from, or ``None`` after refusing it.

    Must stay ahead of the point where ``download`` and ``extracted
    --download-all`` create the destination directory and print "about to
    write live malware", so a typo leaves no empty directory and no warning
    about a sample that was never going to be fetched. The service validates
    too; this is about the order, not about trusting it less.

    It answers the normalized hash because the file name, the extracted
    directory and the malware warning are all built from it: a hash pasted
    with padding or in upper case would otherwise name a path that does not
    match the sample the appliance was asked for.
    """
    normalised = normalize_hash(hash_value)
    if normalised is None:
        output.error(f"Invalid hash format: {hash_value}")
    return normalised


def _offer_the_remaining_pages(
    service: A1000SampleService, output: RichOutput, *, limit: int, page: int = 1
) -> None:
    """Say how to fetch the pages the appliance left behind, if it did.

    The service reports that pages were left behind; how to ask for the rest
    is this layer's word, because ``--limit`` and ``--page`` are this
    layer's options — a service naming them could not be reused by a caller
    that has neither.

    The remedy has to fit the fetch that was cut short: "raise --limit" is
    no advice to a walk that already spanned pages, where the next page is
    what helps, and at
    :data:`~rl_cli.cli.commands._shared_inputs.MAX_LIMIT` there is no bigger
    ``--limit`` to raise it to — advising one names a command line this CLI
    refuses with exit 2.

    Both commands offer it because both take one page by default: ``list``
    reaches Advanced Search through ``list_samples``.
    """
    if page > 1:
        remedy = f"pass --page {page + 1}"
    elif limit >= MAX_LIMIT:
        remedy = "narrow the search and run it again"
    else:
        remedy = f"raise --limit above {limit}"
    partial_answer_notice(output, remedy, pages_left=service.pages_left_unfetched)


def _report_reanalysis(
    output: RichOutput, results: list[dict[str, Any]], *, submitted: int
) -> None:
    """Say what the appliance made of the answer, and fail the run if it took nothing.

    Every number here must come from :class:`ReanalysisBatch`, which is also
    what the Status column below draws from: one reading of an entry rather
    than one per layer that reports it, so the sentence and the rows cannot
    disagree.

    ``submitted`` is what was asked for rather than what was answered, and
    the two differing is a fact no other line states.
    """
    batch = ReanalysisBatch.of(results)
    if not batch.accepted:
        output.error(f"Reanalysis refused for all {batch.answered} samples")
    elif batch.refused:
        output.success(f"Reanalysis started for {batch.accepted} of {batch.answered} samples")
        output.warning(f"{batch.refused} samples were refused by the appliance")
    else:
        output.success(f"Reanalysis started for {batch.accepted} samples")
    if batch.answered < submitted:
        output.warning(
            f"The appliance answered for {batch.answered} of the {submitted} samples submitted"
        )


@a1000.command()
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--comment", "-c", help="Optional comment for the submission")
@click.option("--wait", "-w", is_flag=True, help="Wait for analysis to complete")
@click.option(
    "--timeout",
    "-t",
    type=click.IntRange(min=0),
    default=300,
    help="Analysis timeout in seconds",
)
@click.pass_context
def upload(
    ctx: click.Context, file_path: Path, comment: str | None, wait: bool, timeout: int
) -> None:
    """Upload file to A1000 for analysis."""
    service, output, formatter = probed_a1000(ctx, A1000SampleService)

    output.info(f"Uploading file: {file_path.name}")
    result = run_step(
        output,
        "Uploading file...",
        lambda: service.upload_file(file_path, comment),
        success=f"File uploaded successfully: {file_path.name}",
        failure="Failed to upload file",
        # With ``--wait`` the receipt is a step towards the answer, not the
        # answer: rendering it here would put a second document on stdout
        # ahead of the analysis status below, and ``-o json | jq`` reads one.
        render=None if wait else formatter.display,
    )

    if result and wait:
        final_status = wait_for_uploaded_analysis(service, result, timeout)

        if final_status:
            # Not "Analysis completed": the only progress endpoint answers
            # "processed"/"not_found" about the hash, so for a file the
            # appliance already held it says "processed" on the first poll,
            # out of the previous analysis. Claim what the answer establishes.
            output.success("Analysis results are available")
            output.info("For a file A1000 already had, these may be from its earlier analysis")
            formatter.display(final_status)
        # No else: the workflow reports a receipt with no digest, and
        # wait_for_analysis reports the timeout and the give-up, both as
        # errors. Repeating the timeout here as a warning is how a --wait
        # that never landed ends up exiting 0.


@a1000.command()
@click.argument("hash_or_task_id")
@click.pass_context
def status(ctx: click.Context, hash_or_task_id: str) -> None:
    """Check analysis status for a hash or task ID."""
    if not validate_hash(hash_or_task_id):
        # The analysis-status endpoint posts its argument as a hash and
        # validates nothing, so a task id comes back as an empty result set
        # and a bare "Failed to retrieve task status". The only task ids this
        # CLI hands out come from submit-url, which url-status reads.
        _, output, _ = cli_objects(ctx)
        output.error(f"Not a hash: {hash_or_task_id}")
        output.info(f"For a submit-url task ID, use: rl-cli a1000 url-status {hash_or_task_id}")
        return

    metadata, output, formatter = probed_a1000(ctx, A1000MetadataService)
    run_step(
        output,
        "Checking sample status...",
        lambda: metadata.get_classification(hash_or_task_id),
        success=f"Sample status for: {hash_or_task_id}",
        failure="Sample not found or analysis not available",
        render=partial(display, formatter, print_analysis_status),
    )


@a1000.command()
@click.argument("hash_value")
@yes_flag
@click.pass_context
def delete(ctx: click.Context, hash_value: str, yes: bool) -> None:
    """Delete sample from A1000."""
    service, output, _ = probed_a1000(ctx, A1000SampleService)

    if not confirmed(
        output,
        yes,
        f"Are you sure you want to delete sample {hash_value}?",
        cancelled="Deletion cancelled",
    ):
        return

    deleted = run_step(
        output,
        "Deleting sample...",
        lambda: service.delete_sample(hash_value),
        success=f"Sample deleted: {hash_value}",
        failure="Failed to delete sample",
    )
    if not deleted:
        # This command uses the single-sample endpoint, which an API token
        # may not be permitted to call even when the bulk one behind
        # batch-delete accepts the same account.
        output.info(f"If this is a rights error, try: rl-cli a1000 batch-delete -h {hash_value}")


@a1000.command()
@click.argument("hash_value")
@click.pass_context
def reanalyze(ctx: click.Context, hash_value: str) -> None:
    """Reanalyze existing sample."""
    # No --wait: the only progress endpoint knows two answers about a hash,
    # "processed" and "not_found". A sample you can reanalyze is already
    # processed, so a wait would report "completed" on its first poll
    # whatever the appliance is still doing.
    service, output, formatter = probed_a1000(ctx, A1000SampleService)

    run_step(
        output,
        "Initiating reanalysis...",
        lambda: service.reanalyze_sample(hash_value),
        success=f"Reanalysis started for {hash_value}",
        failure="Failed to start reanalysis",
        render=formatter.display,
    )


@a1000.command()
@click.argument("hash_value")
@output_dir_option
@click.pass_context
def download(ctx: click.Context, hash_value: str, output_dir: Path) -> None:
    """Download original sample from A1000."""
    service, output, _ = probed_a1000(ctx, A1000SampleService)

    download_to(
        output,
        hash_value,
        output_dir,
        validate=partial(_validated_hash, output),
        fetch=service.download_sample,
    )


@a1000.command()
@click.argument("hash_value")
@click.option("--download-all", "-a", is_flag=True, help="Download all extracted files")
@click.option(
    "--all",
    "fetch_all",
    is_flag=True,
    hidden=True,
    help="Deprecated no-op: the listing already includes every file",
)
@output_dir_option
@click.pass_context
def extracted(
    ctx: click.Context, hash_value: str, download_all: bool, fetch_all: bool, output_dir: Path
) -> None:
    """List or download files extracted from sample."""
    service, output, formatter = probed_a1000(ctx, A1000SampleService)

    sample = _validated_hash(output, hash_value)
    if sample is None:
        return

    if fetch_all:
        # The endpoint returns every extracted file in one response when
        # asked for no page, which is what the listing below does. Kept
        # accepted so existing scripts do not break.
        output.warning(
            "--all is a deprecated no-op; the listing already covers every extracted file"
        )

    extracted_files = run_step(
        output,
        "Fetching extracted files list...",
        lambda: service.list_extracted_files(sample),
        success=lambda files: f"Found {len(files)} extracted files",
        failure="Failed to list extracted files",
        empty="No extracted files for this sample",
        empty_formatter=formatter,
    )
    if not extracted_files:
        return

    if not download_all:
        render = capped_listing(output, print_extracted_files_table)
        display(formatter, render, extracted_files)
        output.info("Use --download-all to download all extracted files")
        return

    extracted_dir = output_dir / f"extracted_{sample[:16]}"
    output.info(f"Downloading to {extracted_dir}")
    # The third command that writes live malware. The warning comes before
    # the write for the same reason as `download` above, and on the same
    # terms — no classification is consulted for any of these files.
    fetch_into(
        output,
        extracted_dir,
        private_root=output_dir,
        warnings=[
            f"WARNING: about to write files that may contain malware to {extracted_dir}. "
            "Handle with caution!"
        ],
        spinner="Downloading extracted files...",
        success=f"Extracted files downloaded to {extracted_dir}",
        failure="Failed to download extracted files",
        fetch=lambda: service.download_extracted_files(sample, extracted_dir),
    )


@a1000.command(name="list")
@limit_option
@click.pass_context
def list_samples(ctx: click.Context, limit: int) -> None:
    """List samples in A1000."""
    service, output, formatter = probed_a1000(ctx, A1000SampleService)

    render = capped_listing(output, print_samples_panels)
    run_step(
        output,
        "Fetching samples...",
        lambda: service.list_samples(limit),
        success=lambda samples: f"Found {len(samples)} samples",
        failure="Failed to retrieve the sample list",
        empty="No samples found",
        render=partial(display, formatter, render),
        empty_formatter=formatter,
    )
    _offer_the_remaining_pages(service, output, limit=limit)


@a1000.command(
    epilog="""\
\b
Examples:
  rl-cli a1000 search --malicious
  rl-cli a1000 search -q "available:true"
  rl-cli a1000 search -q "classification:malicious"
\b
Query fields:
  available:true            All available samples
  classification:malicious  Malicious samples
  classification:suspicious Suspicious samples
  classification:clean      Clean samples
  riskscore:10              High risk samples
  sha256:<hash>             Specific sample
""",
)
@click.option("--query", "-q", help="Search query (see the examples below)")
@click.option("--malicious", is_flag=True, help="Search for malicious files")
@click.option("--clean", is_flag=True, help="Search for clean files")
@limit_option
@click.option("--page", "-p", type=click.IntRange(min=1), default=1, help="Page number")
@click.pass_context
def search(
    ctx: click.Context, query: str, malicious: bool, clean: bool, limit: int, page: int
) -> None:
    """Advanced search for samples."""
    service, output, formatter = probed_a1000(ctx, A1000SampleService)

    # Named before the resolved query replaces the arguments it resolved,
    # and warned about: nothing else durable states which flags survived
    # the resolution, since the spinner showing the query is transient.
    ignored = unused_search_inputs(query, malicious=malicious, clean=clean)
    query = service.build_search_query(query, malicious=malicious, clean=clean)
    if ignored:
        output.warning(f"Ignoring {', '.join('--' + name for name in ignored)}: searching {query}")

    render = capped_listing(output, print_search_results_table)
    run_step(
        output,
        f"Searching: {query}",
        lambda: service.advanced_search(query, limit, page),
        # The query is in the answer and not only in the erased spinner,
        # so that the line says what it answered.
        success=lambda results: f"Found {len(results)} samples matching {query}",
        failure=f"Search failed: {query}",
        empty=f"No samples matched: {query}",
        render=partial(display, formatter, render),
        empty_formatter=formatter,
    )
    _offer_the_remaining_pages(service, output, limit=limit, page=page)


@a1000.command()
@hash_inputs("Individual hashes to delete")
@yes_flag
@click.pass_context
def batch_delete(
    ctx: click.Context, hash_file: Path | None, hashes: tuple[str, ...], yes: bool
) -> None:
    """Delete multiple samples."""
    hash_list = required_hashes(
        hashes, hash_file, "Pass -h <hash> (repeatable) or -f <file of hashes>"
    )

    service, output, _ = probed_a1000(ctx, A1000SampleService)

    if not confirmed(output, yes, f"Delete {len(hash_list)} samples?"):
        return

    # The bulk endpoint answers for the whole batch, not per sample, and
    # the appliance's own completion message says finishing the removal
    # task does not guarantee every sample was removed — one submitted by
    # several users survives it. So acceptance of the batch is all we know.
    #
    # Not ``run_step``: it reads ``None`` and a count of zero as one
    # outcome, and for this endpoint they are opposite facts. ``None`` is a
    # removal the appliance was never asked for — a batch it cannot take,
    # or a call that did not land, and the service has already said which.
    with output.progress_spinner(f"Deleting {len(hash_list)} samples..."):
        accepted = service.batch_delete_samples(hash_list)

    if accepted is None:
        output.error(f"Removal was never submitted; none of the {len(hash_list)} samples was sent")
        return
    # The count is the service's, which for this endpoint is the size of the
    # batch it posted: the appliance answers 202 for the whole task and names
    # no sample. The line says "accepted", not "removed", for that reason.
    output.success(f"Removal accepted for {accepted} samples")
    output.info("Confirm with 'rl-cli a1000 search -q sha256:<hash>'")


@a1000.command()
@hash_inputs("Individual hashes to reanalyze")
@click.option("--titanium-core/--no-titanium-core", default=True, help="Use Titanium Core analysis")
@click.option(
    "--titanium-cloud/--no-titanium-cloud", default=True, help="Use Titanium Cloud analysis"
)
@click.pass_context
def batch_reanalyze(
    ctx: click.Context,
    hash_file: Path | None,
    hashes: tuple[str, ...],
    titanium_core: bool,
    titanium_cloud: bool,
) -> None:
    """Reanalyze multiple samples with Titanium Core and Cloud."""
    hash_list = required_hashes(
        hashes,
        hash_file,
        "Pass -h <hash1> -h <hash2>, or -f hashes.txt",
    )

    engines = [
        name
        for name, wanted in (("Titanium Core", titanium_core), ("Titanium Cloud", titanium_cloud))
        if wanted
    ]
    if not engines:
        raise click.UsageError(
            "Nothing to do: both --no-titanium-core and --no-titanium-cloud given"
        )

    service, output, formatter = probed_a1000(ctx, A1000SampleService)

    output.info(f"Reanalyzing {len(hash_list)} samples")
    output.info(f"Analysis engines: {', '.join(engines)}")

    # Not ``run_step``: it reports an answer as a success, and an answer is
    # not one here — the appliance replies 200 with a per-sample refusal
    # inside, so a batch it turned down whole has to exit non-zero and be
    # drawn anyway. ``run_step`` renders on the success path alone, and the
    # refusals are the rows an analyst most needs to read.
    with output.progress_spinner(f"Submitting {len(hash_list)} samples for reanalysis..."):
        results = service.batch_reanalyze_samples(
            hash_list, titanium_core=titanium_core, titanium_cloud=titanium_cloud
        )

    if not results:
        # ``None`` is a failed call; an empty answer is a batch the
        # appliance said nothing at all about. Neither started anything.
        output.error("Failed to start reanalysis")
        return

    _report_reanalysis(output, results, submitted=len(hash_list))
    render = capped_listing(output, print_reanalyze_results_table)
    display(formatter, render, results)
