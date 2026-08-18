"""Analysis, PDF and dynamic-analysis reports."""

from functools import partial
from pathlib import Path
from typing import Any

import click

from rl_cli.cli.commands.a1000._shared import a1000, json_flag, report_format_option
from rl_cli.cli.context import cli_session, display, probed_a1000, run_step
from rl_cli.render.formatters import (
    print_report_summary,
    print_summary_panel,
    print_titanium_report,
)
from rl_cli.render.output import RichOutput
from rl_cli.services.a1000 import A1000ReportService, A1000SampleService, upload_and_get_report
from rl_cli.storage.files import save_private_report

# The two rendered variants of a dynamic-analysis report, spelled once for
# the three commands that fetch, create and poll for one.
_dynamic_format_option = report_format_option(
    ["pdf", "html"], default="pdf", help_text="Report format"
)


def _save_report(output: RichOutput, path: Path, content: Any, *, binary: bool) -> bool:
    """Report the outcome of saving ``content`` to ``path``.

    The 0600, no-symlink, all-or-nothing write and the overwrite warning
    are :func:`~rl_cli.storage.files.save_private_report`'s: how a report
    file has to be written is not something a Click handler should be the
    keeper of. What is left here is the telling.

    A failed write is the destination's problem, not a CLI bug, so it is
    reported as itself rather than through main()'s "Unexpected error".
    There is no fallback to stdout — the pdf variant is bytes, and half a
    fallback is worse than a message naming the path and the reason.
    """
    try:
        save_private_report(path, content, binary=binary, warn=output.warning)
    except OSError as exc:
        output.error(f"Report retrieved but could not be saved to {path}: {exc.strerror or exc}")
        return False
    output.success(f"Report saved to {path}")
    return True


@a1000.command()
@click.argument("hash_value")
@json_flag
@report_format_option(
    ["json", "titanium", "xml", "pdf"],
    default="json",
    help_text="Report variant: json, titanium (TitaniumCore static analysis), pdf; "
    "xml is a deprecated alias for titanium",
)
@click.option(
    "--output-file", type=click.Path(dir_okay=False, path_type=Path), help="Save report to file"
)
@click.pass_context
def report(
    ctx: click.Context,
    hash_value: str,
    show_json: bool,
    report_format: str,
    output_file: Path | None,
) -> None:
    """Get analysis report for a file."""
    service, output, formatter = probed_a1000(ctx, A1000ReportService)

    if report_format == "xml":
        # There is no XML anywhere in this path: "xml" selects the
        # TitaniumCore endpoint, which answers JSON. Still accepted so
        # existing scripts do not break.
        output.warning("--format xml returns JSON; it is the TitaniumCore report")
        output.info("Use --format titanium instead")

    result = run_step(
        output,
        "Fetching report...",
        lambda: service.get_report(hash_value, report_format),
        success=f"Report retrieved for {hash_value}",
        failure="Failed to retrieve report",
    )
    if result is None:
        return

    is_binary = service.report_is_binary(report_format)
    if output_file:
        _save_report(output, output_file, result, binary=is_binary)
    elif is_binary:
        output.info(f"Binary report ({len(result):,} bytes). Use --output-file to save.")
    else:
        # The summary panel only knows the detailed json report; --json
        # asks for the dump of whichever variant was fetched.
        summarisable = report_format == "json" and not show_json
        display(formatter, print_report_summary, result, plain=not summarisable)


@a1000.command()
@click.argument("hash_value")
@_dynamic_format_option
@click.pass_context
def dynamic_report_create(ctx: click.Context, hash_value: str, report_format: str) -> None:
    """Initiate creation of a dynamic-analysis report (requires SHA1 hash)."""
    service, output, formatter = probed_a1000(ctx, A1000ReportService)

    run_step(
        output,
        "Initiating dynamic-analysis report creation...",
        lambda: service.create_dynamic_report(hash_value, report_format),
        success=f"Dynamic {report_format.upper()} report creation initiated",
        failure="Failed to initiate dynamic-analysis report",
        render=formatter.display,
    )


@a1000.command()
@click.argument("hash_value")
@_dynamic_format_option
@click.option(
    "--output-file", type=click.Path(dir_okay=False, path_type=Path), help="Output file path"
)
@click.pass_context
def dynamic_report(
    ctx: click.Context, hash_value: str, report_format: str, output_file: Path | None
) -> None:
    """Download dynamic analysis report (requires SHA1 hash)."""
    service, output, _ = probed_a1000(ctx, A1000ReportService)

    report_content = run_step(
        output,
        "Downloading dynamic analysis report...",
        lambda: service.download_dynamic_report(hash_value, report_format),
        success="Dynamic analysis report retrieved",
        failure="No dynamic analysis report available",
    )
    if not report_content:
        output.info(
            f"Create it first: rl-cli a1000 dynamic-report-create {hash_value} -f {report_format}"
        )
        return

    if not output_file:
        # No default path: a report is not written into wherever the analyst
        # happens to be standing, possibly over an annotated PDF of the same
        # name. Same rule as its sibling ``report -f pdf``, in its words.
        output.info(f"Binary report ({len(report_content):,} bytes). Use --output-file to save.")
        return

    if _save_report(output, output_file, report_content, binary=True):
        output.info(f"Report format: {report_format.upper()}")
        output.info(f"File size: {len(report_content):,} bytes")


@a1000.command()
@click.argument("hash_value")
@_dynamic_format_option
@click.pass_context
def dynamic_report_status(ctx: click.Context, hash_value: str, report_format: str) -> None:
    """Check whether a dynamic-analysis report is ready (requires SHA1 hash)."""
    service, output, formatter = probed_a1000(ctx, A1000ReportService)

    run_step(
        output,
        "Checking dynamic-analysis report status...",
        lambda: service.check_dynamic_analysis_status(hash_value, report_format),
        success="Dynamic-analysis report status retrieved",
        failure="Failed to get dynamic-analysis report status",
        render=formatter.display,
    )


@a1000.command()
@click.argument("hash_value")
@json_flag
@click.pass_context
def summary_report(ctx: click.Context, hash_value: str, show_json: bool) -> None:
    """Get summary report for sample."""
    service, output, formatter = probed_a1000(ctx, A1000ReportService)

    run_step(
        output,
        "Fetching summary report...",
        lambda: service.get_summary_report_v2(hash_value),
        success="Summary report retrieved",
        failure="Failed to get summary report",
        # --json asks for the dump; -o json insists on it whatever the
        # command would rather have printed.
        render=partial(display, formatter, print_summary_panel, plain=show_json),
    )


@a1000.command()
@click.argument("hash_value")
@json_flag
@click.pass_context
def titanium_report(ctx: click.Context, hash_value: str, show_json: bool) -> None:
    """Get Titanium Core report for sample."""
    service, output, formatter = probed_a1000(ctx, A1000ReportService)

    run_step(
        output,
        "Fetching Titanium Core report...",
        lambda: service.get_titanium_core_report_v2(hash_value),
        success="Titanium Core report retrieved",
        failure="Failed to get Titanium Core report",
        # --json asks for the dump; -o json insists on it whatever the
        # command would rather have printed.
        render=partial(display, formatter, print_titanium_report, plain=show_json),
    )


@a1000.command()
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--comment", "-c", help="Optional comment")
@click.option("--get-report", "-r", is_flag=True, help="Get detailed report immediately")
@click.pass_context
def upload_and_analyze(
    ctx: click.Context, file_path: Path, comment: str | None, get_report: bool
) -> None:
    """Upload file and get immediate analysis."""
    # The one command spanning two areas: both go over the one session and
    # report through the one notifier, so the upload and the report cost a
    # single authentication and share the invocation's exit status.
    samples, output, formatter = probed_a1000(ctx, A1000SampleService)
    reports = cli_session(ctx).service(A1000ReportService, output)

    output.info(f"Uploading and analyzing: {file_path.name}")
    if comment:
        output.info(f"Comment: {comment}")
    wanted = "titanium" if get_report else "summary"
    run_step(
        output,
        "Uploading and analyzing...",
        lambda: upload_and_get_report(
            samples, reports, file_path, report_type=wanted, comment=comment
        ),
        success="File analyzed successfully",
        failure="Failed to upload and analyze file",
        render=formatter.display,
    )
