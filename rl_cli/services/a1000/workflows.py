"""Workflows that span more than one area of the appliance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_cli.services.a1000.reports import A1000ReportService
from rl_cli.services.a1000.samples import A1000SampleService


def upload_and_get_report(
    samples: A1000SampleService,
    reports: A1000ReportService,
    file_path: Path,
    report_type: str,
    comment: str | None = None,
) -> dict[str, Any] | None:
    """Upload ``file_path`` then fetch its ``summary`` or ``titanium`` report.

    A function over the two services rather than a method on either: the
    work spans both areas and belongs to neither, and a service that built
    the other one for itself would open a second connection to do it. Hand
    it two services from the same :class:`A1000Session` and the upload and
    the report go over the one connection.

    ``report_type`` names the report, not a file format — the
    ``report_format`` its neighbours take spells json/pdf/html — so those
    spellings are refused here rather than answered with a summary report.
    """
    if report_type not in ("summary", "titanium"):
        samples.output.error(f"Invalid report type '{report_type}'. Expected summary or titanium.")
        return None
    upload_result = samples.upload_file(file_path, comment)
    if not upload_result:
        return None
    # upload_file already put the sample's digest here.
    hash_value = upload_result.get("task_id")
    if not hash_value:
        # Not the bare upload receipt: a caller cannot tell that from a
        # report, and ``run_step`` announces it as "File analyzed
        # successfully".
        samples.output.error("Could not extract hash from upload result")
        return None
    if report_type == "titanium":
        return reports.get_titanium_core_report_v2(hash_value)
    return reports.get_summary_report_v2(hash_value)


def wait_for_uploaded_analysis(
    samples: A1000SampleService, upload_result: dict[str, Any], timeout: int
) -> dict[str, Any] | None:
    """Wait for A1000 to hold an analysis of the sample ``upload_result`` receipts.

    Two appliance steps and the decision between them: read the digest the
    upload receipt carries, then poll the analysis-status endpoint for it.
    Which field of the receipt names the sample is the appliance's business
    and so belongs here, not in a Click handler.

    A receipt with no digest leaves nothing to poll, and skipping the wait
    in silence reads as a wait that succeeded — so it is reported and
    answered ``None``, exactly as a wait that ran out of time is.
    """
    task_id = upload_result.get("task_id")
    if not task_id:
        samples.output.error("Could not extract hash from upload result; cannot wait for analysis")
        return None
    samples.output.info("Waiting for analysis to complete...")
    return samples.wait_for_analysis(task_id, timeout)
