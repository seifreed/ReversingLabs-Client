"""A1000 sample, report, tag and network commands, exercised through the click layer.

The yara, download, config and ticloud commands have files of their own,
and the sweeps over the whole command matrix live in
tests/test_command_matrix.py.
"""

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner

import rl_cli.cli.commands
from rl_cli.cli.commands._shared_inputs import MAX_LIMIT, partial_answer_notice
from rl_cli.cli.commands.a1000 import a1000
from rl_cli.cli.commands.a1000 import samples as samples_module
from rl_cli.cli.main import cli
from rl_cli.models.payload import ReanalysisOutcome
from rl_cli.render.output import OutputFormat, OutputFormatter, RichOutput
from rl_cli.services.a1000 import (
    A1000MetadataService,
    A1000NetworkService,
    A1000ReportService,
    A1000SampleService,
    A1000Session,
    A1000YaraService,
)
from rl_cli.services.a1000 import samples as sample_service_module
from rl_cli.services.availability import APIAvailabilityChecker
from tests.cli_support import (
    A1000_SERVICES,
    SHA1,
    SHA256,
    commands_offering_yes,
    commands_that_ask,
    flat,
    invoke,
    make_context,
    stub_client,
    stub_response,
    stub_service,
)
from tests.conftest import sdk_response


@pytest.fixture
def ctx_obj(tmp_path):
    return make_context(tmp_path)


class TestA1000Commands:
    @pytest.mark.parametrize(
        "args",
        [
            ["upload", __file__, "--wait", "--timeout", "-1"],
            ["search", "--page", "0"],
            ["yara-repo-list", "--page", "0"],
            ["yara-repo-list", "--page-size", "0"],
        ],
    )
    def test_paging_and_wait_budgets_must_be_positive(self, ctx_obj, args):
        result = invoke(a1000, args, ctx_obj)

        assert result.exit_code == 2
        assert "Invalid value" in result.output

    def test_report_success(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(
            A1000ReportService, "get_report", lambda self, h, fmt="json": {"sha256": h}
        )
        result = invoke(a1000, ["report", SHA256], ctx_obj)
        assert result.exit_code == 0
        assert SHA256 in result.output

    def test_report_failure_is_reported_with_the_group_invoked_alone(self, ctx_obj, monkeypatch):
        """The non-zero exit status comes from ``cli.result_callback``.

        Invoking the ``a1000`` group directly bypasses it, so this pins the
        message only; ``tests/test_main_cli.py`` pins the exit status.
        """
        monkeypatch.setattr(A1000ReportService, "get_report", lambda self, h, fmt="json": None)
        result = invoke(a1000, ["report", SHA256], ctx_obj)
        assert result.exit_code == 0
        assert "Failed" in result.output

    def test_delete_asks_for_confirmation(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(A1000SampleService, "delete_sample", lambda self, h: True)
        result = invoke(a1000, ["delete", SHA256], ctx_obj, input="n\n")
        assert result.exit_code == 0
        assert "cancelled" in result.output.lower()

    def test_delete_confirmed(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(A1000SampleService, "delete_sample", lambda self, h: True)
        result = invoke(a1000, ["delete", SHA256], ctx_obj, input="y\n")
        assert result.exit_code == 0
        assert "deleted" in result.output.lower()

    def test_status_routes_hash_to_classification(self, ctx_obj, monkeypatch):
        calls = []

        def classify(self: A1000MetadataService, h: str) -> dict[str, str]:
            calls.append(h)
            return {"classification": "malicious"}

        monkeypatch.setattr(A1000MetadataService, "get_classification", classify)
        result = invoke(a1000, ["status", SHA256], ctx_obj)
        assert result.exit_code == 0
        assert calls == [SHA256]


class TestWaitingIsOnlyOfferedWhereItWorks:
    """``--wait`` polls the file analysis status endpoint by hash."""

    def test_upload_wait_polls_the_uploaded_hash(self, ctx_obj, monkeypatch, tmp_path):
        polled: list[tuple] = []
        sample = tmp_path / "sample.bin"
        sample.write_bytes(b"x")
        monkeypatch.setattr(
            A1000SampleService, "upload_file", lambda self, path, comment=None: {"task_id": SHA256}
        )

        def wait(self: A1000SampleService, task_id: str, timeout: int = 300) -> dict[str, str]:
            polled.append((task_id, timeout))
            return {"status": "processed"}

        monkeypatch.setattr(A1000SampleService, "wait_for_analysis", wait)

        result = invoke(a1000, ["upload", str(sample), "--wait", "-t", "7"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert polled == [(SHA256, 7)], "--wait stopped waiting"

    def test_upload_wait_puts_one_document_on_stdout_not_two(self, ctx_obj, monkeypatch, tmp_path):
        """``-o json | jq`` reads one document; ``--wait`` was writing two.

        The upload receipt and the final analysis status were both rendered
        to stdout, so a machine-format run emitted two concatenated
        documents. Under ``--wait`` the status is the answer and the receipt
        is a step towards it, so only the status lands on stdout.
        """
        sample = tmp_path / "sample.bin"
        sample.write_bytes(b"x")
        monkeypatch.setattr(
            A1000SampleService,
            "upload_file",
            lambda self, path, comment=None: {"task_id": SHA256, "receipt": True},
        )
        monkeypatch.setattr(
            A1000SampleService,
            "wait_for_analysis",
            lambda self, task_id, timeout=300: {"status": "processed"},
        )

        result = invoke(a1000, ["upload", str(sample), "--wait"], ctx_obj)

        assert json.loads(result.stdout) == {"status": "processed"}

    def test_upload_wait_says_so_when_the_receipt_carries_no_hash(
        self, ctx_obj, monkeypatch, tmp_path
    ):
        """A receipt without a digest leaves nothing to poll.

        Skipping the wait in silence — and exiting 0 — told the user the
        analysis had been waited for.
        """
        sample = tmp_path / "sample.bin"
        sample.write_bytes(b"x")
        monkeypatch.setattr(
            A1000SampleService, "upload_file", lambda self, path, comment=None: {"message": "ok"}
        )
        monkeypatch.setattr(
            A1000SampleService,
            "wait_for_analysis",
            lambda self, task_id, timeout=300: pytest.fail("there is no hash to wait on"),
        )

        result = invoke(a1000, ["upload", str(sample), "--wait"], ctx_obj)

        assert "Could not extract hash from upload result" in result.output
        # ``cli.result_callback`` turns this into the non-zero exit status;
        # invoking the group alone bypasses it.
        assert ctx_obj.output.status.failed

    def test_reanalyze_offers_no_wait_it_cannot_perform(self, ctx_obj):
        """The status endpoint answers "processed" for any known sample.

        A sample you can reanalyze is one, so a wait would report success
        on its first poll whatever the appliance is still doing.
        """
        result = invoke(a1000, ["reanalyze", SHA256, "--wait"], ctx_obj)

        assert result.exit_code != 0
        assert "no such option" in result.output.lower()


class TestReportOutput:
    """PDF is binary; json and xml (TitaniumCore) come back as structures."""

    def test_dash_o_is_not_a_filename(self, ctx_obj, monkeypatch, tmp_path):
        """`report <hash> -o json` used to save the report to a file named "json"."""
        monkeypatch.setattr(A1000ReportService, "get_report", lambda self, h, fmt="json": {"x": 1})
        monkeypatch.chdir(tmp_path)
        result = invoke(a1000, ["report", SHA256, "-o", "json"], ctx_obj)
        assert result.exit_code != 0
        assert "no such option" in result.output.lower()
        assert not (tmp_path / "json").exists()

    def test_xml_report_written_as_text_not_bytes(self, ctx_obj, monkeypatch, tmp_path):
        monkeypatch.setattr(
            A1000ReportService, "get_report", lambda self, h, fmt="json": {"ticore": {"x": 1}}
        )
        out = tmp_path / "report.xml"
        result = invoke(a1000, ["report", SHA256, "-f", "xml", "--output-file", str(out)], ctx_obj)
        assert result.exit_code == 0, result.output
        assert "ticore" in out.read_text(encoding="utf-8")

    def test_pdf_report_written_as_bytes(self, ctx_obj, monkeypatch, tmp_path):
        monkeypatch.setattr(
            A1000ReportService, "get_report", lambda self, h, fmt="json": b"%PDF-1.4"
        )
        out = tmp_path / "report.pdf"
        result = invoke(a1000, ["report", SHA256, "-f", "pdf", "--output-file", str(out)], ctx_obj)
        assert result.exit_code == 0, result.output
        assert out.read_bytes() == b"%PDF-1.4"

    def test_a_binary_report_without_an_output_file_says_how_to_save_it(self, ctx_obj, monkeypatch):
        """A pdf to stdout would be a wall of bytes, so it is described, not dumped."""
        monkeypatch.setattr(
            A1000ReportService, "get_report", lambda self, h, fmt="json": b"%PDF-1.4 body"
        )

        result = invoke(a1000, ["report", SHA256, "-f", "pdf"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "Binary report" in flat(result)
        assert "--output-file" in flat(result)

    def test_xml_report_displayed_without_output_file(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(
            A1000ReportService, "get_report", lambda self, h, fmt="json": {"ticore": {"x": 1}}
        )
        result = invoke(a1000, ["report", SHA256, "-f", "xml"], ctx_obj)
        assert result.exit_code == 0
        assert "ticore" in result.output

    def test_titanium_is_the_accurate_name_for_the_xml_variant(self, ctx_obj, monkeypatch):
        seen: list[str] = []

        def report(self: A1000ReportService, h: str, fmt: str = "json") -> dict[str, object]:
            seen.append(fmt)
            return {"ticore": {"x": 1}}

        monkeypatch.setattr(A1000ReportService, "get_report", report)
        result = invoke(a1000, ["report", SHA256, "-f", "titanium"], ctx_obj)
        assert result.exit_code == 0, result.output
        assert seen == ["titanium"]

    def test_legacy_xml_still_works_but_is_relabelled(self, ctx_obj, monkeypatch):
        """It never emitted XML, so scripts asking for it get a heads-up.

        The alias itself belongs to the service; the CLI only warns.
        """
        seen: list[str] = []

        def report(self: A1000ReportService, h: str, fmt: str = "json") -> dict[str, object]:
            seen.append(fmt)
            return {"ticore": {"x": 1}}

        monkeypatch.setattr(A1000ReportService, "get_report", report)
        result = invoke(a1000, ["report", SHA256, "-f", "xml"], ctx_obj)
        assert result.exit_code == 0, result.output
        assert seen == ["xml"]
        assert "TitaniumCore report" in result.output


class TestADynamicReportThatCannotBeSavedSaysSo:
    """A write that fails is the destination's problem, reported not swallowed."""

    def test_a_failed_write_skips_the_success_lines(self, ctx_obj, monkeypatch, tmp_path):
        monkeypatch.setattr(
            A1000ReportService,
            "download_dynamic_report",
            lambda self, h, fmt: b"%PDF-1.4 body",
        )

        def boom(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr("rl_cli.cli.commands.a1000.reports.save_private_report", boom)

        result = invoke(
            a1000,
            ["dynamic-report", SHA1, "--output-file", str(tmp_path / "r.pdf")],
            ctx_obj,
        )

        assert result.exit_code == 0, result.output
        assert "could not be saved" in flat(result)
        assert "File size" not in flat(result)


class TestUploadAndAnalyzeEchoesItsComment:
    """The optional comment is echoed before the work, so the run records it."""

    def test_a_comment_is_reported(self, ctx_obj, monkeypatch, tmp_path):
        sample = tmp_path / "sample.bin"
        sample.write_bytes(b"MZ")
        monkeypatch.setattr(
            "rl_cli.cli.commands.a1000.reports.upload_and_get_report",
            lambda *args, **kwargs: {"status": "done"},
        )

        result = invoke(a1000, ["upload-and-analyze", str(sample), "-c", "triage note"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "Comment: triage note" in flat(result)


class TestNothingFoundIsNotTheSameAsFailed:
    """A service answering ``None`` could not do the work; ``[]`` did it."""

    def test_none_is_an_error(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(A1000SampleService, "list_samples", lambda self, limit=100: None)
        result = invoke(a1000, ["list"], ctx_obj)
        assert "✗" in result.output
        assert ctx_obj.output.status.failed

    def test_empty_is_only_a_warning(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(A1000SampleService, "list_samples", lambda self, limit=100: [])
        result = invoke(a1000, ["list"], ctx_obj)
        assert "⚠" in result.output
        assert not ctx_obj.output.status.failed


class TestListEmitsWholeRecords:
    """The panel's column widths must not reach the machine-readable formats."""

    def test_json_output_keeps_the_whole_filename(self, ctx_obj, monkeypatch):
        entry = {
            "sha256": SHA256,
            "file_names": ["a-really-long-sample-name-the-panel-would-shorten.exe"],
            "classification": "malicious",
            "sample_type": "PE32 executable (GUI) Intel 80386, for MS Windows",
        }
        monkeypatch.setattr(A1000SampleService, "list_samples", lambda self, limit=100: [entry])
        result = invoke(a1000, ["list"], ctx_obj)
        assert json.loads(result.stdout) == [entry]


class TestEmptyIsNotWordedLikeAFailure:
    """A 500 used to print "No URLs found" right under the real error."""

    def test_failed_lookup_does_not_claim_nothing_matched(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(A1000NetworkService, "get_urls_from_ip", lambda self, ip: None)
        result = invoke(a1000, ["ip-urls", "1.2.3.4"], ctx_obj)
        assert "Failed to look up URLs" in result.output
        assert "No URLs found" not in result.output

    def test_empty_lookup_still_says_nothing_matched(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(A1000NetworkService, "get_urls_from_ip", lambda self, ip: [])
        result = invoke(a1000, ["ip-urls", "1.2.3.4"], ctx_obj)
        assert "⚠" in result.output
        assert "No URLs found for this IP" in result.output


class TestTruncationNotice:
    """The note used to quote the cap, which is not what got drawn.

    Result arrays carry entries some of these tables skip, so a run that
    drew 12 of 22 rows announced "Showing 20 of 22", and one that drew 10
    of 15 — under a cap of 20 — announced nothing at all.

    The reanalysis table is no longer one of the tables that skip: it
    draws an entry it cannot read as a refused row rather than dropping
    it, so its notice counts every row up to the cap. The claim each test
    here makes is unchanged — the notice counts the rows the table drew —
    which is why the number moves with the renderer rather than the
    assertion loosening to accept either.
    """

    def _results(self, dicts: int, strings: int) -> list:
        return [{"sha256": f"{index:064x}"} for index in range(dicts)] + ["skipped"] * strings

    def _invoke(self, ctx_obj, monkeypatch, service, method, args, results):
        monkeypatch.setattr(service, method, lambda self, *a, **kw: results)
        ctx_obj = replace(ctx_obj, formatter=OutputFormatter(OutputFormat.RICH))
        return invoke(a1000, args, ctx_obj)

    def test_search_counts_the_rows_the_table_drew(self, ctx_obj, monkeypatch):
        result = self._invoke(
            ctx_obj,
            monkeypatch,
            A1000SampleService,
            "advanced_search",
            ["search", "-q", "available:true"],
            self._results(dicts=12, strings=10),
        )
        assert result.exit_code == 0, result.output
        assert "Showing 12 of 22" in result.output

    def test_search_notices_truncation_below_the_cap(self, ctx_obj, monkeypatch):
        """10 drawn of 15 results is truncation, even though 15 < 20."""
        result = self._invoke(
            ctx_obj,
            monkeypatch,
            A1000SampleService,
            "advanced_search",
            ["search", "-q", "available:true"],
            self._results(dicts=10, strings=5),
        )
        assert result.exit_code == 0, result.output
        assert "Showing 10 of 15" in result.output

    def test_yara_matches_counts_the_rows_the_table_drew(self, ctx_obj, monkeypatch):
        result = self._invoke(
            ctx_obj,
            monkeypatch,
            A1000YaraService,
            "get_yara_matches",
            ["yara-matches", "ruleset"],
            self._results(dicts=12, strings=10),
        )
        assert result.exit_code == 0, result.output
        assert "Showing 12 of 22" in result.output

    def test_batch_reanalyze_counts_the_rows_the_table_drew(self, ctx_obj, monkeypatch):
        """Ten of eighteen: the table now fills its cap, drawing refusals too."""
        result = self._invoke(
            ctx_obj,
            monkeypatch,
            A1000SampleService,
            "batch_reanalyze_samples",
            ["batch-reanalyze", "-h", SHA256],
            self._results(dicts=8, strings=10),
        )
        assert result.exit_code == 0, result.output
        assert "Showing 10 of 18" in result.output

    def test_extracted_files_say_it_the_same_way(self, ctx_obj, monkeypatch):
        """It used to grow an in-table "... and N more" tail naming no way to see them."""
        result = self._invoke(
            ctx_obj,
            monkeypatch,
            A1000SampleService,
            "list_extracted_files",
            ["extracted", SHA256],
            [{"filename": f"stage{index}.bin"} for index in range(25)],
        )
        assert result.exit_code == 0, result.output
        assert "Showing 20 of 25" in result.output
        assert "more extracted files" not in result.output

    def test_yara_list_says_it_the_same_way(self, ctx_obj, monkeypatch):
        result = self._invoke(
            ctx_obj,
            monkeypatch,
            A1000YaraService,
            "list_yara_rulesets",
            ["yara-list"],
            [{"name": f"ruleset-{index}"} for index in range(25)],
        )
        assert result.exit_code == 0, result.output
        assert "Showing 20 of 25" in result.output
        assert "more rulesets" not in result.output

    def test_nothing_is_said_when_every_result_was_drawn(self, ctx_obj, monkeypatch):
        result = self._invoke(
            ctx_obj,
            monkeypatch,
            A1000SampleService,
            "advanced_search",
            ["search", "-q", "available:true"],
            self._results(dicts=3, strings=0),
        )
        assert result.exit_code == 0, result.output
        assert "Showing" not in result.output


class TestBatchDeleteReporting:
    """The bulk endpoint answers for the batch, so there is no per-sample count.

    ``batch_delete_samples`` sends every hash in one call and answers
    ``len(hashes)`` or ``None`` — never anything between — so the CLI
    reports that the batch was accepted and never claims a per-sample
    outcome the appliance did not confirm.

    ``None`` is "the appliance was never asked": a batch the endpoint
    cannot take, or a call that did not land, both already reported by
    the service. It used to be a ``0`` indistinguishable from a batch the
    appliance accepted nothing from, and this command printed "Removal
    failed for all N samples" over it — asserting a refusal nobody heard,
    for a request that was never made.
    """

    def _run(self, ctx_obj, monkeypatch, *, submitted: bool, requested: int):
        sent: list[list[str]] = []

        def batch_delete_samples(service, hashes):
            sent.append(list(hashes))
            return len(hashes) if submitted else None

        monkeypatch.setattr(A1000SampleService, "batch_delete_samples", batch_delete_samples)
        args = ["batch-delete"]
        for index in range(requested):
            args += ["-h", f"{index:064x}"]
        return invoke(a1000, args, ctx_obj, input="y\n"), sent

    def test_the_batch_goes_out_in_one_call_and_is_reported_as_accepted(self, ctx_obj, monkeypatch):
        result, sent = self._run(ctx_obj, monkeypatch, submitted=True, requested=2)
        assert sent == [[f"{index:064x}" for index in range(2)]]
        assert "✓ Removal accepted for 2 samples" in result.output
        assert "completed" not in result.output.lower(), "the appliance confirmed no such thing"

    def test_a_removal_that_was_never_submitted_says_so_and_fails(self, ctx_obj, monkeypatch):
        result, _ = self._run(ctx_obj, monkeypatch, submitted=False, requested=1)

        assert "✗ Removal was never submitted" in result.output
        assert "failed for all" not in result.output, "no refusal was heard from the appliance"
        assert "Confirm with" not in result.output

    def test_a_status_the_endpoint_does_not_use_is_named_and_not_counted(self, ctx_obj):
        """End to end over a real service: the silent branch, from the CLI down."""
        client = stub_client(ctx_obj)
        client.delete_samples.return_value = sdk_response(302)

        result = invoke(a1000, ["batch-delete", "-h", SHA256], ctx_obj, input="y\n")

        assert "302" in flat(result)
        assert "Removal was never submitted" in flat(result)
        assert "failed for all" not in flat(result)
        assert "Removal accepted" not in result.output


class TestCtrlCAbortsBatchDelete:
    """The removal wait caught KeyboardInterrupt and returned normally.

    ``batch_delete_samples`` then answered ``len(hashes)``, ``run_step``
    saw a truthy int, and Ctrl-C printed "Removal accepted" and exited 0
    — the one command where the analyst most wants out.
    """

    def test_an_interrupted_wait_neither_claims_success_nor_exits_zero(self, ctx_obj):
        client = MagicMock()
        client.delete_samples.return_value = stub_response({"id": "task-9"})
        client.check_sample_removal_status_v2.side_effect = KeyboardInterrupt
        ctx_obj.session.client = client

        result = invoke(a1000, ["batch-delete", "-h", SHA256], ctx_obj, input="y\n")

        assert "Removal accepted" not in result.output
        assert result.exit_code != 0


class TestAvailabilityProbeRunsWhenAServiceIsBuilt:
    """It used to be skipped by looking for "--help" in ``sys.argv``.

    That is not click's state: an embedded ``cli(...)`` call left argv
    pointing at the host program, and a real invocation whose option
    value happened to be the string "--help" skipped the probe. Neither
    test below touches a global; the args given to the runner are all
    that decide it.
    """

    def _args(self, tmp_path, *args):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("default:\n  a1000:\n    host: https://a1000.invalid\n")
        return ["--config", str(config_file), *args]

    def test_help_does_not_probe(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            APIAvailabilityChecker,
            "check_all",
            lambda self, force=False: pytest.fail("--help must not probe the API"),
        )

        result = CliRunner().invoke(cli, self._args(tmp_path, "a1000", "report", "--help"))

        assert result.exit_code == 0, result.output
        assert "Usage" in result.output

    def test_an_option_value_of_help_still_probes(self, tmp_path, monkeypatch):
        calls: list[bool] = []

        def check_all(
            self: APIAvailabilityChecker, force: bool = False
        ) -> dict[str, dict[str, str]]:
            calls.append(True)
            return {"a1000": {"status": "available", "message": "ok"}}

        monkeypatch.setattr(APIAvailabilityChecker, "check_all", check_all)
        monkeypatch.setattr(
            A1000SampleService, "advanced_search", lambda self, q, limit=100, page=1: []
        )

        CliRunner().invoke(cli, self._args(tmp_path, "a1000", "search", "-q", "--help"))

        assert calls == [True]


class TestRemoveTagsWithoutATagList:
    """``--tag`` is documented as optional, and the endpoint has no "all" mode.

    The SDK rejects a missing tag list before sending anything, so the
    command failed every time it was trusted with its own help text.
    """

    def _client(self, ctx_obj, tags) -> MagicMock:
        client = stub_client(ctx_obj)
        client.get_user_tags.return_value = stub_response(tags)
        client.delete_user_tags.return_value = stub_response({})
        return client

    def test_the_samples_own_tags_are_what_gets_removed(self, ctx_obj):
        client = self._client(ctx_obj, ["alpha", "beta"])

        # Answering the remove-everything prompt; see
        # TestRemovingEveryTagIsConfirmedFirst for the prompt itself.
        result = invoke(a1000, ["remove-tags", SHA256], ctx_obj, input="y\n")

        assert result.exit_code == 0, result.output
        client.delete_user_tags.assert_called_once_with(SHA256, ["alpha", "beta"])
        assert not ctx_obj.output.status.failed

    def test_a_sample_with_no_tags_is_not_a_failure(self, ctx_obj):
        client = self._client(ctx_obj, [])

        result = invoke(a1000, ["remove-tags", SHA256], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "No user tags to remove" in result.output
        client.delete_user_tags.assert_not_called()
        assert not ctx_obj.output.status.failed

    def test_named_tags_are_not_looked_up_first(self, ctx_obj):
        client = self._client(ctx_obj, ["alpha"])

        result = invoke(a1000, ["remove-tags", SHA256, "-t", "beta"], ctx_obj)

        assert result.exit_code == 0, result.output
        client.get_user_tags.assert_not_called()
        client.delete_user_tags.assert_called_once_with(SHA256, ["beta"])


class TestDeleteClassificationTargetsOneSystem:
    """``set-classification`` can write to TitaniumCloud, so deleting must reach it."""

    def _client(self, ctx_obj) -> MagicMock:
        client = stub_client(ctx_obj)
        client.delete_classification.return_value = stub_response({})
        return client

    def test_ticloud_classification_is_the_one_deleted(self, ctx_obj):
        client = self._client(ctx_obj)

        result = invoke(
            a1000, ["delete-classification", SHA256, "--system", "ticloud"], ctx_obj, input="y\n"
        )

        assert result.exit_code == 0, result.output
        client.delete_classification.assert_called_once_with(SHA256, system="ticloud")
        assert "ticloud" in result.output

    def test_the_default_is_still_local(self, ctx_obj):
        client = self._client(ctx_obj)

        result = invoke(a1000, ["delete-classification", SHA256], ctx_obj, input="y\n")

        assert result.exit_code == 0, result.output
        client.delete_classification.assert_called_once_with(SHA256, system="local")


class TestATypedNoIsAnsweredNotAborted:
    """The guarded sweep sends EOF, which aborts; a typed 'n' is the answer path.

    ``if not confirmed(...): return`` only runs when ``click.confirm`` hands
    back ``False`` -- an empty stdin raises ``Abort`` instead and never
    reaches it. These two commands had only the abort exercised.
    """

    def test_delete_classification_no_cancels_and_says_so(self, ctx_obj, monkeypatch):
        called: list[tuple] = []
        monkeypatch.setattr(
            A1000MetadataService,
            "delete_classification",
            lambda self, *args, **kwargs: called.append(args),
        )

        result = invoke(a1000, ["delete-classification", SHA256], ctx_obj, input="n\n")

        assert result.exit_code == 0, result.output
        assert "Cancelled" in flat(result)
        assert called == []

    def test_batch_delete_no_cancels_and_says_so(self, ctx_obj, monkeypatch):
        called: list[tuple] = []
        monkeypatch.setattr(
            A1000SampleService,
            "batch_delete_samples",
            lambda self, *args, **kwargs: called.append(args),
        )

        result = invoke(a1000, ["batch-delete", "-h", SHA256], ctx_obj, input="n\n")

        assert result.exit_code == 0, result.output
        assert "Cancelled" in flat(result)
        assert called == []


class TestSetClassificationEchoesTheThreatName:
    """A malicious verdict carries a threat name, and the success line names it."""

    def test_the_threat_name_is_reported_after_a_successful_set(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(
            A1000MetadataService,
            "set_classification",
            lambda self, *args, **kwargs: True,
        )

        result = invoke(a1000, ["set-classification", SHA256, "malicious", "-t", "Conti"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "Threat name: Conti" in flat(result)


class TestClaimsThatOutranTheEndpoint:
    """What the sample commands may say, given what A1000 actually answers."""

    def test_upload_wait_does_not_claim_this_submission_completed(
        self, ctx_obj, monkeypatch, tmp_path
    ):
        """The status endpoint answers about the hash, not about the upload.

        Re-uploading a file the appliance already holds makes the first
        poll answer "processed" out of the previous analysis, so claiming
        "Analysis completed" reported someone else's finished run as this
        submission's.
        """
        sample = tmp_path / "sample.bin"
        sample.write_bytes(b"x")
        monkeypatch.setattr(
            A1000SampleService, "upload_file", lambda self, path, comment=None: {"task_id": SHA256}
        )
        monkeypatch.setattr(
            A1000SampleService,
            "wait_for_analysis",
            lambda self, task_id, timeout=300: {"status": "processed"},
        )

        result = invoke(a1000, ["upload", str(sample), "--wait"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "Analysis completed" not in result.output
        assert "earlier analysis" in result.output

    def test_status_points_a_task_id_at_url_status(self, ctx_obj):
        """A task id posted to the hash-only endpoint just returns nothing."""
        # There is no wrapper left to reach the hash-only analysis-status
        # endpoint: the command reads the classification instead.
        assert not hasattr(A1000SampleService, "get_analysis_status")

        result = invoke(a1000, ["status", "3f1c9d20-0000-4000-8000-000000000000"], ctx_obj)

        # Rich wraps the hint, so the task id may land on the next line.
        assert "url-status" in result.output
        assert ctx_obj.output.status.failed

    def test_extracted_all_does_not_page_for_the_same_list(self, ctx_obj, monkeypatch):
        """The listing endpoint returns every file when asked for no page."""
        monkeypatch.setattr(
            A1000SampleService, "list_extracted_files", lambda self, h: [{"sha1": SHA1}]
        )
        # There is no aggregated wrapper left to call: paging for a list
        # one request already returns was the whole of what --all did.
        assert not hasattr(A1000SampleService, "list_extracted_files_aggregated")

        result = invoke(a1000, ["extracted", SHA256, "--all"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "no-op" in result.output


class TestConfigDumpDescribesWhatItDoes:
    """``config-dump`` prints local settings; its --help text has claimed twice not to."""

    def test_it_asks_the_appliance_nothing(self, ctx_obj, a1000_connections):
        result = invoke(a1000, ["config-dump"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert all(not client.method_calls for client in a1000_connections), (
            "config-dump called the appliance"
        )

    def test_its_help_does_not_offer_the_appliances_configuration(self, ctx_obj):
        result = invoke(a1000, ["config-dump", "--help"], ctx_obj)

        assert "Get A1000 configuration" not in result.output
        assert "local" in result.output.lower()


class TestUploadIsNotFailedByAnUnreadableBody:
    """A1000 answered 201 and a proxy truncated the body.

    "Expecting value: line 1 column 1" plus exit 1 had the analyst
    re-upload a sample already queued, or record it as never submitted.
    """

    def _appliance(self, ctx_obj, status: int) -> MagicMock:
        response = MagicMock(status_code=status)
        response.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
        client = MagicMock()
        client.submit_file_from_path.return_value = response
        ctx_obj.session.client = client
        return client

    def test_the_upload_is_reported_as_the_success_it_was(self, ctx_obj, tmp_path):
        sample = tmp_path / "sample.bin"
        sample.write_bytes(b"MZ")
        self._appliance(ctx_obj, 201)

        result = invoke(a1000, ["upload", str(sample)], ctx_obj)

        assert "Failed to upload file" not in flat(result)
        assert "File uploaded successfully" in flat(result)
        assert not ctx_obj.output.status.failed

    def test_wait_says_why_it_cannot_wait(self, ctx_obj, tmp_path):
        sample = tmp_path / "sample.bin"
        sample.write_bytes(b"MZ")
        client = self._appliance(ctx_obj, 201)

        result = invoke(a1000, ["upload", str(sample), "--wait"], ctx_obj)

        assert "Could not extract hash from upload result" in flat(result)
        client.file_analysis_status.assert_not_called()


class TestRemovingEveryTagIsConfirmedFirst:
    """``remove-tags <hash>`` with no ``--tag`` is the delete-everything mode.

    It wiped a week of manual triage tagging on a mistyped flag, silently
    and irreversibly, while ``delete-classification`` — which removes
    strictly less analyst-entered metadata — prompted.
    """

    def _client(self, ctx_obj, tags) -> MagicMock:
        client = stub_client(ctx_obj)
        client.get_user_tags.return_value = stub_response(tags)
        client.delete_user_tags.return_value = stub_response({})
        return client

    def test_declining_leaves_every_tag_alone(self, ctx_obj):
        client = self._client(ctx_obj, ["alpha", "beta", "gamma"])

        result = invoke(a1000, ["remove-tags", SHA256], ctx_obj, input="n\n")

        assert result.exit_code == 0, result.output
        client.delete_user_tags.assert_not_called()
        assert "Cancelled" in result.output

    def test_an_unanswered_prompt_is_not_taken_for_a_yes(self, ctx_obj):
        """Empty stdin used to print "Removed tags: a, b, c" and exit 0."""
        client = self._client(ctx_obj, ["alpha", "beta", "gamma"])

        invoke(a1000, ["remove-tags", SHA256], ctx_obj, input="")

        client.delete_user_tags.assert_not_called()

    def test_the_prompt_says_what_is_about_to_go(self, ctx_obj):
        client = self._client(ctx_obj, ["alpha", "beta", "gamma"])

        result = invoke(a1000, ["remove-tags", SHA256], ctx_obj, input="y\n")

        assert result.exit_code == 0, result.output
        assert "alpha" in flat(result) and "3" in flat(result)
        client.delete_user_tags.assert_called_once_with(SHA256, ["alpha", "beta", "gamma"])

    def test_naming_the_tags_is_still_unprompted(self, ctx_obj):
        """``-t`` already says what it removes, so it must not stop to ask."""
        client = self._client(ctx_obj, ["alpha"])

        result = invoke(a1000, ["remove-tags", SHA256, "-t", "beta"], ctx_obj, input="")

        assert result.exit_code == 0, result.output
        client.delete_user_tags.assert_called_once_with(SHA256, ["beta"])


class TestBothReportCommandsTreatADestinationAlike:
    """``report`` refused to write without one; ``dynamic-report`` invented one.

    So the same invocation shape either printed a size or dropped a file
    into whatever directory the analyst was standing in — over an
    annotated copy of the same name, announced as a success. Both now
    refuse without ``--output-file`` and, like ``download``, say so before
    replacing a file they were given.
    """

    def test_dynamic_report_writes_nothing_it_was_not_given_a_path_for(
        self, ctx_obj, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        annotated = tmp_path / f"{SHA1}_dynamic_report.pdf"
        annotated.write_bytes(b"an annotated copy")
        monkeypatch.setattr(
            A1000ReportService, "download_dynamic_report", lambda self, h, fmt="pdf": b"%PDF-fresh"
        )

        result = invoke(a1000, ["dynamic-report", SHA1], ctx_obj)

        assert result.exit_code == 0, result.output
        assert annotated.read_bytes() == b"an annotated copy", "it overwrote what it was not given"
        assert "--output-file" in flat(result)
        assert not ctx_obj.output.status.failed

    @pytest.mark.parametrize(
        "args, method, payload",
        [
            (["report", SHA256, "--output-file"], "get_report", {"sha256": SHA256}),
            (["dynamic-report", SHA1, "--output-file"], "download_dynamic_report", b"%PDF"),
        ],
        ids=["report", "dynamic-report"],
    )
    def test_replacing_a_named_file_is_announced_first(
        self, ctx_obj, monkeypatch, tmp_path, args, method, payload
    ):
        keep = tmp_path / "keep.out"
        keep.write_text("an earlier report")
        monkeypatch.setattr(A1000ReportService, method, lambda self, h, fmt="json": payload)

        result = invoke(a1000, [*args, str(keep)], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "already exists and will be replaced" in flat(result)


class TestAnEmptyNetworkAnswerIsAnAnswer:
    """These endpoints answer 200 with an empty report for a clean lookup.

    Three of the five called that a failure and exited 1, so an enrichment
    run filed every clean domain, URL and IP as a lookup error — while the
    ticloud twin of each said "No network intelligence for ..." and exited
    0 about the same answer.
    """

    @pytest.mark.parametrize(
        "command, method",
        [
            (["network-url-report", "example.com"], "get_network_url_report"),
            (["domain-report", "example.com"], "get_domain_report"),
            (["ip-report", "8.8.8.8"], "get_ip_report"),
        ],
        ids=["network-url-report", "domain-report", "ip-report"],
    )
    def test_an_empty_report_is_not_a_failed_lookup(self, ctx_obj, monkeypatch, command, method):
        monkeypatch.setattr(A1000NetworkService, method, lambda self, value: {})

        result = invoke(a1000, command, ctx_obj)

        assert result.exit_code == 0, result.output
        assert not ctx_obj.output.status.failed, "a clean lookup was reported as a failure"
        assert "Failed" not in flat(result)

    def test_domain_report_names_a_typo_instead_of_asking_the_appliance(self, ctx_obj):
        client = stub_client(ctx_obj)

        result = invoke(a1000, ["domain-report", "not a domain!!"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "not a domain!!" in flat(result)
        client.network_domain_report.assert_not_called()


class TestAnEmptyAnswerIsToldFromACallThatFailed:
    """An endpoint answering 200 with nothing has answered about the subject.

    ``get-classification`` and ``url-report`` read that as a failed lookup
    and exited 1, while ``get-tags`` and ``containers`` next to them warned
    and exited 0 about the same fact. Driven through the root group, since
    that is where the exit status a script reads is decided.
    """

    @pytest.mark.parametrize(
        "command, service_cls, method, notice",
        [
            (
                ["a1000", "get-classification", SHA256],
                A1000MetadataService,
                "get_classification",
                "No classification for this sample",
            ),
            (
                ["a1000", "url-report", "task-1"],
                A1000NetworkService,
                "get_url_report",
                "No URL report for this task yet",
            ),
        ],
        ids=["get-classification", "url-report"],
    )
    def test_an_empty_answer_warns_and_exits_zero(
        self, monkeypatch, command, service_cls, method, notice
    ):
        monkeypatch.setattr(service_cls, method, lambda self, value: {})

        result = CliRunner().invoke(cli, command)

        assert result.exit_code == 0, result.output
        assert notice in flat(result)
        assert "Failed" not in flat(result)

    def test_a_call_that_never_landed_still_exits_one(self, monkeypatch):
        monkeypatch.setattr(A1000MetadataService, "get_classification", lambda self, h: None)

        result = CliRunner().invoke(cli, ["a1000", "get-classification", SHA256])

        assert result.exit_code == 1, result.output
        assert "Failed to get classification" in flat(result)

    def test_url_status_keeps_a_stateless_answer_a_failure(self, monkeypatch):
        """A poll needs a state to act on, so nothing back is not an answer."""
        monkeypatch.setattr(A1000NetworkService, "url_status", lambda self, task_id: {})

        result = CliRunner().invoke(cli, ["a1000", "url-status", "task-1"])

        assert result.exit_code == 1, result.output
        assert "Failed to get URL status" in flat(result)


class TestAPartialAnswerSaysSoInOneWording:
    """ "There is more than this" was three sentences in three commands.

    ``list``/``search`` named ``--limit``, the IP lookups named ``--all``,
    and each wrote the rest of the sentence its own way. One wording now,
    with the option that fetches the rest as its only variable.
    """

    def _left_a_page_behind(self, entries):
        def fetch(service, *_args, **_kwargs):
            service.pages_left_unfetched = True
            return entries

        return fetch

    @pytest.mark.parametrize(
        "command, service_cls, method, option",
        [
            (["a1000", "list"], A1000SampleService, "list_samples", "raise --limit above 100"),
            (
                ["a1000", "ip-urls", "8.8.8.8"],
                A1000NetworkService,
                "get_urls_from_ip",
                "re-run with --all",
            ),
        ],
        ids=["list", "ip-urls"],
    )
    def test_the_notice_names_the_option_that_fetches_the_rest(
        self, monkeypatch, command, service_cls, method, option
    ):
        monkeypatch.setattr(service_cls, method, self._left_a_page_behind([{"sha256": SHA256}]))

        result = CliRunner().invoke(cli, ["-o", "json", *command])

        assert result.exit_code == 0, result.output
        assert f"More results may be waiting; {option} to fetch them." in flat(result)

    @pytest.mark.parametrize(
        "args,remedy",
        [
            (["--limit", "150"], "raise --limit above 150"),
            (["--limit", "150", "--page", "3"], "pass --page 4"),
        ],
        ids=["aggregated-walk", "explicit-page"],
    )
    def test_the_remedy_fits_the_fetch_that_was_cut_short(self, monkeypatch, args, remedy):
        """ "Raise --limit past one page" is no advice to a walk that spanned pages.

        Nor is a bigger ``--limit`` to a request that named ``--page``,
        where the two cannot be combined at all.
        """
        monkeypatch.setattr(
            A1000SampleService, "advanced_search", self._left_a_page_behind([{"sha256": SHA256}])
        )

        result = CliRunner().invoke(
            cli, ["-o", "json", "a1000", "search", "-q", "available:true", *args]
        )

        assert result.exit_code == 0, result.output
        assert f"More results may be waiting; {remedy} to fetch them." in flat(result)

    def test_at_the_ceiling_the_remedy_is_one_the_cli_would_accept(self, monkeypatch):
        """ "Raise --limit above 100000" is a command line this CLI refuses.

        ``--limit`` is bounded at ``MAX_LIMIT`` so a mistyped digit cannot
        spend thousands of metered requests; at that bound there is no
        bigger one to raise it to, and advising it sends the analyst to
        find out by typing it and getting exit 2. ``ticloud search``
        answers this the same way.
        """
        monkeypatch.setattr(
            A1000SampleService, "advanced_search", self._left_a_page_behind([{"sha256": SHA256}])
        )

        result = CliRunner().invoke(
            cli,
            ["-o", "json", "a1000", "search", "-q", "available:true", "--limit", str(MAX_LIMIT)],
        )

        assert result.exit_code == 0, result.output
        assert "More results may be waiting" in flat(result)
        assert f"raise --limit above {MAX_LIMIT}" not in flat(result)

    def test_a_complete_answer_says_nothing(self, monkeypatch):
        monkeypatch.setattr(
            A1000SampleService, "list_samples", lambda self, limit: [{"sha256": SHA256}]
        )

        result = CliRunner().invoke(cli, ["-o", "json", "a1000", "list"])

        assert result.exit_code == 0, result.output
        assert "More results" not in flat(result)


class TestTheNoticeOwnsTheCapComparison:
    """ "Did the walk fill its cap" is one reading, made in one place.

    Two commands each computed ``max_results is not None and len(x) >=
    max_results`` and passed the answer in, under the same paragraph
    explaining why it is ``>=`` and not ``==``. A third caller writing
    ``==`` would go on printing nothing for the walk an SDK release
    overshot, and no test of either command would notice.
    """

    def _said(self, capsys, **kwargs) -> str:
        partial_answer_notice(RichOutput(), "raise --max-results", **kwargs)
        return " ".join(capsys.readouterr().err.split())

    @pytest.mark.parametrize(
        "collected,cap,expected",
        [
            ([1, 2], 2, True),
            ([1, 2, 3], 2, True),
            ([1], 2, False),
            ([1, 2], None, False),
            (None, 2, False),
        ],
        ids=["at-the-cap", "overshot", "under-the-cap", "uncapped", "failed-fetch"],
    )
    def test_the_notice_decides_from_the_cap_and_the_answer(self, capsys, collected, cap, expected):
        said = self._said(capsys, collected=collected, max_results=cap)

        assert ("More results may be waiting" in said) is expected, said

    def test_a_page_left_behind_is_still_the_other_way_of_being_partial(self, capsys):
        """The paging services report the fact; only the cap is computed here."""
        said = self._said(capsys, pages_left=True)

        assert "More results may be waiting; raise --max-results to fetch them." in said


class TestUploadWaitKeepsItsPromiseOrFails:
    """``upload --wait --timeout 60 && report -o json > report.json``.

    The timeout warned and exited 0, so the ``&&`` fired and the report
    was written out of an analysis that had never finished.
    """

    def test_a_wait_that_runs_out_of_time_exits_non_zero(self, ctx_obj, monkeypatch, tmp_path):
        sample = tmp_path / "sample.bin"
        sample.write_bytes(b"MZ")
        monkeypatch.setattr(
            A1000SampleService, "upload_file", lambda self, path, comment=None: {"task_id": SHA256}
        )
        client = stub_client(ctx_obj)
        client.file_analysis_status.return_value = stub_response(
            {"results": [{"status": "not_found"}]}
        )

        invoke(a1000, ["upload", str(sample), "--wait", "--timeout", "0"], ctx_obj)

        assert ctx_obj.output.status.failed, "a --wait that never landed exited 0"


class TestAWrongCommandLineIsAUsageError:
    """One class of mistake, one answer: exit 2 with the usage.

    ``yara-repo-list`` raised ``UsageError``; the others printed an error
    and exited 1, so ``rl-cli ... || retry`` saw the same operator
    mistake as a transient appliance failure in four commands and as a
    bad command line in a fifth.
    """

    @pytest.mark.parametrize(
        "args",
        [
            ["yara-publish", "ruleset", "--all"],
            ["yara-publish"],
            ["yara-update-interval", "3600", "--reset"],
            ["yara-update-interval"],
            ["batch-delete"],
            ["batch-reanalyze"],
            ["batch-reanalyze", "-h", SHA256, "--no-titanium-core", "--no-titanium-cloud"],
        ],
        ids=[
            "publish-both",
            "publish-neither",
            "interval-both",
            "interval-neither",
            "batch-delete-no-hashes",
            "batch-reanalyze-no-hashes",
            "batch-reanalyze-no-engines",
        ],
    )
    def test_the_combination_is_refused_with_the_usage(self, ctx_obj, monkeypatch, args):
        for service_cls in A1000_SERVICES:
            stub_service(monkeypatch, service_cls, {"stub": "payload"})

        result = invoke(a1000, args, ctx_obj)

        assert result.exit_code == 2, result.output
        assert "Usage:" in result.output

    def test_a_refused_combination_reaches_no_appliance(self, ctx_obj):
        """Exit 2 has to mean nothing was attempted, not something half-was."""
        client = stub_client(ctx_obj)

        result = invoke(a1000, ["yara-publish", "ruleset", "--all"], ctx_obj)

        assert result.exit_code == 2
        client.publish_yara_ruleset.assert_not_called()
        client.publish_all_yara_rulesets.assert_not_called()


def _appliance(monkeypatch, **responses) -> MagicMock:
    """An A1000 whose named SDK calls answer 200 with the given bodies.

    Wired at ``A1000Session._open`` so the command can be driven through
    the root group, which is where the exit status is decided: the
    ``a1000`` group on its own runs no ``result_callback``, so a test
    invoking it can pin the message but not the status a script reads.
    """
    client = MagicMock()
    for name, payload in responses.items():
        getattr(client, name).return_value = stub_response(payload)
    monkeypatch.setattr(A1000Session, "_open", lambda session: setattr(session, "client", client))
    return client


class TestBatchReanalyzeCountsWhatTheApplianceTook:
    """The headline count named the entries in the answer, not the samples taken.

    ``print_reanalyze_results_table`` already reads a per-engine status of
    400 or more as a refusal — a batch of 50 unknown hashes came back as
    50 successes — but the success line counted the response array and the
    run exited 0, so ``batch-reanalyze -f hashes.txt && wait-and-report``
    fired over a batch the appliance had rejected outright. Under ``-o
    json`` no table is drawn at all, which leaves that line as the whole
    report.
    """

    def _entry(self, index: int, *codes: int) -> dict:
        return {
            "detail": {"sha256": f"{index:064x}"},
            "analysis": [
                {"name": "titanium_core", "code": code, "message": "Sample not found"}
                for code in codes
            ],
        }

    def _run(self, monkeypatch, entries, *, submitted=None, output_format=None):
        # Over the appliance rather than over the service method: what the
        # sentence counts is the answer as the service grades it, and a
        # stubbed service would be a grading of the test's own.
        _appliance(monkeypatch, reanalyze_samples_v2=entries)
        args = ["a1000", "batch-reanalyze"]
        for index in range(submitted if submitted is not None else len(entries)):
            args += ["-h", f"{index:064x}"]
        if output_format:
            args = ["-o", output_format, *args]
        return CliRunner().invoke(cli, args)

    def test_a_batch_the_appliance_refused_entirely_is_a_failure(self, monkeypatch):
        result = self._run(monkeypatch, [self._entry(index, 404) for index in range(3)])

        assert "Reanalysis refused for all 3 samples" in flat(result)
        assert "Reanalysis started" not in flat(result)
        assert result.exit_code == 1, result.output

    def test_the_count_names_the_samples_that_were_queued(self, monkeypatch):
        entries = [self._entry(0, 201), self._entry(1, 404), self._entry(2, 404)]

        result = self._run(monkeypatch, entries)

        assert "Reanalysis started for 1 of 3 samples" in flat(result)
        assert "2 samples were refused by the appliance" in flat(result)
        assert result.exit_code == 0, result.output

    def test_a_batch_the_appliance_took_reports_the_whole_batch(self, monkeypatch):
        result = self._run(monkeypatch, [self._entry(index, 201) for index in range(3)])

        assert "Reanalysis started for 3 samples" in flat(result)
        assert "refused" not in flat(result)
        assert result.exit_code == 0, result.output

    def test_an_engine_that_took_the_sample_carries_the_entry(self, monkeypatch):
        """One refusing engine is not the appliance refusing the sample."""
        result = self._run(monkeypatch, [self._entry(0, 201, 404)])

        assert "Reanalysis started for 1 samples" in flat(result)
        assert result.exit_code == 0, result.output

    def test_an_entry_with_no_per_engine_verdict_is_the_submitted_the_table_draws(
        self, monkeypatch
    ):
        result = self._run(monkeypatch, [{"detail": {"sha256": SHA256}}])

        assert "Reanalysis started for 1 samples" in flat(result)
        assert result.exit_code == 0, result.output

    def test_the_refusal_survives_output_json_where_no_table_is_drawn(self, monkeypatch):
        entries = [self._entry(index, 404) for index in range(2)]

        result = self._run(monkeypatch, entries, output_format="json")

        assert json.loads(result.stdout) == entries, "the appliance's answer was not rendered"
        assert "Reanalysis refused for all 2 samples" in flat(result)
        assert result.exit_code == 1, result.output

    def test_an_answer_covering_fewer_samples_than_were_sent_says_so(self, monkeypatch):
        result = self._run(monkeypatch, [self._entry(0, 201)], submitted=3)

        assert "answered for 1 of the 3 samples submitted" in flat(result)
        assert result.exit_code == 0, result.output

    def test_an_answer_about_nothing_is_not_a_reanalysis(self, monkeypatch):
        result = self._run(monkeypatch, [], submitted=2)

        assert "Failed to start reanalysis" in flat(result)
        assert "Reanalysis started" not in flat(result)
        assert result.exit_code == 1, result.output


class TestBatchReanalyzeReadsTheAnswerTheWayTheTableDoes:
    """The count read per-engine codes only; the table read three other things.

    Every shape here was reproduced through the root ``cli`` group against
    the first fix: an entry refused before any engine saw it, a refusal
    stated in words, a scalar where the engine list belongs, and a body
    carrying no per-sample answers at all. Each was announced as
    "Reanalysis started", exit 0 — or, for the scalar, as an unhandled
    ``TypeError`` — while the table underneath drew the refusal.
    """

    def _run(self, monkeypatch, entries, *, submitted=None):
        _appliance(monkeypatch, reanalyze_samples_v2=entries)
        args = ["a1000", "batch-reanalyze"]
        for index in range(submitted if submitted is not None else len(entries)):
            args += ["-h", f"{index:064x}"]
        return CliRunner().invoke(cli, args)

    def test_a_refusal_stated_on_the_entry_itself_is_not_an_acceptance(self, monkeypatch):
        """``{"code": 404, ...}`` — the shape a submission is answered with."""
        entries = [
            {"code": 404, "message": "Sample not found", "detail": {"sha256": f"{index:064x}"}}
            for index in range(3)
        ]

        result = self._run(monkeypatch, entries)

        assert "Reanalysis refused for all 3 samples" in flat(result)
        assert "Reanalysis started" not in flat(result)
        assert "Submitted" not in flat(result), "the table read it as taken"
        assert result.exit_code == 1, result.output

    def test_a_refusal_stated_in_words_is_not_an_acceptance(self, monkeypatch):
        """The table drew "Sample not found" while the headline said started."""
        entries = [{"status": "Sample not found", "analysis": []} for _ in range(3)]

        result = self._run(monkeypatch, entries)

        assert "Reanalysis refused for all 3 samples" in flat(result)
        assert "Reanalysis started" not in flat(result)
        assert result.exit_code == 1, result.output

    def test_a_queued_spelling_of_status_is_still_an_acceptance(self, monkeypatch):
        result = self._run(monkeypatch, [{"status": "Submitted", "analysis": []}])

        assert "Reanalysis started for 1 samples" in flat(result)
        assert result.exit_code == 0, result.output

    def test_a_scalar_where_the_engine_list_belongs_does_not_crash_the_command(self, monkeypatch):
        """``{"analysis": 7}`` raised TypeError out of the count; the table survived it."""
        result = self._run(monkeypatch, [{"detail": {"sha256": SHA256}, "analysis": 7}])

        assert result.exception is None, result.exception
        assert "Reanalysis started for 1 samples" in flat(result)
        assert result.exit_code == 0, result.output

    def test_a_body_carrying_no_results_is_not_one_accepted_sample(self, monkeypatch):
        """The service wrapped an unrecognised 200 as a single fabricated entry."""
        _appliance(monkeypatch, reanalyze_samples_v2={"code": 403, "message": "no rights"})
        args = ["a1000", "batch-reanalyze"]
        for index in range(5):
            args += ["-h", f"{index:064x}"]

        result = CliRunner().invoke(cli, args)

        assert "Reanalysis started" not in flat(result)
        assert "Failed to start reanalysis" in flat(result)
        assert result.exit_code == 1, result.output


class TestTheHeadlineCountAndTheStatusColumnAreOneGrading:
    """ "Reanalysis started for 3 samples", in green, over three refused rows.

    The count above the table was graded in the command body and the
    Status column was graded again in ``print_reanalyze_results_table``,
    with a comment between them saying the two must agree. Nothing
    enforced it: a change to how a renderer reads a row moved the table and
    left the headline — and the exit status — where it was.

    So the pairing is asserted the way it can break: the grading rule is
    changed, and both the number in the sentence and the rows underneath it
    have to move together.
    """

    QUEUED: ClassVar[dict[str, Any]] = {"analysis": [{"name": "core", "code": 201}]}
    REFUSED: ClassVar[dict[str, Any]] = {"code": 404, "message": "gone"}

    def _entries(self, *shapes: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {**shape, "detail": {"sha256": f"{index:064x}"}} for index, shape in enumerate(shapes)
        ]

    def _run(self, monkeypatch, entries: list[dict[str, Any]]):
        """Drive the whole command over an appliance answering ``entries``.

        Through the root group and the real service, because the grading
        under test is the one the command and the table share: a stubbed
        service would be a third reading of the answer.
        """
        _appliance(monkeypatch, reanalyze_samples_v2=entries)
        args = ["a1000", "batch-reanalyze"]
        for index in range(len(entries)):
            args += ["-h", f"{index:064x}"]
        return CliRunner().invoke(cli, args)

    @staticmethod
    def _headline_count(rendered: str) -> int:
        """How many samples the headline claims a reanalysis started for."""
        if "Reanalysis refused for all" in rendered:
            return 0
        started = re.search(r"Reanalysis started for (\d+)", rendered)
        assert started is not None, rendered
        return int(started.group(1))

    def _assert_the_count_is_the_rows(self, result, *, answered: int) -> None:
        """The headline number is the number of rows the table drew as taken.

        "Refused" is the word the Status cell reads a turned-down sample by,
        and the only capitalised one in the output: the sentences say
        "refused".
        """
        rendered = flat(result)
        drawn_as_accepted = answered - rendered.count("Refused")
        assert self._headline_count(rendered) == drawn_as_accepted, rendered

    def test_the_headline_counts_the_rows_the_table_drew_as_accepted(self, monkeypatch):
        entries = self._entries(self.QUEUED, self.REFUSED, self.REFUSED)

        self._assert_the_count_is_the_rows(self._run(monkeypatch, entries), answered=3)

    @pytest.mark.parametrize("shape", ["QUEUED", "REFUSED"], ids=["taken", "turned-down"])
    def test_the_pairing_survives_a_change_to_the_grading_rule(self, monkeypatch, shape):
        """One grading, so inverting it moves the sentence and the rows alike."""
        graded = ReanalysisOutcome.of

        def inverted(entry) -> ReanalysisOutcome:
            outcome = graded(entry)
            return replace(outcome, accepted=not outcome.accepted)

        monkeypatch.setattr(ReanalysisOutcome, "of", staticmethod(inverted))
        entries = self._entries(*[getattr(self, shape)] * 3)

        self._assert_the_count_is_the_rows(self._run(monkeypatch, entries), answered=3)


class TestTheReanalysisVerdictIsAModelAndNotARendering:
    """``batch-reanalyze``'s exit status was decided inside ``render/formatters``.

    The command deep-imported the one non-``print_*`` name in the
    presentation package to compute it, and the answer arrived welded to
    ``[red]…[/red]`` — markup in the value the exit status is read from. The
    decision is a payload reading, so it lives with the other payload
    readings; the service grades the answer with it, and the command and
    the table read that one grading rather than each repeating it.
    """

    def test_the_command_grades_nothing_and_the_service_grades_once(self):
        # Read out of the module namespaces: the claim is which name each
        # module binds, and an import is not a re-export.
        assert "ReanalysisOutcome" not in vars(samples_module), "the command graded the answer"
        assert vars(sample_service_module)["ReanalysisOutcome"] is ReanalysisOutcome

    def test_no_command_module_reaches_into_the_formatters_package(self):
        """``print_*`` off ``render.formatters`` only: a renderer decides nothing."""
        commands = Path(rl_cli.cli.commands.__file__).parent
        for source in sorted(commands.rglob("*.py")):
            for line in source.read_text(encoding="utf-8").splitlines():
                assert "from rl_cli.render.formatters." not in line, f"{source.name}: {line}"

    def test_the_reasons_it_states_carry_no_markup(self):
        """Escaping is the renderer's: a value with ``[red]`` in it cannot be counted."""
        outcome = ReanalysisOutcome.of(
            {"code": 404, "message": "Sample not found", "analysis": [{"name": "tc", "code": 404}]}
        )

        assert outcome.accepted is False
        assert outcome.refusal == "404: Sample not found"
        assert outcome.failed == ("tc (404): rejected",)
        assert "[red]" not in str(outcome)

    def test_an_entry_that_is_not_a_record_is_not_an_acceptance(self):
        """The count walks the answer as it came; only the table filters it."""
        assert ReanalysisOutcome.of("nonsense").accepted is False
        assert ReanalysisOutcome.of(None).accepted is False


class TestBatchDeleteCountsTheAnswerToo:
    """Its message was built from ``len(hash_list)`` — the request, not the answer."""

    def test_the_count_comes_from_what_the_service_reported(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(A1000SampleService, "batch_delete_samples", lambda self, hashes: 1)

        result = invoke(a1000, ["batch-delete", "-h", SHA256, "-h", SHA1], ctx_obj, input="y\n")

        assert "Removal accepted for 1 samples" in flat(result)
        assert not ctx_obj.output.status.failed


class TestAnAcceptedRemovalIsNotReportedAsATotalFailure:
    """A 202 the analyst is told failed is the dangerous direction.

    ``response.json()`` was called unguarded on the bulk-removal answer,
    so an empty or proxy-truncated body raised, the service answered its
    ``default=0``, and this command printed "Removal failed for all 1
    samples" and exited 1 — while the appliance was removing the samples.
    """

    def test_a_202_whose_body_cannot_be_read_still_reports_acceptance(self, monkeypatch):
        client = _appliance(monkeypatch)
        accepted = MagicMock()
        accepted.status_code = 202
        accepted.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
        client.delete_samples.return_value = accepted

        result = CliRunner().invoke(cli, ["a1000", "batch-delete", "-h", SHA256], input="y\n")

        assert "Removal failed" not in flat(result)
        assert "Removal accepted for 1 samples" in flat(result)
        assert result.exit_code == 0, result.output


class TestAnUnreadableAnswerIsNotAnEmptyOne:
    """``data.get(<key>, [])`` graded a 200 we could not parse as "nothing found".

    For an analyst that is the dangerous direction: "No extracted files
    for this sample", exit 0, over an answer this reader could not parse
    reads as a clean result rather than as a failed read.
    """

    def test_extracted_files_stated_as_something_else_are_not_no_files(self, monkeypatch):
        _appliance(monkeypatch, list_extracted_files_v2={"results": {"0": {"sha1": SHA1}}})

        result = CliRunner().invoke(cli, ["a1000", "extracted", SHA256])

        assert "No extracted files" not in flat(result)
        assert "Failed to list extracted files" in flat(result)
        assert result.exit_code == 1, result.output

    def test_a_sample_with_no_extracted_files_still_says_so(self, monkeypatch):
        _appliance(monkeypatch, list_extracted_files_v2={"count": 0, "results": []})

        result = CliRunner().invoke(cli, ["a1000", "extracted", SHA256])

        assert "No extracted files for this sample" in flat(result)
        assert result.exit_code == 0, result.output

    def test_tags_stated_under_another_key_are_not_no_tags(self, monkeypatch):
        _appliance(monkeypatch, get_user_tags={"user_tags": ["apt"]})

        result = CliRunner().invoke(cli, ["a1000", "get-tags", SHA256])

        assert "No user tags" not in flat(result)
        assert "Failed to get user tags" in flat(result)
        assert result.exit_code == 1, result.output

    def test_a_sample_with_no_tags_still_says_so(self, monkeypatch):
        _appliance(monkeypatch, get_user_tags=[])

        result = CliRunner().invoke(cli, ["a1000", "get-tags", SHA256])

        assert "No user tags for this sample" in flat(result)
        assert result.exit_code == 0, result.output

    def test_the_wrapped_tag_list_is_still_read(self, monkeypatch):
        _appliance(monkeypatch, get_user_tags={"tags": ["apt"]})

        result = CliRunner().invoke(cli, ["-o", "json", "a1000", "get-tags", SHA256])

        assert json.loads(result.stdout) == ["apt"]
        assert result.exit_code == 0, result.output

    def test_containers_stated_under_another_key_are_not_no_containers(self, monkeypatch):
        _appliance(monkeypatch, list_containers_for_hashes={"containers": [{"sha1": SHA1}]})

        result = CliRunner().invoke(cli, ["a1000", "containers", SHA256])

        assert "No containers found" not in flat(result)
        assert "Failed to list containers" in flat(result)
        assert result.exit_code == 1, result.output

    def test_an_empty_container_answer_is_not_a_failed_lookup(self, monkeypatch):
        """The bulk endpoint leaves out a hash that has no container at all."""
        _appliance(monkeypatch, list_containers_for_hashes={})

        result = CliRunner().invoke(cli, ["a1000", "containers", SHA256])

        assert "No containers found for this sample" in flat(result)
        assert "Failed" not in flat(result)
        assert result.exit_code == 0, result.output

    def test_the_containers_it_did_state_are_counted(self, monkeypatch):
        _appliance(monkeypatch, list_containers_for_hashes={"results": [{"sha1": SHA1}]})

        result = CliRunner().invoke(cli, ["a1000", "containers", SHA256])

        assert "Found 1 containers" in flat(result)
        assert result.exit_code == 0, result.output

    def test_removing_every_tag_does_not_report_a_wipe_it_could_not_read(self, monkeypatch):
        """The remove-everything mode resolves "all" against the sample's tags."""
        client = _appliance(monkeypatch, get_user_tags={"user_tags": ["apt"]})

        result = CliRunner().invoke(cli, ["a1000", "remove-tags", SHA256])

        assert "No user tags to remove" not in flat(result)
        assert "Failed to remove tags" in flat(result)
        client.delete_user_tags.assert_not_called()
        assert result.exit_code == 1, result.output


class TestASha512IsRefusedByTheCommandThatCannotUseIt:
    """The tag and container endpoints take no SHA512.

    It used to reach the SDK, whose refusal — "Only hash strings of the
    following types are allowed as input values" with all 128 characters
    echoed back — named neither the command nor what it should have been
    given.
    """

    SHA512 = "c" * 128

    @pytest.mark.parametrize(
        "args, sdk_method",
        [
            (["get-tags"], "get_user_tags"),
            (["add-tags", "-t", "apt"], "post_user_tags"),
            (["containers"], "list_containers_for_hashes"),
        ],
        ids=["get-tags", "add-tags", "containers"],
    )
    def test_the_command_says_what_the_endpoint_takes(self, monkeypatch, args, sdk_method):
        command, *options = args
        client = _appliance(monkeypatch)

        result = CliRunner().invoke(cli, ["a1000", command, self.SHA512, *options])

        assert "does not accept SHA512" in flat(result)
        assert "MD5, SHA1, SHA256" in flat(result)
        assert result.exit_code == 1, result.output
        getattr(client, sdk_method).assert_not_called()

    def test_a_command_whose_endpoint_takes_one_still_accepts_it(self, monkeypatch):
        client = _appliance(monkeypatch, get_classification_v3={"classification": "malicious"})

        result = CliRunner().invoke(cli, ["-o", "json", "a1000", "get-classification", self.SHA512])

        assert result.exit_code == 0, result.output
        client.get_classification_v3.assert_called_once_with(self.SHA512)


class TestEveryCommandThatPromptsCanStillBeScripted:
    """A command that asks before it acts takes ``--yes``, and every one of them
    takes the same one.

    ``yara-repo-delete`` was given its prompt with no way past it, and a
    working ``rl-cli a1000 yara-repo-delete 7`` in a cron job started
    exiting 1 having deleted nothing: ``click.confirm`` reads a caller with
    nothing on stdin as an abort. So the escape hatch is swept for rather
    than remembered — an eighth destructive command cannot ship without
    one.

    Read from the command bodies, like the ``capped_listing`` sweep in
    tests/test_command_matrix.py, because a test that only invoked the
    seven would pass the day the eighth is written.

    Which commands ask is resolved once, in ``tests/cli_support.py``, and
    read here and by tests/test_readme_conformance.py. Two of the three
    used to spell it as the same substring search, which is one derivation
    wearing two hats: it saw a question put through a helper as no
    question at all, and a comment mentioning ``confirmed()`` as one. What
    is independent here is the *claim* — that the set which asks is the
    set carrying ``--yes``, taken off click's parsed parameters, and that
    the set reaching an irreversible call is a subset of both, which
    tests/test_command_matrix.py makes.
    """

    # The seven that ask today, plus ``create-profile``. Named so that a
    # sweep which has stopped seeing them fails rather than passing empty.
    ASKING: ClassVar = {
        "a1000 delete",
        "a1000 batch-delete",
        "a1000 delete-classification",
        "a1000 remove-tags",
        "a1000 yara-delete",
        "a1000 yara-cloud-retro",
        "a1000 yara-repo-delete",
        "config create-profile",
    }

    # One row per command whose prompt guards a single service call, and the
    # call it guards. ``remove-tags`` is not here: its prompt is answered
    # inside the service, against the tags "all" resolved to, so it is
    # exercised below through the appliance instead.
    GUARDED: ClassVar = [
        (["delete", SHA256], A1000SampleService, "delete_sample"),
        (["batch-delete", "-h", SHA256], A1000SampleService, "batch_delete_samples"),
        (["delete-classification", SHA256], A1000MetadataService, "delete_classification"),
    ]
    GUARDED_IDS: ClassVar = ["delete", "batch-delete", "delete-classification"]

    # ``config init`` prompts and takes no ``--yes``, on purpose: it is the
    # wizard, and every value it writes comes from a prompt, so a run that
    # cannot answer has nothing to save. Named here rather than by excluding
    # the whole ``config`` group, which is how ``create-profile`` -- a
    # command that takes its name from argv and overwrites a stored
    # profile's credentials -- shipped prompting with no way past.
    NOT_SCRIPTABLE: ClassVar = {"config init"}

    def _prompting(self) -> dict[str, click.Command]:
        return {
            where: command
            for where, command in commands_that_ask().items()
            if where not in self.NOT_SCRIPTABLE
        }

    def _offering_yes(self) -> set[str]:
        return commands_offering_yes()

    def test_every_exempt_command_still_prompts(self):
        """An exemption for a command that stopped prompting is a stale waiver."""
        asking = set(commands_that_ask())

        assert asking >= self.NOT_SCRIPTABLE, (
            f"exempted but no longer prompting: {sorted(self.NOT_SCRIPTABLE - asking)}"
        )

    def test_every_command_that_prompts_declares_yes(self):
        without = sorted(
            where
            for where, command in self._prompting().items()
            if not any("--yes" in param.opts for param in command.params)
        )

        assert not without, f"a command prompts with no --yes to get past it: {without}"

    def test_every_command_declaring_yes_asks_before_it_acts(self):
        """The other direction, which is the one a ninth command breaks.

        ``--yes`` on a command that never puts the question is a flag
        promising an escape from a prompt that is not there — and, far
        worse, a destructive call with nothing in front of it. Counted
        against the commands that declare the flag rather than against a
        number written down here, so the eventual ninth is swept too.
        """
        silent = sorted(self._offering_yes() - set(self._prompting()))

        assert not silent, f"a command declares --yes and asks nothing: {silent}"

    def test_the_sweep_still_finds_the_commands_that_prompt(self):
        """The sentinel: the markers above have to mark something.

        Equality, not a floor: every destructive command in this CLI is
        one that declares ``--yes``, so the set the source sweep finds is
        the set the parser reports, and a command that drifts out of
        either is named.
        """
        found = set(self._prompting())

        assert found == self.ASKING, f"the sweep sees {sorted(found)}, not {sorted(self.ASKING)}"
        assert found == self._offering_yes(), (
            f"prompting and --yes disagree: {sorted(found ^ self._offering_yes())}"
        )

    def test_the_flag_is_one_flag_with_one_help_line(self):
        published = {
            (tuple(flag.opts), flag.default, getattr(flag, "help", None))
            for command in self._prompting().values()
            for flag in command.params
            if "--yes" in flag.opts
        }

        assert len(published) == 1, f"--yes is offered {len(published)} ways: {published}"
        opts, default, help_text = published.pop()
        assert opts == ("--yes",)
        assert default is False, "--yes is on unless the caller gives it"
        assert help_text, "--yes publishes no help line, so --help does not mention it"

    def _record(self, monkeypatch, service_cls, method) -> list[tuple]:
        made: list[tuple] = []

        def answer(self, *arguments, **_keywords):
            made.append(arguments)
            return 1

        monkeypatch.setattr(service_cls, method, answer)
        return made

    @pytest.mark.parametrize("args, service_cls, method", GUARDED, ids=GUARDED_IDS)
    def test_the_flag_carries_the_call_through_with_nothing_on_stdin(
        self, ctx_obj, monkeypatch, args, service_cls, method
    ):
        made = self._record(monkeypatch, service_cls, method)

        result = invoke(a1000, [*args, "--yes"], ctx_obj, input="")

        assert result.exit_code == 0, result.output
        assert made, "the call the prompt guards was never made"
        assert "?" not in flat(result), "the prompt was drawn anyway"

    @pytest.mark.parametrize("args, service_cls, method", GUARDED, ids=GUARDED_IDS)
    def test_the_same_command_without_the_flag_still_asks(
        self, ctx_obj, monkeypatch, args, service_cls, method
    ):
        made = self._record(monkeypatch, service_cls, method)

        result = invoke(a1000, args, ctx_obj, input="")

        assert result.exit_code == 1
        assert not made, "an unanswered prompt was taken for a yes"

    def test_removing_every_tag_is_scripted_by_the_same_flag(self, ctx_obj):
        client = stub_client(ctx_obj)
        client.get_user_tags.return_value = stub_response(["alpha", "beta"])
        client.delete_user_tags.return_value = stub_response({})

        result = invoke(a1000, ["remove-tags", SHA256, "--yes"], ctx_obj, input="")

        assert result.exit_code == 0, result.output
        client.delete_user_tags.assert_called_once_with(SHA256, ["alpha", "beta"])
