"""Tests for the A1000 service wrappers and module-level helpers."""

from __future__ import annotations

import errno
import io
import itertools
import os
import stat
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from ReversingLabs.SDK.a1000 import A1000

from rl_cli.cli.commands._shared_inputs import MAX_LIMIT
from rl_cli.cli.main import cli
from rl_cli.config import Settings
from rl_cli.render.output import RichOutput
from rl_cli.services import http as http_module
from rl_cli.services.a1000 import (
    A1000MetadataService,
    A1000NetworkService,
    A1000ReportService,
    A1000SampleService,
    A1000Service,
    A1000Session,
    A1000YaraService,
    upload_and_get_report,
)
from rl_cli.services.a1000 import reports as reports_module
from rl_cli.services.a1000 import samples as samples_module
from rl_cli.services.a1000.network import _MAX_PIVOT_PAGES, _RECORDS_PER_PAGE
from rl_cli.services.a1000.samples import (
    _MAX_RECORDS_PER_PAGE,
    _MAX_SEARCH_PAGES,
    ReanalysisBatch,
    _extract_hash_from_upload,
    _records_per_page,
    unused_search_inputs,
)
from rl_cli.services.a1000.service import list_from_envelope
from rl_cli.services.base import BaseService
from rl_cli.storage import archives as archives_module
from rl_cli.storage.archives import (
    _MAX_ARCHIVE_BYTES,
    _MAX_ARCHIVE_MEMBERS,
    _MAX_MEMBER_RATIO,
    copy_member,
    extract_private,
)
from rl_cli.storage.files import private_writer, write_private_bytes
from tests.cli_support import flat
from tests.conftest import sdk_response

SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SHA1 = "da39a3ee5e6b4b0d3255bfef95601890afd80709"


@pytest.fixture
def session(a1000_session: A1000Session) -> A1000Session:
    """One connected session; the fixtures below are areas over the one client.

    Each test takes the service that owns the wrapper it exercises, so a
    call reaching for a neighbouring area fails here rather than being
    covered by an object that happened to hold every area.
    """
    a1000_session.client = MagicMock()
    return a1000_session


@pytest.fixture
def a1000_client(session: A1000Session) -> A1000Service:
    return session.service(A1000Service, RichOutput())


@pytest.fixture
def samples(session: A1000Session) -> A1000SampleService:
    return session.service(A1000SampleService, RichOutput())


@pytest.fixture
def reports(session: A1000Session) -> A1000ReportService:
    return session.service(A1000ReportService, RichOutput())


@pytest.fixture
def network(session: A1000Session) -> A1000NetworkService:
    return session.service(A1000NetworkService, RichOutput())


@pytest.fixture
def yara(session: A1000Session) -> A1000YaraService:
    return session.service(A1000YaraService, RichOutput())


@pytest.fixture
def metadata(session: A1000Session) -> A1000MetadataService:
    return session.service(A1000MetadataService, RichOutput())


class TestUploadFile:
    def test_injects_task_id_from_sha1(self, samples, tmp_path):
        sample = tmp_path / "sample.bin"
        sample.write_bytes(b"x")
        samples.client.submit_file_from_path.return_value = sdk_response(
            200, {"rl": {"sample": {"sha1": SHA1}}}
        )
        result = samples.upload_file(sample)
        assert result["task_id"] == SHA1

    def test_missing_file_returns_none(self, samples, tmp_path):
        assert samples.upload_file(tmp_path / "missing.bin") is None

    def test_a_body_that_is_not_an_object_is_handed_back_as_it_came(self, samples, tmp_path):
        """No task_id to inject into a list, so it passes through untouched."""
        sample = tmp_path / "sample.bin"
        sample.write_bytes(b"x")
        samples.client.submit_file_from_path.return_value = sdk_response(200, [{"accepted": True}])

        assert samples.upload_file(sample) == [{"accepted": True}]


class TestAnAnalysisStatusThatCarriedNoResults:
    """The SDK gates these two reports on an analysis-status page it counts.

    ``len(analysis_status.json().get("results"))`` raises ``TypeError``
    for any 200 whose body lacks ``results`` — a proxy interstitial, a
    rewritten error page, a future API version — and the read happens
    inside the SDK, out of reach of the envelope guards every other
    listing here goes through. So the user-facing error was
    ``object of type 'NoneType' has no len() (get_report(...))``, which
    names nothing and reads like an rl-cli bug.
    """

    _CRASH = TypeError("object of type 'NoneType' has no len()")

    @pytest.mark.parametrize(
        "call,sdk_method",
        [
            (lambda service: service.get_report(SHA256), "get_detailed_report_v2"),
            (lambda service: service.get_summary_report_v2(SHA256), "get_summary_report_v2"),
        ],
        ids=["report", "summary-report"],
    )
    def test_it_is_named_as_the_envelope_failure_it_is(self, reports, call, sdk_method):
        reports.output = MagicMock()
        getattr(reports.client, sdk_method).side_effect = self._CRASH

        assert call(reports) is None

        said = reports.output.error.call_args.args[0]
        assert said == "Analysis-status response carried no results list"
        # The status is not read a second time to find out: a pre-flight
        # would double the requests this command makes.
        assert getattr(reports.client, sdk_method).call_count == 1

    def test_a_readable_answer_is_untouched(self, reports):
        reports.client.get_summary_report_v2.return_value = sdk_response(200, {"sha1": SHA1})
        assert reports.get_summary_report_v2(SHA256) == {"sha1": SHA1}


class TestTheGuardReadsWhereTheTypeErrorCameFrom:
    """Not what CPython called it: ``has no len()`` is one release's wording.

    What separates the two cases is the frame: a call the installed SDK no
    longer accepts fails in the frame that makes it, and a failure inside
    the SDK has a frame of its own underneath.
    """

    def test_a_reworded_message_is_still_the_envelope_failure_it_is(self, reports):
        reports.output = MagicMock()
        reports.client.get_summary_report_v2.side_effect = TypeError(
            "object of type 'NoneType' cannot be interpreted as having a length"
        )

        assert reports.get_summary_report_v2(SHA256) is None

        assert reports.output.error.call_args.args[0] == (
            "Analysis-status response carried no results list"
        )

    def test_a_call_the_installed_sdk_refuses_is_not_dressed_as_the_appliances_answer(
        self, reports
    ):
        """This SDK deletes public API in minors; that TypeError is ours to fix."""

        def get_summary_report_v2(sample_hash, *, results):
            raise AssertionError("a call that does not type-check never runs")

        reports.output = MagicMock()
        reports.client.get_summary_report_v2 = get_summary_report_v2

        assert reports.get_summary_report_v2(SHA256) is None

        said = reports.output.error.call_args.args[0]
        assert "carried no results list" not in said
        assert "results" in said


class TestGetReport:
    def test_invalid_hash_rejected(self, reports):
        assert reports.get_report("nothash") is None
        reports.client.get_detailed_report_v2.assert_not_called()

    def test_json_report(self, reports):
        reports.client.get_detailed_report_v2.return_value = sdk_response(200, {"rl": {}})
        assert reports.get_report(SHA256) == {"rl": {}}

    def test_unsupported_format_rejected(self, reports):
        assert reports.get_report(SHA256, report_format="docx") is None

    def test_xml_is_an_alias_for_the_titanium_endpoint(self, reports):
        """The alias lives here, not in the CLI: it is a wire-format detail."""
        reports.client.get_titanium_core_report_v2.return_value = sdk_response(200, {"ticore": {}})
        assert reports.get_report(SHA256, report_format="xml") == {"ticore": {}}
        reports.client.get_titanium_core_report_v2.assert_called_once_with(SHA256)

    def test_pdf_is_created_waited_for_and_downloaded(self, reports, monkeypatch):
        reports.client.create_pdf_report.return_value = sdk_response(200)
        monkeypatch.setattr(A1000ReportService, "_wait_for_report", lambda self, check, **kw: True)
        reports.client.download_pdf_report.return_value = sdk_response(200, content=b"%PDF-1.4")
        assert reports.get_report(SHA256, report_format="pdf") == b"%PDF-1.4"

    def test_failed_pdf_creation_returns_none_not_the_error_body(self, reports):
        reports.client.create_pdf_report.return_value = sdk_response(500, content=b"nope")
        assert reports.get_report(SHA256, report_format="pdf") is None
        reports.client.download_pdf_report.assert_not_called()

    @pytest.mark.parametrize(
        "report_format,binary",
        [("json", False), ("titanium", False), ("xml", False), ("pdf", True)],
    )
    def test_report_is_binary_names_the_variant(self, report_format, binary):
        assert A1000ReportService.report_is_binary(report_format) is binary

    def test_a_format_is_dispatched_and_described_out_of_one_table(self, reports, monkeypatch):
        """What ``get_report`` fetches and what it answers are one fact.

        Written down twice, a format added to the dispatch alone answers
        bytes to a caller that ``report_is_binary`` told to expect a
        structure — and the caller writes the repr of a PDF to its report.
        """
        monkeypatch.setitem(
            reports_module._REPORT_VARIANTS,
            "csv",
            reports_module._ReportVariant(
                binary=True, fetch=lambda service, hash_value, digest: b"sha1,name\n"
            ),
        )

        assert A1000ReportService.report_is_binary("csv") is True
        assert reports.get_report(SHA256, report_format="csv") == b"sha1,name\n"


class TestNoReportFailsWithoutSayingSomething:
    """The PDF flow held two ``return None``s that emitted nothing at all.

    Both sat behind ``if not succeeded(...)``, the branch a status that
    was neither an acceptance nor a refusal fell into: ``report --format
    pdf`` printed a bare "Failed to get report" about an appliance that
    had answered something, and the ``-o json`` variants of the same
    commands wrote nothing and exited 1 with no reason on stderr.
    Neither line was ever executed by the suite.
    """

    def test_a_redirect_from_the_creation_call_names_itself(self, reports, capsys):
        reports.client.create_pdf_report.return_value = sdk_response(302)

        assert reports.get_report(SHA256, report_format="pdf") is None

        reports.client.download_pdf_report.assert_not_called()
        assert "302" in capsys.readouterr().err

    def test_a_redirect_from_the_download_names_itself_too(self, reports, monkeypatch, capsys):
        reports.client.create_pdf_report.return_value = sdk_response(200)
        monkeypatch.setattr(A1000ReportService, "_wait_for_report", lambda self, check, **kw: True)
        reports.client.download_pdf_report.return_value = sdk_response(302)

        assert reports.get_report(SHA256, report_format="pdf") is None
        assert "302" in capsys.readouterr().err

    def test_a_refused_creation_still_carries_the_appliances_own_words(self, reports, capsys):
        reports.client.create_pdf_report.return_value = sdk_response(
            403, {"message": "PDF reports are not licensed"}
        )

        assert reports.get_report(SHA256, report_format="pdf") is None
        assert "not licensed" in capsys.readouterr().err

    def test_a_report_that_was_never_built_is_not_a_redirect(self, reports, monkeypatch):
        """The wait reports its own failure; this branch adds nothing to it."""
        reports.client.create_pdf_report.return_value = sdk_response(200)
        monkeypatch.setattr(A1000ReportService, "_wait_for_report", lambda self, check, **kw: False)

        assert reports.get_report(SHA256, report_format="pdf") is None
        reports.client.download_pdf_report.assert_not_called()

    def test_the_pdf_endpoint_refuses_a_sha512_before_the_appliance_does(self, reports, capsys):
        """The narrower guard, and the branch that reaches ``_pdf_report``'s exit."""
        assert reports.get_report(SHA512, report_format="pdf") is None

        reports.client.create_pdf_report.assert_not_called()
        assert "does not accept SHA512" in capsys.readouterr().err

    def test_the_dynamic_download_names_a_redirect_as_well(self, reports, monkeypatch, capsys):
        monkeypatch.setattr(A1000ReportService, "_wait_for_report", lambda self, check, **kw: True)
        reports.client.download_dynamic_analysis_report.return_value = sdk_response(302)

        assert reports.download_dynamic_report(SHA1) is None
        assert "302" in capsys.readouterr().err

    def test_the_dynamic_download_refuses_a_hash_that_is_not_a_sha1(self, reports, capsys):
        assert reports.download_dynamic_report(SHA256) is None

        reports.client.download_dynamic_analysis_report.assert_not_called()
        assert "SHA1" in capsys.readouterr().err

    def test_the_dynamic_download_stops_when_the_report_never_builds(self, reports, monkeypatch):
        """A wait that times out returns without ever asking for the bytes."""
        monkeypatch.setattr(A1000ReportService, "_wait_for_report", lambda self, check, **kw: False)

        assert reports.download_dynamic_report(SHA1) is None
        reports.client.download_dynamic_analysis_report.assert_not_called()

    def test_a_bare_bool_wrapper_no_longer_fails_in_silence(self, samples, capsys):
        """``return succeeded(response)``: fourteen wrappers are written this way.

        A 3xx answered ``False`` and printed nothing, so ``a1000 delete``
        said "Failed to delete sample" and stopped — no status, no
        explanation, nothing to act on.
        """
        samples.client.delete_samples.return_value = sdk_response(302)

        assert samples.delete_sample(SHA256) is False
        assert "302" in capsys.readouterr().err


class TestBuildSearchQuery:
    """The A1000 search DSL is the service's business, not the CLI's."""

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({}, "available:true"),
            ({"malicious": True}, "classification:malicious"),
            ({"clean": True}, "classification:clean"),
            ({"malicious": True, "clean": True}, "classification:malicious"),
        ],
    )
    def test_flags_translate_to_queries(self, kwargs, expected):
        assert A1000SampleService.build_search_query(None, **kwargs) == expected

    def test_explicit_query_wins(self):
        assert A1000SampleService.build_search_query("sha256:abc", malicious=True) == "sha256:abc"

    @pytest.mark.parametrize(
        "query,kwargs,expected",
        [
            (None, {}, []),
            (None, {"malicious": True}, []),
            (None, {"clean": True}, []),
            (None, {"malicious": True, "clean": True}, ["clean"]),
            ("riskscore:1", {"clean": True}, ["clean"]),
            ("riskscore:1", {"malicious": True, "clean": True}, ["malicious", "clean"]),
            ("riskscore:1", {}, []),
        ],
    )
    def test_the_resolution_names_what_it_dropped(self, query, kwargs, expected):
        """Resolving a conflict in silence searched for something else."""
        assert unused_search_inputs(query, **kwargs) == expected


class TestWhatWasIgnoredFollowsWhatWon:
    """The two answers must come from one ordered source, not from two rules.

    ``_CLASSIFICATION_QUERIES`` is ordered and its order is the
    precedence. Written out a second time as a literal, the report of what
    was dropped named the shortcut that had in fact been used the moment
    the precedence moved.
    """

    def test_moving_the_precedence_moves_both_answers(self, monkeypatch):
        monkeypatch.setattr(
            samples_module,
            "_CLASSIFICATION_QUERIES",
            dict(reversed(list(samples_module._CLASSIFICATION_QUERIES.items()))),
        )

        assert (
            A1000SampleService.build_search_query(None, malicious=True, clean=True)
            == "classification:clean"
        )
        assert unused_search_inputs(None, malicious=True, clean=True) == ["malicious"]


class TestYaraMatches:
    """The ``results`` envelope is unwrapped here so the CLI sees a list."""

    def test_returns_the_matches_list(self, yara):
        yara.client.get_yara_ruleset_matches_v2.return_value = sdk_response(
            200, {"count": 1, "results": [{"sha256": SHA256}]}
        )
        assert yara.get_yara_matches("myrules") == [{"sha256": SHA256}]

    def test_filters_to_the_requested_sample(self, yara):
        yara.client.get_yara_ruleset_matches_v2.return_value = sdk_response(
            200, {"results": [{"sha256": SHA256}, {"sha1": SHA1}]}
        )
        assert yara.get_yara_matches("myrules", SHA1) == [{"sha1": SHA1}]

    def test_no_matches_is_an_empty_list(self, yara):
        yara.client.get_yara_ruleset_matches_v2.return_value = sdk_response(200, {"results": []})
        assert yara.get_yara_matches("myrules") == []


class TestConnectionSummary:
    """The CLI asks the service how it is authenticated, not the settings."""

    def test_token_auth(self, a1000_client):
        a1000_client.a1000_settings.token = "secret"
        summary = a1000_client.connection_summary()
        assert summary["host"] == a1000_client.a1000_settings.host
        assert summary["auth"] == "Token"
        assert "secret" not in str(summary)

    def test_username_password_auth(self, a1000_client):
        a1000_client.a1000_settings.token = None
        assert a1000_client.connection_summary()["auth"] == "Username/Password"


class TestATestedConnectionSaysWhyItFailed:
    """It hand-rolled the decorator's guard and then swallowed everything.

    The availability probe is the only place a user ever hears about this
    call, and it can report only what the notifier was told — so a bare
    ``except Exception: return False`` left it reconstructing the reason
    from elsewhere, or naming none at all.
    """

    def test_a_failure_is_reported_and_still_answers_false(self, a1000_client, capsys):
        a1000_client.client.file_analysis_status.side_effect = TimeoutError("read timed out")

        assert a1000_client.test_connection() is False
        assert "read timed out" in capsys.readouterr().err

    def test_check_status_is_what_is_asked(self, a1000_client):
        """The SDK's own probe call, made here so its answer can be judged."""
        a1000_client.client.file_analysis_status.return_value = sdk_response(
            200, {"results": [{"status": "not_found"}]}
        )

        assert a1000_client.test_connection() is True
        a1000_client.client.file_analysis_status.assert_called_once_with(
            sample_hashes=["0" * 40], sample_status="processed"
        )

    def test_an_accepted_probe_says_nothing(self, a1000_client, capsys):
        a1000_client.client.file_analysis_status.return_value = sdk_response(200, {"results": []})

        assert a1000_client.test_connection() is True
        assert capsys.readouterr().err == ""

    @pytest.mark.parametrize("status", [504, 407, 511, 501, 410, 422, 428, 302])
    def test_a_status_the_sdk_stays_quiet_about_is_still_a_failure(
        self, a1000_client, capsys, status
    ):
        """The SDK's error map holds none of these, and silence is not health.

        A dead gateway, an authenticating proxy and a captive portal each
        answer one of these, so a probe that read the SDK's silence as an
        acceptance called an unreachable appliance available — and the
        availability cache then repeated it for 24 h.
        """
        a1000_client.client.file_analysis_status.return_value = sdk_response(status)

        assert a1000_client.test_connection() is False
        assert f"HTTP {status}" in " ".join(capsys.readouterr().err.split())


class TestDownloads:
    def test_download_sample_writes_bytes(self, samples, tmp_path):
        samples.client.download_sample.return_value = sdk_response(200, content=b"payload")
        out = tmp_path / "out.bin"
        assert samples.download_sample(SHA256, out) is True
        assert out.read_bytes() == b"payload"

    def test_downloaded_sample_is_never_world_readable(self, samples, tmp_path):
        samples.client.download_sample.return_value = sdk_response(200, content=b"MZ malware")
        out = tmp_path / "sample.malware"
        samples.download_sample(SHA256, out)
        assert stat.S_IMODE(out.stat().st_mode) == 0o600

    def test_extracted_archive_is_never_world_readable(self, samples, tmp_path, monkeypatch):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("inner.txt", "data")
        samples.client.download_extracted_files.return_value = sdk_response(
            200, content=buffer.getvalue()
        )
        captured: list[int] = []
        real_unlink = Path.unlink

        def capture_then_unlink(self, *args, **kwargs):
            captured.append(stat.S_IMODE(self.stat().st_mode))
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", capture_then_unlink)
        samples.download_extracted_files(SHA256, tmp_path)
        assert captured == [0o600]

    def test_download_sample_failure_status(self, samples, tmp_path):
        samples.client.download_sample.return_value = sdk_response(404)
        assert samples.download_sample(SHA256, tmp_path / "out.bin") is False

    def test_a_refused_download_reports_what_the_appliance_said(self, samples, tmp_path):
        """It used to print the status and throw the explanation away.

        ``validate_response`` answered ``False`` and the caller wrote
        "Download failed with status: 500" over the top of a body that
        said which quota had run out. Every status check now goes through
        the reader that keeps it.
        """
        samples.output = MagicMock()
        samples.client.download_sample.return_value = sdk_response(
            500, {"message": "Sample storage is unavailable"}
        )

        assert samples.download_sample(SHA256, tmp_path / "out.bin") is False

        reported = samples.output.error.call_args.args[0]
        assert "HTTP 500" in reported and "Sample storage is unavailable" in reported

    @pytest.mark.parametrize(
        "call,sdk_method",
        [
            (lambda service, path: service.download_sample(SHA256, path), "download_sample"),
            (
                lambda service, path: service.download_extracted_files(SHA256, path),
                "download_extracted_files",
            ),
        ],
        ids=["sample", "extracted-files"],
    )
    def test_a_status_that_is_neither_success_nor_refusal_is_named(
        self, samples, tmp_path, call, sdk_method
    ):
        """A redirect is not a sample, and carries nothing to raise about."""
        samples.output = MagicMock()
        getattr(samples.client, sdk_method).return_value = sdk_response(302)

        assert call(samples, tmp_path) is False

        assert "302" in samples.output.error.call_args.args[0]

    def test_download_extracted_files_unzips(self, samples, tmp_path):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("inner.txt", "data")
        samples.client.download_extracted_files.return_value = sdk_response(
            200, content=buffer.getvalue()
        )
        assert samples.download_extracted_files(SHA256, tmp_path) is True
        assert (tmp_path / "inner.txt").read_text() == "data"
        assert not (tmp_path / "extracted_files.zip").exists()


class TestIpGuard:
    def test_invalid_ip_rejected_without_sdk_call(self, network):
        assert network.get_ip_report("999.1.1.1") is None
        network.client.network_ip_addr_report.assert_not_called()

    def test_valid_ip_passes_through(self, network):
        network.client.network_ip_addr_report.return_value = sdk_response(200, {"ip": "1.2.3.4"})
        assert network.get_ip_report("1.2.3.4") == {"ip": "1.2.3.4"}

    @pytest.mark.parametrize(
        "method",
        [
            "get_files_from_ip",
            "get_domains_from_ip",
            "get_urls_from_ip",
            "get_files_from_ip_aggregated",
            "get_domains_from_ip_aggregated",
            "get_urls_from_ip_aggregated",
        ],
    )
    def test_every_ip_pivot_rejects_an_invalid_address(self, network, method):
        """The same guard fronts every pivot, not just get_ip_report."""
        assert getattr(network, method)("999.1.1.1") is None


class TestAnUnsupportedHashIsRejectedBeforeTheSdk:
    """Every hash-taking mutation and download guards on the parsed hash."""

    @pytest.mark.parametrize(
        "call, expected",
        [
            (lambda s, p: s.delete_sample("nothash"), False),
            (lambda s, p: s.reanalyze_sample("nothash"), None),
            (lambda s, p: s.download_sample("nothash", p / "out.bin"), False),
            (lambda s, p: s.list_extracted_files("nothash"), None),
            (lambda s, p: s.download_extracted_files("nothash", p), False),
        ],
    )
    def test_a_bad_hash_short_circuits(self, samples, tmp_path, call, expected):
        assert call(samples, tmp_path) is expected


class TestSearchRefusesANonPositiveWindow:
    """A page holds a positive number of records starting at a positive page."""

    def test_a_limit_below_one_is_refused(self, samples):
        samples.output = MagicMock()
        assert samples.advanced_search("available:true", limit=0) is None
        assert samples.output.error.called

    def test_a_page_below_one_is_refused(self, samples):
        samples.output = MagicMock()
        assert samples.advanced_search("available:true", page=0) is None
        assert samples.output.error.called


class TestListSamples:
    """``list`` and ``search`` read one endpoint; they must answer alike."""

    def test_returns_whole_entries(self, samples):
        entry = {
            "sha256": SHA256,
            "threat_status": "malicious",
            "file_names": ["a-really-long-sample-file-name-nobody-should-truncate.exe"],
            "sample_type": "PE32 executable (GUI) Intel 80386, for MS Windows",
        }
        samples.client.advanced_search_v3.return_value = sdk_response(
            200, {"rl": {"web_search_api": {"entries": [entry]}}}
        )
        assert samples.list_samples(limit=5) == [entry]

    @pytest.mark.parametrize("call", ["list_samples", "advanced_search"])
    def test_missing_envelope_is_a_failure_not_an_empty_result(self, samples, call):
        samples.output = MagicMock()
        samples.client.advanced_search_v3.return_value = sdk_response(200, {"unexpected": True})
        args = () if call == "list_samples" else ("available:true",)
        assert getattr(samples, call)(*args) is None
        assert samples.output.error.called

    @pytest.mark.parametrize("call", ["list_samples", "advanced_search"])
    def test_non_2xx_is_reported_by_both(self, samples, call):
        samples.output = MagicMock()
        samples.client.advanced_search_v3.return_value = sdk_response(500, {})
        args = () if call == "list_samples" else ("available:true",)
        assert getattr(samples, call)(*args) is None
        assert "500" in samples.output.error.call_args.args[0]


class TestUserTags:
    def test_list_payload(self, metadata):
        metadata.client.get_user_tags.return_value = sdk_response(200, ["a", "b"])
        assert metadata.get_user_tags(SHA256) == ["a", "b"]

    def test_dict_payload(self, metadata):
        metadata.client.get_user_tags.return_value = sdk_response(200, {"tags": ["a"]})
        assert metadata.get_user_tags(SHA256) == ["a"]

    def test_a_body_with_no_tag_list_is_unreadable_not_empty(self, metadata):
        """``.get("tags", [])`` reported a shape it cannot read as no tags."""
        metadata.output = MagicMock()
        metadata.client.get_user_tags.return_value = sdk_response(200, {"user_tags": ["apt"]})

        assert metadata.get_user_tags(SHA256) is None
        assert metadata.output.error.called

    def test_a_body_that_hides_no_records_is_unreadable_too(self, metadata):
        """The ``required`` key is what makes this one a failed read.

        A body carrying records under another name is refused whether or
        not the key is required; this one — an object with no tag list at
        all — is refused only because the key has to be there.
        """
        metadata.output = MagicMock()
        metadata.client.get_user_tags.return_value = sdk_response(200, {"detail": "Not found."})

        assert metadata.get_user_tags(SHA256) is None
        assert "carried no tag list" in metadata.output.error.call_args.args[0]

    def test_removing_every_tag_stops_when_the_tag_list_was_absent(self, metadata):
        """ "All" resolved against an unread list removes nothing and says it worked."""
        metadata.output = MagicMock()
        metadata.client.get_user_tags.return_value = sdk_response(200, {"detail": "Not found."})

        assert metadata.remove_all_user_tags(SHA256) is None
        metadata.client.delete_user_tags.assert_not_called()

    def test_removing_every_tag_stops_when_the_tags_could_not_be_read(self, metadata):
        """ "All" is resolved against the tags the sample has; unread is not none."""
        metadata.output = MagicMock()
        metadata.client.get_user_tags.return_value = sdk_response(200, {"user_tags": ["apt"]})

        assert metadata.remove_all_user_tags(SHA256) is None
        metadata.client.delete_user_tags.assert_not_called()

    @pytest.mark.parametrize(
        "call,sdk_method",
        [
            (lambda service: service.get_user_tags(SHA256), "get_user_tags"),
            (lambda service: service.list_containers(SHA256), "list_containers_for_hashes"),
        ],
        ids=["tags", "containers"],
    )
    def test_a_redirect_is_not_an_empty_answer(self, metadata, call, sdk_method, capsys):
        """A 3xx is not a sample with no tags and no parent container.

        ``requests`` would have followed a redirect, so one arriving here
        means the appliance answered something this reader has no body
        for — which is a failed read. These wrappers went through a
        bridge that answered ``False`` on the same status and said
        nothing; now ``succeeded`` names the status and the wrapper keeps
        saying "we do not know" rather than "none".
        """
        getattr(metadata.client, sdk_method).return_value = sdk_response(302)

        assert call(metadata) is None
        assert "302" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "call,sdk_method",
        [
            (lambda service: service.get_user_tags(SHA256), "get_user_tags"),
            (lambda service: service.list_containers(SHA256), "list_containers_for_hashes"),
        ],
        ids=["tags", "containers"],
    )
    def test_a_body_stating_null_is_not_an_empty_answer_either(self, metadata, call, sdk_method):
        """``null`` is the one body ``json_on`` still answers ``None`` for.

        Every status that is not an acceptance now leaves ``json_on``
        through a raise, so the ``is None`` guard these readers carry is
        about the body alone: a 200 whose payload is literally ``null``
        states nothing we can read, and that is not "this sample has
        none".
        """
        response = sdk_response(200)
        response.json.return_value = None
        getattr(metadata.client, sdk_method).return_value = response

        assert call(metadata) is None


class TestContainersAndExtractedFilesReadTheirEnvelope:
    """An answer we cannot parse is a failed read, not an empty result.

    ``models.payload.search_page`` already decides it that way for
    Advanced Search; these three read the same class of envelope and used
    a ``[]`` default instead, so a sample whose extracted files, tags or
    parent containers the reader simply could not find was reported as
    having none — exit 0, in a malware-analysis tool.
    """

    def test_extracted_files_stated_as_something_other_than_a_list_are_a_failed_read(self, samples):
        samples.output = MagicMock()
        samples.client.list_extracted_files_v2.return_value = sdk_response(
            200, {"results": {"sha1": SHA1}}
        )

        assert samples.list_extracted_files(SHA256) is None
        assert samples.output.error.called

    def test_the_documented_envelope_is_still_read(self, samples):
        samples.client.list_extracted_files_v2.return_value = sdk_response(
            200, {"count": 1, "results": [{"sha1": SHA1}]}
        )

        assert samples.list_extracted_files(SHA256) == [{"sha1": SHA1}]

    def test_a_body_stating_null_is_a_failed_read_not_an_empty_one(self, samples):
        """``null`` is the one body ``json_on`` still answers ``None`` for."""
        response = sdk_response(200)
        response.json.return_value = None
        samples.client.list_extracted_files_v2.return_value = response

        assert samples.list_extracted_files(SHA256) is None

    def test_containers_under_an_unknown_key_are_a_failed_read(self, metadata):
        metadata.output = MagicMock()
        metadata.client.list_containers_for_hashes.return_value = sdk_response(
            200, {"containers": [{"sha1": SHA1}]}
        )

        assert metadata.list_containers(SHA256) is None
        assert metadata.output.error.called

    def test_containers_keyed_by_hash_are_read_as_the_map_they_look_like(self, metadata):
        """The bulk endpoint omits the hashes it has nothing for.

        That is a map's behaviour, not a list's, and nothing in the
        vendored SDK — no aggregator, no test, no README line — says
        which of the two the appliance sends. Reading only the list shape
        exited 1 on an answer that had the containers in it.
        """
        metadata.output = MagicMock()
        metadata.client.list_containers_for_hashes.return_value = sdk_response(
            200, {"results": {SHA256: [{"sha1": SHA1}]}}
        )

        assert metadata.list_containers(SHA256) == [{"sha1": SHA1}]
        assert not metadata.output.error.called

    def test_a_hash_keyed_map_of_single_containers_is_read_too(self, metadata):
        metadata.client.list_containers_for_hashes.return_value = sdk_response(
            200, {"results": {SHA256: {"sha1": SHA1}}}
        )

        assert metadata.list_containers(SHA256) == [{"sha1": SHA1}]

    def test_a_map_entry_that_is_neither_is_still_a_failed_read(self, metadata):
        metadata.output = MagicMock()
        metadata.client.list_containers_for_hashes.return_value = sdk_response(
            200, {"results": {SHA256: "archive.zip"}}
        )

        assert metadata.list_containers(SHA256) is None
        assert metadata.output.error.called

    def test_a_hash_keyed_container_answer_exits_zero(self, a1000_cli):
        _, result = a1000_cli(
            ["containers", SHA256],
            "list_containers_for_hashes",
            {"results": {SHA256: [{"sha1": SHA1}]}},
        )

        assert result.exit_code == 0, result.output
        assert "Found 1 containers" in flat(result)

    def test_containers_under_an_unknown_key_exit_one_saying_why(self, a1000_cli):
        """This envelope has no corroboration in the SDK, so it stays strict."""
        _, result = a1000_cli(
            ["containers", SHA256], "list_containers_for_hashes", {"containers": [{"sha1": SHA1}]}
        )

        assert result.exit_code == 1, result.output
        assert "carried no results list" in flat(result)
        assert "No containers found" not in flat(result)

    def test_an_empty_container_body_is_the_answer_that_it_has_none(self, metadata):
        """The bulk endpoint omits a hash that has no container at all."""
        metadata.output = MagicMock()
        metadata.client.list_containers_for_hashes.return_value = sdk_response(200, {})

        assert metadata.list_containers(SHA256) == []
        assert not metadata.output.error.called


class TestWaitForAnalysis:
    def test_polls_the_real_spinner_task_until_processed(self, samples):
        """Exercises the spinner the service actually gets, not a mock."""
        samples.client.file_analysis_status.side_effect = [
            sdk_response(200, {"results": [{"status": "not_found"}]}),
            sdk_response(200, {"results": [{"status": "processed"}]}),
        ]
        assert samples.wait_for_analysis(SHA1, interval=0) == {"status": "processed"}

    def test_an_appliance_that_never_answers_is_not_asked_sixty_times(self, samples, monkeypatch):
        """Giving up is reported as itself, not as the timeout it is not."""
        samples.output = MagicMock()
        samples.output.progress_spinner.return_value.__enter__.return_value.task_ids = [0]
        monkeypatch.setattr(http_module.time, "sleep", lambda _: None)
        samples.client.file_analysis_status.return_value = sdk_response(
            429, {"message": "Too many requests"}
        )

        assert samples.wait_for_analysis(SHA1) is None

        assert samples.client.file_analysis_status.call_count == 3
        assert "Giving up" in samples.output.error.call_args.args[0]

    def test_an_answer_the_reader_cannot_make_sense_of_still_counts_as_no_answer(
        self, samples, monkeypatch
    ):
        """The give-up rule is for answers we could not read, and only those."""
        samples.output = MagicMock()
        samples.output.progress_spinner.return_value.__enter__.return_value.task_ids = [0]
        monkeypatch.setattr(http_module.time, "sleep", lambda _: None)
        samples.client.file_analysis_status.return_value = sdk_response(200, ["not an envelope"])

        assert samples.wait_for_analysis(SHA1) is None

        assert samples.client.file_analysis_status.call_count == 3
        assert "Giving up" in samples.output.error.call_args.args[0]


class TestAHashTheApplianceHasNotRegisteredYetIsNotAFailedCheck:
    """``{"results": []}`` is the answer to a freshly uploaded hash.

    It is a 200: the appliance answered, it simply holds no entry for
    this hash yet. Folded into ``PollState.UNANSWERED`` alongside "the
    SDK raised" and "the status was not one we read", three of them
    tripped :class:`PollBackoff`'s give-up rule — so ``upload --wait
    --timeout 300`` stopped after three polls, about 35 seconds in, and
    reported "Giving up: the last status checks all failed" over an
    analysis that was running normally. It is exactly the taxonomy
    ``UNANSWERED`` was introduced to keep separate, flattened one call
    further down.
    """

    def _waiting(self, samples) -> None:
        samples.output = MagicMock()
        samples.output.progress_spinner.return_value.__enter__.return_value.task_ids = [0]

    def test_an_empty_results_page_is_waited_through_not_given_up_on(self, samples):
        self._waiting(samples)
        samples.client.file_analysis_status.side_effect = [
            sdk_response(200, {"results": []}),
            sdk_response(200, {"results": []}),
            sdk_response(200, {"results": []}),
            sdk_response(200, {"results": [{"status": "processed"}]}),
        ]

        assert samples.wait_for_analysis(SHA1, interval=0) == {"status": "processed"}

        assert samples.client.file_analysis_status.call_count == 4
        assert not samples.output.error.called

    def test_a_body_with_no_results_key_at_all_is_waited_through_too(self, samples):
        """A 200 is an answer whatever it spells; only a failure to read is not."""
        self._waiting(samples)
        samples.client.file_analysis_status.side_effect = [
            sdk_response(200, {"count": 0}),
            sdk_response(200, {"count": 0}),
            sdk_response(200, {"count": 0}),
            sdk_response(200, {"results": [{"status": "processed"}]}),
        ]

        assert samples.wait_for_analysis(SHA1, interval=0) == {"status": "processed"}
        assert not samples.output.error.called

    def test_the_wait_still_runs_the_whole_timeout_it_was_given(self, samples, monkeypatch):
        """The 300 seconds the caller asked for, not 35 of them.

        With the clock stopped the loop leaves through the deadline, and
        the failure it reports is the timeout — never the give-up, which
        is about an appliance that did not answer.
        """
        self._waiting(samples)
        monkeypatch.setattr(http_module.time, "sleep", lambda _: None)
        samples.client.file_analysis_status.return_value = sdk_response(200, {"results": []})
        ticks = iter([0.0, 0.0, 0.0, 0.0, 0.0])
        monkeypatch.setattr(http_module.time, "time", lambda: next(ticks, 10_000.0))

        assert samples.wait_for_analysis(SHA1, timeout=300) is None

        assert samples.client.file_analysis_status.call_count > 3
        assert "did not complete within 300 seconds" in samples.output.error.call_args.args[0]


class TestBatchReanalyze:
    def test_failure_default_is_immutable(self, samples):
        """A mutable default is one object shared by every failing call."""
        samples.client.reanalyze_samples_v2.side_effect = RuntimeError("boom")
        result = samples.batch_reanalyze_samples([SHA256])
        assert not result
        with pytest.raises(AttributeError):
            result.append({"hash": SHA256})

    def test_passes_hash_list_and_unwraps_results(self, samples):
        samples.client.reanalyze_samples_v2.return_value = sdk_response(
            200, {"results": [{"hash": SHA256}]}
        )
        result = samples.batch_reanalyze_samples([SHA256, SHA1])
        assert result == [{"hash": SHA256}]
        args, _kwargs = samples.client.reanalyze_samples_v2.call_args
        assert args[0] == [SHA256, SHA1]

    def test_the_single_sample_wrapper_reports_a_refused_status(self, samples, capsys):
        """A status the SDK does not raise for used to be read as the answer."""
        samples.client.reanalyze_samples_v2.return_value = sdk_response(
            403, {"message": "Reanalysis is not permitted for this account"}
        )

        assert samples.reanalyze_sample(SHA256) is None
        assert "not permitted" in capsys.readouterr().err

    def test_a_results_section_that_is_not_a_list_is_a_failed_submission(self, samples, capsys):
        """The branch the docstring argues hardest for, and the one never run.

        ``result_data.get("results", [result_data])`` turned a body
        carrying no per-sample answers into one entry with no verdict in
        it, which the command counted as one sample accepted — for a
        batch of five — and exited 0. An envelope we cannot read is a
        failed submission, which is the line every A1000 listing draws.
        """
        samples.output = MagicMock()
        samples.client.reanalyze_samples_v2.return_value = sdk_response(
            200, {"results": {"0": {"hash": SHA256}}}
        )

        assert samples.batch_reanalyze_samples([SHA256]) is None
        assert "carried no results list" in samples.output.error.call_args.args[0]

    def test_a_body_that_is_already_the_list_is_taken_as_it_is(self, samples):
        """Some appliances answer the array bare rather than in an envelope."""
        samples.client.reanalyze_samples_v2.return_value = sdk_response(200, [{"hash": SHA256}])

        assert samples.batch_reanalyze_samples([SHA256]) == [{"hash": SHA256}]


class TestABatchIsSummarisedWhereItIsRead:
    """How many were taken was counted in the command, over the table's own grading.

    Two readings of "was this sample taken" — one for the sentence and the
    exit status, one for the Status column — with nothing keeping them in
    step. The counting is a fact about the answer, so it is read off the
    answer once and both the sentence and the rows come from it.
    """

    def _entry(self, index: int, code: int) -> dict[str, Any]:
        return {
            "detail": {"sha256": f"{index:064x}"},
            "analysis": [{"name": "core", "code": code, "message": "Sample not found"}],
        }

    def test_the_counts_are_of_the_answer_entry_by_entry(self):
        batch = ReanalysisBatch.of([self._entry(0, 201), self._entry(1, 404), self._entry(2, 404)])

        assert (batch.answered, batch.accepted, batch.refused) == (3, 1, 2)

    def test_an_entry_that_is_not_a_record_was_answered_and_not_accepted(self):
        """The count walks the answer as it came; only the table filters it."""
        batch = ReanalysisBatch.of(["nonsense", self._entry(0, 201)])

        assert (batch.answered, batch.accepted, batch.refused) == (2, 1, 1)

    def test_the_answer_is_carried_as_it_came(self):
        """``-o json`` emits it and the table draws it: neither gets a rewrite."""
        entries = [self._entry(0, 201)]

        assert ReanalysisBatch.of(entries).entries is entries

    def test_an_answer_about_nothing_counts_nothing(self):
        assert (ReanalysisBatch.of([]).answered, ReanalysisBatch.of([]).accepted) == (0, 0)


class TestNetworkIntelligencePayloads:
    """Non-aggregated endpoints use endpoint-specific payload keys."""

    def test_files_from_ip_reads_downloaded_files(self, network):
        network.client.network_files_from_ip.return_value = sdk_response(
            200, {"downloaded_files": [{"sha256": SHA256}]}
        )
        assert network.get_files_from_ip("1.2.3.4") == [{"sha256": SHA256}]

    def test_domains_from_ip_reads_resolutions(self, network):
        network.client.network_ip_to_domain.return_value = sdk_response(
            200, {"resolutions": [{"host_name": "evil.test"}]}
        )
        assert network.get_domains_from_ip("1.2.3.4") == [{"host_name": "evil.test"}]

    def test_urls_from_ip_reads_urls(self, network):
        network.client.network_urls_from_ip.return_value = sdk_response(
            200, {"urls": [{"url": "http://evil.test"}]}
        )
        assert network.get_urls_from_ip("1.2.3.4") == [{"url": "http://evil.test"}]


def pivot_pages(key: str, *pages: tuple[list[dict[str, Any]], str | None]) -> list[MagicMock]:
    """Successive IP-pivot answers, one ``(records, next_page)`` each."""
    return [sdk_response(200, {key: records, "next_page": cursor}) for records, cursor in pages]


class TestAggregatedWrappers:
    """``--all`` reads the same envelope as the first page, one page at a time."""

    def test_files_from_ip_aggregated(self, network):
        network.client.network_files_from_ip.side_effect = pivot_pages(
            "downloaded_files", ([{"sha256": SHA256}], None)
        )
        assert network.get_files_from_ip_aggregated("1.2.3.4") == [{"sha256": SHA256}]

    def test_domains_from_ip_aggregated(self, network):
        network.client.network_ip_to_domain.side_effect = pivot_pages(
            "resolutions", ([{"host_name": "x.test"}], None)
        )
        assert network.get_domains_from_ip_aggregated("1.2.3.4") == [{"host_name": "x.test"}]

    def test_urls_from_ip_aggregated(self, network):
        network.client.network_urls_from_ip.side_effect = pivot_pages(
            "urls", ([{"url": "http://x.test"}], None)
        )
        assert network.get_urls_from_ip_aggregated("1.2.3.4") == [{"url": "http://x.test"}]

    def test_every_page_the_appliance_offers_is_fetched(self, network):
        network.client.network_ip_to_domain.side_effect = pivot_pages(
            "resolutions",
            ([{"host_name": "a.test"}], "page-2"),
            ([{"host_name": "b.test"}], "page-3"),
            ([{"host_name": "c.test"}], None),
        )

        found = network.get_domains_from_ip_aggregated("1.2.3.4")

        assert found == [{"host_name": n} for n in ("a.test", "b.test", "c.test")]
        cursors = [call.kwargs["page"] for call in network.client.network_ip_to_domain.mock_calls]
        assert cursors == [None, "page-2", "page-3"], "the appliance's own cursor is what is sent"

    def test_the_page_size_is_sent_rather_than_left_to_the_sdk(self, network):
        """What ``_MAX_PIVOT_PAGES`` is worth in records depends on it.

        The SDK defaults it to 500 and neither the first page nor the walk
        sent it, so the budget's arithmetic — 200 pages of 500 is
        ``MAX_LIMIT`` — rested on a default in a vendored library, and the
        two paths could have disagreed about what one page is.
        """
        network.client.network_ip_to_domain.side_effect = pivot_pages("resolutions", ([], None))
        network.get_domains_from_ip("1.2.3.4")
        network.client.network_ip_to_domain.side_effect = pivot_pages("resolutions", ([], None))
        network.get_domains_from_ip_aggregated("1.2.3.4")

        sizes = [
            call.kwargs["page_size"] for call in network.client.network_ip_to_domain.call_args_list
        ]
        assert sizes == [_RECORDS_PER_PAGE, _RECORDS_PER_PAGE]

    def test_a_sparse_page_mid_corpus_does_not_truncate_the_answer(self, network):
        """A page that carried nothing while promising more is not the end.

        Only a missing ``next_page`` is. The appliance filters each page
        server-side, so a page can come back empty with pages of records
        still behind it, and stopping there hands back part of the answer
        as though it were all of it — under no warning, because ``--all``
        is what the analyst already typed.
        """
        network.output = MagicMock()
        network.client.network_ip_to_domain.side_effect = pivot_pages(
            "resolutions",
            ([{"host_name": "a.test"}], "page-2"),
            ([], "page-3"),
            ([{"host_name": "c.test"}], None),
        )

        found = network.get_domains_from_ip_aggregated("1.2.3.4")

        assert found == [{"host_name": "a.test"}, {"host_name": "c.test"}]
        assert network.client.network_ip_to_domain.call_count == 3
        assert not network.output.warning.called

    def test_aggregated_still_guards_invalid_ip(self, network):
        assert network.get_files_from_ip_aggregated("999.1.1.1") is None
        network.client.network_files_from_ip.assert_not_called()

    def test_an_unreadable_page_ends_the_walk_as_a_failed_lookup(self, network):
        network.output = MagicMock()
        network.client.network_urls_from_ip.side_effect = [
            *pivot_pages("urls", ([{"url": "http://x.test"}], "page-2")),
            sdk_response(200, {"urls": "not a list"}),
        ]

        assert network.get_urls_from_ip_aggregated("1.2.3.4") is None

        assert "carried no URLs list" in network.output.error.call_args.args[0]


class TestAnIpWalkThatStopsMakingProgressStops:
    """The three ``--all`` pivots could not terminate, and had no budget.

    Each was the SDK's ``*_aggregated`` method, which is a ``while True``
    whose only exit is ``not next_page`` while no ``max_results`` is set —
    and none was set, because ``--all`` asks for the lot (SDK
    a1000.py:2296-2302, 2364-2370, 2452-2458). Against a page carrying no
    records and a cursor, or a cursor that never moved, the probe was still
    going after 3001 requests; the second shape also collected the same
    page over and over until the memory ran out.

    Nothing else bounded them: a pivot takes no ``--limit`` and, unlike the
    ticloud pivots, no ``--max-results`` either.
    """

    def _endpoint(self, network, pages):
        network.output = MagicMock()
        network.client.network_ip_to_domain.side_effect = pages
        return network.client.network_ip_to_domain

    def test_an_appliance_that_only_ever_sends_empty_pages_costs_the_budget(self, network):
        """Nothing but empty pages and a cursor that keeps moving.

        The walk cannot tell this from a corpus whose records sit past a
        run of filtered pages, so it reads on, and the page budget is what
        ends it — the same bound a corpus of full pages runs into.
        """
        network.output = MagicMock()
        page = itertools.count()
        network.client.network_ip_to_domain.side_effect = lambda *_a, **_kw: sdk_response(
            200, {"resolutions": [], "next_page": f"page-{next(page)}"}
        )

        found = network.get_domains_from_ip_aggregated("1.2.3.4")

        assert found == []
        assert network.client.network_ip_to_domain.call_count == _MAX_PIVOT_PAGES
        assert "holds more domains than this" in network.output.warning.call_args.args[0]

    def test_an_empty_page_on_a_cursor_that_never_moves_ends_the_walk(self, network):
        """The shape the deleted empty-page rule was written for.

        Two requests, and no page of records is lost to it: the repeated
        cursor is what says the appliance has stopped making progress.
        """
        api = self._endpoint(network, pivot_pages("resolutions", ([], "stuck"), ([], "stuck")))

        assert network.get_domains_from_ip_aggregated("1.2.3.4") == []
        assert api.call_count == 2
        assert "already sent" in network.output.warning.call_args.args[0]

    def test_a_cursor_that_never_moves_ends_the_walk(self, network):
        api = self._endpoint(
            network,
            pivot_pages(
                "resolutions",
                ([{"host_name": "a.test"}], "stuck"),
                ([{"host_name": "a.test"}], "stuck"),
            ),
        )

        found = network.get_domains_from_ip_aggregated("1.2.3.4")

        assert len(found) == 2, "the page already fetched is kept; the repeat is what stops it"
        assert api.call_count == 2
        assert "already sent" in network.output.warning.call_args.args[0]

    def test_a_cursor_that_cycles_between_two_pages_ends_the_walk(self, network):
        """Every cursor is remembered, not only the one page back.

        An appliance alternating A, B, A, B … moves its cursor every time,
        so comparing against the previous one alone never fires and the
        walk pays the whole page budget to collect two distinct pages over
        and over.
        """
        network.output = MagicMock()
        cursors = itertools.cycle(("a", "b"))
        network.client.network_ip_to_domain.side_effect = lambda *_a, **_kw: sdk_response(
            200, {"resolutions": [{"host_name": "a.test"}], "next_page": next(cursors)}
        )

        found = network.get_domains_from_ip_aggregated("1.2.3.4")

        assert len(found) == 3
        assert network.client.network_ip_to_domain.call_count == 3
        assert "already sent" in network.output.warning.call_args.args[0]

    def test_an_appliance_that_never_runs_out_costs_the_page_budget_and_no_more(self, network):
        network.output = MagicMock()
        page = itertools.count()
        network.client.network_ip_to_domain.side_effect = lambda *_a, **_kw: sdk_response(
            200, {"resolutions": [{"host_name": "a.test"}], "next_page": f"page-{next(page)}"}
        )

        found = network.get_domains_from_ip_aggregated("1.2.3.4")

        assert network.client.network_ip_to_domain.call_count == _MAX_PIVOT_PAGES
        assert len(found) == _MAX_PIVOT_PAGES
        assert "holds more domains than this" in network.output.warning.call_args.args[0]

    def test_the_walk_does_not_offer_a_remedy_the_analyst_has_already_taken(self, network):
        """``--all`` is what the caller offers on ``pages_left_unfetched``.

        A budget-ended walk says what it left behind itself, because the
        caller's answer to that flag is the flag the analyst just passed.
        """
        network.output = MagicMock()
        page = itertools.count()
        network.client.network_ip_to_domain.side_effect = lambda *_a, **_kw: sdk_response(
            200, {"resolutions": [{"host_name": "a.test"}], "next_page": f"page-{next(page)}"}
        )

        network.get_domains_from_ip_aggregated("1.2.3.4")

        assert not network.pages_left_unfetched


class TestSuccessCodeTolerance:
    """The SDK raises on 4xx/5xx, so any 2xx reaching a wrapper is success."""

    def test_created_status_is_success(self, yara):
        yara.client.create_or_update_yara_ruleset.return_value = sdk_response(201)
        assert yara.create_yara_ruleset("rules", "rule x {}") is True

    def test_no_content_delete_is_success(self, metadata):
        metadata.client.delete_classification.return_value = sdk_response(204)
        assert metadata.delete_classification(SHA256) is True

    def test_accepted_status_returns_json(self, network):
        network.client.submit_url.return_value = sdk_response(202, {"task_id": "t"})
        assert network.submit_url("http://example.com") == {"task_id": "t"}

    def test_client_error_still_fails(self, yara):
        yara.client.create_or_update_yara_ruleset.return_value = sdk_response(409)
        assert yara.create_yara_ruleset("rules", "rule x {}") is False


class TestClassification:
    def test_set_classification_passes_required_system(self, metadata):
        metadata.client.set_classification.return_value = sdk_response(200)
        assert metadata.set_classification(SHA256, "malicious", "Evil") is True
        args, kwargs = metadata.client.set_classification.call_args
        assert args == (SHA256, "malicious", "local")
        assert kwargs["threat_name"] == "Evil"

    def test_set_classification_ticloud_system(self, metadata):
        metadata.client.set_classification.return_value = sdk_response(200)
        metadata.set_classification(SHA256, "goodware", None, "ticloud")
        assert metadata.client.set_classification.call_args.args[2] == "ticloud"

    def test_set_classification_rejects_unknown_system(self, metadata):
        assert metadata.set_classification(SHA256, "malicious", None, "cloud") is False
        metadata.client.set_classification.assert_not_called()


class TestRecordsPerPage:
    """Advanced Search v3 rejects page sizes outside 1-100 with WrongInputError."""

    @pytest.mark.parametrize(
        "limit,expected", [(1, 1), (50, 50), (100, 100), (200, 100), (5000, 100), (0, 1), (-5, 1)]
    )
    def test_clamped_to_valid_range(self, limit, expected):
        assert _records_per_page(limit) == expected

    def test_single_page_request_stays_within_the_sdk_maximum(self, samples):
        samples.client.advanced_search_v3.return_value = sdk_response(
            200, {"rl": {"web_search_api": {"entries": []}}}
        )
        samples.list_samples(limit=100)
        assert samples.client.advanced_search_v3.call_args.kwargs["records_per_page"] == 100


def search_pages(*pages: tuple[int, bool]) -> list[MagicMock]:
    """Successive Advanced Search v3 answers, one ``(records, more_pages)`` each.

    What a walk above one page reads: the service asks for page 1, page 2
    and so on itself, so what it gets back is a sequence rather than one
    repeated envelope.
    """
    return [
        sdk_response(
            200,
            {"rl": {"web_search_api": {"entries": [{"sha1": SHA1}] * records, "more_pages": more}}},
        )
        for records, more in pages
    ]


class TestLimitsAboveOnePage:
    """A page holds 100; a bigger --limit used to truncate in silence."""

    def test_list_samples_aggregates(self, samples):
        samples.client.advanced_search_v3.side_effect = search_pages(
            (100, True), (100, True), (50, False)
        )
        result = samples.list_samples(limit=250)

        assert len(result) == 250
        assert samples.client.advanced_search_v3.call_count == 3

    def test_aggregation_asks_for_the_largest_page(self, samples):
        """The SDK's own default is 20, so --limit 1000 cost 50 POSTs."""
        samples.client.advanced_search_v3.side_effect = search_pages((0, False))
        samples.advanced_search("available:true", limit=1000)

        kwargs = samples.client.advanced_search_v3.call_args.kwargs
        assert kwargs["records_per_page"] == 100

    def test_advanced_search_aggregates(self, samples):
        samples.client.advanced_search_v3.side_effect = search_pages(
            (100, True), (100, True), (100, True)
        )
        result = samples.advanced_search("available:true", limit=250)

        assert len(result) == 250

    def test_each_page_is_asked_for_by_number_rather_than_by_the_cursor_sent_back(self, samples):
        """The SDK posts ``page_number=next_page``, so an omitted cursor asks for ``null``.

        An appliance that sets ``more_pages`` and states no ``next_page``
        was served page 1 again by anything lenient about it, and the walk
        came back with up to ``--limit`` copies of the same records.
        """
        samples.client.advanced_search_v3.side_effect = search_pages(
            (100, True), (100, True), (100, False)
        )
        samples.advanced_search("available:true", limit=300)

        asked = [
            call.kwargs["page_number"] for call in samples.client.advanced_search_v3.mock_calls
        ]
        assert asked == [1, 2, 3]

    def test_explicit_page_keeps_single_page_behaviour_and_warns(self, samples):
        samples.output = MagicMock()
        samples.client.advanced_search_v3.return_value = sdk_response(
            200, {"rl": {"web_search_api": {"entries": []}}}
        )
        samples.advanced_search("available:true", limit=250, page=3)

        assert samples.client.advanced_search_v3.call_count == 1
        assert samples.client.advanced_search_v3.call_args.kwargs["records_per_page"] == 100
        assert samples.client.advanced_search_v3.call_args.kwargs["page_number"] == 3
        assert samples.output.warning.called

    def test_limit_within_one_page_is_unchanged(self, samples):
        samples.client.advanced_search_v3.return_value = sdk_response(
            200, {"rl": {"web_search_api": {"entries": []}}}
        )
        samples.advanced_search("available:true", limit=50)
        assert samples.client.advanced_search_v3.call_count == 1


class TestAdvancedSearchSaysWhenThereIsMore:
    """The v3 envelope states ``more_pages`` and ``next_page``; this read neither.

    The SDK's own aggregator loops on exactly those two, and reading only
    ``entries`` truncated in silence on the *default* path — ``a1000
    search`` and ``a1000 list`` ask for one page of 100 — so a query
    matching thousands answered "Found 100 samples" and exit 0 with
    nothing anywhere saying the rest existed. The ``--limit above 100``
    warning beside it never covered this: no unusual argument is needed.

    The service says what it measured and the command says what to type,
    the same split the IP lookups make: naming ``--limit`` down here
    would make the service unusable to a caller without that option.
    """

    def _answer(self, samples, **envelope):
        samples.output = MagicMock()
        samples.client.advanced_search_v3.return_value = sdk_response(
            200, {"rl": {"web_search_api": {"entries": [{"sha1": SHA1}], **envelope}}}
        )

    @pytest.mark.parametrize(
        "envelope",
        [{"more_pages": True}, {"next_page": 2}, {"more_pages": True, "next_page": 2}],
        ids=["flag", "cursor", "both"],
    )
    def test_a_further_page_is_announced(self, samples, envelope):
        self._answer(samples, **envelope)

        assert samples.advanced_search("available:true", limit=100) == [{"sha1": SHA1}]

        warned = samples.output.warning.call_args.args[0]
        assert "one page" in warned
        assert "--limit" not in warned, "the remedy is the command's word, not the service's"
        assert samples.pages_left_unfetched

    def test_a_last_page_is_announced_as_nothing(self, samples):
        self._answer(samples, more_pages=False, next_page=None)

        assert samples.advanced_search("available:true", limit=100) == [{"sha1": SHA1}]

        assert not samples.output.warning.called
        assert not samples.pages_left_unfetched

    def test_the_listing_command_reads_the_same_endpoint_and_the_same_flag(self, samples):
        """``a1000 list`` is ``advanced_search`` with the match-anything query."""
        self._answer(samples, more_pages=True)

        assert samples.list_samples(limit=100) == [{"sha1": SHA1}]
        assert samples.pages_left_unfetched

    def test_a_page_that_was_the_last_one_clears_a_previous_answer(self, samples):
        """The flag is about this call, so a second search must not inherit it."""
        self._answer(samples, more_pages=True)
        samples.advanced_search("available:true", limit=100)

        self._answer(samples, more_pages=False)
        samples.advanced_search("available:true", limit=100)

        assert not samples.pages_left_unfetched

    def test_a_walk_that_stopped_at_the_cap_left_the_rest_behind(self, samples):
        """A walk stops at ``limit`` even mid-corpus, and trims to it.

        So the path that fetches *more* was the one that said nothing:
        ``--limit 150`` over 50000 samples read as a complete answer,
        while ``--limit 100`` announced the pages it had left.
        """
        samples.output = MagicMock()
        samples.client.advanced_search_v3.side_effect = search_pages(
            (100, True), (100, True), (100, True)
        )

        assert len(samples.advanced_search("available:true", limit=250)) == 250

        assert samples.pages_left_unfetched

    def test_a_walk_that_ran_out_before_the_cap_fetched_everything(self, samples):
        samples.output = MagicMock()
        samples.client.advanced_search_v3.side_effect = search_pages((100, True), (30, False))

        assert len(samples.advanced_search("available:true", limit=250)) == 130

        assert not samples.pages_left_unfetched
        assert not samples.output.warning.called

    def test_a_corpus_of_exactly_the_cap_left_nothing_behind(self, samples):
        """Measured, not inferred: an answer the size of ``limit`` may be complete.

        Reading the size of the answer alone had to call this partial,
        because it could not tell a corpus that ran out at ``limit`` from
        one the walk cut off there.
        """
        samples.output = MagicMock()
        samples.client.advanced_search_v3.side_effect = search_pages((100, True), (100, False))

        assert len(samples.advanced_search("available:true", limit=200)) == 200

        assert not samples.pages_left_unfetched

    def test_a_walk_the_appliance_left_short_of_the_cap_says_so(self, samples):
        """An empty page is not the end, and what it leaves behind is stated.

        The appliance says ``more_pages`` past a page it filtered down to
        nothing, so the walk reads on and stops where the corpus does; the
        page budget is what stops it against an appliance that never runs
        out.
        """
        samples.output = MagicMock()
        samples.client.advanced_search_v3.side_effect = search_pages(
            (100, True), (0, True), (100, True)
        )

        assert len(samples.advanced_search("available:true", limit=200)) == 200

        assert samples.client.advanced_search_v3.call_count == 3
        assert samples.pages_left_unfetched

    def test_a_page_the_walk_could_not_read_is_a_failed_search(self, samples):
        """Not the end of the corpus: the records already collected are dropped."""
        samples.output = MagicMock()
        samples.client.advanced_search_v3.side_effect = [
            *search_pages((100, True)),
            sdk_response(200, {"rl": "nonsense"}),
        ]

        assert samples.advanced_search("available:true", limit=1000) is None

        assert "carried no results envelope" in samples.output.error.call_args.args[0]

    def test_an_aggregated_walk_clears_a_previous_answer(self, samples):
        """The flag is about this call, whichever path measured it."""
        self._answer(samples, more_pages=True)
        samples.advanced_search("available:true", limit=100)

        samples.client.advanced_search_v3.side_effect = search_pages((10, False))
        samples.advanced_search("available:true", limit=250)

        assert not samples.pages_left_unfetched

    def test_an_unreadable_envelope_is_still_a_failed_search(self, samples):
        samples.output = MagicMock()
        samples.client.advanced_search_v3.return_value = sdk_response(200, {"rl": "nonsense"})

        assert samples.advanced_search("available:true", limit=100) is None

        assert "carried no results envelope" in samples.output.error.call_args.args[0]
        assert not samples.pages_left_unfetched

    def test_the_analyst_is_told_how_to_fetch_the_rest(self, a1000_cli):
        _, result = a1000_cli(
            ["search", "-q", "available:true"],
            "advanced_search_v3",
            {"rl": {"web_search_api": {"entries": [{"sha256": SHA256}], "more_pages": True}}},
        )

        assert result.exit_code == 0, result.output
        assert "--limit" in flat(result)

    def test_the_listing_command_offers_it_too(self, a1000_cli):
        _, result = a1000_cli(
            ["list"],
            "advanced_search_v3",
            {"rl": {"web_search_api": {"entries": [{"sha256": SHA256}], "more_pages": True}}},
        )

        assert result.exit_code == 0, result.output
        assert "--limit" in flat(result)

    def test_a_complete_answer_offers_nothing(self, a1000_cli):
        _, result = a1000_cli(
            ["search", "-q", "available:true"],
            "advanced_search_v3",
            {"rl": {"web_search_api": {"entries": [{"sha256": SHA256}]}}},
        )

        assert result.exit_code == 0, result.output
        assert "--limit" not in flat(result)


class _CountingSearchEndpoint:
    """An Advanced Search v3 endpoint that answers a page and counts the asking.

    Every page it hands out is a POST to the appliance, which is the whole
    subject of the tests below: what bounds them is not the corpus and not
    what the analyst typed, but the walk's own budget.

    It carries the SDK's real ``advanced_search_v3_aggregated`` bound to
    itself, so a walk delegated to the SDK is measured by the same double
    as a walk this service does itself — the claim is about the walk,
    whoever runs it.
    """

    # A stand-in that let the walk run forever would hang the suite rather
    # than fail it. Well above any budget a walk may legitimately have.
    _RUNAWAY = 3000

    advanced_search_v3_aggregated = A1000.advanced_search_v3_aggregated

    def __init__(self, page: dict[str, Any] | list[dict[str, Any]]):
        """One page answered to every request, or a page per request in turn."""
        self._page = page
        self.calls = 0

    def advanced_search_v3(self, **_kwargs: Any) -> MagicMock:
        self.calls += 1
        if self.calls > self._RUNAWAY:
            raise AssertionError(f"the walk made {self.calls} requests and was still going")
        page = self._page[self.calls - 1] if isinstance(self._page, list) else self._page
        return sdk_response(200, {"rl": {"web_search_api": page}})


class TestAnA1000WalkThatStopsMakingProgressStops:
    """The walk had no stop of its own, and no bound on what it could spend.

    Delegated to the SDK's ``advanced_search_v3_aggregated``, this had the
    shape the TitaniumCloud walk was rewritten to escape: ``while
    more_pages:`` … ``if not more_pages or len(results) >= max_results``.
    Against an endpoint answering ``{"entries": [], "more_pages": true}``
    — a well-formed page promising another — neither condition can ever
    come true, because nothing is ever added to ``results``. The probe was
    still looping after 3001 requests.

    ``MAX_LIMIT`` bounds the honest case and not this one: it says how many
    records may be asked for, and a page that carries none costs a request
    all the same.
    """

    def _endpoint(self, samples, page) -> _CountingSearchEndpoint:
        api = _CountingSearchEndpoint(page)
        samples.session.client = api
        samples.output = MagicMock()
        return api

    def test_an_endpoint_that_promises_pages_and_sends_none_does_not_loop(self, samples):
        api = self._endpoint(samples, {"entries": [], "more_pages": True})

        results = samples.advanced_search("available:true", limit=MAX_LIMIT)

        assert api.calls <= _MAX_SEARCH_PAGES, "the walk did not stop making requests"
        assert results == []

    def test_a_limit_past_the_corpus_costs_the_page_budget_and_no_more(self, samples):
        """A mistyped ``--limit`` must not buy requests without end."""
        page = {"entries": [{"n": 1}] * _MAX_RECORDS_PER_PAGE, "more_pages": True}
        api = self._endpoint(samples, page)

        results = samples.advanced_search("available:true", limit=1_000_000)

        assert api.calls == _MAX_SEARCH_PAGES
        assert len(results) == _MAX_SEARCH_PAGES * _MAX_RECORDS_PER_PAGE
        assert samples.pages_left_unfetched

    def test_a_sparse_page_mid_corpus_does_not_truncate_the_answer(self, samples):
        """A page that promised more and carried nothing is not the end.

        Only ``more_pages`` says where the corpus ends. Stopping on an
        empty page hands back the records fetched so far as though they
        were all of them, and ``--limit`` is already what the analyst
        typed, so nothing offers to fetch the rest.
        """
        api = self._endpoint(
            samples,
            [
                {"entries": [{"n": 1}], "more_pages": True},
                {"entries": [], "more_pages": True},
                {"entries": [{"n": 3}], "more_pages": False},
            ],
        )

        results = samples.advanced_search("available:true", limit=1_000_000)

        assert results == [{"n": 1}, {"n": 3}]
        assert api.calls == 3
        assert not samples.pages_left_unfetched

    def test_a_corpus_that_runs_out_first_stops_the_walk_there(self, samples):
        api = self._endpoint(samples, {"entries": [{"n": 1}], "more_pages": False})

        results = samples.advanced_search("available:true", limit=1_000_000)

        assert api.calls == 1
        assert results == [{"n": 1}]
        assert not samples.pages_left_unfetched

    def test_the_page_budget_reaches_the_largest_limit_the_cli_accepts(self):
        """No ``--limit`` click accepts may be capped a second time inside.

        The twin of the claim ``tests/test_cli_ticloud.py`` makes for the
        other Advanced Search: a ceiling the CLI publishes and the service
        silently undercuts is a truncation nothing announces.
        """
        assert MAX_LIMIT <= _MAX_SEARCH_PAGES * _MAX_RECORDS_PER_PAGE

    def test_the_pivot_budget_reaches_the_same_ceiling(self):
        """The IP walks claimed the same product and nothing pinned it.

        A pivot takes no ``--limit``, so what the budget has to reach is
        the largest listing this CLI will fetch at all — and it reaches it
        only at the page size the walk now sends rather than inherits.
        """
        assert MAX_LIMIT <= _MAX_PIVOT_PAGES * _RECORDS_PER_PAGE


class TestWhatOneAggregatedWalkMaySpend:
    """``--max-results`` was bounded from below and not from above.

    It is the same question ``--limit`` asks — how many records may this
    walk collect — and every walk it caps spends a metered request per
    page: ``--max-results 100000000`` bought a thousand of them off
    ``yara-repo-list``, whose page holds 100, and a hundred thousand off a
    ticloud pivot, from a command line that reads like a typo of 100000.

    A ceiling is all it is. None of the SDK pagers it reaches can stop on
    an endpoint that answers an empty page and another cursor, because
    ``len(results) >= max_results`` is what they stop on; that is a page
    budget's job, and it belongs in the service.
    """

    @pytest.mark.parametrize("budget", [str(MAX_LIMIT + 1), "100000000"])
    def test_a_budget_past_the_ceiling_is_refused_before_anything_is_fetched(
        self, a1000_cli, budget
    ):
        client, result = a1000_cli(["yara-repo-list", "--all", "--max-results", budget])

        assert result.exit_code == 2
        assert str(MAX_LIMIT) in flat(result), "the analyst is told the range they may type"
        client.get_yara_repositories_aggregated.assert_not_called()

    def test_the_ceiling_itself_is_a_budget_the_command_takes(self, a1000_cli):
        _, result = a1000_cli(["yara-repo-list", "--all", "--max-results", str(MAX_LIMIT)])

        assert result.exit_code != 2, result.output


class TestSearchSaysWhichQueryItRan:
    """The resolved query lived only in the spinner, which is erased.

    ``--malicious --clean`` posts ``classification:malicious`` and drops
    ``--clean`` without a word, so a script setting both from variables,
    or an analyst adding ``--clean`` to a saved ``-q`` command line, read
    malicious samples under a bare "Found N samples".
    """

    def _search(self, a1000_cli, args):
        page = {"rl": {"web_search_api": {"entries": [{"sha256": SHA256}]}}}
        client, result = a1000_cli(["search", *args], "advanced_search_v3", page)
        assert result.exit_code == 0, result.output
        return client.advanced_search_v3.call_args.kwargs["query_string"], flat(result)

    @pytest.mark.parametrize(
        "args,posted,dropped",
        [
            (["--malicious", "--clean"], "classification:malicious", "--clean"),
            (["-q", "riskscore:1", "--clean"], "riskscore:1", "--clean"),
        ],
        ids=["both-flags", "query-and-flag"],
    )
    def test_an_ignored_input_is_named(self, a1000_cli, args, posted, dropped):
        query, output = self._search(a1000_cli, args)

        assert query == posted
        assert f"Ignoring {dropped}" in output
        assert posted in output

    @pytest.mark.parametrize("args", [["--clean"], ["--malicious"], ["-q", "riskscore:1"]])
    def test_an_input_that_was_used_is_not_reported_as_dropped(self, a1000_cli, args):
        assert "Ignoring" not in self._search(a1000_cli, args)[1]

    def test_the_success_line_names_the_query_it_answered(self, a1000_cli):
        query, output = self._search(a1000_cli, ["--clean"])

        assert query == "classification:clean"
        assert f"Found 1 samples matching {query}" in output


class TestUploadAndGetReport:
    """The workflow spans two areas, so it is a function over both services."""

    def _sample(self, samples, tmp_path: Path, upload_payload) -> Path:
        sample = tmp_path / "sample.bin"
        sample.write_bytes(b"x")
        samples.client.submit_file_from_path.return_value = sdk_response(200, upload_payload)
        return sample

    def test_upload_without_a_usable_hash_is_a_failure(self, samples, reports, tmp_path):
        """It used to answer the upload receipt: "✓ File analyzed" and exit 1."""
        sample = self._sample(samples, tmp_path, {"code": 201, "detail": {"id": 1}})
        samples.output = MagicMock()

        assert upload_and_get_report(samples, reports, sample, "summary") is None
        assert samples.output.error.called

    def test_summary_report_is_fetched_for_the_uploaded_hash(self, samples, reports, tmp_path):
        sample = self._sample(samples, tmp_path, {"detail": {"sha1": SHA1}})
        reports.client.get_summary_report_v2.return_value = sdk_response(200, {"sha1": SHA1})

        assert upload_and_get_report(samples, reports, sample, "summary") == {"sha1": SHA1}
        reports.client.get_summary_report_v2.assert_called_once_with(SHA1)

    def test_titanium_report_is_fetched_for_the_uploaded_hash(self, samples, reports, tmp_path):
        sample = self._sample(samples, tmp_path, {"detail": {"sha1": SHA1}})
        reports.client.get_titanium_core_report_v2.return_value = sdk_response(200, {"ticore": {}})

        assert upload_and_get_report(samples, reports, sample, "titanium") == {"ticore": {}}
        reports.client.get_titanium_core_report_v2.assert_called_once_with(SHA1)

    def test_a_report_format_spelling_is_rejected_before_the_upload(
        self, samples, reports, tmp_path
    ):
        """A neighbouring method accepts "pdf"; here it used to mean "summary"."""
        sample = self._sample(samples, tmp_path, {"detail": {"sha1": SHA1}})
        samples.output = MagicMock()

        assert upload_and_get_report(samples, reports, sample, "pdf") is None
        assert samples.output.error.called
        samples.client.submit_file_from_path.assert_not_called()
        reports.client.get_summary_report_v2.assert_not_called()

    def test_it_uploads_through_the_service_it_was_given(self, session, reports, tmp_path):
        """A caller's own sample service must do the uploading, not a new one."""

        class CustomUpload(A1000SampleService):
            def upload_file(self, file_path: Path, comment: str | None = None) -> dict[str, Any]:
                return {"task_id": SHA1}

        samples = session.service(CustomUpload)
        reports.client.get_summary_report_v2.return_value = sdk_response(200, {"sha1": SHA1})

        report = upload_and_get_report(samples, reports, tmp_path / "sample.bin", "summary")

        assert report == {"sha1": SHA1}
        samples.client.submit_file_from_path.assert_not_called()


class TestServicesBuiltOnTheirOwn:
    """A service must hold no second service — that was a second connection.

    The composed facade used to be what every test built, so a focused
    service quietly constructing a neighbour went unnoticed. These build
    them the way the CLI does.
    """

    def _settings(self, tmp_path) -> Settings:
        return Settings(cache_dir=tmp_path / "cache", config_dir=tmp_path / "config")

    def test_a_bare_report_service_builds_no_second_service(self, tmp_path):
        session = A1000Session(self._settings(tmp_path))
        service = session.service(A1000ReportService)

        assert not [held for held in vars(service).values() if isinstance(held, BaseService)]

        session.client = MagicMock()
        service.client.get_summary_report_v2.return_value = sdk_response(200, {"sha1": SHA1})
        assert service.get_summary_report_v2(SHA1) == {"sha1": SHA1}

    def test_a_service_cannot_swap_the_connection_out_from_under_its_siblings(self, tmp_path):
        """The client is the session's; a service reads it and nothing more.

        The setter existed only for tests, and what it offered production
        was a way to give one service a client its siblings on the same
        session would never see.
        """
        session = A1000Session(self._settings(tmp_path))
        # Deliberately widened: the claim under test is that the assignment
        # below is refused at runtime, which mypy would otherwise report as
        # the error it is -- from inside the ``pytest.raises`` that asserts it.
        reports: Any = session.service(A1000ReportService)

        with pytest.raises(AttributeError):
            reports.client = MagicMock()


class TestModuleHelpers:
    def test_extract_hash_prefers_sha256(self):
        upload = {"rl": {"sample": {"sha256": SHA256, "sha1": SHA1}}}
        assert _extract_hash_from_upload(upload) == SHA256

    def test_extract_hash_falls_back_to_sha1(self):
        assert _extract_hash_from_upload({"rl": {"sample": {"sha1": SHA1}}}) == SHA1

    def test_extract_hash_from_the_a1000_submit_response(self):
        """What the appliance actually answers a submission with."""
        upload = {
            "code": 201,
            "message": "Done.",
            "detail": {"id": 25261421, "sha1": SHA1, "filename": "eicar.com"},
            "warnings": [],
        }
        assert _extract_hash_from_upload(upload) == SHA1

    def test_extract_hash_handles_garbage(self):
        assert _extract_hash_from_upload("not a dict") is None
        assert _extract_hash_from_upload({}) is None
        assert _extract_hash_from_upload({"detail": {"id": 1}}) is None

    @pytest.mark.parametrize("rl", [["sample"], "sample", 7], ids=["list", "string", "number"])
    def test_an_rl_section_that_is_not_a_mapping_reads_as_nothing(self, rl):
        """``(x or {}).get`` raised AttributeError on a truthy non-mapping.

        ``with_client`` swallowed it, so an upload the appliance had
        accepted was reported to the analyst as a failed one — over a
        section this repo reads through ``shapes.mapping`` everywhere else.
        """
        assert _extract_hash_from_upload({"rl": rl, "detail": {"sha1": SHA1}}) == SHA1


class TestWaitForReport:
    """PDF / dynamic-analysis status endpoints answer 200 while still building."""

    def _status(self, body: Any) -> MagicMock:
        return sdk_response(200, body)

    def test_returns_true_once_status_reaches_ready(self, reports, monkeypatch) -> None:
        monkeypatch.setattr(http_module.time, "sleep", lambda _: None)
        replies = iter(
            [
                self._status({"status": 0, "status_message": "PDF is being created."}),
                self._status({"status": 0, "status_message": "PDF is being created."}),
                self._status({"status": 2, "status_message": "PDF is ready for download."}),
            ]
        )
        assert reports._wait_for_report(lambda: next(replies), interval=0) is True

    def test_gives_up_on_an_unrecognised_status_instead_of_polling_out(
        self, reports, monkeypatch
    ) -> None:
        """A body we cannot read must not block for the full timeout."""
        slept: list[Any] = []
        monkeypatch.setattr(http_module.time, "sleep", slept.append)

        assert reports._wait_for_report(lambda: self._status({"unexpected": "shape"})) is False
        assert slept == []

    def test_reports_the_servers_message_on_failure(self, reports, monkeypatch) -> None:
        monkeypatch.setattr(http_module.time, "sleep", lambda _: None)
        reports.output = MagicMock()

        assert (
            reports._wait_for_report(
                lambda: self._status({"status": 5, "message": "HTML does not exist."})
            )
            is False
        )
        assert "HTML does not exist." in reports.output.error.call_args[0][0]

    def test_running_out_of_time_is_a_warning_and_not_a_ready_report(self, reports) -> None:
        reports.output = MagicMock()

        assert reports._wait_for_report(lambda: self._status({"status": 0}), timeout=0) is False

        assert "not ready within 0 seconds" in reports.output.warning.call_args.args[0]

    def test_a_refused_status_check_reports_what_the_appliance_said(self, reports) -> None:
        """The wait used to end on a bare ``False``, saying nothing at all."""
        reports.output = MagicMock()
        reports.client.create_pdf_report.return_value = sdk_response(200)
        reports.client.check_pdf_report_creation.return_value = sdk_response(
            507, {"message": "No room to build the report"}
        )

        assert reports.get_report(SHA256, report_format="pdf") is None

        assert "No room to build the report" in reports.output.error.call_args.args[0]
        reports.client.download_pdf_report.assert_not_called()


class TestUrlValidation:
    """URL commands validated nothing, so a typo cost a round trip."""

    @pytest.mark.parametrize("bad_url", ["not a url", "///bad", ""])
    def test_nonsense_never_reaches_the_appliance(self, network, bad_url: str) -> None:
        assert network.submit_url(bad_url) is None
        assert network.get_network_url_report(bad_url) is None
        network.client.submit_url.assert_not_called()
        network.client.network_url_report.assert_not_called()

    def test_submitting_a_bare_host_is_refused(self, network) -> None:
        """The appliance answers submit_url("example.com") with Bad request."""
        assert network.submit_url("example.com") is None
        network.client.submit_url.assert_not_called()

    def test_looking_up_a_bare_host_is_allowed(self, network) -> None:
        """network_url_report answers 200 for a hostname, so do not block it."""
        network.client.network_url_report.return_value = sdk_response(200, {"ok": True})
        assert network.get_network_url_report("example.com") == {"ok": True}
        network.client.network_url_report.assert_called_once()

    def test_valid_url_is_forwarded(self, network) -> None:
        network.client.submit_url.return_value = sdk_response(200, {"ok": True})
        assert network.submit_url("https://example.com") == {"ok": True}
        network.client.submit_url.assert_called_once()


class TestExtractedFilePermissions:
    """The archive was written 0600 but extractall recreated members under the umask."""

    def _archive_response(self, names: dict[str, str]) -> MagicMock:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            for name, content in names.items():
                zf.writestr(name, content)
        return sdk_response(200, content=buffer.getvalue())

    def test_every_extracted_member_is_owner_only(self, samples, tmp_path):
        samples.client.download_extracted_files.return_value = self._archive_response(
            {"infected/alpha.bin": "MZ", "infected/nested/beta.bin": "MZ"}
        )
        assert samples.download_extracted_files(SHA256, tmp_path) is True

        written = sorted(p for p in tmp_path.rglob("*") if p.is_file())
        assert len(written) == 2
        for path in written:
            assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_traversal_member_is_refused(self, samples, tmp_path):
        """Archive members come from the appliance, not from us.

        Refusing a member used to raise past ``zip_path.unlink()``, so the
        downloaded archive — every extracted file, live malware — was left
        in --output-dir beside the members unpacked before the refusal,
        under a bare "Failed to download extracted files".
        """
        samples.client.download_extracted_files.return_value = self._archive_response(
            {"ok.bin": "MZ", "../escaped.bin": "MZ"}
        )
        assert samples.download_extracted_files(SHA256, tmp_path) is False
        assert not (tmp_path.parent / "escaped.bin").exists()
        assert not (tmp_path / "extracted_files.zip").exists()
        assert not (tmp_path / "ok.bin").exists()


class TestBatchDeleteUsesTheBulkEndpoint:
    """The SDK picks the endpoint from the argument type."""

    def test_hashes_are_passed_as_a_list(self, samples):
        """A string would take the single-sample DELETE, one call per hash."""
        samples.client.delete_samples.return_value = sdk_response(200, {})
        samples.batch_delete_samples([SHA256, SHA1])

        assert samples.client.delete_samples.call_count == 1
        assert samples.client.delete_samples.call_args.args[0] == [SHA256, SHA1]

    def test_asynchronous_task_is_polled_to_completion(self, samples, monkeypatch):
        monkeypatch.setattr(http_module.time, "sleep", lambda _: None)
        samples.client.delete_samples.return_value = sdk_response(200, {"id": "task-1"})
        samples.client.check_sample_removal_status_v2.side_effect = [
            sdk_response(202),
            sdk_response(202),
            sdk_response(200),
        ]
        assert samples.batch_delete_samples([SHA256]) == 1
        assert samples.client.check_sample_removal_status_v2.call_count == 3

    def test_a_rejected_submission_is_not_a_count(self, samples, capsys):
        """A refusal means nothing was asked for, which is not "nothing was taken"."""
        samples.client.delete_samples.return_value = sdk_response(
            403, {"message": "Bulk removal is not permitted"}
        )

        assert samples.batch_delete_samples([SHA256]) is None
        assert "not permitted" in capsys.readouterr().err

    def test_a_batch_the_endpoint_cannot_take_is_not_a_count_either(self, samples, capsys):
        """The guard already said why; a zero on top of it read as an answer."""
        assert samples.batch_delete_samples([SHA256, "not-a-hash"]) is None

        samples.client.delete_samples.assert_not_called()
        assert "entry 2 of 2" in capsys.readouterr().err


class TestOverwritingAnExistingSamplePath:
    """O_CREAT leaves an existing file's mode alone; the sample is still malware."""

    def test_download_tightens_a_preexisting_world_readable_file(self, tmp_path):
        target = tmp_path / "sample.bin"
        target.write_bytes(b"stale")
        target.chmod(0o644)

        write_private_bytes(target, b"live-malware")

        assert target.read_bytes() == b"live-malware"
        assert target.stat().st_mode & 0o777 == 0o600


class TestTheTemporaryFileIsThisCallsOwn:
    """A fixed ``<name>.part`` was a file anyone could get there first."""

    def test_a_hard_link_planted_at_the_temporary_is_not_written_through(self, tmp_path):
        """O_NOFOLLOW refuses a symlink and says nothing about a hard link.

        With the name predictable, anyone able to create files in
        ``--output-dir`` - /tmp, a team share - pre-created it as a hard
        link to a file the analyst owns, and the download replaced that
        file's contents and set it 0600.
        """
        victim = tmp_path / "victim.txt"
        victim.write_text("payroll")
        victim.chmod(0o644)
        download_dir = tmp_path / "downloads"
        download_dir.mkdir()
        destination = download_dir / "abc123.malware"
        os.link(victim, destination.with_name(destination.name + ".part"))

        write_private_bytes(destination, b"attacker-supplied")

        assert victim.read_text() == "payroll", "the write went through the planted link"
        assert victim.stat().st_mode & 0o777 == 0o644, "the victim file was chmodded 0600"
        assert destination.read_bytes() == b"attacker-supplied"

    def test_a_file_already_at_the_temporary_is_never_written_through(self, tmp_path, monkeypatch):
        """``O_EXCL`` is the half of this guard the unpredictable name hid.

        The sibling test above plants its hard link at the fixed
        ``<name>.part`` the writer stopped using, so it passes on the name
        alone: removing ``O_EXCL`` from the open flags left the whole
        suite green, and with it a pre-existing file at the temporary path
        would be opened, written and ``fchmod``-ed to 0600.
        ``O_NOFOLLOW`` cannot cover that - measured on this platform, it
        refuses a symlink and opens a hard link straight through, while
        ``O_CREAT|O_EXCL`` refuses both. Pinning the random half of the
        name puts the plant where the writer will actually open, so the
        flag is the only thing left to refuse it.
        """
        monkeypatch.setattr("rl_cli.storage.files.secrets.token_hex", lambda _: "0badc0de")
        victim = tmp_path / "victim.txt"
        victim.write_text("payroll")
        victim.chmod(0o644)
        destination = tmp_path / "abc123.malware"
        os.link(victim, tmp_path / f"abc123.malware.{os.getpid()}.0badc0de.part")

        with pytest.raises(FileExistsError):
            write_private_bytes(destination, b"attacker-supplied")

        assert victim.read_text() == "payroll", "the write went through the planted file"
        assert victim.stat().st_mode & 0o777 == 0o644, "the victim file was chmodded 0600"
        assert not destination.exists()

    def test_the_temporary_is_owner_only_from_the_moment_it_exists(self, tmp_path, monkeypatch):
        """0600 is the creation mode, not something the next line repairs.

        ``rl_cli.storage.files`` promises the file is owner-only *from the
        moment it exists*, and every existing assertion looks at the
        finished one - which the ``fchmod`` makes 0600 whatever
        ``os.open`` was handed. So creating it 0644 and tightening it a
        line later left the suite green while opening a window in which
        any local process could read the live malware. The mode reaches
        the kernel once, at creation, so that is where it is asserted.
        """
        modes: list[int] = []
        real_open = os.open

        def recording_open(path, flags, mode=0o777, **kwargs):
            if str(path).startswith(str(tmp_path)):
                modes.append(mode)
            return real_open(path, flags, mode, **kwargs)

        # ``rl_cli.storage.files`` does ``import os``, so its ``os`` is this one:
        # patching the canonical module is the same seam, named where it lives.
        monkeypatch.setattr(os, "open", recording_open)

        write_private_bytes(tmp_path / "sample.malware", b"MZ")

        assert modes == [0o600]

    def test_two_writers_to_one_destination_do_not_share_a_temporary(self, tmp_path):
        """The bug ``download_extracted_files`` grew a staging directory for.

        Sharing one ``.part`` published the first writer's bytes into the
        second writer's file and then killed the first with
        FileNotFoundError, after its data was already elsewhere.
        """
        destination = tmp_path / "sample.bin"

        with private_writer(destination) as first, private_writer(destination) as second:
            first.write(b"first")
            second.write(b"second")

        assert destination.read_bytes() == b"first", "the last writer in did not win"

    def test_the_temporary_is_gone_when_the_block_is_interrupted(self, tmp_path):
        destination = tmp_path / "sample.bin"

        with pytest.raises(KeyboardInterrupt), private_writer(destination) as handle:
            handle.write(b"half")
            raise KeyboardInterrupt

        assert not destination.exists()
        assert not list(tmp_path.glob("sample.bin*")), "a temporary outlived the interrupt"


class TestPromotedDirectories:
    """The files are 0600; the carved-out path names were 0755."""

    def test_the_unpacked_tree_lands_owner_only(self, samples, tmp_path):
        samples.client.download_extracted_files.return_value = sdk_response(
            200, content=_zip_bytes({"a/b/payload.bin": b"MZ"})
        )

        assert samples.download_extracted_files(SHA256, tmp_path) is True

        assert stat.S_IMODE((tmp_path / "a").stat().st_mode) == 0o700
        assert stat.S_IMODE((tmp_path / "a" / "b").stat().st_mode) == 0o700

    def test_replacing_an_existing_extracted_file_is_announced(self, samples, tmp_path, capsys):
        (tmp_path / "payload.bin").write_bytes(b"analyst-copy")
        samples.client.download_extracted_files.return_value = sdk_response(
            200, content=_zip_bytes({"payload.bin": b"MZ"})
        )

        assert samples.download_extracted_files(SHA256, tmp_path) is True

        warned = capsys.readouterr().err
        assert "payload.bin" in warned
        assert "already exists and will be" in warned


class TestUploadAndAnalyzeCarriesTheComment:
    """-c was echoed back to the user and then dropped on the floor."""

    def test_the_comment_reaches_the_upload(self, tmp_path):
        sample = tmp_path / "s.bin"
        sample.write_bytes(b"x")
        seen: dict[str, object] = {}

        class StubSamples(A1000SampleService):
            def upload_file(self, file_path, comment=None):
                seen["comment"] = comment
                return {"task_id": "a" * 40}

        class StubReports(A1000ReportService):
            def get_summary_report_v2(self, hash_value):
                return {"ok": True}

        session = A1000Session(Settings())
        session.client = object()
        upload_and_get_report(
            session.service(StubSamples),
            session.service(StubReports),
            sample,
            report_type="summary",
            comment="triage-42",
        )

        assert seen["comment"] == "triage-42"


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    """A stored (uncompressed) archive, so declared ratios stay 1:1 until forged."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _forged_archive(members: dict[str, bytes], **declared: int) -> zipfile.ZipFile:
    """An archive whose central directory claims ``declared`` about every member.

    The claim is what a bomb is refused on, and refusing on it is what
    makes the refusal free: a 337 KB archive can say it holds 300 MB
    without a test ever writing 300 MB.
    """
    archive = zipfile.ZipFile(io.BytesIO(_zip_bytes(members)))
    for info in archive.infolist():
        for attribute, value in declared.items():
            setattr(info, attribute, value)
    return archive


class TestExtractionRefusesZipBombs:
    """A 337 KB archive expanded to 300 MB at 932:1 with no complaint."""

    def test_an_archive_declaring_more_than_the_total_cap_is_refused(self, tmp_path):
        archive = _forged_archive({"big.bin": b"MZ"}, file_size=_MAX_ARCHIVE_BYTES + 1)
        with pytest.raises(ValueError, match="uncompressed bytes"):
            extract_private(archive, tmp_path / "out")
        assert not (tmp_path / "out").exists()

    def test_more_members_than_the_cap_is_refused(self, tmp_path):
        members = {f"{index}.bin": b"" for index in range(_MAX_ARCHIVE_MEMBERS + 1)}
        archive = zipfile.ZipFile(io.BytesIO(_zip_bytes(members)))
        with pytest.raises(ValueError, match="member limit"):
            extract_private(archive, tmp_path / "out")
        assert not (tmp_path / "out").exists()

    def test_a_member_over_the_ratio_cap_is_refused(self, tmp_path):
        """The reproduced bomb, priced at two bytes: 300 MB out of 337 KB."""
        archive = _forged_archive(
            {"bomb.bin": b"MZ"}, file_size=300 * 1024**2, compress_size=337 * 1024
        )
        with pytest.raises(ValueError, match=f"{_MAX_MEMBER_RATIO}:1 compression-ratio limit"):
            extract_private(archive, tmp_path / "out")

    def test_the_members_before_the_bomb_are_not_written_either(self, tmp_path):
        """The whole central directory is read before anything is decompressed."""
        archive = zipfile.ZipFile(io.BytesIO(_zip_bytes({"ok.bin": b"MZ", "bomb.bin": b"MZ"})))
        bomb = archive.infolist()[1]
        bomb.file_size, bomb.compress_size = 300 * 1024**2, 337 * 1024
        out = tmp_path / "out"

        with pytest.raises(ValueError):
            extract_private(archive, out)

        assert not out.exists()

    def test_an_ordinary_archive_still_extracts(self, tmp_path):
        archive = zipfile.ZipFile(io.BytesIO(_zip_bytes({"a/alpha.bin": b"MZ" * 5000})))
        out = tmp_path / "out"
        extract_private(archive, out)
        assert (out / "a" / "alpha.bin").read_bytes() == b"MZ" * 5000

    def test_a_member_both_a_file_and_a_directory_is_refused_before_anything_is_written(
        self, tmp_path
    ):
        """ "payload" as a file, beside "payload/stage2.dll" as a member.

        The ``mkdir`` for the second raised ``FileExistsError`` out of the
        middle of the write, after the members before it were already on
        disk - live malware unpacked out of an archive the CLI then
        reported as refused. The clash is why the plan is built whole
        before a byte is written, and removing the check left the suite
        green: nothing in it fed the extractor a pair that collides.
        """
        archive = zipfile.ZipFile(
            io.BytesIO(_zip_bytes({"payload": b"MZ", "payload/stage2.dll": b"MZ"}))
        )
        out = tmp_path / "out"

        with pytest.raises(ValueError, match="is both a file and a directory member"):
            extract_private(archive, out)

        assert list(out.iterdir()) == [], "a member was unpacked before the clash was named"


class TestTheDeclaredSizesAreNotTakenOnTrust:
    """The central directory is attacker-controlled, so it may also lie low.

    The three caps in ``refuse_bomb`` all judge what the archive *claims*,
    and every test of them forges a claim that is too big. The two guards
    that catch a claim which is too small - the per-member ceiling in
    ``copy_member`` and the running total in ``_write_plan`` - are what
    stand between us and an archive that declares two bytes and delivers
    two gigabytes, and neither had a test: both could be lifted a
    hundredfold with the whole suite green.
    """

    def test_a_member_one_byte_longer_than_it_declared_is_refused(self, tmp_path):
        """One byte over is the whole margin, so no looser ceiling passes.

        ``zipfile`` bounds its own ``read`` at ``file_size``, which is why
        this is exercised against the function rather than through the
        extractor: the guard is the second line, for a source that does
        not bound itself.
        """
        member = zipfile.ZipInfo("bomb.bin")
        member.file_size = 10

        with pytest.raises(ValueError, match=r"bomb\.bin decompressed past the 10 bytes"):
            copy_member(io.BytesIO(b"A" * 11), io.BytesIO(), member)

    def test_a_member_of_exactly_its_declared_size_is_copied_whole(self, tmp_path):
        """The other side of the same byte: the ceiling must not cut a member
        that kept its word."""
        member = zipfile.ZipInfo("ok.bin")
        member.file_size = 4096
        destination = io.BytesIO()

        assert copy_member(io.BytesIO(b"A" * 4096), destination, member) == 4096
        assert destination.getvalue() == b"A" * 4096

    def test_the_running_total_is_what_was_written_not_what_was_promised(
        self, tmp_path, monkeypatch
    ):
        """A member that writes more than the whole archive declared still
        stops the extraction, which is the only thing standing between a
        lying central directory and the analyst's disk. The lie is the
        stub: ``refuse_bomb`` has already passed on the two bytes this
        archive claims."""
        monkeypatch.setattr(archives_module, "copy_member", lambda *_: _MAX_ARCHIVE_BYTES + 1)
        archive = zipfile.ZipFile(io.BytesIO(_zip_bytes({"a.bin": b"MZ"})))

        with pytest.raises(ValueError, match="decompressed past the"):
            extract_private(archive, tmp_path / "out")


class _ChunkRecorder:
    """A write handle that remembers how much it was handed at a time."""

    def __init__(self, handle: Any, sizes: list[int]):
        self._handle = handle
        self._sizes = sizes

    def write(self, chunk: bytes) -> int:
        self._sizes.append(len(chunk))
        return int(self._handle.write(chunk))


class TestMembersAreCopiedInBoundedChunks:
    """``source.read()`` put a whole member in RAM: 1.0x of member size."""

    def test_no_single_write_holds_the_whole_member(self, tmp_path, monkeypatch):
        sizes: list[int] = []

        # ``private_writer`` is patched on ``archives`` because that is the
        # name ``extract_private`` calls, and read from ``storage.files``
        # because that is where it is defined -- the same object either way.
        @contextmanager
        def recording(path: Path, *, binary: Literal[True] = True) -> Iterator[_ChunkRecorder]:
            with private_writer(path, binary=binary) as handle:
                yield _ChunkRecorder(handle, sizes)

        monkeypatch.setattr(archives_module, "private_writer", recording)

        member = bytes(3 * 1024 * 1024)
        archive = zipfile.ZipFile(io.BytesIO(_zip_bytes({"big.bin": member})))
        out = tmp_path / "out"

        extract_private(archive, out)

        assert (out / "big.bin").read_bytes() == member
        assert len(sizes) > 1, "the member was handed over in one piece"
        assert max(sizes) <= archives_module._COPY_CHUNK_BYTES


class TestAFailedExtractionLeavesNothingUnpacked:
    """The archive layout decides when the analyst is told nothing was written."""

    def test_a_file_another_member_needs_as_a_directory_is_refused_up_front(
        self, samples, tmp_path
    ):
        out = tmp_path / "out"
        out.mkdir()
        samples.client.download_extracted_files.return_value = sdk_response(
            200,
            content=_zip_bytes(
                {"dropper.exe": b"MZ", "payload": b"MZ", "payload/stage2.dll": b"MZ"}
            ),
        )

        assert samples.download_extracted_files(SHA256, out) is False
        assert list(out.iterdir()) == [], "live malware survived a reported failure"

    def test_a_traversal_member_leaves_no_staging_directory(self, samples, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        samples.client.download_extracted_files.return_value = sdk_response(
            200, content=_zip_bytes({"ok.bin": b"MZ", "../escaped.bin": b"MZ"})
        )

        assert samples.download_extracted_files(SHA256, out) is False
        assert list(out.iterdir()) == []


class TestConcurrentExtractedDownloads:
    """Two runs shared extracted_<hash16>/extracted_files.zip.

    The second's truncating write made the first fail with "File is not a
    zip file", and its unlink took the other run's archive with it.
    """

    def test_no_archive_is_written_under_a_shared_name(self, samples, tmp_path):
        other = tmp_path / "extracted_files.zip"
        other.write_bytes(b"the other run's archive")
        samples.client.download_extracted_files.return_value = sdk_response(
            200, content=_zip_bytes({"alpha.bin": b"MZ"})
        )

        assert samples.download_extracted_files(SHA256, tmp_path) is True

        assert (tmp_path / "alpha.bin").read_bytes() == b"MZ"
        assert other.read_bytes() == b"the other run's archive"


class TestAFailedDownloadLeavesNoTruncatedSample:
    """Nothing distinguishes a truncated sample from a complete one."""

    def _write_then_fail(self, monkeypatch, error: BaseException) -> None:
        real_fdopen = os.fdopen

        class _FullDisk:
            """7000 bytes land, then the write gives up — ENOSPC, or Ctrl-C."""

            def __init__(self, handle: Any):
                self._handle = handle

            def __enter__(self) -> _FullDisk:
                return self

            def __exit__(self, *exc_info: Any) -> Literal[False]:
                # The write below is meant to reach the caller: a stand-in
                # that swallowed it would leave the test asserting on a
                # download that never failed.
                self._handle.close()
                return False

            def write(self, data: bytes) -> int:
                self._handle.write(data[:7000])
                raise error

        monkeypatch.setattr(os, "fdopen", lambda fd, mode: _FullDisk(real_fdopen(fd, mode)))

    def test_a_failed_write_leaves_nothing(self, samples, tmp_path, monkeypatch):
        out = tmp_path / "out"
        out.mkdir()
        target = out / f"{SHA256}.malware"
        samples.client.download_sample.return_value = sdk_response(200, content=b"MZ" * 10000)
        self._write_then_fail(monkeypatch, OSError(errno.ENOSPC, "No space left on device"))

        assert samples.download_sample(SHA256, target) is False
        assert list(out.iterdir()) == []

    def test_an_interrupted_write_leaves_nothing(self, samples, tmp_path, monkeypatch):
        out = tmp_path / "out"
        out.mkdir()
        target = out / f"{SHA256}.malware"
        samples.client.download_sample.return_value = sdk_response(200, content=b"MZ" * 10000)
        self._write_then_fail(monkeypatch, KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            samples.download_sample(SHA256, target)
        assert list(out.iterdir()) == []


class TestUploadAcceptedWithAnUnreadableBody:
    """201 plus a truncating proxy is still a sample on the appliance."""

    def _bodyless(self, status: int) -> MagicMock:
        response = sdk_response(status)
        response.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
        return response

    def test_a_2xx_is_reported_as_accepted(self, samples, tmp_path):
        sample = tmp_path / "sample.bin"
        sample.write_bytes(b"x")
        samples.client.submit_file_from_path.return_value = self._bodyless(201)

        result = samples.upload_file(sample)

        assert result, "the sample is on the appliance and queued for analysis"
        assert "task_id" not in result, "there is no digest to wait on"

    def test_a_rejection_with_the_same_body_still_fails(self, samples, tmp_path):
        sample = tmp_path / "sample.bin"
        sample.write_bytes(b"x")
        samples.client.submit_file_from_path.return_value = self._bodyless(500)

        assert samples.upload_file(sample) is None


class TestBatchDeleteAcceptanceOutlivesTheStatusPoll:
    """The 202 is the acceptance; the poll only confirms it."""

    def _accepted(self, samples) -> None:
        samples.client.delete_samples.return_value = sdk_response(202, {"id": "task-9"})

    def test_a_failing_status_poll_does_not_unaccept_the_removal(self, samples, capsys):
        self._accepted(samples)
        samples.client.check_sample_removal_status_v2.return_value = sdk_response(
            500, {"message": "Removal service is down"}
        )

        assert samples.batch_delete_samples([SHA256, SHA1]) == 2
        # Rich wraps the warning to the console width, so read it unwrapped.
        warned = " ".join(capsys.readouterr().err.split())
        assert "task-9" in warned
        assert "Removal service is down" in warned, "the appliance's reason was dropped"

    def test_a_status_poll_the_endpoint_does_not_use_is_named(self, samples, capsys):
        """Neither a 202 nor a refusal: the wait ends and says which it was."""
        self._accepted(samples)
        samples.client.check_sample_removal_status_v2.return_value = sdk_response(302)

        assert samples.batch_delete_samples([SHA256]) == 1
        assert "302" in capsys.readouterr().err

    def test_a_202_whose_body_cannot_be_read_is_still_accepted(self, samples, capsys):
        """An empty or proxy-truncated body does not withdraw the 202.

        ``response.json()`` was called unguarded, so a ``ValueError``
        reached ``with_client`` and became this method's ``default=0`` —
        and ``batch-delete`` printed "Removal failed for all N samples"
        and exited 1 while the appliance was removing them. It is the
        case ``upload_file`` already guards, on the safety-relevant side.
        """
        response = sdk_response(202)
        response.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
        samples.client.delete_samples.return_value = response

        assert samples.batch_delete_samples([SHA256, SHA1]) == 2

        # No task id came back, so there is nothing to poll and the
        # analyst has to be told the removal cannot be followed.
        samples.client.check_sample_removal_status_v2.assert_not_called()
        warned = " ".join(capsys.readouterr().err.split())
        assert "Removal accepted" in warned
        assert "could not be read" in warned

    def test_a_redirect_is_not_an_accepted_batch_and_does_not_go_unsaid(self, samples, capsys):
        """The silent branch: a status ``succeeded`` neither took nor raised on.

        It answered ``False``, this method turned that into its
        ``default=0``, and ``batch-delete`` printed "Removal failed for
        all N samples" and exited 1 — a refusal by the appliance that
        nobody had heard — with no other line anywhere naming the status.
        A removal we never got to ask for is ``None``, and the status
        names itself.
        """
        samples.client.delete_samples.return_value = sdk_response(302)

        assert samples.batch_delete_samples([SHA256, SHA1]) is None

        samples.client.check_sample_removal_status_v2.assert_not_called()
        assert "302" in capsys.readouterr().err

    def test_a_poll_timeout_does_not_unaccept_the_removal(self, samples, monkeypatch, capsys):
        monkeypatch.setattr(http_module.time, "sleep", lambda _: None)
        ticks = iter([0.0, 0.0])
        monkeypatch.setattr(http_module.time, "time", lambda: next(ticks, 10_000.0))
        self._accepted(samples)
        samples.client.check_sample_removal_status_v2.return_value = sdk_response(202)

        assert samples.batch_delete_samples([SHA256]) == 1
        assert "task-9" in capsys.readouterr().err

    def test_ctrl_c_during_the_poll_names_the_task_and_still_aborts(self, samples, capsys):
        """Swallowing it printed "Removal accepted" and exited 0.

        Ctrl-C then could not abort ``batch-delete`` at all, while the
        task it names really is still running on the appliance.
        """
        self._accepted(samples)
        samples.client.check_sample_removal_status_v2.side_effect = KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            samples.batch_delete_samples([SHA256])

        assert "task-9" in capsys.readouterr().err

    def test_a_messageless_failure_names_its_exception_type(self, samples, capsys):
        """``exc or type(exc).__name__``: an exception object is truthy."""
        self._accepted(samples)
        samples.client.check_sample_removal_status_v2.side_effect = RuntimeError()

        assert samples.batch_delete_samples([SHA256]) == 1
        assert "(RuntimeError)" in capsys.readouterr().err


class TestEveryHashEntryPointIsGuarded:
    """Two commands over one endpoint used to disagree about what a hash is.

    Validation sat in whichever module happened to have grown it, so
    ``report -f titanium bogus`` was refused by name while
    ``titanium-report bogus`` — the same endpoint, the same argument —
    handed the analyst the SDK's "not a valid hexadecimal value".
    """

    @pytest.mark.parametrize(
        "call",
        [
            lambda s: s.get_summary_report_v2("bogus"),
            lambda s: s.get_titanium_core_report_v2("bogus"),
        ],
        ids=["summary-report", "titanium-report"],
    )
    def test_a_report_wrapper_names_the_bad_hash_itself(self, reports, call):
        reports.output = MagicMock()

        assert call(reports) is None

        assert "bogus" in reports.output.error.call_args.args[0]
        assert not reports.client.method_calls

    @pytest.mark.parametrize(
        "call",
        [
            lambda s: s.add_user_tags("bogus", ["t"]),
            lambda s: s.get_user_tags("bogus"),
            lambda s: s.remove_user_tags("bogus", ["t"]),
            lambda s: s.set_classification("bogus", "malicious"),
            lambda s: s.get_classification("bogus"),
            lambda s: s.delete_classification("bogus"),
            lambda s: s.list_containers("bogus"),
        ],
        ids=[
            "add-tags",
            "get-tags",
            "remove-tags",
            "set-classification",
            "get-classification",
            "delete-classification",
            "containers",
        ],
    )
    def test_a_metadata_wrapper_names_the_bad_hash_itself(self, metadata, call):
        metadata.output = MagicMock()

        assert not call(metadata)

        assert "bogus" in metadata.output.error.call_args.args[0]
        assert not metadata.client.method_calls

    def test_the_dynamic_status_wants_the_sha1_its_siblings_want(self, reports):
        """``dynamic-report-create`` said so; its status twin leaked ``('sha1',)``."""
        reports.output = MagicMock()

        assert reports.check_dynamic_analysis_status("0" * 32) is None

        assert reports.output.error.call_args.args[0] == "Dynamic reports require SHA1 hash"
        assert not reports.client.method_calls


class TestABatchNamesTheHashThatSankIt:
    """One typo in a 500-line --hash-file aborted the batch anonymously."""

    @pytest.mark.parametrize(
        "call", ["batch_delete_samples", "batch_reanalyze_samples"], ids=["delete", "reanalyze"]
    )
    def test_the_offending_entry_is_named_and_nothing_is_sent(self, samples, call):
        samples.output = MagicMock()
        batch = [SHA256, SHA1, "n0tahash", SHA256]

        assert not getattr(samples, call)(batch)

        message = samples.output.error.call_args.args[0]
        assert "n0tahash" in message, "the batch died without naming the offender"
        assert "3" in message, "the batch died without saying which entry"
        assert not samples.client.method_calls


class TestDomainReportValidatesLikeItsSiblings:
    """The one network lookup that sent anything the user typed."""

    def test_nonsense_never_reaches_the_appliance(self, network):
        network.output = MagicMock()

        assert network.get_domain_report("not a domain!!") is None

        assert "not a domain!!" in network.output.error.call_args.args[0]
        network.client.network_domain_report.assert_not_called()

    def test_a_real_domain_still_goes_through(self, network):
        network.client.network_domain_report.return_value = sdk_response(200, {"ok": True})

        assert network.get_domain_report("example.com") == {"ok": True}


class TestAWaitThatRanOutIsAFailure:
    """``--wait`` is a promise; not keeping it exited 0 and fed the next command."""

    def test_the_timeout_is_reported_as_an_error(self, samples, monkeypatch):
        samples.output = MagicMock()
        samples.output.progress_spinner.return_value.__enter__.return_value.task_ids = [0]
        monkeypatch.setattr(http_module.time, "sleep", lambda _seconds: None)
        samples.client.file_analysis_status.return_value = sdk_response(
            200, {"results": [{"status": "not_found"}]}
        )

        assert samples.wait_for_analysis(SHA256, timeout=0) is None

        assert samples.output.error.called, "a --wait that never landed exited 0"
        assert not samples.output.warning.called


class TestDeleteClassificationGuardsItsSystemLikeSetDoes:
    """The two are twins, and only one used to check where it was aiming.

    ``set_classification`` refused anything but local/ticloud; the delete
    path took ``--system cloud`` straight to the SDK, so the same typo
    was caught on one call and silently mishandled on its twin.
    """

    def test_an_unknown_system_never_reaches_the_appliance(self, metadata, capsys):
        assert metadata.delete_classification(SHA256, system="cloud") is False
        metadata.client.delete_classification.assert_not_called()
        assert "Invalid system 'cloud'" in capsys.readouterr().err

    def test_both_known_systems_still_go_through(self, metadata):
        metadata.client.delete_classification.return_value = sdk_response(200)
        for system in ("local", "ticloud"):
            assert metadata.delete_classification(SHA256, system=system) is True

    def test_setting_one_is_guarded_exactly_the_same_way(self, metadata, capsys):
        assert metadata.set_classification(SHA256, "malicious", system="cloud") is False
        metadata.client.set_classification.assert_not_called()
        assert "Invalid system 'cloud'" in capsys.readouterr().err


# --- Network intelligence: a wrong answer must not read as a reassuring one ---

# The three "what else sits on this IP" lookups: (wrapper, SDK method, the
# key its payload carries the records under).
_PAGED_IP_LOOKUPS = [
    ("get_files_from_ip", "network_files_from_ip", "downloaded_files"),
    ("get_domains_from_ip", "network_ip_to_domain", "resolutions"),
    ("get_urls_from_ip", "network_urls_from_ip", "urls"),
]


@pytest.fixture
def a1000_cli(monkeypatch):
    """Run one ``a1000`` command through the root CLI.

    The non-zero exit status comes from ``cli.result_callback``, which a
    run started at the ``a1000`` group never reaches — and the difference
    between "nothing found" and "could not look it up" is exactly an exit
    status, so these go through the root.
    """
    client = MagicMock()
    monkeypatch.setattr(A1000Session, "_open", lambda session: setattr(session, "client", client))

    def run(args, sdk_method=None, body=None):
        if sdk_method is not None:
            getattr(client, sdk_method).return_value = sdk_response(200, body)
        return client, CliRunner().invoke(cli, ["a1000", *args])

    return run


class TestDomainReportTakesADomainAndNothingElse:
    """The endpoint interpolates its argument into a path segment unquoted.

    ``network_url_report`` quotes with ``parse.quote_plus``; this one does
    not, so ``https://evil.com/a/b?q=1`` was a request about something
    else, answered 200 and empty, and reported as "No network
    intelligence for https://evil.com/a/b?q=1" with exit 0.
    """

    @pytest.mark.parametrize(
        "argument",
        ["https://evil.com/a/b?q=1", "evil.com:8080", "user@evil.com", "evil.com/path"],
    )
    def test_anything_that_reshapes_the_path_is_refused(self, network, argument):
        network.output = MagicMock()

        assert network.get_domain_report(argument) is None

        assert argument in network.output.error.call_args.args[0]
        network.client.network_domain_report.assert_not_called()

    def test_the_domain_is_sent_lowercased_and_unpadded(self, network):
        network.client.network_domain_report.return_value = sdk_response(200, {"ok": True})

        assert network.get_domain_report(" EVIL.COM ") == {"ok": True}

        network.client.network_domain_report.assert_called_once_with("evil.com")

    def test_a_url_exits_non_zero_instead_of_saying_no_intelligence(self, a1000_cli):
        client, result = a1000_cli(["domain-report", "https://evil.com/a/b?q=1"])

        assert result.exit_code == 1
        assert "Invalid domain" in flat(result)
        assert "No network intelligence" not in flat(result)
        client.network_domain_report.assert_not_called()

    def test_a_real_domain_with_nothing_on_it_still_exits_zero(self, a1000_cli):
        _, result = a1000_cli(["domain-report", "evil.com"], "network_domain_report", {})

        assert result.exit_code == 0
        assert "No network intelligence" in flat(result)


class TestIpLookupsSayWhenThereIsMore:
    """These endpoints take 500 records a page and carry ``next_page``.

    Dropping it announced "Found 500 files for 8.8.8.8", exit 0, with
    nothing to say the rest existed — ``advanced_search`` guards the same
    case for Advanced Search.

    The service says what it measured and the command says what to type:
    naming ``--all`` down here made the service unusable to any caller
    without that option, so the flag is asserted at the CLI below.
    """

    @pytest.mark.parametrize("wrapper,sdk_method,key", _PAGED_IP_LOOKUPS)
    def test_a_further_page_is_announced(self, network, wrapper, sdk_method, key):
        network.output = MagicMock()
        getattr(network.client, sdk_method).return_value = sdk_response(
            200, {key: [{"x": 1}], "next_page": "abc123"}
        )

        assert getattr(network, wrapper)("8.8.8.8") == [{"x": 1}]

        assert "first page only" in network.output.warning.call_args.args[0]
        assert "--all" not in network.output.warning.call_args.args[0]
        assert network.pages_left_unfetched

    @pytest.mark.parametrize("wrapper,sdk_method,key", _PAGED_IP_LOOKUPS)
    def test_a_last_page_is_announced_as_nothing(self, network, wrapper, sdk_method, key):
        network.output = MagicMock()
        getattr(network.client, sdk_method).return_value = sdk_response(200, {key: [{"x": 1}]})

        assert getattr(network, wrapper)("8.8.8.8") == [{"x": 1}]

        assert not network.output.warning.called
        assert not network.pages_left_unfetched

    def test_the_analyst_is_told_to_re_run_with_all(self, a1000_cli):
        _, result = a1000_cli(
            ["ip-files", "8.8.8.8"],
            "network_files_from_ip",
            {"downloaded_files": [{"sha256": SHA256}], "next_page": "abc123"},
        )

        assert result.exit_code == 0
        assert "--all" in flat(result)


class TestNoCallInheritsThePagesTheLastOneLeftBehind:
    """The flag is about the call that just finished, and about nothing else.

    Every entry point clears it before its first request, so a lookup that
    measures no paging at all cannot hand the previous call's ``True`` to
    a caller — which reports a whole answer as a partial one, under a
    remedy that fetches nothing.
    """

    def _left_a_page_behind(self, network):
        network.output = MagicMock()
        network.client.network_files_from_ip.return_value = sdk_response(
            200, {"downloaded_files": [{"x": 1}], "next_page": "abc123"}
        )
        network.get_files_from_ip("8.8.8.8")
        assert network.pages_left_unfetched

    def test_a_domain_report_does_not_inherit_it(self, network):
        self._left_a_page_behind(network)
        network.client.network_domain_report.return_value = sdk_response(200, {"ok": True})

        assert network.get_domain_report("evil.com") == {"ok": True}

        assert not network.pages_left_unfetched

    def test_an_ip_report_does_not_inherit_it(self, network):
        self._left_a_page_behind(network)
        network.client.network_ip_addr_report.return_value = sdk_response(200, {"ok": True})

        assert network.get_ip_report("8.8.8.8") == {"ok": True}

        assert not network.pages_left_unfetched

    def test_an_address_the_guard_refused_does_not_leave_it_set(self, network):
        """The clearing happens on the way in, so a call that never lands clears too."""
        self._left_a_page_behind(network)

        assert network.get_files_from_ip("not-an-ip") is None

        assert not network.pages_left_unfetched

    def test_a_walk_that_fetched_every_page_does_not_inherit_it(self, network):
        """An aggregated walk leaves nothing behind and records nothing."""
        self._left_a_page_behind(network)
        network.client.network_files_from_ip.side_effect = pivot_pages(
            "downloaded_files", ([{"x": 1}], None)
        )

        assert network.get_files_from_ip_aggregated("8.8.8.8") == [{"x": 1}]

        assert not network.pages_left_unfetched

    def test_a_call_that_raises_does_not_leave_it_set(self, network):
        """The flag is cleared on the way in, so nothing has to unwind it."""
        self._left_a_page_behind(network)
        network.client.network_files_from_ip.side_effect = RuntimeError("appliance said no")

        assert network.get_files_from_ip("8.8.8.8") is None

        assert not network.pages_left_unfetched

    def test_listing_extracted_files_does_not_inherit_it(self, samples):
        samples.output = MagicMock()
        samples.client.advanced_search_v3.return_value = sdk_response(
            200, {"rl": {"web_search_api": {"entries": [{"sha1": SHA1}], "more_pages": True}}}
        )
        samples.advanced_search("available:true", limit=100)
        assert samples.pages_left_unfetched

        samples.client.list_extracted_files_v2.return_value = sdk_response(
            200, {"count": 1, "results": [{"sha1": SHA1}]}
        )

        assert samples.list_extracted_files(SHA256) == [{"sha1": SHA1}]

        assert not samples.pages_left_unfetched


class TestAnUnreadableAnswerIsNotAnEmptyOne:
    """A key we cannot read is a failure; a key that is absent is an empty page.

    ``data.get(key, [])`` reported a shape it did not recognise as a
    miss. Refusing both directions was the over-correction: every A1000
    aggregator in the SDK reads its page key with ``.get(key, [])``, so a
    reader that fails on a missing key is stricter than the API it wraps
    and grades a legitimately empty page — ``{"count": 0}`` — as a failed
    lookup.
    """

    @pytest.mark.parametrize("wrapper,sdk_method,key", _PAGED_IP_LOOKUPS)
    def test_a_key_that_is_not_a_list_is_still_a_failure(self, network, wrapper, sdk_method, key):
        network.output = MagicMock()
        getattr(network.client, sdk_method).return_value = sdk_response(
            200, {key: {"sha256": SHA256}}
        )

        assert getattr(network, wrapper)("8.8.8.8") is None

        assert key in network.output.error.call_args.args[0]

    @pytest.mark.parametrize("wrapper,sdk_method,key", _PAGED_IP_LOOKUPS)
    def test_a_body_without_the_key_is_the_empty_page_the_sdk_reads(
        self, network, wrapper, sdk_method, key
    ):
        getattr(network.client, sdk_method).return_value = sdk_response(
            200, {"count": 0, "next_page": None}
        )

        assert getattr(network, wrapper)("8.8.8.8") == []

    @pytest.mark.parametrize("wrapper,sdk_method,key", _PAGED_IP_LOOKUPS)
    def test_an_empty_body_is_the_empty_answer_it_looks_like(
        self, network, wrapper, sdk_method, key
    ):
        getattr(network.client, sdk_method).return_value = sdk_response(200, {})

        assert getattr(network, wrapper)("8.8.8.8") == []

    def test_the_analyst_sees_a_failure_not_no_files_found(self, a1000_cli):
        _, result = a1000_cli(
            ["ip-files", "8.8.8.8"],
            "network_files_from_ip",
            {"downloaded_files": {"sha256": SHA256}},
        )

        assert result.exit_code == 1
        assert "Failed to look up files" in flat(result)
        assert "No files found" not in flat(result)

    def test_the_failure_says_which_part_of_the_answer_was_unreadable(self, a1000_cli):
        """A bare "Failed to look up files for 8.8.8.8" named no reason at all."""
        _, result = a1000_cli(
            ["ip-files", "8.8.8.8"],
            "network_files_from_ip",
            {"downloaded_files": {"sha256": SHA256}},
        )

        assert "carried no files list under 'downloaded_files'" in flat(result)

    def test_an_empty_page_exits_zero(self, a1000_cli):
        _, result = a1000_cli(
            ["ip-files", "8.8.8.8"], "network_files_from_ip", {"count": 0, "next_page": None}
        )

        assert result.exit_code == 0
        assert "No files found for this IP" in flat(result)

    def test_an_empty_body_still_exits_zero(self, a1000_cli):
        _, result = a1000_cli(["ip-files", "8.8.8.8"], "network_files_from_ip", {})

        assert result.exit_code == 0
        assert "No files found for this IP" in flat(result)


class TestGuardsSendWhatTheyJudged:
    """The guard decided on ``value.strip()``; the caller sent ``value``."""

    def test_a_padded_address_reaches_the_sdk_stripped(self, network):
        network.client.network_ip_addr_report.return_value = sdk_response(200, {"ok": True})

        assert network.get_ip_report(" 8.8.8.8 ") == {"ok": True}

        network.client.network_ip_addr_report.assert_called_once_with("8.8.8.8")

    def test_a_padded_url_reaches_the_sdk_stripped(self, network):
        network.client.network_url_report.return_value = sdk_response(200, {"ok": True})

        assert network.get_network_url_report(" https://evil.com/a ") == {"ok": True}

        network.client.network_url_report.assert_called_once_with("https://evil.com/a")

    def test_an_aggregated_lookup_strips_the_address_too(self, network):
        network.client.network_files_from_ip.side_effect = pivot_pages(
            "downloaded_files", ([], None)
        )

        assert network.get_files_from_ip_aggregated(" 8.8.8.8 ") == []

        network.client.network_files_from_ip.assert_called_once_with(
            "8.8.8.8", page=None, page_size=_RECORDS_PER_PAGE
        )


class TestTheAddressReachesThePathAsOneKey:
    """``__ip_addr_endpoints`` does ``specific_endpoint.format(ip=ip_addr)``.

    So whatever the guard returns becomes a path segment, unquoted. Two
    things followed from stripping and nothing else: an expanded or
    uppercased IPv6 address asked about a host the appliance files under
    its short form — the ``evil.com.`` / ``evil.com`` split
    ``valid_domain`` exists to close — and ``fe80::1%eth0`` wrote a
    percent-escape introducer into the path, which is the reshaping the
    domain guard exists to prevent.
    """

    @pytest.mark.parametrize(
        "typed,sent",
        [
            ("2001:DB8:0000:0000:0000:0000:0000:0001", "2001:db8::1"),
            ("2001:db8::1", "2001:db8::1"),
            ("fe80::1%eth0", "fe80::1"),
            (" 8.8.8.8 ", "8.8.8.8"),
        ],
    )
    def test_the_canonical_address_is_what_is_interpolated(self, network, typed, sent):
        network.client.network_ip_addr_report.return_value = sdk_response(200, {"ok": True})

        assert network.get_ip_report(typed) == {"ok": True}

        network.client.network_ip_addr_report.assert_called_once_with(sent)

    def test_the_paging_lookups_key_the_same_address_the_same_way(self, network):
        network.client.network_files_from_ip.return_value = sdk_response(
            200, {"downloaded_files": []}
        )

        assert network.get_files_from_ip("2001:DB8::0001") == []

        network.client.network_files_from_ip.assert_called_once_with(
            "2001:db8::1", page_size=_RECORDS_PER_PAGE
        )


class TestAnIdnDomainIsLookedUpRatherThanRefused:
    """The guard matched an ASCII-only label pattern against the typed name.

    So ``münchen.de`` — the spelling in the lure the analyst is pasting
    from — was refused before the appliance was asked, while its punycode
    twin went through. The endpoint keys on the punycode form, so the
    encoding is this CLI's job.
    """

    @pytest.mark.parametrize(
        "typed,sent",
        [
            ("münchen.de", "xn--mnchen-3ya.de"),
            ("évil.com", "xn--vil-9la.com"),
            (" MÜNCHEN.de. ", "xn--mnchen-3ya.de"),
            ("xn--mnchen-3ya.de", "xn--mnchen-3ya.de"),
        ],
    )
    def test_the_domain_reaches_the_path_punycode_encoded(self, network, typed, sent):
        network.client.network_domain_report.return_value = sdk_response(200, {"ok": True})

        assert network.get_domain_report(typed) == {"ok": True}

        network.client.network_domain_report.assert_called_once_with(sent)

    def test_an_idn_lookup_exits_zero_instead_of_naming_a_typo(self, a1000_cli):
        _, result = a1000_cli(["domain-report", "münchen.de"], "network_domain_report", {})

        assert result.exit_code == 0
        assert "Invalid domain" not in flat(result)


SHA512 = "c" * 128


class TestEndpointsStricterThanTheAppliance:
    """The tag endpoints and the container lookup take no SHA512.

    The SDK's ``allowed_hash_types`` is ``(MD5, SHA1, SHA256)`` for these
    four, so a SHA512 was accepted at the CLI and refused inside the SDK
    — "Only hash strings of the following types are allowed", with all
    128 characters echoed back and nothing said about which endpoint.
    """

    @pytest.mark.parametrize(
        "call, sdk_method",
        [
            (lambda service: service.add_user_tags(SHA512, ["apt"]), "post_user_tags"),
            (lambda service: service.get_user_tags(SHA512), "get_user_tags"),
            (lambda service: service.remove_user_tags(SHA512, ["apt"]), "delete_user_tags"),
            (lambda service: service.list_containers(SHA512), "list_containers_for_hashes"),
        ],
        ids=["add-tags", "get-tags", "remove-tags", "containers"],
    )
    def test_a_sha512_is_refused_here_and_never_sent(self, metadata, call, sdk_method):
        metadata.output = MagicMock()

        assert not call(metadata)

        message = metadata.output.error.call_args.args[0]
        assert "SHA512" in message
        assert "MD5, SHA1, SHA256" in message
        assert not metadata.client.method_calls

    @pytest.mark.parametrize(
        "call, sdk_method",
        [
            (lambda service: service.get_classification(SHA512), "get_classification_v3"),
            (
                lambda service: service.set_classification(SHA512, "malicious"),
                "set_classification",
            ),
            (lambda service: service.delete_classification(SHA512), "delete_classification"),
        ],
        ids=["get-classification", "set-classification", "delete-classification"],
    )
    def test_the_classification_endpoints_still_take_one(self, metadata, call, sdk_method):
        """These three do accept SHA512, so the stricter tuple must not reach them."""
        getattr(metadata.client, sdk_method).return_value = sdk_response(200, {})

        call(metadata)

        assert getattr(metadata.client, sdk_method).called

    def test_a_sha512_still_reaches_the_endpoints_that_take_it(self, samples):
        samples.client.delete_samples.return_value = sdk_response(200, {})

        assert samples.delete_sample(SHA512)

        samples.client.delete_samples.assert_called_once_with(SHA512)


class TestTheHashSentIsTheHashTheGuardJudged:
    """``validate_hash`` decides on the stripped, lowercased value.

    The callers forwarded their raw argument, so a hash pasted with a
    trailing newline passed the guard and was then refused inside the
    SDK, and an uppercase one reached the appliance uppercased — asking
    about a sample under a spelling the API does not use.
    """

    PADDED = f"  {SHA256.upper()}\n"

    @pytest.mark.parametrize(
        "call, sdk_method",
        [
            (lambda service, value: service.delete_sample(value), "delete_samples"),
            (lambda service, value: service.reanalyze_sample(value), "reanalyze_samples_v2"),
            (
                lambda service, value: service.list_extracted_files(value),
                "list_extracted_files_v2",
            ),
        ],
        ids=["delete", "reanalyze", "extracted"],
    )
    def test_a_sample_wrapper_sends_the_normalized_hash(self, samples, call, sdk_method):
        getattr(samples.client, sdk_method).return_value = sdk_response(200, {"results": []})

        call(samples, self.PADDED)

        assert getattr(samples.client, sdk_method).call_args.args[0] == SHA256

    def test_download_asks_for_the_normalized_hash(self, samples, tmp_path):
        samples.client.download_sample.return_value = sdk_response(200, {}, content=b"MZ")

        assert samples.download_sample(self.PADDED, tmp_path / "sample.malware")

        samples.client.download_sample.assert_called_once_with(SHA256)

    @pytest.mark.parametrize(
        "call, sdk_method",
        [
            (lambda service, value: service.add_user_tags(value, ["apt"]), "post_user_tags"),
            (lambda service, value: service.get_user_tags(value), "get_user_tags"),
            (
                lambda service, value: service.remove_user_tags(value, ["apt"]),
                "delete_user_tags",
            ),
            (lambda service, value: service.get_classification(value), "get_classification_v3"),
        ],
        ids=["add-tags", "get-tags", "remove-tags", "get-classification"],
    )
    def test_a_metadata_wrapper_sends_the_normalized_hash(self, metadata, call, sdk_method):
        getattr(metadata.client, sdk_method).return_value = sdk_response(200, [])

        call(metadata, self.PADDED)

        assert getattr(metadata.client, sdk_method).call_args.args[0] == SHA256

    @pytest.mark.parametrize(
        "call, sdk_method",
        [
            (lambda service, value: service.get_summary_report_v2(value), "get_summary_report_v2"),
            (
                lambda service, value: service.get_titanium_core_report_v2(value),
                "get_titanium_core_report_v2",
            ),
            (
                lambda service, value: service.get_report(value, "titanium"),
                "get_titanium_core_report_v2",
            ),
        ],
        ids=["summary-report", "titanium-report", "report-titanium"],
    )
    def test_a_report_wrapper_sends_the_normalized_hash(self, reports, call, sdk_method):
        """``report --format pdf`` was converted; its five siblings were not."""
        getattr(reports.client, sdk_method).return_value = sdk_response(200, {"ok": True})

        call(reports, self.PADDED)

        assert getattr(reports.client, sdk_method).call_args.args[0] == SHA256

    def test_the_json_report_sends_the_normalized_hash_by_keyword(self, reports):
        reports.client.get_detailed_report_v2.return_value = sdk_response(200, {"ok": True})

        reports.get_report(self.PADDED, "json")

        assert reports.client.get_detailed_report_v2.call_args.kwargs["sample_hashes"] == SHA256

    @pytest.mark.parametrize(
        "call, sdk_method",
        [
            (
                lambda service, value: service.create_dynamic_report(value),
                "create_dynamic_analysis_report",
            ),
            (
                lambda service, value: service.check_dynamic_analysis_status(value),
                "check_dynamic_analysis_report_status",
            ),
        ],
        ids=["dynamic-create", "dynamic-status"],
    )
    def test_a_dynamic_report_wrapper_sends_the_normalized_sha1(self, reports, call, sdk_method):
        getattr(reports.client, sdk_method).return_value = sdk_response(200, {"ok": True})

        call(reports, f"  {SHA1.upper()}\n")

        assert getattr(reports.client, sdk_method).call_args.args[0] == SHA1

    def test_the_dynamic_download_waits_on_the_normalized_sha1_too(self, reports):
        reports.client.check_dynamic_analysis_report_status.return_value = sdk_response(
            200, {"status": 2}
        )
        reports.client.download_dynamic_analysis_report.return_value = sdk_response(
            200, {}, content=b"%PDF"
        )

        assert reports.download_dynamic_report(f"  {SHA1.upper()}\n") == b"%PDF"

        reports.client.check_dynamic_analysis_report_status.assert_called_once_with(SHA1, "pdf")
        reports.client.download_dynamic_analysis_report.assert_called_once_with(
            SHA1, report_format="pdf"
        )

    @pytest.mark.parametrize(
        "call, sdk_method",
        [
            (
                lambda service, fmt: service.create_dynamic_report(SHA1, fmt),
                "create_dynamic_analysis_report",
            ),
            (
                lambda service, fmt: service.check_dynamic_analysis_status(SHA1, fmt),
                "check_dynamic_analysis_report_status",
            ),
            (
                lambda service, fmt: service.download_dynamic_report(SHA1, fmt),
                "download_dynamic_analysis_report",
            ),
        ],
        ids=["dynamic-create", "dynamic-status", "dynamic-download"],
    )
    def test_a_dynamic_report_format_the_endpoint_cannot_build_is_refused(
        self, reports, call, sdk_method
    ):
        """``download`` carried no format check at all, so it forwarded anything.

        Its two siblings each carried their own copy of the guard; the one
        method that writes a file to disk had none, and handed the
        appliance whatever string the caller passed.
        """
        assert call(reports, "docx") is None

        getattr(reports.client, sdk_method).assert_not_called()

    def test_a_dynamic_report_format_is_normalized_before_it_is_sent(self, reports):
        reports.client.create_dynamic_analysis_report.return_value = sdk_response(200, {"ok": True})

        reports.create_dynamic_report(SHA1, "  PDF  ")

        assert reports.client.create_dynamic_analysis_report.call_args.args[1] == "pdf"

    def test_a_padded_hash_gets_a_summary_report_instead_of_an_sdk_error(self, a1000_cli):
        """The SDK's ``validate_hashes`` raised on exactly this input."""
        _, result = a1000_cli(
            ["summary-report", self.PADDED], "get_summary_report_v2", {"sha1": SHA1}
        )

        assert result.exit_code == 0, result.output
        assert "not a valid hexadecimal value" not in flat(result)

    def test_the_container_lookup_normalizes_the_hash_it_wraps_in_a_list(self, metadata):
        metadata.client.list_containers_for_hashes.return_value = sdk_response(200, {"results": []})

        metadata.list_containers(self.PADDED)

        metadata.client.list_containers_for_hashes.assert_called_once_with([SHA256])

    @pytest.mark.parametrize(
        "call, sdk_method",
        [
            (lambda service, batch: service.batch_delete_samples(batch), "delete_samples"),
            (
                lambda service, batch: service.batch_reanalyze_samples(batch),
                "reanalyze_samples_v2",
            ),
        ],
        ids=["batch-delete", "batch-reanalyze"],
    )
    def test_a_batch_is_normalized_entry_by_entry(self, samples, call, sdk_method):
        getattr(samples.client, sdk_method).return_value = sdk_response(200, {"results": []})

        call(samples, [f" {SHA256.upper()} ", f"{SHA1.upper()}\n"])

        assert getattr(samples.client, sdk_method).call_args.args[0] == [SHA256, SHA1]


class TestRecordsUnderAnUnexpectedKeyAreNotAnEmptyPage:
    """Leniency is for a page with nothing on it, not for a spelling we misread.

    Treating an absent key as a failure made the CLI stricter than the SDK
    and reported `{"count": 0, "next": null}` as an error. Treating it as
    empty let `{"extracted_files": [...]}` report "no extracted files"
    over the top of the records it was holding. The distinguishing signal
    is whether the body carries records at all.
    """

    def test_a_page_with_nothing_on_it_is_empty(self):
        assert list_from_envelope({"count": 0, "next": None}, "results") == []

    def test_an_empty_list_elsewhere_is_still_an_empty_page(self):
        assert list_from_envelope({"extracted_files": []}, "results") == []

    def test_records_under_another_name_are_unreadable(self):
        assert list_from_envelope({"extracted_files": [{"sha1": "a"}]}, "results") is None

    def test_a_present_key_of_the_wrong_type_is_still_unreadable(self):
        assert list_from_envelope({"results": {"a": 1}}, "results") is None

    def test_an_empty_body_is_nobody_s_failure(self):
        assert list_from_envelope({}, "results", required=True) == []

    @pytest.mark.parametrize(
        "body", [{"user_tags": []}, {"detail": "Not found."}], ids=["list", "detail"]
    )
    def test_a_required_key_that_is_absent_is_unreadable(self, body):
        """``required`` is the whole difference for a body that hides no records.

        Without it the same body is the empty page the SDK's aggregators
        take it for. With it — the two tag reads and the container lookup,
        whose envelope key this repo cannot corroborate — a missing key is
        far likelier to mean we are reading the wrong one.
        """
        assert list_from_envelope(body, "tags", required=True) is None
        assert list_from_envelope(body, "tags") == []
