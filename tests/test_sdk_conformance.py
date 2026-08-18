"""Signature-conformance tests: every wrapper must call the real SDK legally.

The service decorators swallow exceptions, so a wrapper calling an SDK
method with a wrong name or signature fails silently at runtime. Here the
SDK client is replaced with ``create_autospec`` of the real classes, which
raises TypeError on any call that does not match the installed SDK's
signatures — and the tests assert no such error was recorded.
"""

from __future__ import annotations

import inspect
import io
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from ReversingLabs.SDK import ticloud
from ReversingLabs.SDK.a1000 import A1000

from rl_cli.config import Settings
from rl_cli.models.yara_repository import YaraRepositorySpec
from rl_cli.services.a1000 import (
    A1000MetadataService,
    A1000NetworkService,
    A1000ReportService,
    A1000SampleService,
    A1000Service,
    A1000Session,
    A1000YaraService,
    workflows,
)
from rl_cli.services.titanium_cloud import (
    TitaniumCloudApi,
    TitaniumCloudNetworkService,
    TitaniumCloudService,
)
from tests.conftest import autospec_sdk_client

SHA1 = "da39a3ee5e6b4b0d3255bfef95601890afd80709"
SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
# A second SHA-256, for the bulk endpoints: they send one ``hash_type``
# taken from the first entry, so a batch has to be all of one type.
SHA256_OTHER = "b5bb9d8014a0f9b1d61e21e796d78dccdf1352f23cd32812f4850b878ae4944c"

URL = "http://example.com/dropper.exe"
DOMAIN = "example.com"
IP = "1.2.3.4"

# Where the download wrappers write. Markers rather than paths, so no row
# in the call tables names a location on the machine running them; each
# test resolves them against its own ``tmp_path``. What they are handed is
# never sent to the appliance, which is what the ``download_sample``
# binding row below says in so many words.
#
# The file an upload *reads* is not one of these: it has to exist before
# the wrapper will call the SDK at all, and it reaches the SDK as a value
# a binding row has to name, so those rows pass this module's own path —
# as the TitaniumCloud upload row already did.
SAMPLE_DESTINATION = object()
OUTPUT_DIRECTORY = object()


def _output_directory(tmp_path: Path) -> Path:
    path = tmp_path / "extracted"
    path.mkdir(exist_ok=True)
    return path


_MARKERS: tuple[tuple[object, Any], ...] = (
    (SAMPLE_DESTINATION, lambda tmp_path: tmp_path / "sample.bin"),
    (OUTPUT_DIRECTORY, _output_directory),
)


def _resolved(args: tuple[Any, ...], tmp_path: Path) -> tuple[Any, ...]:
    """``args`` with every path marker bound under this test's tmp dir."""

    def resolve(arg: Any) -> Any:
        for marker, build in _MARKERS:
            if arg is marker:
                path: Path = build(tmp_path)
                return path
        return arg

    return tuple(resolve(arg) for arg in args)


# The five fields the repository endpoints write as a unit; both wrappers
# take them as this one value rather than as five same-typed strings.
REPOSITORY_SPEC = YaraRepositorySpec(
    repository_url="http://repo",
    name="name",
    source_branch="main",
    api_token="",
    import_update_preferences="manual",
)

# Every A1000 service there is. Each wrapper below is exercised on the one
# that defines it, and the coverage test walks all of them, so a wrapper
# added to any service has to be listed here or fail.
# Cross-area work lives in ``workflows`` rather than on a facade over all
# five, so the coverage test walks that module too.
A1000_SERVICES = (
    A1000SampleService,
    A1000ReportService,
    A1000NetworkService,
    A1000YaraService,
    A1000MetadataService,
)


def owning_service(method: str) -> type[A1000Service]:
    """The focused service that defines ``method``, or the shared client.

    Resolving it rather than tabulating it keeps the call lists to
    (wrapper, args) — and asserts on the way past that no two services
    define the same wrapper, which is the thing that would make "the
    service that owns it" ambiguous.
    """
    owners = [service for service in A1000_SERVICES if method in vars(service)]
    assert len(owners) <= 1, f"{method} is defined by more than one service: {owners}"
    return owners[0] if owners else A1000Service


@pytest.fixture(autouse=True)
def surface_swallowed_errors(monkeypatch):
    """Let signature mismatches escape the service decorators.

    ``handle_error`` normally logs and returns a default, which is exactly
    what hides a bad SDK call. Re-raising TypeError/AttributeError here
    makes any signature mismatch fail the test instead of passing silently.
    """
    from rl_cli.services.base import BaseService

    def reraise(self, error: Exception, context: str = "") -> None:
        if isinstance(error, TypeError | AttributeError):
            raise AssertionError(f"invalid SDK call in {context}: {error}") from error

    monkeypatch.setattr(BaseService, "handle_error", reraise)


A1000_WRAPPER_CALLS = [
    # SDK-backed since it stopped trusting the SDK's silence: it calls
    # `file_analysis_status` and judges the status itself, so a kwarg
    # renamed in an SDK minor has to fail here.
    ("test_connection", ()),
    ("get_report", (SHA256,)),
    ("get_report", (SHA256, "xml")),
    ("get_summary_report_v2", (SHA256,)),
    ("get_titanium_core_report_v2", (SHA256,)),
    ("delete_sample", (SHA256,)),
    ("reanalyze_sample", (SHA256,)),
    ("download_sample", (SHA256, SAMPLE_DESTINATION)),
    ("list_extracted_files", (SHA256,)),
    ("download_extracted_files", (SHA256, OUTPUT_DIRECTORY)),
    ("upload_file", (Path(__file__),)),
    ("upload_file", (Path(__file__), "a comment")),
    ("get_report", (SHA256, "pdf")),
    ("create_dynamic_report", (SHA1, "pdf")),
    ("download_dynamic_report", (SHA1, "pdf")),
    ("check_dynamic_analysis_status", (SHA1, "pdf")),
    ("list_samples", ()),
    ("advanced_search", ("classification:malicious",)),
    ("submit_url", ("http://example.com",)),
    ("get_url_report", ("task-id",)),
    ("url_status", ("task-id",)),
    ("get_domain_report", ("example.com",)),
    ("get_ip_report", ("1.2.3.4",)),
    ("get_files_from_ip", ("1.2.3.4",)),
    ("get_domains_from_ip", ("1.2.3.4",)),
    ("get_urls_from_ip", ("1.2.3.4",)),
    ("get_files_from_ip_aggregated", ("1.2.3.4",)),
    ("get_domains_from_ip_aggregated", ("1.2.3.4",)),
    ("get_urls_from_ip_aggregated", ("1.2.3.4",)),
    ("get_network_url_report", ("http://example.com",)),
    ("list_yara_rulesets", ()),
    ("create_yara_ruleset", ("name", "rule x {}")),
    ("toggle_yara_ruleset", ("name", True)),
    ("get_yara_matches", ("ruleset", SHA256)),
    ("delete_yara_ruleset", ("name",)),
    ("get_yara_content", ("name",)),
    ("yara_cloud_retro_scan", ("ruleset", "start")),
    ("yara_local_retro_scan", ("start",)),
    ("get_yara_cloud_retro_scan_status", ("ruleset",)),
    ("get_yara_local_retro_scan_status", ()),
    ("list_yara_repositories", ()),
    ("list_yara_repositories_aggregated", ()),
    ("create_yara_repository", (REPOSITORY_SPEC,)),
    ("update_yara_repository", (1, REPOSITORY_SPEC)),
    ("delete_yara_repository", (1,)),
    ("publish_yara_ruleset", ("name",)),
    ("publish_all_yara_rulesets", ()),
    ("run_yara_update", ()),
    ("set_yara_update_interval", (3600,)),
    ("reset_yara_update_interval", ()),
    ("add_user_tags", (SHA256, ["tag"])),
    ("get_user_tags", (SHA256,)),
    ("remove_user_tags", (SHA256, ["tag"])),
    ("set_classification", (SHA256, "malicious")),
    ("get_classification", (SHA256,)),
    ("delete_classification", (SHA256,)),
    ("list_containers", (SHA256,)),
    ("batch_reanalyze_samples", ([SHA256, SHA1],)),
    ("batch_delete_samples", ([SHA256, SHA1],)),
]


# Bodies that get a polling wrapper past its wait and on to the call it
# exists to make. The status endpoints answer 200 whether or not the
# report is ready, so with an empty body every poll ended as "unexpected
# status" and `download_pdf_report` / `download_dynamic_analysis_report`
# were never reached. 2 is ``rl_cli.services.a1000.reports._REPORT_READY``; the
# bulk delete needs a task id before it polls for removal.
_READY_BODIES: dict[str, dict[str, object]] = {
    "check_pdf_report_creation": {"status": 2},
    "check_dynamic_analysis_report_status": {"status": 2},
    "delete_samples": {"id": 1},
}


def _one_member_zip() -> bytes:
    """The archive body ``download_extracted_files`` unpacks.

    Its wrapper hands ``response.content`` straight to the unpacker, so an
    empty body is a ``BadZipFile`` before the SDK call is ever reached and
    the wrapper cannot be conformance-tested at all. This is why that
    wrapper was waived; a payload is cheaper than a waiver.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("extracted/sample.bin", b"payload")
    return buffer.getvalue()


_RESPONSE_CONTENT: dict[str, bytes] = {"download_extracted_files": _one_member_zip()}


def _autospec_client(sdk_class: type) -> MagicMock:
    """Autospec ``sdk_class``, with the bodies a polling wrapper has to get past."""
    return autospec_sdk_client(sdk_class, bodies=_READY_BODIES, contents=_RESPONSE_CONTENT)


@pytest.fixture
def build_service(tmp_path):
    """Build any A1000 service on an autospec of the real SDK client."""

    def build(service_cls: type[A1000Service]) -> A1000Service:
        settings = Settings(cache_dir=tmp_path / "cache", config_dir=tmp_path / "config")
        session = A1000Session(settings)
        # The client belongs to the session, so that is where it is put:
        # a service only reads it.
        session.client = _autospec_client(A1000)
        svc = session.service(service_cls)
        svc.output = MagicMock()
        return svc

    return build


class TestA1000WrapperSignatures:
    @pytest.mark.parametrize(
        "method,args",
        A1000_WRAPPER_CALLS,
        ids=[f"{m}-{i}" for i, (m, _) in enumerate(A1000_WRAPPER_CALLS)],
    )
    def test_wrapper_calls_sdk_with_valid_signature(self, build_service, tmp_path, method, args):
        service = build_service(owning_service(method))

        getattr(service, method)(*_resolved(args, tmp_path))

    def test_call_list_covers_every_sdk_backed_wrapper(self):
        """New SDK-backed wrappers must be added to A1000_WRAPPER_CALLS."""
        covered = {name for name, _ in A1000_WRAPPER_CALLS}
        exempt = {
            # compositions/polling helpers that only call other wrappers.
            # ``upload_and_get_report`` and ``wait_for_uploaded_analysis``
            # are functions rather than methods, and reach ``public``
            # through the walk of ``workflows``.
            "upload_and_get_report",
            "wait_for_uploaded_analysis",
            "remove_all_user_tags",
            "wait_for_analysis",
            "connect",
            "disconnect",
            "handle_error",
            # pure local knowledge (settings, query DSL, report variants);
            # they touch no SDK method, so there is no signature to conform to
            "build_search_query",
            "connection_summary",
            "report_is_binary",
        }
        public = {
            name
            for service in (A1000Service, *A1000_SERVICES)
            for name in dir(service)
            if not name.startswith("_") and callable(getattr(service, name))
        } | {
            name
            for name, value in vars(workflows).items()
            if not name.startswith("_")
            and inspect.isfunction(value)
            and value.__module__ == workflows.__name__
        }
        missing = public - covered - exempt
        assert not missing, f"wrappers without signature-conformance coverage: {missing}"


# (the service that builds the handle, the handle, its SDK class, wrapper,
# args). The owning service is named rather than assumed: the network
# lookups are their own service, and back when a facade forwarded to one,
# putting the stand-in on the facade left the real SDK object in place —
# the wrapper then talked past every autospec in this table with the test
# still green.
TICLOUD_WRAPPER_CALLS = [
    (
        TitaniumCloudService,
        "_file_reputation",
        ticloud.FileReputation,
        "get_file_reputation",
        (SHA256,),
    ),
    (
        TitaniumCloudService,
        "_file_reputation",
        ticloud.FileReputation,
        "get_bulk_file_reputation",
        ([SHA256, SHA256_OTHER],),
    ),
    # Answers one flat record per hash by picking an endpoint from the
    # count, so both branches are its SDK surface and both are listed.
    (
        TitaniumCloudService,
        "_file_reputation",
        ticloud.FileReputation,
        "get_reputation_records",
        ([SHA256],),
    ),
    (
        TitaniumCloudService,
        "_file_reputation",
        ticloud.FileReputation,
        "get_reputation_records",
        ([SHA256, SHA256_OTHER],),
    ),
    (TitaniumCloudService, "_av_scanners", ticloud.AVScanners, "get_av_scanners", (SHA256,)),
    (TitaniumCloudService, "_file_upload", ticloud.FileUpload, "upload_file", (Path(__file__),)),
    (
        TitaniumCloudService,
        "_file_analysis",
        ticloud.FileAnalysis,
        "get_file_analysis",
        (SHA256,),
    ),
    (
        TitaniumCloudService,
        "_advanced_search",
        ticloud.AdvancedSearch,
        "search_samples",
        ("classification:malicious",),
    ),
    (
        TitaniumCloudService,
        "_file_download",
        ticloud.FileDownload,
        "get_download_status",
        (SHA256,),
    ),
    (
        TitaniumCloudService,
        "_file_download",
        ticloud.FileDownload,
        "download_sample",
        (SHA256, SAMPLE_DESTINATION),
    ),
    # ---- the network pivots: TCA-0401 through TCA-0407 ----
    # Each comes in two, a first page and a whole walk, and both go through
    # the helpers that name the handle and the SDK method as strings — so
    # neither the handle nor the method is checked by anything until here.
    # The aggregated rows are called with a budget as well as without,
    # because ``max_results`` is the one argument those wrappers add.
    (
        TitaniumCloudNetworkService,
        "_url_threat",
        ticloud.URLThreatIntelligence,
        "analyze_url",
        (URL,),
    ),
    (
        TitaniumCloudNetworkService,
        "_url_threat",
        ticloud.URLThreatIntelligence,
        "get_files_from_url",
        (URL,),
    ),
    (
        TitaniumCloudNetworkService,
        "_url_threat",
        ticloud.URLThreatIntelligence,
        "get_files_from_url_aggregated",
        (URL, 5),
    ),
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_domain_report",
        (DOMAIN,),
    ),
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_files_from_domain",
        (DOMAIN,),
    ),
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_files_from_domain_aggregated",
        (DOMAIN,),
    ),
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_files_from_domain_aggregated",
        (DOMAIN, 5),
    ),
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_urls_from_domain",
        (DOMAIN,),
    ),
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_urls_from_domain_aggregated",
        (DOMAIN, 5),
    ),
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_ips_from_domain",
        (DOMAIN,),
    ),
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_ips_from_domain_aggregated",
        (DOMAIN, 5),
    ),
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_related_domains",
        (DOMAIN,),
    ),
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_related_domains_aggregated",
        (DOMAIN, 5),
    ),
    (
        TitaniumCloudNetworkService,
        "_ip_threat",
        ticloud.IPThreatIntelligence,
        "get_ip_report",
        (IP,),
    ),
    (
        TitaniumCloudNetworkService,
        "_ip_threat",
        ticloud.IPThreatIntelligence,
        "get_files_from_ip",
        (IP,),
    ),
    (
        TitaniumCloudNetworkService,
        "_ip_threat",
        ticloud.IPThreatIntelligence,
        "get_files_from_ip_aggregated",
        (IP, 5),
    ),
    (
        TitaniumCloudNetworkService,
        "_ip_threat",
        ticloud.IPThreatIntelligence,
        "get_urls_from_ip",
        (IP,),
    ),
    (
        TitaniumCloudNetworkService,
        "_ip_threat",
        ticloud.IPThreatIntelligence,
        "get_urls_from_ip_aggregated",
        (IP, 5),
    ),
    (
        TitaniumCloudNetworkService,
        "_ip_threat",
        ticloud.IPThreatIntelligence,
        "get_domains_from_ip",
        (IP,),
    ),
    (
        TitaniumCloudNetworkService,
        "_ip_threat",
        ticloud.IPThreatIntelligence,
        "get_domains_from_ip_aggregated",
        (IP, 5),
    ),
    (
        TitaniumCloudNetworkService,
        "_uri_index",
        ticloud.URIIndex,
        "get_uri_index",
        (URL,),
    ),
    (
        TitaniumCloudNetworkService,
        "_uri_index",
        ticloud.URIIndex,
        "get_uri_index_aggregated",
        (URL, 5),
    ),
]


class TestTiCloudWrapperSignatures:
    @pytest.fixture
    def build_ticloud(self, tmp_path):
        def build(service_cls: type[TitaniumCloudApi]) -> TitaniumCloudApi:
            settings = Settings(cache_dir=tmp_path / "cache", config_dir=tmp_path / "config")
            return service_cls(settings, output=MagicMock())

        return build

    @pytest.fixture
    def service(self, build_ticloud) -> TitaniumCloudService:
        service: TitaniumCloudService = build_ticloud(TitaniumCloudService)
        return service

    @pytest.mark.parametrize(
        "service_cls,handle,sdk_class,method,args",
        TICLOUD_WRAPPER_CALLS,
        ids=[f"{row[3]}-{index}" for index, row in enumerate(TICLOUD_WRAPPER_CALLS)],
    )
    def test_wrapper_calls_sdk_with_valid_signature(
        self, build_ticloud, tmp_path, service_cls, handle, sdk_class, method, args
    ):
        service = build_ticloud(service_cls)
        service.__dict__[handle] = _autospec_client(sdk_class)

        getattr(service, method)(*_resolved(args, tmp_path))

    def test_search_samples_signature(self, service, monkeypatch):
        api = _autospec_client(ticloud.AdvancedSearch)
        monkeypatch.setattr(
            "rl_cli.services.titanium_cloud.service.ticloud.AdvancedSearch", lambda **kw: api
        )

        service.search_samples("classification:malicious", limit=50)

    def test_call_list_covers_every_ticloud_wrapper(self):
        """New SDK-backed wrappers must be added to TICLOUD_WRAPPER_CALLS.

        The wrappers are read off the classes rather than tabulated a
        second time: the network pivots reach the SDK through helpers that
        name the handle and the method as strings, so nothing else in the
        suite would notice a twenty-second pivot appearing — or a name
        going wrong in the twenty-first.
        """
        covered = {row[3] for row in TICLOUD_WRAPPER_CALLS}
        exempt = {
            # reporting, shared by every service; talks to no SDK method
            "handle_error",
            # pure local knowledge: which hash types these endpoints take.
            # Public so ``ticloud download`` can ask before it creates a
            # directory, but there is no SDK call to conform to.
            "supported_hash",
        }
        public = {
            name
            for service in (
                TitaniumCloudApi,
                TitaniumCloudService,
                TitaniumCloudNetworkService,
            )
            for name in dir(service)
            if not name.startswith("_") and callable(getattr(service, name))
        }
        missing = public - covered - exempt
        assert not missing, f"wrappers without signature-conformance coverage: {missing}"


# (wrapper, wrapper args, sdk method, expected parameter -> value binding)
#
# create_autospec validates arity and parameter names but not which value
# lands on which parameter, so it passed happily while
# enable_or_disable_yara_ruleset(enabled, name) was being called as
# (name, enabled) and get_yara_ruleset_matches_v2's `page` was receiving a
# file hash. Both shipped broken. These bind the recorded call against the
# real signature and assert the values by parameter name.
#
# This is where "argument X reaches parameter Y" is asserted for the whole
# suite. The per-wrapper tests assert what a wrapper does with an answer;
# spelling the same claim there as ``assert_called_once_with(a, b)`` pins the
# call form instead of the binding, and fails a wrapper that starts passing
# the very same values by keyword.
A1000_ARGUMENT_BINDINGS = [
    (
        "toggle_yara_ruleset",
        ("myrules", True),
        "enable_or_disable_yara_ruleset",
        {"name": "myrules", "enabled": True},
    ),
    (
        # A hash must never reach `page`, which is the second positional
        # parameter and has nothing to do with filtering by sample.
        "get_yara_matches",
        ("myrules", SHA256),
        "get_yara_ruleset_matches_v2",
        {"ruleset_name": "myrules", "page": None, "page_size": None},
    ),
    (
        # The wrapper takes (ruleset, operation); the SDK takes them the
        # other way round. That reversal is exactly what this guards.
        "yara_cloud_retro_scan",
        ("myrules", "start"),
        "start_or_stop_yara_cloud_retro_scan",
        {"operation": "START", "ruleset_name": "myrules"},
    ),
    (
        # The endpoint takes the operation upper-cased, so the wrapper
        # cases it for the caller — onto ``operation``, never onto the name.
        "yara_cloud_retro_scan",
        ("myrules", "Stop"),
        "start_or_stop_yara_cloud_retro_scan",
        {"operation": "STOP", "ruleset_name": "myrules"},
    ),
    (
        "create_yara_ruleset",
        ("myrules", "rule x {}"),
        "create_or_update_yara_ruleset",
        {"name": "myrules", "content": "rule x {}", "publish": None, "ticloud": None},
    ),
    (
        # publish/ticloud are forwarded, and to the parameters of those
        # names: they are two same-typed optionals in a row, so a swap
        # would sync a ruleset to TitaniumCloud on a request to publish it
        # to the cluster.
        "create_yara_ruleset",
        ("myrules", "rule y {}", True, False),
        "create_or_update_yara_ruleset",
        {"name": "myrules", "content": "rule y {}", "publish": True, "ticloud": False},
    ),
    (
        "set_classification",
        (SHA256, "malicious"),
        "set_classification",
        {"sample_hash": SHA256, "classification": "malicious"},
    ),
    (
        "add_user_tags",
        (SHA256, ["alpha"]),
        "post_user_tags",
        {"sample_hash": SHA256, "tags": ["alpha"]},
    ),
    (
        "remove_user_tags",
        (SHA256, ["alpha"]),
        "delete_user_tags",
        {"sample_hash": SHA256, "tags": ["alpha"]},
    ),
    (
        "create_dynamic_report",
        (SHA1, "html"),
        "create_dynamic_analysis_report",
        {"sample_hash": SHA1, "report_format": "html"},
    ),
    (
        "create_dynamic_report",
        (SHA1, "pdf"),
        "create_dynamic_analysis_report",
        {"sample_hash": SHA1, "report_format": "pdf"},
    ),
    (
        # Hash and format are both strings, and the status endpoint used to
        # be called with the format missing altogether.
        "check_dynamic_analysis_status",
        (SHA1, "html"),
        "check_dynamic_analysis_report_status",
        {"sample_hash": SHA1, "report_format": "html"},
    ),
    (
        "submit_url",
        ("https://example.com", "cloud"),
        "submit_url",
        {"url_string": "https://example.com", "crawler": "cloud"},
    ),
    # ---- the rest of the wrappers, so that the table is the call list ----
    #
    # Thirteen rows guarded the wrappers somebody had already been bitten
    # by, and the other forty-one were checked for arity and nothing else:
    # ``delete_sample``, ``batch_delete_samples``, ``delete_classification``
    # and ``download_sample`` among them, which are the calls a transposed
    # pair does the most damage on. The completeness test below is what
    # keeps this list level with the call list from here on.
    (
        # The liveness probe is a real lookup: a fixed hash asked about
        # with a status filter, two arguments the SDK takes in the other
        # order than a reader expects.
        "test_connection",
        (),
        "file_analysis_status",
        {"sample_hashes": ["0" * 40], "sample_status": "processed"},
    ),
    (
        # Two booleans in a row, and the wrapper sets them opposite ways:
        # a swap reanalyses the sample on every report request and drops
        # the network intelligence the analyst asked for.
        "get_report",
        (SHA256,),
        "get_detailed_report_v2",
        {
            "sample_hashes": SHA256,
            "skip_reanalysis": True,
            "include_networkthreatintelligence": False,
        },
    ),
    ("get_report", (SHA256, "xml"), "get_titanium_core_report_v2", {"sample_hash": SHA256}),
    (
        # The PDF path creates, polls and downloads; what this pins is
        # that the hash reached the download, which is the call that
        # answers with the document.
        "get_report",
        (SHA256, "pdf"),
        "download_pdf_report",
        {"sample_hash": SHA256},
    ),
    ("get_summary_report_v2", (SHA256,), "get_summary_report_v2", {"sample_hashes": SHA256}),
    (
        "get_titanium_core_report_v2",
        (SHA256,),
        "get_titanium_core_report_v2",
        {"sample_hash": SHA256},
    ),
    (
        # The single-sample delete goes through the bulk endpoint with one
        # hash, so what stops it deleting a batch is the value's shape.
        "delete_sample",
        (SHA256,),
        "delete_samples",
        {"hash_input": SHA256},
    ),
    (
        "reanalyze_sample",
        (SHA256,),
        "reanalyze_samples_v2",
        {"hash_input": SHA256, "titanium_cloud": True, "titanium_core": True},
    ),
    (
        # The destination is ours: it is where the bytes are written, and
        # it must not reach the appliance as part of the request.
        "download_sample",
        (SHA256, SAMPLE_DESTINATION),
        "download_sample",
        {"sample_hash": SHA256},
    ),
    ("list_extracted_files", (SHA256,), "list_extracted_files_v2", {"sample_hash": SHA256}),
    (
        "download_extracted_files",
        (SHA256, OUTPUT_DIRECTORY),
        "download_extracted_files",
        {"sample_hash": SHA256},
    ),
    (
        # Path and comment are two strings in a row, and a swap submits
        # the analyst's note as the file to analyse.
        "upload_file",
        (Path(__file__),),
        "submit_file_from_path",
        {"file_path": str(Path(__file__)), "comment": None},
    ),
    (
        "upload_file",
        (Path(__file__), "triage batch 7"),
        "submit_file_from_path",
        {"file_path": str(Path(__file__)), "comment": "triage batch 7"},
    ),
    (
        "download_dynamic_report",
        (SHA1, "pdf"),
        "download_dynamic_analysis_report",
        {"sample_hash": SHA1, "report_format": "pdf"},
    ),
    (
        "list_samples",
        (),
        "advanced_search_v3",
        {"query_string": "available:true", "page_number": 1, "records_per_page": 100},
    ),
    (
        # Page number and page size are both ints, and a swap asks for
        # page 100 of one record each.
        "advanced_search",
        ("classification:malicious",),
        "advanced_search_v3",
        {"query_string": "classification:malicious", "page_number": 1, "records_per_page": 100},
    ),
    (
        "get_url_report",
        ("task-id",),
        "get_submitted_url_report",
        {"task_id": "task-id", "retry": False},
    ),
    ("url_status", ("task-id",), "check_submitted_url_status", {"task_id": "task-id"}),
    # The seven network pivots: each names its subject differently, and
    # each reaches an SDK method whose name says the same thing twice.
    ("get_domain_report", (DOMAIN,), "network_domain_report", {"domain": DOMAIN}),
    ("get_ip_report", (IP,), "network_ip_addr_report", {"ip_addr": IP}),
    ("get_files_from_ip", (IP,), "network_files_from_ip", {"ip_addr": IP}),
    ("get_domains_from_ip", (IP,), "network_ip_to_domain", {"ip_addr": IP}),
    ("get_urls_from_ip", (IP,), "network_urls_from_ip", {"ip_addr": IP}),
    (
        "get_files_from_ip_aggregated",
        (IP,),
        "network_files_from_ip",
        {"ip_addr": IP},
    ),
    (
        "get_domains_from_ip_aggregated",
        (IP,),
        "network_ip_to_domain",
        {"ip_addr": IP},
    ),
    (
        "get_urls_from_ip_aggregated",
        (IP,),
        "network_urls_from_ip",
        {"ip_addr": IP},
    ),
    ("get_network_url_report", (URL,), "network_url_report", {"requested_url": URL}),
    (
        "list_yara_rulesets",
        (),
        "get_yara_rulesets_on_the_appliance_v2",
        {"owner_type": "all"},
    ),
    ("delete_yara_ruleset", ("myrules",), "delete_yara_ruleset", {"name": "myrules"}),
    ("get_yara_content", ("myrules",), "get_yara_ruleset_contents", {"ruleset_name": "myrules"}),
    (
        "yara_local_retro_scan",
        ("start",),
        "start_or_stop_yara_local_retro_scan",
        {"operation": "START"},
    ),
    (
        "get_yara_cloud_retro_scan_status",
        ("myrules",),
        "get_yara_cloud_retro_scan_status",
        {"ruleset_name": "myrules"},
    ),
    ("get_yara_local_retro_scan_status", (), "get_yara_local_retro_scan_status", {}),
    (
        "list_yara_repositories",
        (),
        "get_yara_repositories",
        {"active_filter": None, "page_size": None, "page_number": None},
    ),
    (
        "list_yara_repositories_aggregated",
        (),
        "get_yara_repositories",
        {"active_filter": None, "page_size": 100, "page_number": 1},
    ),
    (
        # Four same-typed strings written as one unit, which is the whole
        # reason ``YaraRepositorySpec`` exists — and the fifth field is
        # translated on the way, so its landing place is a claim too.
        "create_yara_repository",
        (REPOSITORY_SPEC,),
        "create_yara_repository",
        {
            "repository_url": "http://repo",
            "name": "name",
            "source_branch": "main",
            "api_token": "",
            "import_update_preferences": 0,
        },
    ),
    (
        # The same five, plus the id of the repository they overwrite: an
        # update replaces every field, so a transposed pair here is a
        # repository silently pointed somewhere else.
        "update_yara_repository",
        (1, REPOSITORY_SPEC),
        "update_yara_repository",
        {
            "repository_id": 1,
            "repository_url": "http://repo",
            "name": "name",
            "source_branch": "main",
            "api_token": "",
            "import_update_preferences": 0,
        },
    ),
    (
        # ``remove_rulesets`` decides whether the rulesets the repository
        # imported go with it, so it is the difference between deleting a
        # link and deleting the hunt.
        "delete_yara_repository",
        (1,),
        "delete_yara_repository",
        {"repository_id": 1, "remove_rulesets": False},
    ),
    (
        "publish_yara_ruleset",
        ("myrules",),
        "publish_single_yara_ruleset",
        {"ruleset_name": "myrules"},
    ),
    ("publish_all_yara_rulesets", (), "publish_all_yara_rulesets", {}),
    ("run_yara_update", (), "run_yara_update", {}),
    ("set_yara_update_interval", (3600,), "set_yara_update_interval", {"seconds": 3600}),
    ("reset_yara_update_interval", (), "reset_yara_update_interval", {}),
    ("get_user_tags", (SHA256,), "get_user_tags", {"sample_hash": SHA256}),
    ("get_classification", (SHA256,), "get_classification_v3", {"sample_hash": SHA256}),
    (
        # Hash and system are two strings in a row, and the system decides
        # whether a local verdict or the TitaniumCloud one is discarded.
        "delete_classification",
        (SHA256,),
        "delete_classification",
        {"sample_hash": SHA256, "system": "local"},
    ),
    (
        "delete_classification",
        (SHA256, "ticloud"),
        "delete_classification",
        {"sample_hash": SHA256, "system": "ticloud"},
    ),
    ("list_containers", (SHA256,), "list_containers_for_hashes", {"sample_hashes": [SHA256]}),
    (
        "batch_reanalyze_samples",
        ([SHA256, SHA1],),
        "reanalyze_samples_v2",
        {"hash_input": [SHA256, SHA1], "titanium_cloud": True, "titanium_core": True},
    ),
    (
        # The whole batch reaches ``hash_input`` as one list: the removal
        # submitted is the one asked for, and no other.
        "batch_delete_samples",
        ([SHA256, SHA1],),
        "delete_samples",
        {"hash_input": [SHA256, SHA1]},
    ),
]


class TestArgumentsLandOnTheIntendedParameters:
    def test_every_wrapper_that_is_called_is_also_bound(self):
        """The binding table is the call table, not a subset somebody kept up.

        There was a completeness test over the calls and none over the
        bindings, so a new wrapper satisfied the gate with an arity row
        alone — and arity is what ``create_autospec`` already checks.
        Forty-one of the fifty-four wrappers were in that position,
        ``delete_sample`` and ``batch_delete_samples`` among them.

        Equality, and no exempt set: a wrapper that reaches the SDK at all
        sends it values, and where those values land is the claim this
        file exists to make. A wrapper that sends none still sends the
        constants the wrapper itself chose — ``test_connection``'s
        ``sample_status``, ``list_yara_rulesets``'s ``owner_type`` — which
        are the same claim about a value nobody passed in.
        """
        called = {name for name, _ in A1000_WRAPPER_CALLS}
        bound = {row[0] for row in A1000_ARGUMENT_BINDINGS}

        assert called == bound, (
            f"called with no binding: {sorted(called - bound)}; "
            f"bound but never called: {sorted(bound - called)}"
        )

    @pytest.mark.parametrize(
        "wrapper,args,sdk_method,expected",
        A1000_ARGUMENT_BINDINGS,
        ids=[f"{row[0]}-{index}" for index, row in enumerate(A1000_ARGUMENT_BINDINGS)],
    )
    def test_binding(self, build_service, tmp_path, wrapper, args, sdk_method, expected):
        service = build_service(owning_service(wrapper))
        getattr(service, wrapper)(*_resolved(args, tmp_path))

        recorded = getattr(service.client, sdk_method)
        assert recorded.call_count == 1, f"{sdk_method} was not called exactly once"

        signature = inspect.signature(getattr(A1000, sdk_method))
        bound = signature.bind(None, *recorded.call_args.args, **recorded.call_args.kwargs)
        for parameter, value in expected.items():
            assert bound.arguments.get(parameter) == value, (
                f"{sdk_method}: {parameter} received "
                f"{bound.arguments.get(parameter)!r}, expected {value!r}"
            )


# (owning service, handle, SDK class, wrapper, wrapper args, sdk method,
#  expected parameter -> value binding).
#
# The same claim the A1000 table above makes, for the TitaniumCloud half:
# autospec validates arity and parameter names, never which value landed on
# which parameter. It matters more here, because every network pivot reaches
# its endpoint through ``_endpoint(api, method)`` — the handle and the SDK
# method are *strings*, and the subject is forwarded positionally — so the
# only thing that can state "this pivot asks TCA-0403 about a URL, with
# ``private`` on ``private`` and not on the boolean next to it" is a
# binding assertion.
TICLOUD_ARGUMENT_BINDINGS = [
    (
        # The failure the whole file is for: ``uri_input`` renamed, or a
        # URI landing on ``page_sha1``, which is a paging cursor and would
        # quietly answer the first page of nothing.
        TitaniumCloudNetworkService,
        "_uri_index",
        ticloud.URIIndex,
        "get_uri_index",
        (URL,),
        "get_uri_index",
        {"uri_input": URL, "classification": None, "page_sha1": None},
    ),
    (
        TitaniumCloudNetworkService,
        "_uri_index",
        ticloud.URIIndex,
        "get_uri_index_aggregated",
        (URL, 5),
        "get_uri_index",
        {"uri_input": URL, "classification": None, "page_sha1": None},
    ),
    (
        # ``private`` keeps the analyst's target out of the public feeds.
        # It sits among three booleans on this endpoint, so a swap onto
        # ``extended`` or ``last_analysis`` publishes the lookup while
        # still passing every arity check there is.
        TitaniumCloudNetworkService,
        "_url_threat",
        ticloud.URLThreatIntelligence,
        "get_files_from_url",
        (URL,),
        "get_downloaded_files",
        {
            "url_input": URL,
            "private": True,
            "extended": True,
            "last_analysis": False,
            "classification": None,
            "analysis_id": None,
        },
    ),
    (
        TitaniumCloudNetworkService,
        "_url_threat",
        ticloud.URLThreatIntelligence,
        "get_files_from_url_aggregated",
        (URL, 7),
        "get_downloaded_files",
        {"url_input": URL, "private": True, "page_string": None},
    ),
    (
        TitaniumCloudNetworkService,
        "_url_threat",
        ticloud.URLThreatIntelligence,
        "analyze_url",
        (URL,),
        "get_url_report",
        {"url_input": URL, "private": True},
    ),
    (
        # Four domain pivots share one handle and take their subject on
        # four differently-named parameters; only the binding says which.
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_domain_report",
        (DOMAIN,),
        "get_domain_report",
        {"domain_input": DOMAIN},
    ),
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_files_from_domain_aggregated",
        (DOMAIN, 3),
        "get_downloaded_files",
        {"domain": DOMAIN, "page_string": None},
    ),
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_ips_from_domain",
        (DOMAIN,),
        "domain_to_ip_resolutions",
        {"domain": DOMAIN, "page_string": None},
    ),
    (
        TitaniumCloudNetworkService,
        "_ip_threat",
        ticloud.IPThreatIntelligence,
        "get_ip_report",
        (IP,),
        "get_ip_report",
        {"ip_address_input": IP},
    ),
    (
        TitaniumCloudNetworkService,
        "_ip_threat",
        ticloud.IPThreatIntelligence,
        "get_domains_from_ip_aggregated",
        (IP, 4),
        "ip_to_domain_resolutions",
        {"ip_address": IP, "page_string": None},
    ),
    (
        # Two booleans in a row: extended results and hashes in results.
        TitaniumCloudService,
        "_file_reputation",
        ticloud.FileReputation,
        "get_file_reputation",
        (SHA256,),
        "get_file_reputation",
        {"hash_input": SHA256, "extended_results": True, "show_hashes_in_results": True},
    ),
    (
        # The whole batch reaches ``hash_input`` as one list, which is what
        # makes it one metered request instead of N.
        TitaniumCloudService,
        "_file_reputation",
        ticloud.FileReputation,
        "get_bulk_file_reputation",
        ([SHA256, SHA256_OTHER],),
        "get_file_reputation",
        {"hash_input": [SHA256, SHA256_OTHER], "extended_results": True},
    ),
    (
        TitaniumCloudService,
        "_av_scanners",
        ticloud.AVScanners,
        "get_av_scanners",
        (SHA256,),
        "get_scan_results",
        {"hash_input": SHA256, "historical_results": False},
    ),
    (
        # Two strings in a row, and a swap sends the path as the name.
        TitaniumCloudService,
        "_file_upload",
        ticloud.FileUpload,
        "upload_file",
        (Path(__file__),),
        "upload_sample_from_path",
        {"file_path": str(Path(__file__)), "sample_name": Path(__file__).name},
    ),
    (
        TitaniumCloudService,
        "_file_analysis",
        ticloud.FileAnalysis,
        "get_file_analysis",
        (SHA256,),
        "get_analysis_results",
        {"hash_input": SHA256},
    ),
    (
        # The page size is clamped into the endpoint's 1-10000 range, and
        # must land on ``records_per_page`` rather than on ``page_number``.
        TitaniumCloudService,
        "_advanced_search",
        ticloud.AdvancedSearch,
        "search_samples",
        ("classification:malicious", 50),
        "search",
        {"query_string": "classification:malicious", "records_per_page": 50, "page_number": 1},
    ),
    (
        TitaniumCloudService,
        "_advanced_search",
        ticloud.AdvancedSearch,
        "search_samples",
        ("classification:malicious", 0),
        "search",
        {"query_string": "classification:malicious", "records_per_page": 1},
    ),
    (
        TitaniumCloudService,
        "_file_download",
        ticloud.FileDownload,
        "get_download_status",
        (SHA256,),
        "get_download_status",
        {"hash_input": SHA256},
    ),
    (
        # The hash reaches the SDK; the destination is ours and must not.
        TitaniumCloudService,
        "_file_download",
        ticloud.FileDownload,
        "download_sample",
        (SHA256, SAMPLE_DESTINATION),
        "download_sample",
        {"hash_input": SHA256},
    ),
    (
        # One hash goes to the single endpoint, several to the bulk one.
        TitaniumCloudService,
        "_file_reputation",
        ticloud.FileReputation,
        "get_reputation_records",
        ([SHA256],),
        "get_file_reputation",
        {"hash_input": SHA256},
    ),
    (
        TitaniumCloudService,
        "_file_reputation",
        ticloud.FileReputation,
        "get_reputation_records",
        ([SHA256, SHA256_OTHER],),
        "get_file_reputation",
        {"hash_input": [SHA256, SHA256_OTHER]},
    ),
    # The remaining pivots take their subject and nothing else, so what
    # these rows are for is the pair of strings the collapse introduced:
    # that this pivot reached *this* handle — ``get_downloaded_files``
    # exists on the domain and the IP class alike — and that the subject
    # landed on the subject parameter rather than on the paging cursor
    # next to it, which would answer page one of some other query.
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_files_from_domain",
        (DOMAIN,),
        "get_downloaded_files",
        {"domain": DOMAIN, "extended": True, "classification": None, "page_string": None},
    ),
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_urls_from_domain",
        (DOMAIN,),
        "urls_from_domain",
        {"domain": DOMAIN, "page_string": None},
    ),
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_urls_from_domain_aggregated",
        (DOMAIN, 6),
        "urls_from_domain",
        {"domain": DOMAIN, "page_string": None},
    ),
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_ips_from_domain_aggregated",
        (DOMAIN, 6),
        "domain_to_ip_resolutions",
        {"domain": DOMAIN, "page_string": None},
    ),
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_related_domains",
        (DOMAIN,),
        "related_domains",
        {"domain": DOMAIN, "page_string": None},
    ),
    (
        TitaniumCloudNetworkService,
        "_domain_threat",
        ticloud.DomainThreatIntelligence,
        "get_related_domains_aggregated",
        (DOMAIN, 6),
        "related_domains",
        {"domain": DOMAIN, "page_string": None},
    ),
    (
        TitaniumCloudNetworkService,
        "_ip_threat",
        ticloud.IPThreatIntelligence,
        "get_files_from_ip",
        (IP,),
        "get_downloaded_files",
        {"ip_address": IP, "extended": True, "classification": None, "page_string": None},
    ),
    (
        TitaniumCloudNetworkService,
        "_ip_threat",
        ticloud.IPThreatIntelligence,
        "get_files_from_ip_aggregated",
        (IP, 6),
        "get_downloaded_files",
        {"ip_address": IP, "page_string": None},
    ),
    (
        TitaniumCloudNetworkService,
        "_ip_threat",
        ticloud.IPThreatIntelligence,
        "get_urls_from_ip",
        (IP,),
        "urls_from_ip",
        {"ip_address": IP, "page_string": None},
    ),
    (
        TitaniumCloudNetworkService,
        "_ip_threat",
        ticloud.IPThreatIntelligence,
        "get_urls_from_ip_aggregated",
        (IP, 6),
        "urls_from_ip",
        {"ip_address": IP, "page_string": None},
    ),
    (
        TitaniumCloudNetworkService,
        "_ip_threat",
        ticloud.IPThreatIntelligence,
        "get_domains_from_ip",
        (IP,),
        "ip_to_domain_resolutions",
        {"ip_address": IP, "page_string": None},
    ),
]


class TestTiCloudArgumentsLandOnTheIntendedParameters:
    def test_every_wrapper_that_is_called_is_also_bound(self):
        """The same completeness claim, for the half that already held it.

        This half was level by hand — thirty wrappers called, the same
        thirty bound — with nothing saying so, which is one edit from the
        gap the A1000 table had.
        """
        called = {row[3] for row in TICLOUD_WRAPPER_CALLS}
        bound = {row[3] for row in TICLOUD_ARGUMENT_BINDINGS}

        assert called == bound, (
            f"called with no binding: {sorted(called - bound)}; "
            f"bound but never called: {sorted(bound - called)}"
        )

    @pytest.mark.parametrize(
        "service_cls,handle,sdk_class,wrapper,args,sdk_method,expected",
        TICLOUD_ARGUMENT_BINDINGS,
        ids=[f"{row[3]}-{index}" for index, row in enumerate(TICLOUD_ARGUMENT_BINDINGS)],
    )
    def test_binding(
        self, tmp_path, service_cls, handle, sdk_class, wrapper, args, sdk_method, expected
    ):
        settings = Settings(cache_dir=tmp_path / "cache", config_dir=tmp_path / "config")
        service = service_cls(settings, output=MagicMock())
        api = _autospec_client(sdk_class)
        service.__dict__[handle] = api

        getattr(service, wrapper)(*_resolved(args, tmp_path))

        recorded = getattr(api, sdk_method)
        assert recorded.call_count == 1, f"{sdk_method} was not called exactly once"

        signature = inspect.signature(getattr(sdk_class, sdk_method))
        bound = signature.bind(None, *recorded.call_args.args, **recorded.call_args.kwargs)
        bound.apply_defaults()
        for parameter, value in expected.items():
            assert bound.arguments.get(parameter) == value, (
                f"{sdk_method}: {parameter} received "
                f"{bound.arguments.get(parameter)!r}, expected {value!r}"
            )
