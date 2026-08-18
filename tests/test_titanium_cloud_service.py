"""Tests for the two TitaniumCloud services' response shaping and validation."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import MagicMock

import pytest
from ReversingLabs.SDK import ticloud

from rl_cli.config import Settings
from rl_cli.services.a1000 import A1000NetworkService, A1000Session
from rl_cli.services.titanium_cloud import TitaniumCloudNetworkService, TitaniumCloudService
from rl_cli.services.titanium_cloud.api import TitaniumCloudApi
from rl_cli.services.titanium_cloud.network import _MAX_PIVOT_PAGES, _RECORDS_PER_PAGE
from rl_cli.services.titanium_cloud.service import _MAX_SEARCH_PAGES
from tests.conftest import sdk_response

SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# The SDK class behind each cached API handle, and the service that builds
# it: six families are the file-oriented half's and four are the network
# half's, which is the split itself. Every stand-in below is built with the
# matching class as its ``spec``, which is what makes a wrapper calling a
# method the real class does not have an AttributeError in the suite rather
# than in front of an analyst: a bare ``MagicMock()`` answers any name at
# all, so a wrapper and its test could agree on a method that does not
# exist and stay green forever.
#
# ``test_every_handle_is_specced_against_the_class_it_builds`` below keeps
# this table honest against the services themselves.
_FILE_SDK_CLASS = {
    "_file_reputation": ticloud.FileReputation,
    "_av_scanners": ticloud.AVScanners,
    "_file_upload": ticloud.FileUpload,
    "_file_analysis": ticloud.FileAnalysis,
    "_advanced_search": ticloud.AdvancedSearch,
    "_file_download": ticloud.FileDownload,
}
_NETWORK_SDK_CLASS = {
    "_url_threat": ticloud.URLThreatIntelligence,
    "_domain_threat": ticloud.DomainThreatIntelligence,
    "_ip_threat": ticloud.IPThreatIntelligence,
    "_uri_index": ticloud.URIIndex,
}
_SDK_CLASS = _FILE_SDK_CLASS | _NETWORK_SDK_CLASS


def sdk_api(attribute: str) -> MagicMock:
    """A stand-in for one SDK handle that only answers to that class's names."""
    return MagicMock(spec=_SDK_CLASS[attribute])


def _api_for(service, attribute) -> MagicMock:
    """Put that stand-in where the service's cached handle would be."""
    api = sdk_api(attribute)
    service.__dict__[attribute] = api
    return api


def _settings(tmp_path: Path) -> Settings:
    """What either half is built from, proxy and credentials included.

    Shared rather than set in the one test that reads it back: the proxy
    reaches every SDK handle through ``_api_config`` on the base both
    halves inherit, so it is a property of the credentials and not of a
    single service.

    A profile with credentials in it, because that is the state every
    claim below is about — what a service does with an answer. A profile
    with none of them builds no SDK handle at all
    (``test_no_credential_means_no_client_is_built_at_all``), so a fixture
    without them would test the refusal over and over under other names.
    """
    settings = Settings(cache_dir=tmp_path / "cache", config_dir=tmp_path / "config")
    settings.titanium_cloud.proxy = "http://proxy:8080"
    settings.titanium_cloud.username = "analyst"
    settings.titanium_cloud.password = "s3cret"
    return settings


# Both fixtures are handed a recorder at construction rather than having
# ``output`` replaced afterwards, because ``safe_call`` and the guards read
# ``self.output`` from the first call onwards.
@pytest.fixture
def service(tmp_path) -> TitaniumCloudService:
    """The file-oriented half: hashes, samples and Advanced Search."""
    return TitaniumCloudService(_settings(tmp_path), output=MagicMock())


@pytest.fixture
def network(tmp_path) -> TitaniumCloudNetworkService:
    """The network half, built by name now that nothing forwards to it."""
    return TitaniumCloudNetworkService(_settings(tmp_path), output=MagicMock())


class TestTheStandInsMatchTheRealSdk:
    """What makes every other test in this file able to fail."""

    @pytest.mark.parametrize("attribute,api_class", sorted(_SDK_CLASS.items()))
    def test_every_handle_is_specced_against_the_class_it_builds(
        self, service, network, monkeypatch, attribute, api_class
    ):
        """The table above has to name the class the service actually builds.

        Specced against the wrong sibling, the stand-in would allow the
        wrong method names just as freely as a bare mock did — the guard
        would look present and check nothing.

        Patched on the base both halves inherit ``_api`` from: patching it
        on ``TitaniumCloudService`` alone would leave the four network
        handles building real SDK objects.
        """
        built: list[type] = []

        def record_api(self: TitaniumCloudApi, cls: type) -> type:
            built.append(cls)
            return cls

        monkeypatch.setattr(TitaniumCloudApi, "_api", record_api)

        owner = network if attribute in _NETWORK_SDK_CLASS else service
        getattr(owner, attribute)

        assert built == [api_class]

    def test_a_method_the_sdk_class_does_not_have_is_refused(self):
        """The property the bare ``MagicMock()`` these replaced did not have."""
        api = sdk_api("_file_reputation")

        assert api.get_file_reputation is not None
        with pytest.raises(AttributeError):
            api.get_file_reputashun(hash_input=SHA256)


class TestApiConfig:
    def test_proxy_expands_to_both_schemes(self, service):
        proxies = service._api_config["proxies"]
        assert proxies == {"http": "http://proxy:8080", "https": "http://proxy:8080"}

    @pytest.mark.parametrize(
        ("field", "stand_in"),
        [("username", "your_ticloud_username"), ("password", "your_ticloud_password")],
    )
    def test_a_stand_in_is_not_what_the_handles_authenticate_with(self, service, field, stand_in):
        """The probe refuses this string and caches the refusal for 24 h.

        Passed through, every ``ticloud`` command in that day authenticates
        with the example's own value while ``check-access`` reports the
        service unconfigured — two readings of one config, kept in step by
        nothing.
        """
        setattr(service.ti_cloud_settings, field, stand_in)

        assert service._api_config[field] is None

    @pytest.mark.parametrize("padded", ["your_ticloud_username ", " your_ticloud_username"])
    def test_a_stand_in_padded_with_whitespace_is_refused_too(self, service, padded):
        service.ti_cloud_settings.username = padded

        assert service._api_config["username"] is None

    def test_a_credential_the_user_set_still_reaches_the_sdk(self, service):
        service.ti_cloud_settings.username = "analyst"
        service.ti_cloud_settings.password = "s3cret"

        assert service._api_config["username"] == "analyst"
        assert service._api_config["password"] == "s3cret"

    def test_a_credential_the_user_set_reaches_the_handle_that_is_built(self, service, monkeypatch):
        """The refusal below must not stand in front of a configured service."""
        service.ti_cloud_settings.username = "analyst"
        service.ti_cloud_settings.password = "s3cret"
        factory = MagicMock(return_value=sdk_api("_file_reputation"))
        monkeypatch.setattr(
            "rl_cli.services.titanium_cloud.service.ticloud.FileReputation", factory
        )

        service.get_file_reputation(SHA256)

        assert factory.call_args.kwargs["username"] == "analyst"
        assert factory.call_args.kwargs["password"] == "s3cret"

    @pytest.mark.parametrize(
        ("username", "password"),
        [
            (None, None),
            ("your_ticloud_username", "your_ticloud_password"),
            ("analyst", None),
            (None, "s3cret"),
        ],
        ids=["unset", "stand-ins", "no-password", "no-username"],
    )
    def test_no_credential_means_no_client_is_built_at_all(
        self, service, monkeypatch, username, password
    ):
        """A credential that was never supplied is not one to send as "None".

        ``supplied_credential`` turns a stand-in into ``None`` and the SDK
        takes the pair as it is given: ``session.auth = (None, None)``,
        which requests sends as ``Basic Tm9uZTpOb25l`` — the literal string
        "None" as username and password — after a ``DeprecationWarning``
        about non-string usernames. So an unconfigured profile spent a
        request per command to be told 401, and the analyst was told their
        credentials were rejected rather than that there were none.

        The A1000 refuses the same state before it opens anything
        (``WrongInputError`` out of the SDK's own constructor: "If token is
        not provided username and password are required"), and this is that
        refusal on the other side.
        """
        service.ti_cloud_settings.username = username
        service.ti_cloud_settings.password = password
        factory = MagicMock()
        monkeypatch.setattr(
            "rl_cli.services.titanium_cloud.service.ticloud.FileReputation", factory
        )

        assert service.get_file_reputation(SHA256) is None
        assert not factory.called
        assert "credentials" in service.output.error.call_args.args[0]

    def test_the_sdk_arguments_are_not_public_to_a_formatter(self, service):
        """It carries the password and ``http://user:pass@proxy:8080``.

        A public name is one an output formatter can be handed, and
        anything that renders this dict prints the credentials to stdout.
        """
        assert not hasattr(service, "api_config")


SHA512 = "b" * 128

# (wrapper, cached API attribute, SDK method) for every hash endpoint.
_HASH_WRAPPERS = [
    ("get_file_reputation", "_file_reputation", "get_file_reputation"),
    ("get_av_scanners", "_av_scanners", "get_scan_results"),
    ("get_file_analysis", "_file_analysis", "get_analysis_results"),
]


class TestValidationBoundary:
    def test_invalid_hash_rejected(self, service):
        assert service.get_file_reputation("nothash") is None
        assert service.get_av_scanners("zzz") is None

    def test_invalid_url_rejected(self, network):
        assert network.analyze_url("not a url") is None

    def test_missing_file_rejected(self, service, tmp_path):
        assert service.upload_file(tmp_path / "missing.bin") is None


class TestFailedStatusIsReported:
    """A failure must carry the server's explanation, not collapse to None."""

    def test_error_status_surfaces_server_message(self, service):
        api = _api_for(service, "_file_reputation")
        api.get_file_reputation.return_value = sdk_response(429, {"message": "Quota exceeded"})

        assert service.get_file_reputation(SHA256) is None

        reported = service.output.error.call_args[0][0]
        assert "HTTP 429" in reported
        assert "Quota exceeded" in reported


class TestHashTypesTheseEndpointsAccept:
    """MD5/SHA1/SHA256 only, and sent as the value the CLI validated."""

    @pytest.mark.parametrize("wrapper,attribute,sdk_method", _HASH_WRAPPERS)
    def test_the_normalised_hash_is_what_reaches_the_sdk(
        self, service, wrapper, attribute, sdk_method
    ):
        """A hash pasted with a space passed the CLI check, then the SDK
        answered "not a valid hexadecimal value" for the raw string."""
        api = _api_for(service, attribute)
        getattr(api, sdk_method).return_value = sdk_response(json_payload={"rl": {}})

        getattr(service, wrapper)(f"  {SHA256.upper()}\n")

        assert getattr(api, sdk_method).call_args.kwargs["hash_input"] == SHA256

    @pytest.mark.parametrize("wrapper,attribute,sdk_method", _HASH_WRAPPERS)
    def test_sha512_is_refused_with_the_supported_types_named(
        self, service, wrapper, attribute, sdk_method
    ):
        api = _api_for(service, attribute)

        assert getattr(service, wrapper)(SHA512) is None

        assert not getattr(api, sdk_method).called
        reported = service.output.error.call_args[0][0]
        assert "MD5" in reported and "SHA1" in reported and "SHA256" in reported


class TestTheRequestSpellsOutWhatItAsksFor:
    """What was *asked for*, not just which method was called.

    Two mutations of these keywords survived the entire suite: flipping
    ``historical_results`` to True, and ``extended_results`` and
    ``show_hashes_in_results`` to False at both reputation call sites.
    Each of them changes the record every caller then renders — the
    hashes disappear from a bulk answer, the AV section grows every scan
    ever run — and nothing failed. The whole kwargs dict is asserted, so
    a keyword added or dropped fails here too.
    """

    def test_the_single_reputation_query_asks_for_the_extended_record(self, service):
        api = _api_for(service, "_file_reputation")
        api.get_file_reputation.return_value = sdk_response(json_payload={"rl": {}})

        service.get_file_reputation(SHA256)

        assert api.get_file_reputation.call_args.kwargs == {
            "hash_input": SHA256,
            "extended_results": True,
            "show_hashes_in_results": True,
        }

    def test_the_bulk_reputation_query_asks_for_the_same_record(self, service):
        """Or a batch is graded on less than the same hash queried alone."""
        api = _api_for(service, "_file_reputation")
        api.get_file_reputation.return_value = sdk_response(json_payload={"rl": {"entries": []}})

        service.get_bulk_file_reputation([SHA256])

        assert api.get_file_reputation.call_args.kwargs == {
            "hash_input": [SHA256],
            "extended_results": True,
            "show_hashes_in_results": True,
        }

    def test_the_av_query_asks_for_current_results_not_every_scan_ever_run(self, service):
        api = _api_for(service, "_av_scanners")
        api.get_scan_results.return_value = sdk_response(json_payload={"rl": {}})

        service.get_av_scanners(SHA256)

        assert api.get_scan_results.call_args.kwargs == {
            "hash_input": SHA256,
            "historical_results": False,
        }

    def test_the_analysis_query_asks_by_hash_and_nothing_else(self, service):
        api = _api_for(service, "_file_analysis")
        api.get_analysis_results.return_value = sdk_response(json_payload={"rl": {}})

        service.get_file_analysis(SHA256)

        assert api.get_analysis_results.call_args.kwargs == {"hash_input": SHA256}


class TestUploadReportsTheSpexAnswer:
    """The upload endpoint is XML and answers 2xx with no JSON body."""

    def _upload(self, service, tmp_path, response):
        sample = tmp_path / "sample.bin"
        sample.write_bytes(b"payload")
        api = _api_for(service, "_file_upload")
        api.upload_sample_from_path.return_value = response
        return service.upload_file(sample)

    def test_an_accepted_upload_with_no_body_is_a_success(self, service, tmp_path):
        """Reading it as JSON turned an accepted upload into "Failed to
        upload file" and an exit code of 1."""
        response = MagicMock()
        response.status_code = 200
        response.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")

        result = self._upload(service, tmp_path, response)

        assert result == {"file_name": "sample.bin", "status": "queued for analysis"}
        assert not service.output.error.called

    def test_a_rejected_upload_still_carries_the_server_message(self, service, tmp_path):
        response = sdk_response(403, {"message": "Quota exceeded"})

        assert self._upload(service, tmp_path, response) is None

        reported = service.output.error.call_args[0][0]
        assert "HTTP 403" in reported and "Quota exceeded" in reported


class TestSearchSamples:
    def test_hash_query_short_circuits_to_reputation(self, service, monkeypatch):
        monkeypatch.setattr(
            TitaniumCloudService, "get_file_reputation", lambda self, h: {"sha256": h}
        )
        assert service.search_samples(SHA256) == [{"sha256": SHA256}]

    def test_the_reputation_envelope_is_unwrapped_like_every_other_entry(
        self, service, monkeypatch
    ):
        """Search entries are flat records; handing back the envelope in
        their place graded a known-malicious sample "none" in SARIF."""
        record = {"status": "MALICIOUS", "threat_name": "Win32.Trojan.Emotet"}
        monkeypatch.setattr(
            TitaniumCloudService,
            "get_file_reputation",
            lambda self, h: {"rl": {"malware_presence": record}},
        )
        assert service.search_samples(SHA256) == [record]

    def _search_api(self, service, monkeypatch, payload) -> MagicMock:
        api = sdk_api("_advanced_search")
        api.search.return_value = sdk_response(json_payload=payload)
        monkeypatch.setattr(
            "rl_cli.services.titanium_cloud.service.ticloud.AdvancedSearch", lambda **kw: api
        )
        return api

    @pytest.mark.parametrize("entries", [[{"a": 1}, {"b": 2}, {"c": 3}], []])
    def test_unwraps_the_advanced_search_envelope(self, service, monkeypatch, entries):
        """Advanced Search answers rl.web_search_api.entries, and only that.

        Replaces a parametrization that pinned the bug: it fed
        ``{"rl": {"entries": ...}}`` — a shape the API never returns — and
        asserted that an unrecognised payload came back as ``[payload]``,
        which is exactly how every real search reported one "sample".
        """
        self._search_api(service, monkeypatch, {"rl": {"web_search_api": {"entries": entries}}})
        assert service.search_samples("threat_name:Evil") == entries

    @pytest.mark.parametrize("payload", [{"unshaped": True}, {"rl": {"entries": [{"a": 1}]}}, []])
    def test_unreadable_envelope_is_a_failure_not_a_result(self, service, monkeypatch, payload):
        """An answer we cannot read is a failed search, as on the A1000 side."""
        self._search_api(service, monkeypatch, payload)

        assert service.search_samples("threat_name:Evil") is None
        assert service.output.error.called

    def test_the_search_handle_is_built_once_and_reused(self, service, monkeypatch):
        """Its five sibling API handles are cached; this one was not.

        Every search paid for a fresh ``ticloud.AdvancedSearch`` — and the
        timeout wrapper around its session with it.
        """
        api = sdk_api("_advanced_search")
        api.search.return_value = sdk_response(
            json_payload={"rl": {"web_search_api": {"entries": []}}}
        )
        built: list[dict[str, Any]] = []

        def record_construction(**kwargs: Any) -> Any:
            built.append(kwargs)
            return api

        monkeypatch.setattr(
            "rl_cli.services.titanium_cloud.service.ticloud.AdvancedSearch",
            record_construction,
        )

        service.search_samples("threat_name:Evil")
        service.search_samples("threat_name:Evil")

        assert len(built) == 1
        assert api.search.call_count == 2

    @pytest.mark.parametrize(
        "limit,expected", [(0, 1), (-5, 1), (1, 1), (100, 100), (10000, 10000)]
    )
    def test_a_limit_below_one_is_still_a_page_the_endpoint_takes(
        self, service, monkeypatch, limit, expected
    ):
        """``-l 0`` used to surface the SDK's own WrongInputError text."""
        api = self._search_api(service, monkeypatch, {"rl": {"web_search_api": {"entries": []}}})
        service.search_samples("threat_name:Evil", limit=limit)
        assert api.search.call_args.kwargs["records_per_page"] == expected

    def test_the_query_reaches_the_endpoint_with_nothing_else_asked_for(self, service, monkeypatch):
        api = self._search_api(service, monkeypatch, {"rl": {"web_search_api": {"entries": []}}})
        service.search_samples("threat_name:Evil")
        assert api.search.call_args.kwargs == {
            "query_string": "threat_name:Evil",
            "page_number": 1,
            "records_per_page": 100,
        }


class TestSearchSaysWhenItLeftHitsBehind:
    """A 20000-hit answer came back as "Found 10000 samples".

    ``--limit 20000`` was clamped to the endpoint's page size and the other
    half left in the corpus, with no warning and no info — the analyst
    reads a whole answer. The A1000 half already routes a limit above one
    page into the SDK's paged walk and records what the walk left behind;
    this half did neither.
    """

    def _search_api(self, service, monkeypatch, *, page=None, pages=None) -> MagicMock:
        """The endpoint, answering one page or a corpus of them.

        ``pages`` is the corpus a walk reads: page N of the answer is the
        Nth envelope given, and the last one answers anything past it — a
        walk that asks for a page the corpus does not have is reading past
        the end of the corpus, which is the walk's own bug and not
        something to hide behind an ``IndexError``.
        """
        answers = [page or {}] if pages is None else pages
        api = sdk_api("_advanced_search")

        def search(*, page_number: int = 1, **_kwargs: Any) -> MagicMock:
            envelope = answers[min(page_number, len(answers)) - 1]
            # A page stated as anything but an object is an answer no
            # reader can descend into, and is handed over as it stands.
            if isinstance(envelope, dict):
                envelope = {"entries": [], **envelope}
            return sdk_response(json_payload={"rl": {"web_search_api": envelope}})

        api.search.side_effect = search
        monkeypatch.setattr(
            "rl_cli.services.titanium_cloud.service.ticloud.AdvancedSearch", lambda **kw: api
        )
        return api

    @staticmethod
    def _full_page(more_pages: bool) -> dict[str, Any]:
        """One whole page of the endpoint's maximum, and what follows it."""
        return {"entries": [{"n": 1}] * 10000, "more_pages": more_pages}

    def test_a_limit_of_exactly_one_page_is_not_sent_to_the_walk(self, service, monkeypatch):
        """10000 is one page, so the routing is ``>``, not ``>=``: the single
        page is fetched once even when it comes back under-full and says more
        matched, rather than being handed to the multi-page walk."""
        half = {"entries": [{"n": 1}] * 5000, "more_pages": True}
        api = self._search_api(service, monkeypatch, pages=[half, half, half])

        results = service.search_samples("classification:malicious", 10000)

        assert results is not None
        assert len(results) == 5000
        assert api.search.call_count == 1

    def test_a_limit_above_one_page_is_walked_rather_than_cut_down(self, service, monkeypatch):
        api = self._search_api(
            service,
            monkeypatch,
            pages=[self._full_page(True), {"entries": [{"sha1": "a"}], "more_pages": False}],
        )

        results = service.search_samples("classification:malicious", 20000)

        assert results is not None
        assert len(results) == 10001
        assert results[-1] == {"sha1": "a"}
        assert api.search.call_count == 2
        # Every page of a walk is asked for at the endpoint's maximum: the
        # records are the same either way and the requests are metered.
        assert [call.kwargs["records_per_page"] for call in api.search.call_args_list] == [
            10000,
            10000,
        ]
        assert [call.kwargs["page_number"] for call in api.search.call_args_list] == [1, 2]

    def test_a_walk_that_came_back_at_its_cap_says_it_left_more(self, service, monkeypatch):
        self._search_api(service, monkeypatch, pages=[self._full_page(True)] * 2)

        service.search_samples("classification:malicious", 20000)

        assert service.pages_left_unfetched

    def test_a_walk_that_ran_out_of_hits_first_says_it_left_none(self, service, monkeypatch):
        self._search_api(service, monkeypatch, pages=[{"entries": [{"n": 1}]}])

        service.search_samples("classification:malicious", 20000)

        assert not service.pages_left_unfetched

    def test_a_corpus_of_exactly_the_limit_is_a_whole_answer(self, service, monkeypatch):
        """The walk measures what it left rather than reading the size of the answer.

        Reported from the size alone, an answer of exactly ``--limit`` is
        indistinguishable from one the cap cut short, so a complete answer
        was told to "raise --limit above 20000" — over-reporting on the
        argument the analyst is most likely to have chosen to fit.
        """
        self._search_api(
            service, monkeypatch, pages=[self._full_page(True), self._full_page(False)]
        )

        results = service.search_samples("classification:malicious", 20000)

        assert results is not None
        assert len(results) == 20000
        assert not service.pages_left_unfetched

    def test_records_the_walk_fetched_and_could_not_hand_over_are_left_behind(
        self, service, monkeypatch
    ):
        """A page overshoots the limit; the trimmed records are still missing hits."""
        self._search_api(
            service, monkeypatch, pages=[self._full_page(True), self._full_page(False)]
        )

        results = service.search_samples("classification:malicious", 15000)

        assert results is not None
        assert len(results) == 15000
        assert service.pages_left_unfetched

    def test_a_page_the_walk_cannot_read_is_a_failed_search(self, service, monkeypatch):
        """Not the end of the corpus: a half-walked answer must not read as whole."""
        self._search_api(service, monkeypatch, pages=[self._full_page(True), "not a page"])

        assert service.search_samples("classification:malicious", 20000) is None
        assert service.output.error.called

    def test_a_page_the_envelope_says_is_not_the_last_is_reported(self, service, monkeypatch):
        self._search_api(service, monkeypatch, page={"entries": [{"a": 1}], "more_pages": True})

        service.search_samples("classification:malicious", 100)

        assert service.pages_left_unfetched
        assert service.output.warning.called

    def test_the_last_page_leaves_nothing_behind(self, service, monkeypatch):
        self._search_api(service, monkeypatch, page={"entries": [{"a": 1}], "more_pages": False})

        service.search_samples("classification:malicious", 100)

        assert not service.pages_left_unfetched
        assert not service.output.warning.called

    def test_a_hash_query_measures_no_paging_at_all(self, service, monkeypatch):
        monkeypatch.setattr(
            TitaniumCloudService, "get_file_reputation", lambda self, h: {"sha256": h}
        )

        assert service.search_samples(SHA256) == [{"sha256": SHA256}]
        assert not service.pages_left_unfetched

    def test_the_verdict_does_not_outlive_the_call_that_measured_it(self, service, monkeypatch):
        """A later call that measured no paging must not inherit this one's ``True``."""
        self._search_api(service, monkeypatch, page={"entries": [{"a": 1}], "more_pages": True})
        service.search_samples("classification:malicious", 100)
        assert service.pages_left_unfetched

        service.supported_hash(SHA256)

        assert not service.pages_left_unfetched


class _CountingAdvancedSearch:
    """An Advanced Search endpoint that answers a page and counts the asking.

    Every page it hands out is a metered request, which is the whole
    subject of the tests below: what bounds them is not the corpus and not
    what the analyst typed, but the walk's own budget.

    It carries the SDK's real ``search_aggregated`` bound to itself, so a
    walk delegated to the SDK is measured by the same double as a walk this
    service does itself — the claim is about the walk, whoever runs it.
    """

    # A stand-in that let the walk run forever would hang the suite rather
    # than fail it. Well above any budget a walk may legitimately have.
    _RUNAWAY = 200

    search_aggregated = ticloud.AdvancedSearch.search_aggregated

    def __init__(self, page: dict[str, Any] | list[dict[str, Any]]):
        """One page answered to every request, or a page per request in turn."""
        self._page = page
        self.calls = 0

    def search(self, **_kwargs: Any) -> MagicMock:
        self.calls += 1
        if self.calls > self._RUNAWAY:
            raise AssertionError(f"the walk made {self.calls} requests and was still going")
        page = self._page[self.calls - 1] if isinstance(self._page, list) else self._page
        return sdk_response(json_payload={"rl": {"web_search_api": page}})


class TestAWalkThatStopsMakingProgressStops:
    """The walk had no stop of its own, and no bound on what it could spend.

    Against an endpoint answering ``{"entries": [], "more_pages": true}`` —
    a well-formed page promising another — neither of the two conditions
    the SDK's aggregator returns on ever comes true: ``len(results) >=
    max_results`` cannot, because no result is ever added, and
    ``more_pages`` stays set. The probe was still looping after 5001
    requests, each one metered.

    The cost is the other half. ``--limit`` had no ceiling, so
    ``--limit 1000000`` walked a corpus of 1000000 in 100 metered requests
    and ``--limit 100000000`` would have taken 10000, none of it confirmed
    and none of it capped.
    """

    def _endpoint(self, service, monkeypatch, page) -> _CountingAdvancedSearch:
        api = _CountingAdvancedSearch(page)
        monkeypatch.setattr(
            "rl_cli.services.titanium_cloud.service.ticloud.AdvancedSearch", lambda **kw: api
        )
        return api

    def test_an_endpoint_that_promises_pages_and_sends_none_does_not_loop(
        self, service, monkeypatch
    ):
        api = self._endpoint(service, monkeypatch, {"entries": [], "more_pages": True})

        results = service.search_samples("classification:malicious", 1_000_000)

        assert api.calls <= _MAX_SEARCH_PAGES, "the walk did not stop making requests"
        assert results == []

    def test_a_limit_past_the_corpus_costs_the_page_budget_and_no_more(self, service, monkeypatch):
        """A mistyped ``--limit`` must not buy requests by the thousand."""
        page = {"entries": [{"n": 1}] * 10000, "more_pages": True}
        api = self._endpoint(service, monkeypatch, page)

        results = service.search_samples("classification:malicious", 1_000_000)

        assert api.calls == _MAX_SEARCH_PAGES
        assert len(results) == _MAX_SEARCH_PAGES * 10000
        assert service.pages_left_unfetched

    def test_a_corpus_that_runs_out_first_stops_the_walk_there(self, service, monkeypatch):
        api = self._endpoint(service, monkeypatch, {"entries": [{"n": 1}], "more_pages": False})

        results = service.search_samples("classification:malicious", 1_000_000)

        assert api.calls == 1
        assert results == [{"n": 1}]
        assert not service.pages_left_unfetched


# (wrapper, cached API attribute, SDK method, the rl.* key the page is under,
# a subject the endpoint accepts, one it does not) for every network pivot.
_PIVOTS = [
    ("get_files_from_ip", "_ip_threat", "get_downloaded_files", "downloaded_files"),
    ("get_urls_from_ip", "_ip_threat", "urls_from_ip", "urls"),
    ("get_domains_from_ip", "_ip_threat", "ip_to_domain_resolutions", "resolutions"),
    ("get_files_from_domain", "_domain_threat", "get_downloaded_files", "downloaded_files"),
    ("get_urls_from_domain", "_domain_threat", "urls_from_domain", "urls"),
    ("get_ips_from_domain", "_domain_threat", "domain_to_ip_resolutions", "resolutions"),
    ("get_related_domains", "_domain_threat", "related_domains", "related_domains"),
    ("get_files_from_url", "_url_threat", "get_downloaded_files", "files"),
]

# What each pivot is asked about, and what it must refuse before asking.
_SUBJECTS = {
    "_ip_threat": ("1.2.3.4", "not-an-ip"),
    "_domain_threat": ("example.com", "not a domain"),
    "_url_threat": ("https://example.com/x", "example.com"),
    "_uri_index": ("https://example.com/x", "not a uri"),
}


class TestNetworkPivots:
    """The "what else is associated with this" lookups, and their guards."""

    @pytest.mark.parametrize("wrapper,attribute,sdk_method,key", _PIVOTS)
    def test_the_page_under_the_rl_envelope_is_what_comes_back(
        self, network, wrapper, attribute, sdk_method, key
    ):
        entries = [{"sha1": "a" * 40}, {"sha1": "b" * 40}]
        api = _api_for(network, attribute)
        getattr(api, sdk_method).return_value = sdk_response(
            json_payload={"rl": {key: entries, "next_page": "page2"}}
        )
        subject, _ = _SUBJECTS[attribute]

        assert getattr(network, wrapper)(subject) == entries

    @pytest.mark.parametrize("wrapper,attribute,sdk_method,key", _PIVOTS)
    def test_a_page_the_endpoint_left_empty_is_an_empty_result_not_a_failure(
        self, network, wrapper, attribute, sdk_method, key
    ):
        """The SDK's own aggregators read a missing key as an empty page."""
        api = _api_for(network, attribute)
        getattr(api, sdk_method).return_value = sdk_response(json_payload={"rl": {}})
        subject, _ = _SUBJECTS[attribute]

        assert getattr(network, wrapper)(subject) == []
        assert not network.output.error.called

    @pytest.mark.parametrize("wrapper,attribute,sdk_method,key", _PIVOTS)
    def test_a_subject_the_endpoint_would_refuse_never_reaches_it(
        self, network, wrapper, attribute, sdk_method, key
    ):
        api = _api_for(network, attribute)
        _, rejected = _SUBJECTS[attribute]

        assert getattr(network, wrapper)(rejected) is None

        assert not getattr(api, sdk_method).called
        assert rejected in network.output.error.call_args[0][0]

    @pytest.mark.parametrize("wrapper,attribute,sdk_method,key", _PIVOTS)
    def test_an_error_status_carries_the_servers_own_explanation(
        self, network, wrapper, attribute, sdk_method, key
    ):
        api = _api_for(network, attribute)
        getattr(api, sdk_method).return_value = sdk_response(429, {"message": "Quota exceeded"})
        subject, _ = _SUBJECTS[attribute]

        assert getattr(network, wrapper)(subject) is None

        reported = network.output.error.call_args[0][0]
        assert "HTTP 429" in reported and "Quota exceeded" in reported

    @pytest.mark.parametrize("wrapper,attribute,sdk_method,key", _PIVOTS)
    def test_an_answer_with_no_envelope_is_a_failed_lookup(
        self, network, wrapper, attribute, sdk_method, key
    ):
        """Not an empty page: an answer we cannot read is not "nothing found"."""
        api = _api_for(network, attribute)
        getattr(api, sdk_method).return_value = sdk_response(json_payload={"unshaped": True})
        subject, _ = _SUBJECTS[attribute]

        assert getattr(network, wrapper)(subject) is None
        assert "rl envelope" in network.output.error.call_args[0][0]

    @pytest.mark.parametrize("wrapper,attribute,sdk_method,key", _PIVOTS)
    def test_the_all_variant_walks_the_same_endpoint_a_page_at_a_time(
        self, network, wrapper, attribute, sdk_method, key
    ):
        """``--all`` asks the very endpoint the first page came from, again.

        The SDK's ``_aggregated`` sibling is what it used to hand the walk
        to, and that sibling cannot stop against an endpoint that promises
        a page and sends none.
        """
        every_page = [{"sha1": "c" * 40}]
        api = _api_for(network, attribute)
        getattr(api, sdk_method).return_value = sdk_response(json_payload={"rl": {key: every_page}})
        subject, _ = _SUBJECTS[attribute]

        assert getattr(network, f"{wrapper}_aggregated")(subject) == every_page
        assert not getattr(api, f"{sdk_method}_aggregated").called

    @pytest.mark.parametrize("wrapper,attribute,sdk_method,key", _PIVOTS)
    @pytest.mark.parametrize("aggregated", [False, True], ids=["page", "all"])
    def test_the_page_size_is_sent_rather_than_left_to_the_sdk(
        self, network, wrapper, attribute, sdk_method, key, aggregated
    ):
        """What ``_MAX_PIVOT_PAGES`` is worth in records depends on it.

        The SDK defaults it to 1000 and this walk stopped sending it, so
        the budget's arithmetic — 100 pages of 1000 is ``MAX_LIMIT`` —
        rested on a default in a vendored library.
        """
        api = _api_for(network, attribute)
        getattr(api, sdk_method).return_value = sdk_response(json_payload={"rl": {key: []}})
        subject, _ = _SUBJECTS[attribute]
        asked = f"{wrapper}_aggregated" if aggregated else wrapper

        getattr(network, asked)(subject)

        assert getattr(api, sdk_method).call_args.kwargs["results_per_page"] == _RECORDS_PER_PAGE

    @pytest.mark.parametrize("wrapper", ["get_uri_index", "get_uri_index_aggregated"])
    def test_the_one_endpoint_with_no_page_size_is_not_sent_one(self, network, wrapper):
        """TCA-0401 takes no ``results_per_page`` at all (SDK ticloud.py:1024).

        So its budget bounds requests and not records, and sending the
        keyword anyway would be a ``TypeError`` in front of an analyst.
        """
        api = _api_for(network, "_uri_index")
        api.get_uri_index.return_value = sdk_response(
            json_payload={"rl": {"uri_index": {"sha1_list": []}}}
        )

        getattr(network, wrapper)("https://example.com/x")

        assert "results_per_page" not in api.get_uri_index.call_args.kwargs

    @pytest.mark.parametrize("wrapper,attribute,sdk_method,key", _PIVOTS)
    def test_the_all_variant_guards_its_subject_too(
        self, network, wrapper, attribute, sdk_method, key
    ):
        api = _api_for(network, attribute)
        _, rejected = _SUBJECTS[attribute]

        assert getattr(network, f"{wrapper}_aggregated")(rejected) is None
        assert not getattr(api, sdk_method).called


class _Walk(NamedTuple):
    """One paging wrapper, and everything its walk has to get right.

    The endpoint it asks is the *non-aggregated* sibling, and the three
    names beside it differ per pivot: the keyword the cursor rides back
    on, where the page sits under the ``rl`` envelope and where the
    cursor for the next page sits. Read off the SDK aggregator each
    wrapper used to delegate to (SDK ticloud.py:1087, 1420, 1785, 1851,
    1910, 1970, 2136, 2203, 2263), which is the only statement of them
    there has ever been.
    """

    wrapper: str
    handle: str
    sdk_class: type
    sibling: str
    page: str
    path: tuple[str, ...]
    cursor: tuple[str, ...]
    aggregator: str

    def envelope(self, entries: list[Any], cursor: Any) -> dict[str, Any]:
        """One page as this pivot's endpoint states it.

        The page and the cursor share every step but the last — the ``rl``
        envelope itself for eight of the nine, ``rl.uri_index`` for
        TCA-0401 — which is what lets one builder state all nine.
        """
        assert self.path[:-1] == self.cursor[:-1], "these two are stated in the same place"
        node: dict[str, Any] = {self.path[-1]: entries, self.cursor[-1]: cursor}
        for step in reversed(self.path[:-1]):
            node = {step: node}
        return {"rl": node}


# Spelled out here rather than imported from the service, because a name
# the service got wrong is exactly what these tests are for.
_WALKS = [
    _Walk(
        f"{wrapper}_aggregated",
        attribute,
        _SDK_CLASS[attribute],
        sdk_method,
        "page_string",
        (key,),
        ("next_page",),
        f"{sdk_method}_aggregated",
    )
    for wrapper, attribute, sdk_method, key in _PIVOTS
] + [
    # TCA-0401 is the odd one: it pages by the first SHA-1 of the page it
    # wants, and states both the hashes and the cursor a level down.
    _Walk(
        "get_uri_index_aggregated",
        "_uri_index",
        ticloud.URIIndex,
        "get_uri_index",
        "page_sha1",
        ("uri_index", "sha1_list"),
        ("uri_index", "next_page_sha1"),
        "get_uri_index_aggregated",
    )
]


def _walk_ids(walk: _Walk) -> str:
    return walk.wrapper


def _subject(walk: _Walk) -> str:
    """A subject this pivot's guard accepts."""
    return _SUBJECTS[walk.handle][0]


def _record(number: int) -> Any:
    """One entry of whatever this pivot lists.

    A record for eight of the nine and a SHA-1 string for TCA-0401; the
    walk never looks inside either, so one distinguishable value stands
    for both.
    """
    return {"n": number}


def _pages_of(walk: _Walk, size: int, *, last: int | None) -> Callable[[int], Any]:
    """An endpoint answering ``size`` records a page, ending after ``last``.

    ``last=None`` never stops naming a cursor, which is the corpus larger
    than any budget: what bounds that walk is the walk's own.
    """

    def page(call: int) -> Any:
        return walk.envelope(
            [_record(call * 10 + n) for n in range(1, size + 1)],
            None if call == last else f"cursor-{call + 1}",
        )

    return page


class _CountingPivot:
    """A pivot endpoint that answers one page and counts the asking.

    Every page is a metered request, which is the whole subject of the
    tests here: what bounds a walk is not the corpus and not what the
    analyst typed but the walk's own budget.

    It carries the real SDK aggregator bound to itself, so a walk handed
    to the SDK is measured by the same double as one this service runs
    — the claim is about the walk, whoever runs it.
    """

    # A stand-in that let the walk run forever would hang the suite rather
    # than fail it. Well above any budget a walk may legitimately have.
    _RUNAWAY = 3000

    def __init__(self, walk: _Walk, page: Callable[[int], Any]):
        self._walk = walk
        self._page = page
        # The cursor each request carried, which is also the count of them.
        self.cursors: list[Any] = []
        setattr(self, walk.sibling, self._answer)
        setattr(self, walk.aggregator, partial(getattr(walk.sdk_class, walk.aggregator), self))

    @property
    def calls(self) -> int:
        return len(self.cursors)

    def _answer(self, *_args: Any, **kwargs: Any) -> Any:
        # ``""`` is what the SDK's own walks open with and ``None`` is what
        # ours does; both mean "the first page", so both are recorded as one.
        self.cursors.append(kwargs.get(self._walk.page) or None)
        if self.calls > self._RUNAWAY:
            raise AssertionError(f"the walk made {self.calls} requests and was still going")
        return sdk_response(json_payload=self._page(self.calls))


def _walking(
    network: TitaniumCloudNetworkService, walk: _Walk, page: Callable[[int], Any]
) -> _CountingPivot:
    """Put that endpoint where the service's cached handle would be."""
    double = _CountingPivot(walk, page)
    network.__dict__[walk.handle] = double
    return double


class TestTheSubjectReachesTheEndpointAsTheGuardJudgedIt:
    """The guards judged one string and the wrappers sent another.

    ``normalize_ip_address`` and ``validate_url_or_host`` strip before they
    judge, so ``" 8.8.8.8 "``, ``"EVIL.COM"`` and ``"evil.com."`` passed
    the guard and then reached the API in a form no guard had looked at.
    The API answers those with an empty page, and the CLI announced
    "nothing found" for a subject it had never asked about — the same
    failure ``supported_hash`` was already fixed for.
    """

    @pytest.mark.parametrize("typed", ["  8.8.8.8 ", "\t8.8.8.8\n"])
    def test_a_padded_address_reaches_the_endpoint_stripped(self, network, typed):
        api = _api_for(network, "_ip_threat")
        api.get_ip_report.return_value = sdk_response(json_payload={"rl": {}})

        network.get_ip_report(typed)

        assert api.get_ip_report.call_args.args[0] == "8.8.8.8"

    @pytest.mark.parametrize("typed", ["EVIL.COM", " evil.com. ", "Evil.Com."])
    def test_a_domain_reaches_the_endpoint_spelled_as_it_is_keyed(self, network, typed):
        """A trailing root dot and a capital are the same zone to a
        resolver and two more keys the endpoint holds nothing under."""
        api = _api_for(network, "_domain_threat")
        api.get_domain_report.return_value = sdk_response(json_payload={"rl": {}})

        network.get_domain_report(typed)

        assert api.get_domain_report.call_args.args[0] == "evil.com"

    def test_a_padded_url_reaches_the_endpoint_stripped(self, network):
        api = _api_for(network, "_url_threat")
        api.get_url_report.return_value = sdk_response(json_payload={"rl": {}})

        network.analyze_url(" https://evil.com/x ")

        assert api.get_url_report.call_args.args[0] == "https://evil.com/x"

    def test_a_padded_uri_reaches_the_endpoint_stripped(self, network):
        """Stripped and no more: a URL path is case-sensitive."""
        api = _api_for(network, "_uri_index")
        api.get_uri_index.return_value = sdk_response(json_payload={"rl": {}})

        network.get_uri_index(" https://evil.com/X ")

        assert api.get_uri_index.call_args.args[0] == "https://evil.com/X"

    @pytest.mark.parametrize("wrapper,attribute,sdk_method,key", _PIVOTS)
    def test_every_pivot_sends_the_subject_it_validated(
        self, network, wrapper, attribute, sdk_method, key
    ):
        api = _api_for(network, attribute)
        getattr(api, sdk_method).return_value = sdk_response(json_payload={"rl": {}})
        subject, _ = _SUBJECTS[attribute]

        getattr(network, wrapper)(f"  {subject} ")

        assert getattr(api, sdk_method).call_args.args[0] == subject

    @pytest.mark.parametrize("walk", _WALKS, ids=_walk_ids)
    def test_every_all_variant_sends_the_subject_it_validated(self, network, walk):
        api = _api_for(network, walk.handle)
        getattr(api, walk.sibling).return_value = sdk_response(json_payload=walk.envelope([], None))

        getattr(network, walk.wrapper)(f"  {_subject(walk)} ")

        assert getattr(api, walk.sibling).call_args.args[0] == _subject(walk)


class TestAnAnswerThatCannotBeParsedIsNotNothingFound:
    """Below the top level, ``_rl_list`` used to fall through to ``[]``.

    Only a missing ``rl`` was treated as unreadable. A step that was
    present but was not an object, or a page that was present but was not
    a list, came back as "no files found for this address" — a confident
    wrong answer about the subject, which is the failure this helper's
    own docstring says it exists to prevent.
    """

    def test_a_step_that_is_not_an_object_is_a_failed_lookup(self, network):
        api = _api_for(network, "_uri_index")
        api.get_uri_index.return_value = sdk_response(json_payload={"rl": {"uri_index": "n/a"}})

        assert network.get_uri_index("https://example.com/x") is None

        reported = network.output.error.call_args[0][0]
        assert "rl.uri_index" in reported and "not dict" in reported

    def test_a_leaf_below_a_step_that_is_not_a_list_is_a_failed_lookup(self, network):
        api = _api_for(network, "_uri_index")
        api.get_uri_index.return_value = sdk_response(
            json_payload={"rl": {"uri_index": {"sha1_list": {"count": 0}}}}
        )

        assert network.get_uri_index("https://example.com/x") is None
        assert "rl.uri_index.sha1_list" in network.output.error.call_args[0][0]

    @pytest.mark.parametrize("page", [{"count": 0}, "none", 0, None])
    def test_a_page_that_is_not_a_list_is_a_failed_lookup(self, network, page):
        """``null`` included: the SDK's own pager would raise on extending
        a list with it, so it is not a shape this endpoint family speaks."""
        api = _api_for(network, "_ip_threat")
        api.get_downloaded_files.return_value = sdk_response(
            json_payload={"rl": {"downloaded_files": page}}
        )

        assert network.get_files_from_ip("1.2.3.4") is None
        assert "rl.downloaded_files" in network.output.error.call_args[0][0]

    def test_a_step_that_is_simply_absent_is_still_an_empty_page(self, network):
        """The one shape the SDK's aggregators do read as "nothing here"."""
        api = _api_for(network, "_uri_index")
        api.get_uri_index.return_value = sdk_response(json_payload={"rl": {"next_page": None}})

        assert network.get_uri_index("https://example.com/x") == []
        assert not network.output.error.called


# One page of whatever a pivot lists, for the tests about what came with it.
_ONE_PAGE = [{"sha1": "a" * 40}]


class TestAFirstPageSaysWhenTheEnvelopeHeldAnother:
    """A first page holds 1000 records of a corpus that may hold 40000.

    The cursor for the page after it sits in the same envelope as the
    entries — which is where every SDK aggregator reads it — so a wrapper
    that reads only the entries answers one page as if it were the whole
    set, on the flagless invocation.
    """

    @pytest.mark.parametrize("wrapper,attribute,sdk_method,key", _PIVOTS)
    def test_a_cursor_beside_the_entries_is_a_page_left_unfetched(
        self, network, wrapper, attribute, sdk_method, key
    ):
        api = _api_for(network, attribute)
        getattr(api, sdk_method).return_value = sdk_response(
            json_payload={"rl": {key: _ONE_PAGE, "next_page": "page2"}}
        )
        subject, _ = _SUBJECTS[attribute]

        assert getattr(network, wrapper)(subject) == _ONE_PAGE
        assert network.pages_left_unfetched

    @pytest.mark.parametrize("wrapper,attribute,sdk_method,key", _PIVOTS)
    @pytest.mark.parametrize("cursor", [{}, {"next_page": None}, {"next_page": ""}])
    def test_a_page_with_no_cursor_is_the_whole_answer(
        self, network, wrapper, attribute, sdk_method, key, cursor
    ):
        """Absent, ``null`` and empty are all the last page the SDK stops on."""
        api = _api_for(network, attribute)
        getattr(api, sdk_method).return_value = sdk_response(
            json_payload={"rl": {key: _ONE_PAGE} | cursor}
        )
        subject, _ = _SUBJECTS[attribute]

        assert getattr(network, wrapper)(subject) == _ONE_PAGE
        assert not network.pages_left_unfetched

    def test_the_uri_index_cursor_is_the_one_that_endpoint_states(self, network):
        """TCA-0401 states a SHA-1 cursor inside ``uri_index``, not beside it."""
        api = _api_for(network, "_uri_index")
        api.get_uri_index.return_value = sdk_response(
            json_payload={
                "rl": {"uri_index": {"sha1_list": ["a" * 40], "next_page_sha1": "b" * 40}}
            }
        )

        assert network.get_uri_index("https://example.com/x") == ["a" * 40]
        assert network.pages_left_unfetched

    @pytest.mark.parametrize(
        "envelope",
        [
            {"uri_index": {"sha1_list": ["a" * 40], "next_page_sha1": None}},
            {"uri_index": {"sha1_list": ["a" * 40]}, "next_page": "page2"},
        ],
        ids=["no_cursor", "a_key_this_endpoint_does_not_page_by"],
    )
    def test_a_uri_index_page_with_no_sha1_cursor_is_the_whole_answer(self, network, envelope):
        api = _api_for(network, "_uri_index")
        api.get_uri_index.return_value = sdk_response(json_payload={"rl": envelope})

        assert network.get_uri_index("https://example.com/x") == ["a" * 40]
        assert not network.pages_left_unfetched

    def test_an_answer_that_could_not_be_read_claims_nothing_about_pages(self, network):
        """A lookup that failed leaves no verdict from the one before it."""
        api = _api_for(network, "_ip_threat")
        api.get_downloaded_files.return_value = sdk_response(
            json_payload={"rl": {"downloaded_files": _ONE_PAGE, "next_page": "page2"}}
        )
        assert network.get_files_from_ip("1.2.3.4") == _ONE_PAGE

        api.get_downloaded_files.return_value = sdk_response(json_payload={"unshaped": True})

        assert network.get_files_from_ip("1.2.3.4") is None
        assert not network.pages_left_unfetched


class TestNoLookupInheritsTheLastOnesPagingVerdict:
    """``pages_left_unfetched`` is one cell read after every pivot.

    A verdict left standing from the lookup before is a partial-answer
    notice printed over a whole answer, so every lookup has to start from
    a cleared cell — not only the ones that page.
    """

    def _left_a_page_behind(self, network) -> None:
        api = _api_for(network, "_ip_threat")
        api.get_downloaded_files.return_value = sdk_response(
            json_payload={"rl": {"downloaded_files": _ONE_PAGE, "next_page": "page2"}}
        )
        assert network.get_files_from_ip("1.2.3.4") == _ONE_PAGE
        assert network.pages_left_unfetched

    def test_a_refused_subject_clears_the_verdict_before_it_returns(self, network):
        """The guard returns before any paging code runs, so it must clear too."""
        self._left_a_page_behind(network)

        assert network.get_files_from_ip("not-an-ip") is None
        assert not network.pages_left_unfetched

    def test_a_lookup_that_pages_nothing_clears_the_verdict(self, network):
        """A report is one whole answer; it must not be announced as partial."""
        self._left_a_page_behind(network)
        api = _api_for(network, "_ip_threat")
        api.get_ip_report.return_value = sdk_response(json_payload={"rl": {}})

        network.get_ip_report("1.2.3.4")

        assert not network.pages_left_unfetched

    def test_a_walk_that_fetched_every_page_clears_the_verdict(self, network):
        """An aggregated walk leaves nothing behind and records nothing."""
        self._left_a_page_behind(network)
        api = _api_for(network, "_ip_threat")
        api.get_downloaded_files.return_value = sdk_response(
            json_payload={"rl": {"downloaded_files": _ONE_PAGE}}
        )

        assert network.get_files_from_ip_aggregated("1.2.3.4") == _ONE_PAGE

        assert not network.pages_left_unfetched

    def test_a_call_that_raises_clears_the_verdict(self, network):
        """The verdict is cleared on the way in, so nothing has to unwind it."""
        self._left_a_page_behind(network)
        api = _api_for(network, "_ip_threat")
        api.get_downloaded_files.side_effect = RuntimeError("TitaniumCloud said no")

        assert network.get_files_from_ip("1.2.3.4") is None

        assert not network.pages_left_unfetched


class TestAPageBodyIsReadOnce:
    """``requests.Response.json`` re-runs ``json.loads`` on every call.

    These pages hold 1000 records, and the entries and the cursor for the
    next page sit in the same body, so reading them from two separate
    ``json()`` calls parses that body twice — and checks the status twice
    — for one metered request.
    """

    @pytest.mark.parametrize("wrapper,attribute,sdk_method,key", _PIVOTS)
    def test_a_first_page_is_parsed_once(self, network, wrapper, attribute, sdk_method, key):
        api = _api_for(network, attribute)
        response = sdk_response(json_payload={"rl": {key: _ONE_PAGE, "next_page": "page2"}})
        getattr(api, sdk_method).return_value = response
        subject, _ = _SUBJECTS[attribute]

        assert getattr(network, wrapper)(subject) == _ONE_PAGE

        assert response.json.call_count == 1

    def test_the_uri_index_page_is_parsed_once(self, network):
        """The one pivot whose cursor sits below the entries, not beside them."""
        api = _api_for(network, "_uri_index")
        response = sdk_response(
            json_payload={
                "rl": {"uri_index": {"sha1_list": ["a" * 40], "next_page_sha1": "b" * 40}}
            }
        )
        api.get_uri_index.return_value = response

        assert network.get_uri_index("https://example.com/x") == ["a" * 40]

        assert response.json.call_count == 1


class TestAggregatedLookupsCanBeBounded:
    """``--all`` was every page of a busy subject, at 1000 records a page.

    Each page is its own metered request and the lot is held in memory
    before a row is drawn, so ``ticloud ip-files <busy IP> --all`` was an
    unbounded bill with no way to say how much of the corpus was wanted.
    """

    @pytest.mark.parametrize("walk", _WALKS, ids=_walk_ids)
    def test_the_budget_caps_the_records_exactly(self, network, walk):
        """Two records a page and a cap of five: five, not six."""
        double = _walking(network, walk, _pages_of(walk, 2, last=None))

        assert len(getattr(network, walk.wrapper)(_subject(walk), max_results=5)) == 5
        assert double.calls == 3

    @pytest.mark.parametrize("walk", _WALKS, ids=_walk_ids)
    def test_every_page_is_what_no_budget_means(self, network, walk):
        """Uncapped, the walk goes on until the endpoint stops naming a page."""
        double = _walking(network, walk, _pages_of(walk, 2, last=3))

        assert len(getattr(network, walk.wrapper)(_subject(walk))) == 6
        assert double.calls == 3

    @pytest.mark.parametrize("walk", _WALKS, ids=_walk_ids)
    @pytest.mark.parametrize("budget", [0, -5])
    def test_a_budget_of_nothing_is_not_read_as_everything(self, network, walk, budget):
        """0 is the one number that plainly means "stop immediately", and it
        is what every SDK aggregator reads as "fetch the whole corpus"."""
        double = _walking(network, walk, _pages_of(walk, 3, last=None))

        assert len(getattr(network, walk.wrapper)(_subject(walk), max_results=budget)) == 1
        assert double.calls == 1


class TestAPivotWalkThatStopsMakingProgressStops:
    """The nine pivot walks had no stop of their own and no budget.

    Every one of them was handed to an SDK ``*_aggregated`` method, whose
    ``while True`` returns on ``not next_page`` and on ``len(results) >=
    max_results`` and on nothing else (SDK ticloud.py:1104-1124,
    1447-1471, 1807-1829, 1868-1888 …). Against an endpoint answering
    ``{"rl": {"<key>": [], "next_page": "cursor-x"}}`` neither can ever
    come true: nothing is added to ``results``, so the cap is never
    reached, and the cursor never goes false. The probe was still going
    after 3001 metered requests, with and without ``--max-results`` —
    the budget caps records, not pages — and ``--all`` passes no cap at
    all.

    A cursor that never moves is the other shape of the same standstill:
    the same page collected over and over until the process runs out of
    memory.
    """

    @pytest.mark.parametrize("walk", _WALKS, ids=_walk_ids)
    @pytest.mark.parametrize("budget", [None, 5], ids=["all", "max_results"])
    def test_an_endpoint_that_promises_pages_and_sends_none_does_not_loop(
        self, network, walk, budget
    ):
        double = _walking(network, walk, lambda _call: walk.envelope([], "cursor-x"))

        results = getattr(network, walk.wrapper)(_subject(walk), max_results=budget)

        assert double.calls == 2, "the cursor it repeated is what stops this one"
        assert results == []
        assert network.output.warning.called

    @pytest.mark.parametrize("walk", _WALKS, ids=_walk_ids)
    def test_empty_pages_on_a_moving_cursor_cost_the_page_budget_and_no_more(self, network, walk):
        """The same standstill with a cursor that keeps changing.

        Indistinguishable from a corpus whose records sit past a run of
        filtered pages, so the walk reads on and the budget ends it.
        """
        double = _walking(network, walk, lambda call: walk.envelope([], f"cursor-{call + 1}"))

        results = getattr(network, walk.wrapper)(_subject(walk))

        assert double.calls == _MAX_PIVOT_PAGES
        assert results == []
        assert network.output.warning.called

    @pytest.mark.parametrize("walk", _WALKS, ids=_walk_ids)
    def test_a_cursor_that_never_moves_does_not_loop(self, network, walk):
        """A page that repeats itself is a standstill, not a corpus."""
        double = _walking(network, walk, lambda call: walk.envelope([_record(call)], "cursor-x"))

        results = getattr(network, walk.wrapper)(_subject(walk))

        assert double.calls == 2, "the walk kept asking for the page it already had"
        assert results == [_record(1), _record(2)]
        assert network.output.warning.called

    @pytest.mark.parametrize("walk", _WALKS, ids=_walk_ids)
    def test_a_cursor_that_cycles_between_two_pages_does_not_run_out_the_budget(
        self, network, walk
    ):
        """Every cursor stated on this walk is remembered, not only the last.

        An endpoint alternating A, B, A, B … moves its cursor at every
        step, so comparing against the previous one alone never fires: the
        walk paid all 100 requests to collect two distinct pages over and
        over, under a warning saying the corpus held more.
        """
        double = _walking(
            network, walk, lambda call: walk.envelope([_record(call)], f"cursor-{call % 2}")
        )

        results = getattr(network, walk.wrapper)(_subject(walk))

        assert double.calls == 3
        assert results == [_record(1), _record(2), _record(3)]
        assert "already sent" in network.output.warning.call_args[0][0]

    @pytest.mark.parametrize("walk", _WALKS, ids=_walk_ids)
    def test_a_walk_that_keeps_making_progress_stops_at_the_page_budget(self, network, walk):
        """Full pages and a moving cursor forever: the walk's own bound."""
        double = _walking(network, walk, _pages_of(walk, 2, last=None))

        results = getattr(network, walk.wrapper)(_subject(walk))

        assert double.calls == _MAX_PIVOT_PAGES
        assert len(results) == _MAX_PIVOT_PAGES * 2
        assert network.output.warning.called

    @pytest.mark.parametrize("walk", _WALKS, ids=_walk_ids)
    def test_a_corpus_that_runs_out_first_stops_the_walk_there(self, network, walk):
        double = _walking(network, walk, _pages_of(walk, 2, last=2))

        assert len(getattr(network, walk.wrapper)(_subject(walk))) == 4
        assert double.calls == 2
        assert not network.output.warning.called


class TestAWalkCollectsWhatTheEndpointStated:
    """The records the SDK's own walk brought back, in the order it had them.

    The pages are now fetched here rather than by the SDK, so what a
    well-behaved corpus answers must not have changed: the same records,
    in the endpoint's order, none of them twice, and each page asked for
    with the cursor the page before it named.
    """

    @pytest.mark.parametrize("walk", _WALKS, ids=_walk_ids)
    def test_the_records_come_back_in_order_and_once_each(self, network, walk):
        _walking(network, walk, _pages_of(walk, 3, last=3))

        collected = getattr(network, walk.wrapper)(_subject(walk))

        assert collected == [_record(page * 10 + n) for page in (1, 2, 3) for n in (1, 2, 3)]

    @pytest.mark.parametrize("walk", _WALKS, ids=_walk_ids)
    def test_each_page_is_asked_for_by_the_cursor_the_one_before_it_named(self, network, walk):
        double = _walking(network, walk, _pages_of(walk, 1, last=3))

        getattr(network, walk.wrapper)(_subject(walk))

        assert double.cursors == [None, "cursor-2", "cursor-3"]

    @pytest.mark.parametrize("walk", _WALKS, ids=_walk_ids)
    def test_a_subject_the_endpoint_would_refuse_never_reaches_it(self, network, walk):
        double = _walking(network, walk, _pages_of(walk, 1, last=1))
        _, rejected = _SUBJECTS[walk.handle]

        assert getattr(network, walk.wrapper)(rejected) is None
        assert double.calls == 0

    @pytest.mark.parametrize("walk", _WALKS, ids=_walk_ids)
    def test_a_page_that_cannot_be_read_ends_the_walk_as_a_failed_lookup(self, network, walk):
        """Not a short answer: a page we could not read is not "nothing left"."""
        _walking(
            network,
            walk,
            lambda call: walk.envelope([_record(call)], "cursor-x") if call == 1 else {"nope": 1},
        )

        assert getattr(network, walk.wrapper)(_subject(walk)) is None
        assert "rl envelope" in network.output.error.call_args[0][0]


class TestASparsePageIsNotTheEndOfTheCorpus:
    """A page that carried nothing while naming another used to stop the walk.

    Only the cursor says where the corpus ends. An endpoint filters each
    page server-side, so a page can come back empty with pages of records
    still behind it, and stopping there hands back part of the answer as
    though it were all of it: ``pages_left_unfetched`` stays false and no
    partial-answer notice is printed. A first page shaped ``{"rl":
    {"next_page": "c2"}}`` — which is an empty page by every reader here —
    made that "no files found" for a subject that has files.

    Nothing was bought with it either: an empty page whose cursor never
    moves is stopped by the repeated-cursor check, and one whose cursor
    moves is bounded by the page budget.
    """

    @pytest.mark.parametrize("walk", _WALKS, ids=_walk_ids)
    def test_a_sparse_pivot_page_does_not_truncate_the_answer(self, network, walk):
        double = _walking(
            network,
            walk,
            lambda call: walk.envelope(
                [] if call == 2 else [_record(call)], None if call == 3 else f"cursor-{call + 1}"
            ),
        )

        assert getattr(network, walk.wrapper)(_subject(walk)) == [_record(1), _record(3)]
        assert double.calls == 3
        assert not network.output.warning.called

    @pytest.mark.parametrize("walk", _WALKS, ids=_walk_ids)
    def test_a_first_page_carrying_only_a_cursor_is_not_an_empty_corpus(self, network, walk):
        double = _walking(
            network,
            walk,
            lambda call: walk.envelope(
                [] if call == 1 else [_record(call)], None if call > 1 else "cursor-2"
            ),
        )

        assert getattr(network, walk.wrapper)(_subject(walk)) == [_record(2)]
        assert double.calls == 2

    def test_a_sparse_search_page_does_not_truncate_the_answer(self, service, monkeypatch):
        api = _CountingAdvancedSearch(
            [
                {"entries": [{"n": 1}], "more_pages": True},
                {"entries": [], "more_pages": True},
                {"entries": [{"n": 3}], "more_pages": False},
            ]
        )
        monkeypatch.setattr(
            "rl_cli.services.titanium_cloud.service.ticloud.AdvancedSearch", lambda **kw: api
        )

        results = service.search_samples("classification:malicious", 1_000_000)

        assert results == [{"n": 1}, {"n": 3}]
        assert api.calls == 3
        assert not service.pages_left_unfetched


class TestUriIndex:
    """TCA-0401 answers hashes rather than records, two levels down."""

    def test_the_sha1_list_is_what_comes_back(self, network):
        api = _api_for(network, "_uri_index")
        api.get_uri_index.return_value = sdk_response(
            json_payload={"rl": {"uri_index": {"sha1_list": ["a" * 40], "next_page_sha1": None}}}
        )

        assert network.get_uri_index("https://example.com/x") == ["a" * 40]

    def test_a_uri_that_is_not_one_never_reaches_the_endpoint(self, network):
        api = _api_for(network, "_uri_index")

        assert network.get_uri_index("not a uri") is None

        assert not api.get_uri_index.called
        assert "not a uri" in network.output.error.call_args[0][0]

    def test_an_error_status_carries_the_servers_own_explanation(self, network):
        api = _api_for(network, "_uri_index")
        api.get_uri_index.return_value = sdk_response(403, {"message": "Not licensed"})

        assert network.get_uri_index("https://example.com/x") is None

        reported = network.output.error.call_args[0][0]
        assert "HTTP 403" in reported and "Not licensed" in reported

    def test_the_all_variant_walks_the_same_endpoint_a_page_at_a_time(self, network):
        api = _api_for(network, "_uri_index")
        api.get_uri_index.return_value = sdk_response(
            json_payload={"rl": {"uri_index": {"sha1_list": ["b" * 40, "c" * 40]}}}
        )

        assert network.get_uri_index_aggregated("example.com") == ["b" * 40, "c" * 40]
        assert not api.get_uri_index_aggregated.called

    def test_the_all_variant_guards_its_uri_too(self, network):
        api = _api_for(network, "_uri_index")

        assert network.get_uri_index_aggregated("not a uri") is None
        assert not api.get_uri_index.called


class TestDomainAndIpReports:
    @pytest.mark.parametrize(
        "wrapper,attribute,sdk_method,subject,rejected",
        [
            ("get_domain_report", "_domain_threat", "get_domain_report", "evil.com", "not a host"),
            ("get_ip_report", "_ip_threat", "get_ip_report", "8.8.8.8", "999.1.1.1"),
        ],
    )
    def test_the_report_is_returned_whole(
        self, network, wrapper, attribute, sdk_method, subject, rejected
    ):
        report = {"rl": {"malicious": True}}
        api = _api_for(network, attribute)
        getattr(api, sdk_method).return_value = sdk_response(json_payload=report)

        assert getattr(network, wrapper)(subject) == report

    @pytest.mark.parametrize(
        "wrapper,attribute,sdk_method,subject,rejected",
        [
            ("get_domain_report", "_domain_threat", "get_domain_report", "evil.com", "not a host"),
            ("get_ip_report", "_ip_threat", "get_ip_report", "8.8.8.8", "999.1.1.1"),
        ],
    )
    def test_a_subject_the_endpoint_would_refuse_never_reaches_it(
        self, network, wrapper, attribute, sdk_method, subject, rejected
    ):
        api = _api_for(network, attribute)

        assert getattr(network, wrapper)(rejected) is None

        assert not getattr(api, sdk_method).called
        assert rejected in network.output.error.call_args[0][0]

    @pytest.mark.parametrize(
        "wrapper,attribute,sdk_method,subject,rejected",
        [
            ("get_domain_report", "_domain_threat", "get_domain_report", "evil.com", "not a host"),
            ("get_ip_report", "_ip_threat", "get_ip_report", "8.8.8.8", "999.1.1.1"),
        ],
    )
    def test_an_error_status_carries_the_servers_own_explanation(
        self, network, wrapper, attribute, sdk_method, subject, rejected
    ):
        api = _api_for(network, attribute)
        getattr(api, sdk_method).return_value = sdk_response(500, {"error": "backend down"})

        assert getattr(network, wrapper)(subject) is None

        reported = network.output.error.call_args[0][0]
        assert "HTTP 500" in reported and "backend down" in reported


# Measured, one implementation against the other: ticloud sent every one of
# these to the endpoint; the A1000 half refused every one.
_DISAGREED_DOMAINS = ["1.2.3.4", "user@example.com", "a..b.com"]


class TestBothHalvesOfTheCliAgreeOnWhatADomainIs:
    """One CLI held two ``_valid_domain`` implementations, and they disagreed.

    The A1000 half asked :func:`~rl_cli.models.validators.normalize_domain`;
    this half hand-rolled ``strip``/``rstrip('.')``/``lower`` over a "no
    ``/`` and no ``:``" check and ``validate_url_or_host``. So an IPv4
    address, an email address and a name with an empty label were refused
    by ``a1000 domain-report`` and sent by ``ticloud domain-report`` —
    where the endpoint answers them with an empty page and the CLI
    reported "No files found for a..b.com", which is the confident wrong
    answer both guards' docstrings say they exist to prevent.

    Both halves read the shared :mod:`rl_cli.services.addresses` now, so
    what is pinned here is that they answer the same thing: the same
    verdict, in the same words, for the subjects they used to differ over.
    """

    @pytest.fixture
    def a1000_network(self, a1000_session: A1000Session) -> A1000NetworkService:
        a1000_session.client = MagicMock()
        return a1000_session.service(A1000NetworkService, MagicMock())

    @pytest.mark.parametrize("subject", _DISAGREED_DOMAINS)
    def test_neither_half_sends_it_and_both_refuse_it_in_the_same_words(
        self, network, a1000_network, subject
    ):
        api = _api_for(network, "_domain_threat")

        assert network.get_domain_report(subject) is None
        assert a1000_network.get_domain_report(subject) is None

        assert not api.get_domain_report.called, "ticloud asked about a subject it cannot key"
        assert not a1000_network.client.network_domain_report.called
        refusals = {
            network.output.error.call_args.args[0],
            a1000_network.output.error.call_args.args[0],
        }
        assert len(refusals) == 1, f"the two halves refuse {subject} differently: {refusals}"
        assert subject in refusals.pop()

    @pytest.mark.parametrize(
        "typed,sent",
        [
            ("münchen.de", "xn--mnchen-3ya.de"),
            ("évil.com", "xn--vil-9la.com"),
            (" MÜNCHEN.de. ", "xn--mnchen-3ya.de"),
        ],
    )
    def test_an_idn_is_encoded_by_both_halves_rather_than_refused_by_this_one(
        self, network, a1000_network, typed, sent
    ):
        """The regression the extraction introduced, and the agreement it kept.

        Choosing the A1000 reading wholesale also took its ASCII-only
        label pattern, which had been matched against the typed name. The
        hand-rolled guard this half retired answered ``münchen.de`` with
        ``münchen.de``; the shared one answered ``None``, so nine
        TitaniumCloud lookups began refusing a domain spelled the way a
        phishing lure spells it while accepting its punycode twin. Both
        endpoints key on punycode, so both halves send that now.
        """
        api = _api_for(network, "_domain_threat")
        api.get_domain_report.return_value = sdk_response(json_payload={"rl": {}})
        a1000_network.client.network_domain_report.return_value = sdk_response(200, {"ok": True})

        network.get_domain_report(typed)
        a1000_network.get_domain_report(typed)

        assert api.get_domain_report.call_args.args[0] == sent
        assert a1000_network.client.network_domain_report.call_args.args[0] == sent
        assert not network.output.error.called
        assert not a1000_network.output.error.called

    @pytest.mark.parametrize(
        "typed,sent",
        [
            ("2001:DB8:0000:0000:0000:0000:0000:0001", "2001:db8::1"),
            ("fe80::1%eth0", "fe80::1"),
        ],
    )
    def test_one_address_is_one_key_on_both_halves(self, network, a1000_network, typed, sent):
        """TCA-0406 POSTs the address in a JSON body and the A1000 puts it in
        a path segment, so only one half could have its path reshaped — but
        both file their records under the canonical spelling, and both used
        to forward whatever expanded or scoped form was typed."""
        api = _api_for(network, "_ip_threat")
        api.get_ip_report.return_value = sdk_response(json_payload={"rl": {}})
        a1000_network.client.network_ip_addr_report.return_value = sdk_response(200, {"ok": True})

        network.get_ip_report(typed)
        a1000_network.get_ip_report(typed)

        assert api.get_ip_report.call_args.args[0] == sent
        assert a1000_network.client.network_ip_addr_report.call_args.args[0] == sent

    def test_a_domain_both_halves_accept_reaches_both_endpoints_alike(self, network, a1000_network):
        """The other half of the claim: agreement, not a matched refusal."""
        api = _api_for(network, "_domain_threat")
        api.get_domain_report.return_value = sdk_response(json_payload={"rl": {}})
        a1000_network.client.network_domain_report.return_value = sdk_response(200, {"ok": True})

        network.get_domain_report(" EVIL.COM. ")
        a1000_network.get_domain_report(" EVIL.COM. ")

        assert api.get_domain_report.call_args.args[0] == "evil.com"
        assert a1000_network.client.network_domain_report.call_args.args[0] == "evil.com"


class TestBulkFileReputation:
    """One request for the whole batch, and the two rules the endpoint has."""

    def _api(self, service, payload=None, status=200) -> MagicMock:
        api = _api_for(service, "_file_reputation")
        api.get_file_reputation.return_value = sdk_response(status, payload)
        return api

    def test_the_batch_is_sent_as_one_list_and_the_entries_come_back(self, service):
        entries = [{"sha1": "a" * 40, "status": "MALICIOUS"}, {"sha1": "b" * 40}]
        api = self._api(service, {"rl": {"entries": entries}})

        assert service.get_bulk_file_reputation(["a" * 40, "b" * 40]) == entries

        assert api.get_file_reputation.call_count == 1
        assert api.get_file_reputation.call_args.kwargs["hash_input"] == ["a" * 40, "b" * 40]

    def test_the_batch_reaches_the_sdk_normalised(self, service):
        api = self._api(service, {"rl": {"entries": []}})

        service.get_bulk_file_reputation([f"  {SHA256.upper()}\n", SHA256])

        assert api.get_file_reputation.call_args.kwargs["hash_input"] == [SHA256, SHA256]

    def test_a_bad_entry_is_named_by_position_and_nothing_is_sent(self, service):
        api = self._api(service)

        assert service.get_bulk_file_reputation(["a" * 40, "nothash", "c" * 40]) is None

        assert not api.get_file_reputation.called
        reported = service.output.error.call_args[0][0]
        assert "entry 2 of 3" in reported and "nothash" in reported

    def test_sha512_is_refused_in_a_batch_as_it_is_on_its_own(self, service):
        api = self._api(service)

        assert service.get_bulk_file_reputation([SHA512]) is None

        assert not api.get_file_reputation.called

    def test_a_mixed_batch_is_refused_rather_than_queried_under_one_type(self, service):
        """The bulk query sends the first entry's hash type for the whole list."""
        api = self._api(service)

        assert service.get_bulk_file_reputation(["a" * 40, SHA256]) is None

        assert not api.get_file_reputation.called
        reported = service.output.error.call_args[0][0]
        assert "Mixed hash types" in reported and "entry 2" in reported

    def test_an_error_status_carries_the_servers_own_explanation(self, service):
        self._api(service, {"message": "Quota exceeded"}, status=429)

        assert service.get_bulk_file_reputation(["a" * 40]) is None

        reported = service.output.error.call_args[0][0]
        assert "HTTP 429" in reported and "Quota exceeded" in reported

    def test_an_answer_with_no_envelope_is_a_failed_lookup(self, service):
        self._api(service, {"unshaped": True})

        assert service.get_bulk_file_reputation(["a" * 40]) is None
        assert "rl envelope" in service.output.error.call_args[0][0]

    def test_an_envelope_answered_empty_is_an_answer_about_the_batch(self, service):
        self._api(service, {"rl": {"entries": []}})

        assert service.get_bulk_file_reputation(["a" * 40]) == []
        assert not service.output.error.called

    def test_an_envelope_with_no_entries_key_is_a_failed_lookup(self, service):
        """``rl.entries`` is the one key here with no SDK corroboration.

        Nothing in the vendored SDK parses a ``malware_presence`` bulk
        answer — the single ``rl.entries`` reader it does have belongs to
        TCA-0407 Network Reputation — so this key is asserted by this
        wrapper alone. If it is ever wrong, or the endpoint renames it,
        the honest report is "we could not read this answer" and not "no
        reputation record for any hash in this batch", which is the same
        sentence a genuinely empty batch would get.
        """
        self._api(service, {"rl": {"requested_hash_type": "sha1"}})

        assert service.get_bulk_file_reputation(["a" * 40]) is None
        assert "rl.entries" in service.output.error.call_args[0][0]


class TestReputationAnswersOneShapeWhateverTheBatchSize:
    """One hash and several were graded at two different fidelities.

    One went through the verdict path — verdict, severity colour, threat
    name — and several were rendered from ``rl.entries``, whose records
    the renderer never unwrapped, so ``-o rich`` and ``-o sarif`` answered
    the same question differently depending on how many hashes were named.
    The service reconciles the two envelopes here, once, in the same way
    the SARIF exporter and the rich renderer already unwrap a single one.
    """

    def _api(self, service, payload) -> MagicMock:
        api = _api_for(service, "_file_reputation")
        api.get_file_reputation.return_value = sdk_response(json_payload=payload)
        return api

    def test_one_hash_answers_the_record_not_the_envelope_it_arrived_in(self, service):
        record = {"sha1": "a" * 40, "status": "MALICIOUS", "threat_name": "Win32.Trojan.Emotet"}
        self._api(service, {"rl": {"malware_presence": record}})

        assert service.get_reputation_records(["a" * 40]) == [record]

    def test_several_hashes_answer_the_bulk_entries(self, service):
        entries = [{"sha1": "a" * 40, "status": "MALICIOUS"}, {"sha1": "b" * 40}]
        self._api(service, {"rl": {"entries": entries}})

        assert service.get_reputation_records(["a" * 40, "b" * 40]) == entries

    def test_the_same_sample_grades_the_same_alone_and_in_a_batch(self, service):
        record = {"sha1": "a" * 40, "status": "MALICIOUS", "threat_name": "Win32.Evil"}

        self._api(service, {"rl": {"malware_presence": record}})
        alone = service.get_reputation_records(["a" * 40])
        self._api(service, {"rl": {"entries": [record, {"sha1": "b" * 40}]}})
        batched = service.get_reputation_records(["a" * 40, "b" * 40])

        assert alone is not None and batched is not None
        assert alone[0] == batched[0]

    def test_one_hash_still_reaches_the_single_endpoint(self, service):
        """A batch of one must not pay for a bulk POST, and the wrappers
        are what the CLI tests stub, so the dispatch is pinned here."""
        api = self._api(service, {"rl": {"malware_presence": {"status": "KNOWN"}}})

        service.get_reputation_records([SHA256])

        assert api.get_file_reputation.call_args.kwargs["hash_input"] == SHA256

    @pytest.mark.parametrize("payload", [{}, {"rl": {"malware_presence": {}}}])
    def test_an_empty_answer_for_one_hash_is_a_failure_not_an_empty_record(self, service, payload):
        """TCA-0101 grades a hash it has never seen "UNKNOWN"; it does not
        answer nothing, so nothing is an answer we could not read."""
        self._api(service, payload)

        assert service.get_reputation_records([SHA256]) is None

    def test_a_refused_hash_never_reaches_either_endpoint(self, service):
        api = _api_for(service, "_file_reputation")

        assert service.get_reputation_records([SHA512]) is None
        assert not api.get_file_reputation.called


class TestSampleDownload:
    """The wrapper that writes live malware, and what it refuses to write."""

    @pytest.mark.posix_only
    def test_the_sample_is_written_owner_only(self, service, tmp_path):
        api = _api_for(service, "_file_download")
        api.download_sample.return_value = sdk_response(content=b"MZ malware")
        destination = tmp_path / "sample.malware"

        assert service.download_sample(SHA256, destination) is True

        assert destination.read_bytes() == b"MZ malware"
        assert destination.stat().st_mode & 0o777 == 0o600
        assert api.download_sample.call_args.args == (SHA256,)

    def test_a_hash_the_endpoint_would_refuse_writes_nothing(self, service, tmp_path):
        api = _api_for(service, "_file_download")
        destination = tmp_path / "sample.malware"

        assert service.download_sample(SHA512, destination) is False

        assert not api.download_sample.called
        assert not destination.exists()

    def test_a_failed_download_leaves_no_file_and_names_the_status(self, service, tmp_path):
        api = _api_for(service, "_file_download")
        api.download_sample.return_value = sdk_response(404, {"message": "Sample not found"})
        destination = tmp_path / "sample.malware"

        assert service.download_sample(SHA256, destination) is False

        assert not destination.exists()
        reported = service.output.error.call_args[0][0]
        assert "HTTP 404" in reported and "Sample not found" in reported

    @pytest.mark.posix_only
    def test_a_symlink_at_the_destination_is_refused(self, service, tmp_path):
        """The sample must not be written through a link the analyst left."""
        api = _api_for(service, "_file_download")
        api.download_sample.return_value = sdk_response(content=b"MZ malware")
        target = tmp_path / "innocent.txt"
        target.write_text("keep me")
        link = tmp_path / "sample.malware"
        link.symlink_to(target)

        assert service.download_sample(SHA256, link) is False

        assert target.read_text(encoding="utf-8") == "keep me"
        assert service.output.error.called

    def test_a_status_that_is_neither_success_nor_error_is_named_not_written(
        self, service, tmp_path
    ):
        """A redirect is not a sample: json_on has nothing to raise about it."""
        api = _api_for(service, "_file_download")
        api.download_sample.return_value = sdk_response(302, content=b"")
        destination = tmp_path / "sample.malware"

        assert service.download_sample(SHA256, destination) is False

        assert not destination.exists()
        assert "302" in service.output.error.call_args[0][0]

    def test_download_status_reports_what_the_endpoint_said(self, service):
        api = _api_for(service, "_file_download")
        status = {"rl": {"status": [{"sha1": "a" * 40, "status": "SAMPLE_AVAILABLE"}]}}
        api.get_download_status.return_value = sdk_response(json_payload=status)

        assert service.get_download_status(SHA256) == status
        assert api.get_download_status.call_args.args == (SHA256,)

    def test_download_status_refuses_a_hash_these_endpoints_do_not_take(self, service):
        api = _api_for(service, "_file_download")

        assert service.get_download_status(SHA512) is None
        assert not api.get_download_status.called


# Every public method each half defines. Peers, so neither list may leak
# into the other class.
_NETWORK_WRAPPERS = sorted(
    name
    for name, value in vars(TitaniumCloudNetworkService).items()
    if not name.startswith("_") and callable(value)
)
_FILE_WRAPPERS = sorted(
    name
    for name, value in vars(TitaniumCloudService).items()
    if not name.startswith("_") and callable(value)
)


class TestTheTwoServicesArePeers:
    """Neither half reaches the other: no facade, no forwarding, no handles.

    ``TitaniumCloudService`` used to answer to every network name as well,
    forwarding twenty-one methods to a network service it built inside
    itself. The forwarders are gone, and this is what stops them growing
    back — as a re-added forwarder, as a copied guard, or as one class
    quietly building all ten API families again, which is the thirty-two
    public methods over ten families this was split to stop being.
    """

    def test_the_network_service_answers_on_its_own(self, network):
        """The point of the split: enriching an address needs this and no more."""
        entries = [{"sha1": "a" * 40}]
        api = sdk_api("_ip_threat")
        api.get_downloaded_files.return_value = sdk_response(
            json_payload={"rl": {"downloaded_files": entries}}
        )
        network.__dict__["_ip_threat"] = api

        assert network.get_files_from_ip("1.2.3.4") == entries

    @pytest.mark.parametrize("handle", sorted(_FILE_SDK_CLASS))
    def test_the_network_half_builds_none_of_the_file_oriented_handles(self, network, handle):
        """Four of the ten API families, and no way to reach the other six."""
        assert not hasattr(network, handle)

    @pytest.mark.parametrize("handle", sorted(_NETWORK_SDK_CLASS))
    def test_the_file_half_builds_none_of_the_network_handles(self, service, handle):
        """The other direction, which the facade used to fail by holding a
        whole network service — a second copy of the credentials and the
        proxy URL for endpoint families a hash lookup never calls."""
        assert not hasattr(service, handle)

    @pytest.mark.parametrize("wrapper", _NETWORK_WRAPPERS)
    def test_the_file_half_no_longer_answers_to_a_network_name(self, service, wrapper):
        """A breaking change on purpose: ``ticloud.get_files_from_ip`` used
        to work and now raises, so the network half is asked by name."""
        assert not hasattr(service, wrapper)

    @pytest.mark.parametrize("wrapper", _FILE_WRAPPERS)
    def test_the_network_half_never_answered_to_a_file_name(self, network, wrapper):
        assert not hasattr(network, wrapper)


class TestAUrlLookupIsNotPublishedByDefault:
    """The SDK's `private` default is False; nobody here chose that.

    The parameter appeared in SDK 2.14 and the call sites never mentioned
    it, so every URL an analyst looked up -- and the answer -- was shared
    with third-party sources and entered public feeds. The vendored 2.13
    source a developer would read does not have the parameter at all,
    which is how it stayed invisible.
    """

    URL = "https://evil.example/payload"

    def _service(self, *, share):
        settings = Settings()
        settings.titanium_cloud.username = "u"
        settings.titanium_cloud.password = "p"
        settings.titanium_cloud.share_url_lookups = share
        service = TitaniumCloudNetworkService(settings)
        api = MagicMock()
        api.get_url_report.return_value = sdk_response(200, {"rl": {}})
        api.get_downloaded_files.return_value = sdk_response(200, {"rl": {"files": []}})
        service.__dict__["_url_threat"] = api
        return service, api

    def test_the_report_is_asked_for_privately(self):
        service, api = self._service(share=False)

        service.analyze_url(self.URL)

        assert api.get_url_report.call_args.kwargs["private"] is True

    def test_every_url_endpoint_agrees(self):
        """All three take the flag; one of them forgetting is the whole bug."""
        service, api = self._service(share=False)

        service.analyze_url(self.URL)
        service.get_files_from_url(self.URL)
        service.get_files_from_url_aggregated(self.URL, max_results=10)

        for call in (api.get_url_report, api.get_downloaded_files):
            assert call.call_args.kwargs["private"] is True, call
        # The walk asks the same endpoint the first page came from, so the
        # flag has to be on every page of it and not only the first.
        assert api.get_downloaded_files.call_count == 2

    def test_sharing_stays_one_setting_away(self):
        """Contributing to the feeds is legitimate -- it just has to be chosen."""
        service, api = self._service(share=True)

        service.analyze_url(self.URL)

        assert api.get_url_report.call_args.kwargs["private"] is False
