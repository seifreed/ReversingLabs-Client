"""Tests for newly added and fixed A1000 YARA and dynamic-report wrappers.

These tests cover:
- Fixed wrappers: ``create_yara_ruleset`` (now exposes publish/ticloud),
  ``yara_cloud_retro_scan`` (now mode-aware), ``yara_local_retro_scan``
  (mode-aware), ``get_yara_cloud_retro_scan_status`` (now takes
  ``ruleset_name``), ``get_yara_local_retro_scan_status`` (no args),
  ``check_dynamic_analysis_status`` (now requires ``report_format``),
  ``get_yara_content`` (now calls ``get_yara_ruleset_contents``).
- New wrappers: ``create_dynamic_report``, the YARA repository
  CRUD/list family, publish single/all, and the YARA update job
  controls.

The SDK client is replaced by a ``MagicMock`` so no real network calls
happen. The tests verify which SDK method a wrapper reaches for and what
it makes of the answer on the success and failure paths. Which argument
lands on which SDK parameter is asserted by name in
``tests/test_sdk_conformance.py``'s ``A1000_ARGUMENT_BINDINGS``, against
the installed SDK's real signature, rather than by pinning the call form
here.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner, Result

from rl_cli.cli.commands._shared_inputs import MAX_LIMIT
from rl_cli.cli.main import cli
from rl_cli.models.yara_repository import YaraRepositorySpec
from rl_cli.services.a1000 import A1000ReportService, A1000Session, A1000YaraService
from rl_cli.services.a1000.yara import _MAX_MATCH_PAGES, _MAX_REPOSITORY_PAGES
from tests.cli_support import SHA256, flat
from tests.conftest import sdk_response

# Run one ``a1000`` command through the root CLI, optionally deciding what
# the matches endpoint answers.
CliRun = Callable[..., Result]


def repository_spec(
    *,
    source_branch: str | None = "main",
    api_token: str = "",
    mode: str = "manual",
) -> YaraRepositorySpec:
    """A repository spec, varying only the field a test is about.

    The five fields go to the service as one value, so a test that names
    all five to change one is retyping the very pairing the spec exists to
    make — and would be the last place a transposition could still be
    written by hand.
    """
    return YaraRepositorySpec(
        repository_url="https://github.com/example/rules",
        name="example",
        source_branch=source_branch,
        api_token=api_token,
        import_update_preferences=mode,
    )


@pytest.fixture
def session(a1000_session: A1000Session) -> A1000Session:
    """A session wired to a mock SDK client (no network)."""
    a1000_session.client = MagicMock()
    return a1000_session


@pytest.fixture
def cli_run(monkeypatch: pytest.MonkeyPatch) -> CliRun:
    """Invoke through ``cli`` rather than the ``a1000`` group.

    The non-zero exit status comes from ``cli.result_callback``, so a run
    started at the group cannot tell a refusal from an empty answer —
    which is the whole difference these tests are about.
    """
    client = MagicMock()
    monkeypatch.setattr(A1000Session, "_open", lambda session: setattr(session, "client", client))

    def run(args: list[str], matches: dict[str, Any] | None = None) -> Result:
        client.get_yara_ruleset_matches_v2.return_value = sdk_response(
            200, matches if matches is not None else {"count": 0, "next": None, "results": []}
        )
        return CliRunner().invoke(cli, ["a1000", *args])

    return run


@pytest.fixture
def appliance_cli(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Any]:
    """One ``a1000`` command through the root CLI, with the endpoint armed.

    ``cli_run`` speaks for the matches endpoint only; the listings answer
    their own SDK methods, and the difference between "this appliance has
    no rulesets" and "we could not read the answer" is an exit status.
    """
    client = MagicMock()
    monkeypatch.setattr(A1000Session, "_open", lambda session: setattr(session, "client", client))

    def run(args: list[str], sdk_method: str, body: Any) -> Result:
        getattr(client, sdk_method).return_value = sdk_response(200, body)
        return CliRunner().invoke(cli, ["a1000", *args])

    return run


class TestListingsAreNotStricterThanTheSdk:
    """The SDK reads every page key as ``.get(key, [])``; so must this.

    Refusing a body that simply does not carry the key turned a 200 with
    an empty page into "✗ Failed to list YARA rulesets", exit 1, for an
    appliance that was answering perfectly well.
    """

    def test_a_ruleset_page_without_the_key_exits_zero(self, appliance_cli) -> None:
        result = appliance_cli(
            ["yara-list"], "get_yara_rulesets_on_the_appliance_v2", {"rulesets": []}
        )

        assert result.exit_code == 0, result.output
        assert "No YARA rulesets on this appliance" in flat(result)

    def test_a_ruleset_key_that_is_not_a_list_still_exits_one(self, appliance_cli) -> None:
        result = appliance_cli(
            ["yara-list"], "get_yara_rulesets_on_the_appliance_v2", {"results": "conti"}
        )

        assert result.exit_code == 1, result.output
        assert "Failed to list YARA rulesets" in flat(result)

    def test_a_repository_page_without_the_key_exits_zero(self, yara: A1000YaraService) -> None:
        yara.client.get_yara_repositories.return_value = sdk_response(200, {"count": 0})

        assert yara.list_yara_repositories() == []

    def test_a_repository_key_that_is_not_a_list_is_a_failure(self, yara: A1000YaraService) -> None:
        yara.client.get_yara_repositories.return_value = sdk_response(200, {"results": {}})

        assert yara.list_yara_repositories() is None

    def test_a_non_positive_page_size_is_refused(self, yara: A1000YaraService) -> None:
        yara.output = MagicMock()

        assert yara.list_yara_repositories(page_size=0) is None
        assert yara.output.error.called
        yara.client.get_yara_repositories.assert_not_called()

    def test_a_non_positive_page_number_is_refused(self, yara: A1000YaraService) -> None:
        yara.output = MagicMock()

        assert yara.list_yara_repositories(page_number=0) is None
        assert yara.output.error.called
        yara.client.get_yara_repositories.assert_not_called()

    def test_the_aggregated_walk_refuses_a_non_positive_page_size(
        self, yara: A1000YaraService
    ) -> None:
        yara.output = MagicMock()

        assert yara.list_yara_repositories_aggregated(page_size=0) is None
        assert yara.output.error.called
        yara.client.get_yara_repositories.assert_not_called()


@pytest.fixture
def yara(session: A1000Session) -> A1000YaraService:
    return session.service(A1000YaraService)


@pytest.fixture
def reports(session: A1000Session) -> A1000ReportService:
    return session.service(A1000ReportService)


# --- Fixed wrappers ----------------------------------------------------------


class TestCreateYaraRulesetExposesAllParams:
    """``publish`` and ``ticloud`` are the caller's to set, defaulting to unset.

    Which parameter each of the four values lands on is asserted by name in
    ``A1000_ARGUMENT_BINDINGS``; here it is the answer the analyst gets.
    """

    def test_a_stored_ruleset_is_a_success(self, yara: A1000YaraService) -> None:
        yara.client.create_or_update_yara_ruleset.return_value = sdk_response(200)

        assert yara.create_yara_ruleset("my-rules", "rule x {}") is True

    @pytest.mark.parametrize("content", ["", "   ", "\n\t"])
    def test_a_ruleset_with_no_rules_is_refused_before_the_post(
        self, yara: A1000YaraService, content: str
    ) -> None:
        """The SDK drops empty content instead of sending it.

        Its payload builder assembles the body under ``if content:``, so an
        empty ruleset became a POST of a name alone -- which any 2xx then
        reported as stored.
        """
        assert yara.create_yara_ruleset("my-rules", content) is False
        yara.client.create_or_update_yara_ruleset.assert_not_called()

    def test_publish_and_ticloud_are_accepted(self, yara: A1000YaraService) -> None:
        yara.client.create_or_update_yara_ruleset.return_value = sdk_response(200)

        assert yara.create_yara_ruleset("myrules", "rule y {}", publish=True, ticloud=False) is True

    def test_non_200_returns_false(self, yara: A1000YaraService) -> None:
        yara.client.create_or_update_yara_ruleset.return_value = sdk_response(409)
        assert yara.create_yara_ruleset("myrules", "c") is False


class TestYaraCloudRetroScanMode:
    @pytest.mark.parametrize("mode", ["start", "stop", "clear", "START", "Stop"])
    def test_every_spelling_of_a_valid_mode_is_accepted(
        self, yara: A1000YaraService, mode: str
    ) -> None:
        """Accepted means the endpoint was asked, which is what ``True`` reports.

        That the mode goes to ``operation`` upper-cased and the name to
        ``ruleset_name`` — the two are reversed between wrapper and SDK — is
        asserted by name in ``A1000_ARGUMENT_BINDINGS``.
        """
        yara.client.start_or_stop_yara_cloud_retro_scan.return_value = sdk_response(
            200, {"ok": True}
        )

        assert yara.yara_cloud_retro_scan("myrules", mode) is True

    def test_invalid_mode_is_a_failure_without_calling_the_sdk(
        self, yara: A1000YaraService
    ) -> None:
        assert yara.yara_cloud_retro_scan("myrules", "delete") is False
        yara.client.start_or_stop_yara_cloud_retro_scan.assert_not_called()


class TestYaraLocalRetroScanMode:
    @pytest.mark.parametrize("mode", ["start", "stop"])
    def test_valid_modes(self, yara: A1000YaraService, mode: str) -> None:
        yara.client.start_or_stop_yara_local_retro_scan.return_value = sdk_response(
            200, {"ok": True}
        )

        assert yara.yara_local_retro_scan(mode) is True
        yara.client.start_or_stop_yara_local_retro_scan.assert_called_once_with(mode.upper())

    def test_clear_is_rejected_for_local(self, yara: A1000YaraService) -> None:
        assert yara.yara_local_retro_scan("clear") is False
        yara.client.start_or_stop_yara_local_retro_scan.assert_not_called()


class TestRetroScanActionsReadTheStatusNotTheBody:
    """Both endpoints are action POSTs that document no response body.

    Reading one as JSON reported a scan that had in fact started as a
    failure: a 200 with ``{}`` printed "Failed to start", and a 204 got as
    far as the raw ``Expecting value: line 1 column 1``.
    """

    def test_cloud_empty_body_is_success(self, yara: A1000YaraService) -> None:
        yara.client.start_or_stop_yara_cloud_retro_scan.return_value = sdk_response(200, {})

        assert yara.yara_cloud_retro_scan("myrules", "start") is True

    def test_cloud_no_content_is_success(self, yara: A1000YaraService) -> None:
        response = sdk_response(204)
        response.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
        yara.client.start_or_stop_yara_cloud_retro_scan.return_value = response

        assert yara.yara_cloud_retro_scan("myrules", "stop") is True

    def test_local_no_content_is_success(self, yara: A1000YaraService) -> None:
        response = sdk_response(204)
        response.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
        yara.client.start_or_stop_yara_local_retro_scan.return_value = response

        assert yara.yara_local_retro_scan("start") is True

    def test_a_rejected_action_is_still_a_failure(self, yara: A1000YaraService) -> None:
        yara.client.start_or_stop_yara_local_retro_scan.return_value = sdk_response(500)

        assert yara.yara_local_retro_scan("start") is False


class TestYaraRetroScanStatusSignatures:
    def test_cloud_status_uses_ruleset_name(self, yara: A1000YaraService) -> None:
        yara.client.get_yara_cloud_retro_scan_status.return_value = sdk_response(
            200, {"state": "running"}
        )

        result = yara.get_yara_cloud_retro_scan_status("ruleset-A")

        assert result == {"state": "running"}
        yara.client.get_yara_cloud_retro_scan_status.assert_called_once_with("ruleset-A")

    def test_local_status_takes_no_args(self, yara: A1000YaraService) -> None:
        yara.client.get_yara_local_retro_scan_status.return_value = sdk_response(
            200, {"state": "idle"}
        )

        result = yara.get_yara_local_retro_scan_status()

        assert result == {"state": "idle"}
        yara.client.get_yara_local_retro_scan_status.assert_called_once_with()


class TestCheckDynamicAnalysisStatusSignature:
    """The format is required, and only pdf/html exist.

    That it reaches ``report_format`` rather than being dropped — the state
    the wrapper shipped in — is asserted by name in ``A1000_ARGUMENT_BINDINGS``.
    """

    def test_a_ready_status_is_read_through(self, reports: A1000ReportService) -> None:
        reports.client.check_dynamic_analysis_report_status.return_value = sdk_response(
            200, {"status": "done"}
        )

        result = reports.check_dynamic_analysis_status("a" * 40, "html")

        assert result == {"status": "done"}

    def test_invalid_format_short_circuits(self, reports: A1000ReportService) -> None:
        result = reports.check_dynamic_analysis_status("a" * 40, "docx")

        assert result is None
        reports.client.check_dynamic_analysis_report_status.assert_not_called()


class TestGetYaraContentCallsCorrectSdkMethod:
    def test_uses_get_yara_ruleset_contents(self, yara: A1000YaraService) -> None:
        yara.client.get_yara_ruleset_contents.return_value = sdk_response(200, text="rule abc {}")

        text = yara.get_yara_content("abc")

        assert text == "rule abc {}"
        yara.client.get_yara_ruleset_contents.assert_called_once_with("abc")


class TestYaraRulesetListCoversTheAppliance:
    """``owner_type`` defaults to ``my``; the command speaks for the appliance."""

    def test_list_asks_for_every_owner(self, yara: A1000YaraService) -> None:
        yara.client.get_yara_rulesets_on_the_appliance_v2.return_value = sdk_response(
            200, {"results": [{"name": "system-ruleset"}]}
        )

        result = yara.list_yara_rulesets()

        assert result == [{"name": "system-ruleset"}]
        yara.client.get_yara_rulesets_on_the_appliance_v2.assert_called_once_with(owner_type="all")

    def test_a_bare_list_body_is_the_rulesets(self, yara: A1000YaraService) -> None:
        """Reading ``results`` off a list leaked an AttributeError at the user."""
        yara.client.get_yara_rulesets_on_the_appliance_v2.return_value = sdk_response(
            200, [{"name": "a"}, {"name": "b"}]
        )

        assert yara.list_yara_rulesets() == [{"name": "a"}, {"name": "b"}]

    def test_a_body_that_is_neither_is_a_failure(self, yara: A1000YaraService) -> None:
        yara.client.get_yara_rulesets_on_the_appliance_v2.return_value = sdk_response(200, "nope")

        assert yara.list_yara_rulesets() is None


class TestYaraMatchesReadEveryPage:
    """A hash filter is only as truthful as the set it runs over."""

    @staticmethod
    def _page(entries: list[Any], *, count: int, more: bool) -> MagicMock:
        return sdk_response(
            200,
            {
                "count": count,
                "next": "https://appliance/api/v2/matches?page=2" if more else None,
                "results": entries,
            },
        )

    def test_a_hash_on_a_later_page_is_found(self, yara: A1000YaraService) -> None:
        first = self._page([{"sha256": "a" * 64}], count=2, more=True)
        second = self._page([{"sha256": "b" * 64}], count=2, more=False)
        yara.client.get_yara_ruleset_matches_v2.side_effect = [first, second]

        assert yara.get_yara_matches("myrules", "B" * 64) == [{"sha256": "b" * 64}]
        # Page one is asked for exactly as before; the follow-ups reuse the
        # size the appliance chose, or the slices would not line up.
        assert yara.client.get_yara_ruleset_matches_v2.call_args_list[0].args == ("myrules",)
        assert yara.client.get_yara_ruleset_matches_v2.call_args_list[1].kwargs == {
            "page": 2,
            "page_size": 1,
        }

    def test_every_page_is_collected(self, yara: A1000YaraService) -> None:
        yara.client.get_yara_ruleset_matches_v2.side_effect = [
            self._page([{"sha1": "1"}, {"sha1": "2"}], count=5, more=True),
            self._page([{"sha1": "3"}, {"sha1": "4"}], count=5, more=True),
            self._page([{"sha1": "5"}], count=5, more=False),
        ]

        matches = yara.get_yara_matches("myrules")

        assert matches is not None and len(matches) == 5

    def test_a_single_page_is_still_one_request(self, yara: A1000YaraService) -> None:
        yara.client.get_yara_ruleset_matches_v2.return_value = self._page(
            [{"sha1": "1"}], count=1, more=False
        )

        assert yara.get_yara_matches("myrules") == [{"sha1": "1"}]
        assert yara.client.get_yara_ruleset_matches_v2.call_count == 1

    def test_a_page_that_repeats_next_forever_still_terminates(
        self, yara: A1000YaraService
    ) -> None:
        """An envelope that always promises more must not loop the CLI."""
        yara.output = MagicMock()
        yara.client.get_yara_ruleset_matches_v2.return_value = self._page(
            [{"sha1": "1"}], count=9, more=True
        )

        matches = yara.get_yara_matches("myrules")

        assert matches is not None and len(matches) == _MAX_MATCH_PAGES
        assert yara.client.get_yara_ruleset_matches_v2.call_count == _MAX_MATCH_PAGES
        assert "partial" in yara.output.warning.call_args.args[0]

    def test_the_match_budget_reaches_the_largest_listing_this_cli_fetches(self) -> None:
        """The number, not just the fact of a cap.

        A walk that stops short of a match reports a *failed* lookup for a
        ruleset it simply did not finish reading, so what the budget has to
        cover is the largest listing this CLI fetches at all. The page size
        is the appliance's to choose here — this endpoint is asked for
        whatever it sent on page one — and 500 is the largest page any
        A1000 listing serves (SDK ``a1000.py`` defaults ``page_size=500``
        on every network pivot).
        """
        assert MAX_LIMIT <= _MAX_MATCH_PAGES * 500

    def test_an_empty_page_in_the_middle_does_not_end_the_walk(
        self, yara: A1000YaraService
    ) -> None:
        """Stopping there handed back a partial set reported as the whole answer.

        The appliance filters each page server-side, so a page can come
        back empty with pages of matches still behind it.
        """
        yara.client.get_yara_ruleset_matches_v2.side_effect = [
            self._page([{"sha1": "1"}], count=2, more=True),
            self._page([], count=2, more=True),
            self._page([{"sha1": "2"}], count=2, more=False),
        ]

        assert yara.get_yara_matches("myrules") == [{"sha1": "1"}, {"sha1": "2"}]

    def test_the_page_size_counts_what_the_server_sent_not_what_survived(
        self, yara: A1000YaraService
    ) -> None:
        """Sizing page two by the filtered list shifted every later boundary.

        Page one holds three records, one of them malformed. Asking for
        page two in a size of two re-reads a record page one already had.
        """
        yara.client.get_yara_ruleset_matches_v2.side_effect = [
            self._page([{"sha1": "1"}, "malformed", {"sha1": "2"}], count=4, more=True),
            self._page([{"sha1": "3"}], count=4, more=False),
        ]

        yara.get_yara_matches("myrules")

        assert yara.client.get_yara_ruleset_matches_v2.call_args_list[1].kwargs == {
            "page": 2,
            "page_size": 3,
        }

    def test_a_results_key_that_is_not_a_list_is_a_failure(self, yara: A1000YaraService) -> None:
        """``.get("results", [])`` reported an unreadable answer as no matches."""
        yara.output = MagicMock()
        yara.client.get_yara_ruleset_matches_v2.return_value = sdk_response(
            200, {"results": {"sha1": "1"}}
        )

        assert yara.get_yara_matches("myrules") is None

        assert "carried no results list" in yara.output.error.call_args.args[0]

    def test_the_failure_names_the_page_it_could_not_read(self, yara: A1000YaraService) -> None:
        """A walk that broke on page two looked exactly like one that broke on page one."""
        yara.output = MagicMock()
        yara.client.get_yara_ruleset_matches_v2.side_effect = [
            sdk_response(200, {"results": [{"sha1": "1"}], "next": "?page=2"}),
            sdk_response(200, {"results": "later"}),
        ]

        assert yara.get_yara_matches("myrules") is None

        assert "page 2 of matches for 'myrules'" in yara.output.error.call_args.args[0]

    def test_a_page_with_no_results_key_is_empty_not_failed(self, yara: A1000YaraService) -> None:
        """The SDK reads its page key with ``.get(key, [])``; so must this."""
        yara.output = MagicMock()
        yara.client.get_yara_ruleset_matches_v2.return_value = sdk_response(
            200, {"count": 0, "next": None}
        )

        assert yara.get_yara_matches("myrules") == []
        assert not yara.output.error.called


class TestYaraMatchesGuardsTheHashItFilters:
    """The only sample-identifying argument in these services without a guard.

    The filter is ours, not the endpoint's, so nonsense simply matched
    nothing: ``yara-matches conti not-a-hash`` printed "No YARA matches
    found" and exited 0, which is exactly what a real miss looks like.
    """

    def test_nonsense_never_reaches_the_appliance(self, yara: A1000YaraService) -> None:
        yara.output = MagicMock()

        assert yara.get_yara_matches("conti", "not-a-hash") is None

        assert "not-a-hash" in yara.output.error.call_args.args[0]
        yara.client.get_yara_ruleset_matches_v2.assert_not_called()

    def test_a_sha512_is_refused_rather_than_answered_no(self, yara: A1000YaraService) -> None:
        """The filter reads sha256/sha1/md5, so a SHA512 could never match."""
        yara.output = MagicMock()

        assert yara.get_yara_matches("conti", "e" * 128) is None

        assert "SHA512" in yara.output.error.call_args.args[0]
        yara.client.get_yara_ruleset_matches_v2.assert_not_called()

    def test_a_padded_uppercase_hash_is_normalised_before_it_filters(
        self, yara: A1000YaraService
    ) -> None:
        yara.client.get_yara_ruleset_matches_v2.return_value = sdk_response(
            200, {"count": 1, "next": None, "results": [{"sha256": "a" * 64}]}
        )

        assert yara.get_yara_matches("myrules", f"  {'A' * 64}\n") == [{"sha256": "a" * 64}]


class TestEveryWrapperThatTakesARulesetNameGuardsIt:
    """The name goes into a URL the SDK builds by string formatting.

    ``get_yara_ruleset_matches_v2`` appends ``f"name={ruleset_name}"`` to
    a query string and the retro and publish endpoints format it into a
    path segment, with no ``quote()`` between the command line and the
    request. An unguarded name is therefore not a lookup that fails, it
    is a different request that succeeds — the defect
    ``normalize_domain`` closed for the domain endpoints, left open here.

    Table-driven because the previous state of this service was the whole
    bug: the same argument was guarded on some of these methods and not
    on others, and it was the URL-building half that went unguarded.
    """

    WRAPPERS: ClassVar[dict[str, Callable[[A1000YaraService, str], Any]]] = {
        "get_yara_content": lambda service, name: service.get_yara_content(name),
        "get_yara_matches": lambda service, name: service.get_yara_matches(name),
        "publish_yara_ruleset": lambda service, name: service.publish_yara_ruleset(name),
        "yara_cloud_retro_scan": lambda service, name: service.yara_cloud_retro_scan(name, "start"),
        "get_yara_cloud_retro_scan_status": (
            lambda service, name: service.get_yara_cloud_retro_scan_status(name)
        ),
        "create_yara_ruleset": lambda service, name: service.create_yara_ruleset(name, "rule x {}"),
        "toggle_yara_ruleset": lambda service, name: service.toggle_yara_ruleset(name, True),
        "delete_yara_ruleset": lambda service, name: service.delete_yara_ruleset(name),
    }

    # The parameter names a ruleset name arrives under. The sweep below
    # matches on these, so a wrapper spelling it some other way is one the
    # sentinel cannot see -- keep the spellings few and keep them here.
    RULESET_PARAMETERS: ClassVar = frozenset({"name", "ruleset", "ruleset_name"})

    REFUSED = (
        "../../../api/samples/v2/list/details",
        "prod#ignored",
        "a&name=core",
        "prod?page=2",
        "prod%2fcore",
        "prod core",
        "my\nrules",
        "",
        "   ",
        "ab",
        "a" * 49,
    )

    def _takes_a_ruleset_name(self) -> set[str]:
        """Every public wrapper on the service with a ruleset-name parameter."""
        return {
            name
            for name, function in inspect.getmembers(A1000YaraService, inspect.isfunction)
            if not name.startswith("_")
            and function.__module__ == A1000YaraService.__module__
            and self.RULESET_PARAMETERS & set(inspect.signature(function).parameters)
        }

    def test_the_table_lists_every_wrapper_that_takes_a_ruleset_name(self) -> None:
        """The sentinel: the table has to cover the real set, read off the service.

        A hand-maintained list of "every wrapper that takes a ruleset name"
        is the guard's coverage, so a ninth wrapper added without a row
        would be swept over by nothing and shipped unguarded.
        """
        real = self._takes_a_ruleset_name()

        assert real, "no wrapper takes a ruleset name; this table now covers nothing"
        assert real == set(self.WRAPPERS), (
            f"the table misses {sorted(real - set(self.WRAPPERS))} "
            f"and invents {sorted(set(self.WRAPPERS) - real)}"
        )

    @pytest.mark.parametrize("wrapper", WRAPPERS.values(), ids=list(WRAPPERS))
    @pytest.mark.parametrize("name", REFUSED)
    def test_a_name_the_appliance_would_not_store_never_leaves_here(
        self, yara: A1000YaraService, wrapper: Callable[..., Any], name: str
    ) -> None:
        yara.output = MagicMock()

        assert not wrapper(yara, name)

        assert "Invalid YARA ruleset name" in yara.output.error.call_args.args[0]
        assert yara.client.method_calls == [], "an unchecked ruleset name reached the SDK"

    @pytest.mark.parametrize("wrapper", WRAPPERS.values(), ids=list(WRAPPERS))
    def test_a_pasted_name_is_sent_trimmed_but_not_recased(
        self, yara: A1000YaraService, wrapper: Callable[..., Any]
    ) -> None:
        """Ruleset names are case-sensitive; lowercasing asks about another one."""
        yara.client.get_yara_ruleset_matches_v2.return_value = sdk_response(
            200, {"count": 0, "next": None, "results": []}
        )
        for method in ("get_yara_ruleset_contents", "publish_single_yara_ruleset"):
            getattr(yara.client, method).return_value = sdk_response(200, text="rule x {}")
        for method in (
            "start_or_stop_yara_cloud_retro_scan",
            "get_yara_cloud_retro_scan_status",
            "create_or_update_yara_ruleset",
            "enable_or_disable_yara_ruleset",
            "delete_yara_ruleset",
        ):
            getattr(yara.client, method).return_value = sdk_response(200, {})

        wrapper(yara, "  MixedCase_2024\n")

        sent = [
            argument
            for _, args, kwargs in yara.client.method_calls
            for argument in (*args, *kwargs.values())
            if isinstance(argument, str)
        ]
        assert "MixedCase_2024" in sent


class TestATruncatedWalkCannotAnswerNo:
    """The cap is ours and the per-sample filter runs on this side.

    So an empty answer off a walk that stopped early is not "ruleset X
    did not hit sample Y", it is "we did not look at all of it". It used
    to warn that the list was partial and hand back the empty list
    anyway, which the command reported as a miss at exit 0.
    """

    @staticmethod
    def _endless(yara: A1000YaraService, entries: list[Any]) -> None:
        yara.client.get_yara_ruleset_matches_v2.return_value = sdk_response(
            200, {"count": 9999, "next": "https://appliance/next", "results": entries}
        )

    def test_no_hit_over_a_truncated_walk_is_a_failure_not_a_miss(
        self, yara: A1000YaraService
    ) -> None:
        yara.output = MagicMock()
        self._endless(yara, [{"sha256": "a" * 64}])

        assert yara.get_yara_matches("myrules", "b" * 64) is None

        assert yara.matches_are_partial
        assert "Cannot say whether" in yara.output.error.call_args.args[0]

    def test_a_hit_inside_the_pages_that_were_read_is_still_answered(
        self, yara: A1000YaraService
    ) -> None:
        yara.output = MagicMock()
        self._endless(yara, [{"sha256": "a" * 64}])

        assert yara.get_yara_matches("myrules", "a" * 64)

        assert yara.matches_are_partial
        assert not yara.output.error.called

    def test_a_complete_walk_that_found_nothing_still_answers_nothing(
        self, yara: A1000YaraService
    ) -> None:
        yara.output = MagicMock()
        yara.client.get_yara_ruleset_matches_v2.return_value = sdk_response(
            200, {"count": 0, "next": None, "results": []}
        )

        assert yara.get_yara_matches("myrules", "a" * 64) == []

        assert not yara.matches_are_partial
        assert not yara.output.error.called

    def test_the_flag_does_not_survive_into_the_next_lookup(self, yara: A1000YaraService) -> None:
        """A stale flag would make a complete walk look truncated."""
        yara.output = MagicMock()
        self._endless(yara, [{"sha256": "a" * 64}])
        yara.get_yara_matches("myrules")
        assert yara.matches_are_partial

        yara.client.get_yara_ruleset_matches_v2.return_value = sdk_response(
            200, {"count": 1, "next": None, "results": [{"sha256": "a" * 64}]}
        )

        assert yara.get_yara_matches("myrules") == [{"sha256": "a" * 64}]
        assert not yara.matches_are_partial

    def test_a_ruleset_name_the_guard_refused_does_not_leave_the_flag_set(
        self, yara: A1000YaraService
    ) -> None:
        """The clearing happens on the way in, so a call that never lands clears too."""
        yara.output = MagicMock()
        self._endless(yara, [{"sha256": "a" * 64}])
        yara.get_yara_matches("myrules")
        assert yara.matches_are_partial

        assert yara.get_yara_matches("no") is None
        assert not yara.matches_are_partial

    def test_a_walk_that_raises_does_not_leave_the_flag_set(self, yara: A1000YaraService) -> None:
        """The flag is cleared on the way in, so nothing has to unwind it."""
        yara.output = MagicMock()
        self._endless(yara, [{"sha256": "a" * 64}])
        yara.get_yara_matches("myrules")
        assert yara.matches_are_partial

        yara.client.get_yara_ruleset_matches_v2.side_effect = RuntimeError("appliance said no")

        assert yara.get_yara_matches("myrules") is None
        assert not yara.matches_are_partial

    def test_the_flag_does_not_survive_into_a_lookup_that_walks_nothing(
        self, yara: A1000YaraService
    ) -> None:
        """A call that measures no walk must not answer for the one before it."""
        yara.output = MagicMock()
        self._endless(yara, [{"sha256": "a" * 64}])
        yara.get_yara_matches("myrules")
        assert yara.matches_are_partial

        yara.client.get_yara_rulesets_on_the_appliance_v2.return_value = sdk_response(
            200, {"count": 1, "results": [{"name": "myrules"}]}
        )

        assert yara.list_yara_rulesets() == [{"name": "myrules"}]
        assert not yara.matches_are_partial


class TestYaraMatchesThroughTheCli:
    """The analyst sees a message and an exit status, not a return value."""

    def test_a_bad_hash_is_named_and_exits_non_zero(self, cli_run: CliRun) -> None:
        result = cli_run(["yara-matches", "conti", "not-a-hash"])

        assert result.exit_code == 1
        assert "Invalid hash format: not-a-hash" in flat(result)
        assert "No YARA matches found" not in flat(result)

    def test_a_real_miss_still_reads_as_a_miss(self, cli_run: CliRun) -> None:
        result = cli_run(
            ["yara-matches", "conti", SHA256],
            {"count": 0, "next": None, "results": []},
        )

        assert result.exit_code == 0
        assert "No YARA matches found" in flat(result)


# --- New G1: dynamic report create -------------------------------------------


class TestCreateDynamicReport:
    """A 201 and a 202 both mean the report was accepted, body and all.

    Both formats reach ``report_format`` by name in ``A1000_ARGUMENT_BINDINGS``.
    """

    def test_a_created_report_hands_back_its_urls(self, reports: A1000ReportService) -> None:
        reports.client.create_dynamic_analysis_report.return_value = sdk_response(
            201, {"status_url": "/x", "download_url": "/y"}
        )

        result = reports.create_dynamic_report("a" * 40, "pdf")

        assert result == {"status_url": "/x", "download_url": "/y"}

    def test_an_accepted_report_is_not_read_as_a_failure(self, reports: A1000ReportService) -> None:
        reports.client.create_dynamic_analysis_report.return_value = sdk_response(
            202, {"queued": True}
        )

        result = reports.create_dynamic_report("a" * 40, "html")

        assert result == {"queued": True}

    def test_rejects_non_sha1(self, reports: A1000ReportService) -> None:
        # A 32-char MD5 should be refused per the SDK contract.
        result = reports.create_dynamic_report("d" * 32, "pdf")

        assert result is None
        reports.client.create_dynamic_analysis_report.assert_not_called()

    def test_rejects_invalid_hash(self, reports: A1000ReportService) -> None:
        result = reports.create_dynamic_report("not-a-hash", "pdf")

        assert result is None
        reports.client.create_dynamic_analysis_report.assert_not_called()

    def test_rejects_invalid_format(self, reports: A1000ReportService) -> None:
        result = reports.create_dynamic_report("a" * 40, "xml")

        assert result is None
        reports.client.create_dynamic_analysis_report.assert_not_called()


# --- New G2-G6: YARA Online Source repositories ------------------------------


class TestYaraRepositories:
    def test_list_paged_unwraps_the_envelope(self, yara: A1000YaraService) -> None:
        """Repositories, like every sibling wrapper — not the paging envelope."""
        yara.client.get_yara_repositories.return_value = sdk_response(
            200, {"count": 1, "next": None, "previous": None, "results": [{"id": 1}]}
        )

        result = yara.list_yara_repositories(active_filter="user", page_size=50, page_number=1)

        assert result == [{"id": 1}]
        yara.client.get_yara_repositories.assert_called_once_with(
            active_filter="user", page_size=50, page_number=1
        )

    def test_no_repositories_is_empty_not_truthy(self, yara: A1000YaraService) -> None:
        """An envelope around nothing is still nothing, so the command can say so."""
        yara.client.get_yara_repositories.return_value = sdk_response(
            200, {"count": 0, "next": None, "previous": None, "results": []}
        )

        assert yara.list_yara_repositories() == []

    @staticmethod
    def _pages(*pages: list[dict[str, Any]]) -> list[Any]:
        """Each page as the endpoint answers it, the last carrying no ``next``."""
        return [
            sdk_response(200, {"next": "/x" if rest else None, "results": page})
            for rest, page in ((pages[index + 1 :], page) for index, page in enumerate(pages))
        ]

    def test_list_aggregated_joins_the_pages(self, yara: A1000YaraService) -> None:
        yara.client.get_yara_repositories.side_effect = self._pages(
            [{"id": 1}, {"id": 2}], [{"id": 3}]
        )

        assert yara.list_yara_repositories_aggregated() == [{"id": 1}, {"id": 2}, {"id": 3}]

    def test_a_sparse_page_mid_listing_does_not_truncate_the_answer(
        self, yara: A1000YaraService
    ) -> None:
        """A page that carried nothing is not the end of the listing.

        Only a missing ``next`` is. Stopping on an empty page hands back
        the repositories fetched so far as though they were all of them,
        and this walk says nothing about having stopped early.
        """
        yara.client.get_yara_repositories.side_effect = [
            sdk_response(200, {"next": "/2", "results": [{"id": 1}]}),
            sdk_response(200, {"next": "/3", "results": []}),
            sdk_response(200, {"next": None, "results": [{"id": 3}]}),
        ]

        assert yara.list_yara_repositories_aggregated() == [{"id": 1}, {"id": 3}]
        assert yara.client.get_yara_repositories.call_count == 3

    def test_list_aggregated_stops_at_the_cap_it_was_given(self, yara: A1000YaraService) -> None:
        """The cap counts repositories, and no page is fetched to be discarded."""
        yara.client.get_yara_repositories.side_effect = self._pages(
            [{"id": 1}, {"id": 2}], [{"id": 3}, {"id": 4}]
        )

        result = yara.list_yara_repositories_aggregated(max_results=3)

        assert result == [{"id": 1}, {"id": 2}, {"id": 3}]
        assert yara.client.get_yara_repositories.call_count == 2

    def test_a_cap_of_zero_is_a_cap(self, yara: A1000YaraService) -> None:
        """``max_results=0`` used to be read as "no cap" and returned everything."""
        yara.client.get_yara_repositories.side_effect = self._pages([{"id": 1}, {"id": 2}])

        assert yara.list_yara_repositories_aggregated(max_results=0) == []

    def test_an_unreadable_page_is_a_failure_not_an_empty_listing(
        self, yara: A1000YaraService
    ) -> None:
        yara.client.get_yara_repositories.return_value = sdk_response(200, {"results": "conti"})

        assert yara.list_yara_repositories_aggregated() is None

    def test_a_walk_that_stops_making_progress_stops(self, yara: A1000YaraService) -> None:
        """``--all`` passes no cap, and the SDK's loop exits on ``next`` alone.

        Against an appliance answering an empty page that promises another,
        ``not next_page_url`` never becomes true and nothing is ever added
        for a cap to bind on, so the walk asks forever. The three other
        walks in this repo were rewritten for exactly this; this was the
        last one still handed to the SDK.

        What stops it here is the page budget rather than the empty page:
        this walk cannot tell an appliance that has stopped making progress
        from a listing whose repositories sit past a run of filtered pages,
        and only one of those two readings loses records.
        """
        asked = 0

        def a_page(**_kwargs: Any) -> Any:
            nonlocal asked
            asked += 1
            if asked > 5_000:
                raise AssertionError("the walk did not stop making requests")
            return sdk_response(200, {"count": 1, "next": "/next", "results": []})

        yara.client.get_yara_repositories.side_effect = a_page

        assert yara.list_yara_repositories_aggregated() == []
        assert asked == _MAX_REPOSITORY_PAGES

    def test_a_walk_is_bounded_even_when_every_page_is_full(self, yara: A1000YaraService) -> None:
        """A page that always carries records and always promises more.

        Nothing in the answer ever ends this, so the budget has to.
        """

        def a_page(**_kwargs: Any) -> Any:
            return sdk_response(200, {"count": 1, "next": "/next", "results": [{"id": 1}]})

        yara.client.get_yara_repositories.side_effect = a_page

        walked = yara.list_yara_repositories_aggregated()

        assert walked is not None
        assert len(walked) == _MAX_REPOSITORY_PAGES, "one record per page, one page per request"
        assert yara.client.get_yara_repositories.call_count == _MAX_REPOSITORY_PAGES

    def test_each_page_is_asked_for_by_number(self, yara: A1000YaraService) -> None:
        """The page number is this walk's own, never read out of the envelope."""
        pages = [
            sdk_response(200, {"next": "/2", "results": [{"id": 1}]}),
            sdk_response(200, {"next": None, "results": [{"id": 2}]}),
        ]
        yara.client.get_yara_repositories.side_effect = pages

        assert yara.list_yara_repositories_aggregated() == [{"id": 1}, {"id": 2}]
        assert [
            call.kwargs["page_number"] for call in yara.client.get_yara_repositories.call_args_list
        ] == [1, 2]

    def test_create(self, yara: A1000YaraService) -> None:
        yara.client.create_yara_repository.return_value = sdk_response(201, {"id": 7})

        result = yara.create_yara_repository(repository_spec(mode="auto-update"))

        assert result == {"id": 7}
        yara.client.create_yara_repository.assert_called_once_with(
            repository_url="https://github.com/example/rules",
            name="example",
            source_branch="main",
            api_token="",
            import_update_preferences=1,
        )

    @pytest.mark.parametrize("wrapper", ["create_yara_repository", "update_yara_repository"])
    def test_no_branch_leaves_the_endpoints_main_or_master_fallback_alone(
        self, yara: A1000YaraService, wrapper: str
    ) -> None:
        """Naming ``main`` for the caller broke every repo whose default is master.

        Asserted on both wrappers because only ``create`` used to make the
        substitution: ``update`` took ``source_branch: str`` and passed it
        straight through, so the one spec now reaching both had to grow the
        fallback on the update side or a ``None`` branch would have hit the
        SDK's ``WrongInputError``.
        """
        sdk_method = getattr(yara.client, wrapper)
        sdk_method.return_value = sdk_response(201, {"id": 7})
        spec = repository_spec(source_branch=None)

        if wrapper == "create_yara_repository":
            yara.create_yara_repository(spec)
        else:
            yara.update_yara_repository(7, spec)

        assert sdk_method.call_args.kwargs["source_branch"] == ""

    def test_update(self, yara: A1000YaraService) -> None:
        yara.client.update_yara_repository.return_value = sdk_response(200, {"id": 7})

        result = yara.update_yara_repository(7, repository_spec(mode="auto-import"))

        assert result == {"id": 7}
        yara.client.update_yara_repository.assert_called_once_with(
            repository_id=7,
            repository_url="https://github.com/example/rules",
            name="example",
            source_branch="main",
            api_token="",
            import_update_preferences=2,
        )

    @pytest.mark.parametrize(
        "mode,expected", [("manual", 0), ("auto-update", 1), ("auto-import", 2)]
    )
    def test_update_mode_names_map_to_the_wire_codes(
        self, yara: A1000YaraService, mode: str, expected: int
    ) -> None:
        """The integer preference code is the service's business, not the CLI's."""
        yara.client.create_yara_repository.return_value = sdk_response(201, {"id": 7})

        yara.create_yara_repository(repository_spec(mode=mode))

        kwargs = yara.client.create_yara_repository.call_args.kwargs
        assert kwargs["import_update_preferences"] == expected

    @pytest.mark.parametrize(
        "call,sdk_method",
        [
            (
                lambda service, spec: service.create_yara_repository(spec),
                "create_yara_repository",
            ),
            (
                lambda service, spec: service.update_yara_repository(7, spec),
                "update_yara_repository",
            ),
        ],
        ids=["create", "update"],
    )
    def test_unknown_update_mode_is_reported_and_never_reaches_the_sdk(
        self,
        yara: A1000YaraService,
        call: Callable[[A1000YaraService, YaraRepositorySpec], dict[str, Any] | None],
        sdk_method: str,
    ) -> None:
        """Both wrappers refuse it; only one of them used to be asked to."""
        yara.output = MagicMock()

        assert call(yara, repository_spec(mode="bogus")) is None

        getattr(yara.client, sdk_method).assert_not_called()
        (message,) = yara.output.error.call_args.args
        assert "bogus" in message and "auto-import" in message

    @pytest.mark.parametrize(
        "call,first_argument",
        [
            (
                lambda service, spec: service.create_yara_repository(spec),
                "api_token='***'",
            ),
            (
                lambda service, spec: service.update_yara_repository(7, spec),
                "update_yara_repository(7)",
            ),
        ],
        ids=["create", "update"],
    )
    def test_a_failed_call_never_prints_the_token_it_was_given(
        self,
        yara: A1000YaraService,
        call: Callable[[A1000YaraService, YaraRepositorySpec], dict[str, Any] | None],
        first_argument: str,
    ) -> None:
        """The decorator reprs the first positional argument into the error line.

        ``create_yara_repository(spec)`` makes the spec that argument, so a
        refusal the reporter has no advice for would have printed a live
        GitHub PAT to the terminal, to the log and into any ``-o json``
        pipeline. ``first_argument`` is asserted rather than only the
        token's absence, because a line that never carried the spec at all
        would pass an absence check while proving nothing: on create the
        spec is in the line, masked; on update the id is what precedes it.
        """
        yara.output = MagicMock()
        refusal = sdk_response(500, {"message": "Storage is full"})
        yara.client.create_yara_repository.return_value = refusal
        yara.client.update_yara_repository.return_value = refusal
        spec = repository_spec(api_token="ghp_livetoken")

        assert call(yara, spec) is None

        (reported,) = yara.output.error.call_args.args
        assert "ghp_livetoken" not in reported
        assert "Storage is full" in reported
        assert first_argument in reported

    def test_delete_default(self, yara: A1000YaraService) -> None:
        yara.client.delete_yara_repository.return_value = sdk_response(204)

        ok = yara.delete_yara_repository(7)

        assert ok is True
        yara.client.delete_yara_repository.assert_called_once_with(
            repository_id=7, remove_rulesets=False
        )

    def test_delete_with_rulesets(self, yara: A1000YaraService) -> None:
        yara.client.delete_yara_repository.return_value = sdk_response(200)

        ok = yara.delete_yara_repository(7, remove_rulesets=True)

        assert ok is True
        yara.client.delete_yara_repository.assert_called_once_with(
            repository_id=7, remove_rulesets=True
        )


# --- New G7-G8: publish ------------------------------------------------------


class TestAnAnswerThatIsNotAResponseIsNotASuccess:
    """``validate_response`` returned ``True`` for anything without a status.

    ``None`` was a failure and a real response was range-checked, but
    every other value fell through a ``hasattr`` to ``return True`` — so
    an SDK method that answered a dict, a string, or an object of the
    wrong kind entirely (a stub, a future SDK returning parsed bodies, a
    wrapper calling the wrong method) reported the action as done. These
    wrappers are the ones the analyst reads as "it worked".
    """

    @pytest.mark.parametrize(
        "call,sdk_method",
        [
            (lambda service: service.publish_all_yara_rulesets(), "publish_all_yara_rulesets"),
            (lambda service: service.run_yara_update(), "run_yara_update"),
            (lambda service: service.delete_yara_ruleset("myrules"), "delete_yara_ruleset"),
            (
                lambda service: service.create_yara_ruleset("myrules", "rule x {}"),
                "create_or_update_yara_ruleset",
            ),
        ],
        ids=["publish-all", "run-update", "delete", "create"],
    )
    @pytest.mark.parametrize(
        "answer", [{"code": 200, "message": "ok"}, "200 OK", object()], ids=["dict", "str", "obj"]
    )
    def test_a_body_where_a_response_belongs_is_reported_as_a_failure(
        self, yara: A1000YaraService, call: Callable[..., Any], sdk_method: str, answer: Any
    ) -> None:
        yara.output = MagicMock()
        getattr(yara.client, sdk_method).return_value = answer

        assert call(yara) is False, "a value that is not a response passed as success"

        assert "Expected a response" in yara.output.error.call_args.args[0]


class TestYaraPublish:
    def test_publish_single(self, yara: A1000YaraService) -> None:
        yara.client.publish_single_yara_ruleset.return_value = sdk_response(202)

        ok = yara.publish_yara_ruleset("ruleset-A")

        assert ok is True
        yara.client.publish_single_yara_ruleset.assert_called_once_with("ruleset-A")

    def test_publish_all(self, yara: A1000YaraService) -> None:
        yara.client.publish_all_yara_rulesets.return_value = sdk_response(202)

        ok = yara.publish_all_yara_rulesets()

        assert ok is True
        yara.client.publish_all_yara_rulesets.assert_called_once_with()

    def test_publish_failure(self, yara: A1000YaraService) -> None:
        yara.client.publish_single_yara_ruleset.return_value = sdk_response(500)

        ok = yara.publish_yara_ruleset("ruleset-A")

        assert ok is False


# --- New G9-G11: update job control ------------------------------------------


class TestYaraUpdateJob:
    def test_run_now(self, yara: A1000YaraService) -> None:
        yara.client.run_yara_update.return_value = sdk_response(200)

        ok = yara.run_yara_update()

        assert ok is True
        yara.client.run_yara_update.assert_called_once_with()

    def test_set_interval_in_seconds(self, yara: A1000YaraService) -> None:
        yara.client.set_yara_update_interval.return_value = sdk_response(200)

        ok = yara.set_yara_update_interval(3600)

        assert ok is True
        yara.client.set_yara_update_interval.assert_called_once_with(3600)

    def test_set_interval_zero_disables(self, yara: A1000YaraService) -> None:
        yara.client.set_yara_update_interval.return_value = sdk_response(200)

        ok = yara.set_yara_update_interval(0)

        assert ok is True
        yara.client.set_yara_update_interval.assert_called_once_with(0)

    def test_set_interval_rejects_negative(self, yara: A1000YaraService) -> None:
        ok = yara.set_yara_update_interval(-1)

        assert ok is False
        yara.client.set_yara_update_interval.assert_not_called()

    def test_reset_interval(self, yara: A1000YaraService) -> None:
        yara.client.reset_yara_update_interval.return_value = sdk_response(200)

        ok = yara.reset_yara_update_interval()

        assert ok is True
        yara.client.reset_yara_update_interval.assert_called_once_with()
