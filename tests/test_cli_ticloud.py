"""``rl-cli ticloud`` — reputation, search and the URL analysis guard."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from rl_cli.cli.commands._shared_inputs import MAX_LIMIT
from rl_cli.cli.commands.ticloud import ticloud
from rl_cli.cli.main import cli
from rl_cli.render.output import OutputFormat, OutputFormatter
from rl_cli.services.titanium_cloud import TitaniumCloudNetworkService, TitaniumCloudService
from rl_cli.services.titanium_cloud.api import TitaniumCloudApi
from rl_cli.services.titanium_cloud.network import _MAX_PIVOT_PAGES, _RECORDS_PER_PAGE
from rl_cli.services.titanium_cloud.service import _MAX_RECORDS_PER_PAGE, _MAX_SEARCH_PAGES
from rl_cli.storage.files import write_private_bytes
from tests.cli_support import SHA1, SHA256, flat, invoke, make_context, stub_response


@pytest.fixture
def ctx_obj(tmp_path):
    """The invocation's context, with a TitaniumCloud profile filled in.

    Every command here is about what it does with an answer, and a profile
    with no credentials gets no answer to work with: the service refuses to
    build an SDK handle rather than authenticate as the word "None"
    (``test_no_credential_means_no_client_is_built_at_all``).
    """
    context = make_context(tmp_path)
    context.settings.titanium_cloud.username = "analyst"
    context.settings.titanium_cloud.password = "s3cret"
    return context


# What a pivot says when it answered less than the corpus, whichever of the
# two reasons it was. Both remedies are named because one sentence covers
# both: a cap that stopped the walk, and a page the endpoint held back.
_NOTICE = "More results may be waiting; re-run with --all or raise --max-results to fetch them."


class TestTicloudCommands:
    def test_reputation_success(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(
            TitaniumCloudService,
            "get_file_reputation",
            lambda self, h: {"threat_status": "known"},
        )
        result = invoke(ticloud, ["reputation", SHA256], ctx_obj)
        assert result.exit_code == 0
        assert "threat_status" in result.output

    def test_reputation_failure(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(TitaniumCloudService, "get_file_reputation", lambda self, h: None)
        result = invoke(ticloud, ["reputation", SHA256], ctx_obj)
        assert result.exit_code == 0
        assert "Failed" in result.output


class TestTicloudSearchReportsWhatTheEnvelopeHeld:
    """The whole search envelope used to be handed back as one "sample"."""

    def _search(self, ctx_obj, monkeypatch, entries, args=("search", "threat_name:Evil")):
        api = MagicMock()
        api.search.return_value = stub_response({"rl": {"web_search_api": {"entries": entries}}})
        monkeypatch.setattr(
            "rl_cli.services.titanium_cloud.service.ticloud.AdvancedSearch", lambda **kw: api
        )
        return invoke(ticloud, list(args), ctx_obj)

    def test_three_hits_are_reported_and_rendered_as_three(self, ctx_obj, monkeypatch):
        entries = [{"sha1": h * 40} for h in "abc"]
        result = self._search(ctx_obj, monkeypatch, entries)
        assert result.exit_code == 0
        assert "Found 3 samples" in result.output
        assert json.loads(result.stdout) == entries

    def test_a_walk_stopped_at_its_cap_says_so(self, ctx_obj, monkeypatch):
        """The A1000 side has said this for a while; ticloud search did not.

        The service reports a partial answer through ``pages_left_unfetched``
        and deliberately names no remedy — which option asks for the rest is
        the caller's to know. Without this the command reported "Found 10000
        samples" over a corpus of 20000 and said nothing about the rest.
        """

        def left_hits_behind(service, *_args, **_kwargs):
            service.pages_left_unfetched = True
            return [{"sha1": "a" * 40}]

        monkeypatch.setattr(TitaniumCloudService, "search_samples", left_hits_behind)

        result = invoke(ticloud, ["search", "threat_name:Evil", "--limit", "10"], ctx_obj)

        assert result.exit_code == 0
        assert "raise --limit above 10" in result.output

    def test_a_complete_answer_says_nothing_about_pages(self, ctx_obj, monkeypatch):
        """The notice must be about the answer, not about every search."""
        result = self._search(ctx_obj, monkeypatch, [{"sha1": "a" * 40}])

        assert "raise --limit" not in result.output

    def test_zero_hits_report_nothing_matched(self, ctx_obj, monkeypatch):
        result = self._search(ctx_obj, monkeypatch, [])
        assert result.exit_code == 0
        assert "No samples matched" in result.output
        assert "Found 1 samples" not in result.output
        assert not ctx_obj.output.status.failed

    def test_zero_hits_on_the_console_stay_a_warning_not_an_empty_table(self, ctx_obj, monkeypatch):
        """The parseable empty dump is for machine formats; ``rich`` keeps its warning.

        A ``-o json`` consumer wants ``[]`` on stdout, but the console reader
        already has the "No samples matched" line and an empty ``[]`` under it
        would be noise — so the dump is withheld for the ``rich`` format.
        """
        ctx_obj = replace(ctx_obj, formatter=OutputFormatter(OutputFormat.RICH))
        result = self._search(ctx_obj, monkeypatch, [])
        assert result.exit_code == 0
        assert "No samples matched" in result.output
        assert result.stdout == ""

    def test_zero_hits_emit_a_clean_sarif_run_not_a_phantom_finding(self, ctx_obj, monkeypatch):
        """``-o sarif`` on a zero-hit search emits a valid, empty SARIF run.

        A machine format has to stay parseable when the answer is empty: a
        zero-result SARIF document is how a scanner says "ran, found
        nothing", and emitting nothing at all leaves a consumer unable to
        tell a clean scan from one that never ran. The run still carries no
        finding and no phantom ``web_search_api`` rule — only the envelope.
        """
        ctx_obj = replace(ctx_obj, formatter=OutputFormatter(OutputFormat.SARIF))
        result = self._search(ctx_obj, monkeypatch, [])
        document = json.loads(result.stdout)
        assert document["runs"][0]["results"] == []
        assert "web_search_api" not in result.stdout


class TestWhatOneSearchMaySpend:
    """``--limit`` was unbounded, and every page past the first is metered.

    ``IntRange(min=1)`` bounded the bottom alone, so ``--limit 1000000``
    walked a corpus in 100 metered requests and one more mistyped digit
    would have taken 10000 — with no confirmation, no cap and nothing said
    about the cost. The ceiling is refused by click at the point of the
    typo, which is the only place the analyst can still see the number
    they meant.
    """

    def _search(self, ctx_obj, monkeypatch, limit):
        monkeypatch.setattr(
            TitaniumCloudService, "search_samples", lambda self, q, limit: [{"sha1": "a" * 40}]
        )
        return invoke(ticloud, ["search", "threat_name:Evil", "--limit", str(limit)], ctx_obj)

    def test_a_limit_past_the_ceiling_is_a_usage_error(self, ctx_obj, monkeypatch):
        result = self._search(ctx_obj, monkeypatch, 100_000_000)

        assert result.exit_code == 2
        assert str(MAX_LIMIT) in flat(result)

    def test_the_ceiling_itself_is_a_limit_the_analyst_may_type(self, ctx_obj, monkeypatch):
        assert self._search(ctx_obj, monkeypatch, MAX_LIMIT).exit_code == 0

    def test_the_ceiling_is_within_reach_of_the_walk_that_serves_it(self):
        """A ceiling the service cannot walk to is an answer capped twice.

        The analyst would be allowed to ask for records the walk stops
        short of, and told to raise a ``--limit`` that is already as high
        as it goes.
        """
        assert MAX_LIMIT <= _MAX_SEARCH_PAGES * _MAX_RECORDS_PER_PAGE

    def test_the_pivot_ceiling_is_within_reach_of_the_walk_that_serves_it(self):
        """The same claim for ``--max-results``, which the pivot walks take.

        The search budget had this test and the pivot budget only said it
        in a comment: 100 pages of 1000 records is the same ceiling, and a
        page size left to an SDK default is one release away from making it
        false, which is why the walk sends the size it counts on.
        """
        assert MAX_LIMIT <= _MAX_PIVOT_PAGES * _RECORDS_PER_PAGE

    def test_a_partial_answer_at_the_ceiling_names_a_remedy_that_exists(self, ctx_obj, monkeypatch):
        """ "Raise --limit above 100000" is advice the CLI itself refuses."""

        def left_hits_behind(service, *_args, **_kwargs):
            service.pages_left_unfetched = True
            return [{"sha1": "a" * 40}]

        monkeypatch.setattr(TitaniumCloudService, "search_samples", left_hits_behind)

        result = invoke(ticloud, ["search", "threat_name:Evil", "--limit", str(MAX_LIMIT)], ctx_obj)

        assert result.exit_code == 0
        assert "More results may be waiting" in result.output
        assert f"raise --limit above {MAX_LIMIT}" not in result.output


class TestAnalyzeUrlIsCheckedInOnePlace:
    """The CLI carried the service's guard verbatim, and returned before the call.

    So the service's own check — the one standing in front of the request
    — could never run on this path, and the two would have to be kept
    saying the same thing by hand. The layer that would send the URL is
    the layer that refuses it.
    """

    def _api(self, monkeypatch) -> MagicMock:
        api = MagicMock()
        monkeypatch.setattr(
            "rl_cli.services.titanium_cloud.network.ticloud.URLThreatIntelligence", lambda **kw: api
        )
        return api

    def test_a_url_that_is_not_one_is_named_and_refused(self, ctx_obj, monkeypatch):
        api = self._api(monkeypatch)

        result = invoke(ticloud, ["analyze-url", "not a url"], ctx_obj)

        assert "Invalid URL format: not a url" in flat(result)
        api.get_url_report.assert_not_called()
        # ``cli.result_callback`` turns this into the non-zero exit status;
        # invoking the group alone bypasses it.
        assert ctx_obj.output.status.failed
        assert result.stdout == ""

    def test_a_url_that_is_one_reaches_the_endpoint(self, ctx_obj, monkeypatch):
        api = self._api(monkeypatch)
        api.get_url_report.return_value = stub_response({"classification": "malicious"})

        result = invoke(ticloud, ["analyze-url", "https://example.com/x"], ctx_obj)

        assert result.exit_code == 0, result.output
        api.get_url_report.assert_called_once_with("https://example.com/x", private=True)
        assert json.loads(result.stdout) == {"classification": "malicious"}


class TestNetworkPivotCommands:
    """The nine "what else is associated with this" commands and their --all."""

    PIVOTS: ClassVar = [
        ("ip-files", "1.2.3.4", "get_files_from_ip"),
        ("ip-urls", "1.2.3.4", "get_urls_from_ip"),
        ("ip-domains", "1.2.3.4", "get_domains_from_ip"),
        ("domain-files", "evil.com", "get_files_from_domain"),
        ("domain-urls", "evil.com", "get_urls_from_domain"),
        ("domain-ips", "evil.com", "get_ips_from_domain"),
        ("domain-related", "evil.com", "get_related_domains"),
        ("url-files", "https://evil.com/x", "get_files_from_url"),
        ("uri-index", "evil.com", "get_uri_index"),
    ]

    @pytest.mark.parametrize("command,subject,method", PIVOTS)
    def test_the_first_page_is_reported_and_rendered(
        self, ctx_obj, monkeypatch, command, subject, method
    ):
        found = [{"sha1": "a" * 40}, {"sha1": "b" * 40}]
        monkeypatch.setattr(TitaniumCloudNetworkService, method, lambda self, subject: found)

        result = invoke(ticloud, [command, subject], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "Found 2 " in flat(result)
        assert json.loads(result.stdout) == found

    def _aggregated(self, monkeypatch, method) -> list[tuple[str, int | None]]:
        """Record what the paging variant was asked for, and refuse the other."""
        asked: list[tuple[str, int | None]] = []

        def aggregated(self, subject, max_results=None):
            asked.append((subject, max_results))
            return [{"sha1": "c" * 40}]

        monkeypatch.setattr(TitaniumCloudNetworkService, f"{method}_aggregated", aggregated)
        monkeypatch.setattr(
            TitaniumCloudNetworkService,
            method,
            lambda self, subject: pytest.fail("paged variant used"),
        )
        return asked

    @pytest.mark.parametrize("command,subject,method", PIVOTS)
    def test_all_asks_for_every_page_and_the_first_page_call_is_not_made(
        self, ctx_obj, monkeypatch, command, subject, method
    ):
        asked = self._aggregated(monkeypatch, method)

        result = invoke(ticloud, [command, subject, "--all"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert asked == [(subject, None)]

    @pytest.mark.parametrize("command,subject,method", PIVOTS)
    def test_a_budget_bounds_the_paging_and_reaches_the_service(
        self, ctx_obj, monkeypatch, command, subject, method
    ):
        """``--all`` alone is the whole corpus at 1000 records a metered page."""
        asked = self._aggregated(monkeypatch, method)

        result = invoke(ticloud, [command, subject, "--all", "--max-results", "250"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert asked == [(subject, 250)]

    @pytest.mark.parametrize("command,subject,method", PIVOTS)
    def test_a_budget_pages_without_all_being_typed_as_well(
        self, ctx_obj, monkeypatch, command, subject, method
    ):
        """Asking for 250 results is asking for as many pages as that takes;
        the first page alone could not honour it."""
        asked = self._aggregated(monkeypatch, method)

        result = invoke(ticloud, [command, subject, "--max-results", "250"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert asked == [(subject, 250)]

    @pytest.mark.parametrize("budget", ["0", "-5"])
    def test_a_budget_of_nothing_is_refused_before_anything_is_fetched(
        self, ctx_obj, monkeypatch, budget
    ):
        """The SDK pagers read a 0 budget as "every page there is"."""
        monkeypatch.setattr(
            TitaniumCloudNetworkService,
            "get_files_from_ip_aggregated",
            lambda self, subject, max_results=None: pytest.fail("a 0 budget reached the endpoint"),
        )

        result = invoke(ticloud, ["ip-files", "1.2.3.4", "--max-results", budget], ctx_obj)

        assert result.exit_code == 2
        assert "max-results" in flat(result)

    @pytest.mark.parametrize("command,subject,method", PIVOTS)
    def test_nothing_found_is_an_answer_not_a_failure(
        self, ctx_obj, monkeypatch, command, subject, method
    ):
        monkeypatch.setattr(TitaniumCloudNetworkService, method, lambda self, subject: [])

        result = invoke(ticloud, [command, subject], ctx_obj)

        assert result.exit_code == 0
        assert "No " in flat(result) and subject in flat(result)
        assert not ctx_obj.output.status.failed
        # The human "No ... found" note rides the status stream; stdout stays
        # a parseable empty document so a ``-o json`` pipeline reads ``[]``
        # rather than the empty input it would choke on.
        assert json.loads(result.stdout) == []

    @pytest.mark.parametrize("command,subject,method", PIVOTS)
    def test_a_failed_lookup_is_reported_as_one(
        self, ctx_obj, monkeypatch, command, subject, method
    ):
        monkeypatch.setattr(TitaniumCloudNetworkService, method, lambda self, subject: None)

        result = invoke(ticloud, [command, subject], ctx_obj)

        assert "Failed to look up" in flat(result)
        assert ctx_obj.output.status.failed

    @pytest.mark.parametrize(
        "command,rejected,api_class,sdk_method",
        [
            ("ip-files", "not-an-ip", "IPThreatIntelligence", "get_downloaded_files"),
            ("ip-urls", "999.1.1.1", "IPThreatIntelligence", "urls_from_ip"),
            ("ip-domains", "not-an-ip", "IPThreatIntelligence", "ip_to_domain_resolutions"),
            ("domain-files", "not a domain", "DomainThreatIntelligence", "get_downloaded_files"),
            ("domain-urls", "not a domain", "DomainThreatIntelligence", "urls_from_domain"),
            ("domain-ips", "not a domain", "DomainThreatIntelligence", "domain_to_ip_resolutions"),
            ("domain-related", "not a domain", "DomainThreatIntelligence", "related_domains"),
            ("url-files", "evil.com", "URLThreatIntelligence", "get_downloaded_files"),
            ("uri-index", "not a uri", "URIIndex", "get_uri_index"),
        ],
    )
    @pytest.mark.parametrize("paging", [[], ["--all"]], ids=["first_page", "all"])
    def test_input_the_endpoint_would_refuse_never_reaches_it(
        self, ctx_obj, monkeypatch, command, rejected, api_class, sdk_method, paging
    ):
        """Both halves of a pivot now ask the same endpoint, so both guard it."""
        api = MagicMock()
        monkeypatch.setattr(
            f"rl_cli.services.titanium_cloud.network.ticloud.{api_class}", lambda **kwargs: api
        )

        result = invoke(ticloud, [command, rejected, *paging], ctx_obj)

        assert rejected in flat(result)
        assert not getattr(api, sdk_method).called
        assert ctx_obj.output.status.failed
        assert result.stdout == ""


class TestAPivotSaysWhenTheCapCutTheWalkShort:
    """A capped walk reported "Found 5000 files for 8.8.8.8" and stopped there.

    All nine pivots take the same ``--max-results`` as ``yara-repo-list``
    and clamp it through the same helper, and only that one said the
    corpus might be larger — so an analyst read a capped pivot as the
    whole answer.
    """

    def _stub(self, monkeypatch, method, found):
        monkeypatch.setattr(
            TitaniumCloudNetworkService,
            f"{method}_aggregated",
            lambda self, subject, max_results=None: found[:max_results],
        )

    @pytest.mark.parametrize("command,subject,method", TestNetworkPivotCommands.PIVOTS)
    def test_a_result_the_size_of_the_cap_is_announced_as_partial(
        self, ctx_obj, monkeypatch, command, subject, method
    ):
        self._stub(monkeypatch, method, [{"sha1": h * 40} for h in "abc"])

        result = invoke(ticloud, [command, subject, "--max-results", "2"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert _NOTICE in flat(result)

    @pytest.mark.parametrize("command,subject,method", TestNetworkPivotCommands.PIVOTS)
    def test_a_result_short_of_the_cap_is_the_whole_set(
        self, ctx_obj, monkeypatch, command, subject, method
    ):
        self._stub(monkeypatch, method, [{"sha1": "a" * 40}])

        result = invoke(ticloud, [command, subject, "--max-results", "2"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "More results" not in flat(result)

    @pytest.mark.parametrize("command,subject,method", TestNetworkPivotCommands.PIVOTS)
    def test_an_uncapped_lookup_says_nothing_about_a_cap(
        self, ctx_obj, monkeypatch, command, subject, method
    ):
        """``--all`` with no budget fetched every page there was, so nothing was left."""
        monkeypatch.setattr(
            TitaniumCloudNetworkService,
            f"{method}_aggregated",
            lambda self, subject, max_results=None: [{"sha1": "a" * 40}],
        )

        result = invoke(ticloud, [command, subject, "--all"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "More results" not in flat(result)

    @pytest.mark.parametrize("command,subject,method", TestNetworkPivotCommands.PIVOTS)
    def test_a_walk_that_overshot_its_cap_is_partial_too(
        self, ctx_obj, monkeypatch, command, subject, method
    ):
        """The walk trims to the cap; a service that hands back more is no
        less partial, and the notice is read off what arrived."""
        monkeypatch.setattr(
            TitaniumCloudNetworkService,
            f"{method}_aggregated",
            lambda self, subject, max_results=None: [{"sha1": h * 40} for h in "abc"],
        )

        result = invoke(ticloud, [command, subject, "--max-results", "2"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert _NOTICE in flat(result)

    def test_a_failed_capped_lookup_claims_nothing_about_the_corpus(self, ctx_obj, monkeypatch):
        """``None`` is a call that did not land, not a walk that filled its cap."""
        monkeypatch.setattr(
            TitaniumCloudNetworkService,
            "get_files_from_ip_aggregated",
            lambda self, subject, max_results=None: None,
        )

        result = invoke(ticloud, ["ip-files", "1.2.3.4", "--max-results", "2"], ctx_obj)

        assert "Failed to look up" in flat(result)
        assert "More results" not in flat(result)


class TestAPivotSaysWhenTheEndpointHeldAnotherPage:
    """The flagless invocation printed "Found 1000 files" over a 40000 corpus.

    The notice fired only for a walk that filled ``--max-results``, so the
    default — one page, and the cursor for the next one thrown away —
    reported a page as the whole answer and exited 0.
    """

    def _first_page(self, monkeypatch, method, *, more_pages):
        """Answer one page, as the service does when it read the cursor."""

        def page(service, subject):
            service.pages_left_unfetched = more_pages
            return [{"sha1": "a" * 40}]

        monkeypatch.setattr(TitaniumCloudNetworkService, method, page)

    @pytest.mark.parametrize("command,subject,method", TestNetworkPivotCommands.PIVOTS)
    def test_a_page_left_unfetched_is_announced_with_no_flags_typed(
        self, ctx_obj, monkeypatch, command, subject, method
    ):
        self._first_page(monkeypatch, method, more_pages=True)

        result = invoke(ticloud, [command, subject], ctx_obj)

        assert result.exit_code == 0, result.output
        assert _NOTICE in flat(result)

    @pytest.mark.parametrize("command,subject,method", TestNetworkPivotCommands.PIVOTS)
    def test_the_only_page_there_was_is_reported_as_the_whole_answer(
        self, ctx_obj, monkeypatch, command, subject, method
    ):
        self._first_page(monkeypatch, method, more_pages=False)

        result = invoke(ticloud, [command, subject], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "More results" not in flat(result)

    def _ip_files(self, ctx_obj, monkeypatch, envelope):
        """``ticloud ip-files 8.8.8.8`` against an endpoint answering ``envelope``."""
        api = MagicMock()
        api.get_downloaded_files.return_value = stub_response({"rl": envelope})
        monkeypatch.setattr(
            "rl_cli.services.titanium_cloud.network.ticloud.IPThreatIntelligence",
            lambda **kwargs: api,
        )
        return invoke(ticloud, ["ip-files", "8.8.8.8"], ctx_obj)

    def test_a_cursor_in_the_answer_reaches_the_notice(self, ctx_obj, monkeypatch):
        """The whole path: the envelope states a cursor, the analyst is told."""
        result = self._ip_files(
            ctx_obj, monkeypatch, {"downloaded_files": [{"sha1": "a" * 40}], "next_page": "page2"}
        )

        assert result.exit_code == 0, result.output
        assert "Found 1 files for 8.8.8.8" in flat(result)
        assert _NOTICE in flat(result)

    def test_an_answer_with_no_cursor_says_nothing(self, ctx_obj, monkeypatch):
        result = self._ip_files(ctx_obj, monkeypatch, {"downloaded_files": [{"sha1": "a" * 40}]})

        assert result.exit_code == 0, result.output
        assert "More results" not in flat(result)


class TestEachCommandBuildsTheHalfItUses:
    """A network command is handed ``TitaniumCloudNetworkService``, and only that.

    Every ``ticloud`` command used to be handed ``TitaniumCloudService``,
    which built a network service of its own and forwarded twenty-one
    methods to it — so a pivot was answered by two API subclasses holding
    the same credentials and the same proxy URL, having called only one of
    them.

    Patching the network service's methods did not show this on its own:
    the facade forwarded to exactly the same patched method, which is how
    the split could have been shipped with nothing pointed at it. Refusing
    to let the other service be constructed does.
    """

    def test_a_pivot_never_builds_the_file_oriented_service(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(
            TitaniumCloudService,
            "__init__",
            lambda self, *args, **kwargs: pytest.fail("a pivot built the file-side service"),
        )
        monkeypatch.setattr(
            TitaniumCloudNetworkService, "get_files_from_ip", lambda self, ip: [{"sha1": "a" * 40}]
        )

        result = invoke(ticloud, ["ip-files", "1.2.3.4"], ctx_obj)

        assert result.exit_code == 0, result.output

    def test_a_file_lookup_never_builds_a_network_service(self, ctx_obj, monkeypatch):
        """The other direction: ``reputation`` asks for the file half by name.

        It used to be able to build only that and still reach the network
        endpoints, because the file half held a network service to forward
        with. Nothing holds one now, so this counts what was built.
        """
        built: list[object] = []

        def reputation(self: TitaniumCloudService, h: str) -> dict[str, str]:
            built.append(self)
            return {"threat_status": "known"}

        monkeypatch.setattr(TitaniumCloudService, "get_file_reputation", reputation)
        monkeypatch.setattr(
            TitaniumCloudNetworkService,
            "__init__",
            lambda self, *args, **kwargs: pytest.fail("a file lookup built the network service"),
        )

        result = invoke(ticloud, ["reputation", SHA256], ctx_obj)

        assert result.exit_code == 0, result.output
        assert [type(service) for service in built] == [TitaniumCloudService]


class TestDomainAndIpReportCommands:
    @pytest.mark.parametrize(
        "command,subject,method",
        [
            ("domain-report", "evil.com", "get_domain_report"),
            ("ip-report", "8.8.8.8", "get_ip_report"),
        ],
    )
    def test_the_report_is_rendered(self, ctx_obj, monkeypatch, command, subject, method):
        report = {"classification": "malicious"}
        monkeypatch.setattr(TitaniumCloudNetworkService, method, lambda self, subject: report)

        result = invoke(ticloud, [command, subject], ctx_obj)

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == report

    @pytest.mark.parametrize(
        "command,subject,method",
        [
            ("domain-report", "evil.com", "get_domain_report"),
            ("ip-report", "8.8.8.8", "get_ip_report"),
        ],
    )
    def test_a_failure_is_reported_as_one(self, ctx_obj, monkeypatch, command, subject, method):
        monkeypatch.setattr(TitaniumCloudNetworkService, method, lambda self, subject: None)

        result = invoke(ticloud, [command, subject], ctx_obj)

        assert "Failed to get" in flat(result)
        assert ctx_obj.output.status.failed

    @pytest.mark.parametrize(
        "command,rejected,api_class,sdk_method",
        [
            ("domain-report", "not a domain", "DomainThreatIntelligence", "get_domain_report"),
            ("ip-report", "999.1.1.1", "IPThreatIntelligence", "get_ip_report"),
        ],
    )
    def test_input_the_endpoint_would_refuse_never_reaches_it(
        self, ctx_obj, monkeypatch, command, rejected, api_class, sdk_method
    ):
        api = MagicMock()
        monkeypatch.setattr(
            f"rl_cli.services.titanium_cloud.network.ticloud.{api_class}", lambda **kwargs: api
        )

        result = invoke(ticloud, [command, rejected], ctx_obj)

        assert rejected in flat(result)
        assert not getattr(api, sdk_method).called
        assert ctx_obj.output.status.failed


class TestReputationTakesABatch:
    """One hash goes to the single endpoint; several go in one bulk request."""

    def test_one_hash_uses_the_single_endpoint(self, ctx_obj, monkeypatch):
        asked: list[str] = []

        def reputation(self: TitaniumCloudService, h: str) -> dict[str, str]:
            asked.append(h)
            return {"threat_status": "known"}

        monkeypatch.setattr(TitaniumCloudService, "get_file_reputation", reputation)
        monkeypatch.setattr(
            TitaniumCloudService,
            "get_bulk_file_reputation",
            lambda self, hashes: pytest.fail("one hash went to the bulk endpoint"),
        )

        result = invoke(ticloud, ["reputation", SHA256], ctx_obj)

        assert result.exit_code == 0, result.output
        assert asked == [SHA256]

    def test_several_hashes_are_one_bulk_request(self, ctx_obj, monkeypatch):
        """Triaging a batch used to cost one metered round-trip per hash."""
        batches: list[list[str]] = []
        entries = [{"sha1": "a" * 40}, {"sha1": "b" * 40}, {"sha1": "c" * 40}]

        def bulk(self: TitaniumCloudService, hashes: list[str]) -> list[dict[str, str]]:
            batches.append(hashes)
            return entries

        monkeypatch.setattr(TitaniumCloudService, "get_bulk_file_reputation", bulk)
        monkeypatch.setattr(
            TitaniumCloudService,
            "get_file_reputation",
            lambda self, h: pytest.fail("a batch went to the single endpoint"),
        )

        result = invoke(ticloud, ["reputation", "a" * 40, "-h", "b" * 40, "-h", "c" * 40], ctx_obj)

        assert result.exit_code == 0, result.output
        assert batches == [["a" * 40, "b" * 40, "c" * 40]]
        assert json.loads(result.stdout) == entries
        assert "3 of 3 hashes" in flat(result)

    def test_a_hash_file_is_read_a_hash_per_line(self, ctx_obj, monkeypatch, tmp_path):
        batches: list[list[str]] = []

        def bulk(self: TitaniumCloudService, hashes: list[str]) -> list[dict[str, str]]:
            batches.append(hashes)
            return [{"sha1": "a" * 40}]

        monkeypatch.setattr(TitaniumCloudService, "get_bulk_file_reputation", bulk)
        hash_file = tmp_path / "hashes.txt"
        hash_file.write_text(f"{'b' * 40}\n\n  {'c' * 40}  \n")

        result = invoke(ticloud, ["reputation", "a" * 40, "-f", str(hash_file)], ctx_obj)

        assert result.exit_code == 0, result.output
        assert batches == [["a" * 40, "b" * 40, "c" * 40]]

    def test_a_binary_hash_file_is_named_not_an_unexpected_error(self, ctx_obj, tmp_path):
        """``-f`` aimed at a sample, not its hash list, must say which mistake it is.

        Pointing ``--hash-file`` at a binary made ``read_text`` raise a
        ``UnicodeDecodeError`` that escaped to the top-level handler as
        "Unexpected error", telling the analyst to file a bug about their own
        slip. It is a usage error naming the file instead.
        """
        binary = tmp_path / "sample.bin"
        binary.write_bytes(b"MZ\x90\x00\x03\xff\xfe")

        result = invoke(ticloud, ["reputation", "-f", str(binary)], ctx_obj)

        assert result.exit_code == 2
        assert "not a UTF-8 text file" in result.output
        assert "Unexpected error" not in result.output

    def test_a_batch_the_endpoint_would_refuse_never_reaches_it(self, ctx_obj, monkeypatch):
        api = MagicMock()
        monkeypatch.setattr(
            "rl_cli.services.titanium_cloud.service.ticloud.FileReputation", lambda **kwargs: api
        )

        result = invoke(ticloud, ["reputation", "a" * 40, "-h", "nothash"], ctx_obj)

        assert "entry 2 of 2" in flat(result)
        assert not api.get_file_reputation.called
        assert ctx_obj.output.status.failed
        assert result.stdout == ""

    def test_naming_no_hash_at_all_is_a_usage_error(self, ctx_obj):
        result = invoke(ticloud, ["reputation"], ctx_obj)

        assert result.exit_code == 2
        assert "Usage" in result.output


# One malicious record, and a second sample to make a batch of it.
MALICIOUS = {
    "sha1": "a" * 40,
    "status": "MALICIOUS",
    "threat_name": "Win32.Trojan.Emotet",
    "scanner_count": 37,
    "scanner_match": 31,
}
SECOND = {"sha1": "b" * 40, "status": "KNOWN"}


class TestReputationLooksUpEachHashOnce:
    """``reputation <H> -h <H>`` sent the hash twice and misreported the answer.

    The count was of what was typed rather than of what was asked about,
    so the answer to two identical hashes read "retrieved for 1 of 2
    hashes" — which is how the CLI says one hash was not found.
    """

    def _single(self, monkeypatch) -> list[str]:
        asked: list[str] = []

        def get_file_reputation(self, hash_value):
            asked.append(hash_value)
            return {"rl": {"malware_presence": MALICIOUS}}

        monkeypatch.setattr(TitaniumCloudService, "get_file_reputation", get_file_reputation)
        monkeypatch.setattr(
            TitaniumCloudService,
            "get_bulk_file_reputation",
            lambda self, batch: pytest.fail("one distinct hash went to the bulk endpoint"),
        )
        return asked

    def test_the_same_hash_named_twice_is_one_lookup(self, ctx_obj, monkeypatch):
        asked = self._single(monkeypatch)

        result = invoke(ticloud, ["reputation", SHA256, "-h", SHA256], ctx_obj)

        assert result.exit_code == 0, result.output
        assert asked == [SHA256]
        assert "1 of 2" not in flat(result)

    def test_two_spellings_of_one_hash_are_one_lookup(self, ctx_obj, monkeypatch):
        """The endpoints are asked in lower case, so this is one hash."""
        asked = self._single(monkeypatch)

        result = invoke(ticloud, ["reputation", SHA256, "-h", f"  {SHA256.upper()}  "], ctx_obj)

        assert result.exit_code == 0, result.output
        assert asked == [SHA256]

    def test_a_batch_is_counted_by_the_hashes_it_asks_about(self, ctx_obj, monkeypatch, tmp_path):
        seen: list[list[str]] = []

        def bulk(self: TitaniumCloudService, batch: list[str]) -> list[dict[str, Any]]:
            seen.append(batch)
            return [MALICIOUS, SECOND]

        monkeypatch.setattr(TitaniumCloudService, "get_bulk_file_reputation", bulk)
        listed = tmp_path / "hashes.txt"
        listed.write_text(f"{'a' * 40}\n{'b' * 40}\n{'a' * 40}\n")

        result = invoke(ticloud, ["reputation", "-f", str(listed)], ctx_obj)

        assert result.exit_code == 0, result.output
        assert seen == [["a" * 40, "b" * 40]]
        assert "2 of 2 hashes" in flat(result)

    def test_the_dropped_duplicates_are_said_out_loud(self, ctx_obj, monkeypatch):
        self._single(monkeypatch)

        result = invoke(ticloud, ["reputation", SHA256, "-h", SHA256], ctx_obj)

        assert "1 duplicate hash(es) dropped" in flat(result)


class TestReputationGradesTheSampleWhateverTheBatchSize:
    """One hash was graded; several were dumped as a table of raw keys.

    The single query answers ``rl.malware_presence`` and the bulk query
    answers ``rl.entries``, and the CLI rendered the first through the
    verdict path and the second through the generic formatter — so the
    same question was answered at two fidelities depending on how many
    hashes were named, in ``-o rich`` and in ``-o sarif`` alike.
    """

    def _endpoints(self, monkeypatch) -> None:
        monkeypatch.setattr(
            TitaniumCloudService,
            "get_file_reputation",
            lambda self, h: {"rl": {"malware_presence": MALICIOUS}},
        )
        monkeypatch.setattr(
            TitaniumCloudService,
            "get_bulk_file_reputation",
            lambda self, batch: [MALICIOUS, SECOND],
        )

    def test_rich_renders_a_report_per_sample_not_a_table_of_keys(self, ctx_obj, monkeypatch):
        self._endpoints(monkeypatch)
        rich = replace(ctx_obj, formatter=OutputFormatter(OutputFormat.RICH))

        alone = flat(invoke(ticloud, ["reputation", "a" * 40], rich))
        batched = flat(invoke(ticloud, ["reputation", "a" * 40, "-h", "b" * 40], rich))

        assert alone.count("File Analysis Result") == 1
        assert batched.count("File Analysis Result") == 2
        assert "Win32.Trojan.Emotet" in alone and "Win32.Trojan.Emotet" in batched

    def test_sarif_grades_the_same_sample_identically_in_both_paths(self, ctx_obj, monkeypatch):
        self._endpoints(monkeypatch)
        sarif = replace(ctx_obj, formatter=OutputFormatter(OutputFormat.SARIF))

        alone = json.loads(invoke(ticloud, ["reputation", "a" * 40], sarif).stdout)
        batched = json.loads(
            invoke(ticloud, ["reputation", "a" * 40, "-h", "b" * 40], sarif).stdout
        )

        graded = alone["runs"][0]["results"][0]
        assert graded["level"] == "error"
        # The family name is a property of the result; the rule id is the
        # severity's. See tests/test_sarif.py.
        assert graded["properties"]["threatName"] == "Win32.Trojan.Emotet"
        assert graded == batched["runs"][0]["results"][0]

    def test_json_answers_records_whether_one_hash_was_named_or_two(self, ctx_obj, monkeypatch):
        """The envelope used to go out whole for one hash and never for two."""
        self._endpoints(monkeypatch)

        alone = json.loads(invoke(ticloud, ["reputation", "a" * 40], ctx_obj).stdout)
        batched = json.loads(
            invoke(ticloud, ["reputation", "a" * 40, "-h", "b" * 40], ctx_obj).stdout
        )

        assert alone == [MALICIOUS]
        assert batched == [MALICIOUS, SECOND]


class TestDownloadWritesLiveMalware:
    """Same order and the same write as the A1000 download: check, warn, write."""

    def _service(self, monkeypatch, written):
        def download_sample(self, hash_value, output_path):
            written.append(output_path)
            write_private_bytes(output_path, b"MZ malware")
            return True

        monkeypatch.setattr(TitaniumCloudService, "download_sample", download_sample)

    def test_the_sample_is_written_where_it_was_asked_for(self, ctx_obj, monkeypatch, tmp_path):
        written: list = []
        self._service(monkeypatch, written)
        target = tmp_path / "samples"

        result = invoke(ticloud, ["download", SHA256, "--output-dir", str(target)], ctx_obj)

        assert result.exit_code == 0, result.output
        assert written == [target / f"{SHA256}.malware"]
        assert (target / f"{SHA256}.malware").read_bytes() == b"MZ malware"

    def test_the_warning_comes_before_the_write(self, ctx_obj, monkeypatch, tmp_path):
        """--output-dir defaults to cwd, so a warning after the fact tells the
        analyst where the malware already is."""
        order: list[str] = []

        def download(self: TitaniumCloudService, hash_value: str, output_path: Path) -> bool:
            order.append("write")
            return True

        monkeypatch.setattr(TitaniumCloudService, "download_sample", download)
        monkeypatch.setattr(
            ctx_obj.output.__class__,
            "warning",
            lambda self, message: order.append(f"warn:{message}"),
        )

        invoke(ticloud, ["download", SHA256, "--output-dir", str(tmp_path / "out")], ctx_obj)

        assert order[0].startswith("warn:")
        assert "live malware" in order[0]
        assert order[-1] == "write"

    def test_a_hash_the_endpoints_refuse_creates_no_directory_and_warns_of_nothing(
        self, ctx_obj, monkeypatch, tmp_path
    ):
        api = MagicMock()
        monkeypatch.setattr(
            "rl_cli.services.titanium_cloud.service.ticloud.FileDownload", lambda **kwargs: api
        )
        target = tmp_path / "samples"

        result = invoke(ticloud, ["download", "b" * 128, "--output-dir", str(target)], ctx_obj)

        assert not target.exists(), "an invalid hash left an empty output directory behind"
        assert "live malware" not in flat(result)
        assert not api.download_sample.called
        assert ctx_obj.output.status.failed

    def test_an_existing_file_is_announced_before_it_is_replaced(
        self, ctx_obj, monkeypatch, tmp_path
    ):
        self._service(monkeypatch, [])
        target = tmp_path / "samples"
        target.mkdir()
        (target / f"{SHA256}.malware").write_bytes(b"older copy")

        result = invoke(ticloud, ["download", SHA256, "--output-dir", str(target)], ctx_obj)

        assert "already exists and will be replaced" in flat(result)
        assert result.exit_code == 0, result.output

    def test_a_failed_download_is_reported_as_one(self, ctx_obj, monkeypatch, tmp_path):
        monkeypatch.setattr(
            TitaniumCloudService,
            "download_sample",
            lambda self, hash_value, output_path: False,
        )

        result = invoke(
            ticloud, ["download", SHA256, "--output-dir", str(tmp_path / "out")], ctx_obj
        )

        assert "Failed to download sample" in flat(result)
        assert ctx_obj.output.status.failed


class TestDownloadStatus:
    def test_the_status_is_rendered(self, ctx_obj, monkeypatch):
        status = {"rl": {"status": [{"status": "SAMPLE_AVAILABLE"}]}}
        monkeypatch.setattr(TitaniumCloudService, "get_download_status", lambda self, h: status)

        result = invoke(ticloud, ["download-status", SHA256], ctx_obj)

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == status

    def test_a_hash_the_endpoints_refuse_never_reaches_them(self, ctx_obj, monkeypatch):
        api = MagicMock()
        monkeypatch.setattr(
            "rl_cli.services.titanium_cloud.service.ticloud.FileDownload", lambda **kwargs: api
        )

        result = invoke(ticloud, ["download-status", "b" * 128], ctx_obj)

        assert "MD5" in flat(result) and "SHA256" in flat(result)
        assert not api.get_download_status.called
        assert ctx_obj.output.status.failed

    def test_a_failure_is_reported_as_one(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(TitaniumCloudService, "get_download_status", lambda self, h: None)

        result = invoke(ticloud, ["download-status", SHA256], ctx_obj)

        assert "Failed to check download status" in flat(result)
        assert ctx_obj.output.status.failed


class TestBulkReputationIsInvokableAsDocumented:
    """`-f hashes.txt` must work without also naming a hash on the line.

    The positional argument was `required=True`, so the README's own bulk
    example exited 2 with "Missing argument 'HASH...'" — the feature the
    change was named for could not be invoked the way it was documented.
    """

    def test_a_hash_file_alone_is_enough(self, tmp_path, ctx_obj, monkeypatch):
        listed = tmp_path / "hashes.txt"
        listed.write_text(f"{SHA256}\n{SHA1}\n")
        seen: list[list[str]] = []

        def bulk(self: TitaniumCloudService, batch: list[str]) -> list[dict[str, str]]:
            seen.append(batch)
            return [{"sha1": SHA1}]

        monkeypatch.setattr(TitaniumCloudService, "get_bulk_file_reputation", bulk)

        result = invoke(ticloud, ["reputation", "-f", str(listed)], ctx_obj)

        assert result.exit_code == 0, result.output
        assert seen == [[SHA256, SHA1]]

    def test_repeated_hash_options_alone_are_enough(self, ctx_obj, monkeypatch):
        seen: list[list[str]] = []

        def bulk(self: TitaniumCloudService, batch: list[str]) -> list[dict[str, str]]:
            seen.append(batch)
            return [{"sha1": SHA1}]

        monkeypatch.setattr(TitaniumCloudService, "get_bulk_file_reputation", bulk)

        result = invoke(ticloud, ["reputation", "-h", SHA256, "-h", SHA1], ctx_obj)

        assert result.exit_code == 0, result.output
        assert seen == [[SHA256, SHA1]]

    def test_naming_nothing_at_all_is_still_a_usage_error(self, ctx_obj):
        assert invoke(ticloud, ["reputation"], ctx_obj).exit_code != 0


class TestAnEmptyTicloudAnswerIsAnAnswer:
    """A subject TitaniumCloud holds nothing on is answered 200 with nothing.

    Four commands read that as a failed lookup and exited 1, while
    ``reputation``, ``search`` and the two network reports in the same file
    warned and exited 0 about the same fact — so an enrichment run over
    clean hashes and URLs was filed as a run of errors. Driven through the
    root group, since that is where the exit status a script reads is set.
    """

    @pytest.mark.parametrize(
        "command, service_cls, method, notice",
        [
            (
                ["av-scanners", SHA256],
                TitaniumCloudService,
                "get_av_scanners",
                "holds no AV scanner results for",
            ),
            (
                ["analysis", SHA256],
                TitaniumCloudService,
                "get_file_analysis",
                "holds no analysis for",
            ),
            (
                ["download-status", SHA256],
                TitaniumCloudService,
                "get_download_status",
                "holds no download status for",
            ),
            (
                ["analyze-url", "https://example.com/x"],
                TitaniumCloudNetworkService,
                "analyze_url",
                "holds no analysis for",
            ),
        ],
        ids=["av-scanners", "analysis", "download-status", "analyze-url"],
    )
    def test_an_empty_answer_warns_and_exits_zero(
        self, monkeypatch, command, service_cls, method, notice
    ):
        monkeypatch.setattr(service_cls, method, lambda self, subject: {})

        result = CliRunner().invoke(cli, ["ticloud", *command])

        assert result.exit_code == 0, result.output
        assert notice in flat(result)
        assert "Failed" not in flat(result)

    @pytest.mark.parametrize(
        "command, service_cls, method, failure",
        [
            (
                ["av-scanners", SHA256],
                TitaniumCloudService,
                "get_av_scanners",
                "Failed to retrieve AV scanner results",
            ),
            (
                ["analyze-url", "https://example.com/x"],
                TitaniumCloudNetworkService,
                "analyze_url",
                "Failed to analyze URL",
            ),
        ],
        ids=["av-scanners", "analyze-url"],
    )
    def test_a_call_that_never_landed_still_exits_one(
        self, monkeypatch, command, service_cls, method, failure
    ):
        monkeypatch.setattr(service_cls, method, lambda self, subject: None)

        result = CliRunner().invoke(cli, ["ticloud", *command])

        assert result.exit_code == 1, result.output
        assert failure in flat(result)


class TestDomainEndpointsWantADomain:
    """A URL reaching a domain endpoint answers an empty page.

    That renders as "no files found for https://evil.com/path" — a wrong
    answer shaped like a real one, which is the failure this layer exists
    to prevent.
    """

    @pytest.mark.parametrize(
        "command", ["domain-report", "domain-files", "domain-urls", "domain-ips"]
    )
    def test_a_url_never_reaches_the_endpoint(self, command, ctx_obj, monkeypatch):
        # Patched on the base both halves build their handles through: the
        # domain lookups are the network service's, so refusing ``_api`` on
        # ``TitaniumCloudService`` alone would fail nothing it does.
        monkeypatch.setattr(
            TitaniumCloudApi,
            "_api",
            lambda self, cls: pytest.fail("a URL must not reach a domain endpoint"),
        )

        result = invoke(ticloud, [command, "https://evil.com/path"], ctx_obj)

        assert ctx_obj.output.status.failed
        assert "bare domain" in result.output

    def test_a_bare_domain_still_goes_through(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(
            TitaniumCloudNetworkService,
            "get_domain_report",
            lambda self, domain: {"domain": domain},
        )
        assert invoke(ticloud, ["domain-report", "evil.com"], ctx_obj).exit_code == 0
