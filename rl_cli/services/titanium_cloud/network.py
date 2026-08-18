"""Domain, IP, URL and URI-index intelligence: the "what else is associated with this" half."""

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from typing import Any

from ReversingLabs.SDK import ticloud

from rl_cli.config import Settings
from rl_cli.services.addresses import valid_domain, valid_ip, valid_url, valid_url_or_host
from rl_cli.services.decorators import safe_call
from rl_cli.services.http import json_on
from rl_cli.services.partial_answers import cleared_per_call
from rl_cli.services.protocols import Notifier
from rl_cli.services.titanium_cloud.api import TitaniumCloudApi
from rl_cli.services.walks import CursorPage, cursor_walk

# Where a first page states the cursor for the one after it. Eight of the
# nine pivots carry it directly under the ``rl`` envelope and TCA-0401
# carries its SHA-1 cursor inside ``uri_index``, which is where each SDK
# aggregator reads it from (SDK ticloud.py:1116, 1463, 1821 …).
_NEXT_PAGE = ("next_page",)
_NEXT_PAGE_SHA1 = ("uri_index", "next_page_sha1")

# How large a page these endpoints are asked for, and how many pages one
# ``--all`` walk of a pivot may fetch. 100 pages of 1000 is 100000 records,
# the ceiling ``_shared_inputs.MAX_LIMIT`` puts on the largest listing this
# CLI will fetch, for 100 metered requests. The tests pin that product, so
# the page size is sent rather than left to an SDK default that could
# change under it.
#
# TCA-0401 is the exception and takes no page size at all (SDK
# ticloud.py:1024), so there the budget bounds requests and not records: a
# walk that reaches it says so, and that warning is the whole of what can
# be said about how much of the corpus was read.
#
# It is the walk's own bound and not the caller's: ``--max-results`` counts
# records, so it stops a walk only once records arrive, and ``--all`` names
# no number at all.
_RECORDS_PER_PAGE = 1000
_MAX_PIVOT_PAGES = 100

# Which keyword carries the cursor back to the endpoint that stated it.
# Eight of the nine pivots take a page string; TCA-0401 takes the first
# SHA-1 of the page it is being asked for.
_PAGE_STRING = "page_string"
_PAGE_SHA1 = "page_sha1"

# How every lookup below judges its subject: the guard answers the value to
# send, or ``None`` once it has said why through the callable it is handed.
type _Guard = Callable[[str, Callable[[str], None]], str | None]


def _stated_cursor(body: Any, cursor: tuple[str, ...]) -> Any:
    """The cursor a parsed envelope names at ``rl.<cursor>``, or ``None``.

    A cursor that is absent, null or empty is the last page; anything else
    names a page this lookup did not fetch.

    Takes the body the entries were read from, not the response they came
    in: ``requests.Response.json`` caches nothing and re-runs
    ``json.loads`` over the whole page on every call.
    """
    node: Any = body
    for step in ("rl", *cursor):
        if not isinstance(node, dict):
            return None
        node = node.get(step)
    return node or None


def _record_budget(max_results: int | None) -> int | None:
    """``max_results`` as a walk may act on it; ``None`` means every page.

    Zero and negatives are clamped to one record rather than passed on:
    every SDK aggregator branches on ``if not max_results`` and so reads
    either as "fetch the whole corpus" (SDK ticloud.py:1118, 1236, 1823 …),
    billed per page. ``--max-results`` is an ``IntRange(min=1)``, so no
    analyst can type one.
    """
    return None if max_results is None else max(1, max_results)


@cleared_per_call("pages_left_unfetched")
class TitaniumCloudNetworkService(TitaniumCloudApi):
    """The network-intelligence endpoints: TCA-0401 through TCA-0407.

    Four of the ten ``ticloud`` API families;
    :class:`~rl_cli.services.titanium_cloud.service.TitaniumCloudService`
    is its peer, holds the other six, and answers to no name defined here.

    Each pivot comes in two, and both ask the same endpoint: the plain
    wrapper answers its first page, and the ``_aggregated`` one walks it
    page by page through :func:`~rl_cli.services.walks.cursor_walk`, which
    is where the stops are set out. Neither hands the paging to the SDK's
    own ``*_aggregated`` methods, which cannot stop against an endpoint
    that names a further page forever.

    ``max_results`` bounds records rather than requests, so
    :data:`_MAX_PIVOT_PAGES` bounds the requests. The record budget is a
    parameter rather than a constant because only the caller knows how much
    of the corpus it meant to pay for.
    """

    def __init__(self, settings: Settings, output: Notifier | None = None):
        super().__init__(settings, output)
        # Whether the last lookup left pages behind, for a caller that can
        # offer to fetch them: a fact about the answer, not advice about
        # what to do with it. It speaks for the call that just finished and
        # for no earlier one, which is why the class decorator clears it at
        # every entry point and :meth:`_page` is the only thing that sets it.
        self.pages_left_unfetched = False

    # Each ticloud API class talks to a different endpoint family. Cache
    # them so repeated calls reuse the same handle.
    @cached_property
    def _url_threat(self) -> Any:
        return self._api(ticloud.URLThreatIntelligence)

    @cached_property
    def _domain_threat(self) -> Any:
        return self._api(ticloud.DomainThreatIntelligence)

    @cached_property
    def _ip_threat(self) -> Any:
        return self._api(ticloud.IPThreatIntelligence)

    @cached_property
    def _uri_index(self) -> Any:
        return self._api(ticloud.URIIndex)

    # The four guards these lookups need are shared with the A1000 half,
    # in :mod:`rl_cli.services.addresses`. Every domain and IP call below
    # POSTs its subject in a JSON body, so nothing here can be reshaped the
    # way the A1000's unquoted path segment can; what the shared guard buys
    # these endpoints is not announcing a non-answer about a key the corpus
    # never held.

    @property
    def _url_lookups_are_private(self) -> bool:
        """Whether these three endpoints keep the URL out of public feeds.

        The SDK's own default is ``private=False``: omit the parameter and
        every URL an analyst looks up — and the answer — is shared with
        third-party sources and enters public feeds. A CLI's queries are its
        user's investigation targets, so the flag is inverted into
        ``share_url_lookups`` and defaults to not sharing. Deliberately
        contributing to the feeds stays one setting away.
        """
        return not self.ti_cloud_settings.share_url_lookups

    # Every pivot below is one of three shapes — an envelope, a first page,
    # or a whole walk — so each one is its guard, its endpoint and nothing
    # else. The guard, the deferred handle and the paging verdict live in
    # the three helpers here rather than in 21 copies.

    def _endpoint(self, api: str, method: str) -> Callable[..., Any]:
        """The SDK call named by handle and method, built now and not sooner.

        Named rather than passed in because ``_domain_threat`` and its
        siblings are ``cached_property`` objects that build an SDK handle
        from the credentials, and an argument is evaluated before the call
        it is written in: a handle passed as one is built even for a subject
        the guard is about to refuse. Nothing in that construction reaches
        the network (SDK ticloud.py:60-83), so what naming it buys is that a
        refused subject's error stays the guard's own rather than the SDK's
        complaint about an unrelated setting.
        """
        return getattr(getattr(self, api), method)

    def _envelope(self, guard: _Guard, subject: str, api: str, method: str, **kwargs: Any) -> Any:
        """The whole body ``api.method`` answers about ``subject``."""
        accepted = guard(subject, self.output.error)
        if accepted is None:
            return None
        return json_on(self._endpoint(api, method)(accepted, **kwargs))

    def _page(
        self,
        guard: _Guard,
        subject: str,
        api: str,
        method: str,
        *path: str,
        cursor: tuple[str, ...] = _NEXT_PAGE,
        records_per_page: int | None = _RECORDS_PER_PAGE,
        **kwargs: Any,
    ) -> list[Any] | None:
        """The first page ``api.method`` answers about ``subject``, out of its envelope.

        A first page is not the whole answer and must not be reported as
        one: these endpoints hold up to 1000 records a page and state the
        cursor for the next one in the same envelope, so
        :attr:`pages_left_unfetched` records whether one was left behind —
        the measurement only, never the remedy. Naming ``--all`` would make
        this service unusable to a caller without that option, which is the
        dependency on the presentation layer the ``Notifier`` protocol
        exists to keep out.

        ``records_per_page`` is ``None`` for the one endpoint that takes no
        page size, and sending it is what keeps the size this CLI's rather
        than an SDK default's. The entries and the cursor are read from one
        parsed body, because ``requests.Response.json`` re-parses the page
        on every call.

        The only place :attr:`pages_left_unfetched` is set, and it is set
        after the request rather than before it, so a lookup that never
        landed leaves the flag as the entry point cleared it.
        """
        accepted = guard(subject, self.output.error)
        if accepted is None:
            return None
        if records_per_page is not None:
            kwargs["results_per_page"] = records_per_page
        body = json_on(self._endpoint(api, method)(accepted, **kwargs))
        entries = self._rl_list(body, *path)
        if entries is None:
            return None
        self.pages_left_unfetched = _stated_cursor(body, cursor) is not None
        return entries

    def _aggregated(
        self,
        guard: _Guard,
        subject: str,
        api: str,
        method: str,
        max_results: int | None,
        *path: str,
        page: str = _PAGE_STRING,
        cursor: tuple[str, ...] = _NEXT_PAGE,
        records_per_page: int | None = _RECORDS_PER_PAGE,
        **kwargs: Any,
    ) -> list[Any] | None:
        """Every page ``api.method`` holds about ``subject``, within a budget.

        The walk itself, and what each of its stops is for, is
        :func:`~rl_cli.services.walks.cursor_walk`; what is decided here is
        the request it makes and the words it stops with.

        ``method`` is the *non-aggregated* sibling — the one :meth:`_page`
        asks — and ``page`` is the keyword it carries the cursor on. The
        entries and the cursor come out of one parsed body, because
        ``requests.Response.json`` re-parses the page on every call.

        :attr:`pages_left_unfetched` is deliberately left as the entry
        point cleared it, even where the budget ended the walk early: it is
        read by a caller offering to fetch the rest, and this walk was
        already asked for as much as the analyst wanted.
        """
        accepted = guard(subject, self.output.error)
        if accepted is None:
            return None
        # Bound to a name of its own because the check above narrows
        # ``accepted`` for this body and not for a nested one.
        sent = accepted
        endpoint = self._endpoint(api, method)
        if records_per_page is not None:
            kwargs["results_per_page"] = records_per_page

        def fetched(stated: Any) -> CursorPage[Any] | None:
            body = json_on(endpoint(sent, **{page: stated}, **kwargs))
            entries = self._rl_list(body, *path)
            if entries is None:
                return None
            return CursorPage(entries, _stated_cursor(body, cursor))

        return cursor_walk(
            fetched,
            max_pages=_MAX_PIVOT_PAGES,
            budget=_record_budget(max_results),
            warn=self.output.warning,
            repeated="The endpoint sent a page it had already sent; stopping here.",
            exhausted=(
                f"Stopped after {_MAX_PIVOT_PAGES} pages; TitaniumCloud holds more than this."
            ),
        )

    @safe_call(default=None)
    def analyze_url(self, url: str) -> dict[str, Any] | None:
        report: dict[str, Any] | None = self._envelope(
            valid_url, url, "_url_threat", "get_url_report", private=self._url_lookups_are_private
        )
        return report

    @safe_call(default=None)
    def get_domain_report(self, domain: str) -> dict[str, Any] | None:
        report: dict[str, Any] | None = self._envelope(
            valid_domain, domain, "_domain_threat", "get_domain_report"
        )
        return report

    @safe_call(default=None)
    def get_files_from_domain(self, domain: str) -> list[dict[str, Any]] | None:
        return self._page(
            valid_domain, domain, "_domain_threat", "get_downloaded_files", "downloaded_files"
        )

    @safe_call(default=None)
    def get_files_from_domain_aggregated(
        self, domain: str, max_results: int | None = None
    ) -> list[dict[str, Any]] | None:
        return self._aggregated(
            valid_domain,
            domain,
            "_domain_threat",
            "get_downloaded_files",
            max_results,
            "downloaded_files",
        )

    @safe_call(default=None)
    def get_urls_from_domain(self, domain: str) -> list[dict[str, Any]] | None:
        return self._page(valid_domain, domain, "_domain_threat", "urls_from_domain", "urls")

    @safe_call(default=None)
    def get_urls_from_domain_aggregated(
        self, domain: str, max_results: int | None = None
    ) -> list[dict[str, Any]] | None:
        return self._aggregated(
            valid_domain, domain, "_domain_threat", "urls_from_domain", max_results, "urls"
        )

    @safe_call(default=None)
    def get_ips_from_domain(self, domain: str) -> list[dict[str, Any]] | None:
        return self._page(
            valid_domain, domain, "_domain_threat", "domain_to_ip_resolutions", "resolutions"
        )

    @safe_call(default=None)
    def get_ips_from_domain_aggregated(
        self, domain: str, max_results: int | None = None
    ) -> list[dict[str, Any]] | None:
        return self._aggregated(
            valid_domain,
            domain,
            "_domain_threat",
            "domain_to_ip_resolutions",
            max_results,
            "resolutions",
        )

    @safe_call(default=None)
    def get_related_domains(self, domain: str) -> list[dict[str, Any]] | None:
        return self._page(
            valid_domain, domain, "_domain_threat", "related_domains", "related_domains"
        )

    @safe_call(default=None)
    def get_related_domains_aggregated(
        self, domain: str, max_results: int | None = None
    ) -> list[dict[str, Any]] | None:
        return self._aggregated(
            valid_domain,
            domain,
            "_domain_threat",
            "related_domains",
            max_results,
            "related_domains",
        )

    @safe_call(default=None)
    def get_ip_report(self, ip: str) -> dict[str, Any] | None:
        report: dict[str, Any] | None = self._envelope(valid_ip, ip, "_ip_threat", "get_ip_report")
        return report

    @safe_call(default=None)
    def get_files_from_ip(self, ip: str) -> list[dict[str, Any]] | None:
        return self._page(valid_ip, ip, "_ip_threat", "get_downloaded_files", "downloaded_files")

    @safe_call(default=None)
    def get_files_from_ip_aggregated(
        self, ip: str, max_results: int | None = None
    ) -> list[dict[str, Any]] | None:
        return self._aggregated(
            valid_ip, ip, "_ip_threat", "get_downloaded_files", max_results, "downloaded_files"
        )

    @safe_call(default=None)
    def get_urls_from_ip(self, ip: str) -> list[dict[str, Any]] | None:
        return self._page(valid_ip, ip, "_ip_threat", "urls_from_ip", "urls")

    @safe_call(default=None)
    def get_urls_from_ip_aggregated(
        self, ip: str, max_results: int | None = None
    ) -> list[dict[str, Any]] | None:
        return self._aggregated(valid_ip, ip, "_ip_threat", "urls_from_ip", max_results, "urls")

    @safe_call(default=None)
    def get_domains_from_ip(self, ip: str) -> list[dict[str, Any]] | None:
        return self._page(valid_ip, ip, "_ip_threat", "ip_to_domain_resolutions", "resolutions")

    @safe_call(default=None)
    def get_domains_from_ip_aggregated(
        self, ip: str, max_results: int | None = None
    ) -> list[dict[str, Any]] | None:
        return self._aggregated(
            valid_ip, ip, "_ip_threat", "ip_to_domain_resolutions", max_results, "resolutions"
        )

    @safe_call(default=None)
    def get_files_from_url(self, url: str) -> list[dict[str, Any]] | None:
        # TCA-0403's downloaded-files query names a URL that was fetched,
        # so a scheme is required here where the domain lookups take a bare
        # host.
        return self._page(
            valid_url,
            url,
            "_url_threat",
            "get_downloaded_files",
            "files",
            private=self._url_lookups_are_private,
        )

    @safe_call(default=None)
    def get_files_from_url_aggregated(
        self, url: str, max_results: int | None = None
    ) -> list[dict[str, Any]] | None:
        return self._aggregated(
            valid_url,
            url,
            "_url_threat",
            "get_downloaded_files",
            max_results,
            "files",
            private=self._url_lookups_are_private,
        )

    @safe_call(default=None)
    def get_uri_index(self, uri: str) -> list[str] | None:
        """The SHA-1s of every sample seen at ``uri``.

        TCA-0401 takes an email address, URL, DNS name or IPv4 string and
        answers a list of hashes rather than records, which is why this
        one returns strings.
        """
        return self._page(
            valid_url_or_host,
            uri,
            "_uri_index",
            "get_uri_index",
            "uri_index",
            "sha1_list",
            cursor=_NEXT_PAGE_SHA1,
            records_per_page=None,
        )

    @safe_call(default=None)
    def get_uri_index_aggregated(
        self, uri: str, max_results: int | None = None
    ) -> list[str] | None:
        return self._aggregated(
            valid_url_or_host,
            uri,
            "_uri_index",
            "get_uri_index",
            max_results,
            "uri_index",
            "sha1_list",
            page=_PAGE_SHA1,
            cursor=_NEXT_PAGE_SHA1,
            records_per_page=None,
        )
