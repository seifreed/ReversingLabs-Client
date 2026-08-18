"""URL submissions and network threat intelligence lookups."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rl_cli.services.a1000.service import A1000Service, list_from_envelope
from rl_cli.services.a1000.session import A1000Session
from rl_cli.services.addresses import valid_domain, valid_ip, valid_url, valid_url_or_host
from rl_cli.services.decorators import with_client
from rl_cli.services.http import json_on
from rl_cli.services.partial_answers import cleared_per_call
from rl_cli.services.protocols import Notifier
from rl_cli.services.walks import CursorPage, cursor_walk

# How large a page these three endpoints are asked for, and how many pages
# one ``--all`` walk of an IP pivot may fetch. 200 pages of 500 is 100000
# records, the ceiling ``_shared_inputs.MAX_LIMIT`` puts on the largest
# listing this CLI will fetch. The tests pin that product, so the page size
# is sent rather than left to an SDK default that could change under it.
#
# The page budget is the only bound these three have: unlike a listing, a
# pivot takes no ``--limit``, and unlike the ticloud pivots it takes no
# ``--max-results`` either, so the walk's own budget is the whole of what
# stands between a busy address and an unbounded loop.
_RECORDS_PER_PAGE = 500
_MAX_PIVOT_PAGES = 200


@cleared_per_call("pages_left_unfetched")
class A1000NetworkService(A1000Service):
    """Service for URL analysis and the network intelligence endpoints."""

    def __init__(self, session: A1000Session, output: Notifier | None = None):
        super().__init__(session, output)
        # Whether the last first-page lookup left pages behind, for a caller
        # that can offer to fetch them: a fact about the answer, not advice
        # about what to do with it. It speaks for the call that just
        # finished and for no earlier one, which is why the class decorator
        # clears it at every entry point and :meth:`_first_page` is the only
        # thing that sets it.
        self.pages_left_unfetched = False

    # ---------- URL analysis ----------
    @with_client(default=None)
    def submit_url(self, url: str, crawler: str = "local") -> dict[str, Any] | None:
        """Submit a URL for analysis.

        ``crawler`` selects where the URL is fetched from; valid values are
        ``"local"`` (the appliance) and ``"cloud"`` per the SDK contract.
        """
        target = valid_url(url, self.output.error)
        if target is None:
            return None
        return json_on(self.client.submit_url(target, crawler))

    @with_client(default=None)
    def get_url_report(self, task_id: str) -> dict[str, Any] | None:
        """Fetch the analysis report for a URL submission ``task_id``.

        Use the ``task_id`` returned by :meth:`submit_url`.
        """
        return json_on(self.client.get_submitted_url_report(task_id, retry=False))

    @with_client(default=None)
    def url_status(self, task_id: str) -> dict[str, Any] | None:
        """Check the status of a URL submission ``task_id``."""
        return json_on(self.client.check_submitted_url_status(task_id))

    # ---------- Network intelligence ----------
    def _first_page(
        self, address: str, call: Callable[..., Any], key: str, noun: str
    ) -> list[dict[str, Any]] | None:
        """One page of an IP lookup, saying so when the appliance has more.

        ``address`` must already have been accepted by the wrapper's guard,
        and the wrapper is what returns for a refused one: ``call`` is
        spelled ``self.client.<method>`` at every call site, so it is
        evaluated before this body runs.

        These three endpoints carry the cursor for the next page in
        ``next_page``, so a first page is not the whole answer and must not
        be reported as one. It is asked for at the same size the walk asks
        for, so what "one page" means does not depend on which of the two
        the analyst reached.

        What is said here is what was measured and nothing about what to
        type next: naming ``--all`` would make this service unusable to a
        caller without that option, which is the dependency on the
        presentation layer the ``Notifier`` protocol exists to keep out.

        An answer we could not read is named rather than returned as a bare
        ``None``, so the analyst is told which part of it was missing.

        The only place :attr:`pages_left_unfetched` is set, and it is set
        after the request rather than before it, so a lookup that never
        landed leaves the flag as the entry point cleared it.
        """
        body = json_on(call(address, page_size=_RECORDS_PER_PAGE))
        entries = list_from_envelope(body, key)
        if entries is None:
            self.output.error(f"Network response carried no {noun} list under '{key}'")
            return None
        self.pages_left_unfetched = bool(body.get("next_page"))
        if self.pages_left_unfetched:
            self.output.warning(f"More {noun} than one page holds; showing the first page only.")
        return entries

    def _every_page(
        self, address: str, call: Callable[..., Any], key: str, noun: str
    ) -> list[dict[str, Any]] | None:
        """Every page of an IP lookup, within a budget, saying where it stopped.

        The walk itself, and what each of its stops is for, is
        :func:`~rl_cli.services.walks.cursor_walk`; what is decided here is
        the request it makes, the words it stops with, and that it takes no
        record budget — an ``--all`` on a busy address has no ``--limit``
        beside it and, unlike the ticloud pivots, no ``--max-results``
        either, so the page budget is the whole of it.

        An answer we could not read is named rather than returned as a bare
        ``None``, so the analyst is told which part of it was missing.

        :attr:`pages_left_unfetched` is deliberately left as the entry
        point cleared it, even where the budget ended the walk early: it is
        read by a caller offering to fetch the rest, and this walk was
        already asked for every page there is.
        """

        def fetched(cursor: Any) -> CursorPage[dict[str, Any]] | None:
            body = json_on(call(address, page=cursor, page_size=_RECORDS_PER_PAGE))
            entries = list_from_envelope(body, key)
            if entries is None:
                self.output.error(f"Network response carried no {noun} list under '{key}'")
                return None
            return CursorPage(entries, body.get("next_page"))

        return cursor_walk(
            fetched,
            max_pages=_MAX_PIVOT_PAGES,
            budget=None,
            warn=self.output.warning,
            repeated=f"The appliance sent a page of {noun} it had already sent; stopping here.",
            exhausted=(
                f"Stopped after {_MAX_PIVOT_PAGES} pages; "
                f"the appliance holds more {noun} than this."
            ),
        )

    @with_client(default=None)
    def get_domain_report(self, domain: str) -> dict[str, Any] | None:
        # The endpoint interpolates this into a path segment without
        # quoting it — unlike ``network_url_report``, which quotes — so a
        # URL here is a request about something else, answered 200 and
        # empty, and reported as "no intelligence" for the domain.
        name = valid_domain(domain, self.output.error)
        if name is None:
            return None
        return json_on(self.client.network_domain_report(name))

    @with_client(default=None)
    def get_ip_report(self, ip: str) -> dict[str, Any] | None:
        address = valid_ip(ip, self.output.error)
        if address is None:
            return None
        return json_on(self.client.network_ip_addr_report(address))

    @with_client(default=None)
    def get_files_from_ip(self, ip: str) -> list[dict[str, Any]] | None:
        address = valid_ip(ip, self.output.error)
        if address is None:
            return None
        return self._first_page(
            address, self.client.network_files_from_ip, "downloaded_files", "files"
        )

    @with_client(default=None)
    def get_domains_from_ip(self, ip: str) -> list[dict[str, Any]] | None:
        address = valid_ip(ip, self.output.error)
        if address is None:
            return None
        return self._first_page(address, self.client.network_ip_to_domain, "resolutions", "domains")

    @with_client(default=None)
    def get_files_from_ip_aggregated(self, ip: str) -> list[dict[str, Any]] | None:
        address = valid_ip(ip, self.output.error)
        if address is None:
            return None
        return self._every_page(
            address, self.client.network_files_from_ip, "downloaded_files", "files"
        )

    @with_client(default=None)
    def get_domains_from_ip_aggregated(self, ip: str) -> list[dict[str, Any]] | None:
        address = valid_ip(ip, self.output.error)
        if address is None:
            return None
        return self._every_page(address, self.client.network_ip_to_domain, "resolutions", "domains")

    @with_client(default=None)
    def get_urls_from_ip_aggregated(self, ip: str) -> list[dict[str, Any]] | None:
        address = valid_ip(ip, self.output.error)
        if address is None:
            return None
        return self._every_page(address, self.client.network_urls_from_ip, "urls", "URLs")

    @with_client(default=None)
    def get_urls_from_ip(self, ip: str) -> list[dict[str, Any]] | None:
        address = valid_ip(ip, self.output.error)
        if address is None:
            return None
        return self._first_page(address, self.client.network_urls_from_ip, "urls", "URLs")

    @with_client(default=None)
    def get_network_url_report(self, url: str) -> dict[str, Any] | None:
        # This endpoint quotes its argument and documents both forms, so
        # a bare host is as good as a full URL here.
        target = valid_url_or_host(url, self.output.error)
        if target is None:
            return None
        return json_on(self.client.network_url_report(target))
