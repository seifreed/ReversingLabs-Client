"""The two page walks this package runs, each written once.

Both halves of this CLI page the same two ways — a cursor the endpoint
states and hands back, or a page number the walk counts itself — so the
stop order and the wording are settled here, once, for both. The
endpoint-specific parts stay with the endpoint: the guard, the envelope
key, the SDK call and the words for a warning are what each service passes
in.

Neither walk is handed to the SDK's own ``*_aggregated`` methods, every one
of which is a ``while True`` that returns on the endpoint's own answer
alone (SDK ticloud.py:1104-1124, 1215-1242, 1447-1471 …; SDK
a1000.py:2116-2146, 2296-2302, 2364-2370 …). Those cannot stop against an
endpoint that names a further page forever, and every page is a metered
request, so a budget the walk owns is what bounds it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from rl_cli.models.payload import SearchPage
from rl_cli.models.payload import search_page as parse_search_page
from rl_cli.services.http import json_on


class CursorPage[Record](NamedTuple):
    """One page of a cursor walk: its records, and where the next one is.

    ``following`` is the cursor as the endpoint stated it, which is how it
    has to be handed back. Absent, null or empty is the last page; anything
    else names a page this walk has not fetched.
    """

    records: list[Record]
    following: Any


class NumberedWalk(NamedTuple):
    """What a numbered walk collected, and whether the corpus held more."""

    records: list[dict[str, Any]]
    pages_left_unfetched: bool


def fetch_search_page(
    request: Callable[..., Any],
    query: str,
    page_number: int,
    records_per_page: int,
    report: Callable[[str], None],
) -> SearchPage | None:
    """Call either search endpoint and read its shared page envelope."""
    found = parse_search_page(
        json_on(
            request(
                query_string=query,
                page_number=page_number,
                records_per_page=records_per_page,
            )
        )
    )
    if found is None:
        report("Search response carried no results envelope")
    return found


def cursor_walk[Record](
    page: Callable[[Any], CursorPage[Record] | None],
    *,
    max_pages: int,
    budget: int | None,
    warn: Callable[[str], None],
    repeated: str,
    exhausted: str,
) -> list[Record] | None:
    """Every page ``page`` states, within a budget; ``None`` for an unreadable one.

    ``page`` is handed the cursor to ask for — ``None`` for the first page
    — and answers the records and the cursor after them, or ``None`` for an
    answer it could not read, which it has already reported and which ends
    the walk as the failed lookup it is rather than as the end of the
    corpus.

    Four things stop the walk. Two are the endpoint's answer: it named no
    further cursor, or it has handed over ``budget`` records. The third is
    ``max_pages``, which is the walk's own, and it is here because neither
    of the other two bounds what a walk may spend — a record budget caps
    records, not requests, and ``--all`` names no number at all. The fourth
    is a cursor the endpoint has already stated on this walk, which is the
    walk having stopped making progress.

    That last one is the counterpart to counting pages, which a cursor
    endpoint cannot do. Every cursor is remembered rather than only the
    last, so a server alternating between two of them stops on the third
    request instead of running out the page budget.

    A page that carries no records is *not* a stop: only the cursor says
    where the corpus ends. These endpoints filter pages server-side, so an
    empty page can sit between two full ones, and ending the walk there
    hands back part of an answer as though it were all of it. Nothing is
    bought by stopping either — an empty page whose cursor repeats is the
    stop above, and one whose cursor moves is bounded by ``max_pages``.
    """
    collected: list[Record] = []
    stated: list[Any] = []
    cursor: Any = None
    for _ in range(max_pages):
        fetched = page(cursor)
        if fetched is None:
            return None
        collected.extend(fetched.records)
        if budget is not None and len(collected) >= budget:
            return collected[:budget]
        if not fetched.following:
            return collected
        if fetched.following in stated:
            warn(repeated)
            return collected
        stated.append(fetched.following)
        cursor = fetched.following
    warn(exhausted)
    return collected


def numbered_walk(
    page: Callable[[int], SearchPage | None], *, limit: int, max_pages: int
) -> NumberedWalk | None:
    """Pages 1, 2, 3 … up to ``limit`` records; ``None`` for an unreadable one.

    ``page`` is handed the number to ask for and answers that page, or
    ``None`` for an answer it could not read, which it has already reported
    and which ends the walk as the failed search it is rather than as the
    end of the corpus.

    The page asked for is counted here and never read out of the envelope,
    which is one of this walk's two departures from the SDK's: that one
    posts ``page_number=next_page`` straight back, so an endpoint that sets
    ``more_pages`` and omits ``next_page`` is asked for page ``null`` —
    served page 1 again by anything lenient, and answered with ``limit``
    duplicate records. The other is ``max_pages``, the walk's own bound and
    the only thing standing between a mistyped ``--limit`` and a metered
    request per page of a corpus nobody meant to buy.

    An empty page is not a stop, because only ``more_pages`` says where the
    corpus ends: a page filtered down to nothing can sit between two full
    ones, and stopping there returns part of an answer as the whole one.

    What is left behind is measured rather than inferred — the last page's
    own ``more_pages``, plus the records fetched past ``limit`` and trimmed
    off — so a corpus of exactly ``limit`` reports nothing waiting.
    """
    collected: list[dict[str, Any]] = []
    more_pages = True
    number = 1
    while more_pages and len(collected) < limit and number <= max_pages:
        found = page(number)
        if found is None:
            return None
        collected.extend(found.entries)
        more_pages = found.more_pages
        number += 1
    return NumberedWalk(collected[:limit], more_pages or len(collected) > limit)


def numbered_search(
    request: Callable[..., Any],
    query: str,
    *,
    limit: int,
    records_per_page: int,
    max_pages: int,
    report: Callable[[str], None],
) -> NumberedWalk | None:
    """Walk the numbered search pages exposed by either service."""
    return numbered_walk(
        lambda number: fetch_search_page(request, query, number, records_per_page, report),
        limit=limit,
        max_pages=max_pages,
    )
