"""Direct tests for the pagination walks in ``rl_cli.services.walks``.

The walks are exercised end to end through the A1000 and TitaniumCloud
services, but the record budget's arithmetic is easiest to pin here, over
a ``page`` callable that counts how many requests the walk actually spent:
a budget caps records, and it must cap requests at the same boundary
rather than spending one more page to collect records it then discards.
"""

from __future__ import annotations

from rl_cli.services.walks import CursorPage, cursor_walk


def _silent(_message: str) -> None:
    raise AssertionError("the walk warned when it should not have")


def test_a_budget_reached_exactly_stops_without_fetching_another_page():
    """``>= budget`` returns on the page that fills it; ``> budget`` would
    fetch one more page only to slice it back off, spending a request the
    budget was there to save."""
    calls: list[object] = []

    def page(cursor: object) -> CursorPage[str]:
        index = len(calls)
        calls.append(cursor)
        # Every page carries one record and always names a further cursor,
        # so nothing but the budget can end this walk.
        return CursorPage([f"r{index}"], f"c{index}")

    result = cursor_walk(
        page,
        max_pages=100,
        budget=2,
        warn=_silent,
        repeated="repeated cursor",
        exhausted="page budget spent",
    )

    assert result == ["r0", "r1"]
    assert len(calls) == 2, "the walk fetched a page past the budget it had already filled"
