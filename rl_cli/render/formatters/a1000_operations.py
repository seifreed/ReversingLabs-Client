"""The A1000 renderings of appliance operations: YARA rulesets and reanalysis.

Every renderer here writes to the ``Console`` it is given; the two tables
that cap what they draw return how many rows they drew, so the caller can
say "Showing N of M" without assuming the cap.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from rl_cli.models.payload import ReanalysisOutcome, SampleFacts
from rl_cli.models.shapes import count, dict_rows, mapping
from rl_cli.render.formatters.panels import stated_cell
from rl_cli.render.markup import safe
from rl_cli.text import digest_cell, sanitize

# The match counts a ruleset reports, which the "Matches" column totals.
_MATCH_KINDS = ("malicious", "suspicious", "goodware", "unknown")

# How many rulesets the table draws for a caller that binds no cap: an
# appliance holding several hundred of them would otherwise answer
# `yara-list` with a screenful per scroll.
_RULESET_ROW_LIMIT = 20

# The reanalysis listing's default, which is a preview of a batch just
# submitted rather than a listing to read through: the panels' number, not
# the tables'.
_REANALYSIS_ROW_LIMIT = 10


def print_yara_rulesets_table(
    rulesets: list[dict[str, Any]], *, max_rows: int = _RULESET_ROW_LIMIT, console: Console
) -> int:
    """Render up to ``max_rows`` of the YARA rulesets the appliance holds.

    Returns the number of rows drawn, which is how every capped listing in
    this package reports a cut: the command says "Showing N of M; use -o
    json for the full set" over it. ``dict_rows`` drops the entries this
    table cannot render, so the count is not ``min(max_rows, len(rulesets))``.
    """
    table = Table(title="YARA Rulesets", show_header=True)
    # The appliance reports status/<x>_match_count/last_matched; it has no
    # enabled/rule_count/modified fields.
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Malicious", style="red")
    table.add_column("Matches", style="yellow")
    table.add_column("Last Matched", style="white")
    table.add_column("Owner", style="blue")
    rows = dict_rows(rulesets)[:max_rows]
    for ruleset in rows:
        status = ruleset.get("status")
        if status is None:
            status = "active" if ruleset.get("enabled") else "inactive"
        status_display = safe(status)
        # A ruleset that failed to compile says so in "error_message".
        error = ruleset.get("error_message")
        if error:
            status_display = f"[red]{status_display}: {safe(error)}[/red]"
        # The total is dominated by goodware — a packer ruleset with 48,213
        # goodware hits outranks a Conti ruleset with 12 malicious ones — so
        # the malicious count gets a column of its own.
        matches = sum(count(ruleset.get(f"{kind}_match_count")) for kind in _MATCH_KINDS)
        table.add_row(
            safe(stated_cell(ruleset.get("name"))),
            status_display,
            f"{count(ruleset.get('malicious_match_count')):,}",
            f"{matches:,}",
            safe(ruleset.get("last_matched") or ruleset.get("modified") or "Never"),
            safe(stated_cell(ruleset.get("owner"))),
        )
    console.print(table)
    return len(rows)


def print_yara_content(content: str, console: Console) -> None:
    """Render YARA rule text with monokai syntax highlighting.

    ``Syntax`` does not read markup, so the rule text only needs the
    control characters taken out — it happily emits an escape sequence
    from inside a rule's comment or string literal otherwise.
    """
    console.print(Syntax(sanitize(content), "yara", theme="monokai", line_numbers=True))


def _reanalysis_status(outcome: ReanalysisOutcome) -> str:
    """The Status cell for one graded entry: the verdict, then the reasons for it.

    Whether the sample was taken is read off ``accepted`` and never decided
    again from the reasons beside it. That is the whole point of grading an
    entry once: the count above the table and the exit status read the same
    flag, so a row cannot draw "Refused" under a green "Reanalysis started
    for 5 samples" — change how an entry grades and both move together.

    Escaping happens here rather than in :class:`ReanalysisOutcome`: the
    ``[red]`` is markup this function wrote and must survive, while every
    string the appliance supplied must not be read as markup at all.
    """
    reasons: list[str] = []
    if outcome.refusal is not None:
        reasons.append(safe(outcome.refusal))
    if outcome.stated is not None:
        reasons.append(safe(outcome.stated))
    if outcome.queued:
        reasons.append(f"Queued: {', '.join(safe(name) for name in outcome.queued)}")
    if outcome.failed:
        reasons.append(f"Failed: {'; '.join(safe(engine) for engine in outcome.failed)}")
    detail = " | ".join(reasons)
    if outcome.accepted:
        # Nothing in the answer turned this one down; "Submitted" is what
        # an entry that states no reason at all says.
        return detail or "Submitted"
    return f"[red]Refused: {detail}[/red]" if detail else "[red]Refused[/red]"


def print_reanalyze_results_table(
    results: list[Any], *, max_rows: int = _REANALYSIS_ROW_LIMIT, console: Console
) -> int:
    """Render up to ``max_rows`` reanalysis entries, and say how many were drawn.

    Every entry the caller was given gets a row, including one that is not a
    record at all. The sentence above this table counts the same entries
    graded the same way — ``ReanalysisOutcome.of`` reads a non-record as "not
    accepted" precisely so that count can have it — so an entry skipped here
    is a refusal announced with no row to attribute it to: ``["oops"]``
    printed "Reanalysis refused for all 1 samples", exited 1, and drew an
    empty table. ``mapping`` turns the unreadable entry into the empty
    record, which is a Hash cell saying the payload stated none.

    The sentence and this table share one *rule* — ``ReanalysisOutcome.of``
    — and not one *grading*: the caller grades the answer into a
    ``ReanalysisBatch`` and hands this function the raw list, which grades
    it again, entry by entry. Two invocations of one rule agree only while
    the batch grades exactly the entries it was given, in order, which
    ``tests/test_formatters.py`` pins from the side that can see both.

    "Every entry" is every entry of a *list*; an answer that is not one
    holds no entries at all and draws the empty table. The endpoint answers
    with a list and an error body is a mapping, which this slice read as a
    key — ``KeyError: slice(None, 10, None)`` out of a renderer nothing
    wraps in a ``try`` — while a string sliced and iterated happily, drawing
    one row per character about samples never mentioned.
    """
    table = Table(title="Reanalysis Results", show_header=True)
    table.add_column("Hash", style="yellow")
    table.add_column("Status", style="green")
    # The list is taken whole, entry by entry, rather than filtered to the
    # mappings: the count above this table grades the unreadable entries
    # too, so one skipped here is a refusal announced with no row under it.
    rows = results[:max_rows] if isinstance(results, list) else []
    for result in rows:
        # A reanalyze entry is {"detail": {hashes...}, "analysis": [{"name",
        # "code", "message"}, ...]}; there is no flat hash/status field.
        entry = mapping(result)
        detail = mapping(entry.get("detail"))
        hash_val = stated_cell(entry.get("hash") or SampleFacts.of(detail).digest)
        table.add_row(safe(digest_cell(hash_val)), _reanalysis_status(ReanalysisOutcome.of(result)))
    console.print(table)
    return len(rows)
