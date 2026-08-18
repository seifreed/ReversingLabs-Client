"""The ``a1000`` group itself, and what its command modules share.

The group has to live here rather than in ``__init__``: the command modules
decorate with it, and ``__init__`` imports them to register their commands,
so a group defined there would be a circular import. The leading underscore
is the module's, not its contents': the names inside it are for the sibling
modules to use.

What the ``ticloud`` group asks for as well — the hash batch, whose ``-h``
must go on meaning a hash rather than help, ``--output-dir``, ``--yes``, and
the partial-answer sentence — sits one directory up in
``cli/commands/_shared_inputs``. :func:`partial_answer_notice` is
re-exported from there because it has to be one sentence for both groups.
"""

from collections.abc import Sequence
from typing import Any, Protocol

import click

from rl_cli.cli.commands._shared_inputs import OptionStack

# The redundant alias is what makes the re-export explicit rather than
# accidental, which is what mypy asks for under ``no_implicit_reexport``;
# the separate statement is where ruff's isort puts it.
from rl_cli.cli.commands._shared_inputs import (
    partial_answer_notice as partial_answer_notice,
)
from rl_cli.cli.context import RenderConsole, RichRenderer
from rl_cli.render.output import RichOutput


def truncation_notice(output: RichOutput, shown: int, total: int) -> None:
    """Say that a listing was capped by the renderer, in one wording for all of them.

    The whole set is in hand and only part of it was drawn, so the advice is
    the dump and not ``--limit``, which decides how many results are fetched
    rather than how many are shown. ``shown`` must be the count the renderer
    drew, never the cap it was given: a result array carries entries the
    renderers skip.

    For a command's listing only. The report panels cap their own lists in
    place, where a note after the last panel could not say which of eight it
    was about.
    """
    if total > shown:
        output.info(f"Showing {shown} of {total}; use -o json for the full set")


class CappedPrinter(Protocol):
    """A renderer with its cap already bound: draws ``data``, answers how many it drew.

    The cap is the caller's because the renderers do not spell it the same:
    ``print_samples_panels`` counts panels, the tables count rows.
    """

    def __call__(self, data: Any, /, *, console: RenderConsole) -> int: ...


def capped_listing(output: RichOutput, draw: CappedPrinter) -> RichRenderer:
    """Draw a capped listing, then say it was capped, in one place."""

    def render(data: list[Any], *, console: RenderConsole) -> None:
        truncation_notice(output, draw(data, console=console), len(data))

    return render


def report_format_option(choices: Sequence[str], *, default: str, help_text: str) -> OptionStack:
    """``--format``: which variant of a report is meant.

    Named ``report_format`` because the parameter name click would derive
    from the flag shadows the builtin in the command body. The choices, the
    default and the sentence describing them are the caller's, since the
    commands do not offer the same variants.
    """

    def decorate(command: Any) -> Any:
        return click.option(
            "--format",
            "-f",
            "report_format",
            type=click.Choice(list(choices)),
            default=default,
            help=help_text,
        )(command)

    return decorate


def json_flag(command: Any) -> Any:
    """``--json``: dump the fetched document instead of drawing a panel for it.

    Only for a command that fetches **one document** and renders a lossy
    summary of it, which is what makes the flag's name true: ``display``
    defers to ``OutputFormatter.display``, which under the default ``rich``
    format dumps a mapping as JSON but renders a *list* as another rich
    table and a bare string as plain text. On a listing, ``--json`` would
    print a different table rather than the document, so listings offer no
    such flag and point at ``-o json``.

    A command that has it must pass the flag to ``display`` as ``plain``, so
    that the two spellings of "defer to the dump" stay one decision.
    """
    return click.option(
        "--json", "show_json", is_flag=True, help="Dump the fetched document instead of a summary"
    )(command)


@click.group()
def a1000() -> None:
    """A1000 malware analysis platform commands."""
