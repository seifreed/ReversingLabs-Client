"""The inputs both command groups take, spelled once for both of them.

A hash batch, a directory to write a downloaded sample into, the two
result budgets, the ``--yes`` a scripted destructive command needs, and the
bodies that act on them.

Where the two groups must say different words to the analyst, the words are
a parameter: sharing a declaration is not licence to change a published
``--help``.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import click

from rl_cli.render.output import RichOutput

# The ceiling on both result budgets, and the largest answer TitaniumCloud
# will go and fetch: its Advanced Search page holds 10000 records and the
# walk's own budget is ten of them. Click refuses a larger number rather
# than clamping it, so the analyst sees the range while the digit they
# typed is still in front of them.
MAX_LIMIT = 100_000

# Both result budgets — ``--limit`` and ``--max-results`` — are declared
# ``IntRange(min=1, max=MAX_LIMIT)``. Zero or less is not a smaller fetch:
# the SDK pagers branch on ``if not max_results`` and read it as "every page
# there is", and the listing endpoints clamp it to one record.

# What a stack of ``click.option`` calls does to a command function. These
# are only ever applied underneath a ``@group.command()``, so what they take
# and give back is the undecorated function, not click's function-or-Command.
type OptionStack = Callable[[Any], Any]


def hash_inputs(hashes_help: str) -> OptionStack:
    """The two ways to name a batch of samples: ``--hash-file`` and ``--hashes``.

    ``hashes_help`` is the whole ``--hashes`` help line, because the two
    groups do not word it the same. The flags, their short forms and the
    file's "one hash per line" shape are the same batch contract everywhere.
    """

    def decorate(command: Any) -> Any:
        command = click.option("--hashes", "-h", multiple=True, help=hashes_help)(command)
        return click.option(
            "--hash-file",
            "-f",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help="File with hashes (one per line)",
        )(command)

    return decorate


def required_hashes(
    hashes: tuple[str, ...],
    hash_file: Path | None,
    usage: str,
    *,
    positional: tuple[str, ...] = (),
) -> list[str]:
    """The batch to act on, or a usage error naming every way to give one.

    The hashes come back in the order the command line named them:
    positionals first, then the repeated ``-h`` flags, then the lines of
    ``--hash-file`` with their blanks dropped.

    Naming no hashes at all is a wrong command line rather than a failed
    operation, so it exits 2 with the usage. ``usage`` is the caller's
    because only some commands take a positional hash.
    """
    batch = [*positional, *hashes]
    if hash_file:
        # ``--hash-file`` aimed at a binary — a sample rather than the list
        # of its hashes is an easy slip — makes ``read_text`` raise a
        # ``UnicodeDecodeError`` that escapes to the top-level handler as
        # "Unexpected error", which tells the analyst to file a bug about
        # their own typo. Named here as the wrong file it is.
        try:
            text = hash_file.read_text()
        except UnicodeDecodeError as exc:
            raise click.UsageError(
                f"--hash-file {hash_file} is not a UTF-8 text file (expected one hash per line)"
            ) from exc
        batch.extend(line.strip() for line in text.splitlines() if line.strip())
    if not batch:
        raise click.UsageError(f"No hashes provided. {usage}")
    return batch


def output_dir_option(command: Any) -> Any:
    """``--output-dir``: where a command that writes live malware writes it.

    ``Path.cwd`` must stay uncalled: click evaluates a callable default once
    per invocation, while calling it here would bind the directory current
    at *import* and write live malware where the analyst no longer stands.
    """
    return click.option(
        "--output-dir",
        type=click.Path(path_type=Path),
        default=Path.cwd,
        help="Output directory  [default: the current directory]",
    )(command)


def fetch_all_flag(command: Any) -> Any:
    """``--all``: walk every page of a pivot lookup, not just the first.

    Both groups' pivot lookups offer it and both must keep meaning the same
    thing by it. What they offer alongside differs, so only the flag is
    shared: ticloud's endpoints take a ``--max-results`` bound the A1000
    ones have no aggregated form for.
    """
    return click.option(
        "--all", "fetch_all", is_flag=True, help="Fetch every page, not just the first"
    )(command)


def max_results_option(help_text: str) -> OptionStack:
    """``--max-results``: how many records an aggregated walk may collect.

    It does not bound a walk that has stopped making progress: the pagers
    stop on ``len(results) >= max_results``, which an endpoint answering an
    empty page and another cursor never reaches. Only a page budget inside
    the service stops that, and the walks that have one must keep it.

    The help line is the caller's because the two groups do not offer it on
    the same terms: ticloud's pivots take it as implying ``--all``, while
    ``yara-repo-list`` refuses it without ``--all``.
    """

    def decorate(command: Any) -> Any:
        return click.option(
            "--max-results", type=click.IntRange(min=1, max=MAX_LIMIT), help=help_text
        )(command)

    return decorate


def limit_option(command: Any) -> Any:
    """``--limit``: a ceiling on the records a listing brings back.

    One help line for all three commands that take it, because all three
    offer it on those same terms.
    """
    return click.option(
        "--limit",
        "-l",
        type=click.IntRange(min=1, max=MAX_LIMIT),
        default=100,
        help="Maximum number of results",
    )(command)


def yes_flag(command: Any) -> Any:
    """``--yes``: take the confirmation as given, for a caller that cannot answer.

    Every command that prompts before an irreversible call takes it. It
    defaults to ``False`` and an analyst at a terminal is still asked; it
    exists for the scripted invocation, which has no answer to give, since
    ``click.confirm`` reads a closed stdin as an abort.

    Honour it through :func:`confirmed`, never in a command's own words.
    """
    return click.option(
        "--yes",
        is_flag=True,
        help="Skip the confirmation prompt (for non-interactive use)",
    )(command)


def confirmed(
    output: RichOutput, yes: bool, question: str, *, cancelled: str = "Cancelled"
) -> bool:
    """Put ``question`` unless ``--yes`` answered it, and say so when refused.

    Answers whether to go on, so the caller's guard is ``if not
    confirmed(...): return``. ``cancelled`` is the line printed when the
    analyst says no, and is the caller's because two commands name what
    they did not do.
    """
    if yes or click.confirm(question):
        return True
    output.info(cancelled)
    return False


def partial_answer_notice(
    output: RichOutput,
    how_to_fetch: str,
    *,
    pages_left: bool = False,
    collected: Sequence[Any] | None = None,
    max_results: int | None = None,
) -> None:
    """Say the fetch left records behind, in the one wording every command uses.

    What differs between the commands is the option that asks for the rest
    — ``--limit``, ``--all``, ``--max-results`` — so that phrase is the
    caller's and the sentence is not.

    A command may be partial in either of two ways. ``pages_left`` is the
    paging services' own report that the appliance promised another page.
    The cap is taken here rather than as a boolean each caller computes, so
    that the ``>=`` stays: only the vendored SDK trims a walk to the cap
    exactly, and a release returning whole pages overshoots it.

    Hence "may be", which the ``capped`` half needs: a set of exactly the
    cap's size may be the whole of it, and counting cannot tell which.

    **Call it unguarded**, straight after the step it reports on, at every
    call site. A call that did not land leaves ``pages_left`` cleared —
    that is what ``cleared_per_call`` guarantees — and hands ``collected``
    a ``None`` that fills no cap, so both halves already say nothing.
    Guarding on the step's result would instead silence an answer that was
    empty *and* stated another page, since ``run_step`` returns ``None``
    for both.

    A listing may say this and still be drawn short of what it did fetch,
    which is a separate sentence about the rendering.
    """
    capped = max_results is not None and collected is not None and len(collected) >= max_results
    if pages_left or capped:
        output.info(f"More results may be waiting; {how_to_fetch} to fetch them.")
