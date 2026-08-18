"""Yara Online Source repositories: the outside sources the appliance imports rulesets from."""

from collections.abc import Callable
from typing import Any

import click

from rl_cli.cli.commands._shared_inputs import (
    OptionStack,
    confirmed,
    fetch_all_flag,
    max_results_option,
    yes_flag,
)
from rl_cli.cli.commands.a1000._shared import a1000, partial_answer_notice
from rl_cli.cli.context import probed_a1000, run_step
from rl_cli.models.yara_repository import YaraRepositorySpec, redact_api_token
from rl_cli.render.output import OutputFormatter
from rl_cli.services.a1000 import A1000YaraService


def _redacted(formatter: OutputFormatter) -> Callable[[Any], None]:
    """Show an answer with the repository token taken out of it.

    Every command in this file must render through this: the token the
    analyst supplied — a live GitHub PAT — is echoed back by the appliance,
    and stdout is where a redirect or a pasted terminal takes it somewhere
    far less protected than the 0600 file it belongs in.
    """
    return lambda answer: formatter.display(redact_api_token(answer))


@a1000.command()
@click.option(
    "--filter",
    "active_filter",
    type=click.Choice(["all", "user", "system"]),
    help="Filter repositories by type",
)
@click.option(
    "--page", "page_number", type=click.IntRange(min=1), help="Page number (single-page mode)"
)
@click.option("--page-size", type=click.IntRange(min=1), help="Results per page")
@fetch_all_flag
@max_results_option("Cap aggregated results (requires --all)")
@click.pass_context
def yara_repo_list(
    ctx: click.Context,
    active_filter: str | None,
    page_number: int | None,
    page_size: int | None,
    fetch_all: bool,
    max_results: int | None,
) -> None:
    """List Yara Online Source repositories."""
    # Each of these options is read on one branch only, so an accepted pair
    # would be half-ignored: --page vanishes under --all, and --max-results
    # without --all caps nothing.
    if fetch_all and page_number is not None:
        raise click.UsageError("Pass either --page or --all, not both")
    if max_results is not None and not fetch_all:
        raise click.UsageError("--max-results caps the aggregated walk; pass it with --all")

    service, output, formatter = probed_a1000(ctx, A1000YaraService)

    def fetch() -> list[dict[str, Any]] | None:
        if fetch_all:
            return service.list_yara_repositories_aggregated(
                active_filter=active_filter,
                page_size=page_size or 100,
                max_results=max_results,
            )
        return service.list_yara_repositories(
            active_filter=active_filter,
            page_size=page_size,
            page_number=page_number,
        )

    repositories = run_step(
        output,
        "Fetching YARA repositories...",
        fetch,
        success="YARA repositories retrieved",
        failure="Failed to list YARA repositories",
        # An appliance with no user repositories answers an empty list,
        # which is an answer rather than a failure.
        empty="No YARA repositories configured",
        render=_redacted(formatter),
        empty_formatter=formatter,
    )
    # A walk that fills its cap was cut short by it; unsaid, the analyst
    # reads the first N repositories as all of them.
    partial_answer_notice(
        output, "raise --max-results", collected=repositories, max_results=max_results
    )


def _api_token_options(token_help: str) -> OptionStack:
    """The two ways to supply the repository token, declared for both commands.

    The token authenticates to somebody else's system — typically a GitHub
    PAT — and an argument is the one place this CLI cannot keep a credential:
    argv is readable in ``ps`` for as long as the call runs, and the line
    lands in the shell history afterwards. ``--api-token`` stays for scripts
    and for the public repositories that pass ``--api-token ''``;
    ``--api-token-stdin`` is the way that keeps it off the command line.

    Only those two sources, and no environment variable: the ``.env`` this
    CLI reads is a plaintext file in the working directory that nothing here
    creates 0600 or keeps out of a commit.

    ``token_help`` is the caller's because the two commands do not offer the
    flag on the same terms — create defaults it to a public repository's
    empty token, update replaces whatever is stored.
    """

    def decorate(command: Any) -> Any:
        command = click.option(
            "--api-token-stdin",
            is_flag=True,
            help="Read the token from stdin instead (hidden prompt at a terminal)",
        )(command)
        # ``default=None`` on both commands, so "not given" stays tellable
        # from "given as the empty string", which is how a public repository
        # is stated. Not ``required=True`` either: click would refuse the
        # command line that supplies the token over stdin, so the
        # requirement lives in :func:`_supplied_api_token`, which sees both.
        return click.option("--api-token", default=None, help=token_help)(command)

    return decorate


def _token_from_stdin() -> str:
    """The token off stdin: hidden at a terminal, verbatim from a pipe.

    A pipe gets a plain read, so nothing decides anything about the bytes
    the caller sent. A terminal gets click's hidden prompt, which is
    ``getpass``: the token is not echoed and never reaches the scrollback.
    The prompt goes to stderr, because stdout is the command's document.
    """
    stream = click.get_text_stream("stdin")
    if stream.isatty():
        # Annotated because ``click.prompt`` is typed to return ``Any``,
        # and this function promises a ``str`` to the spec it fills.
        typed: str = click.prompt("Repository API token", hide_input=True, err=True)
        return typed
    return _without_one_line_ending(stream.read())


def _without_one_line_ending(piped: str) -> str:
    """One trailing line ending removed, and nothing else touched.

    A piped token almost always carries the newline ``echo`` or an editor
    put there, and sending it on authenticates nothing. Only the last one
    goes, and never ``.strip()``: that would also eat a character the token
    really ends in, and a credential silently reshaped on the way out is
    unfindable from the 401 it causes. CRLF counts as the one ending.
    """
    for ending in ("\r\n", "\n"):
        if piped.endswith(ending):
            return piped[: -len(ending)]
    return piped


def _supplied_api_token(api_token: str | None, api_token_stdin: bool, *, required: bool) -> str:
    """The token, from exactly one of the two ways of giving it.

    Both at once is a usage error rather than a precedence rule to look up:
    a script that pipes one and passes the other would never learn which of
    its two secrets travelled.

    ``required`` is the difference between the commands. For create, an
    absent token means a public repository, which the endpoint is told by an
    empty string. Update replaces all five fields on every call, so a token
    it was not given is a token it would blank.
    """
    if api_token_stdin and api_token is not None:
        raise click.UsageError("Pass either --api-token or --api-token-stdin, not both")
    if api_token_stdin:
        return _token_from_stdin()
    if api_token is not None:
        return api_token
    if required:
        raise click.UsageError(
            "Missing option '--api-token'. Pass it, or --api-token-stdin to keep the token "
            "out of your shell history; a public repository takes --api-token ''"
        )
    return ""


@a1000.command()
@click.option("--url", "repository_url", required=True, help="Repository URL")
@click.option("--name", required=True, help="Display name")
@click.option(
    "--branch",
    "source_branch",
    default=None,
    help="Source branch; omit to let the appliance use main or master",
)
@_api_token_options(
    "Token for private repos (empty for public); prefer --api-token-stdin, "
    "since an argument is visible in ps and kept in your shell history"
)
@click.option(
    "--mode",
    "import_update_preferences",
    type=click.Choice(["manual", "auto-update", "auto-import"]),
    default="manual",
    help="Update mode (manual=0, auto-update=1, auto-import=2)",
)
@click.pass_context
def yara_repo_create(
    ctx: click.Context,
    api_token: str | None,
    api_token_stdin: bool,
    repository_url: str,
    name: str,
    source_branch: str | None,
    import_update_preferences: str,
) -> None:
    """Create a Yara Online Source repository.

    The token may be piped in or typed at a hidden prompt with
    `--api-token-stdin`, which is the way that keeps it out of `ps` and
    out of your shell history. Omit both for a public repository.
    """
    # Judge the command line before probing the appliance: a token stated
    # twice is wrong whatever the appliance is doing.
    token = _supplied_api_token(api_token, api_token_stdin, required=False)

    service, output, formatter = probed_a1000(ctx, A1000YaraService)
    # Every field is named on both sides of the ``=``. All five are strings,
    # so two transposed — the name into the URL slot, the token into the
    # branch — would pass every check there is and be stored.
    spec = YaraRepositorySpec(
        repository_url=repository_url,
        name=name,
        source_branch=source_branch,
        api_token=token,
        import_update_preferences=import_update_preferences,
    )

    run_step(
        output,
        f"Creating YARA repository '{spec.name}'...",
        lambda: service.create_yara_repository(spec),
        success=f"Created YARA repository: {spec.name}",
        failure="Failed to create YARA repository",
        render=_redacted(formatter),
    )


@a1000.command()
@click.argument("repository_id", type=int)
@click.option("--url", "repository_url", required=True, help="Repository URL (replaced)")
@click.option("--name", required=True, help="Display name (replaced)")
@click.option("--branch", "source_branch", required=True, help="Source branch (replaced)")
@_api_token_options(
    "Token for private repos, empty string for public (replaced); prefer "
    "--api-token-stdin, since an argument is visible in ps and kept in your shell history"
)
@click.option(
    "--mode",
    "import_update_preferences",
    type=click.Choice(["manual", "auto-update", "auto-import"]),
    required=True,
    help="Update mode (replaced)",
)
@click.pass_context
def yara_repo_update(
    ctx: click.Context,
    repository_id: int,
    api_token: str | None,
    api_token_stdin: bool,
    repository_url: str,
    name: str,
    source_branch: str,
    import_update_preferences: str,
) -> None:
    """Replace a Yara Online Source repository, in full.

    The endpoint writes all five fields on every call, so this is a
    replacement and not a patch: every option is required because a
    default would silently overwrite what is configured. Read the current
    values with `yara-repo-list` and pass back the ones you are keeping —
    the token is not listed, so supply it (or "") yourself, either with
    `--api-token` or over stdin with `--api-token-stdin`, which keeps it
    out of `ps` and out of your shell history.
    """
    # The same order as create: the command line is judged first.
    token = _supplied_api_token(api_token, api_token_stdin, required=True)

    service, output, formatter = probed_a1000(ctx, A1000YaraService)
    # The same five fields as create: the two commands differ in what they
    # require of the analyst, not in what they write. ``repository_id`` says
    # which repository is replaced rather than being one of the fields
    # replaced, so it stays out of the spec.
    spec = YaraRepositorySpec(
        repository_url=repository_url,
        name=name,
        source_branch=source_branch,
        api_token=token,
        import_update_preferences=import_update_preferences,
    )

    run_step(
        output,
        f"Updating YARA repository {repository_id}...",
        lambda: service.update_yara_repository(repository_id, spec),
        success=f"Updated YARA repository {repository_id}",
        failure="Failed to update YARA repository",
        render=_redacted(formatter),
    )


@a1000.command()
@click.argument("repository_id", type=int)
@click.option(
    "--remove-rulesets",
    is_flag=True,
    help="Also delete every ruleset associated with the repository",
)
@yes_flag
@click.pass_context
def yara_repo_delete(
    ctx: click.Context, repository_id: int, remove_rulesets: bool, yes: bool
) -> None:
    """Delete a custom Yara Online Source repository."""
    service, output, _ = probed_a1000(ctx, A1000YaraService)

    # Irreversible, like every other destructive command in this group,
    # which all prompt: with ``--remove-rulesets`` this takes every ruleset
    # the repository imported, and those live on the appliance, not in the
    # repository it synced them from.
    also = " and every ruleset it imported" if remove_rulesets else ""
    if not confirmed(output, yes, f"Delete YARA repository {repository_id}{also}?"):
        return

    run_step(
        output,
        f"Deleting YARA repository {repository_id}...",
        lambda: service.delete_yara_repository(repository_id, remove_rulesets=remove_rulesets),
        success=f"Deleted YARA repository {repository_id}",
        failure="Failed to delete YARA repository",
    )
