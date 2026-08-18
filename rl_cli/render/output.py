"""Output formatting utilities using Rich."""

import json
from collections.abc import Callable
from typing import Any

import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from tabulate import tabulate

from rl_cli.models.output_format import OutputFormat
from rl_cli.models.run_status import RunStatus
from rl_cli.render.markup import safe
from rl_cli.render.sarif import _json_safe, to_sarif
from rl_cli.render.toon import encode as toon_encode
from rl_cli.text import sanitize

# Re-exported: this module renders the formats, so every reader of the
# vocabulary already imports it from here. The enum itself lives one
# module down, where the config loader can validate against it without
# importing Rich, tabulate and the SARIF writer to do so.
__all__ = [
    "OutputFormat",
    "OutputFormatter",
    "RichOutput",
]


def _for_terminal(data: Any) -> Any:
    """Strip terminal control sequences from every string in ``data``.

    For the renderers that write their strings out verbatim — tabulate
    and TOON. JSON and YAML escape control characters themselves, and
    what they emit is for a consumer to parse, not for a terminal to
    interpret, so they are left carrying the value they were given.

    Tuples are walked as well as lists, and stay tuples: ``_tabulated``
    hands a flattened record over as ``list(...items())``, a list of
    tuples, so a list-only walk leaves every cell of every single-record
    ``-o table`` unsanitised, ESC and right-to-left override included.
    """
    if isinstance(data, str):
        return sanitize(data)
    if isinstance(data, dict):
        return {_for_terminal(key): _for_terminal(value) for key, value in data.items()}
    if isinstance(data, tuple):
        return tuple(_for_terminal(item) for item in data)
    if isinstance(data, list):
        return [_for_terminal(item) for item in data]
    return data


def _cell(value: Any) -> str:
    """Render one table cell, unpacking lists onto their own lines.

    ``str()`` on a list of hashes yields a quoted Python repr, which is
    noise in a table; one item per line is what the column is for.

    ``safe`` because the cell holds whatever the malware author typed:
    without it an ESC in a file name clears the analyst's screen, a
    threat name containing ``[a-z0-9]`` loses it to Rich's markup
    parser, and one containing ``[/]`` raises MarkupError mid-report.
    """
    if isinstance(value, list):
        return "\n".join(safe(item) for item in value)
    return safe(value)


class RichOutput:
    """Rich output formatter for CLI responses."""

    def __init__(
        self,
        console: Console | None = None,
        status_console: Console | None = None,
        quiet: bool = False,
    ):
        """Initialize Rich output formatter.

        Status lines go to ``status_console`` (stderr by default) so that
        stdout carries nothing but the requested data: otherwise ``-o json``
        emits "✓ Report retrieved" ahead of the document and the result
        cannot be piped into jq. Both streams still land on the terminal in
        an interactive session.

        ``quiet`` drops the progress chatter — successes, notes and the
        spinner. Warnings and errors always survive it: suppressing those
        would hide the reason a command produced nothing.
        """
        self.console = console or Console()
        # A default stderr console inherits the given console's colour
        # policy: `color: false` that reached stdout but not the status
        # stream would honour the setting for only half the output.
        self.status_console = status_console or Console(
            stderr=True, no_color=self.console.no_color or None
        )
        self.quiet = quiet
        # Held, not owned: error() records a failure here and cli/main.py
        # reads it for the exit status. Whether the process failed is not a
        # rendering decision — the reporter carries the flag because it is
        # what every layer already holds, and does not define it.
        self.status = RunStatus()

    def success(self, message: str) -> None:
        """Display success message.

        ``safe`` here, and in the four below, because these five are the
        only place appliance-authored text reaches a console with markup
        on: an HTTP error carries up to 200 bytes of response body into
        ``handle_error`` and from there straight into ``error``, where a
        ``[/]`` raises MarkupError out of the error reporting itself, an
        ESC clears the screen or sets the window title, and ``[a-z0-9]`` is
        lost silently. Nothing reaches these methods pre-escaped — the
        formatters call ``safe`` on values they interpolate into their own
        markup and print the built Panel/Table through ``console``, never
        through here — so there is nothing to double-escape.
        """
        if not self.quiet:
            self.status_console.print(f"[green]✓[/green] {safe(message)}")

    def error(self, message: str) -> None:
        """Report that this run failed, and why.

        Always fails the run, and takes no argument about that: the exit
        status is not something a print call negotiates, or the same red
        line means "the process failed" from one caller and "here is what I
        measured" from the next. The measurement has its own method,
        :meth:`problem`.
        """
        self.status.fail()
        self.status_console.print(f"[red]✗[/red] {safe(message)}")

    def problem(self, message: str) -> None:
        """Report bad news this run measured, without failing the run.

        `config check-access` is the command the CLI points an analyst at
        when a service is unreachable, so "A1000: connection refused" is its
        answer rather than its failure: exiting 1 for printing it is the
        diagnostic reporting itself broken. Drawn exactly like
        :meth:`error`, because to the reader it is the same bad news; only
        the process's fate differs.
        """
        self.status_console.print(f"[red]✗[/red] {safe(message)}")

    def warning(self, message: str) -> None:
        """Display warning message."""
        self.status_console.print(f"[yellow]⚠[/yellow] {safe(message)}")

    def info(self, message: str) -> None:
        """Display info message."""
        if not self.quiet:
            self.status_console.print(f"[blue]ℹ[/blue] {safe(message)}")

    def table(
        self,
        data: list[dict[str, Any]],
        title: str | None = None,
    ) -> None:
        """Display data in a table."""
        if not data:
            self.warning("No data to display")
            return

        table = Table(title=title, show_header=True, header_style="bold magenta")

        # Columns are every row's keys in first-seen order, and each cell is
        # looked up by its own key. Taking the columns from the first row
        # while adding cells positionally puts a row with a different key set
        # under whatever headings line up, and Rich says nothing: it appends
        # a blank-headed column for the overflow.
        keys = list(dict.fromkeys(key for row in data for key in row))
        for key in keys:
            column_title = safe(str(key).replace("_", " ").title())
            table.add_column(column_title)

        # Add rows
        for row in data:
            table.add_row(*[_cell(row.get(key, "")) for key in keys])

        self.console.print(table)

    def json(self, data: Any, indent: int = 2) -> None:
        """Display JSON data with syntax highlighting."""
        json_str = json.dumps(_json_safe(data), indent=indent)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
        self.console.print(syntax)

    def progress_spinner(self, message: str = "Processing...") -> Progress:
        """Create a progress spinner already showing ``message``.

        The task is added here rather than left to the caller: a
        ``Progress`` with no tasks renders an empty Live, so a caller that
        forgets the ``add_task`` shows a blank line for the whole wait — 45s
        of nothing while a PDF report is generated. Callers that want to
        advance it can use ``progress.task_ids[0]``.
        """
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.status_console,
            transient=True,
            disable=self.quiet,
        )
        progress.add_task(message, total=None)
        return progress


def _flattened(data: Any, prefix: str = "") -> dict[str, Any]:
    """A nested record as one level of dotted keys.

    A verdict arrives as ``{"rl": {"malware_presence": {...}}}``, and a
    table of one row whose single column holds that whole structure tells
    the reader nothing. Flattening names each leaf by the path it sat on,
    so ``rl.malware_presence.status`` is a row.
    """
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            flat.update(_flattened(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def _json_cell(value: Any) -> str:
    """A structured cell as compact JSON, with its characters intact.

    ``_raw`` escapes non-ASCII because a raw dump is a machine artefact;
    a grid is read by a person, and this CLI's subjects have non-ASCII
    names.

    Round-tripped through UTF-8 because not escaping is what makes a lone
    surrogate reachable: ``ensure_ascii=True`` renders ``\\ud800`` as text,
    while leaving it intact carries it to the console, where writing it
    raises ``UnicodeEncodeError`` and takes the whole report down. A
    payload holding one is malformed JSON from the appliance, so the
    replacement character is the honest rendering of a character that is
    not one.
    """
    dumped = json.dumps(value, separators=(",", ":"), default=str, ensure_ascii=False)
    return dumped.encode("utf-8", "replace").decode("utf-8")


def _tabulated(data: Any) -> str:
    """``data`` as a grid, whatever shape it arrived in.

    A list of records is the shape ``tabulate`` wants, and most commands
    answer a single record instead: those shapes are gridded here rather
    than falling through to a single-quoted, unparseable ``str(data)``.

    Every shape goes through the shared JSON-safe conversion, so a non-finite score reads the
    same here as under every other format — the record branch skipping it
    printed a literal ``inf`` in the cell where JSON, YAML and TOON all
    write null.
    """
    if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
        return str(tabulate(_for_terminal(_json_safe(data)), headers="keys", tablefmt="grid"))
    if isinstance(data, dict):
        rows = _for_terminal(list(_flattened(_json_safe(data)).items()))
        # A list is a leaf ``_flattened`` does not walk into, and the shape
        # most commands answer with -- ``{"results": [...], "count": 1}`` --
        # puts the whole answer in one such leaf. ``tabulate`` would
        # ``str()`` it into 2000 characters of single-quoted Python repr in a
        # single cell, which is the unparseable rendering this branch exists
        # to avoid; ``_raw`` writes the same value as compact JSON.
        # ``ensure_ascii=False``, unlike ``_raw``: this cell is read on a
        # terminal, and a sample named "документ.exe" is the ordinary case
        # here. Escaping it to \uXXXX hides the name the analyst is
        # identifying. ``_for_terminal`` has already stripped the control
        # characters, including inside dicts nested in lists.
        cells = [
            (key, _json_cell(value) if isinstance(value, dict | list) else value)
            for key, value in rows
        ]
        return str(tabulate(cells, headers=["field", "value"], tablefmt="grid"))
    if isinstance(data, list):
        return str(
            tabulate(
                [[item] for item in _for_terminal(_json_safe(data))],
                headers=["value"],
                tablefmt="grid",
            )
        )
    # Sanitised like every other cell above: a grid is a rendering for a
    # terminal, and the answer that arrives as a bare string — a YARA rule,
    # an appliance message — is as attacker-authored as the rest. Only
    # ``-o raw`` promises the bytes back unaltered.
    return sanitize(data)


def _raw(data: Any) -> str:
    """``data`` with nothing added, for a pipeline to read.

    A string is what it is — this is how ``yara-content -o raw > rule.yar``
    gets a rule and not a quoted one. Anything structured becomes compact
    JSON rather than ``str(dict)``, whose single-quoted repr no parser on
    the other end of the pipe accepts under a format the README lists for
    automation.

    This is the one format that does not sanitize a bare string, and the
    only one that promises the bytes back unaltered — deliberately, since
    a redaction is not what ``> rule.yar`` asked for. An ESC in appliance
    text therefore reaches a terminal under ``-o raw`` and nowhere else;
    every rendering format, including ``-o table``, strips it.
    """
    if isinstance(data, str):
        return data
    if isinstance(data, dict | list):
        return json.dumps(_json_safe(data), separators=(",", ":"))
    return str(data)


def _json_document(data: Any) -> str:
    """``data`` as the indented JSON document ``-o json`` writes."""
    return json.dumps(_json_safe(data), indent=2)


# One renderer per format, so the vocabulary and the renderings are one
# table rather than a ladder with a fallback under it. RICH is the console
# rendering and has no string form of its own; asked for one it answers the
# JSON document, because the fallback it used to reach was ``str(data)`` —
# the single-quoted Python repr ``_raw`` exists to avoid.
#
# Every member of the enum is here, and ``tests/test_output.py`` says so:
# a format added without a renderer must fail loudly rather than fall back
# to a rendering no parser accepts.
_RENDERERS: dict[OutputFormat, Callable[[Any], str]] = {
    OutputFormat.RICH: _json_document,
    OutputFormat.JSON: _json_document,
    OutputFormat.YAML: lambda data: yaml.dump(_json_safe(data), default_flow_style=False),
    OutputFormat.TOON: lambda data: toon_encode(_for_terminal(_json_safe(data))),
    OutputFormat.SARIF: lambda data: json.dumps(to_sarif(data), indent=2),
    OutputFormat.TABLE: _tabulated,
    OutputFormat.RAW: _raw,
}


class OutputFormatter:
    """General output formatter supporting multiple formats."""

    def __init__(self, format: OutputFormat = OutputFormat.RICH, rich: RichOutput | None = None):
        """Initialize output formatter over the invocation's ``RichOutput``.

        Takes the invocation's reporter rather than a bare console, because
        there is one answer per invocation to "is this quiet?" and "did the
        run fail?": a second ``RichOutput`` built here would hold its own,
        and only the CLI's is told about ``--quiet`` or read for the exit
        status.

        ``console`` is this reporter's stdout console, not a parameter of
        its own: the two must be the same object, or ``display`` hands
        renderers a console the rest of the output does not use.
        """
        self.format = format
        self.rich = rich or RichOutput()
        self.console = self.rich.console

    def format_output(self, data: Any, format: OutputFormat | None = None) -> str:
        """``data`` as the string the format writes. See :data:`_RENDERERS`."""
        return _RENDERERS[format or self.format](data)

    def display(self, data: Any, format: OutputFormat | None = None) -> None:
        """Display formatted data."""
        output_format = format or self.format

        if output_format == OutputFormat.RICH:
            if isinstance(data, dict):
                self.rich.json(data)
            elif isinstance(data, list):
                if all(isinstance(item, dict) for item in data):
                    self.rich.table(data)
                else:
                    self.rich.json(data)
            else:
                # markup=False: Rich reads [tags] out of arbitrary text and
                # silently deletes them — a YARA rule printed here loses
                # [a-z0-9] from its regexes with no error at all. It still
                # forwards ESC, so the string goes through sanitize for the
                # same reason a table cell does.
                self.console.print(sanitize(data) if isinstance(data, str) else data, markup=False)
        else:
            formatted = self.format_output(data, output_format)
            # ``raw`` promises the bytes back, which is what makes
            # ``yara-content -o raw > rule.yar`` produce a usable rule. That
            # promise is to a *file*: on a terminal the same bytes are an
            # escape sequence that clears the screen, retitles the window,
            # or hides an extension behind U+202E, so the redirected case
            # keeps its fidelity and the displayed case is made safe. Every
            # other format escapes already; this is a no-op for them.
            if formatted and self.console.is_terminal:
                formatted = sanitize(formatted)
            print(formatted)
