"""Tests for OutputFormatter serialization paths and RichOutput rendering."""

import contextlib
import inspect
import io
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from rich.console import Console

from rl_cli.render.output import OutputFormat, OutputFormatter, RichOutput, _for_terminal


def _consoles() -> tuple[RichOutput, io.StringIO, io.StringIO]:
    """RichOutput with both streams captured: (output, data, status)."""
    data, status = io.StringIO(), io.StringIO()
    output = RichOutput(
        Console(file=data, force_terminal=False, width=120),
        Console(file=status, force_terminal=False, width=120),
    )
    return output, data, status


def _recorded_output() -> tuple[RichOutput, io.StringIO]:
    output, data, _ = _consoles()
    return output, data


class TestRichOutput:
    def test_message_helpers_render_text(self):
        output, _, status = _consoles()
        output.success("done")
        output.error("boom")
        output.warning("careful")
        output.info("fyi")
        text = status.getvalue()
        assert all(word in text for word in ("done", "boom", "careful", "fyi"))

    def test_status_messages_stay_off_the_data_stream(self):
        """`-o json` is unpipeable if status lines share stdout with the data."""
        output, data, status = _consoles()
        output.success("Report retrieved")
        output.info("Fetching...")
        output.json({"sha256": "abc"})

        assert data.getvalue().strip().startswith("{")
        assert "Report retrieved" not in data.getvalue()
        assert "Report retrieved" in status.getvalue()

    def test_progress_spinner_renders_its_message(self):
        """A task-less Progress draws an empty Live: 45s of blank line."""

        def captured(**kwargs) -> str:
            buffer = io.StringIO()
            output = RichOutput(
                status_console=Console(file=buffer, force_terminal=True, width=80), **kwargs
            )
            with output.progress_spinner("Fetching report..."):
                pass
            return buffer.getvalue()

        assert "Fetching report..." in captured()
        assert captured(quiet=True) == ""

    def test_table_renders_rows_and_headers(self):
        output, buffer = _recorded_output()
        output.table([{"sha256": "abc", "status": "clean"}], title="Samples")
        text = buffer.getvalue()
        assert "Samples" in text and "abc" in text and "Sha256" in text

    def test_table_with_no_data_warns(self):
        output, _, status = _consoles()
        output.table([])
        assert "No data" in status.getvalue()


class TestFormatOutput:
    def test_json_serializes_non_primitive_values(self):
        formatter = OutputFormatter(OutputFormat.JSON)
        result = formatter.format_output({"path": Path("/tmp/x")})
        assert json.loads(result) == {"path": "/tmp/x"}

    def test_table_formats_list_of_dicts(self):
        formatter = OutputFormatter(OutputFormat.TABLE)
        result = formatter.format_output([{"a": 1, "b": 2}])
        assert "a" in result and "1" in result

    def test_raw_gives_a_pipeline_something_it_can_parse(self):
        """This used to assert ``"{'a': 1}"`` — a Python repr, pinning the bug.

        ``raw`` is listed under Automation, and single-quoted repr output
        is not JSON: nothing on the far end of the pipe accepts it.
        """
        formatter = OutputFormatter(OutputFormat.RAW)
        assert json.loads(formatter.format_output({"a": 1})) == {"a": 1}

    def test_raw_passes_a_string_through_untouched(self):
        """``yara-content -o raw > rule.yar`` has to get the rule, not a quoted one."""
        formatter = OutputFormatter(OutputFormat.RAW)
        assert formatter.format_output("rule x { condition: true }") == "rule x { condition: true }"

    def test_raw_renders_a_bare_scalar_as_its_text(self):
        """A number is neither a string to pass through nor a JSON container."""
        formatter = OutputFormatter(OutputFormat.RAW)
        assert formatter.format_output(42) == "42"

    def test_table_renders_a_single_record_as_field_and_value(self):
        """A dict fell through to ``str(data)``, so most commands printed a repr."""
        formatter = OutputFormatter(OutputFormat.TABLE)
        rendered = formatter.format_output({"rl": {"malware_presence": {"status": "MALICIOUS"}}})
        assert "rl.malware_presence.status" in rendered
        assert "MALICIOUS" in rendered
        assert "{'rl'" not in rendered

    def test_table_still_renders_a_list_of_records_as_columns(self):
        formatter = OutputFormatter(OutputFormat.TABLE)
        rendered = formatter.format_output([{"sha1": "a", "status": "clean"}])
        assert "sha1" in rendered and "status" in rendered

    def test_toon_serializes_tabular_data(self):
        formatter = OutputFormatter(OutputFormat.TOON)
        result = formatter.format_output(
            {"items": [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 1}]}
        )
        assert result == "items[2]{sku,qty}:\n  A1,2\n  B2,1"

    def test_toon_serializes_non_primitive_values(self):
        formatter = OutputFormatter(OutputFormat.TOON)
        assert formatter.format_output({"path": Path("/tmp/x")}) == "path: /tmp/x"

    def test_sarif_wraps_payload_in_valid_log(self):
        formatter = OutputFormatter(OutputFormat.SARIF)
        result = json.loads(formatter.format_output({"classification": "malicious"}))
        assert result["version"] == "2.1.0"
        assert result["runs"][0]["results"][0]["level"] == "error"


class TestQuietMode:
    """-q claims to suppress non-essential output; it has to actually do it."""

    def _quiet(self):
        data, status = io.StringIO(), io.StringIO()
        output = RichOutput(
            Console(file=data, force_terminal=False, width=120),
            Console(file=status, force_terminal=False, width=120),
            quiet=True,
        )
        return output, data, status

    def test_success_and_info_are_suppressed(self):
        output, _, status = self._quiet()
        output.success("Report retrieved")
        output.info("Fetching...")
        assert status.getvalue() == ""

    def test_warnings_and_errors_still_get_through(self):
        output, _, status = self._quiet()
        output.warning("Analysis timed out")
        output.error("Invalid hash format")
        text = status.getvalue()
        assert "Analysis timed out" in text
        assert "Invalid hash format" in text

    def test_data_is_unaffected(self):
        output, data, _ = self._quiet()
        output.json({"sha256": "abc"})
        assert "abc" in data.getvalue()


class TestBracketsSurviveTheConsole:
    """Rich reads [tags] out of any string and silently deletes them."""

    def test_a_yara_regex_keeps_its_character_class(self):
        console = Console(width=100, force_terminal=False)
        formatter = OutputFormatter(OutputFormat.RICH, RichOutput(console))
        with console.capture() as capture:
            formatter.display(r"$re = /https?:\/\/[a-z0-9]+\.com/")
        assert "[a-z0-9]" in capture.get()

    def test_a_list_that_is_not_all_records_is_shown_as_json(self):
        """A table needs uniform records; a mixed list falls back to JSON."""
        console = Console(width=100, force_terminal=False)
        formatter = OutputFormatter(OutputFormat.RICH, RichOutput(console))
        with console.capture() as capture:
            formatter.display([1, {"a": 2}, "three"])
        rendered = capture.get()
        assert "three" in rendered
        assert '"a"' in rendered

    def test_a_table_cell_keeps_its_character_class(self):
        """A threat name is as attacker-authored as a YARA rule."""
        output, buffer = _recorded_output()
        output.table([{"threat": "W32.[a-z0-9]x"}])
        assert "W32.[a-z0-9]x" in buffer.getvalue()

    def test_unbalanced_markup_in_a_cell_does_not_abort_the_report(self):
        """`[/]` raised MarkupError, which the CLI reports as "Unexpected error"."""
        output, buffer = _recorded_output()
        output.table([{"name": "x[/]y"}])
        assert "x[/]y" in buffer.getvalue()


class TestTerminalEscapesNeverReachTheTerminal:
    """A file name or threat name is written by whoever built the malware."""

    HOSTILE = "\x1b[2Jwiped.exe"

    def test_rich_table_strips_them(self):
        output, buffer = _recorded_output()
        output.table([{"file_name": self.HOSTILE}])
        text = buffer.getvalue()
        assert "\x1b" not in text
        assert "wiped.exe" in text

    def test_rich_table_strips_them_from_list_cells_and_headers(self):
        output, buffer = _recorded_output()
        output.table([{"\x1b]0;title\x07": ["\x1b[2Jone", "two"]}])
        text = buffer.getvalue()
        assert "\x1b" not in text
        assert "one" in text and "two" in text

    def test_tabulate_strips_them(self):
        formatter = OutputFormatter(OutputFormat.TABLE)
        result = formatter.format_output([{"threat_name": self.HOSTILE}])
        assert "\x1b" not in result
        assert "wiped.exe" in result

    def test_toon_strips_them(self):
        formatter = OutputFormatter(OutputFormat.TOON)
        result = formatter.format_output({"threat_name": self.HOSTILE})
        assert "\x1b" not in result
        assert "wiped.exe" in result

    def test_the_rich_fallback_for_a_bare_string_strips_them(self):
        console = Console(width=100, force_terminal=False)
        formatter = OutputFormatter(OutputFormat.RICH, RichOutput(console))
        with console.capture() as capture:
            formatter.display(self.HOSTILE)
        assert "\x1b" not in capture.get()

    def test_json_escapes_rather_than_strips_them(self):
        """The machine-readable dump owes its consumer the real name."""
        formatter = OutputFormatter(OutputFormat.JSON)
        result = formatter.format_output({"file_name": self.HOSTILE})
        assert "\x1b" not in result
        assert json.loads(result) == {"file_name": self.HOSTILE}


class TestEveryTableCellIsSanitised:
    """``-o table`` flattens a single record with ``dict.items()``.

    That is a list of *tuples*, and ``_for_terminal`` recursed through
    ``str``, ``dict`` and ``list`` only — ``isinstance(x, list)`` is False
    for a tuple — so every cell of every single-record table went out
    carrying whatever the malware author wrote. Most commands answer a
    single record.
    """

    # ESC, DEL, and the right-to-left override that makes a dropper's name
    # end in ".jpg" in the one place an analyst reads it.
    HOSTILE = "Win32.Evil\x1b[2J\x7f‮gpj.exe"
    UNSAFE = ("\x1b", "\x7f", "‮")

    def _rendered(self, data) -> str:
        return OutputFormatter(OutputFormat.TABLE).format_output(data)

    def test_a_flattened_single_record_carries_none_of_them(self):
        rendered = self._rendered({"rl": {"malware_presence": {"threat_name": self.HOSTILE}}})
        assert not [char for char in self.UNSAFE if char in rendered]
        assert "Win32.Evil" in rendered

    def test_a_hostile_key_is_stripped_too(self):
        rendered = self._rendered({"\x1b]0;pwned\x07": "value"})
        assert not [char for char in self.UNSAFE if char in rendered]
        assert "value" in rendered

    def test_a_list_of_records_stays_covered(self):
        rendered = self._rendered([{"threat_name": self.HOSTILE}])
        assert not [char for char in self.UNSAFE if char in rendered]
        assert "Win32.Evil" in rendered

    def test_a_bare_string_answer_is_sanitised_as_well(self):
        """``yara-content -o table`` answers with the rule itself."""
        rendered = self._rendered(self.HOSTILE)
        assert not [char for char in self.UNSAFE if char in rendered]
        assert "Win32.Evil" in rendered

    def test_for_terminal_walks_a_tuple_and_keeps_it_one(self):
        """The actual gap: tabulate is handed rows as tuples, not lists."""
        walked = _for_terminal([("threat", self.HOSTILE)])
        assert walked == [("threat", "Win32.Evil[2Jgpj.exe")]
        assert isinstance(walked[0], tuple)


class TestALoneSurrogateNeverReachesTheConsole:
    """``"\\ud800"`` is valid JSON, so ``.json()`` hands one back as a str.

    A lone surrogate is not encodable UTF-8: writing it out took the whole
    render down with ``UnicodeEncodeError`` under the two formats that emit
    their strings verbatim — ``-o table`` and ``-o toon`` — while the
    escaping formats survived. ``sanitize`` strips it now, so every format
    stays writable.
    """

    @pytest.mark.parametrize("fmt", list(OutputFormat))
    def test_every_format_stays_utf8_writable(self, fmt):
        payload = [{"sha256": "a" * 64, "threat_name": "Win32.\ud800.Evil"}]
        rendered = OutputFormatter(fmt).format_output(payload)
        # The print() at the sink encodes to a UTF-8 stdout; a lone
        # surrogate raises there, so this is the crash the fix prevents.
        rendered.encode("utf-8")

    @pytest.mark.parametrize("fmt", [OutputFormat.TABLE, OutputFormat.TOON])
    def test_the_verbatim_formats_drop_the_surrogate(self, fmt):
        rendered = OutputFormatter(fmt).format_output({"name": "keep\ud800me"})
        rendered.encode("utf-8")
        assert "\ud800" not in rendered
        assert "keepme" in rendered


class TestAStructuredCellIsJsonNotAPythonRepr:
    """``_flattened`` walks dicts, so a list is a leaf ``tabulate`` ``str()``s.

    The commonest answer shape in this CLI — ``{"results": [...], "count": 1}``
    — therefore came out as one cell holding 2000 characters of
    single-quoted Python repr, under a format the README lists for
    automation, from the branch whose docstring says these shapes are
    gridded "rather than falling through to a single-quoted, unparseable
    ``str(data)``".
    """

    ANSWER: ClassVar[dict[str, Any]] = {
        "results": [{"sha1": "a" * 40, "av_scanners": {"scanner_count": 3}, "tags": ["x", "y"]}],
        "count": 1,
    }

    def _cell(self, data, field: str) -> str:
        rendered = OutputFormatter(OutputFormat.TABLE).format_output(data)
        row = next(line for line in rendered.splitlines() if line.startswith(f"| {field} "))
        return row.strip("|").split("|")[1].strip()

    def test_the_results_cell_parses(self):
        assert json.loads(self._cell(self.ANSWER, "results")) == self.ANSWER["results"]

    def test_no_single_quoted_repr_reaches_the_grid(self):
        rendered = OutputFormatter(OutputFormat.TABLE).format_output(self.ANSWER)
        assert "'sha1'" not in rendered

    def test_a_scalar_cell_is_unchanged(self):
        """A number stays a number, and a null stays the empty cell it was."""
        assert self._cell(self.ANSWER, "count") == "1"
        assert self._cell({"threat_name": None}, "threat_name") == ""

    def test_a_hostile_string_inside_a_list_is_still_stripped(self):
        rendered = OutputFormatter(OutputFormat.TABLE).format_output(
            {"results": [{"threat_name": "Win32.Evil\x1b[2J\x7f‮gpj.exe"}]}
        )
        assert not [char for char in ("\x1b", "\x7f", "‮") if char in rendered]
        assert "Win32.Evil" in rendered


class TestNonFiniteNumbersStayParseable:
    """`requests`' .json() accepts NaN/Infinity, so a payload can carry them."""

    def test_json_emits_null(self):
        formatter = OutputFormatter(OutputFormat.JSON)
        result = formatter.format_output({"a": float("nan"), "b": float("inf")})
        assert "NaN" not in result and "Infinity" not in result
        assert json.loads(result) == {"a": None, "b": None}

    def test_sarif_emits_null(self):
        """A .sarif with a bare NaN is rejected by GitHub Code Scanning and jq."""
        formatter = OutputFormatter(OutputFormat.SARIF)
        result = formatter.format_output({"classification": "malicious", "score": float("nan")})
        assert "NaN" not in result
        assert json.loads(result)["runs"][0]["results"][0]["properties"]["score"] is None

    def test_rich_json_emits_null(self):
        output, buffer = _recorded_output()
        output.json({"a": float("inf")})
        assert "Infinity" not in buffer.getvalue()

    def test_a_table_of_records_writes_no_literal_inf(self):
        """The list-of-records branch was the one grid skipping JSON-safe conversion.

        A riskscore of Infinity printed as a bare ``inf`` in the cell where
        the single-record grid beside it — and JSON, YAML and TOON — all
        write null.
        """
        formatter = OutputFormatter(OutputFormat.TABLE)
        rendered = formatter.format_output([{"riskscore": float("inf"), "name": "stage2.dll"}])

        assert "inf" not in rendered
        assert "stage2.dll" in rendered

    def test_a_table_of_bare_values_writes_no_literal_inf(self):
        formatter = OutputFormatter(OutputFormat.TABLE)
        assert "inf" not in formatter.format_output([float("inf")])

    def test_a_single_record_table_was_already_covered(self):
        formatter = OutputFormatter(OutputFormat.TABLE)
        assert "inf" not in formatter.format_output({"riskscore": float("inf")})


class TestApplianceAuthoredMessages:
    """The four status helpers carry text the appliance wrote.

    An HTTP failure reaches ``RichOutput.error`` through
    ``BaseService.handle_error`` with up to 200 bytes of response body in
    it, and these were the only Rich renderers left printing a value with
    markup enabled and no ``safe()``.
    """

    def test_a_close_tag_does_not_abort_the_command(self):
        """`[/]` raised MarkupError out of the error reporting itself."""
        output, _, status = _consoles()
        output.error("HTTP 502: [/]")
        assert "[/]" in status.getvalue()

    def test_escape_sequences_do_not_reach_the_terminal(self):
        output, _, status = _consoles()
        output.warning("probe failed: \x1b[2Jwiped\x1b]0;HIJACKED\x07")
        text = status.getvalue()
        assert "\x1b" not in text
        assert "wiped" in text and "HIJACKED" in text

    def test_bracketed_text_is_not_silently_deleted(self):
        """A YARA-derived name loses `[a-z0-9]` to Rich's markup parser."""
        output, _, status = _consoles()
        output.info("rule matched [a-z0-9]")
        assert "[a-z0-9]" in status.getvalue()

    def test_a_well_formed_message_is_unchanged(self):
        output, _, status = _consoles()
        output.success("Report retrieved")
        assert "✓ Report retrieved" in status.getvalue()


class TestRawIsFaithfulToAFileAndSafeOnATerminal:
    """``raw`` is the one format that does not strip what malware wrote.

    That promise is to a file — it is what makes ``yara-content -o raw >
    rule.yar`` produce a rule that still matches. Printed to a terminal
    the same bytes are an escape sequence, so the destination decides.
    """

    RULE = 'rule x { strings: $a = "\x1b[2J\x7f‮" }'

    def _displayed(self, *, is_terminal: bool) -> str:
        console = Console(file=io.StringIO(), force_terminal=is_terminal)
        formatter = OutputFormatter(OutputFormat.RAW, RichOutput(console))
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            formatter.display(self.RULE)
        return captured.getvalue()

    def test_a_redirected_rule_keeps_every_byte(self):
        assert self.RULE in self._displayed(is_terminal=False)

    def test_a_displayed_rule_cannot_drive_the_terminal(self):
        shown = self._displayed(is_terminal=True)
        assert "\x1b" not in shown and "\x7f" not in shown and "‮" not in shown

    def test_a_displayed_rule_is_still_the_rule(self):
        assert "rule x" in self._displayed(is_terminal=True)


class TestWhetherTheRunFailedIsNotTheConsolesToOwn:
    """The exit status was a mutable ``had_error`` attribute on the printer.

    ``cli/main.py`` reads it to decide ``ctx.exit(1)``, so the class that
    draws spinners and tables also decided whether ``rl-cli ... || alert``
    alerted — and anything holding the reporter could assign the flag back
    to ``False``. It is a :class:`RunStatus` now: the reporter holds one
    and reports into it, and setting it takes reporting an error.
    """

    def test_an_error_marks_the_run_as_failed(self):
        output, _, _ = _consoles()
        output.error("boom")
        assert output.status.failed

    def test_a_measurement_is_not_a_failure(self):
        """``config check-access`` prints red for what it measured, and exits 0."""
        output, _, _ = _consoles()
        output.problem("A1000: unreachable")
        assert not output.status.failed

    def test_a_measurement_still_reaches_the_reader(self):
        """Same red line as an error: only the process's fate differs."""
        output, _, status = _consoles()
        output.problem("A1000: unreachable")
        assert "A1000: unreachable" in status.getvalue()

    def test_a_measurement_survives_quiet(self):
        """``--quiet`` promises that bad news survives it."""
        output, _, status = _consoles()
        output.quiet = True
        output.problem("A1000: unreachable")
        assert "A1000: unreachable" in status.getvalue()

    def test_a_failed_run_stays_failed(self):
        output, _, _ = _consoles()
        output.error("boom")
        output.success("but this bit worked")
        output.problem("a measurement")
        assert output.status.failed

    def test_the_exit_status_is_not_negotiable_through_a_print_call(self):
        """``error(..., fail=False)`` is gone, and so is assigning the flag.

        Both were ways to print a line and pick the run's fate with it —
        the first per call site, the second from anything holding the
        reporter.
        """
        output, _, _ = _consoles()
        assert list(inspect.signature(RichOutput.error).parameters) == ["self", "message"]

        output.error("boom")

        assert output.status.failed is True
        # Deliberately widened: the read-only property is what is being
        # asserted, so mypy must not report the assignment it refuses.
        status: Any = output.status
        with pytest.raises(AttributeError):
            status.failed = False


class TestAStructuredCellSurvivesWhatTheApplianceMaySend:
    """The grid stopped escaping non-ASCII so a filename reads as itself.

    That is what makes a lone surrogate reachable: escaped it was text,
    intact it reaches the console and writing it raises
    ``UnicodeEncodeError`` mid-report.
    """

    def test_a_non_ascii_name_reads_as_itself(self):
        rendered = OutputFormatter(OutputFormat.TABLE).format_output(
            {"results": [{"fn": "\u0434\u043e\u043a.exe"}]}
        )

        assert "\u0434\u043e\u043a.exe" in rendered
        assert "\\u0434" not in rendered

    def test_a_lone_surrogate_does_not_take_the_report_down(self):
        rendered = OutputFormatter(OutputFormat.TABLE).format_output(
            {"results": [{"fn": "\ud800bad.exe"}]}
        )

        assert "bad.exe" in rendered
        assert rendered.encode("utf-8"), "the cell must be writable to a console"


class TestNoFormatFallsBackToAPythonRepr:
    """``format_output`` is public, and its ``else`` returned ``str(data)``.

    ``display`` never routes RICH here, but nothing stops a caller, and
    ``OutputFormatter(OutputFormat.RICH).format_output({...})`` answered the
    single-quoted repr that ``_raw``'s own docstring says no parser on the
    other end of a pipe accepts. RICH is the console rendering; asked for a
    string, it answers the JSON document.
    """

    @pytest.mark.parametrize("output_format", list(OutputFormat), ids=lambda fmt: fmt.value)
    def test_no_format_answers_a_python_repr(self, output_format: OutputFormat):
        rendered = OutputFormatter(output_format).format_output({"threat": "Win32.Conti"})

        assert "{'threat': 'Win32.Conti'}" not in rendered

    def test_rich_answers_the_json_document(self):
        rendered = OutputFormatter(OutputFormat.RICH).format_output({"threat": "Win32.Conti"})

        assert json.loads(rendered) == {"threat": "Win32.Conti"}
