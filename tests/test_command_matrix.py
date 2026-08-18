"""The sweeps over the whole 52-command matrix, in one file of their own.

Every command must render its --help, defer to ``-o json``, and open one
connection per invocation. Each claim is parametrized over every command
in the group, so together they are the bulk of the CLI suite's tests —
which is why they are not in with the per-command behaviour tests: those
were buried under three hundred parametrized cases about something else.

The last two sweeps here are over the source of the command modules
rather than over the commands: the inputs both groups take must keep
having one definition between them, and the edge from a command to the
service method it calls must keep being one a type checker can follow.
"""

import ast
import inspect
import json
import re
import sys
import textwrap
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import ClassVar

import click
import pytest
from click.testing import CliRunner

from rl_cli.cli.commands._shared_inputs import confirmed, yes_flag
from rl_cli.cli.commands.a1000 import a1000
from rl_cli.cli.commands.ticloud import ticloud
from rl_cli.cli.main import cli
from rl_cli.render.output import OutputFormat, OutputFormatter, RichOutput
from rl_cli.services.a1000 import A1000ReportService, A1000SampleService
from rl_cli.services.titanium_cloud import TitaniumCloudNetworkService, TitaniumCloudService
from tests.cli_support import (
    A1000_SERVICES,
    ASKING_CALLS,
    SHA1,
    SHA256,
    command_module_imports,
    command_modules,
    commands,
    commands_offering_yes,
    commands_that_ask,
    invoke,
    make_context,
    module_tree,
    reached_by,
    reached_in,
    stub_service,
)


@pytest.fixture
def ctx_obj(tmp_path):
    return make_context(tmp_path)


class TestCommandRegistration:
    """Every registered command must parse its options and render --help."""

    @pytest.mark.parametrize(
        "group,name",
        [(g, n) for g in (a1000, ticloud) for n in sorted(g.commands)],
        ids=lambda v: v if isinstance(v, str) else v.name,
    )
    def test_help_renders(self, group, name, ctx_obj):
        result = invoke(group, [name, "--help"], ctx_obj)
        assert result.exit_code == 0
        assert "Usage" in result.output


class StubPayload(dict):
    """A payload that answers any key, so a command reading a field survives."""

    def __missing__(self, key):
        return "stub"


# Commands whose result is a status rather than data: they print nothing to
# stdout by design. Named here so a command is exempted on purpose and not
# by being left off a hand-written list.
NO_PAYLOAD = {
    "add-tags",
    "batch-delete",
    "delete",
    "delete-classification",
    "download",
    "dynamic-report",
    "remove-tags",
    "set-classification",
    "test",
    "yara-cloud-retro",
    "yara-create",
    "yara-delete",
    "yara-local-retro",
    "yara-publish",
    "yara-repo-delete",
    "yara-toggle",
    "yara-update-interval",
    "yara-update-now",
}

# A TitaniumCore document as /api/v2/samples/{hash}/ticore/ answers it: the
# sections the SDK asks for by name, verdict as the numeric code.
TICORE_REPORT = {
    "sha256": "e" * 64,
    "info": {"file": {"file_type": "PE+", "size": 4096}},
    "classification": {"classification": 3, "factor": 5, "result": "Win32.Trojan.Emotet"},
    "indicators": [{"description": "Contains a suspicious section name"}],
}

# The payload is a dict everywhere except where the command writes it out.
PAYLOADS = {"dynamic-report": b"%PDF-1.4"}

# Commands that need more than their required parameters to reach a service.
EXTRA_ARGS = {
    "batch-delete": ["-h", SHA256],
    "batch-reanalyze": ["-h", SHA256],
    "yara-publish": ["ruleset"],
    "yara-update-interval": ["3600"],
    # `reputation` takes a variadic HASH..., which stub_args skips because
    # it is not required — and it is not required so that `-f hashes.txt`
    # alone works, which is how the README documents batch triage.
    "reputation": [SHA256],
    # The token this command must be given is not a click-required option,
    # so stub_args skips it: it has a second source (`--api-token-stdin`,
    # which keeps a live PAT out of `ps` and the shell history), and click
    # cannot demand a flag that another flag replaces. The demand is made
    # in the command body instead, which is where a run without either
    # source stops — before any service is reached.
    "yara-repo-update": ["--api-token", "stub"],
}


def stub_value(param, tmp_path):
    """Something the parameter accepts, so the command reaches its service."""
    if isinstance(param.type, click.Choice):
        return param.type.choices[0]
    if isinstance(param.type, click.Path):
        target = tmp_path / param.name
        target.write_text("rule stub {condition: false}")
        return str(target)
    if param.type is click.INT:
        return "1"
    for fragment, value in (
        ("url", "https://example.com"),
        ("ip", "1.2.3.4"),
        ("hash", SHA256),
        ("domain", "example.com"),
    ):
        if fragment in param.name:
            return value
    return "stub"


def stub_args(command, tmp_path):
    args = []
    for param in command.params:
        if not param.required:
            continue
        value = stub_value(param, tmp_path)
        args += [value] if isinstance(param, click.Argument) else [param.opts[0], value]
    return args


class TestGlobalOutputFormatIsHonoured:
    """A rich panel on stdout defeats -o json; every command must defer to it.

    Hand-listing a handful of commands here let three of them go on
    printing panels under -o json with the suite green, so the whole group
    is parametrized instead.
    """

    @pytest.mark.parametrize(
        "group,service_classes,name",
        [
            (group, service_classes, name)
            for group, service_classes in (
                (a1000, A1000_SERVICES),
                (ticloud, (TitaniumCloudService, TitaniumCloudNetworkService)),
            )
            for name in sorted(group.commands)
        ],
        ids=lambda value: value if isinstance(value, str) else getattr(value, "name", None),
    )
    def test_stdout_carries_json_and_nothing_else(
        self, group, service_classes, name, ctx_obj, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        # click.confirm prompts on stdout, which would drown out what the
        # command itself rendered there.
        monkeypatch.setattr(click, "confirm", lambda *args, **kwargs: True)
        payload = PAYLOADS.get(name, StubPayload(stub="payload"))
        for service_cls in service_classes:
            stub_service(monkeypatch, service_cls, payload)
        # The one command whose work is a function over two services rather
        # than a call on one, so the walk above cannot reach it.
        monkeypatch.setattr(
            "rl_cli.cli.commands.a1000.reports.upload_and_get_report",
            lambda *args, **kwargs: payload,
        )
        ctx_obj = replace(ctx_obj, formatter=OutputFormatter(OutputFormat.JSON))

        args = [name, *stub_args(group.commands[name], tmp_path), *EXTRA_ARGS.get(name, [])]
        result = invoke(group, args, ctx_obj)

        if name in NO_PAYLOAD:
            assert result.stdout == "", f"{name} is listed as having no payload but rendered one"
            return
        assert result.exit_code == 0, result.output
        # Reported rather than asserted, like the `--json` sweep below:
        # `json.loads` raises before the message on the `assert` can be
        # read, so a command still printing a panel under -o json showed up
        # as a bare JSONDecodeError naming neither the command nor what it
        # put on stdout.
        try:
            dumped = json.loads(result.stdout)
        except json.JSONDecodeError:
            pytest.fail(f"{name} did not render the payload as JSON: {result.stdout!r}")
        assert dumped == payload, f"{name} rendered something other than the payload"

    def test_every_exempt_command_exists(self):
        assert NO_PAYLOAD.issubset(a1000.commands)

    def _stub_titanium(self, monkeypatch):
        monkeypatch.setattr(
            A1000ReportService, "get_titanium_core_report_v2", lambda self, h: TICORE_REPORT
        )

    def test_titanium_report_renders_the_document_on_a_terminal(self, ctx_obj, monkeypatch):
        self._stub_titanium(monkeypatch)
        ctx_obj = replace(ctx_obj, formatter=OutputFormatter(OutputFormat.RICH))
        result = invoke(a1000, ["titanium-report", SHA256], ctx_obj)
        assert result.exit_code == 0, result.output
        assert "Win32.Trojan.Emotet" in result.output
        assert "Contains a suspicious section name" in result.output
        # Panels, not the structure the --json flag is for.
        assert '"classification"' not in result.stdout

    def test_titanium_report_json_flag_dumps_on_a_terminal(self, ctx_obj, monkeypatch):
        self._stub_titanium(monkeypatch)
        ctx_obj = replace(ctx_obj, formatter=OutputFormatter(OutputFormat.RICH))
        result = invoke(a1000, ["titanium-report", SHA256, "--json"], ctx_obj)
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == TICORE_REPORT

    def test_titanium_report_output_json_dumps(self, ctx_obj, monkeypatch):
        self._stub_titanium(monkeypatch)
        ctx_obj = replace(ctx_obj, formatter=OutputFormatter(OutputFormat.JSON))
        result = invoke(a1000, ["titanium-report", SHA256], ctx_obj)
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == TICORE_REPORT

    def test_summary_report_uses_the_panel_on_a_terminal(self, ctx_obj, monkeypatch):
        panelled = []
        monkeypatch.setattr(
            A1000ReportService,
            "get_summary_report_v2",
            lambda self, h: {"classification": "malicious"},
        )

        def capture(data, *, console):
            """Signature-compatible with the renderer: it is handed a console."""
            panelled.append(data)

        monkeypatch.setattr("rl_cli.cli.commands.a1000.reports.print_summary_panel", capture)
        ctx_obj = replace(ctx_obj, formatter=OutputFormatter(OutputFormat.RICH))
        result = invoke(a1000, ["summary-report", SHA256], ctx_obj)
        assert result.exit_code == 0, result.output
        assert panelled == [{"classification": "malicious"}]


def _json_flag_commands():
    """Every registered command offering ``--json``, however it was declared."""
    return [
        (group, service_classes, name)
        for group, service_classes in (
            (a1000, A1000_SERVICES),
            (ticloud, (TitaniumCloudService, TitaniumCloudNetworkService)),
        )
        for name in sorted(group.commands)
        if any(param.name == "show_json" for param in group.commands[name].params)
    ]


class TestTheJsonFlagDumpsWhereverItIsOffered:
    """``--json`` promises the fetched document, which only holds for a document.

    ``OutputFormatter.display`` under the default rich format dumps a
    mapping as JSON — but renders a list of records as *another rich table*
    and a bare string as plain text. So the flag belongs to the commands
    that fetch one document and summarise it, and the capped listings point
    at the global flag instead, which is what ``truncation_notice`` tells
    the analyst.

    Which shape a command answers with is a property of the service method,
    and a sweep that stubs every service call has replaced exactly that. So
    the rule is checked where the command decides — a command that draws
    through ``capped_listing`` is a listing, whatever its stub returns —
    and the invocation below is left to assert the narrower claim it can
    actually see. Asserted through the payload alone, this passed with
    ``--json`` wired onto ``a1000 list``: the stub answers a mapping, and
    ``display`` dumps a mapping whichever command it came from.
    """

    @pytest.mark.parametrize(
        "group,service_classes,name",
        _json_flag_commands(),
        ids=lambda value: value if isinstance(value, str) else getattr(value, "name", None),
    )
    def test_stdout_carries_the_document_and_not_a_rendering(
        self, group, service_classes, name, ctx_obj, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        payload = StubPayload(stub="payload")
        for service_cls in service_classes:
            stub_service(monkeypatch, service_cls, payload)
        ctx_obj = replace(ctx_obj, formatter=OutputFormatter(OutputFormat.RICH))

        args = [name, *stub_args(group.commands[name], tmp_path), "--json"]
        result = invoke(group, args, ctx_obj)

        assert result.exit_code == 0, result.output
        # Reported rather than asserted: `json.loads` raises before the
        # message on the `assert` can be read, so a rendering shows up as a
        # bare JSONDecodeError with the table nowhere in the output.
        try:
            dumped = json.loads(result.stdout)
        except json.JSONDecodeError:
            pytest.fail(f"{name} rendered rather than dumped: {result.stdout!r}")
        assert dumped == payload

    def test_no_command_offering_json_draws_a_capped_listing(self):
        """The rule itself, read off the command bodies.

        ``capped_listing`` is what a listing command draws through, so it is
        the marker that separates the two halves of the split -- and unlike
        the invocation above it does not depend on what a stub answered.
        """
        offenders = sorted(
            name
            for group, _, name in _json_flag_commands()
            if "capped_listing" in inspect.getsource(group.commands[name].callback)
        )
        assert not offenders, (
            f"--json on a capped listing, which renders rather than dumps: {offenders}"
        )

    def test_a_capped_listing_is_still_how_a_listing_is_drawn(self):
        """The sentinel: the marker above has to mark something.

        Rename ``capped_listing`` and the sweep would find no offender for
        the same reason it finds none today.
        """
        drawn = [
            name
            for name, command in a1000.commands.items()
            if command.callback and "capped_listing" in inspect.getsource(command.callback)
        ]
        assert drawn, "no command draws through capped_listing; the marker names nothing"

    def test_the_flag_is_offered_at_all(self):
        """A sweep over an empty list is a sweep that proves nothing."""
        assert _json_flag_commands()


class TestEveryCommandAuthenticatesOnce:
    """One invocation, one connection — probe included — and none for ``--help``.

    Counted at ``A1000Session._open``, the call that constructs the SDK
    client and, under username/password auth, POSTs for a token. Run
    through the root group with the real availability probe: stubbing the
    probe out is exactly what let this claim pass while a cold invocation
    opened two connections, one for the probe and one for the command.
    """

    # Answers out of the settings on purpose: reading back how we are
    # configured to reach the appliance asks the appliance nothing, so
    # once the probe is not paying for a connection there is none left.
    NO_CONNECTION: ClassVar[set[str]] = {"config-dump"}

    def _args(self, name, tmp_path, *command_args):
        """A configured A1000, so the probe has credentials to probe with."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "default:\n  a1000:\n    host: https://a1000.invalid\n    token: real-token\n"
        )
        return ["--config", str(config_file), "a1000", name, *command_args]

    def _run(self, name, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(click, "confirm", lambda *args, **kwargs: True)
        command_args = [*stub_args(a1000.commands[name], tmp_path), *EXTRA_ARGS.get(name, [])]
        return CliRunner().invoke(cli, self._args(name, tmp_path, *command_args))

    @pytest.mark.parametrize("name", sorted(a1000.commands))
    def test_one_connection_per_invocation(
        self, name, real_probe, a1000_connections, monkeypatch, tmp_path
    ):
        """Cold cache, so the probe really runs and shares the one connection."""
        self._run(name, tmp_path, monkeypatch)

        assert len(a1000_connections) == 1, f"{name} opened {len(a1000_connections)}"

    @pytest.mark.parametrize("name", sorted(a1000.commands))
    def test_a_cached_probe_leaves_only_the_commands_own_connection(
        self, name, real_probe, a1000_connections, monkeypatch, tmp_path
    ):
        """The second run inside the cache's 24 h answers from the cache."""
        self._run(name, tmp_path, monkeypatch)
        a1000_connections.clear()

        self._run(name, tmp_path, monkeypatch)

        expected = 0 if name in self.NO_CONNECTION else 1
        assert len(a1000_connections) == expected, f"{name} opened {len(a1000_connections)}"

    @pytest.mark.parametrize("name", sorted(a1000.commands))
    def test_an_unreachable_appliance_is_dialled_once_and_the_command_fails(
        self, name, real_probe, a1000_failed_connections, monkeypatch, tmp_path
    ):
        """The count above cannot see this: its stub connection always answers.

        An appliance that refuses used to be dialled three times for one
        command — the probe, the probe's fallback call, and the command's
        own service — each waiting out the connect timeout before the
        user got the one message they were owed.
        """
        result = self._run(name, tmp_path, monkeypatch)

        assert len(a1000_failed_connections) == 1, (
            f"{name} attempted {len(a1000_failed_connections)}"
        )
        # The command that asks the appliance nothing still answers, out
        # of the settings; every other one has failed and must say so.
        assert result.exit_code == (0 if name in self.NO_CONNECTION else 1), result.output

    @pytest.mark.parametrize("name", sorted(a1000.commands))
    def test_help_opens_none(self, name, real_probe, a1000_connections, tmp_path):
        result = CliRunner().invoke(cli, self._args(name, tmp_path, "--help"))

        assert result.exit_code == 0
        assert a1000_connections == []

    def test_every_exempt_command_exists(self):
        assert self.NO_CONNECTION.issubset(a1000.commands)

    def test_upload_and_analyze_reports_over_the_connection_it_uploaded_on(
        self, ctx_obj, a1000_connections, tmp_path
    ):
        """The two areas it spans used to be spanned by one facade object."""
        sample = tmp_path / "sample.bin"
        sample.write_bytes(b"x")

        result = invoke(a1000, ["upload-and-analyze", str(sample)], ctx_obj)

        assert result.exit_code == 0, result.output
        assert len(a1000_connections) == 1, "upload-and-analyze authenticated more than once"
        client = a1000_connections[0]
        client.submit_file_from_path.assert_called_once()
        client.get_summary_report_v2.assert_called_once_with(SHA1)

    def test_upload_and_analyze_reports_through_the_invocations_own_notifier(
        self, ctx_obj, a1000_connections, tmp_path, monkeypatch
    ):
        """The report half was built without one, so it made itself a fresh one.

        That is a second Console, ``quiet=False`` whatever ``-q`` asked
        for, and an ``had_error`` flag nothing reads — so a report that
        reported a failure would have exited 0.
        """
        sample = tmp_path / "sample.bin"
        sample.write_bytes(b"x")
        seen: list[object] = []

        def summary(self: A1000ReportService, h: str) -> dict[str, str]:
            seen.append(self.output)
            return {"sha1": h}

        monkeypatch.setattr(A1000ReportService, "get_summary_report_v2", summary)

        result = invoke(a1000, ["upload-and-analyze", str(sample)], ctx_obj)

        assert result.exit_code == 0, result.output
        assert seen == [ctx_obj.output]


# The lookups that answer "what else is associated with this", per group.
A1000_PIVOTS = ("ip-urls", "ip-files", "ip-domains")
TICLOUD_PIVOTS = (
    "ip-files",
    "ip-urls",
    "ip-domains",
    "domain-files",
    "domain-urls",
    "domain-ips",
    "domain-related",
    "url-files",
    "uri-index",
)


class TestCommandsReachTheirServicesByAttributeNotByName:
    """The edge from a command to its service must be one mypy can see.

    Both pivot factories used to pick the wrapper to call with
    ``getattr(service, f"{method}_aggregated" if paged else method)``,
    which made the ``_aggregated`` suffix convention load-bearing across
    twelve service methods and checkable by nothing: renaming any of them
    passed ruff, passed mypy, and failed in front of the analyst. The only
    guard was a parametrized table in ``tests/test_cli_ticloud.py``, which
    is a test remembering to be updated rather than a compiler refusing.

    Read out of the source rather than out of the behaviour, for the
    reason the sweeps above are: a test that only called the commands
    would pass the day the next one is written the old way. It is the one
    claim in this file that has to be, so it is the only one — and it asks
    about the computed name, not about the word ``getattr``. What the
    factories are handed instead is left to mypy, which is the whole point
    of handing them anything checkable: a lambda, an ``attrgetter``, a
    ``partial`` and an unbound method are all edges it follows, and a test
    that insisted on one of them would fail the three improvements.
    """

    def test_no_command_module_dispatches_on_a_computed_attribute_name(self):
        computed = [
            f"{module.name}:{node.lineno}"
            for module in command_modules()
            for node in ast.walk(module_tree(module))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "setattr"}
            and len(node.args) > 1
            and not isinstance(node.args[1], ast.Constant)
        ]

        assert not computed, f"commands reaching a service by a name they built: {computed}"


class TestBothGroupsShareOneDefinitionOfTheirInputs:
    """The batch of hashes and ``--output-dir`` are declared in one place.

    They were declared in two, once per group, and both copies drifted
    before anyone read them side by side: two different ``--hashes`` help
    strings, no usage error on the ticloud side — which is why
    ``reputation -f hashes.txt`` shipped exiting 2 with "Missing
    argument" — and a ``--output-dir`` default that stayed frozen at
    import there for two commits after the A1000 one was fixed, so
    ``ticloud download`` wrote live malware to whichever directory the
    process had started in.

    A test that only compared the two behaviours would pass the day a
    third copy appeared, so the sweeps that count declarations read the
    source of every command module. What they no longer do is name the
    file the declarations must live in: moving or renaming the module the
    groups share is not a second copy of anything, and the tests that
    spelled it out failed on the improvement as readily as on the drift.

    ``--all`` is checked the other way round, against the commands rather
    than the source, for the reason written on that test.
    """

    # Declared once and no more. ``--all`` is not here: ``a1000 extracted``
    # keeps a hidden, deprecated one of its own, and the flag spells two
    # further contracts in the yara commands. ``--max-results`` is: its
    # second declaration was a bare ``type=int`` where the first was
    # ``IntRange(min=1)``, so ``--max-results -5`` was accepted and capped
    # nothing. ``--limit`` is the same bound on the same kind of budget:
    # three declarations, three help lines and no ``IntRange`` between
    # them, so ``--limit 0`` was clamped to 1 inside the service and then
    # advised, by the CLI, to be raised above 0.
    SHARED_FLAGS: ClassVar = [
        "--hash-file",
        "--hashes",
        "--limit",
        "--max-results",
        "--output-dir",
    ]

    # Every spelling the two copies went by, underscores stripped: the
    # ticloud copies were ``_hash_inputs`` and ``_collect_hashes`` while
    # the A1000 originals were ``hash_inputs`` and ``_collect_hashes``, so
    # a name-for-name comparison would have missed the drift it is here
    # to catch. ``download_to`` joined them for the same reason the flags
    # did: both groups spelled the same download out, and only one copy
    # learned to clear away the directory it had made for a sample that
    # never arrived.
    SHARED_HELPERS: ClassVar = frozenset(
        {
            "hash_inputs",
            "collect_hashes",
            "required_hashes",
            "output_dir_option",
            "fetch_all_flag",
            "max_results_option",
            "download_to",
        }
    )

    @staticmethod
    def _click_options(module: Path) -> list[ast.Call]:
        """Every ``click.option(...)`` call in the module, as written."""
        return [
            node
            for node in ast.walk(module_tree(module))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "option"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "click"
        ]

    @staticmethod
    def _flag(call: ast.Call) -> str | None:
        return next(
            (
                argument.value
                for argument in call.args
                if isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and argument.value.startswith("--")
            ),
            None,
        )

    @pytest.mark.parametrize("flag", SHARED_FLAGS)
    def test_one_module_declares_the_flag(self, flag):
        """One declaration, wherever it lives.

        The module was named here as well, so moving or renaming the file
        the groups share failed a test about them not sharing it. What
        must not happen is a second declaration; which file holds the
        first is a filing decision.
        """
        declared = [
            module.name
            for module in command_modules()
            for call in self._click_options(module)
            if self._flag(call) == flag
        ]

        assert len(declared) == 1, f"{flag} is declared {len(declared)} times, in {declared}"

    def test_every_pivot_lookup_pages_on_the_same_flag(self):
        """Both groups' pivots publish one ``--all``, and publish it alike.

        Compared with each other rather than against a sentence written
        down here: what the two groups owe each other is that the flag
        means the same thing in both, and rewording published help is not
        a change to that — but it used to fail this file, because the help
        line was pinned as a constant and counted in the source.

        The flag name alone is not something to count either: ``--all``
        also publishes every ruleset, pages ``yara-repo-list``, and
        survives as a hidden no-op on ``extracted``, which are four
        contracts rather than one. So it is the pivots' own flags that are
        gathered, by the commands that must agree.
        """
        expected = [
            f"{group.name} {name}"
            for group, names in ((a1000, A1000_PIVOTS), (ticloud, TICLOUD_PIVOTS))
            for name in names
        ]
        paging = [
            (f"{group.name} {name}", param)
            for group, names in ((a1000, A1000_PIVOTS), (ticloud, TICLOUD_PIVOTS))
            for name in names
            for param in group.commands[name].params
            if param.name == "fetch_all"
        ]

        assert [where for where, _ in paging] == expected, "a pivot offers no --all, or offers two"
        flags = [param for _, param in paging if isinstance(param, click.Option)]
        assert len(flags) == len(paging), "a pivot declares --all as an argument, not an option"
        published = {(tuple(param.opts), param.help, param.is_flag) for param in flags}
        assert len(published) == 1, f"the pivots spell --all {len(published)} ways: {published}"
        opts, help_text, is_flag = published.pop()
        assert opts == ("--all",)
        assert is_flag
        assert help_text, "the shared paging flag publishes no help line"

    def test_each_shared_helper_has_one_definition(self):
        """No module writes its own copy — including the one they share.

        The sweep used to exempt ``_shared_inputs.py`` by name and count
        definitions anywhere else, which is the same claim as long as that
        file keeps its name and a weaker one the moment it does not.
        """
        defined: dict[str, list[str]] = {}
        for module in command_modules():
            for node in module_tree(module).body:
                name = node.name.lstrip("_") if isinstance(node, ast.FunctionDef) else ""
                if name in self.SHARED_HELPERS:
                    defined.setdefault(name, []).append(module.name)

        assert defined, "no shared input helper was found; this sweep is reading nothing"
        copies = {name: where for name, where in defined.items() if len(where) > 1}
        assert not copies, f"second definitions of shared input handling: {copies}"

    # The three commands that take a result budget, and the rest of a
    # command line each of them needs to reach its own parsing.
    LIMITED: ClassVar = [
        (a1000, "list", []),
        (a1000, "search", ["-q", "available:true"]),
        (ticloud, "search", ["available:true"]),
    ]

    def test_every_command_offering_limit_is_covered(self):
        """The sentinel for the two below: a fourth ``--limit`` is not exempt."""
        offering = {
            f"{group.name} {name}"
            for group in (a1000, ticloud)
            for name, command in group.commands.items()
            if any("--limit" in param.opts for param in command.params)
        }

        assert offering == {f"{group.name} {name}" for group, name, _ in self.LIMITED}

    @pytest.mark.parametrize(
        "group,name,rest",
        LIMITED,
        ids=["a1000 list", "a1000 search", "ticloud search"],
    )
    def test_a_budget_of_zero_is_refused_where_it_is_typed(self, group, name, rest, ctx_obj):
        """``--limit 0`` is a wrong command line, not a fetch to clamp.

        It was accepted, clamped to one record deep in the service, and
        then answered with "raise --limit above 0" — advice about the
        number the analyst had just been told was fine.
        """
        result = invoke(group, [name, *rest, "--limit", "0"], ctx_obj)

        assert result.exit_code == 2, result.output

    def test_one_help_line_for_the_shared_budget(self):
        published = {
            (tuple(param.opts), param.default, param.help)
            for group, name, _ in self.LIMITED
            for param in group.commands[name].params
            if isinstance(param, click.Option) and "--limit" in param.opts
        }

        assert len(published) == 1, f"--limit is offered {len(published)} ways: {published}"

    @pytest.mark.parametrize("group", [a1000, ticloud], ids=["a1000", "ticloud"])
    def test_every_hash_batch_is_offered_the_same_way(self, group):
        for name, command in group.commands.items():
            batch = [param for param in command.params if param.name == "hashes"]
            for param in batch:
                assert param.opts == ["--hashes", "-h"], name
                assert param.multiple, name
            for param in (p for p in command.params if p.name == "hash_file"):
                assert param.opts == ["--hash-file", "-f"], name
                assert param.help == "File with hashes (one per line)", name
                assert param.type.exists, name

    @pytest.mark.parametrize("group", [a1000, ticloud], ids=["a1000", "ticloud"])
    def test_every_output_dir_resolves_when_the_command_runs(self, group):
        """Not ``Path.cwd()``: called here it would name the directory this
        module was imported from, and the sample would land there rather
        than where the analyst is standing."""
        for name, command in group.commands.items():
            for param in (p for p in command.params if p.name == "output_dir"):
                # ``==`` rather than ``is``: every attribute access on a
                # classmethod builds a fresh bound method, so identity is
                # not the question — whether it is still uncalled is.
                assert param.default == Path.cwd, name
                assert param.help == "Output directory  [default: the current directory]", name


def _decorated_with(tree: ast.Module, decorator: str) -> list[ast.FunctionDef]:
    """Every function in ``tree`` carrying ``@decorator``, by that bare name."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(entry, ast.Name) and entry.id == decorator for entry in node.decorator_list
        )
    ]


class TestEveryDestructiveCommandAsksThroughOneGuard:
    """``@yes_flag`` and the body that honours it are one decision, not two.

    The flag was shared and the four lines that read it were not, so eight
    commands each re-spelled ``if not yes and not click.confirm(...)`` and
    three of them worded the refusal differently. A ninth could ship with
    the flag declared and no guard under it -- click would accept
    ``--yes``, the prompt would be put anyway, and a cron job's delete
    would hang on a closed stdin, which is the exact failure the flag
    exists to prevent.

    Read out of the source rather than out of the behaviour, like the
    sweeps above: a test that only invoked the eight would pass the day the
    ninth is written the old way.
    """

    # The floor, so that a sweep matching nothing cannot pass as
    # compliance. Eight commands ask before they act today.
    ASKING = 8

    def test_every_command_taking_yes_consults_the_shared_guard(self):
        without = [
            f"{module.name}:{function.name}"
            for module in command_modules()
            for function in _decorated_with(module_tree(module), "yes_flag")
            if not any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "confirmed"
                for node in ast.walk(function)
            )
        ]

        assert not without, f"a command declares --yes and never consults it: {without}"

    def test_no_command_taking_yes_spells_the_prompt_itself(self):
        """The guard is the helper's; a body that prompts again is a second copy."""
        own = [
            f"{module.name}:{function.name}"
            for module in command_modules()
            for function in _decorated_with(module_tree(module), "yes_flag")
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "confirm"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "click"
        ]

        assert not own, f"a command carrying --yes prompts on its own: {own}"

    def test_the_sweep_still_finds_the_commands_that_ask(self):
        guarded = [
            f"{module.name}:{function.name}"
            for module in command_modules()
            for function in _decorated_with(module_tree(module), "yes_flag")
        ]

        assert len(guarded) >= self.ASKING, f"the sweep sees only {len(guarded)}: {guarded}"


# The service wrappers whose call an analyst cannot take back: the sample,
# the verdict, the ruleset or the repository is gone from the appliance and
# no flag on this CLI brings it back. This is the oracle the sweeps below
# ask "is this command destructive?" — without one, "destructive" was
# whatever the command already did, and a ``purge-everything`` with an
# empty body passed every assertion in three files.
#
# The line is irreversibility on the *appliance*. What ``config`` writes is
# a local file, and ``config save`` writes the one it was told to; those
# commands are held to the prompt/``--yes`` consistency the sweeps below
# also make, and not to this.
IRREVERSIBLE_OPERATIONS = frozenset(
    {
        "delete_sample",
        "batch_delete_samples",
        "delete_classification",
        "delete_yara_ruleset",
        "delete_yara_repository",
        "remove_user_tags",
        "remove_all_user_tags",
        # Not removal-shaped by name, and irreversible all the same: its
        # ``clear`` operation discards a ruleset's retro-hunt history.
        "yara_cloud_retro_scan",
    }
)

# Removal-shaped names that are not irreversible, and why. The completeness
# test below reads this as the other half of a classification: a wrapper
# whose name says removal is in one list or the other, never in neither.
REVERSIBLE_DESPITE_THE_NAME = {
    "reset_yara_update_interval": "puts a setting back to its default; set it again and it is back",
}

_REMOVAL_VERBS = re.compile(r"(^|_)(delete|remove|purge|drop|clear|reset|wipe|destroy|erase)(_|$)")

_SERVICE_CLASSES = (*A1000_SERVICES, TitaniumCloudService, TitaniumCloudNetworkService)


def _service_surface() -> set[str]:
    """Every public wrapper the services offer the command layer."""
    return {
        name
        for service in _SERVICE_CLASSES
        for name in dir(service)
        if not name.startswith("_") and callable(getattr(service, name))
    }


def _commands_reaching_an_irreversible_operation(
    over: dict[str, click.Command] | None = None,
) -> set[str]:
    """Read off what a command *references*, not only what it calls.

    ``run_step`` takes a callable, so ``run_step(output, "Deleting...",
    partial(service.delete_sample, h))`` is the natural way to write a
    delete here — and under a resolution that recorded only names in call
    position, that command deleted without appearing to touch
    ``delete_sample`` at all. A method handed to something that will run
    it is a method run.
    """
    return {
        where
        for where, command in (commands() if over is None else over).items()
        if reached_by(command).referenced & IRREVERSIBLE_OPERATIONS
    }


class TestEveryIrreversibleCommandAsksBeforeItActs:
    """What "destructive" is, said once, so the sweeps have something to read.

    The ``--yes`` sweeps cross-checked prompting against the flag and
    against the README, and all three took the set of destructive commands
    from the commands that already prompted. Nothing anywhere decided what
    destructive *meant*, so a command that deleted and asked nothing was
    consistent with itself and passed: verified, by registering a
    ``purge-everything`` whose body was empty and watching five assertions
    across three files agree with it.

    The oracle is the service call, which is where the irreversibility
    actually is, and the edge from command to call is resolved rather than
    grepped — see ``tests/cli_support.py``.
    """

    # The seven that reach one today. Pinned so that a resolution which has
    # stopped seeing them fails here rather than passing empty.
    REACHING: ClassVar = {
        "a1000 delete",
        "a1000 batch-delete",
        "a1000 delete-classification",
        "a1000 remove-tags",
        "a1000 yara-delete",
        "a1000 yara-cloud-retro",
        "a1000 yara-repo-delete",
    }

    def test_every_removal_shaped_wrapper_is_classified(self):
        """A new ``purge_samples`` is destructive or is said not to be.

        The table above is what the sweeps mean by destructive, so a
        wrapper that quietly fails to be in it is a command the sweeps
        stop asking about. Only the removal-shaped names are demanded —
        a rule over the service surface rather than a second list to keep.
        """
        surface = _service_surface()
        removal_shaped = {name for name in surface if _REMOVAL_VERBS.search(name)}

        unclassified = removal_shaped - IRREVERSIBLE_OPERATIONS - set(REVERSIBLE_DESPITE_THE_NAME)
        assert not unclassified, (
            f"wrappers whose name says removal and nothing says whether they are "
            f"irreversible: {sorted(unclassified)}"
        )

    def test_the_table_names_wrappers_that_exist(self):
        """A renamed wrapper leaves a row here matching nothing, and no command."""
        gone = (IRREVERSIBLE_OPERATIONS | set(REVERSIBLE_DESPITE_THE_NAME)) - _service_surface()

        assert not gone, f"classified but no longer a service wrapper: {sorted(gone)}"

    def test_the_commands_reaching_one_are_the_ones_we_think(self):
        reaching = _commands_reaching_an_irreversible_operation()

        assert reaching == self.REACHING, (
            f"the resolution sees {sorted(reaching)}, not {sorted(self.REACHING)}"
        )

    def test_every_command_reaching_one_asks_first(self):
        """The claim the ``--yes`` sweeps were assuming rather than making."""
        silent = sorted(_commands_reaching_an_irreversible_operation() - set(commands_that_ask()))

        assert not silent, f"a command deletes and asks nothing: {silent}"

    def test_every_command_reaching_one_can_be_scripted(self):
        without = sorted(_commands_reaching_an_irreversible_operation() - commands_offering_yes())

        assert not without, f"a command deletes with no --yes to get past its question: {without}"


class TestTheResolutionSeesWhatTheSubstringSweepMissed:
    """The three shapes that got past the old sweep, each written out here.

    Written as modules the resolver has never seen rather than as changes
    to the real command modules: what is under test is the resolution, and
    a proof that only works on the eight commands it was written against
    is the thing this replaces.
    """

    def _tree(self, source: str) -> ast.Module:
        return ast.parse(textwrap.dedent(source))

    ASKS_THROUGH_A_HELPER = """
        def _ask_first(output, yes):
            return confirmed(output, yes, "Delete every sample on the appliance?")

        @a1000.command("scrub")
        @yes_flag
        @click.pass_context
        def scrub(ctx, yes):
            service, output, _ = probed_a1000(ctx, A1000SampleService)
            if not _ask_first(output, yes):
                return
            run_step(output, "Scrubbing...", lambda: service.delete_sample("x"))
    """

    DELETES_AND_SAYS_NOTHING = """
        @a1000.command("purge-everything")
        @click.pass_context
        def purge_everything(ctx):
            service, output, _ = probed_a1000(ctx, A1000SampleService)
            run_step(output, "Purging...", lambda: service.batch_delete_samples(["x"]))
    """

    ONLY_TALKS_ABOUT_ASKING = '''
        @a1000.command("wipe")
        @click.pass_context
        def wipe(ctx):
            """Wipe the appliance.

            Guarded by confirmed() -- see the delete command.
            """
            # confirmed(output, yes, "...") used to live here
            service, output, _ = probed_a1000(ctx, A1000SampleService)
            run_step(output, "Wiping...", lambda: service.delete_sample("x"))
    '''

    def test_a_question_put_through_a_helper_is_a_command_that_asks(self):
        """The regression the ``--yes`` flag exists to prevent, one call out.

        The substring sweep read the command's own body, so this command
        shipped a prompt with no ``--yes`` behind it and a cron job's
        delete hung on a closed stdin. Both readings are here: the one
        that missed it, and the one that does not.
        """
        source = textwrap.dedent(self.ASKS_THROUGH_A_HELPER)
        body = source[source.index("def scrub") :]

        assert "confirmed(" not in body, "the old sweep read this body and saw no question"
        assert "confirmed" in reached_in(self._tree(source), "scrub").invoked

    def test_a_command_that_deletes_and_never_asks_is_named(self):
        reached = reached_in(self._tree(self.DELETES_AND_SAYS_NOTHING), "purge_everything")

        assert reached.referenced & IRREVERSIBLE_OPERATIONS == {"batch_delete_samples"}
        assert not reached.invoked & ASKING_CALLS, "it asks nothing, and must read as such"

    def test_a_comment_about_asking_is_not_a_command_that_asks(self):
        """``confirmed(`` in a comment counted as a question; a call is a call."""
        source = textwrap.dedent(self.ONLY_TALKS_ABOUT_ASKING)

        assert "confirmed(" in source, "this is the text the old sweep matched"
        reached = reached_in(self._tree(source), "wipe")
        assert not reached.invoked & ASKING_CALLS
        assert not reached.referenced & ASKING_CALLS, "a docstring is not a question either"
        assert reached.referenced & IRREVERSIBLE_OPERATIONS == {"delete_sample"}


# One command written the way ``run_step`` invites, and no question in
# front of it. Defined here rather than described, so the sweeps below can
# be pointed at a real ``click.Command`` whose callback is a real ``def``
# in a real file — which is what they read.
def _run(step: Callable[[], object]) -> None:
    """Stands in for ``run_step``: something else runs the callable it is given."""
    step()


@click.command("purge-everything")
@click.pass_obj
def _purge_everything(service: A1000SampleService) -> None:
    """Delete a sample by handing the bound method to whatever will run it."""
    _run(partial(service.delete_sample, SHA256))


class TestAMethodHandedOverIsAMethodRun:
    """Five ways to spell one delete. The resolution used to see the first.

    ``_called_names`` recorded an attribute only where it stood in a
    call's ``func``, so a bound method passed as a *value* was a method
    nobody had touched. That is not a hypothetical spelling here:
    ``a1000/samples.py`` already writes ``partial(...)`` for a renderer,
    and ``run_step`` takes a callable, so ``run_step(output,
    "Deleting...", partial(service.delete_sample, h))`` is the obvious way
    to write a delete command — one that would have deleted without asking
    and passed every sweep in this file.

    Each shape is written out as a module the resolver has never seen, and
    each is read twice: once by the reading this replaced, and once by the
    one in ``tests/cli_support.py``.
    """

    SHAPES: ClassVar = {
        "called outright": 'run_step(output, "Deleting...", lambda: service.delete_sample(h))',
        "handed to partial": 'run_step(output, "Deleting...", partial(service.delete_sample, h))',
        "bound to a name": "operation = service.delete_sample\noperation(h)",
        "looked up by string": 'getattr(service, "delete_sample")(h)',
        "dispatched from a table": 'table = {"delete": service.delete_sample}\ntable[action](h)',
    }

    # The four the old reading was blind to. Named rather than counted, so
    # a shape that quietly stops being a shape fails here.
    WAS_BLIND_TO: ClassVar = {
        "handed to partial",
        "bound to a name",
        "looked up by string",
        "dispatched from a table",
    }

    def _command(self, body: str) -> ast.Module:
        head = textwrap.dedent("""
            @a1000.command("scrub")
            @click.pass_context
            def scrub(ctx, action, h):
                service, output, _ = probed_a1000(ctx, A1000SampleService)
        """)
        return ast.parse(head + textwrap.indent(body, "    ") + "\n")

    def _in_call_position(self, tree: ast.Module) -> set[str]:
        """The reading this replaced, written out so the comparison is not a claim."""
        return {
            call.func.attr
            for call in ast.walk(tree)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        }

    @pytest.mark.parametrize("shape", sorted(SHAPES), ids=lambda shape: shape.replace(" ", "-"))
    def test_the_delete_is_seen_however_the_method_is_handed_over(self, shape):
        reached = reached_in(self._command(self.SHAPES[shape]), "scrub")

        assert reached.referenced & IRREVERSIBLE_OPERATIONS == {"delete_sample"}

    @pytest.mark.parametrize("shape", sorted(SHAPES), ids=lambda shape: shape.replace(" ", "-"))
    def test_none_of_them_reads_as_a_command_that_asks(self, shape):
        """Over-reporting a delete is a command made to ask; the reverse excuses one."""
        reached = reached_in(self._command(self.SHAPES[shape]), "scrub")

        assert not reached.invoked & ASKING_CALLS
        assert not reached.referenced & ASKING_CALLS

    def test_the_reading_this_replaced_saw_only_the_first(self):
        """The other half of the comparison: four shapes, four blind spots."""
        blind = {
            shape
            for shape, body in self.SHAPES.items()
            if "delete_sample" not in self._in_call_position(self._command(body))
        }

        assert blind == self.WAS_BLIND_TO

    def test_the_sweep_fails_for_a_command_that_deletes_this_way(self):
        """The teeth, through the real sweep rather than through its ingredients.

        ``_purge_everything`` above is a registered ``click.Command`` whose
        body hands ``delete_sample`` to a runner and asks nothing. Both
        halves of the sweep are run over it: it must read as reaching an
        irreversible operation, and it must not read as asking.
        """
        registered = {"a1000 purge-everything": _purge_everything}

        reaching = _commands_reaching_an_irreversible_operation(registered)
        silent = reaching - set(commands_that_ask(registered))

        assert reaching == {"a1000 purge-everything"}
        assert silent == {"a1000 purge-everything"}, "the sweep passes a silent delete"
        assert not commands_offering_yes(registered), "and it offers no --yes either"


# The guard, and a command that only mentions it. Registered rather than
# described, because what reads it is the real sweep, and the sweep reads
# a ``click.Command``'s callback out of the file that wrote it.
def _guarded_delete(
    service: A1000SampleService, output: RichOutput, yes: bool, sample: str
) -> None:
    """A helper that really does ask, so that naming it is worth something."""
    if confirmed(output, yes, f"Delete {sample}?"):
        service.delete_sample(sample)


@click.command("purge-everything")
@yes_flag
@click.pass_obj
def _purge_naming_the_guard(service: A1000SampleService, yes: bool) -> None:
    """Delete with no question at all; --yes is accepted and ignored."""
    _fallback = _guarded_delete  # named, never called
    service.delete_sample(SHA256)


class TestOnlyAQuestionPutCountsAsAsking:
    """Naming the guard is not asking it, and the closure used to say it was.

    The two readings share a resolution but not a traversal: what a body
    *references* is followed into, and what it *invokes* is folded up only
    along invocations. Under one traversal folding both, a body that
    mentioned ``_guarded_delete`` in a value was walked into — and every
    call that helper makes, the question included, came back as the
    command's own. ``commands_that_ask`` reads ``invoked`` precisely so
    that it cannot be talked into consent; that is what this pins.
    """

    def test_naming_a_helper_that_asks_is_not_asking(self):
        registered = {"a1000 purge-everything": _purge_naming_the_guard}

        reaching = _commands_reaching_an_irreversible_operation(registered)
        asking = set(commands_that_ask(registered))

        assert reaching == {"a1000 purge-everything"}
        assert not asking, f"a body that only names the guard reads as asking: {sorted(asking)}"
        assert reaching - asking == {"a1000 purge-everything"}, "the sweep passes a silent delete"

    def test_the_helper_it_names_is_still_something_it_references(self):
        """The other direction of the same contract, which stays deliberately loose."""
        reached = reached_by(_purge_naming_the_guard)

        assert "confirmed" in reached.referenced, "a name in a value is still a name reached"
        assert "confirmed" not in reached.invoked


class TestOneNameTwoClosuresInOneFile:
    """Two commands in one file both write ``def fetch()``. They are not one def.

    The resolution answered a name with every same-named ``FunctionDef``
    anywhere in the module, nested closures included, so a read-only
    command was credited with the delete its neighbour's ``fetch``
    performs — and was told to grow a prompt and a ``--yes`` for something
    that touches nothing. Only ``_shared_inputs``' two ``decorate``
    closures collide today, which is why this is written out rather than
    left to the real modules to demonstrate.
    """

    TWO_COMMANDS_ONE_CLOSURE_NAME = '''
        @a1000.command("delete")
        @yes_flag
        @click.pass_context
        def delete(ctx, yes, sha1):
            service, output, _ = probed_a1000(ctx, A1000SampleService)

            def fetch():
                return service.delete_sample(sha1)

            if not confirmed(output, yes, "Delete the sample?"):
                return
            run_step(output, "Deleting...", fetch)

        @a1000.command("info")
        @click.pass_context
        def info(ctx, sha1):
            """The most ordinary edit in this repo: one more read-only command."""
            service, output, _ = probed_a1000(ctx, A1000SampleService)

            def fetch():
                return service.get_sample_info(sha1)

            run_step(output, "Fetching...", fetch)
    '''

    def _tree(self) -> ast.Module:
        return ast.parse(textwrap.dedent(self.TWO_COMMANDS_ONE_CLOSURE_NAME))

    def test_the_reading_command_reaches_neither_the_delete_nor_a_question(self):
        reached = reached_in(self._tree(), "info")

        assert not reached.referenced & IRREVERSIBLE_OPERATIONS, (
            "a read-only command is credited with its neighbour's closure"
        )
        assert not reached.invoked & ASKING_CALLS

    def test_the_deleting_command_still_reaches_its_own_closure(self):
        """The direction that must not be traded away to fix the other."""
        reached = reached_in(self._tree(), "delete")

        assert reached.referenced & IRREVERSIBLE_OPERATIONS == {"delete_sample"}
        assert reached.invoked & ASKING_CALLS == {"confirmed"}


class TestADeleteParkedInAValueIsStillADelete:
    """The two shapes the walk started too late to see.

    ``_reach`` starts at a ``def`` and used to walk only into other
    ``def``s, so a method held at module scope was reached by nobody: the
    body names the table, the table names the method, and the resolution
    stopped at the first hop. Both are bindings a name resolves to, so
    both are walked — in the reference direction only, since a value is
    not a call and constructing a class does not put its questions.
    """

    HELD_IN_A_MODULE_CONSTANT = """
        _OPERATIONS = {"purge": A1000SampleService.delete_sample}

        @a1000.command("purge")
        @click.pass_context
        def purge(ctx, h):
            service, output, _ = probed_a1000(ctx, A1000SampleService)
            run_step(output, "Purging...", partial(_OPERATIONS["purge"], service, h))
    """

    HELD_ON_A_CLASS_ATTRIBUTE = """
        class Ops:
            purge = A1000SampleService.delete_sample

        @a1000.command("purge")
        @click.pass_context
        def purge(ctx, h):
            service, output, _ = probed_a1000(ctx, A1000SampleService)
            run_step(output, "Purging...", partial(Ops.purge, service, h))
    """

    SHAPES: ClassVar = {
        "held in a module constant": HELD_IN_A_MODULE_CONSTANT,
        "held on a class attribute": HELD_ON_A_CLASS_ATTRIBUTE,
    }

    @pytest.mark.parametrize("shape", sorted(SHAPES), ids=lambda shape: shape.replace(" ", "-"))
    def test_the_method_the_value_holds_is_reached(self, shape):
        reached = reached_in(ast.parse(textwrap.dedent(self.SHAPES[shape])), "purge")

        assert reached.referenced & IRREVERSIBLE_OPERATIONS == {"delete_sample"}

    @pytest.mark.parametrize("shape", sorted(SHAPES), ids=lambda shape: shape.replace(" ", "-"))
    def test_neither_of_them_reads_as_a_command_that_asks(self, shape):
        reached = reached_in(ast.parse(textwrap.dedent(self.SHAPES[shape])), "purge")

        assert not reached.invoked & ASKING_CALLS


class TestANameResolvesInsideTheModuleThatWroteTheCall:
    """A name two modules define is not one name, and used to resolve as one.

    The lookup was a single ``setdefault`` over every command module, so a
    colliding name resolved to whichever module sorted first. The
    consequence was live rather than theoretical: ``config init`` calls
    ``click.confirm``, which records the bare ``confirm``, and the
    resolver handed it ``a1000/metadata.py``'s nested ``confirm`` — whose
    body calls ``confirmed``. ``config init`` was credited with reaching
    the shared guard it has never called, and came out right only because
    it does ask, by another route.
    """

    # The names more than one command module defines today. Pinned so the
    # tests below cannot pass by there being no collision left to resolve.
    COLLIDING: ClassVar = {
        "__call__",
        "confirm",
        "decorate",
        "domain_report",
        "download",
        "ip_report",
        "lookup",
        "search",
        "upload",
    }

    def _defined_by_module(self) -> dict[str, set[Path]]:
        defined: dict[str, set[Path]] = {}
        for module in command_modules():
            for node in ast.walk(module_tree(module)):
                if isinstance(node, ast.FunctionDef):
                    defined.setdefault(node.name, set()).add(module)
        return defined

    def test_the_command_modules_really_do_share_names(self):
        shared = {name for name, modules in self._defined_by_module().items() if len(modules) > 1}

        assert shared == self.COLLIDING, f"the collisions moved: {sorted(shared ^ self.COLLIDING)}"

    def test_no_resolution_lands_in_a_module_nothing_imports_from(self):
        """The rule, over every command at once: a body is reached or it is imported.

        A definition walked in another file is one the walking module
        wrote an import for. Under the old lookup, seven of the nine names
        above could put a command in a file it has no edge to.
        """
        reachable = {
            module: {module, *command_module_imports()[module].values()}
            for module in command_modules()
        }
        strayed = {}
        for where, command in commands().items():
            home = Path(sys.modules[command.callback.__module__].__file__ or "")
            visited = {walked.split(":")[0] for walked in reached_by(command).walked}
            outside = visited - {module.name for module in reachable[home]}
            if outside:
                strayed[where] = sorted(outside)

        assert not strayed, f"a command's reach walked into a module it has no import to: {strayed}"

    def test_the_two_confirms_stay_in_their_own_files(self):
        """The collision that was live, pinned from both sides.

        ``config init`` writes ``click.confirm`` and ``a1000 remove-tags``
        writes a nested ``confirm`` of its own. Each must walk the one its
        own module defines, and neither the other's.
        """
        walked = {where: reached_by(command).walked for where, command in commands().items()}

        assert "config.py:confirm" in walked["config init"]
        assert "metadata.py:confirm" not in walked["config init"]
        assert "metadata.py:confirm" in walked["a1000 remove-tags"]
        assert "config.py:confirm" not in walked["a1000 remove-tags"]

    def test_a_local_helper_that_asks_nothing_is_not_the_shared_guard(self):
        """What the collision cost: a module's own ``confirm`` read as the guard's.

        The old lookup answered ``confirm`` with ``a1000/metadata.py``'s,
        which calls ``confirmed``. So a command whose *own* module defines
        a ``confirm`` that asks nothing was credited with asking — the
        exact reading that lets a silent delete through. Written as a
        module the resolver has never seen, with no import of the guard.
        """
        source = textwrap.dedent("""
            def confirm(question):
                return True

            @a1000.command("purge")
            @click.pass_context
            def purge(ctx):
                service, output, _ = probed_a1000(ctx, A1000SampleService)
                if not confirm("Delete everything?"):
                    return
                run_step(output, "Purging...", partial(service.delete_sample, "x"))
        """)
        reached = reached_in(ast.parse(source), "purge")

        assert reached.referenced & IRREVERSIBLE_OPERATIONS == {"delete_sample"}
        assert not reached.invoked & ASKING_CALLS, "its own confirm() is not the shared guard"


class TestEveryOptionAppearsInTheCommandItBelongsTo:
    """A declared option is a named parameter, not a key in ``**kwargs``.

    ``yara-repo-create`` and ``yara-repo-update`` swallowed five of their
    options in ``**_options`` and read them back out of ``ctx.params`` by
    string. All five fields are strings and update replaces every one of
    them, so renaming an option's destination type-checked, imported, and
    wrote a transposed repository the appliance accepted.
    """

    @pytest.mark.parametrize("group", [a1000, ticloud], ids=["a1000", "ticloud"])
    def test_no_command_swallows_its_options(self, group):
        hidden = {}
        for name, command in group.commands.items():
            if command.callback is None:
                continue
            signature = inspect.signature(command.callback)
            catch_all = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            missing = [
                param.name
                for param in command.params
                if param.name and param.name not in signature.parameters
            ]
            if catch_all or missing:
                hidden[f"{group.name} {name}"] = missing or ["**kwargs"]

        assert not hidden, f"options that reach the body only as a dict key: {hidden}"
