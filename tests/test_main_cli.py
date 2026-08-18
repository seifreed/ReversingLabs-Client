"""Tests for the root CLI group: option precedence, exit status, and its output helpers."""

import importlib
import io
import json
import os
import re
import signal
import stat
from pathlib import Path
from unittest.mock import MagicMock

import click
import pytest
import yaml
from click.testing import CliRunner
from rich.console import Console

from rl_cli.cli.main import cli
from rl_cli.config import clear_settings_cache
from rl_cli.render.output import RichOutput
from rl_cli.services.a1000 import A1000ReportService, A1000SampleService, A1000Session
from rl_cli.services.availability import ServiceStatus

# ``rl_cli.cli.main`` is shadowed by the ``main`` function its package
# re-exports, so the module itself has to be asked for by name.
main_module = importlib.import_module("rl_cli.cli.main")


def test_the_module_runs_as_a_script(monkeypatch):
    """Running the module as ``__main__`` reaches the entrypoint guard."""
    import runpy
    import sys

    module_file = main_module.__file__
    assert module_file is not None
    monkeypatch.setattr(sys, "argv", ["rl-cli", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(module_file, run_name="__main__")
    assert excinfo.value.code == 0


def _invoke_show(tmp_path, config_profile, *cli_args):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"default": config_profile}))
    runner = CliRunner()
    return runner.invoke(cli, ["--config", str(config_file), *cli_args, "config", "show"])


def _run(args, env=None):
    """Invoke the CLI with a fresh settings cache.

    ``get_settings`` keys on its arguments and the working directory, but
    not on the environment, so two invocations differing only in an
    environment variable would otherwise share one answer.
    """
    clear_settings_cache()
    return CliRunner().invoke(cli, args, env=env)


class TestOneInvocationHasOneReporter:
    """The formatter built a second ``RichOutput`` over the same console.

    Only the CLI's own was told about ``--quiet``, and only its
    ``had_error`` was read for the exit status, so every invocation
    carried a second answer to two questions that have one.
    """

    @pytest.fixture
    def captured(self):
        seen: dict = {}

        @cli.command("_probe_context")
        @click.pass_context
        def probe(ctx: click.Context) -> None:
            seen["obj"] = ctx.obj

        yield seen
        cli.commands.pop("_probe_context")

    def test_the_formatter_renders_through_the_cli_reporter(self, captured):
        result = _run(["_probe_context"])
        assert result.exit_code == 0, result.output
        assert captured["obj"].formatter.rich is captured["obj"].output
        assert captured["obj"].formatter.console is captured["obj"].output.console

    def test_the_formatter_hears_quiet(self, captured):
        result = _run(["-q", "_probe_context"])
        assert result.exit_code == 0, result.output
        assert captured["obj"].formatter.rich.quiet


class TestReportsFromOutsideACommandKeepTheColourPolicy:
    """Ctrl-C and the catch-all built a bare ``RichOutput``.

    Its default console reads the terminal, so both ignored `color:
    false` — the one setting the group callback goes out of its way to put
    on both consoles.
    """

    @pytest.fixture(autouse=True)
    def _forget_the_reporter(self, monkeypatch):
        monkeypatch.setattr(main_module, "_reporter", None)

    def test_an_interrupt_reports_through_the_configured_consoles(self):
        status = io.StringIO()
        configured = RichOutput(Console(no_color=True), Console(file=status, no_color=True))
        main_module._remember_reporter(configured)

        with pytest.raises(SystemExit) as exit_info:
            main_module._interrupted()

        assert exit_info.value.code == 130
        assert "Interrupted" in status.getvalue()

    def test_the_group_callback_hands_over_the_configured_policy(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"default": {"output": {"color": False}}}))

        result = _run(["-c", str(config_file), "config", "list-profiles"])

        assert result.exit_code == 0, result.output
        reporter = main_module._reporting()
        assert reporter.console.no_color
        assert reporter.status_console.no_color

    def test_an_interrupt_before_any_policy_is_read_stays_uncoloured(self):
        reporter = main_module._reporting()
        assert reporter.console.no_color
        assert reporter.status_console.no_color


class TestOneInvocationDoesNotReportThroughAnothersConsoles:
    """The slot is module-level, so it outlived the run that filled it.

    An unusable config file is reported before the colour policy can be
    read, so that path is documented to fall back to an uncoloured
    reporter. It could not: whatever the *previous* in-process invocation
    left in the slot answered instead — which is every ``CliRunner`` test in
    this suite, and any caller embedding ``cli()``. The invocation now
    empties the slot before anything can report.
    """

    @pytest.fixture(autouse=True)
    def _forget_the_reporter(self, monkeypatch):
        monkeypatch.setattr(main_module, "_reporter", None)

    def _write(self, tmp_path, name: str, body: str) -> Path:
        config_file = Path(tmp_path) / name
        config_file.write_text(body)
        return config_file

    def test_a_second_run_does_not_inherit_the_first_runs_reporter(self, tmp_path):
        coloured = self._write(tmp_path, "coloured.yaml", yaml.dump({"default": {}}))
        assert _run(["-c", str(coloured), "config", "list-profiles"]).exit_code == 0
        first = main_module._reporting()
        assert not first.console.no_color, "the first run has to leave a coloured reporter behind"

        # Reported before any policy is read, so it must use the fallback.
        unusable = self._write(tmp_path, "unusable.yaml", "default:\n  a1000:\n    host: []\n")
        result = _run(["-c", str(unusable), "config", "show"])

        assert result.exit_code == 1, result.output
        assert "a1000.host" in result.output
        assert main_module._reporter is None, "the failed run reported through the previous run's"
        assert main_module._reporting().console.no_color

    def test_a_run_that_reads_its_config_still_fills_the_slot(self, tmp_path):
        """Emptying it first must not leave the paths outside click with nothing."""
        config_file = self._write(tmp_path, "config.yaml", yaml.dump({"default": {}}))

        assert _run(["-c", str(config_file), "config", "list-profiles"]).exit_code == 0
        assert main_module._reporter is not None


class TestAnUnusableConfigFileIsNotACliBug:
    """A stated value that cannot be used stops the run, and used to stop
    it through main()'s catch-all — which prefixed the analyst's own typo
    with "Unexpected error", the framing reserved for our faults.
    """

    def _run_with(self, tmp_path, body: str):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(body)
        return _run(["-c", str(config_file), "config", "show"])

    def test_it_names_the_key_and_exits_nonzero(self, tmp_path):
        result = self._run_with(tmp_path, "default:\n  a1000:\n    host: []\n")

        assert result.exit_code == 1
        assert "a1000.host" in result.output
        assert "Unexpected error" not in result.output

    def test_it_reports_rather_than_raising(self, tmp_path):
        result = self._run_with(tmp_path, "default:\n  a1000:\n    host: []\n")

        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "Traceback" not in result.output


class TestOutputFormatPrecedence:
    def test_configured_format_is_respected(self, tmp_path):
        result = _invoke_show(tmp_path, {"output": {"format": "json"}})
        assert result.exit_code == 0
        assert '"profile": "default"' in result.output

    def test_cli_flag_overrides_configured_format(self, tmp_path):
        result = _invoke_show(tmp_path, {"output": {"format": "json"}}, "--output", "yaml")
        assert result.exit_code == 0
        assert "profile: default" in result.output

    def test_an_unusable_configured_format_stops_the_run(self, tmp_path):
        """It used to warn and render rich, alone among rejected config values.

        Every other value the file states and validation refuses stops the
        load, because a field silently left on its default is how a
        mistyped ``a1000.host`` retargeted the CLI at the vendor's public
        cloud. ``output.format`` escaped that only because it was typed as
        a bare ``str``: `-o json`-shaped automation reading a config file
        with a typo in it got rich-rendered tables and exit 0, which is a
        pipeline parsing box-drawing characters rather than a report.
        """
        result = _invoke_show(tmp_path, {"output": {"format": "nonsense"}})

        # Unwrapped: a tmp_path is long enough for rich to break the file
        # name across lines, which is a rendering, not the message.
        reported = result.output.replace("\n", "")

        assert result.exit_code == 1
        assert str(tmp_path / "config.yaml") in reported
        assert "output.format" in reported
        assert "'rich'" in reported, "the message has to say which words are accepted"
        # Named as the user's own typo, not as a fault of ours.
        assert "Unexpected error" not in result.output
        assert "Traceback" not in result.output

    def test_a_usable_configured_format_is_unaffected(self, tmp_path):
        result = _invoke_show(tmp_path, {"output": {"format": "yaml"}})

        assert result.exit_code == 0
        assert "profile: default" in result.output


class TestFlagPrecedence:
    def test_configured_verbose_survives_absent_flag(self, tmp_path):
        result = _invoke_show(tmp_path, {"output": {"verbose": True, "format": "json"}})
        assert result.exit_code == 0
        assert '"verbose": true' in result.output


class TestProfilePrecedence:
    """CLI flag -> environment -> .env file -> the built-in default.

    The profile picks which appliance the invocation talks to, so a
    ``RL_CLI_PROFILE`` that is quietly ignored is not a cosmetic bug.
    """

    def _profile(self, tmp_path, *args, env=None):
        config_file = tmp_path / "profiles.yaml"
        config_file.write_text(yaml.dump({"default": {}, "staging": {}, "prod": {}}))
        result = _run(["--config", str(config_file), "-o", "json", *args, "config", "show"], env)
        assert result.exit_code == 0, result.output
        return result.output

    def test_the_environment_selects_the_profile(self, tmp_path):
        output = self._profile(tmp_path, env={"RL_CLI_PROFILE": "staging"})
        assert '"profile": "staging"' in output

    def test_the_flag_beats_the_environment(self, tmp_path):
        output = self._profile(tmp_path, "--profile", "prod", env={"RL_CLI_PROFILE": "staging"})
        assert '"profile": "prod"' in output

    def test_the_environment_beats_a_dotenv_file(self, tmp_path):
        Path(".env").write_text("RL_CLI_PROFILE=prod\n")
        output = self._profile(tmp_path, env={"RL_CLI_PROFILE": "staging"})
        assert '"profile": "staging"' in output

    def test_a_dotenv_file_beats_the_built_in_default(self, tmp_path):
        Path(".env").write_text("RL_CLI_PROFILE=staging\n")
        assert '"profile": "staging"' in self._profile(tmp_path)

    def test_neither_flag_nor_environment_leaves_the_default(self, tmp_path):
        assert '"profile": "default"' in self._profile(tmp_path)

    def test_a_profile_absent_from_the_file_is_named_and_falls_back(self, tmp_path):
        """A named profile the file does not hold warns rather than acting as it."""
        config_file = tmp_path / "profiles.yaml"
        config_file.write_text(yaml.dump({"default": {}}))

        result = _run(["--config", str(config_file), "--profile", "ghost", "config", "show"])

        assert result.exit_code == 0, result.output
        assert "ghost" in result.output
        assert "not in" in result.output


class TestSaveWritesToTheProfileNotToAFileNamedAfterIt:
    """``-p`` is ``--profile`` on the root group and on every other command.

    ``config save`` alone read it as ``--path``, and both take a free
    string, so ``config save -p prod`` wrote a file literally named
    ``prod`` into the current directory holding the *active* profile's
    settings, left the config file untouched and reported success - and
    the next ``-p prod`` run went on talking to the old appliance.
    """

    def _config_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "default": {"a1000": {"host": "https://default.example"}},
                    "prod": {"a1000": {"host": "https://prod.example"}},
                }
            )
        )
        return config_file

    def test_a_profile_name_is_not_taken_as_a_path(self, tmp_path):
        config_file = self._config_file(tmp_path)

        result = _run(["-c", str(config_file), "config", "save", "-p", "prod"])

        assert result.exit_code == 2, result.output
        # No stray file named after the profile, and nothing claiming success.
        assert not Path("prod").exists()
        assert "saved" not in result.output

    def test_the_saved_file_holds_the_profile_that_was_meant(self, tmp_path):
        config_file = self._config_file(tmp_path)

        result = _run(["-c", str(config_file), "-p", "prod", "config", "save"])

        assert result.exit_code == 0, result.output
        written = yaml.safe_load(config_file.read_text())
        assert written["prod"]["a1000"]["host"] == "https://prod.example"
        assert written["default"]["a1000"]["host"] == "https://default.example"

    def test_the_long_option_still_names_a_destination(self, tmp_path):
        config_file = self._config_file(tmp_path)
        target = tmp_path / "copy.yaml"

        result = _run(
            ["-c", str(config_file), "-p", "prod", "config", "save", "--path", str(target)]
        )

        assert result.exit_code == 0, result.output
        assert yaml.safe_load(target.read_text())["prod"]["a1000"]["host"] == "https://prod.example"


class TestConfigCommandsDeferToTheOutputFormat:
    """Both are commands a script runs to decide what to do next.

    ``check-access`` rendered its panel on stdout, so ``-o json
    check-access > status.json`` produced a file of box-drawing
    characters; ``list-profiles`` sent the whole list to stderr, so
    ``-o json config list-profiles | jq -r '.[]'`` fed a fan-out loop
    nothing at all.
    """

    def _profiles_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"default": {}, "prod": {}}))
        return config_file

    def _nothing_reachable(self, monkeypatch):
        from rl_cli.services.availability import APIAvailabilityChecker

        probe = {
            "status": ServiceStatus.ERROR.value,
            "message": "probe failed: appliance did not answer",
            "credentials_configured": True,
            "api_accessible": False,
        }
        monkeypatch.setattr(
            APIAvailabilityChecker,
            "check_all",
            lambda self, force=False: {
                "timestamp": "2026-01-01T00:00:00",
                "titanium_cloud": probe,
                "a1000": probe,
                "summary": {"services_available": 0, "services_total": 2},
            },
        )

    def test_check_access_puts_the_availability_on_stdout_as_json(self):
        result = _run(["-o", "json", "config", "check-access"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["a1000"]["status"] == ServiceStatus.AVAILABLE.value
        assert payload["summary"]["services_available"] == 2

    def test_check_access_still_draws_the_panel_on_a_terminal(self):
        result = _run(["config", "check-access"])

        assert result.exit_code == 0, result.output
        assert "API Availability Status" in result.stdout

    def test_nothing_reachable_stays_parseable_and_reports_the_count(self, monkeypatch):
        """The count is the document's, not the exit status's.

        This asserted ``exit_code == 1``: the command exited on how many
        services answered, so a diagnostic that had worked failed for what
        it found. It reports and exits 0 now — see
        ``TestTheDiagnosticIsNotTheFailure`` — and the count a script
        branches on is where every other measured fact already was.
        """
        self._nothing_reachable(monkeypatch)

        result = _run(["-o", "json", "config", "check-access"])

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["summary"]["services_available"] == 0

    def test_list_profiles_puts_the_names_on_stdout_as_json(self, tmp_path):
        config_file = self._profiles_file(tmp_path)

        result = _run(["-c", str(config_file), "-o", "json", "config", "list-profiles"])

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == ["default", "prod"]

    def test_list_profiles_still_marks_the_current_one_on_a_terminal(self, tmp_path):
        config_file = self._profiles_file(tmp_path)

        result = _run(["-c", str(config_file), "-p", "prod", "config", "list-profiles"])

        assert result.exit_code == 0, result.output
        assert "prod (current)" in result.output


class TestConfigFileFromTheEnvironment:
    def test_the_environment_names_the_config_file_without_the_flag(self, tmp_path):
        """``--config`` is optional, so its absence must not outrank the env var."""
        config_file = tmp_path / "elsewhere.yaml"
        config_file.write_text(
            yaml.dump({"default": {"a1000": {"host": "https://appliance.example"}}})
        )

        result = _run(["config", "show"], env={"RL_CLI_CONFIG_FILE": str(config_file)})

        assert result.exit_code == 0, result.output
        assert "appliance.example" in result.output


class TestTildePathsAreExpanded:
    """`.env.example` documents ``RL_CLI_CACHE_DIR=~/.cache/rl-cli``.

    Nothing expands a tilde that arrives inside an environment variable,
    so the documented value used to make a directory literally named
    ``~`` under the current one.
    """

    def test_a_tilde_in_a_configured_directory_means_home(self):
        result = _run(
            ["-o", "json", "config", "show"],
            env={"RL_CLI_CACHE_DIR": "~/.cache/rl-cli", "RL_CLI_CONFIG_DIR": "~/.config/rl-cli"},
        )

        assert result.exit_code == 0, result.output
        assert str(Path.home() / ".cache" / "rl-cli") in result.output
        assert str(Path.home() / ".config" / "rl-cli") in result.output


class TestOnlyARealServiceCommandProbes:
    """Only a command that will actually talk to a service may probe.

    The probe fires from the command body, the first hook click runs
    after it has accepted the command line. Everything that exits before
    a body runs - ``--help``, a usage error, shell completion - must not
    reach it, and neither must ``config``.
    """

    @staticmethod
    def _forbid_probing(monkeypatch):
        from rl_cli.services.availability import APIAvailabilityChecker

        def explode(self, force=False):
            raise AssertionError("this invocation must not probe the API")

        monkeypatch.setattr(APIAvailabilityChecker, "check_all", explode)

    def test_help_skips_the_availability_check(self, monkeypatch):
        self._forbid_probing(monkeypatch)
        result = CliRunner().invoke(cli, ["a1000", "report", "--help"])

        assert result.exit_code == 0, result.output
        assert "Usage" in result.output

    def test_a_usage_error_skips_the_availability_check(self, monkeypatch):
        self._forbid_probing(monkeypatch)
        result = CliRunner().invoke(cli, ["a1000", "report"])

        assert result.exit_code == 2, result.output

    def test_shell_completion_skips_the_availability_check(self, monkeypatch):
        self._forbid_probing(monkeypatch)
        result = CliRunner().invoke(
            cli,
            [],
            env={"_CLI_COMPLETE": "bash_complete", "COMP_WORDS": "cli a1000 ", "COMP_CWORD": "2"},
        )

        assert result.exit_code == 0, result.output

    def test_config_commands_skip_the_availability_check(self, monkeypatch):
        self._forbid_probing(monkeypatch)
        result = CliRunner().invoke(cli, ["config", "show"])

        assert result.exit_code == 0, result.output

    def test_a_real_invocation_still_checks(self, monkeypatch):
        from rl_cli.services.availability import APIAvailabilityChecker

        calls: list[bool] = []

        def check_all(
            self: APIAvailabilityChecker, force: bool = False
        ) -> dict[str, dict[str, str]]:
            calls.append(True)
            return {"a1000": {"status": "available", "message": "ok"}}

        monkeypatch.setattr(APIAvailabilityChecker, "check_all", check_all)
        monkeypatch.setattr(A1000SampleService, "list_samples", lambda self, limit=100: [])

        CliRunner().invoke(cli, ["a1000", "list"])

        assert calls == [True]


class TestUnreachableApplianceIsAlwaysReported:
    """--quiet promises warnings survive it; it was hiding this one."""

    def _run(self, monkeypatch, tmp_path, *args, configured_quiet=False):
        from rl_cli.services.availability import APIAvailabilityChecker

        monkeypatch.setattr(
            APIAvailabilityChecker,
            "check_all",
            lambda self, force=False: {"a1000": {"status": "error", "message": "probe failed"}},
        )
        monkeypatch.setattr(
            A1000SampleService, "list_samples", lambda self, limit=100: [{"md5": "abc"}]
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"default": {"output": {"quiet": configured_quiet}}}))
        return CliRunner().invoke(cli, ["--config", str(config_file), *args, "a1000", "list"])

    def test_warning_and_hint_without_the_flag(self, monkeypatch, tmp_path):
        output = self._run(monkeypatch, tmp_path).output
        assert "A1000: probe failed" in output
        assert "check-access" in output

    def test_quiet_flag_keeps_the_warning_and_drops_the_hint(self, monkeypatch, tmp_path):
        output = self._run(monkeypatch, tmp_path, "-q").output
        assert "A1000: probe failed" in output
        assert "check-access" not in output

    def test_configured_quiet_behaves_like_the_flag(self, monkeypatch, tmp_path):
        output = self._run(monkeypatch, tmp_path, configured_quiet=True).output
        assert "A1000: probe failed" in output
        assert "check-access" not in output


class TestExitStatus:
    """Commands reported failure by printing, so every run exited 0."""

    def _invoke(self, args, monkeypatch):
        return CliRunner().invoke(cli, args)

    def test_a_failed_command_exits_nonzero(self, monkeypatch):
        monkeypatch.setattr(A1000ReportService, "get_report", lambda self, h, fmt="json": None)
        result = self._invoke(["a1000", "report", "a" * 64], monkeypatch)
        assert result.exit_code == 1

    def test_a_rejected_argument_exits_nonzero(self, monkeypatch):
        result = self._invoke(["a1000", "report", "notahash"], monkeypatch)
        assert result.exit_code == 1

    def test_a_successful_command_still_exits_zero(self, monkeypatch):
        monkeypatch.setattr(
            A1000ReportService,
            "get_report",
            lambda self, h, fmt="json": {"results": [{"md5": "abc"}]},
        )
        result = self._invoke(["a1000", "report", "a" * 64], monkeypatch)
        assert result.exit_code == 0

    def test_a_warning_alone_does_not_fail_the_run(self, monkeypatch):
        """Empty results are not an error."""
        monkeypatch.setattr(A1000SampleService, "list_samples", lambda self, limit=100: [])
        result = self._invoke(["a1000", "list"], monkeypatch)
        assert result.exit_code == 0

    def test_no_context_is_nothing_to_grade_rather_than_a_crash(self):
        """A group callback that returns early leaves ``ctx.obj`` empty.

        None does today, which is why this reads the callback directly
        rather than arranging one. The invariant was written down and
        enforced by nothing: the first early ``return`` added to ``cli()``
        turns every invocation's exit into ``AttributeError: 'NoneType'
        object has no attribute 'output'``, which main()'s catch-all hands
        the analyst as "Unexpected error" — a bug report about their own
        config path.
        """
        context = click.Context(cli)
        context.obj = None

        with context:
            main_module._exit_nonzero_on_failure(None)


class TestTheDiagnosticIsNotTheFailure:
    """`check-access` is what the CLI points you at when a service is
    unreachable, so a red line per service is the command working.

    Nothing reachable at all used to be treated as a different statement
    and exited 1. That made the exit status a count: one appliance down
    out of two printed red and exited 0, both down exited 1, and two
    unconfigured services - nothing measured the analyst did not already
    know - exited 1 as well. It is the same kind of thing as the red line,
    so it is reported the same way and the run still succeeds; the count
    a script branches on is on stdout, where `config show -a` (which
    renders the same measurement through the same helper, and has always
    exited 0) puts it too. A non-zero exit now means the check could not
    be made at all.
    """

    def _check_access(self, monkeypatch, status):
        from rl_cli.services.availability import APIAvailabilityChecker

        probe = {
            "status": status,
            "message": "probe failed: appliance did not answer",
            "credentials_configured": True,
            "api_accessible": False,
        }
        monkeypatch.setattr(
            APIAvailabilityChecker,
            "check_all",
            lambda self, force=False: {
                "timestamp": "2026-01-01T00:00:00",
                "titanium_cloud": probe,
                "a1000": probe,
                "summary": {"services_available": 0, "services_total": 2},
            },
        )
        # Through the root group, which is where the exit status is decided.
        return CliRunner().invoke(cli, ["config", "check-access"])

    def test_an_unreachable_appliance_is_reported_and_the_run_succeeds(self, monkeypatch):
        result = self._check_access(monkeypatch, ServiceStatus.ERROR.value)

        assert "probe failed" in result.output
        assert result.exit_code == 0, result.output

    def test_missing_credentials_are_measured_the_same_way(self, monkeypatch):
        result = self._check_access(monkeypatch, ServiceStatus.UNAVAILABLE.value)

        assert result.exit_code == 0, result.output

    def test_one_service_up_is_a_healthy_check(self, monkeypatch):
        """A red line for the other service must not fail the command."""
        from rl_cli.services.availability import APIAvailabilityChecker

        monkeypatch.setattr(
            APIAvailabilityChecker,
            "check_all",
            lambda self, force=False: {
                "timestamp": "2026-01-01T00:00:00",
                "titanium_cloud": {
                    "status": ServiceStatus.ERROR.value,
                    "message": "probe failed",
                    "credentials_configured": True,
                    "api_accessible": False,
                },
                "a1000": {
                    "status": ServiceStatus.AVAILABLE.value,
                    "message": "ok",
                    "credentials_configured": True,
                    "api_accessible": True,
                },
                "summary": {"services_available": 1, "services_total": 2},
            },
        )
        result = CliRunner().invoke(cli, ["config", "check-access"])

        assert "probe failed" in result.output
        assert result.exit_code == 0, result.output


class TestTableColumnsCarryTheirOwnValues:
    """Rows need not share a key set, and Rich never complains: a short
    row shifted every later value one column left and appended a
    blank-headed column to take the overflow.
    """

    def _rendered_row(self, data, marker):
        stream = io.StringIO()
        RichOutput(console=Console(file=stream, width=200)).table(data)
        line = next(row for row in stream.getvalue().splitlines() if marker in row)
        return [cell.strip() for cell in re.split(r"[│┃]", line) if cell.strip()]

    def test_a_later_row_with_an_extra_key_stays_under_its_own_headings(self):
        data = [
            {"sha1": "a" * 40, "threatname": "Win32.X"},
            {"sha1": "b" * 40, "filesize": 4096, "threatname": "Win32.Y"},
        ]

        assert self._rendered_row(data, "Threatname") == ["Sha1", "Threatname", "Filesize"]
        assert self._rendered_row(data, "Win32.Y") == ["b" * 40, "Win32.Y", "4096"]

    def test_a_row_missing_a_key_leaves_that_cell_empty(self):
        data = [
            {"sha1": "a" * 40, "threatname": "Win32.X"},
            {"sha1": "b" * 40, "filesize": 4096, "threatname": "Win32.Y"},
        ]

        assert self._rendered_row(data, "Win32.X") == ["a" * 40, "Win32.X"]


class TestColourIsAConfiguredSetting:
    """`color` was documented in both example files and read by nothing.

    Rich disables colour on a non-TTY whatever it is told, so a captured
    run cannot show the difference; what has to hold is that the setting
    reaches both consoles - warnings go to the stderr one, and colouring
    only stdout would honour the setting for half the output.
    """

    def _consoles(self, tmp_path, color: bool) -> list[Console]:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"default:\n  output:\n    color: {str(color).lower()}\n")
        seen: list[Console] = []
        original = Console.__init__

        def record(self, *args, **kwargs):
            original(self, *args, **kwargs)
            seen.append(self)

        monkey = pytest.MonkeyPatch()
        monkey.setattr(Console, "__init__", record)
        try:
            result = CliRunner().invoke(cli, ["-c", str(cfg), "config", "show"])
        finally:
            monkey.undo()
        assert result.exit_code == 0, result.output
        return seen

    def test_colour_off_reaches_every_console(self, tmp_path):
        consoles = self._consoles(tmp_path, color=False)
        assert consoles and all(c.no_color for c in consoles)

    def test_colour_on_leaves_rich_to_decide(self, tmp_path):
        consoles = self._consoles(tmp_path, color=True)
        assert consoles and not any(c.no_color for c in consoles)


class TestSavedReportsGetTheSameCareAsSamples:
    """A report is the analysis of a sample: extracted strings, C2
    indicators, the customer's own file names. Samples were written 0600
    and reports 0644, readable by every other local user, and a failed
    write was announced as a CLI bug with the fetched report dropped.
    """

    def _saved(self, monkeypatch, tmp_path, name="report.json"):
        monkeypatch.setattr(
            A1000ReportService,
            "get_report",
            lambda self, h, fmt="json": {"results": [{"md5": "a"}]},
        )
        target = tmp_path / name
        result = CliRunner().invoke(
            cli, ["a1000", "report", "a" * 64, "--output-file", str(target)]
        )
        return result, target

    @pytest.mark.posix_only
    def test_a_saved_report_is_owner_only(self, monkeypatch, tmp_path):
        result, target = self._saved(monkeypatch, tmp_path)

        assert result.exit_code == 0, result.output
        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    @pytest.mark.posix_only
    def test_a_saved_dynamic_report_is_owner_only(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            A1000ReportService, "download_dynamic_report", lambda self, h, fmt="pdf": b"%PDF-1.4"
        )
        target = tmp_path / "dynamic.pdf"

        result = CliRunner().invoke(
            cli, ["a1000", "dynamic-report", "b" * 40, "--output-file", str(target)]
        )

        assert result.exit_code == 0, result.output
        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_an_unwritable_destination_names_itself(self, monkeypatch, tmp_path):
        result, target = self._saved(monkeypatch, tmp_path, name="missing/report.json")

        assert "could not be saved" in result.output
        assert "Unexpected error" not in result.output
        assert result.exit_code == 1
        assert not target.exists()

    @pytest.mark.posix_only
    def test_a_symlink_at_the_destination_is_not_followed(self, monkeypatch, tmp_path):
        """--output-file pointed at a symlink wrote the report through it,
        and a refused write must leave no ``.part`` wearing a real name.
        """
        elsewhere = tmp_path / "elsewhere.json"
        (tmp_path / "report.json").symlink_to(elsewhere)

        result, _target = self._saved(monkeypatch, tmp_path)

        assert not elsewhere.exists()
        assert not (tmp_path / "report.json.part").exists()
        assert "could not be saved" in result.output
        assert result.exit_code == 1


class TestUnmappedStatusesAreCapped:
    """428 and 507 went through get_report's own error path, which printed
    the status and then the whole response body.
    """

    def test_a_507_reports_the_status_and_caps_the_body(self, monkeypatch):
        response = MagicMock(status_code=507, text="x" * 5000)
        response.json.side_effect = ValueError("not json")
        client = MagicMock()
        client.get_detailed_report_v2.return_value = response
        monkeypatch.setattr(
            A1000Session, "_open", lambda session: setattr(session, "client", client)
        )

        result = CliRunner().invoke(cli, ["a1000", "report", "a" * 64])

        assert "HTTP 507" in result.output
        assert result.output.count("x") < 300
        assert result.exit_code == 1


class TestCtrlCIsNotAPlainFailure:
    """Click turned Ctrl-C into a bare "Aborted!" and exit 1, so a script
    could not tell an interrupted download from a command that failed.
    """

    @pytest.fixture(autouse=True)
    def _restore_sigint(self):
        original = signal.getsignal(signal.SIGINT)
        yield
        signal.signal(signal.SIGINT, original)

    def test_a_real_sigint_exits_130(self, monkeypatch, capsys):
        monkeypatch.setattr(main_module, "cli", lambda: os.kill(os.getpid(), signal.SIGINT))

        with pytest.raises(SystemExit) as exit_info:
            main_module.main()

        assert exit_info.value.code == 130
        assert "Interrupted" in capsys.readouterr().err

    def test_a_keyboard_interrupt_reaching_main_exits_130(self, monkeypatch):
        def interrupt():
            raise KeyboardInterrupt

        monkeypatch.setattr(main_module, "cli", interrupt)

        with pytest.raises(SystemExit) as exit_info:
            main_module.main()

        assert exit_info.value.code == 130

    def test_an_unexpected_failure_still_exits_1(self, monkeypatch):
        def explode():
            raise RuntimeError("boom")

        monkeypatch.setattr(main_module, "cli", explode)

        with pytest.raises(SystemExit) as exit_info:
            main_module.main()

        assert exit_info.value.code == 1
