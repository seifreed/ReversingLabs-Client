"""``rl-cli config`` — what it shows, and which file it writes."""

import ast
import io
import json
from dataclasses import replace
from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner
from rich.console import Console

from rl_cli.cli.commands import config as config_module
from rl_cli.cli.commands.config import config
from rl_cli.cli.main import cli
from rl_cli.config import Settings, clear_settings_cache
from rl_cli.models.availability import availability_block, availability_payload
from rl_cli.render.formatters.config_report import print_service_lines
from rl_cli.render.output import OutputFormat, OutputFormatter, RichOutput
from rl_cli.services.availability import APIAvailabilityChecker, ServiceStatus
from tests.cli_support import flat, invoke, make_context


@pytest.fixture
def ctx_obj(tmp_path):
    return make_context(tmp_path)


def _availability(
    titanium_cloud: str = ServiceStatus.AVAILABLE.value,
    a1000: str = ServiceStatus.ERROR.value,
) -> dict:
    """What ``APIAvailabilityChecker.check_all`` answers, as the commands read it."""
    statuses = (titanium_cloud, a1000)
    return {
        "timestamp": "2026-01-01T00:00:00",
        "titanium_cloud": {
            "status": titanium_cloud,
            "message": f"TitaniumCloud probe: {titanium_cloud}",
            "credentials_configured": True,
            "api_accessible": titanium_cloud == ServiceStatus.AVAILABLE.value,
        },
        "a1000": {
            "status": a1000,
            "message": f"A1000 probe: {a1000}",
            "credentials_configured": True,
            "api_accessible": a1000 == ServiceStatus.AVAILABLE.value,
        },
        "summary": {
            "services_available": sum(
                status == ServiceStatus.AVAILABLE.value for status in statuses
            ),
            "services_total": 2,
        },
    }


class TestConfigCommands:
    def test_show_displays_profile(self, ctx_obj):
        result = invoke(config, ["show"], ctx_obj)
        assert result.exit_code == 0
        assert "default" in result.output

    def test_check_access_detailed_shows_the_probe_and_not_a_catalogue(self, ctx_obj, monkeypatch):
        """Rendered from a cache the previous version wrote, which lives 24 h.

        Its ``available_methods`` are the twelve names that used to be
        printed as "Available" whenever one Check Status call answered.
        """
        monkeypatch.setattr(
            APIAvailabilityChecker,
            "check_all",
            lambda self, force=False: {
                "timestamp": "2026-01-01T00:00:00",
                "titanium_cloud": {
                    "status": "unavailable",
                    "message": "No TitaniumCloud credentials configured",
                    "credentials_configured": False,
                    "api_accessible": False,
                    "available_apis": [],
                    "test_results": {},
                },
                "a1000": {
                    "status": "available",
                    "message": "A1000 API is available (token authentication)",
                    "credentials_configured": True,
                    "api_accessible": True,
                    "available_methods": ["advanced_search_v3", "get_summary_report_v2"],
                    "test_results": {},
                },
                "summary": {"services_available": 1, "services_total": 2},
            },
        )

        result = invoke(config, ["check-access", "--detailed"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "Endpoint answered: True" in result.output
        assert "Credentials configured: False" in result.output
        assert "advanced_search_v3" not in result.output

    def test_list_profiles_without_config_file(self, ctx_obj, tmp_path):
        ctx_obj = replace(
            ctx_obj,
            settings=Settings(
                cache_dir=tmp_path / "cache",
                config_dir=tmp_path / "config",
                config_file=tmp_path / "missing.yaml",
            ),
        )
        result = invoke(config, ["list-profiles"], ctx_obj)
        assert result.exit_code == 0
        assert "No configuration file" in result.output

    def test_list_profiles_reads_the_loaded_config_file(self, ctx_obj, tmp_path):
        config_file = tmp_path / "elsewhere.yaml"
        config_file.write_text("default:\n  output:\n    format: json\nstaging:\n  output: {}\n")
        ctx_obj = replace(
            ctx_obj,
            settings=Settings(
                cache_dir=tmp_path / "cache",
                config_dir=tmp_path / "config",
                config_file=config_file,
            ),
        )
        result = invoke(config, ["list-profiles"], ctx_obj)
        assert result.exit_code == 0
        assert "staging" in result.output

    def test_list_profiles_says_so_when_the_file_holds_none(self, ctx_obj, tmp_path):
        """A file that exists and is empty is not the same as no file."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        ctx_obj = replace(
            ctx_obj,
            settings=Settings(
                cache_dir=tmp_path / "cache",
                config_dir=tmp_path / "config",
                config_file=config_file,
            ),
        )

        result = invoke(config, ["list-profiles"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "No profiles found" in flat(result)

    def test_check_access_force_clears_the_cache_before_probing(self, ctx_obj, monkeypatch):
        """``--force`` is the remedy the cache's own 24 h staleness needs."""
        cleared = []

        def record(self) -> None:
            cleared.append(self)

        monkeypatch.setattr(APIAvailabilityChecker, "clear_cache", record)
        monkeypatch.setattr(
            APIAvailabilityChecker, "check_all", lambda self, force=False: _availability()
        )

        result = invoke(config, ["check-access", "--force"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert len(cleared) == 1
        assert "Cache cleared" in flat(result)


class TestTheFormatDecisionIsDisplaysAndNotEachCommandsOwn:
    """Two commands here hand-rolled ``if format == RICH``, ``display``'s one job.

    Every other command in the CLI routes the choice between "draw it" and
    "dump it" through ``cli.context.display``, so a third spelling of it is
    a place the global ``-o`` can be forgotten. Both renderings are pinned
    on both sides of the branch, since routing them is only safe if it
    changed neither.
    """

    def _profiled(self, ctx_obj, tmp_path, output_format):
        config_file = tmp_path / "profiles.yaml"
        config_file.write_text("default:\n  output: {}\nstaging:\n  output: {}\n")
        return replace(
            ctx_obj,
            settings=Settings(
                cache_dir=tmp_path / "cache",
                config_dir=tmp_path / "config",
                config_file=config_file,
            ),
            formatter=OutputFormatter(output_format, ctx_obj.output),
        )

    def test_list_profiles_marks_the_current_one_on_a_terminal(self, ctx_obj, tmp_path):
        obj = self._profiled(ctx_obj, tmp_path, OutputFormat.RICH)

        result = invoke(config, ["list-profiles"], obj)

        assert result.exit_code == 0, result.output
        assert "default (current)" in flat(result)
        # The marked names are chatter, so stdout stays empty rather than
        # carrying a list the marker has been drawn into.
        assert result.stdout == ""

    def test_list_profiles_dumps_the_bare_names_under_a_format(self, ctx_obj, tmp_path):
        obj = self._profiled(ctx_obj, tmp_path, OutputFormat.JSON)

        result = invoke(config, ["list-profiles"], obj)

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == ["default", "staging"]

    def test_check_access_panels_the_summary_on_a_terminal(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(
            APIAvailabilityChecker, "check_all", lambda self, force=False: _availability()
        )
        obj = replace(ctx_obj, formatter=OutputFormatter(OutputFormat.RICH, ctx_obj.output))

        result = invoke(config, ["check-access"], obj)

        assert result.exit_code == 0, result.output
        assert "API Availability Status" in flat(result)
        assert "Services Available: 1/2" in flat(result)
        assert '"summary"' not in result.stdout

    def test_check_access_dumps_the_measurement_under_a_format(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(
            APIAvailabilityChecker, "check_all", lambda self, force=False: _availability()
        )
        obj = replace(ctx_obj, formatter=OutputFormatter(OutputFormat.JSON, ctx_obj.output))

        result = invoke(config, ["check-access"], obj)

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["summary"] == {
            "services_available": 1,
            "services_total": 2,
        }


class TestConfigWritesToTheActiveFile:
    """init/save must not write to a file the session is not reading."""

    def _settings(self, tmp_path, config_file):
        return Settings(
            cache_dir=tmp_path / "cache",
            config_dir=tmp_path / "configdir",
            config_file=config_file,
        )

    def test_init_saves_to_the_loaded_config_file(self, ctx_obj, tmp_path):
        target = tmp_path / "mine.yaml"
        target.write_text("default:\n  a1000:\n    host: https://original.example\n")
        ctx_obj = replace(ctx_obj, settings=self._settings(tmp_path, target))

        result = invoke(config, ["init"], ctx_obj, input="n\nn\nn\ny\n")

        assert result.exit_code == 0, result.output
        assert not (tmp_path / "configdir" / "config.yaml").exists()
        # save_config rewrites the whole profile, so the file it touched
        # gains the sections a bare host-only file did not have.
        assert "output:" in target.read_text(encoding="utf-8")

    def test_save_without_path_uses_the_loaded_config_file(self, ctx_obj, tmp_path):
        target = tmp_path / "mine.yaml"
        target.write_text("default: {}\n")
        ctx_obj = replace(ctx_obj, settings=self._settings(tmp_path, target))

        result = invoke(config, ["save"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert not (tmp_path / "configdir" / "config.yaml").exists()
        assert "a1000" in target.read_text(encoding="utf-8")


class TestTheWizardIsNotDefinedInTheCommandModule:
    """36 lines of nested confirms and field-by-field mutation, in a click module.

    It knew the full shape of every settings section, so adding a section
    meant editing a CLI module — and it filled in the same ``Settings``
    instance ``cli/main.py`` warns is what ``config save`` serialises, so a
    mistake here lands in the user's 0600 credentials file. The question
    sequence belongs to ``rl_cli/config/wizard.py``; this module supplies
    the terminal and nothing else.
    """

    _SOURCE = Path(config_module.__file__).read_text(encoding="utf-8")

    def test_no_settings_question_is_worded_here(self):
        assert "Configure " not in self._SOURCE, (
            "a settings section's question belongs in rl_cli/config/wizard.py"
        )

    def test_no_settings_field_is_filled_in_here(self):
        """An attribute assignment in this module is a write to that object."""
        written = sorted(
            ast.unparse(target)
            for node in ast.walk(ast.parse(self._SOURCE))
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute)
        )

        assert written == [], f"{written}: the wizard fills the Settings in, not the command"


class TestInitAsksForTheCredentialsAtTheTerminal:
    """`config init` is the one command that reads secrets off a terminal.

    Its whole credential body was uncovered: the only ``init`` test
    answered "no" to every section, so nothing asserted that what is typed
    at "Password" reaches the password field rather than the username one,
    or that it is asked for without being echoed. Both are answered here
    from the file the command wrote, which is the artefact that matters —
    the same 0600 file every later run reads its credentials back from.

    The answers are positional, so these are also the pin on the question
    order: an inserted or dropped question shifts every value after it.
    """

    def _invoke_init(self, ctx_obj, tmp_path, answers):
        target = tmp_path / "mine.yaml"
        ctx_obj = replace(
            ctx_obj,
            settings=Settings(
                cache_dir=tmp_path / "cache",
                config_dir=tmp_path / "configdir",
                config_file=target,
            ),
        )
        result = invoke(config, ["init"], ctx_obj, input="\n".join([*answers, ""]))
        assert result.exit_code == 0, result.output
        saved: dict = yaml.safe_load(target.read_text(encoding="utf-8"))["default"]
        return result, saved

    def test_every_typed_credential_lands_in_its_own_field(self, ctx_obj, tmp_path):
        _, saved = self._invoke_init(
            ctx_obj,
            tmp_path,
            [
                "y",  # Configure TitaniumCloud API?
                "https://tc.example",  # TitaniumCloud host
                "tc-user",  # Username
                "tc-secret",  # Password
                "y",  # Configure A1000 platform?
                "https://a1000.example",  # A1000 host
                "n",  # Use token authentication?
                "a1000-user",  # Username
                "a1000-secret",  # Password
                "y",  # Configure output settings?
                "json",  # Default output format
                "n",  # Enable colored output?
                "y",  # Save configuration?
            ],
        )

        assert saved["titanium_cloud"] == {
            **saved["titanium_cloud"],
            "host": "https://tc.example",
            "username": "tc-user",
            "password": "tc-secret",
        }
        assert saved["a1000"] == {
            **saved["a1000"],
            "host": "https://a1000.example",
            "username": "a1000-user",
            "password": "a1000-secret",
        }
        assert "token" not in saved["a1000"], "the token branch was declined"
        assert saved["output"]["format"] == "json"
        assert saved["output"]["color"] is False

    def test_the_token_branch_writes_a_token_and_asks_for_no_password(self, ctx_obj, tmp_path):
        _, saved = self._invoke_init(
            ctx_obj,
            tmp_path,
            [
                "n",  # Configure TitaniumCloud API?
                "y",  # Configure A1000 platform?
                "https://a1000.example",  # A1000 host
                "y",  # Use token authentication?
                "a1000-token",  # API Token
                "n",  # Configure output settings?
                "y",  # Save configuration?
            ],
        )

        assert saved["a1000"]["token"] == "a1000-token"
        assert "password" not in saved["a1000"]
        assert "username" not in saved["a1000"]

    def test_a_declined_run_writes_nothing(self, ctx_obj, tmp_path):
        target = tmp_path / "mine.yaml"
        ctx_obj = replace(
            ctx_obj,
            settings=Settings(
                cache_dir=tmp_path / "cache",
                config_dir=tmp_path / "configdir",
                config_file=target,
            ),
        )

        result = invoke(config, ["init"], ctx_obj, input="n\nn\nn\nn\n")

        assert result.exit_code == 0, result.output
        assert not target.exists()
        assert "not saved" in flat(result)

    def test_a_secret_is_asked_for_without_being_echoed(self, monkeypatch):
        """``hide_input`` is the whole reason this adapter exists as a class.

        Asserted against click's own keyword rather than against the
        rendered prompt, because a hidden prompt renders exactly like a
        visible one — the difference is only in what the terminal does with
        the keystrokes, which no captured output can show.
        """
        asked = {}

        def record(text, **kwargs):
            asked[text] = kwargs
            return "typed"

        monkeypatch.setattr(click, "prompt", record)
        prompter = config_module._ClickPrompter()

        assert prompter.ask("Password", secret=True) == "typed"
        assert prompter.ask("Username", default="") == "typed"
        assert asked["Password"]["hide_input"] is True
        assert asked["Username"]["hide_input"] is False


class TestSaveWritesTheConfigurationNotTheInvocation:
    """The run's own flags are not settings, and must not end up in the file.

    ``-v``/``-q``/``-o`` used to be applied by assigning to
    ``settings.output``, which is the object ``save`` serialises — so
    ``rl-cli -o json -q config save`` wrote ``format: json`` and
    ``quiet: true`` into the user's config and made them permanent.
    """

    def _saved_output_section(self, tmp_path, *root_args):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"default": {"output": {"format": "rich", "quiet": False, "verbose": False}}})
        )
        clear_settings_cache()
        result = CliRunner().invoke(
            cli, ["--config", str(config_file), *root_args, "config", "save"]
        )
        assert result.exit_code == 0, result.output
        section: dict = yaml.safe_load(config_file.read_text(encoding="utf-8"))["default"]["output"]
        return section

    def test_a_plain_save_round_trips_the_configured_values(self, tmp_path):
        assert self._saved_output_section(tmp_path) == {
            "format": "rich",
            "quiet": False,
            "verbose": False,
            "color": True,
        }

    @pytest.mark.parametrize(
        ("flags", "key"),
        [
            (["-o", "json"], "format"),
            (["-q"], "quiet"),
            (["-v"], "verbose"),
        ],
    )
    def test_a_run_flag_is_not_written_to_the_file(self, tmp_path, flags, key):
        before = self._saved_output_section(tmp_path)[key]

        assert self._saved_output_section(tmp_path, *flags)[key] == before


class TestCheckAccessExitsOnWhetherItCouldMeasure:
    """The exit status used to be a count, and the count was the verdict.

    ``check-access`` is what the CLI tells an analyst to run when a
    service is unreachable, so what it measured is its answer rather than
    its failure — which is why the per-service lines print through
    ``problem`` and leave the run's fate alone. The summary below them did
    not: it exited 1 when ``services_available`` was 0, so one appliance
    down out of two printed a red line and exited 0 while two down exited
    1, and two merely-unconfigured services — nothing measured that the
    analyst did not already know — exited 1 as well. Under ``set -e`` that
    aborted the troubleshooting script at the diagnostic.

    ``config show -a`` renders the same measurement through the same
    helper and has always exited 0. The two agree now.
    """

    def _measured(self, monkeypatch, statuses):
        monkeypatch.setattr(
            APIAvailabilityChecker, "check_all", lambda self, force=False: _availability(*statuses)
        )

    @pytest.mark.parametrize(
        "statuses",
        [
            (ServiceStatus.ERROR.value, ServiceStatus.ERROR.value),
            (ServiceStatus.AVAILABLE.value, ServiceStatus.ERROR.value),
            (ServiceStatus.UNAVAILABLE.value, ServiceStatus.UNAVAILABLE.value),
            (ServiceStatus.AVAILABLE.value, ServiceStatus.AVAILABLE.value),
        ],
    )
    def test_every_measurement_it_managed_to_make_exits_zero(self, ctx_obj, monkeypatch, statuses):
        self._measured(monkeypatch, statuses)

        result = invoke(config, ["check-access"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert not ctx_obj.output.status.failed
        assert "A1000 probe" in flat(result), "it still says what it measured"

    def test_nothing_reachable_still_says_so_and_says_what_to_do(self, ctx_obj, monkeypatch):
        """The guidance is what the exit status used to duplicate."""
        self._measured(monkeypatch, (ServiceStatus.ERROR.value, ServiceStatus.ERROR.value))

        result = invoke(config, ["check-access"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "No services available" in flat(result)
        assert "rl-cli config init" in flat(result)

    def test_the_count_a_script_branches_on_is_on_stdout(self, ctx_obj, monkeypatch):
        """Two values of an exit status cannot carry it; the document can."""
        self._measured(monkeypatch, (ServiceStatus.AVAILABLE.value, ServiceStatus.ERROR.value))

        result = invoke(config, ["check-access"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["summary"]["services_available"] == 1

    def test_a_check_that_could_not_be_made_is_still_a_failure(self, tmp_path):
        """The other half of the contract: exit 1 means the check did not happen.

        An unusable config file is the case the user is told about — the
        run cannot say which appliance it would have probed, so there is
        no measurement to report.
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text("default:\n  a1000: [unclosed\n")
        clear_settings_cache()

        result = CliRunner().invoke(cli, ["--config", str(config_file), "config", "check-access"])

        assert result.exit_code == 1, result.output
        assert "not valid YAML" in flat(result)


class TestShowReportsAvailabilityTheWayCheckAccessDoes:
    """``show -a`` had its own copy of the per-service summary.

    It answered "✓ Available" where the probe had a measurement to report,
    and painted a service whose probe errored in the same yellow as one
    that is merely unconfigured. Both facts are ``check-access``'s to
    render, so both are rendered by the same helper.
    """

    def _probe(self, monkeypatch, status, message):
        monkeypatch.setattr(
            APIAvailabilityChecker,
            "check_all",
            lambda self, force=False: {
                "timestamp": "2026-01-01T00:00:00",
                "titanium_cloud": {
                    "status": status,
                    "message": message,
                    "credentials_configured": True,
                    "api_accessible": False,
                },
                "a1000": {
                    "status": status,
                    "message": message,
                    "credentials_configured": True,
                    "api_accessible": False,
                },
                "summary": {"services_available": 0, "services_total": 2},
            },
        )

    def test_the_summary_reports_what_the_probe_measured(self, ctx_obj):
        result = invoke(config, ["show", "-a"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["api_availability"]["a1000"]["status"] == "available"
        assert "A1000: probe stubbed in tests" in flat(result)
        assert "✓ Available" not in flat(result), "the wording is the probe's, not a second one"

    def test_a_service_that_could_not_be_reached_does_not_fail_the_command(
        self, ctx_obj, monkeypatch
    ):
        """``show`` reports configuration; an unreachable service is not its failure."""
        self._probe(monkeypatch, ServiceStatus.ERROR.value, "probe failed: no answer")

        result = invoke(config, ["show", "-a"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "A1000: probe failed: no answer" in flat(result)
        assert not ctx_obj.output.status.failed

    def test_without_the_flag_nothing_is_probed(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(
            APIAvailabilityChecker,
            "check_all",
            lambda self, force=False: pytest.fail("`config show` probed without --check-access"),
        )

        result = invoke(config, ["show"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "api_availability" not in result.stdout


class TestTheAvailabilityReportIsNotStrandedInTheCliLayer:
    """The report's shape and its drawing were both defined in this command module.

    ``ServiceStatus`` lived in ``services``, which ``render`` may not
    import, so the grading — and with it the whole ``-o json`` document and
    the per-service lines — was written beside the click command instead of
    beside the panel that draws it. The consequence was that a second
    consumer (a ``status`` command, a library caller, a SARIF exporter)
    could only get the document by importing ``rl_cli.cli``.

    The fix is the type's location, not a second copy of the strings: the
    statuses are a rule both sides of the boundary agree on, so they are in
    ``models``, which every layer may read.
    """

    # Read off the source rather than off the imported module, because a
    # helper *re-imported* into the command module would still be the CLI's
    # to change; what must be gone is the definition.
    _SOURCE = Path(config_module.__file__).read_text(encoding="utf-8")
    _STRANDED = ("availability", "service_lines", "report", "prompt")

    def test_the_command_module_defines_no_report_of_its_own(self):
        helpers = {
            node.name
            for node in ast.parse(self._SOURCE).body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_")
        }
        stranded = sorted(name for name in helpers if any(word in name for word in self._STRANDED))

        assert stranded == [], (
            f"{stranded} are the report, not the command: the document belongs in "
            "rl_cli/models/availability.py and the drawing in "
            "rl_cli/render/formatters/config_report.py"
        )

    def test_the_document_and_the_drawing_are_reachable_without_the_cli(self):
        """Both halves import from layers ``test_architecture`` forbids to reach ``cli``.

        Asserted through their new homes rather than by inspecting imports:
        ``tests/test_architecture.py`` already fails any ``models`` or
        ``render`` module that names ``rl_cli.cli``, so naming the module a
        caller imports is the whole claim.
        """
        assert availability_block.__module__ == "rl_cli.models.availability"
        assert availability_payload.__module__ == "rl_cli.models.availability"
        assert print_service_lines.__module__ == "rl_cli.render.formatters.config_report"


class TestTheAvailabilityReportTheseCommandsPrint:
    """The document `-o json` prints, and the lines the terminal gets beside it.

    The statuses are graded in ``models``, so the document and the lines
    read one answer to "is this good news"; the drawing of that answer sits
    beside the panel.
    """

    @staticmethod
    def _lines(availability: dict, **kwargs) -> str:
        """The status console's half of a probe report, which is where it goes."""
        buffer = io.StringIO()
        output = RichOutput(Console(file=io.StringIO(), width=100), Console(file=buffer, width=100))
        print_service_lines(availability, output=output, **kwargs)
        return buffer.getvalue()

    @pytest.mark.parametrize(
        ("status", "glyph"),
        [
            (ServiceStatus.AVAILABLE, "✓"),
            (ServiceStatus.UNAVAILABLE, "⚠"),
            (ServiceStatus.ERROR, "✗"),
        ],
    )
    def test_every_status_is_announced_as_what_it_is(self, status, glyph):
        """A probe that could not reach the appliance is still an answer.

        It prints through ``problem``, which draws ``error``'s red line and
        leaves the exit status alone.
        """
        rendered = self._lines(_availability(status.value, status.value))

        assert rendered.count(glyph) == 2, rendered

    def test_a_failed_probe_does_not_fail_the_run(self):
        buffer = io.StringIO()
        output = RichOutput(Console(file=io.StringIO()), Console(file=buffer, width=100))

        print_service_lines(
            _availability(ServiceStatus.ERROR.value, ServiceStatus.ERROR.value), output=output
        )

        assert "✗" in buffer.getvalue()
        assert not output.status.failed

    def test_the_plain_report_states_only_what_the_probe_said(self):
        rendered = self._lines(_availability())

        assert "TitaniumCloud: TitaniumCloud probe: available" in rendered
        assert "A1000: A1000 probe: error" in rendered
        assert "Credentials configured" not in rendered

    def test_the_detailed_report_adds_the_two_facts_measured(self):
        rendered = self._lines(_availability(), detailed=True)

        assert "Credentials configured: True" in rendered
        assert "Endpoint answered: False" in rendered

    def test_the_caller_nests_the_lines_under_its_own_heading(self):
        """``config show -a`` lists them under "API Status Summary:"."""
        rendered = self._lines(_availability(), indent="  ")

        assert "✓   TitaniumCloud: TitaniumCloud probe: available" in rendered

    def test_the_block_carries_the_verdict_and_the_reason_and_no_more(self):
        """``config show`` puts this on stdout next to the redacted profile."""
        assert availability_block(_availability()) == {
            "titanium_cloud": {"status": "available", "message": "TitaniumCloud probe: available"},
            "a1000": {"status": "error", "message": "A1000 probe: error"},
        }

    def test_the_payload_is_the_block_plus_what_the_panel_reads(self):
        """The whole of `check-access`'s stdout, which is what a script branches on."""
        assert availability_payload(_availability()) == {
            "titanium_cloud": {"status": "available", "message": "TitaniumCloud probe: available"},
            "a1000": {"status": "error", "message": "A1000 probe: error"},
            "timestamp": "2026-01-01T00:00:00",
            "summary": {"services_available": 1, "services_total": 2},
        }
