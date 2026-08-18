"""Tests for the ``config`` commands that handle plaintext secrets.

``create-profile`` writes ``a1000.token`` and ``titanium_cloud.password``
in the clear, and it is the only caller of :func:`write_private_yaml`
outside ``Settings.save_config`` — so the permission contract has to be
pinned here as well, not only where ``save`` is tested. The overwrite
prompt is the other half: without it the command silently replaces a
profile the user is still using.

``show`` is the same secrets going the other way, to stdout, where the
0600 the writers are careful about does not apply.
"""

from __future__ import annotations

import json
import stat

import pytest
import yaml
from click.testing import CliRunner

from rl_cli.cli.commands.config import config
from rl_cli.cli.context import CliContext
from rl_cli.config import Settings
from rl_cli.render.output import OutputFormat, OutputFormatter, RichOutput
from rl_cli.services.a1000 import A1000Session

EXISTING = "default:\n  a1000:\n    host: https://original.example\n"


def _ctx_obj(tmp_path, config_file, fmt: OutputFormat = OutputFormat.JSON) -> CliContext:
    settings = Settings(
        cache_dir=tmp_path / "cache",
        config_dir=tmp_path / "configdir",
        config_file=config_file,
    )
    settings.a1000.token = "supersecret-token"
    settings.titanium_cloud.password = "supersecret-password"
    return CliContext(
        settings=settings,
        output=RichOutput(),
        formatter=OutputFormatter(fmt),
        session=A1000Session(settings),
        warn_if_unavailable=None,
    )


def _invoke(tmp_path, config_file, args, **kwargs):
    return CliRunner().invoke(config, args, obj=_ctx_obj(tmp_path, config_file), **kwargs)


@pytest.fixture
def config_file(tmp_path):
    return tmp_path / "profiles.yaml"


CANARY_TC_PASSWORD = "CANARY_TC_PASSWORD_1234"
CANARY_A1000_PASSWORD = "CANARY_A1_PASSWORD_5678"
CANARY_TOKEN = "CANARY_A1_TOKEN_9012"


def _show(tmp_path, config_file, fmt: OutputFormat = OutputFormat.JSON):
    """Run ``config show`` over a profile whose secrets are all canaries."""
    obj = _ctx_obj(tmp_path, config_file, fmt)
    settings = obj.settings
    settings.titanium_cloud.username = "tc-user"
    settings.titanium_cloud.password = CANARY_TC_PASSWORD
    settings.a1000.username = "a1000-user"
    settings.a1000.password = CANARY_A1000_PASSWORD
    settings.a1000.token = CANARY_TOKEN
    return CliRunner().invoke(config, ["show"], obj=obj)


class TestShowDoesNotPrintTheSecrets:
    """``config show`` writes the profile to stdout, in whatever format.

    ``rl-cli -o json config show > cfg.json`` therefore copied a live
    appliance token out of a 0600 config file and into a 0644 one, and a
    terminal user pasting the output handed it over entirely.
    """

    @pytest.mark.parametrize(
        "fmt", [OutputFormat.RICH, OutputFormat.JSON, OutputFormat.YAML, OutputFormat.TOON]
    )
    def test_no_format_prints_a_secret(self, tmp_path, config_file, fmt):
        result = _show(tmp_path, config_file, fmt)

        assert result.exit_code == 0, result.output
        for canary in (CANARY_TC_PASSWORD, CANARY_A1000_PASSWORD, CANARY_TOKEN):
            assert canary not in result.output

    def test_the_keys_are_unchanged_and_say_which_secret_is_set(self, tmp_path, config_file):
        result = _show(tmp_path, config_file, OutputFormat.JSON)

        shown = json.loads(result.output[result.output.index("{") :])
        # Same shape as before: only the values are masked, and a masked
        # value is still distinguishable from an absent one.
        assert shown["a1000"]["token"] == "***9012"
        assert shown["a1000"]["password"] == "***5678"
        assert shown["titanium_cloud"]["password"] == "***1234"

    def test_what_is_not_a_secret_stays_readable(self, tmp_path, config_file):
        """Host and username are what people run this command to check."""
        result = _show(tmp_path, config_file, OutputFormat.JSON)

        shown = json.loads(result.output[result.output.index("{") :])
        assert shown["a1000"]["username"] == "a1000-user"
        assert shown["titanium_cloud"]["username"] == "tc-user"
        assert shown["a1000"]["host"] == "https://a1000.reversinglabs.com"

    def test_a_short_secret_is_not_partially_revealed(self, tmp_path, config_file):
        """Four characters of a seven-character secret is most of it.

        This test used to assert only that ``hunter2`` was absent and that
        ``***`` was present, and both of those hold for ``***nter2`` — so
        it was named for a property it could not fail on, and removing the
        length guard from ``redact_secret`` left it green while ``config
        show`` printed four of the seven characters to stdout. The whole
        redaction is asserted now, not the absence of the whole secret.
        """
        obj = _ctx_obj(tmp_path, config_file)
        obj.settings.a1000.token = "hunter2"

        result = CliRunner().invoke(config, ["show"], obj=obj)

        shown = json.loads(result.output[result.output.index("{") :])
        assert shown["a1000"]["token"] == "***"
        assert "hunter2" not in result.output


class TestProxyUserinfoIsHiddenWithOrWithoutAScheme:
    """The proxy password rides in the userinfo, scheme or no scheme.

    A proxy written ``user:pass@host:port`` -- no scheme, which the field
    stores verbatim -- put the whole value in ``partition('://')``'s first
    element and left the userinfo unsearched, so ``config show`` printed
    the password in the clear out of a 0600 file.
    """

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("http://user:pass@proxy:8080", "http://user:***@proxy:8080"),
            ("user:pass@proxy:8080", "user:***@proxy:8080"),
            ("user:p@ss@host:1", "user:***@host:1"),
            ("http://proxy:8080", "http://proxy:8080"),
            ("proxy:8080", "proxy:8080"),
            ("bob@host", "bob@host"),
        ],
    )
    def test_the_password_never_survives(self, url, expected):
        from rl_cli.models.redaction import redact_proxy

        assert redact_proxy(url) == expected


class TestTheWritersStillStoreTheRealValues:
    """Redacting for display must not redact what is written back."""

    @pytest.mark.posix_only
    def test_save_round_trips_the_real_secrets(self, tmp_path, config_file):
        obj = _ctx_obj(tmp_path, config_file)
        obj.settings.titanium_cloud.password = CANARY_TC_PASSWORD
        obj.settings.a1000.token = CANARY_TOKEN

        result = CliRunner().invoke(config, ["save"], obj=obj)

        assert result.exit_code == 0, result.output
        written = yaml.safe_load(config_file.read_text(encoding="utf-8"))["default"]
        assert written["a1000"]["token"] == CANARY_TOKEN
        assert written["titanium_cloud"]["password"] == CANARY_TC_PASSWORD
        # And the file it landed in is still the owner-only one.
        assert stat.S_IMODE(config_file.stat().st_mode) == 0o600

    def test_create_profile_round_trips_the_real_secrets(self, tmp_path, config_file):
        obj = _ctx_obj(tmp_path, config_file)
        obj.settings.titanium_cloud.password = CANARY_TC_PASSWORD
        obj.settings.a1000.token = CANARY_TOKEN

        result = CliRunner().invoke(config, ["create-profile", "staging"], obj=obj)

        assert result.exit_code == 0, result.output
        written = yaml.safe_load(config_file.read_text(encoding="utf-8"))["staging"]
        assert written["a1000"]["token"] == CANARY_TOKEN
        assert written["titanium_cloud"]["password"] == CANARY_TC_PASSWORD


class TestCreateProfileSaysWhatItStores:
    """It snapshots the settings this run resolved, not the named profile.

    Defensible, and undocumented: the one-line help read "Create a new
    configuration profile", so ``--profile staging config create-profile
    staging`` against a file with no ``staging`` looked like a clone and
    was really the defaults plus whatever the environment supplied.
    """

    def test_the_help_says_what_the_snapshot_is_of(self, tmp_path, config_file):
        result = _invoke(tmp_path, config_file, ["create-profile", "--help"])

        assert result.exit_code == 0, result.output
        help_text = " ".join(result.output.split())
        assert "not a copy" in help_text
        assert "environment" in help_text


class TestCreatedProfileIsOwnerOnly:
    """The file holds an API token in plaintext; the default umask does not."""

    @pytest.mark.posix_only
    def test_new_file_is_owner_only(self, tmp_path, config_file):
        result = _invoke(tmp_path, config_file, ["create-profile", "staging"])

        assert result.exit_code == 0, result.output
        assert stat.S_IMODE(config_file.stat().st_mode) == 0o600
        # The mode matters because this is what is inside it.
        body = config_file.read_text(encoding="utf-8")
        assert "supersecret-token" in body
        assert "supersecret-password" in body

    @pytest.mark.posix_only
    def test_existing_world_readable_file_is_tightened(self, tmp_path, config_file):
        config_file.write_text(EXISTING)
        config_file.chmod(0o644)

        result = _invoke(tmp_path, config_file, ["create-profile", "staging"])

        assert result.exit_code == 0, result.output
        assert stat.S_IMODE(config_file.stat().st_mode) == 0o600
        assert "supersecret-token" in config_file.read_text(encoding="utf-8")

    def test_other_profiles_survive_the_write(self, tmp_path, config_file):
        config_file.write_text(EXISTING)

        result = _invoke(tmp_path, config_file, ["create-profile", "staging"])

        assert result.exit_code == 0, result.output
        written = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        assert written["default"]["a1000"]["host"] == "https://original.example"
        assert written["staging"]["a1000"]["token"] == "supersecret-token"


class TestOverwritingAnExistingProfileIsConfirmed:
    """A name already in the file is someone's working profile."""

    def _with_staging(self, config_file) -> str:
        body = yaml.dump(
            {
                "default": {"a1000": {"host": "https://original.example"}},
                "staging": {"a1000": {"host": "https://staging.example", "token": "old-token"}},
            }
        )
        config_file.write_text(body)
        return body

    def test_declining_leaves_the_file_exactly_as_it_was(self, tmp_path, config_file):
        before = self._with_staging(config_file)

        result = _invoke(tmp_path, config_file, ["create-profile", "staging"], input="n\n")

        assert result.exit_code == 0, result.output
        assert "cancelled" in result.output
        assert config_file.read_text(encoding="utf-8") == before

    def test_declining_does_not_leak_the_new_secrets(self, tmp_path, config_file):
        self._with_staging(config_file)

        _invoke(tmp_path, config_file, ["create-profile", "staging"], input="n\n")

        assert "supersecret-token" not in config_file.read_text(encoding="utf-8")

    def test_the_prompt_names_the_profile(self, tmp_path, config_file):
        self._with_staging(config_file)

        result = _invoke(tmp_path, config_file, ["create-profile", "staging"], input="n\n")

        assert "staging" in result.output
        assert "Overwrite" in result.output

    @pytest.mark.posix_only
    def test_confirming_replaces_the_profile(self, tmp_path, config_file):
        self._with_staging(config_file)

        result = _invoke(tmp_path, config_file, ["create-profile", "staging"], input="y\n")

        assert result.exit_code == 0, result.output
        written = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        assert written["staging"]["a1000"]["token"] == "supersecret-token"
        assert written["default"]["a1000"]["host"] == "https://original.example"
        assert stat.S_IMODE(config_file.stat().st_mode) == 0o600

    def test_a_new_name_is_not_prompted_for(self, tmp_path, config_file):
        self._with_staging(config_file)

        # No input supplied: a prompt here would fail the invocation.
        result = _invoke(tmp_path, config_file, ["create-profile", "production"])

        assert result.exit_code == 0, result.output
        assert "Overwrite" not in result.output
        assert (
            yaml.safe_load(config_file.read_text(encoding="utf-8"))["staging"]["a1000"]["token"]
            == "old-token"
        )
