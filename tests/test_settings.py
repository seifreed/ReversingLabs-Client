"""Tests for Settings config-file loading and profile handling."""

import json
import stat
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from rl_cli.config import Settings, get_settings
from rl_cli.config import settings as settings_module
from rl_cli.config.settings import (
    A1000Settings,
    ConfigFileError,
    TitaniumCloudSettings,
    _cached_settings,
    _config_sections,
    read_config_file,
    write_private_yaml,
)
from rl_cli.config.wizard import configure
from rl_cli.models.output_format import OutputFormat

CONFIG_EXAMPLE = Path(__file__).resolve().parent.parent / "config.example.yaml"


def _settings(tmp_path, **kwargs):
    """A fully loaded settings object: environment, then config file."""
    return Settings.load(cache_dir=tmp_path / "cache", config_dir=tmp_path / "config", **kwargs)


class TestACredentialIsSentAsTheUserWroteIt:
    """Judging a value and sending it are different questions.

    ``is_real_credential`` strips before deciding whether a value is one of
    the example files' stand-ins, so a stray space inside the quotes
    ``config.example.yaml`` already puts there cannot smuggle a placeholder
    past the probe. Stripping the value itself would be a different and
    worse thing: an appliance may accept a password whose leading space is
    real, and altering it here turns that into a 401 under a green
    ``check-access``, with nothing saying the value was changed.
    """

    @pytest.mark.parametrize(
        "field, value",
        [
            ("password", " pa55 word "),
            ("username", " analyst "),
            ("token", "a1000-token-9f3c\n"),
        ],
    )
    def test_padding_the_user_wrote_survives_to_the_appliance(self, field, value):
        assert getattr(A1000Settings(**{field: value}), field) == value

    def test_a_padded_stand_in_is_still_refused(self):
        """The half that does strip, so the two cannot be conflated."""
        from rl_cli.services.credentials import is_real_credential

        assert not is_real_credential("your_a1000_api_token_here ")
        assert not is_real_credential(" your_ticloud_username\n")


class TestApplyConfig:
    def test_known_keys_are_applied(self, tmp_path):
        settings = _settings(tmp_path)
        settings._apply_config({"a1000": {"host": "https://a1000.local", "token": "tok"}})
        assert settings.a1000.host == "https://a1000.local"
        assert settings.a1000.token == "tok"

    def test_unknown_keys_are_ignored_without_crashing(self, tmp_path):
        settings = _settings(tmp_path)
        settings._apply_config(
            {
                "titanium_cloud": {"api_token": "stale-example-key", "username": "u"},
                "output": {"show_timestamps": True, "format": "json"},
                "not_a_section": {"x": 1},
            }
        )
        assert settings.titanium_cloud.username == "u"
        assert settings.output.format is OutputFormat.JSON
        assert not hasattr(settings.titanium_cloud, "api_token")


class TestConfigFileRoundTrip:
    def test_profile_from_file_is_loaded(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"default": {"a1000": {"host": "https://from-file.local"}}})
        )
        settings = _settings(tmp_path, config_file=config_file)
        assert settings.a1000.host == "https://from-file.local"

    def test_missing_profile_keeps_defaults(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"other": {"a1000": {"host": "https://x"}}}))
        settings = _settings(tmp_path, config_file=config_file)
        assert settings.a1000.host == "https://a1000.reversinglabs.com"

    def test_save_config_round_trips_through_loader(self, tmp_path):
        settings = _settings(tmp_path)
        settings.a1000.host = "https://saved.local"
        saved = tmp_path / "saved.yaml"
        settings.save_config(saved)

        reloaded = _settings(tmp_path, config_file=saved)
        assert reloaded.a1000.host == "https://saved.local"

    def test_example_config_matches_model(self, tmp_path):
        """The shipped config.example.yaml must load without losing keys."""
        example = yaml.safe_load(CONFIG_EXAMPLE.read_text(encoding="utf-8"))
        profile = example["default"]
        settings = _settings(tmp_path)
        for section_name, section_values in profile.items():
            model = getattr(settings, section_name)
            for key in section_values:
                assert key in type(model).model_fields, f"{section_name}.{key} not in model"


class TestPlainConstructionStatesTheSettingsOutright:
    """``Settings(...)`` holds values; ``Settings.load(...)`` goes looking.

    Discovery and the YAML read used to happen inside ``__init__``, so
    there was no way to state a configuration and have it stay stated: a
    caller naming every value it cared about still had whichever of four
    files happened to exist layered over the top, and no settings object
    could be built at all without touching the filesystem.
    """

    def test_a_discoverable_file_is_not_applied_to_a_plain_construction(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yaml").write_text(
            yaml.dump({"default": {"a1000": {"host": "https://discovered.local"}}})
        )

        settings = Settings(cache_dir=tmp_path / "cache", config_dir=tmp_path / "config")

        assert settings.config_file is None
        assert settings.a1000.host == "https://a1000.reversinglabs.com"

    def test_an_explicit_config_file_is_not_read_by_a_plain_construction(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"default": {"a1000": {"host": "https://from-file"}}}))

        settings = Settings(
            cache_dir=tmp_path / "cache", config_dir=tmp_path / "config", config_file=config_file
        )

        assert settings.a1000.host == "https://a1000.reversinglabs.com"

    def test_construction_never_even_looks_for_a_config_file(self, tmp_path, monkeypatch):
        """Not "found nothing" — never looked. Discovery is ``load``'s job."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            settings_module,
            "_discover_config_file",
            lambda config_dir: pytest.fail("a plain construction searched the filesystem"),
        )
        monkeypatch.setattr(
            settings_module,
            "read_config_file",
            lambda path: pytest.fail("a plain construction read a config file"),
        )

        Settings(cache_dir=tmp_path / "cache", config_dir=tmp_path / "config")

    def test_load_still_discovers_and_applies(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yaml").write_text(
            yaml.dump({"default": {"a1000": {"host": "https://discovered.local"}}})
        )

        settings = Settings.load(cache_dir=tmp_path / "cache", config_dir=tmp_path / "config")

        assert settings.config_file == tmp_path / "config.yaml"
        assert settings.a1000.host == "https://discovered.local"

    def test_get_settings_goes_through_load(self, tmp_path, monkeypatch):
        """The CLI's entry point is the caller that must read the file."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yaml").write_text(
            yaml.dump({"default": {"a1000": {"host": "https://discovered.local"}}})
        )

        settings = get_settings(cache_dir=tmp_path / "cache", config_dir=tmp_path / "config")

        assert settings.a1000.host == "https://discovered.local"


class TestConfigFileDiscovery:
    """No ``--config`` means four locations are searched, in a documented order.

    README.md lists them as ``./config.yaml``, ``./.rl-cli.yaml``,
    ``~/.config/rl-cli/config.yaml`` and ``~/.rl-cli.yaml``. Every other
    test here hands ``config_file=`` in, so nothing pinned that the search
    happens at all, let alone which file wins when several exist. The
    hermetic fixture is what makes this safe: ``$HOME`` and the working
    directory are both inside ``tmp_path``.
    """

    def _write(self, path, host):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump({"default": {"a1000": {"host": host}}}))
        return path

    def _locations(self, tmp_path):
        """The four searched paths, in the order the README documents."""
        return [
            tmp_path / "config.yaml",
            tmp_path / ".rl-cli.yaml",
            tmp_path / "config" / "config.yaml",
            Path.home() / ".rl-cli.yaml",
        ]

    @pytest.mark.parametrize("index", range(4))
    def test_each_location_is_searched_when_it_is_the_only_one(self, tmp_path, monkeypatch, index):
        monkeypatch.chdir(tmp_path)
        path = self._write(self._locations(tmp_path)[index], f"https://found-{index}.local")

        settings = _settings(tmp_path)

        assert settings.config_file == path
        assert settings.a1000.host == f"https://found-{index}.local"

    @pytest.mark.parametrize("winner", range(4))
    def test_the_earlier_location_wins(self, tmp_path, monkeypatch, winner):
        """With every location from ``winner`` onwards present, ``winner`` is used."""
        monkeypatch.chdir(tmp_path)
        locations = self._locations(tmp_path)
        for index in range(winner, 4):
            self._write(locations[index], f"https://found-{index}.local")

        settings = _settings(tmp_path)

        assert settings.config_file == locations[winner]
        assert settings.a1000.host == f"https://found-{winner}.local"

    def test_no_file_anywhere_leaves_the_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        settings = _settings(tmp_path)

        assert settings.config_file is None
        assert settings.a1000.host == "https://a1000.reversinglabs.com"

    def test_an_explicit_file_is_not_overridden_by_discovery(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write(tmp_path / "config.yaml", "https://discovered.local")
        chosen = self._write(tmp_path / "chosen.yaml", "https://chosen.local")

        settings = _settings(tmp_path, config_file=chosen)

        assert settings.config_file == chosen
        assert settings.a1000.host == "https://chosen.local"


class TestGetSettingsHandsOutCopies:
    """The CLI applies ``--verbose``/``--output`` by mutating what it got back."""

    def _kwargs(self, tmp_path):
        return {
            "cache_dir": tmp_path / "cache",
            "config_dir": tmp_path / "config",
            "config_file": tmp_path / "missing.yaml",
        }

    def test_a_second_caller_does_not_see_the_first_caller_s_edits(self, tmp_path):
        kwargs = self._kwargs(tmp_path)
        first = get_settings(**kwargs)
        first.profile = "edited"
        first.output.verbose = True
        first.output.format = OutputFormat.JSON
        first.a1000.host = "https://mutated.local"

        second = get_settings(**kwargs)

        assert second is not first
        assert second.profile == "default"
        # The nested sections are separate objects too, not shared
        # references — a shallow copy would leak these.
        assert second.output.verbose is False
        assert second.output.format is OutputFormat.RICH
        assert second.a1000.host == "https://a1000.reversinglabs.com"

    def test_the_copy_is_of_one_parse(self, tmp_path):
        """Copies, not re-parses: the file and environment are read once."""
        kwargs = self._kwargs(tmp_path)
        first = get_settings(**kwargs)
        second = get_settings(**kwargs)

        assert _cached_settings.cache_info().misses == 1
        assert first.a1000 is not second.a1000
        assert first.model_dump() == second.model_dump()

    def test_the_same_arguments_in_another_order_are_the_same_arguments(self, tmp_path):
        """``lru_cache(**kwargs)`` keyed on the order they were typed in.

        ``get_settings(profile=p, config_file=f)`` and
        ``get_settings(config_file=f, profile=p)`` ask for identical
        settings, and used to parse the environment and the config file
        twice to answer, holding two independent objects for the rest of
        the process.
        """
        kwargs = self._kwargs(tmp_path)
        get_settings(profile="default", **kwargs)
        get_settings(**kwargs, profile="default")

        assert _cached_settings.cache_info().misses == 1


class TestSettingsCacheFollowsTheWorkingDirectory:
    """Discovery reads ``Path.cwd()``, so the cache key has to as well.

    ``Settings.load`` searches ``./config.yaml`` before anything else, and
    none of that search was in the memoisation key: a library caller that
    changed directory between two ``get_settings()`` calls — the use
    ``rl_cli/__init__.py`` advertises by exporting ``Settings`` — got the
    first directory's appliance back, silently, for the life of the
    process.
    """

    def _checkout(self, tmp_path, name, host):
        checkout = tmp_path / name
        checkout.mkdir()
        (checkout / "config.yaml").write_text(yaml.dump({"default": {"a1000": {"host": host}}}))
        return checkout

    def test_a_changed_working_directory_is_not_ignored(self, tmp_path, monkeypatch):
        first = self._checkout(tmp_path, "first", "https://first.local")
        second = self._checkout(tmp_path, "second", "https://second.local")
        kwargs = {"cache_dir": tmp_path / "cache", "config_dir": tmp_path / "config"}

        monkeypatch.chdir(first)
        assert get_settings(**kwargs).a1000.host == "https://first.local"

        monkeypatch.chdir(second)
        assert get_settings(**kwargs).a1000.host == "https://second.local"

    def test_returning_to_a_directory_reuses_its_parse(self, tmp_path, monkeypatch):
        """Keying on the directory must not turn the cache off."""
        first = self._checkout(tmp_path, "first", "https://first.local")
        kwargs = {"cache_dir": tmp_path / "cache", "config_dir": tmp_path / "config"}

        monkeypatch.chdir(first)
        get_settings(**kwargs)
        get_settings(**kwargs)

        assert _cached_settings.cache_info().misses == 1


class TestEnvironmentPrecedence:
    """Documented order is CLI options -> env vars -> config file -> defaults."""

    def _config(self, tmp_path, a1000):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"default": {"a1000": a1000}}))
        return config_file

    def test_env_var_supplies_the_token_with_no_config_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("A1000_TOKEN", "token-from-env")
        settings = _settings(tmp_path, config_file=tmp_path / "missing.yaml")
        assert settings.a1000.token == "token-from-env"

    def test_env_var_beats_the_config_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("A1000_TOKEN", "token-from-env")
        config_file = self._config(tmp_path, {"token": "token-from-file"})
        settings = _settings(tmp_path, config_file=config_file)
        assert settings.a1000.token == "token-from-env"

    def test_config_file_still_fills_fields_the_env_left_alone(self, tmp_path, monkeypatch):
        monkeypatch.setenv("A1000_TOKEN", "token-from-env")
        config_file = self._config(
            tmp_path, {"token": "token-from-file", "host": "https://from-file.local"}
        )
        settings = _settings(tmp_path, config_file=config_file)
        assert settings.a1000.token == "token-from-env"
        assert settings.a1000.host == "https://from-file.local"

    def test_dotenv_carrying_other_sections_does_not_break_startup(self, tmp_path, monkeypatch):
        """.env.example tells users to write all four prefixes into one file."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(
            "A1000_TOKEN=token-from-dotenv\n"
            "TICLOUD_USERNAME=someone\n"
            "OUTPUT_FORMAT=json\n"
            "RL_CLI_PROFILE=default\n"
        )
        settings = _settings(tmp_path, config_file=tmp_path / "missing.yaml")
        assert settings.a1000.token == "token-from-dotenv"
        assert settings.titanium_cloud.username == "someone"
        assert settings.output.format is OutputFormat.JSON


class TestSavedConfigPermissions:
    """Config files hold plaintext tokens; the default umask makes them world-readable."""

    @pytest.mark.posix_only
    def test_new_file_is_owner_only(self, tmp_path):
        settings = _settings(tmp_path, config_file=tmp_path / "missing.yaml")
        settings.a1000.token = "supersecret"
        target = tmp_path / "saved.yaml"
        settings.save_config(target)

        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert "supersecret" in target.read_text(encoding="utf-8")

    @pytest.mark.posix_only
    def test_existing_world_readable_file_is_tightened(self, tmp_path):
        target = tmp_path / "saved.yaml"
        target.write_text("default: {}\n")
        target.chmod(0o644)

        settings = _settings(tmp_path, config_file=tmp_path / "missing.yaml")
        settings.a1000.token = "supersecret"
        settings.save_config(target)

        assert stat.S_IMODE(target.stat().st_mode) == 0o600


class TestMissingProfile:
    def test_missing_profile_is_flagged_for_the_caller(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"staging": {"a1000": {"host": "https://x"}}}))
        settings = _settings(tmp_path, config_file=config_file, profile="typo")
        assert settings.missing_profile == "typo"

    def test_present_profile_is_not_flagged(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"staging": {"a1000": {"host": "https://x"}}}))
        settings = _settings(tmp_path, config_file=config_file, profile="staging")
        assert settings.missing_profile is None


class TestProxyRedaction:
    """`http://user:pass@proxy:8080` is how a proxy credential is written.

    `config show` writes the profile to stdout, so an unredacted proxy
    copied the password out of a 0600 config file into whatever the shell
    redirected to.
    """

    CANARY = "hunter2-canary"

    def _profile(self, tmp_path):
        settings = _settings(tmp_path, config_file=tmp_path / "absent.yaml")
        settings.a1000.proxy = f"http://proxyuser:{self.CANARY}@proxy.corp:8080"
        settings.titanium_cloud.proxy = f"https://proxyuser:{self.CANARY}@proxy.corp:3128"
        return settings

    def test_the_password_is_absent_from_every_rendering(self, tmp_path):
        dump = self._profile(tmp_path).profile_dump()
        for text in (
            json.dumps(dump),
            yaml.dump(dump),
            str(dump),
        ):
            assert self.CANARY not in text

    def test_the_host_stays_readable(self, tmp_path):
        dump = self._profile(tmp_path).profile_dump()
        assert dump["a1000"]["proxy"] == "http://proxyuser:***@proxy.corp:8080"
        assert dump["titanium_cloud"]["proxy"] == "https://proxyuser:***@proxy.corp:3128"

    def test_the_writers_still_store_what_was_typed(self, tmp_path):
        settings = self._profile(tmp_path)
        assert self.CANARY in settings.profile_dump(redact=False)["a1000"]["proxy"]

    def test_a_proxy_without_credentials_is_left_alone(self, tmp_path):
        settings = _settings(tmp_path, config_file=tmp_path / "absent.yaml")
        settings.a1000.proxy = "http://proxy.corp:8080"
        assert settings.profile_dump()["a1000"]["proxy"] == "http://proxy.corp:8080"

    def test_config_dump_does_not_leak_it_either(self, tmp_path):
        """`a1000 config-dump` promises credentials are absent."""
        from rl_cli.services.a1000 import A1000Service, A1000Session

        settings = self._profile(tmp_path)
        summary = A1000Service(A1000Session(settings)).connection_summary()
        assert self.CANARY not in str(summary)
        assert summary["proxy"] == "http://proxyuser:***@proxy.corp:8080"


class TestConfigFileValuesAreValidatedLikeEnvironmentOnes:
    """A key the file states unusably keeps its default and says so.

    ``_apply_config`` reaches the sections through ``setattr``, which
    pydantic does not check unless the model asks for it, so a
    ``timeout: "abc"`` in YAML was stored verbatim while the identical
    value in ``A1000_TIMEOUT`` was rejected on the way in. The bad value
    then travelled as far as ``requests``, which reported it as a
    TypeError about a string, naming neither the file nor the key.
    """

    def test_an_unusable_timeout_is_refused_and_named(self, tmp_path):
        settings = _settings(tmp_path)
        default = settings.a1000.timeout

        settings._apply_config({"a1000": {"timeout": "not-a-number"}})

        assert settings.a1000.timeout == default
        assert any("a1000.timeout" in entry for entry in settings.rejected_config_values)

    def test_a_usable_value_in_the_same_section_still_lands(self, tmp_path):
        settings = _settings(tmp_path)

        settings._apply_config({"a1000": {"timeout": "not-a-number", "host": "https://ok.local"}})

        assert settings.a1000.host == "https://ok.local"

    def test_a_well_typed_value_is_recorded_nowhere(self, tmp_path):
        settings = _settings(tmp_path)
        settings._apply_config({"a1000": {"timeout": 45}})
        assert settings.a1000.timeout == 45
        assert settings.rejected_config_values == []

    @pytest.mark.parametrize("section", ["a1000", "titanium_cloud"])
    @pytest.mark.parametrize("timeout", [0, -1])
    def test_a_nonpositive_timeout_is_refused(self, tmp_path, section, timeout):
        settings = _settings(tmp_path)

        settings._apply_config({section: {"timeout": timeout}})

        assert settings.rejected_config_values
        assert getattr(settings, section).timeout == 300


class TestAMalformedConfigFileIsReportedAsOne:
    def test_broken_yaml_names_the_file_instead_of_escaping_raw(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("default:\n  a1000:\n   host: 'unclosed\n")

        with pytest.raises(ConfigFileError) as caught:
            read_config_file(config_file)

        assert str(config_file) in str(caught.value)

    def test_an_absent_file_is_still_simply_empty(self, tmp_path):
        assert read_config_file(tmp_path / "nope.yaml") == {}

    def test_an_empty_file_is_simply_empty(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        assert read_config_file(config_file) == {}

    @pytest.mark.parametrize(
        "body", ["- default\n- staging\n", "just a string\n", "42\n"], ids=["list", "scalar", "int"]
    )
    def test_well_formed_yaml_that_is_not_a_profile_map_is_refused(self, tmp_path, body):
        """It parsed, so it used to travel on as a dict that was not one.

        The profile lookup then quietly found nothing and ``save_config``
        raised ``TypeError: list indices must be integers`` from inside
        pydantic, naming neither the file nor the fact it was the config.
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text(body)

        with pytest.raises(ConfigFileError) as caught:
            read_config_file(config_file)

        assert str(config_file) in str(caught.value)

    def test_a_config_file_that_is_a_list_does_not_crash_the_loader(self, tmp_path):
        """The CLI catches ConfigFileError; it never caught TypeError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("- default\n")

        with pytest.raises(ConfigFileError):
            _settings(tmp_path, config_file=config_file)


class TestTheSectionListComesFromTheModel:
    """A hand-kept list is one edit away from a section the loader ignores."""

    def test_every_settings_subsection_is_loadable(self):
        assert _config_sections() == ("titanium_cloud", "a1000", "output")

    def test_no_plain_field_is_mistaken_for_a_section(self):
        for name in ("config_file", "profile", "cache_dir", "config_dir"):
            assert name not in _config_sections()

    def test_the_loader_and_the_rejection_prefixes_read_one_table(self):
        """What counts as a section is decided once, for both readers.

        The predicate — a field annotated with a ``BaseSettings``
        subclass — was written out twice: once to list the sections the
        loader walks, once to recover the section name a nested
        validation error belongs to. A section declared ``X | None`` or
        ``Annotated[X, ...]`` drops out of both, and whoever fixed one
        would have had no signal the other existed.
        """
        models = settings_module.section_models(Settings)

        assert tuple(models) == _config_sections()
        assert {name: model.__name__ for name, model in models.items()} == {
            "titanium_cloud": "TitaniumCloudSettings",
            "a1000": "A1000Settings",
            "output": "OutputSettings",
        }

    def test_the_predicate_is_spelled_once(self):
        source = Path(settings_module.__file__).read_text(encoding="utf-8")

        assert source.count("issubclass(field.annotation, BaseSettings)") == 1

    def test_every_section_names_itself_in_a_rejection(self):
        """The prefix each section's own validation error is reported under.

        A bare ``format`` does not say which of the sections to go and
        look at, which is what the recovered prefix is for.
        """
        unusable = {
            "titanium_cloud": {"timeout": "abc"},
            "a1000": {"timeout": "abc"},
            "output": {"format": "jsn"},
        }
        assert set(unusable) == set(_config_sections()), "a section grew or moved"

        for name, values in unusable.items():
            with pytest.raises(ValidationError) as refused:
                settings_module.section_models(Settings)[name].model_validate(values)

            assert settings_module._rejections(refused.value).startswith(f"{name}.")


class TestTheConfigFileIsWrittenAsPrivatelyAsASample:
    @pytest.mark.posix_only
    def test_it_lands_owner_only(self, tmp_path):
        path = tmp_path / "config.yaml"
        write_private_yaml(path, {"default": {"a1000": {"token": "secret"}}})
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert (
            yaml.safe_load(path.read_text(encoding="utf-8"))["default"]["a1000"]["token"]
            == "secret"
        )

    @pytest.mark.posix_only
    def test_it_writes_through_a_symlink_to_the_real_file(self, tmp_path):
        """Pointing the config at a dotfiles checkout is how people keep it.

        private_writer refuses a symlink because it guards a download
        directory, where a planted link redirects malware somewhere the
        analyst never named. A config file the user symlinked themselves
        is the opposite case, and refusing it broke `config save` with
        "Unexpected error: Refusing to write through a symlink".
        """
        target = tmp_path / "real.yaml"
        target.write_text("old\n")
        link = tmp_path / "config.yaml"
        link.symlink_to(target)

        write_private_yaml(link, {"default": {"a1000": {"token": "t"}}})

        assert link.is_symlink(), "the link itself must survive"
        assert (
            yaml.safe_load(target.read_text(encoding="utf-8"))["default"]["a1000"]["token"] == "t"
        )
        assert stat.S_IMODE(target.stat().st_mode) == 0o600


class TestAnUnusableEnvironmentValueReadsLikeAnUnusableConfigValue:
    """The same mistake from the other source, phrased the same way.

    A config file stating ``output.format: jsn`` stops the run with a line
    naming the file, the key and the accepted words. The environment
    stating ``OUTPUT_FORMAT=jsn`` raised pydantic's ``ValidationError``
    all the way to ``main()``'s catch-all, which printed it under
    "Unexpected error" — the framing that tells an analyst to file a bug
    about their own typo.
    """

    @pytest.mark.parametrize(
        ("variable", "value", "expected"),
        [
            ("OUTPUT_FORMAT", "jsn", "output.format"),
            ("A1000_TIMEOUT", "abc", "a1000.timeout"),
            ("A1000_VERIFY_SSL", "maybe", "a1000.verify_ssl"),
            ("TICLOUD_TIMEOUT", "soon", "titanium_cloud.timeout"),
        ],
    )
    def test_it_names_the_section_and_the_key(
        self, tmp_path, monkeypatch, variable, value, expected
    ):
        monkeypatch.setenv(variable, value)

        with pytest.raises(ConfigFileError) as refusal:
            Settings.load(cache_dir=tmp_path / "c", config_dir=tmp_path / "cd")

        # The section is the half pydantic cannot supply: a nested section
        # validates on its own and reports a bare `format`, which does not
        # say which of the four sections to go and look at.
        assert expected in str(refusal.value)
        assert "The environment" in str(refusal.value)

    def test_a_usable_environment_value_is_untouched(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OUTPUT_FORMAT", "yaml")

        loaded = Settings.load(cache_dir=tmp_path / "c", config_dir=tmp_path / "cd")

        assert loaded.output.format == OutputFormat.YAML


class TestAnUnusableConfigValueStopsTheRun:
    """A rejected value must not leave the field on its default.

    The defaults for ``host`` are ReversingLabs' public clouds, so an
    empty or mistyped ``a1000.host`` silently retargeted the CLI at the
    vendor — carrying the configured token and every hash the analyst
    looked up — behind a single line of stderr. Which appliance is being
    talked to is not something to get wrong quietly.
    """

    def _load(self, tmp_path, body: str):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(body)
        return Settings.load(
            config_file=config_file, cache_dir=tmp_path / "c", config_dir=tmp_path / "cd"
        )

    def test_an_empty_host_is_refused_rather_than_defaulted(self, tmp_path):
        with pytest.raises(ConfigFileError) as caught:
            self._load(tmp_path, "default:\n  a1000:\n    host:\n    token: tok\n")
        assert "a1000.host" in str(caught.value)

    def test_the_vendor_default_is_never_what_a_stated_host_falls_back_to(self, tmp_path):
        """The specific outcome that made this severe."""
        with pytest.raises(ConfigFileError):
            self._load(tmp_path, "default:\n  a1000:\n    host: [not, a, string]\n")

    def test_an_unusable_timeout_stops_the_run_too(self, tmp_path):
        with pytest.raises(ConfigFileError) as caught:
            self._load(tmp_path, "default:\n  a1000:\n    timeout: not-a-number\n")
        assert "a1000.timeout" in str(caught.value)

    def test_an_unusable_output_format_stops_the_run_like_its_siblings(self, tmp_path):
        """It was the one rejected value that warned and carried on.

        ``output.format`` was typed as a bare ``str`` with the valid words
        written only in the field's description, so anything at all passed
        validation here and the mistake surfaced in ``cli/main.py`` as a
        warning before rendering rich instead — the warn-and-continue this
        whole class exists to argue against.
        """
        with pytest.raises(ConfigFileError) as caught:
            self._load(tmp_path, "default:\n  output:\n    format: jsonn\n")

        message = str(caught.value)
        assert str(tmp_path / "config.yaml") in message
        assert "output.format" in message
        assert "'json'" in message, "the message has to say which words are accepted"

    def test_every_format_the_cli_offers_is_accepted(self, tmp_path):
        for fmt in OutputFormat:
            settings = self._load(tmp_path, f"default:\n  output:\n    format: {fmt.value}\n")
            assert settings.output.format is fmt

    def test_a_usable_file_still_loads(self, tmp_path):
        settings = self._load(tmp_path, "default:\n  a1000:\n    host: https://box.local\n")
        assert settings.a1000.host == "https://box.local"
        assert settings.rejected_config_values == []


class TestVerifySslIsNotCoerced:
    """Turning TLS verification off must be stated, not inferred.

    Pydantic reads "no"/"off"/"0" as False, so a quoted ``verify_ssl:
    "no"`` would silently stop verifying certificates against a malware
    appliance. Before validation was applied at all it was stored as a
    string, which requests rejected as a bad CA-bundle path — wrong, but
    loud.

    The two words spelling a boolean out are the exception, because an
    environment variable is only ever a string: with ``StrictBool`` alone
    the ``A1000_VERIFY_SSL`` .env.example documents could not be set at
    all, and every command exited 1 on it.
    """

    @pytest.mark.parametrize("stated", ["no", "off", "on", "0", "1", "yes", "", 0, 1])
    def test_anything_but_the_two_words_is_refused(self, stated):
        section = A1000Settings()
        with pytest.raises(ValidationError):
            section.verify_ssl = stated

    @pytest.mark.parametrize("stated", [True, False])
    def test_a_real_boolean_is_taken(self, stated):
        section = A1000Settings()
        section.verify_ssl = stated
        assert section.verify_ssl is stated

    @pytest.mark.parametrize(
        ("stated", "expected"),
        [("true", True), ("false", False), ("TRUE", True), ("False", False), (" false ", False)],
    )
    def test_the_word_itself_is_read(self, stated, expected):
        section = A1000Settings()
        section.verify_ssl = stated
        assert section.verify_ssl is expected

    @pytest.mark.parametrize(("stated", "expected"), [("true", True), ("false", False)])
    def test_the_sharing_switch_reads_the_same_words(self, stated, expected):
        """It is a ``StrictBool`` for the same reason, so it takes the same words."""
        section = TitaniumCloudSettings()
        section.share_url_lookups = stated
        assert section.share_url_lookups is expected

    @pytest.mark.parametrize("stated", ["1", "0", "no", ""])
    def test_the_sharing_switch_refuses_the_same_values(self, stated):
        section = TitaniumCloudSettings()
        with pytest.raises(ValidationError):
            section.share_url_lookups = stated

    @pytest.mark.parametrize(
        ("variable", "field"),
        [
            ("A1000_VERIFY_SSL", "a1000"),
            ("TICLOUD_VERIFY_SSL", "titanium_cloud"),
        ],
    )
    def test_the_environment_can_turn_verification_off(
        self, tmp_path, monkeypatch, variable, field
    ):
        """The on-prem, self-signed case, which had no environment path at all."""
        monkeypatch.setenv(variable, "false")

        loaded = _settings(tmp_path)

        assert getattr(loaded, field).verify_ssl is False


class TestAProfileThatIsNotAMappingIsRefusedNotCrashedOn:
    """``read_config_file`` guarded only the top level of the file.

    One level down, four plausible files reached ``_apply_config`` and
    came back out as ``AttributeError: 'NoneType' object has no attribute
    'get'`` under main()'s "Unexpected error" -- naming neither the file,
    nor the profile, nor the fact that it was the config at all.
    ``default:`` with nothing under it is the likeliest way to write one,
    and it broke every command.
    """

    def _load(self, tmp_path, body: str):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(body)
        return Settings.load(
            config_file=config_file, cache_dir=tmp_path / "c", config_dir=tmp_path / "cd"
        )

    @pytest.mark.parametrize(
        ("body", "named"),
        [
            ('default: "oops"', "str"),
            ("default:\n", "NoneType"),
            ("default:\n  - 1\n", "list"),
        ],
        ids=["scalar-profile", "empty-profile", "list-profile"],
    )
    def test_a_profile_holding_no_sections_names_the_profile(self, tmp_path, body, named):
        with pytest.raises(ConfigFileError) as refusal:
            self._load(tmp_path, body)

        assert "profile 'default'" in str(refusal.value)
        assert named in str(refusal.value)

    @pytest.mark.parametrize(
        ("body", "named"),
        [
            ('default:\n  a1000: "host"\n', "str"),
            ("default:\n  a1000:\n    - 1\n", "list"),
        ],
        ids=["scalar-section", "list-section"],
    )
    def test_a_section_holding_no_settings_names_the_section(self, tmp_path, body, named):
        with pytest.raises(ConfigFileError) as refusal:
            self._load(tmp_path, body)

        assert "a1000" in str(refusal.value)
        assert named in str(refusal.value)

    def test_a_section_the_file_simply_omits_is_still_fine(self, tmp_path):
        """Absent is not the same as malformed: the defaults stand."""
        loaded = self._load(tmp_path, "default:\n  a1000:\n    host: https://box.local\n")

        assert loaded.a1000.host == "https://box.local"
        assert loaded.titanium_cloud.host == "https://data.reversinglabs.com"


class _ExtraSettings(BaseSettings):
    """A settings section nobody wrote a question sequence for."""

    endpoint: str = Field(default="https://extra.example")
    api_token: str | None = Field(default=None)
    enabled: bool = Field(default=False)
    # One of each kind a settings field comes in, because each is asked
    # about differently: a bool is a yes/no, an enum is a closed list, and
    # anything else is typed in.
    fmt: OutputFormat = Field(default=OutputFormat.RICH)

    model_config = SettingsConfigDict(env_prefix="RL_CLI_TEST_EXTRA_", extra="ignore")


class _SettingsWithAnExtraSection(Settings):
    """What shipping a fourth settings section looks like, and nothing else."""

    extra_service: _ExtraSettings = Field(default_factory=_ExtraSettings)


class _Answers:
    """A prompter that answers from a script and remembers what it was asked.

    Standing in for the terminal, which is the point of the split: the
    sequence ``config init`` puts is exercised here with no click, no
    console and no stdin.
    """

    def __init__(self, answers: dict[str, object] | None = None) -> None:
        self.answers = answers or {}
        self.asked: list[str] = []
        self.secrets: list[str] = []
        self.offered: dict[str, tuple[str, ...]] = {}

    def confirm(self, question: str, /, *, default: bool = False) -> bool:
        self.asked.append(question)
        answer = self.answers.get(question, default)
        return bool(answer)

    def ask(
        self,
        question: str,
        /,
        *,
        default: str | None = None,
        secret: bool = False,
        choices: Sequence[str] | None = None,
    ) -> str:
        self.asked.append(question)
        if secret:
            self.secrets.append(question)
        if choices is not None:
            self.offered[question] = tuple(choices)
        return str(self.answers.get(question, default if default is not None else ""))


class TestTheInitQuestionSequenceIsTheConfigLayers:
    """``config init``'s wizard used to be 36 lines inside the click module.

    It knew the full shape of every settings section, so a new section
    meant editing a CLI module -- and it filled in the very ``Settings``
    instance ``config save`` serialises, so a mistake landed in the user's
    0600 credentials file. The sequence is stated here; the CLI supplies
    the asking.

    The exact wording and order below are a regression contract: they are
    what an analyst following the README is answering.
    """

    _TITANIUM_CLOUD = (
        "Configure TitaniumCloud API?",
        "TitaniumCloud host",
        "Username",
        "Password",
    )
    _OUTPUT = ("Configure output settings?", "Default output format", "Enable colored output?")

    def test_the_password_route_asks_exactly_what_it_always_asked(self):
        prompter = _Answers(dict.fromkeys(self._sections(), True))

        configure(Settings(), prompter)

        assert prompter.asked == [
            *self._TITANIUM_CLOUD,
            "Configure A1000 platform?",
            "A1000 host",
            "Use token authentication?",
            "Username",
            "Password",
            *self._OUTPUT,
        ]

    def test_the_token_route_replaces_the_username_and_password_questions(self):
        prompter = _Answers(
            {**dict.fromkeys(self._sections(), True), "Use token authentication?": True}
        )

        configure(Settings(), prompter)

        assert prompter.asked == [
            *self._TITANIUM_CLOUD,
            "Configure A1000 platform?",
            "A1000 host",
            "Use token authentication?",
            "API Token",
            *self._OUTPUT,
        ]

    def test_a_declined_section_is_not_asked_about_further(self):
        prompter = _Answers()

        configure(Settings(), prompter)

        assert prompter.asked == self._sections()

    def test_the_secrets_are_the_ones_asked_for_in_confidence(self):
        """What the CLI adapter turns into ``hide_input=True``."""
        prompter = _Answers(dict.fromkeys(self._sections(), True))

        configure(Settings(), prompter)

        assert prompter.secrets == ["Password", "Password"]

    def test_the_token_is_asked_for_in_confidence_too(self):
        prompter = _Answers(
            {**dict.fromkeys(self._sections(), True), "Use token authentication?": True}
        )

        configure(Settings(), prompter)

        assert prompter.secrets == ["Password", "API Token"]

    def test_the_format_question_offers_the_enum_and_no_hand_written_list(self):
        prompter = _Answers(dict.fromkeys(self._sections(), True))

        configure(Settings(), prompter)

        assert prompter.offered["Default output format"] == tuple(fmt.value for fmt in OutputFormat)

    def test_the_answers_land_on_the_settings_object(self):
        settings = Settings()
        prompter = _Answers(
            {
                **dict.fromkeys(self._sections(), True),
                "TitaniumCloud host": "https://tc.example",
                "A1000 host": "https://a1000.example",
                "Use token authentication?": True,
                "API Token": "typed-token",
                "Default output format": "json",
                "Enable colored output?": False,
            }
        )

        configure(settings, prompter)

        assert settings.titanium_cloud.host == "https://tc.example"
        assert settings.a1000.host == "https://a1000.example"
        assert settings.a1000.token == "typed-token"
        assert settings.output.format is OutputFormat.JSON
        assert settings.output.color is False

    def test_a_declined_run_leaves_every_section_alone(self):
        settings = Settings()

        configure(settings, _Answers())

        assert settings.a1000.token is None
        assert settings.titanium_cloud.password is None
        assert settings.output.format is OutputFormat.RICH

    @staticmethod
    def _sections() -> list[str]:
        return [
            "Configure TitaniumCloud API?",
            "Configure A1000 platform?",
            "Configure output settings?",
        ]


class TestANewSettingsSectionIsPromptedForWithoutANewBranch:
    """The sections come off the model, the way the loader's already do.

    A section added to ``Settings`` used to need a fourth ``if
    click.confirm(...)`` in the click module before ``config init`` would
    ask about it at all -- which is the same defect ``section_models``
    exists to prevent on the loading side.
    """

    def test_the_new_section_is_offered(self):
        prompter = _Answers()

        configure(_SettingsWithAnExtraSection(), prompter)

        assert "Configure extra service settings?" in prompter.asked

    def test_its_fields_are_asked_for_and_kept(self):
        settings = _SettingsWithAnExtraSection()
        prompter = _Answers(
            {
                "Configure extra service settings?": True,
                "Endpoint": "https://typed.example",
                "Api token": "typed-token",
                "Enabled?": True,
                "Fmt": "json",
            }
        )

        configure(settings, prompter)

        assert settings.extra_service.endpoint == "https://typed.example"
        assert settings.extra_service.api_token == "typed-token"
        assert settings.extra_service.enabled is True
        assert settings.extra_service.fmt is OutputFormat.JSON

    def test_a_field_holding_an_enum_is_offered_its_members_and_no_more(self):
        """Typed back as the member, not left as the word that was chosen."""
        prompter = _Answers({"Configure extra service settings?": True})

        settings = _SettingsWithAnExtraSection()
        configure(settings, prompter)

        assert prompter.offered["Fmt"] == tuple(fmt.value for fmt in OutputFormat)
        assert settings.extra_service.fmt is OutputFormat.RICH

    def test_a_field_that_reads_as_a_credential_is_asked_for_in_confidence(self):
        prompter = _Answers({"Configure extra service settings?": True})

        configure(_SettingsWithAnExtraSection(), prompter)

        assert prompter.secrets == ["Api token"]
