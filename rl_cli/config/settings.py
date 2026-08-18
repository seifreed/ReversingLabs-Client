"""Configuration settings for ReversingLabs CLI."""

from functools import cache, lru_cache
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BeforeValidator, Field, StrictBool, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rl_cli._version import __version__
from rl_cli.models.output_format import OutputFormat
from rl_cli.models.redaction import redact_section
from rl_cli.storage.files import private_writer


class ConfigFileError(Exception):
    """A config file exists but could not be read as YAML."""


def read_config_file(path: Path) -> dict[str, Any]:
    """Parse a config file, or return an empty mapping if it has nothing.

    A config file is user input, so a stray tab or an unclosed quote is
    reported against the file rather than crashing every command. Well-formed
    YAML that is not a mapping of profiles gets the same treatment: it
    parses, so it would otherwise travel on as a ``dict`` that is really a
    list or a string and fail somewhere that cannot name the config.
    """
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigFileError(f"{path} is not valid YAML: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigFileError(
            f"{path} must map profile names to their settings, not a {type(loaded).__name__}"
        )
    return loaded


def write_private_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write a config file readable only by its owner.

    These files hold API tokens and passwords in plaintext, so they get
    the same guarantees as a downloaded sample: owner-only (0600) from the
    moment the file exists, and a failed write leaves the previous config
    intact instead of a truncated file wearing its name.

    A symlink here is resolved and followed, not refused, which is where
    this parts company with
    :func:`~rl_cli.storage.files.private_writer`'s policy. That refusal
    guards a *download* directory, where the path is chosen by a command
    and a planted link redirects malware or a report somewhere the analyst
    did not name. A config file is the opposite case: pointing
    ``~/.config/rl-cli/config.yaml`` at a dotfiles checkout is how people
    keep it. Resolving first keeps the 0600 and the atomic replace while
    writing the file the user actually meant.
    """
    target = path.resolve() if path.is_symlink() else path
    target.parent.mkdir(parents=True, exist_ok=True)
    with private_writer(target, binary=False) as handle:
        yaml.dump(data, handle, default_flow_style=False)


def _env_config(prefix: str) -> SettingsConfigDict:
    """Environment configuration shared by every settings section.

    ``env_file`` lets a section be populated from a ``.env`` file as well
    as from the real environment. ``extra="ignore"`` is what makes that
    safe: a ``.env`` holding another section's variables (which is exactly
    what .env.example tells users to write) would otherwise fail
    validation and break every CLI invocation.

    ``validate_assignment`` is what makes the config file as safe as the
    environment: ``_apply_config`` reaches these sections with
    ``setattr``, which pydantic does not check by default, so without it a
    ``timeout: "abc"`` in YAML is accepted verbatim and surfaces much
    later inside ``requests``.
    """
    return SettingsConfigDict(
        env_prefix=prefix,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_assignment=True,
        # Deliberately no ``str_strip_whitespace``. A credential is sent as
        # the user wrote it: an appliance may accept a password whose
        # leading or trailing space is real, and altering it here would
        # turn that into a 401 with a green ``check-access`` above it and
        # nothing on screen saying the value had been changed.
        #
        # ``is_real_credential`` strips before *judging* whether a value is
        # a shipped stand-in, which is a different question — the answer to
        # "is this the example file's placeholder" cannot depend on a stray
        # space, while the answer to "what do we authenticate with" must.
    )


def _spelled_out_boolean(value: Any) -> Any:
    """Read the words ``true`` and ``false``, and nothing else, from a string.

    Environment variables and ``.env`` files hold only strings, so a bare
    ``StrictBool`` can be set from neither — including the three variables
    .env.example documents.

    Only these two spellings, stripped and case-folded, are translated.
    "no", "off", "0", "1", "yes" and "" are handed on as the strings they
    are and refused, because those are the readings that let a shell
    default or an empty variable stop certificate verification against a
    malware appliance without anybody writing the word.
    """
    if isinstance(value, str):
        return {"true": True, "false": False}.get(value.strip().casefold(), value)
    return value


# ``verify_ssl`` and ``share_url_lookups`` are the fields that must not be
# coerced: pydantic reads "no"/"off"/"0" as False, so a quoted
# `verify_ssl: "no"` would turn certificate verification off against a
# malware appliance without saying so. StrictBool takes the real YAML
# booleans and, through the validator above, the words ``true`` and
# ``false``; everything else is refused, and a refused value stops the run
# rather than falling back to a default.
StatedBool = Annotated[StrictBool, BeforeValidator(_spelled_out_boolean)]


class TitaniumCloudSettings(BaseSettings):
    """TitaniumCloud API configuration."""

    host: str = Field(
        default="https://data.reversinglabs.com", description="TitaniumCloud API host"
    )
    username: str | None = Field(default=None, description="API username")
    password: str | None = Field(default=None, description="API password")
    user_agent: str = Field(default=f"RL-CLI/{__version__}", description="User agent string")
    verify_ssl: StatedBool = Field(default=True, description="Verify SSL certificates")
    proxy: str | None = Field(default=None, description="Proxy URL")
    timeout: int = Field(
        default=300,
        gt=0,
        description="Per-request read timeout in seconds (file submission can be slow)",
    )
    # The URL endpoints take a `private` flag which the SDK defaults to
    # False, meaning the URL asked about -- and the answer -- are shared
    # with third-party sources and enter public feeds. The vendored 2.13
    # source does not carry the parameter at all, so reading it will not
    # show the default. For a tool whose queries are an analyst's
    # investigation targets, the safe end of that switch is the default,
    # and anyone deliberately contributing to the feeds can say so.
    share_url_lookups: StatedBool = Field(
        default=False,
        description="Share looked-up URLs and their results with third-party feeds",
    )

    model_config = _env_config("TICLOUD_")


class A1000Settings(BaseSettings):
    """A1000 platform configuration."""

    host: str = Field(default="https://a1000.reversinglabs.com", description="A1000 API host")
    username: str | None = Field(default=None, description="API username")
    password: str | None = Field(default=None, description="API password")
    token: str | None = Field(
        default=None, description="API token (alternative to username/password)"
    )
    verify_ssl: StatedBool = Field(default=True, description="Verify SSL certificates")
    proxy: str | None = Field(default=None, description="Proxy URL")
    timeout: int = Field(
        default=300,
        gt=0,
        description="Per-request read timeout in seconds (file submission can be slow)",
    )

    model_config = _env_config("A1000_")


class OutputSettings(BaseSettings):
    """Output formatting configuration."""

    # Typed against the enum the renderer reads, so an ``output.format``
    # the CLI cannot use meets the hard stop in ``load`` like every other
    # rejected config value, rather than the warn-and-fall-back that policy
    # exists to prevent.
    format: OutputFormat = Field(default=OutputFormat.RICH, description="Output format")
    color: bool = Field(default=True, description="Enable colored output")
    verbose: bool = Field(default=False, description="Enable verbose output")
    quiet: bool = Field(default=False, description="Suppress non-essential output")

    model_config = _env_config("OUTPUT_")


def _rejections(exc: ValidationError) -> str:
    """Pydantic's complaint, phrased the way the config-file path phrases it.

    Not memoised: a ``ValidationError`` hashes by identity and every call
    is handed a fresh one, so a cache could never hit and would only hold a
    reference to every exception ever passed.

    The section name has to be recovered from the model that refused the
    value, because a nested section validates on its own and reports
    ``format``, not ``output.format`` — and a bare ``format`` does not say
    which of the four sections to go and look at.
    """
    section_of = {model.__name__: name for name, model in section_models(Settings).items()}
    prefix = section_of.get(exc.title)
    return "; ".join(
        ".".join(filter(None, (prefix, *(str(part) for part in error["loc"]))))
        + f": {error['msg']}"
        for error in exc.errors()
    )


@cache
def section_models(model: type[BaseSettings]) -> dict[str, type[BaseSettings]]:
    """The config sections, as the field name each is set under and the model behind it.

    What counts as a section is decided here and nowhere else: three
    readers need it — the loader, which walks the sections a config file
    may carry; :func:`_rejections`, which has to recover a section's name
    from the model that refused a value; and ``config init``'s wizard,
    which offers each section in turn. Spelled out three times, a section
    declared ``X | None`` or ``Annotated[X, ...]`` would drop out of all
    of them and whoever fixed one would have no signal the others existed.

    A hand-kept list is one edit away from a new section the loader
    silently ignores, so it is read off the model. Takes the model rather
    than closing over ``Settings`` so the answer is about the object in
    hand: the wizard fills in whatever settings it was handed, and a test
    can hand it one carrying a section this release does not ship.
    """
    return {
        name: field.annotation
        for name, field in model.model_fields.items()
        if isinstance(field.annotation, type) and issubclass(field.annotation, BaseSettings)
    }


def _config_sections() -> tuple[str, ...]:
    """The sections a config file may carry, in declaration order."""
    return tuple(section_models(Settings))


def _discover_config_file(config_dir: Path) -> Path | None:
    """The first config file present, in the order README documents.

    ``./config.yaml`` first so a checkout can carry its own, then the
    hidden variant, then the user's config directory, then their home.
    """
    locations = (
        Path.cwd() / "config.yaml",
        Path.cwd() / ".rl-cli.yaml",
        config_dir / "config.yaml",
        Path.home() / ".rl-cli.yaml",
    )
    return next((path for path in locations if path.exists()), None)


class Settings(BaseSettings):
    """Main application settings."""

    config_file: Path | None = Field(default=None, description="Path to configuration file")
    profile: str = Field(default="default", description="Configuration profile to use")
    missing_profile: str | None = Field(
        default=None,
        exclude=True,
        description="Set to the requested profile when the config file has no such section",
    )
    rejected_config_values: list[str] = Field(
        default_factory=list,
        exclude=True,
        description="Config-file keys whose values failed validation and were left at defaults",
    )

    titanium_cloud: TitaniumCloudSettings = Field(default_factory=TitaniumCloudSettings)
    a1000: A1000Settings = Field(default_factory=A1000Settings)
    output: OutputSettings = Field(default_factory=OutputSettings)

    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "rl-cli", description="Cache directory"
    )
    config_dir: Path = Field(
        default_factory=lambda: Path.home() / ".config" / "rl-cli",
        description="Configuration directory",
    )

    @field_validator("config_file", "cache_dir", "config_dir")
    @classmethod
    def _expand_user(cls, value: Path | None) -> Path | None:
        """Make ``~`` mean the home directory, as .env.example promises.

        The shell expands a tilde it is shown, but nothing expands one
        arriving inside an environment variable or a config file, so the
        documented ``RL_CLI_CACHE_DIR=~/.cache/rl-cli`` would otherwise
        make a directory literally named ``~``.
        """
        return value.expanduser() if value is not None else None

    model_config = SettingsConfigDict(
        env_prefix="RL_CLI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def load(cls, **kwargs: Any) -> "Settings":
        """Settings for a run: the environment, then whatever file applies.

        Discovery and the YAML read live here rather than in ``__init__``,
        so that constructing a ``Settings`` is only ever holding values: a
        caller naming every value it cares about gets exactly those, with
        no ``config.yaml`` from the working directory layered on top, and
        can be built without touching disk.
        """
        try:
            settings = cls(**kwargs)
        except ValidationError as exc:
            # An environment variable the user set is a thing to fix, not a
            # fault to report — the same judgement the config-file path
            # makes below. Left to main()'s catch-all, `OUTPUT_FORMAT=jsn`
            # would print under "Unexpected error", which tells an analyst
            # to file a bug about their own typo.
            raise ConfigFileError(
                f"The environment states values that cannot be used: {_rejections(exc)}"
            ) from exc
        # Snapshot what the environment (or a keyword argument) supplied
        # before any config file is layered on top: applying the file marks
        # those fields as set too, so asking later cannot tell the two
        # sources apart.
        from_environment = {
            name: set(getattr(settings, name).model_fields_set) for name in _config_sections()
        }
        if not settings.config_file:
            settings.config_file = _discover_config_file(settings.config_dir)
        settings._load_config_file(from_environment)
        if settings.rejected_config_values:
            # Refusing the run, rather than warning and carrying on with
            # the defaults. A value the file states and validation rejects
            # leaves the field holding whatever it shipped with, and for
            # `host` that default is ReversingLabs' public cloud — so an
            # empty or mistyped `a1000.host` would send the configured token
            # and every hash the analyst looked up to the vendor instead of
            # to their appliance. Which appliance is being talked to is not
            # something to get wrong quietly.
            raise ConfigFileError(
                f"{settings.config_file} states values that cannot be used: "
                + "; ".join(settings.rejected_config_values)
            )
        return settings

    def _load_config_file(self, protected: dict[str, set[str]]) -> None:
        """Layer the active profile of ``config_file`` over what is set."""
        if not self.config_file:
            return
        config_data = read_config_file(self.config_file)

        if config_data and self.profile in config_data:
            profile_data = config_data[self.profile]
            # ``read_config_file`` guards only the top level, so a profile
            # holding anything but a mapping has to be refused here rather
            # than raise out of ``_apply_config`` naming neither the file
            # nor the profile. ``default:`` with nothing under it is the
            # likeliest way to write one.
            if not isinstance(profile_data, dict):
                raise ConfigFileError(
                    f"{self.config_file}: profile '{self.profile}' must map section names to "
                    f"their settings, not a {type(profile_data).__name__}"
                )
            self._apply_config(profile_data, protected)
        elif config_data:
            # A mistyped --profile would otherwise fall back to the
            # defaults in silence, which for this tool means quietly
            # talking to a different appliance than intended.
            self.missing_profile = self.profile

    def _apply_config(
        self, config: dict[str, Any], protected: dict[str, set[str]] | None = None
    ) -> None:
        """Apply file-provided values, leaving ``protected`` fields alone.

        Documented precedence is CLI options -> environment variables ->
        config file -> defaults, so ``protected`` carries the per-section
        field names an env var already supplied; the file may only fill in
        what is still sitting on its default.

        Config files are user input: an unknown key (typo, stale example)
        must not crash every CLI invocation, so it is skipped instead. A
        known key holding an unusable value is the same kind of mistake
        and gets the same treatment — the field keeps its default and the
        key is recorded in ``rejected_config_values`` for the CLI to
        report, rather than travelling on to fail somewhere that cannot
        say which config line caused it.
        """
        protected = protected or {}
        for section_name in _config_sections():
            section = getattr(self, section_name)
            known_fields = type(section).model_fields
            keep = protected.get(section_name, set())
            section_values = config.get(section_name)
            if section_values is not None and not isinstance(section_values, dict):
                # A section holding a scalar or a list is the same class of
                # mistake as a key holding an unusable value, and gets the
                # same treatment.
                self.rejected_config_values.append(
                    f"{section_name}: must map setting names to values, "
                    f"not a {type(section_values).__name__}"
                )
                continue
            for key, value in (section_values or {}).items():
                if key not in known_fields or key in keep:
                    continue
                try:
                    setattr(section, key, value)
                except ValidationError as exc:
                    reason = exc.errors()[0]["msg"] if exc.errors() else "invalid value"
                    self.rejected_config_values.append(f"{section_name}.{key}: {reason}")

    def profile_dump(self, redact: bool = True) -> dict[str, Any]:
        """Serializable snapshot of the per-profile configuration sections.

        Redacted by default because the one caller that shows this to a
        human, ``config show``, writes it to **stdout**: without it
        ``-o json config show > cfg.json`` puts a live appliance token in a
        0644 file, read out of one deliberately kept at 0600. Only the
        writers, which have to store what was typed, ask for
        ``redact=False``.

        ``mode="json"`` because what this feeds is a YAML file and a stdout
        dump, and a plain ``model_dump`` hands back the ``OutputFormat``
        member itself — which ``yaml.dump`` writes as a
        ``!!python/object/apply`` tag that ``safe_load`` then refuses,
        making ``config save`` produce a config file the next run cannot
        read.
        """
        sections = {
            name: getattr(self, name).model_dump(exclude_none=True, mode="json")
            for name in _config_sections()
        }
        if not redact:
            return sections
        return {name: redact_section(values) for name, values in sections.items()}

    def save_config(self, path: Path | None = None) -> None:
        """Save current configuration to file."""
        config_file = path or self.config_dir / "config.yaml"

        config_data = read_config_file(config_file)
        config_data[self.profile] = self.profile_dump(redact=False)

        write_private_yaml(config_file, config_data)


@lru_cache
def _cached_settings(working_directory: Path, arguments: tuple[tuple[str, Any], ...]) -> Settings:
    """Parse the environment and config file once per set of arguments.

    ``working_directory`` is read nowhere in this body and is not passed
    on: it is here to be part of the cache key. ``Settings.load`` discovers
    ``./config.yaml`` and ``./.rl-cli.yaml`` relative to ``Path.cwd()``, so
    with that outside the key a caller that changes directory between two
    calls — the library use ``rl_cli/__init__.py`` advertises by exporting
    ``Settings`` — silently gets the first directory's appliance back for
    the life of the process.

    ``arguments`` is the caller's keyword arguments already sorted, so that
    the same settings asked for with the keywords written in two orders is
    one cache entry rather than two independent objects.
    """
    return Settings.load(**dict(arguments))


def get_settings(**kwargs: Any) -> Settings:
    """Get a settings instance.

    Every caller gets its own deep copy: ``config init`` fills the object
    in from its prompts before saving it, and one shared instance would
    leak those edits into every later caller.
    """
    return _cached_settings(Path.cwd(), tuple(sorted(kwargs.items()))).model_copy(deep=True)


def clear_settings_cache() -> None:
    """Drop the memoised parses so a test can be given a fresh environment.

    The cache key covers the arguments and the working directory, but not
    the contents of ``$HOME``, the environment, or a config file edited in
    place — so anything that changes one of those under a running process
    has to say so here. The suite, which rewrites all three between tests,
    is what this exists for.
    """
    _cached_settings.cache_clear()
