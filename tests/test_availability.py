"""Tests for APIAvailabilityChecker: credential gating, probes, and cache."""

import ast
import hashlib
import importlib
import json
import re
import stat
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Literal
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from rl_cli.cli.main import cli
from rl_cli.config import Settings
from rl_cli.models.availability import ProbeGrade, grade_of
from rl_cli.render.output import RichOutput
from rl_cli.services.a1000 import A1000MetadataService, A1000Session
from rl_cli.services.a1000 import session as session_module
from rl_cli.services.availability import (
    CACHE_DURATION,
    APIAvailabilityChecker,
    Availability,
    ServiceStatus,
    _has_shape,
    _is_probe_run,
    _ProbeOutput,
    _stated_as,
)
from rl_cli.services.credentials import is_real_credential
from rl_cli.services.protocols import NullNotifier
from rl_cli.services.titanium_cloud import TitaniumCloudService
from rl_cli.storage.probe_cache import ProbeCache

# This suite is the probe's own; without the marker the autouse fixture in
# conftest hands it the stub every other suite gets, and every claim below
# would then be made about that stub.
pytestmark = pytest.mark.real_availability_probe

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLE_FILES = ("config.example.yaml", ".env.example")


def _shipped_placeholders(name: str) -> set[str]:
    """Every credential stand-in one example file ships.

    Read out of the file rather than restated here, so a stand-in the
    examples gain or reword is one the probe is tested against. A regex over
    the raw text rather than a YAML or dotenv parse, because the
    username/password stand-ins ship commented out and no parser sees them.

    Per file, not pooled across both. Unioned, a stand-in reworded in one
    file alone left the union intact and every gate below green -- so
    ``config.example.yaml`` could ship ``username: "<ticloud-username>"``,
    which ``is_real_credential`` counts as a real one, and a copied example
    would spend a probe on it and cache the rejection for a day.
    """
    return set(re.findall(r"your_[a-z0-9_]+", (_REPO_ROOT / name).read_text(encoding="utf-8")))


BY_FILE = {name: _shipped_placeholders(name) for name in _EXAMPLE_FILES}
SHIPPED = set().union(*BY_FILE.values())

# Which stand-in spells which credential. A field is checked against every
# example file, so each file has to carry its own.
CREDENTIAL_FIELDS = (
    ("titanium_cloud.username", lambda p: p.endswith("ticloud_username")),
    ("titanium_cloud.password", lambda p: p.endswith("ticloud_password")),
    ("a1000.token", lambda p: "token" in p),
    ("a1000.username", lambda p: p.endswith("username") and "ticloud" not in p),
    ("a1000.password", lambda p: p.endswith("password") and "ticloud" not in p),
)

# Pooled across both files, for the gates that feed a shipped stand-in to a
# probe: whichever file it came from, it must not be authenticated with.
# The per-file check above is what asserts each file carries its own.
TICLOUD_USERNAMES = sorted(p for p in SHIPPED if p.endswith("ticloud_username"))
TICLOUD_PASSWORDS = sorted(p for p in SHIPPED if p.endswith("ticloud_password"))
A1000_TOKENS = sorted(p for p in SHIPPED if "token" in p)
A1000_USERNAMES = sorted(p for p in SHIPPED if p.endswith("username") and "ticloud" not in p)
A1000_PASSWORDS = sorted(p for p in SHIPPED if p.endswith("password") and "ticloud" not in p)

# The whitespace a stand-in can reach the settings wearing.
# ``config.example.yaml`` quotes every credential, so a stray space inside
# the quotes survives the YAML parse, and a *quoted* ``.env`` value keeps
# its padding too (an unquoted one is stripped by dotenv). ``Settings``
# sets no ``str_strip_whitespace``, so the padded string is what both the
# probe and the authentication are handed.
PADDINGS = ("{} ", " {}", "{}\n", "\t{}", "  {}  ")

# How a credential can be spelled in a config file, and what each spelling
# is worth. ``stand-in`` is the example's own string, left where a copied
# example put it.
SPELLINGS = ("set", "stand-in", "absent")


def _spelled(spelling: str, *, stand_in: str, real: str) -> str | None:
    """One credential, written the way ``spelling`` says."""
    return {"set": real, "stand-in": stand_in, "absent": None}[spelling]


def _holds_a_stand_in(sent: dict[str, Any]) -> list[str]:
    """Every shipped stand-in among the values an SDK client was built from."""
    return [str(value) for value in sent.values() if isinstance(value, str) and value in SHIPPED]


def _imported_names(module) -> set[str]:
    """Every module ``module`` imports, read out of its source.

    Read rather than inspected at runtime: by the time this suite runs,
    every module in the package is already in ``sys.modules``, so nothing
    about what one of them reaches for can be observed from there.
    """
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _a_probe_run() -> Availability:
    """A well-formed ``Availability``, for the tests that store one."""

    def probe():
        return {
            "status": ServiceStatus.UNAVAILABLE.value,
            "message": "",
            "credentials_configured": False,
            "api_accessible": False,
        }

    return {
        "timestamp": datetime.now().isoformat(),
        "titanium_cloud": probe(),
        "a1000": probe(),
        "summary": {"services_available": 0, "services_total": 2},
    }


def _a_damageable_probe_run() -> dict[str, Any]:
    """The same document, typed so a test can take a required key out of it.

    ``Availability`` is a ``TypedDict``, and refusing ``del`` on a required
    key is the point of it everywhere else -- but these tests exist to
    damage a stored document and watch the checker re-probe. The round trip
    through JSON is how the document actually reaches disk, so what they
    damage is what the cache would have handed back.
    """
    damaged: dict[str, Any] = json.loads(json.dumps(_a_probe_run()))
    return damaged


@pytest.fixture
def settings(tmp_path):
    s = Settings(cache_dir=tmp_path / "cache", config_dir=tmp_path / "config")
    s.titanium_cloud.username = None
    s.titanium_cloud.password = None
    s.a1000.token = None
    s.a1000.username = None
    s.a1000.password = None
    return s


@pytest.fixture
def checker(settings):
    return APIAvailabilityChecker(A1000Session(settings))


def _cache(settings, verbose: bool = False):
    """The cache the checker builds, reached the way the checker reaches it.

    The file itself is a ``ProbeCache`` under ``rl_cli.storage`` — keeping a
    JSON file is not one of the probe's jobs — so these tests ask the
    checker for the one it wired up rather than assembling a second.
    """
    return APIAvailabilityChecker(A1000Session(settings), RichOutput(), verbose=verbose).cache


class TestTheExamplesStillShipAStandInForEveryCredential:
    """The gate below is parametrised over these, so an empty set proves nothing.

    Five stand-ins across two files, one per credential of each service. A
    field that loses its stand-in silently drops a whole parametrised gate,
    so the counts are asserted before the gates use them.
    """

    @pytest.mark.parametrize("example", _EXAMPLE_FILES)
    @pytest.mark.parametrize(
        ("field", "matches"), CREDENTIAL_FIELDS, ids=[field for field, _ in CREDENTIAL_FIELDS]
    )
    def test_the_field_has_a_stand_in_in_each_example(self, example, field, matches):
        """Each file on its own, so one reworded there cannot hide behind the other."""
        assert any(matches(p) for p in BY_FILE[example]), (
            f"{field} has no your_… stand-in in {example}"
        )

    @pytest.mark.parametrize("placeholder", sorted(SHIPPED))
    def test_no_shipped_stand_in_counts_as_a_supplied_credential(self, placeholder):
        """The one rule both probes gate on, checked against the shipped strings."""
        assert not is_real_credential(placeholder), (
            f"{placeholder!r} ships in an example file but reads as a real credential, "
            "so a copied example spends a probe on it and the rejection is cached for a day"
        )

    def test_a_real_credential_is_supplied(self):
        """The prefix rule must not swallow anything a user would actually paste."""
        assert is_real_credential("a1000-token-9f3c")
        assert is_real_credential("analyst@example.com")


class TestThePlaceholderRuleIsReachableByWhoeverPicksTheCredential:
    """Detecting a stand-in is worth nothing in the module that only reports.

    The probe says "credentials configured" and something else decides
    which credential to authenticate with. Split, a config carrying
    ``token: your_a1000_api_token_here`` beside a real username and
    password is reported as configured under username/password and then
    authenticated with the stand-in, and every command in the run fails
    against a verdict cached for a day.

    So the rule lives in :mod:`rl_cli.services.credentials`, which imports
    nothing from this package and can therefore be imported by the session
    that opens the connection — this module cannot, since it imports that
    session itself.
    """

    def test_the_rule_is_importable_without_reaching_the_probe(self):
        """A leaf module: importable from anything, including its own callers."""
        module = importlib.import_module("rl_cli.services.credentials")

        assert module.is_real_credential is is_real_credential
        assert not any(name.startswith("rl_cli") for name in _imported_names(module))

    @pytest.mark.parametrize("token", A1000_TOKENS)
    def test_a_stand_in_token_beside_real_credentials_is_not_a_token(self, settings, token):
        """What the session has to agree with: bare truthiness sends this one."""
        settings.a1000.token = token
        settings.a1000.username = "analyst"
        settings.a1000.password = "real-pass"

        assert not is_real_credential(settings.a1000.token)
        assert is_real_credential(settings.a1000.username)

    @pytest.mark.parametrize("token", A1000_TOKENS)
    def test_the_probe_reports_the_authentication_a_stand_in_token_leaves(
        self, settings, checker, token, monkeypatch
    ):
        settings.a1000.token = token
        settings.a1000.username = "analyst"
        settings.a1000.password = "real-pass"
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: True)

        assert "username/password" in checker._check_a1000()["message"]


class TestAStandInWithWhitespaceRoundItIsStillAStandIn:
    """One keystroke inside the quotes must not turn a stand-in into a credential.

    ``config.example.yaml`` quotes every credential, so ``token:
    "your_a1000_api_token_here "`` is a plausible typo that YAML hands on
    padding and all, and a quoted ``.env`` value keeps its padding too.
    Judged as-is, the padded stand-in is a credential the user set: the
    probe spends a call on it, reports the service configured, caches that
    for a day, and the session authenticates with it.
    """

    @pytest.mark.parametrize("padding", PADDINGS)
    @pytest.mark.parametrize("placeholder", sorted(SHIPPED))
    def test_a_padded_stand_in_is_not_a_supplied_credential(self, placeholder, padding):
        padded = padding.format(placeholder)

        assert not is_real_credential(padded), (
            f"{padded!r} is a shipped stand-in with whitespace round it, "
            "and reads as a credential the user set"
        )

    @pytest.mark.parametrize("blank", ["", " ", "\t", "\n", "   ", " \t\n "])
    def test_whitespace_alone_is_nothing_to_authenticate_with(self, blank):
        """The same reading as empty: there is no credential in it."""
        assert not is_real_credential(blank)

    def test_padding_does_not_make_a_real_credential_unusable(self):
        """The rule refuses stand-ins, not everything that has been typed loosely."""
        assert is_real_credential(" a1000-token-9f3c ")
        assert is_real_credential("analyst@example.com\n")

    @pytest.mark.parametrize("padding", PADDINGS)
    @pytest.mark.parametrize("token", A1000_TOKENS)
    def test_a_padded_stand_in_token_leaves_the_a1000_unconfigured(
        self, settings, checker, token, padding
    ):
        settings.a1000.token = padding.format(token)

        assert checker._check_a1000()["status"] == ServiceStatus.UNAVAILABLE.value

    @pytest.mark.parametrize("padding", PADDINGS)
    @pytest.mark.parametrize("username", TICLOUD_USERNAMES)
    def test_a_padded_stand_in_leaves_titanium_cloud_unconfigured(
        self, settings, checker, username, padding
    ):
        settings.titanium_cloud.username = padding.format(username)
        settings.titanium_cloud.password = "real-pass"

        assert checker._check_titanium_cloud()["status"] == ServiceStatus.UNAVAILABLE.value


class TestTheProbeAndTheAuthenticationAgree:
    """The verdict that is cached for a day, and the credential that is sent.

    Two readings of one config, and nothing keeps them in step but the
    shared rule. A config half-filled from an example is where they came
    apart: ``check-access`` refused the stand-in and cached
    "not configured (placeholder values)" for 24 h, while every command in
    the run went on authenticating with ``your_ticloud_username``.

    The whole matrix of spellings, not the one case that was reported: a
    stand-in in either half of a credential pair is the same mistake.
    """

    def _titanium_cloud_sends(self, settings) -> dict[str, Any]:
        """The keyword arguments every ``ticloud`` SDK handle is built from."""
        config: dict[str, Any] = TitaniumCloudService(settings)._api_config
        return config

    def _a1000_sends(self, settings, monkeypatch) -> dict[str, Any]:
        """The keyword arguments the A1000 SDK client is constructed with."""
        captured: dict[str, Any] = {}

        def a1000(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(session_module, "A1000", a1000)
        A1000Session(settings).connect()
        return captured

    @pytest.mark.parametrize("password", SPELLINGS)
    @pytest.mark.parametrize("username", SPELLINGS)
    def test_titanium_cloud_authenticates_with_what_the_probe_counted(
        self, settings, checker, monkeypatch, username, password
    ):
        settings.titanium_cloud.username = _spelled(
            username, stand_in=TICLOUD_USERNAMES[0], real="analyst"
        )
        settings.titanium_cloud.password = _spelled(
            password, stand_in=TICLOUD_PASSWORDS[0], real="s3cret"
        )
        monkeypatch.setattr(TitaniumCloudService, "get_file_reputation", lambda self, h: {"rl": {}})

        probe = checker._check_titanium_cloud()
        sent = self._titanium_cloud_sends(settings)

        assert not _holds_a_stand_in(sent), (
            f"the probe reported credentials_configured={probe['credentials_configured']} "
            f"and the SDK handle was built with {_holds_a_stand_in(sent)}"
        )
        assert bool(sent["username"] and sent["password"]) == probe["credentials_configured"]

    @pytest.mark.parametrize("password", SPELLINGS)
    @pytest.mark.parametrize("username", SPELLINGS)
    @pytest.mark.parametrize("token", SPELLINGS)
    def test_the_a1000_authenticates_with_what_the_probe_counted(
        self, settings, checker, monkeypatch, token, username, password
    ):
        settings.a1000.token = _spelled(token, stand_in=A1000_TOKENS[0], real="t0ken")
        settings.a1000.username = _spelled(username, stand_in=A1000_USERNAMES[0], real="analyst")
        settings.a1000.password = _spelled(password, stand_in=A1000_PASSWORDS[0], real="s3cret")
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: True)

        probe = checker._check_a1000()
        sent = self._a1000_sends(settings, monkeypatch)

        assert not _holds_a_stand_in(sent), (
            f"the probe reported credentials_configured={probe['credentials_configured']} "
            f"and the SDK client was built with {_holds_a_stand_in(sent)}"
        )
        authenticates = bool(sent.get("token") or (sent.get("username") and sent.get("password")))
        assert authenticates == probe["credentials_configured"]


class TestTitaniumCloudProbe:
    def test_no_credentials_is_unavailable(self, checker):
        result = checker._check_titanium_cloud()
        assert result["status"] == ServiceStatus.UNAVAILABLE.value
        assert not result["credentials_configured"]

    @pytest.mark.parametrize(
        "username, password",
        [("real-user", None), (None, "real-pass")],
        ids=["password-missing", "username-missing"],
    )
    def test_one_half_of_the_pair_missing_is_unconfigured_not_a_placeholder(
        self, settings, checker, username, password
    ):
        """Either credential absent is "none configured", not the placeholder
        message a real-but-stand-in value earns -- the pair is needed, so a
        half-filled pair is empty, not fake."""
        settings.titanium_cloud.username = username
        settings.titanium_cloud.password = password
        result = checker._check_titanium_cloud()
        assert result["status"] == ServiceStatus.UNAVAILABLE.value
        assert result["message"] == "No TitaniumCloud credentials configured"

    @pytest.mark.parametrize("username", TICLOUD_USERNAMES)
    def test_a_shipped_username_stand_in_is_unavailable(self, settings, checker, username):
        settings.titanium_cloud.username = username
        settings.titanium_cloud.password = "real-pass"
        result = checker._check_titanium_cloud()
        assert result["status"] == ServiceStatus.UNAVAILABLE.value

    @pytest.mark.parametrize("password", TICLOUD_PASSWORDS)
    def test_a_shipped_password_stand_in_is_unavailable(self, settings, checker, password):
        settings.titanium_cloud.username = "real-user"
        settings.titanium_cloud.password = password
        result = checker._check_titanium_cloud()
        assert result["status"] == ServiceStatus.UNAVAILABLE.value

    def test_successful_probe_is_available(self, settings, checker, monkeypatch):
        settings.titanium_cloud.username = "user"
        settings.titanium_cloud.password = "pass"
        monkeypatch.setattr(TitaniumCloudService, "get_file_reputation", lambda self, h: {"rl": {}})
        result = checker._check_titanium_cloud()
        assert result["status"] == ServiceStatus.AVAILABLE.value
        assert result["api_accessible"]

    def test_failed_probe_is_error(self, settings, checker, monkeypatch):
        settings.titanium_cloud.username = "user"
        settings.titanium_cloud.password = "pass"
        monkeypatch.setattr(TitaniumCloudService, "get_file_reputation", lambda self, h: None)
        result = checker._check_titanium_cloud()
        assert result["status"] == ServiceStatus.ERROR.value
        assert "no data and reported no error" in result["message"]


class TestA1000Probe:
    def test_no_credentials_is_unavailable(self, checker):
        result = checker._check_a1000()
        assert result["status"] == ServiceStatus.UNAVAILABLE.value

    @pytest.mark.parametrize("token", A1000_TOKENS)
    def test_a_shipped_token_stand_in_is_unavailable(self, settings, checker, token):
        settings.a1000.token = token
        result = checker._check_a1000()
        assert result["status"] == ServiceStatus.UNAVAILABLE.value

    @pytest.mark.parametrize("password", A1000_PASSWORDS)
    def test_a_shipped_password_stand_in_is_unavailable(self, settings, checker, password):
        settings.a1000.username = "real-user"
        settings.a1000.password = password
        result = checker._check_a1000()
        assert result["status"] == ServiceStatus.UNAVAILABLE.value

    def test_token_probe_success(self, settings, checker, monkeypatch):
        settings.a1000.token = "real-token"
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: True)
        result = checker._check_a1000()
        assert result["status"] == ServiceStatus.AVAILABLE.value
        assert "token" in result["message"]

    def test_failed_probe_is_error(self, settings, checker, monkeypatch):
        settings.a1000.token = "real-token"
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: False)
        monkeypatch.setattr(A1000MetadataService, "get_classification", lambda self, h: None)
        result = checker._check_a1000()
        assert result["status"] == ServiceStatus.ERROR.value
        assert "no data and reported no error" in result["message"]


class TestAFailedProbeSaysWhy:
    """The probe message is the whole answer a user gets about a failure.

    It used to refer them to service logs, and this CLI writes none: the
    probe silences the services it calls so an SDK error is not printed on
    every invocation, which threw away the only account of what happened.
    """

    def test_an_unreachable_appliance_names_the_failure(self, settings, checker, monkeypatch):
        settings.a1000.token = "real-token"

        def refuse(session):
            raise ConnectionError("[Errno 61] Connection refused by a1000.invalid")

        monkeypatch.setattr(A1000Session, "_open", refuse)

        result = checker._check_a1000()

        assert result["status"] == ServiceStatus.ERROR.value
        assert "Connection refused by a1000.invalid" in result["message"]
        assert "logs" not in result["message"]

    def test_an_appliance_that_stops_answering_names_the_call_that_failed(
        self, settings, checker, monkeypatch
    ):
        """Token auth sends nothing until a call is made, so the session remembers no failure."""
        settings.a1000.token = "real-token"

        class _GoesQuiet:
            def test_connection(self):
                raise TimeoutError("read timed out after 300s")

            def get_classification_v3(self, hash_value):
                raise TimeoutError("read timed out after 300s")

        def answer_then_hang(session):
            session.client = _GoesQuiet()

        monkeypatch.setattr(A1000Session, "_open", answer_then_hang)

        result = checker._check_a1000()

        assert checker.session.failure is None
        assert result["status"] == ServiceStatus.ERROR.value
        assert "read timed out after 300s" in result["message"]

    def test_titanium_cloud_names_what_the_api_said(self, settings, checker, monkeypatch):
        settings.titanium_cloud.username = "user"
        settings.titanium_cloud.password = "pass"

        class _Rejecting:
            def get_file_reputation(self, **kwargs):
                raise RuntimeError("HTTP 401: Unauthorized")

        monkeypatch.setattr(
            TitaniumCloudService, "_file_reputation", property(lambda self: _Rejecting())
        )

        result = checker._check_titanium_cloud()

        assert result["status"] == ServiceStatus.ERROR.value
        assert "HTTP 401: Unauthorized" in result["message"]


class TestTheProbeMessageCannotDriveATerminal:
    """The appliance writes this message, and `-o table` prints it verbatim.

    The probe message is the one string in this module built out of what
    a remote host said, it is cached for 24 h and replayed ahead of every
    command, and it does not only reach the user through the notifier:
    ``config check-access`` and ``config show -a`` put it on stdout
    through ``OutputFormatter``, whose table renderer flattens a dict to
    ``(field, value)`` tuples that its own sanitiser walks straight past.
    So the stripping has to happen here, where the message is made.
    """

    ESCAPES = "\x1b[2J\x1b]0;pwned\x07"

    def test_an_a1000_error_reaches_the_message_without_its_escapes(
        self, settings, checker, monkeypatch
    ):
        settings.a1000.token = "real-token"
        escapes = self.ESCAPES

        class _Rude:
            def test_connection(self):
                raise TimeoutError(f"{escapes}read timed out")

            def get_classification_v3(self, hash_value):
                raise TimeoutError(f"{escapes}read timed out")

        def answer_then_shout(session):
            session.client = _Rude()

        monkeypatch.setattr(A1000Session, "_open", answer_then_shout)

        result = checker._check_a1000()

        assert "read timed out" in result["message"]
        assert "\x1b" not in result["message"]
        assert "\x07" not in result["message"]

    def test_a_titanium_cloud_error_reaches_the_message_without_its_escapes(
        self, settings, checker, monkeypatch
    ):
        settings.titanium_cloud.username = "user"
        settings.titanium_cloud.password = "pass"
        escapes = self.ESCAPES

        class _Rejecting:
            def get_file_reputation(self, **kwargs):
                raise RuntimeError(f"{escapes}HTTP 401: Unauthorized")

        monkeypatch.setattr(
            TitaniumCloudService, "_file_reputation", property(lambda self: _Rejecting())
        )

        result = checker._check_titanium_cloud()

        assert "HTTP 401: Unauthorized" in result["message"]
        assert "\x1b" not in result["message"]
        assert "\x07" not in result["message"]


class TestOnlyMeasuredFactsAreReported:
    """A probe answers for the call it made, and for no key nothing writes."""

    MEASURED: ClassVar[set[str]] = {"status", "message", "credentials_configured", "api_accessible"}

    def test_the_a1000_probe_reports_only_what_it_measured(self, settings, checker, monkeypatch):
        settings.a1000.token = "real-token"
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: True)

        assert set(checker._check_a1000()) == self.MEASURED

    def test_the_titanium_cloud_probe_reports_only_what_it_measured(
        self, settings, checker, monkeypatch
    ):
        settings.titanium_cloud.username = "user"
        settings.titanium_cloud.password = "pass"
        monkeypatch.setattr(TitaniumCloudService, "get_file_reputation", lambda self, h: {"rl": {}})

        assert set(checker._check_titanium_cloud()) == self.MEASURED

    def test_the_cache_carries_nothing_unmeasured(self, settings, checker, monkeypatch):
        settings.a1000.token = "real-token"
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: True)

        checker.check_all()

        written = checker.cache.path.read_text(encoding="utf-8")
        assert "test_results" not in written
        assert "available_methods" not in written


class TestEveryProbeAnswersWithAWholeMeasurement:
    """A probe returns a built ``ServiceProbe``; it does not patch one.

    Every exit path builds all four keys, so none can ship a status or a
    message the probe never measured.
    """

    def _probes(self, settings, checker, monkeypatch):
        """One probe from each service, on each of its three exit paths."""
        yield checker._check_titanium_cloud()
        yield checker._check_a1000()

        settings.titanium_cloud.username = TICLOUD_USERNAMES[0]
        settings.titanium_cloud.password = TICLOUD_PASSWORDS[0]
        settings.a1000.token = A1000_TOKENS[0]
        yield checker._check_titanium_cloud()
        yield checker._check_a1000()

        settings.titanium_cloud.username = "user"
        settings.titanium_cloud.password = "pass"
        settings.a1000.token = "real-token"
        monkeypatch.setattr(TitaniumCloudService, "get_file_reputation", lambda self, h: {"rl": {}})
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: True)
        yield checker._check_titanium_cloud()
        yield checker._check_a1000()

        monkeypatch.setattr(TitaniumCloudService, "get_file_reputation", lambda self, h: None)
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: False)
        monkeypatch.setattr(A1000MetadataService, "get_classification", lambda self, h: None)
        yield checker._check_titanium_cloud()
        yield checker._check_a1000()

    def test_every_path_reports_a_real_status_and_a_message(self, settings, checker, monkeypatch):
        statuses = {status.value for status in ServiceStatus}
        for probe in self._probes(settings, checker, monkeypatch):
            assert probe["status"] in statuses
            assert probe["message"]

    def test_a_failed_probe_still_reports_that_credentials_were_configured(
        self, settings, checker, monkeypatch
    ):
        """Otherwise "check your credentials" is the advice given to someone whose are fine."""
        settings.titanium_cloud.username = "user"
        settings.titanium_cloud.password = "pass"
        settings.a1000.token = "real-token"
        monkeypatch.setattr(TitaniumCloudService, "get_file_reputation", lambda self, h: None)
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: False)
        monkeypatch.setattr(A1000MetadataService, "get_classification", lambda self, h: None)

        for probe in (checker._check_titanium_cloud(), checker._check_a1000()):
            assert probe["status"] == ServiceStatus.ERROR.value
            assert probe["credentials_configured"]
            assert not probe["api_accessible"]

    def test_an_unconfigured_service_is_never_reported_as_reachable(self, checker):
        for probe in (checker._check_titanium_cloud(), checker._check_a1000()):
            assert probe["status"] == ServiceStatus.UNAVAILABLE.value
            assert not probe["credentials_configured"]
            assert not probe["api_accessible"]

    @pytest.mark.parametrize("username", A1000_USERNAMES)
    def test_a_shipped_username_is_treated_like_a_blank_one(self, settings, checker, username):
        """Both mean "nothing to authenticate with", so neither spends a probe."""
        settings.a1000.username = username
        settings.a1000.password = "pass"

        assert checker._check_a1000()["status"] == ServiceStatus.UNAVAILABLE.value


class TestCacheTroubleIsReportedOnlyWhenAsked:
    """An unwritable cache costs a re-probe, not an answer.

    So the default run stays quiet rather than prefixing every command
    with a filesystem complaint — but ``--verbose`` has to say it, or a
    cache that never persists looks exactly like one that works.
    """

    def _unwritable(self, settings, monkeypatch, verbose=False):
        cache = _cache(settings, verbose=verbose)
        monkeypatch.setattr(
            cache, "_make_dir", lambda: (_ for _ in ()).throw(OSError("read-only file system"))
        )
        return cache

    def test_a_failed_save_says_nothing_by_default(self, settings, monkeypatch, capsys):
        self._unwritable(settings, monkeypatch).save(_a_probe_run())

        assert capsys.readouterr() == ("", "")

    def test_a_failed_save_names_the_reason_under_verbose(self, settings, monkeypatch, capsys):
        self._unwritable(settings, monkeypatch, verbose=True).save(_a_probe_run())

        assert "read-only file system" in capsys.readouterr().err

    def test_a_failed_load_names_the_reason_under_verbose(self, settings, monkeypatch, capsys):
        cache = _cache(settings, verbose=True)
        cache._make_dir()
        cache.path.write_text("not json{")

        assert cache.load() is None
        assert "Failed to load availability cache" in capsys.readouterr().err


class TestCheckAllAndCache:
    def test_check_all_builds_summary_and_caches(self, settings, checker, monkeypatch):
        settings.a1000.token = "real-token"
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: True)

        availability = checker.check_all()

        assert availability["summary"]["services_available"] == 1
        assert availability["a1000"]["status"] == ServiceStatus.AVAILABLE.value
        assert availability["titanium_cloud"]["status"] == ServiceStatus.UNAVAILABLE.value
        assert checker.cache.path.exists()

    def test_second_call_uses_cache(self, settings, checker, monkeypatch):
        settings.a1000.token = "real-token"
        calls = []

        def probe(self: A1000MetadataService) -> bool:
            calls.append(1)
            return True

        monkeypatch.setattr(A1000MetadataService, "test_connection", probe)
        checker.check_all()
        checker.check_all()
        assert len(calls) == 1

    def test_force_bypasses_cache(self, settings, checker, monkeypatch):
        settings.a1000.token = "real-token"
        calls = []

        def probe(self: A1000MetadataService) -> bool:
            calls.append(1)
            return True

        monkeypatch.setattr(A1000MetadataService, "test_connection", probe)
        checker.check_all()
        checker.check_all(force=True)
        assert len(calls) == 2

    def test_corrupt_cache_is_ignored(self, checker):
        checker.cache.path.parent.mkdir(parents=True, exist_ok=True)
        checker.cache.path.write_text("not json{")
        assert checker.cache.load() is None

    def test_a_cache_written_by_the_previous_version_still_loads(
        self, settings, checker, monkeypatch
    ):
        """A cached answer outlives the upgrade by up to 24 h, keys and all."""
        settings.a1000.token = "real-token"
        old_shape = {
            "timestamp": datetime.now().isoformat(),
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
            "fingerprint": checker.cache.fingerprint(),
        }
        checker.cache.path.parent.mkdir(parents=True, exist_ok=True)
        checker.cache.path.write_text(json.dumps(old_shape))
        monkeypatch.setattr(
            A1000MetadataService, "test_connection", lambda self: pytest.fail("must not re-probe")
        )

        availability = checker.check_all()

        # Everything the old cache said survives - including keys this
        # version no longer writes - except the fingerprint, which is the
        # cache's own bookkeeping and never part of a probe's answer.
        assert availability == {k: v for k, v in old_shape.items() if k != "fingerprint"}
        assert "fingerprint" not in availability
        assert availability["a1000"]["available_methods"] == [
            "advanced_search_v3",
            "get_summary_report_v2",
        ]

        # And writing it back keeps them: the typed shape is an annotation
        # over the payload, not a filter applied to it, so a legacy cache
        # survives being loaded and stored again.
        checker.cache.save(availability)
        assert checker.cache.load() == availability

    def test_an_expired_cache_is_not_served(self, settings, checker, monkeypatch):
        """24 h, counted from the timestamp in the file."""
        settings.a1000.token = "real-token"
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: True)
        checker.check_all()

        stored = json.loads(checker.cache.path.read_text(encoding="utf-8"))
        just_inside = datetime.now() - checker.cache.duration + timedelta(minutes=1)
        checker.cache.path.write_text(json.dumps({**stored, "timestamp": just_inside.isoformat()}))
        assert checker.cache.load() is not None

        just_outside = datetime.now() - checker.cache.duration
        checker.cache.path.write_text(json.dumps({**stored, "timestamp": just_outside.isoformat()}))
        assert checker.cache.load() is None

    @pytest.mark.parametrize(
        "ahead", [timedelta(minutes=1), timedelta(hours=25)], ids=["just", "well"]
    )
    def test_a_cache_stamped_in_the_future_is_not_served_for_ever(
        self, settings, checker, monkeypatch, ahead
    ):
        """A clock that was ahead must not pin one answer indefinitely.

        The age is negative, which is below the duration as surely as a
        fresh answer is, so an upper bound on its own served this file
        until something deleted it. Reachable after a suspended VM resumes,
        a backwards NTP step or a timezone corrected after the fact.
        """
        settings.a1000.token = "real-token"
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: True)
        checker.check_all()

        stored = json.loads(checker.cache.path.read_text(encoding="utf-8"))
        stamped_ahead = datetime.now() + ahead
        checker.cache.path.write_text(
            json.dumps({**stored, "timestamp": stamped_ahead.isoformat()})
        )
        assert checker.cache.load() is None

        # And the re-probe replaces the timestamp, so the analyst does not
        # need ``--force`` to get out of it.
        assert datetime.fromisoformat(checker.check_all()["timestamp"]) <= datetime.now()
        assert checker.cache.load() is not None

    def test_clear_cache_removes_file(self, checker):
        checker.cache.path.parent.mkdir(parents=True, exist_ok=True)
        checker.cache.path.write_text(json.dumps({}))
        checker.clear_cache()
        assert not checker.cache.path.exists()

    def test_clearing_an_absent_cache_is_a_quiet_no_op(self, checker):
        """Nothing stored yet is not an error to forget."""
        assert not checker.cache.path.exists()

        checker.clear_cache()

        assert not checker.cache.path.exists()


class TestACachedDocumentIsCheckedBeforeItIsTrusted:
    """The fingerprint covers the configuration, not the shape.

    A release that probes a third service, renames a key or nests one
    differently leaves every installed copy holding a document written by
    the version before it, valid for another 24 h. Handed on unread, that
    document is annotated as this version's shape and read as one, so the
    renderer raises ``KeyError`` on a path mypy certified and the only
    remedy is a ``--force`` the analyst has no reason to guess.

    A document that does not carry what this version renders is a cache
    miss, not a failure: re-probe and overwrite it.
    """

    def _stored(self, checker, document) -> None:
        checker.cache.path.parent.mkdir(parents=True, exist_ok=True)
        checker.cache.path.write_text(
            json.dumps({**document, "fingerprint": checker.cache.fingerprint()})
        )

    @pytest.mark.parametrize(
        "missing",
        ["timestamp", "titanium_cloud", "a1000", "summary"],
    )
    def test_a_document_missing_a_key_this_version_renders_is_re_probed(
        self, checker, settings, missing
    ):
        """Every top-level key, so a service added later is covered by name."""
        run = _a_damageable_probe_run()
        del run[missing]
        self._stored(checker, run)

        availability = checker.check_all()

        assert missing in availability
        assert set(Availability.__required_keys__) <= availability.keys()

    def test_a_document_whose_probe_lost_a_field_is_re_probed(self, checker):
        """A probe is a shape too: the renderer reads all four of its keys."""
        run = _a_damageable_probe_run()
        del run["a1000"]["api_accessible"]
        self._stored(checker, run)

        availability = checker.check_all()

        assert "api_accessible" in availability["a1000"]

    def test_a_document_whose_summary_lost_a_field_is_re_probed(self, checker):
        run = _a_damageable_probe_run()
        del run["summary"]["services_total"]
        self._stored(checker, run)

        availability = checker.check_all()

        assert availability["summary"]["services_total"] == 2

    @pytest.mark.parametrize("document", [[], "availability", 3, None])
    def test_a_document_that_is_not_an_object_is_re_probed(self, checker, document):
        checker.cache.path.parent.mkdir(parents=True, exist_ok=True)
        checker.cache.path.write_text(json.dumps(document))

        assert checker.check_all()["summary"]["services_total"] == 2

    # A key stated with the wrong type, one per field of the document.
    # The reason for the guard — a document written by another release
    # outliving it — renames and re-types a field just as readily as it
    # adds one, and a value of the wrong type is the half that gets past a
    # check on presence alone.
    _WRONG_TYPES: ClassVar[list[tuple[tuple[str, ...], Any]]] = [
        (("timestamp",), 3),
        (("titanium_cloud",), "available"),
        (("a1000",), ["available"]),
        (("summary",), 2),
        (("a1000", "status"), 123),
        (("a1000", "message"), None),
        (("a1000", "credentials_configured"), "yes"),
        (("a1000", "api_accessible"), "no"),
        (("titanium_cloud", "status"), None),
        (("summary", "services_available"), "two"),
        (("summary", "services_total"), None),
    ]

    @staticmethod
    def _retyped(path: tuple[str, ...], value: Any) -> dict[str, Any]:
        """A stored probe run with the field at ``path`` restated as ``value``."""
        run = _a_damageable_probe_run()
        node: Any = run
        for step in path[:-1]:
            node = node[step]
        node[path[-1]] = value
        return run

    @pytest.mark.parametrize(("path", "value"), _WRONG_TYPES, ids=str)
    def test_a_document_stating_a_field_as_the_wrong_type_is_not_a_probe_run(self, path, value):
        """Read directly, so the claim is about the guard and not about the cache."""
        assert not _is_probe_run(self._retyped(path, value))

    @pytest.mark.parametrize(("path", "value"), _WRONG_TYPES, ids=str)
    def test_a_document_stating_a_field_as_the_wrong_type_is_re_probed(self, checker, path, value):
        self._stored(checker, self._retyped(path, value))

        availability = checker.check_all()

        assert isinstance(availability["summary"]["services_available"], int)
        assert isinstance(availability["summary"]["services_total"], int)
        assert isinstance(availability["a1000"]["message"], str)

    def test_a_whole_probe_run_is_still_a_cache_hit(self, checker):
        """The guard has to let the document this version writes through."""
        assert _is_probe_run(_a_damageable_probe_run())

    def test_a_key_an_older_version_wrote_is_still_welcome(self, checker):
        """Extra keys are an answer the user is entitled to, not damage."""
        run = _a_damageable_probe_run()
        run["retired_service"] = {"status": "gone"}
        run["a1000"]["latency_ms"] = 12

        assert _is_probe_run(run)

    def test_a_summary_stated_as_text_does_not_crash_check_access(self, tmp_path):
        """What the guard is for, at the reader that raised.

        ``check_access`` compares ``services_available`` with 0 on a line
        mypy certified, so a cached ``"two"`` printed a panel reading
        "Services Available: two/None", two green service lines, and then a
        ``TypeError`` — with ``--force``, which nothing in that traceback
        suggests, as the only remedy.
        """
        settings = Settings(cache_dir=Path.home() / ".cache" / "rl-cli", config_dir=tmp_path)
        checker = APIAvailabilityChecker(A1000Session(settings))
        run = _a_damageable_probe_run()
        run["summary"] = {"services_available": "two", "services_total": None}
        self._stored(checker, run)

        result = CliRunner().invoke(cli, ["config", "check-access"])

        assert result.exit_code == 0, result.output
        assert "Services Available: 0/2" in result.output


class _ProbeRunStatingAList(Availability):
    """A probe run of a release that reports which services it probed."""

    services: list[str]


class _ProbeRunStatingAnything(Availability):
    """A probe run carrying a field whose annotation promises nothing."""

    extra: Any


class _ProbeRunStatingAnOptionalNote(Availability):
    """A probe run carrying a field a release may leave null."""

    note: str | None


class TestAnAnnotationTheGuardCannotIsinstanceIsNotACrash:
    """The guard reads the annotations, so it meets whatever they say.

    ``_stated_as`` ends in ``isinstance(value, hint)``, and the interpreter
    refuses a parameterised generic as its second argument:
    ``isinstance([], list[str])`` is ``TypeError: isinstance() argument 2
    cannot be a parameterized generic``, and ``typing.Any`` is refused too.
    ``check_all`` calls the guard with no ``try`` around it, so the release
    that adds ``services: list[str]`` to ``Availability`` — the natural way
    to say which services a run covered — turns every cached run into a
    traceback on every command, where the whole point of the guard is that
    a document this version cannot read is a cache miss.

    The reason it is checked here rather than by editing the shipped
    shapes: the trap is sprung by the *next* field anyone adds, and these
    stand in for it exactly, being ``Availability`` plus one field.
    """

    def _run(self, **fields: Any) -> dict[str, Any]:
        return {**_a_damageable_probe_run(), **fields}

    def test_a_field_stated_as_a_parameterised_generic_is_read_not_raised_on(self):
        assert _has_shape(self._run(services=["titanium_cloud"]), _ProbeRunStatingAList)

    def test_a_parameterised_generic_still_refuses_a_value_of_another_kind(self):
        """The container the annotation names is checked; the crash was not a check."""
        assert not _has_shape(self._run(services="titanium_cloud"), _ProbeRunStatingAList)

    def test_a_field_stated_as_anything_takes_whatever_the_document_carried(self):
        """``Any`` is satisfied by every value there is, including ``None``."""
        assert _has_shape(self._run(extra=None), _ProbeRunStatingAnything)
        assert _has_shape(self._run(extra={"nested": [1]}), _ProbeRunStatingAnything)

    def test_a_field_stated_as_anything_must_still_be_there(self):
        """``Any`` says nothing about the type and everything about the key."""
        assert not _has_shape(_a_damageable_probe_run(), _ProbeRunStatingAnything)

    def test_an_optional_field_takes_either_arm_and_nothing_else(self):
        assert _has_shape(self._run(note="probed twice"), _ProbeRunStatingAnOptionalNote)
        assert _has_shape(self._run(note=None), _ProbeRunStatingAnOptionalNote)
        assert not _has_shape(self._run(note=3), _ProbeRunStatingAnOptionalNote)

    def test_the_shipped_shape_is_unchanged_by_any_of_it(self):
        """The three above are stand-ins; this is the document itself."""
        assert _is_probe_run(_a_damageable_probe_run())

    def test_an_annotation_isinstance_cannot_judge_is_left_standing(self):
        """A Literal is neither class nor container; isinstance raises on it.

        The guard must not — an unreadable annotation is a field left
        unjudged, not damage — so the value is taken as it came.
        """
        assert _stated_as("whatever", Literal["a", "b"])


class TestTheCacheIsAFileHelperNotAProbe:
    """Keeping a JSON file was the only I/O here that was not a network call.

    It lives beside the other file helpers now, knowing nothing about
    settings or services: the checker hands it a path, how long an answer
    is good for, and a callable that says which configuration the answer
    belongs to.
    """

    def test_the_checker_wires_up_the_shared_cache(self, settings, checker):
        assert isinstance(checker.cache, ProbeCache)
        assert checker.cache.path == settings.cache_dir / "api_availability.json"
        assert checker.cache.duration == CACHE_DURATION

    def test_the_identity_still_follows_the_settings_it_was_built_from(self, settings, checker):
        """Captured values would freeze the answer to the appliance at start-up."""
        before = checker.cache.fingerprint()
        settings.a1000.host = "https://somewhere.else"

        assert checker.cache.fingerprint() != before

    def test_a_pipe_in_a_credential_does_not_collide_with_a_shifted_boundary(self, tmp_path):
        """A password may hold the ``|`` the identity was once joined on.

        ``"|".join`` let ``[..., "alice", "x|y"]`` and ``[..., "alice|x",
        "y"]`` hash alike, so one configuration could be served the answer
        measured for another. The two must fingerprint apart.
        """
        path = tmp_path / "api_availability.json"

        def cache_for(identity):
            return ProbeCache(path, lambda: identity, duration=CACHE_DURATION)

        one = cache_for(["prod", "tc.example", "alice", "x|y"])
        other = cache_for(["prod", "tc.example", "alice|x", "y"])

        assert one.fingerprint() != other.fingerprint()

    def test_a_failed_clear_is_reported_without_aborting(self, settings, monkeypatch):
        output = MagicMock()
        checker = APIAvailabilityChecker(A1000Session(settings), output, verbose=True)
        checker.cache.path.parent.mkdir(parents=True, exist_ok=True)
        checker.cache.path.write_text("{}")

        def refuse_unlink(self, *, missing_ok=False):
            raise PermissionError("cache is locked")

        monkeypatch.setattr(Path, "unlink", refuse_unlink)

        checker.clear_cache()

        assert checker.cache.path.exists()
        assert "Failed to clear availability cache" in output.warning.call_args.args[0]


class TestQuietProbeOutput:
    """The probe notifier stays silent without being a terminal renderer."""

    def test_is_not_a_console_renderer(self):
        quiet = _ProbeOutput()
        assert not isinstance(quiet, RichOutput)
        assert not hasattr(quiet, "console")

    def test_it_is_the_silent_notifier(self):
        """Silence is ``NullNotifier``'s job; a copy of it drifted out of contract once.

        ``info``, ``warning`` and the do-nothing spinner were duplicated
        here verbatim, which is how a stand-in stops matching the thing
        it stands in for without anything noticing. Inheriting them back
        is what keeps the two in step; the two tests below drive them.
        """
        assert isinstance(_ProbeOutput(), NullNotifier)

    def test_the_checker_needs_nothing_beyond_the_notifier(self, settings, monkeypatch, capsys):
        """No rendering left in the checker, so a bare Notifier must suffice."""
        checker = APIAvailabilityChecker(A1000Session(settings), output=_ProbeOutput())
        settings.a1000.token = "real-token"
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: True)

        assert checker.check_all()["a1000"]["status"] == ServiceStatus.AVAILABLE.value
        assert checker.cache.load() is not None
        assert capsys.readouterr() == ("", "")

    def test_a_service_using_it_prints_nothing(self, settings, capsys):
        service = A1000Session(settings).service(A1000MetadataService, _ProbeOutput())

        service.handle_error(RuntimeError("boom"), "probe")
        service.output.warning("noisy")
        service.output.info("chatty")
        # The same two members wait_for_analysis polls through, so this
        # fails if the silent stand-in drifts out of the Spinner contract.
        with service.output.progress_spinner("working") as progress:
            progress.advance(progress.task_ids[0])

        assert capsys.readouterr() == ("", "")


class TestCacheIsPerConfiguration:
    """One cache file served every profile, including other appliances."""

    def _checker(self, settings, host, token, profile="default"):
        settings.a1000.host = host
        settings.a1000.token = token
        settings.profile = profile
        return APIAvailabilityChecker(A1000Session(settings))

    def _cache_a_result(self, checker, monkeypatch, available: bool):
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: available)
        monkeypatch.setattr(A1000MetadataService, "get_classification", lambda self, h: None)
        monkeypatch.setattr(TitaniumCloudService, "get_file_reputation", lambda self, h: None)
        return checker.check_all()

    def test_another_host_does_not_reuse_the_cached_answer(self, settings, monkeypatch):
        broken = self._checker(settings, "https://unreachable.invalid", "tok")
        self._cache_a_result(broken, monkeypatch, available=False)

        working = self._checker(settings, "https://real.example", "tok")
        result = self._cache_a_result(working, monkeypatch, available=True)

        assert result["a1000"]["status"] == ServiceStatus.AVAILABLE.value

    def test_a_changed_token_invalidates_the_cache(self, settings, monkeypatch):
        first = self._checker(settings, "https://real.example", "old-token")
        self._cache_a_result(first, monkeypatch, available=False)

        second = self._checker(settings, "https://real.example", "new-token")
        assert second.cache.load() is None

    def test_same_configuration_still_hits_the_cache(self, settings, monkeypatch):
        checker = self._checker(settings, "https://real.example", "tok")
        first = self._cache_a_result(checker, monkeypatch, available=True)

        again = self._checker(settings, "https://real.example", "tok")
        assert again.cache.load()["timestamp"] == first["timestamp"]

    def test_credentials_are_not_written_to_the_cache(self, settings, monkeypatch):
        checker = self._checker(settings, "https://real.example", "super-secret-token")
        self._cache_a_result(checker, monkeypatch, available=True)

        assert "super-secret-token" not in checker.cache.path.read_text(encoding="utf-8")


def _unsalted_fingerprint(settings) -> str:
    """The digest the cache used to store: sha256 over the configuration.

    Reproduced here rather than imported, because the point of the test is
    that anyone can write this function - the format was published in the
    source - and that doing so no longer verifies anything.
    """
    ticloud, a1000 = settings.titanium_cloud, settings.a1000
    identity = "|".join(
        str(part or "")
        for part in (
            settings.profile,
            ticloud.host,
            ticloud.username,
            ticloud.password,
            a1000.host,
            a1000.username,
            a1000.password,
            a1000.token,
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


class TestTheCacheIsNotReadableByOtherUsers:
    """It was written 0644 into a 0755 directory, next to no other secret.

    Nothing in it is a credential verbatim, but the fingerprint is derived
    from every credential, and the answer names the appliances this
    machine talks to.
    """

    @pytest.fixture
    def cached(self, settings, checker, monkeypatch):
        settings.a1000.token = "guessable-token"
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: True)
        checker.check_all()
        return checker.cache

    @pytest.mark.posix_only
    def test_the_cache_file_is_owner_only(self, cached):
        assert stat.S_IMODE(cached.path.stat().st_mode) == 0o600

    @pytest.mark.posix_only
    def test_the_cache_directory_is_owner_only(self, cached):
        assert stat.S_IMODE(cached.path.parent.stat().st_mode) == 0o700

    @pytest.mark.posix_only
    def test_the_salt_is_owner_only(self, cached):
        assert stat.S_IMODE(cached.salt_path.stat().st_mode) == 0o600

    def test_the_fingerprint_does_not_confirm_a_guessed_credential(self, settings, cached):
        """Offline oracle: hash a guess, compare, learn whether it was right."""
        stored = json.loads(cached.path.read_text(encoding="utf-8"))["fingerprint"]

        assert stored != _unsalted_fingerprint(settings)


class TestUpgradingPastTheSaltCostsOneProbe:
    """Every cache written by an earlier version now fails to match."""

    def test_a_cache_from_before_the_salt_is_ignored_rather_than_read(
        self, settings, checker, monkeypatch
    ):
        settings.a1000.token = "real-token"
        legacy = {
            "timestamp": datetime.now().isoformat(),
            "titanium_cloud": {
                "status": "unavailable",
                "message": "",
                "credentials_configured": False,
                "api_accessible": False,
            },
            "a1000": {
                "status": "available",
                "message": "",
                "credentials_configured": True,
                "api_accessible": True,
            },
            "summary": {"services_available": 1, "services_total": 2},
            "fingerprint": _unsalted_fingerprint(settings),
        }
        checker.cache.path.parent.mkdir(parents=True, exist_ok=True)
        checker.cache.path.write_text(json.dumps(legacy))

        assert checker.cache.load() is None

    def test_the_re_probe_replaces_it_without_crashing(self, settings, checker, monkeypatch):
        settings.a1000.token = "real-token"
        checker.cache.path.parent.mkdir(parents=True, exist_ok=True)
        checker.cache.path.write_text(json.dumps({"fingerprint": "stale", "timestamp": "nonsense"}))
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: True)

        assert checker.check_all()["a1000"]["status"] == ServiceStatus.AVAILABLE.value
        assert checker.cache.load() is not None


class TestTheSaltIsStablePerInstall:
    """A random value that changed per run would mean never hitting the cache."""

    def test_the_salt_survives_clearing_the_cache(self, settings, checker, monkeypatch):
        settings.a1000.token = "real-token"
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: True)
        checker.check_all()
        before = checker.cache.salt_path.read_text(encoding="utf-8")

        checker.clear_cache()
        checker.check_all()

        assert checker.cache.salt_path.read_text(encoding="utf-8") == before


class TestCacheHitAndFreshProbeAgree:
    """A cached answer and a fresh one must be the same shape."""

    def test_the_two_paths_carry_the_same_keys(self, settings, checker, monkeypatch):
        settings.a1000.token = "real-token"
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: True)

        fresh = checker.check_all()
        cached = checker.check_all()

        assert cached.keys() == fresh.keys()
        assert "fingerprint" not in cached


class TestHostileProbeMessage:
    """A probe message is replayed before every command for 24 hours."""

    def test_control_characters_never_reach_the_cache(
        self, settings, checker, tmp_path, monkeypatch
    ):
        settings.titanium_cloud.username = "user"
        settings.titanium_cloud.password = "pass"

        def hostile(self, sha256):
            self.output.error("\x1b]0;HIJACKED\x07 502 Bad Gateway [/]")
            return None

        monkeypatch.setattr(TitaniumCloudService, "get_file_reputation", hostile)
        monkeypatch.setattr(A1000MetadataService, "test_connection", lambda self: True)

        checker.check_all()
        stored = json.loads(checker.cache.path.read_text(encoding="utf-8"))

        assert "\x1b" not in stored["titanium_cloud"]["message"]
        assert "\x07" not in stored["titanium_cloud"]["message"]
        assert "502 Bad Gateway" in stored["titanium_cloud"]["message"]


class TestUnreadableSalt:
    """A salt file that is not UTF-8 disabled the cache permanently."""

    def test_a_non_utf8_salt_is_replaced_rather_than_raising(self, settings):
        cache = _cache(settings)
        cache._make_dir()
        cache.salt_path.write_bytes(b"\xff\xfe\x00bad")

        # UnicodeDecodeError escaped _salt into fingerprint, where load()
        # and save() swallowed it: the cache was never read or written.
        assert cache.fingerprint()
        assert cache.salt_path.read_text(encoding="utf-8") != ""

        cache.save(_a_probe_run())
        assert cache.load() is not None

    def test_a_salt_that_cannot_be_stored_still_yields_a_fingerprint(self, settings, monkeypatch):
        """A fresh salt per call costs a re-probe, not a wrong answer."""
        cache = _cache(settings, verbose=True)
        monkeypatch.setattr(
            cache, "_make_dir", lambda: (_ for _ in ()).throw(OSError("read-only file system"))
        )

        assert cache.fingerprint()


class TestHowAProbeOutcomeIsGraded:
    """The one answer to "is this good news", which the report only draws.

    It sits beside the statuses so that the terminal lines and the document
    ``-o json`` writes cannot disagree about a service, and so that no
    renderer has to know which status is which.
    """

    @pytest.mark.parametrize(
        ("status", "grade"),
        [
            (ServiceStatus.AVAILABLE.value, ProbeGrade.GOOD),
            (ServiceStatus.UNAVAILABLE.value, ProbeGrade.CAVEAT),
            (ServiceStatus.ERROR.value, ProbeGrade.BAD),
        ],
        ids=["available", "unavailable", "error"],
    )
    def test_every_status_the_prober_answers_with_has_a_grade(self, status, grade):
        assert grade_of(status) == grade

    @pytest.mark.parametrize("status", ["degraded", ""], ids=["later_version", "empty"])
    def test_a_status_this_version_does_not_know_is_bad_news(self, status):
        """A probe run is a plain dict on disk, so the reader takes what it finds.

        A word a later version wrote, or the empty one a damaged cache
        leaves behind, has to grade as something rather than raise out of
        the report that exists to say what could not be reached. Bad news
        is the grade that cannot announce an unreachable appliance as a
        success.
        """
        assert grade_of(status) is ProbeGrade.BAD
