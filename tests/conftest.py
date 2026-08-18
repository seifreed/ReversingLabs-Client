"""Keep the suite off the developer's own credentials, files and appliance.

``Settings`` finds its own config: ``./config.yaml``, ``./.env``,
``~/.rl-cli.yaml``, ``~/.config/rl-cli/config.yaml`` — and this repo has a
real ``config.yaml`` at its root. The CLI then probes both APIs before it
reaches a subcommand, writing the answer to ``~/.cache/rl-cli``. Run
unguarded, ``pytest`` here talks to a live A1000 with real tokens, and any
``A1000_*``/``TICLOUD_*`` variable in the shell outranks what a test sets up.

The autouse fixture below moves cwd and ``$HOME`` to a tmp path and drops
those variables, so discovery, the cache and ``save_config`` can only ever
find an empty sandbox.
"""

from __future__ import annotations

import io
import os
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest
from requests.adapters import HTTPAdapter
from ReversingLabs.SDK.a1000 import A1000
from rich.console import Console

from rl_cli.config import Settings, clear_settings_cache
from rl_cli.services.a1000 import A1000Session
from rl_cli.services.availability import (
    APIAvailabilityChecker,
    Availability,
    ServiceProbe,
    ServiceStatus,
)

# The suite that exercises the availability probe wears this marker, and the
# autouse fixture below leaves the real ``check_all`` in place for it. It is
# the marker, not the module's name, that decides: naming the file was what
# let a rename or a split hand the probe tests a stub to pass against.
REAL_PROBE_MARKER = "real_availability_probe"

_CREDENTIAL_PREFIXES = ("A1000_", "TICLOUD_", "OUTPUT_", "RL_CLI_")

# Captured before any test can stub it, so a test that wants the real probe
# back can ask for the ``real_probe`` fixture below.
_REAL_CHECK_ALL = APIAvailabilityChecker.check_all

SHA1 = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

# What a probe run looks like when both services answered. Every key the
# real ``check_all`` produces is here: a stub missing ``summary``,
# ``timestamp``, ``credentials_configured`` or ``api_accessible`` passes
# only as long as no caller reads them, which pins today's two-key reader
# rather than the ``Availability`` contract, and lets a command that
# renders the whole run be added with nothing to catch it.
_STUB_PROBE_MESSAGE = "probe stubbed in tests"


def stubbed_availability() -> Availability:
    """A complete ``Availability``, as ``APIAvailabilityChecker`` would build it."""

    def probe() -> ServiceProbe:
        return {
            "status": ServiceStatus.AVAILABLE.value,
            "message": _STUB_PROBE_MESSAGE,
            "credentials_configured": True,
            "api_accessible": True,
        }

    return {
        "timestamp": datetime(2024, 1, 1, 0, 0, 0).isoformat(),
        "titanium_cloud": probe(),
        "a1000": probe(),
        "summary": {"services_available": 2, "services_total": 2},
    }


# Bodies that get a caller past the answer it checks and on to the work it
# was doing: the upload receipt carries the digest the report is fetched
# for, and the analysis-status endpoint reports a sample already analysed.
_STUB_BODIES: dict[str, object] = {
    "submit_file_from_path": {"code": 201, "detail": {"sha1": SHA1}},
    "file_analysis_status": {"results": [{"status": "processed"}]},
    "get_yara_ruleset_matches_v2": {"results": []},
    "advanced_search_v3": {"rl": {"web_search_api": {"entries": []}}},
}


def _runs_whole_suite(config: pytest.Config) -> bool:
    """Whether this invocation collects everything ``testpaths`` names.

    A narrower run is anything that selects: a file or nodeid argument, a
    ``-k``/``-m`` expression, a ``--deselect``, or a last-failed/stepwise
    subset.
    """
    option = config.option
    # Every one of these is read with getattr: `lf` and `stepwise` belong to
    # the cacheprovider plugin, so `-p no:cacheprovider` leaves them off the
    # namespace entirely and reading them as attributes is an INTERNALERROR
    # for the whole run.
    for name in ("keyword", "markexpr", "deselect", "lf", "stepwise"):
        if getattr(option, name, None):
            return False

    testpaths = {Path(p) for p in config.getini("testpaths")}
    for arg in config.args:
        path, _, nodeid = arg.partition("::")
        if nodeid:
            return False
        candidate = Path(path)
        if candidate.is_absolute():
            try:
                candidate = candidate.relative_to(config.rootpath)
            except ValueError:
                return False
        if candidate not in testpaths:
            return False
    return True


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{REAL_PROBE_MARKER}: keep the real APIAvailabilityChecker.check_all for this test",
    )

    # The 100% floor in pyproject's addopts is a claim about the whole suite,
    # but it was applied to every invocation: `pytest tests/test_toon.py`
    # printed "17 passed" and then failed the run at 21.30% coverage of
    # rl_cli, which that file never claimed to cover. Exit code and all.
    # Nothing is measured differently here -- the report still prints -- the
    # floor is simply only enforced by the run it describes, which is the one
    # ci.yml and the pre-push hook make.
    #
    # The floor has to be cleared on the plugin's own namespace:
    # pytest-cov builds its CovPlugin in pytest_load_initial_conftests from
    # `early_config.known_args_namespace`, so `config.option.cov_fail_under`
    # is a different object and setting that one changes nothing.
    if not _runs_whole_suite(config):
        cov_plugin = config.pluginmanager.get_plugin("_cov")
        if cov_plugin is not None:
            cov_plugin.options.cov_fail_under = 0


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the POSIX file-security tests where those semantics do not exist.

    ``rl_cli.storage.files`` keeps its umask-proof 0600, no-symlink write on
    POSIX and degrades to the platform default on Windows, where
    ``O_NOFOLLOW`` and ``fchmod`` are absent. The tests that assert the POSIX
    guarantees -- exact 0600/0700 modes, hard-link and symlink attacks -- are
    marked ``posix_only`` and skipped off POSIX rather than failing on it.
    The whole-suite 100% coverage floor is therefore enforced on a POSIX
    runner, where every branch is reachable; ``ci.yml`` runs Windows without
    the floor.
    """
    if os.name == "posix":
        return
    skip = pytest.mark.skip(reason="POSIX file-mode/link semantics unavailable on this platform")
    for item in items:
        if "posix_only" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def hermetic(request, tmp_path, monkeypatch):
    """Isolate configuration discovery, the cache and the API probe.

    Whatever wears the ``real_availability_probe`` marker is exercising the
    probe itself, so it keeps the real ``check_all``; everything else gets a
    stub. Any other test wanting the real one can ``monkeypatch.setattr`` it
    back, exactly as tests/test_main_cli.py does.
    """
    for name in list(os.environ):
        if name.startswith(_CREDENTIAL_PREFIXES):
            monkeypatch.delenv(name)

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(home)
    clear_settings_cache()

    # A1000 traffic funnels through A1000Session._open, which the connection
    # fixtures replace - but TitaniumCloud has no session, it builds an SDK
    # handle per call, so nothing stopped a ticloud test reaching
    # data.reversinglabs.com for real. One did. Blocking the transport
    # itself is the only seam every client shares, and it does not care
    # which higher level a test chose to stub.
    def _no_network(self: object, request: object, **kwargs: object) -> object:
        raise AssertionError(
            f"A test reached the network: {getattr(request, 'method', '?')} "
            f"{getattr(request, 'url', request)}. Stub the SDK object instead."
        )

    monkeypatch.setattr(HTTPAdapter, "send", _no_network)

    if request.node.get_closest_marker(REAL_PROBE_MARKER) is None:
        monkeypatch.setattr(
            APIAvailabilityChecker,
            "check_all",
            lambda self, force=False: stubbed_availability(),
        )


@pytest.fixture
def real_probe(hermetic, monkeypatch):
    """Put the actual availability probe back for one test.

    The stub above is what let "one connection per invocation" pass while
    the probe was opening a second one, so a test making that claim has to
    ask for this. It stays hermetic: the probe reaches the appliance
    through ``A1000Session._open``, which ``a1000_connections`` replaces.
    """
    monkeypatch.setattr(APIAvailabilityChecker, "check_all", _REAL_CHECK_ALL)


def sdk_response(
    status_code: int = 200,
    json_payload: Any = None,
    *,
    content: bytes = b"",
    text: str = "",
) -> MagicMock:
    """The ``requests.Response`` an SDK method answers with.

    One builder for the whole suite. The three it replaced each set only
    the fields their own tests read, and every field left out stayed a
    MagicMock attribute — truthy, and printable as a repr — so a wrapper
    reading ``response.text`` for the appliance's explanation, or
    ``response.content`` for a body, was handed a mock rather than the
    empty value a real response carries.
    """
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {} if json_payload is None else json_payload
    response.content = content
    response.text = text
    return response


def autospec_sdk_client(
    sdk_class: type,
    *,
    bodies: Mapping[str, Any] | None = None,
    contents: Mapping[str, bytes] | None = None,
) -> MagicMock:
    """An autospec of ``sdk_class`` answering 200 to every call.

    One builder for every stand-in SDK object in the suite. Autospec is
    what makes a stand-in answer only the calls the installed SDK would
    accept: a bare ``MagicMock()`` took ``submit_file_from_path(nonsense=1)``
    and ``a_method_the_sdk_never_had()`` alike, so a wrapper calling a
    renamed method with invented keywords passed against it.

    Autospec alone hands back MagicMocks, whose ``status_code`` breaks
    numeric comparisons for reasons unrelated to call signatures, so each
    method gets a 200 response — ``bodies`` and ``contents`` name the ones
    that need a particular payload — and the aggregating methods get the
    plain list they really return.
    """
    client: MagicMock = create_autospec(sdk_class, instance=True)
    for name in dir(sdk_class):
        if name.startswith("_"):
            continue
        method = getattr(client, name, None)
        if not isinstance(method, MagicMock):
            continue
        if name.endswith("_aggregated"):
            # The SDK's aggregating methods answer a plain list, not a
            # response.
            method.return_value = []
            continue
        method.return_value = sdk_response(
            json_payload=(bodies or {}).get(name, {}),
            content=(contents or {}).get(name, b""),
        )
    return client


def stub_sdk_client() -> MagicMock:
    """A stand-in for a connected ``A1000``, answering 200 to every call."""
    return autospec_sdk_client(A1000, bodies=_STUB_BODIES)


def console_text(console: Console) -> str:
    """Read back what a ``Console`` wired to an in-memory buffer has written.

    ``Console.file`` is declared ``IO[str]``, which has no ``getvalue``;
    the buffer the test handed it does.
    """
    buffer = console.file
    assert isinstance(buffer, io.StringIO), "this console was not given a StringIO to write to"
    return buffer.getvalue()


@pytest.fixture
def a1000_session(tmp_path) -> A1000Session:
    """A session whose cache and config live under the test's own tmp_path.

    ``Settings()`` with neither directory set resolves the developer's real
    ``~/.cache/rl-cli`` and ``~/.config/rl-cli``; the autouse fixture moves
    ``$HOME`` out of the way, but only a session built like this one says so
    at the point of construction.
    """
    return A1000Session(
        Settings(cache_dir=tmp_path / "cache", config_dir=tmp_path / "config"),
    )


@pytest.fixture
def ansi() -> Callable[[Callable[[Console], object]], str]:
    """Render through a colour-emitting console and return the raw ANSI text."""

    def render_to_ansi(render: Callable[[Console], object]) -> str:
        console = Console(
            file=io.StringIO(),
            width=120,
            force_terminal=True,
            no_color=False,
            color_system="truecolor",
        )
        render(console)
        return console_text(console)

    return render_to_ansi


@pytest.fixture
def a1000_connections(monkeypatch) -> list[MagicMock]:
    """Count A1000 connections without opening any; returns the clients handed out.

    ``A1000Session._open`` is the single seam a connection can come through
    — it is the call that constructs the SDK client, which under
    username/password auth is a token POST to the appliance. Counting here
    is what says whether services share a connection or each open one, and
    stubbing it is what keeps the suite off the network.
    """
    opened: list[MagicMock] = []

    def open_(session: A1000Session) -> None:
        session.client = stub_sdk_client()
        opened.append(session.client)

    monkeypatch.setattr(A1000Session, "_open", open_)
    return opened


@pytest.fixture
def a1000_failed_connections(monkeypatch) -> list[A1000Session]:
    """Count attempts against an appliance that never answers; returns the sessions tried.

    The counterpart of ``a1000_connections``, at the same seam: that one
    stubs a connection that always succeeds, which is why "one connection
    per invocation" could hold on the happy path while an unreachable
    appliance was dialled once per thing that wanted it — each attempt
    waiting out the connect timeout in turn.
    """
    attempted: list[A1000Session] = []

    def open_(session: A1000Session) -> None:
        attempted.append(session)
        raise ConnectionError("appliance did not answer")

    monkeypatch.setattr(A1000Session, "_open", open_)
    return attempted
