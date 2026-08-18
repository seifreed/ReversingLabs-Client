"""Tests for how ``A1000Session._open`` authenticates and bounds its client.

This is the one function that decides *how* the CLI reaches an appliance,
and every other fixture in the suite replaces it (``a1000_connections``)
precisely so that no test touches the network. That left the body itself
unpinned: the token branch could stop being taken, or the request-timeout
wrapper could be dropped, without a single test noticing.

Nothing here reaches an appliance. The SDK class is replaced by a
recording stand-in, and the one ``requests.Session`` involved has its
``send`` swapped out, so the only thing observed is what
:meth:`A1000Session._open` asked for.
"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
import requests

from rl_cli.config import Settings
from rl_cli.services.a1000 import A1000Session
from rl_cli.services.a1000 import session as session_module


class _RecordingA1000:
    """Stand-in for ``ReversingLabs.SDK.a1000.A1000`` that keeps its kwargs.

    Carries a real ``requests.Session`` under ``_session`` because that is
    the attribute ``apply_request_timeout`` reaches for; a client without
    one is handed straight back, which would make the timeout assertion
    vacuous.
    """

    constructed: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._session = requests.Session()
        _RecordingA1000.constructed.append(kwargs)


@pytest.fixture
def sdk(monkeypatch) -> type[_RecordingA1000]:
    """Replace the SDK class the session constructs, not the session's method.

    ``session.py`` binds ``A1000`` at import time, so its own module
    global is the seam; patching the name on ``ReversingLabs.SDK.a1000``
    would leave that binding pointing at the real class.
    """
    _RecordingA1000.constructed = []
    monkeypatch.setattr(session_module, "A1000", _RecordingA1000)
    return _RecordingA1000


def _settings(tmp_path, **a1000: Any) -> Settings:
    settings = Settings(
        cache_dir=tmp_path / "cache",
        config_dir=tmp_path / "config",
        config_file=tmp_path / "missing.yaml",
    )
    settings.a1000.host = "https://a1000.invalid"
    settings.a1000.token = None
    settings.a1000.username = None
    settings.a1000.password = None
    settings.a1000.proxy = None
    settings.a1000.verify_ssl = True
    settings.a1000.timeout = 17
    for name, value in a1000.items():
        setattr(settings.a1000, name, value)
    return settings


class TestAuthenticationChoice:
    """A configured token is what the client is built with; nothing else."""

    def test_token_is_used_and_credentials_are_not_sent(self, tmp_path, sdk):
        # Credentials are present too, so "the token branch was taken" is
        # distinguishable from "there was nothing else to send".
        settings = _settings(
            tmp_path,
            token="tok-abc",
            username="user",
            password="pass",
        )
        A1000Session(settings)._open()

        assert len(sdk.constructed) == 1
        kwargs = sdk.constructed[0]
        assert kwargs["token"] == "tok-abc"
        assert "username" not in kwargs
        assert "password" not in kwargs

    def test_username_and_password_are_used_when_no_token(self, tmp_path, sdk):
        settings = _settings(
            tmp_path,
            username="user",
            password="pass",
        )
        A1000Session(settings)._open()

        kwargs = sdk.constructed[0]
        assert kwargs["username"] == "user"
        assert kwargs["password"] == "pass"
        assert "token" not in kwargs

    def test_empty_token_falls_through_to_credentials(self, tmp_path, sdk):
        """An unset token in a config file reads as ``""``, not as absent."""
        settings = _settings(
            tmp_path,
            token="",
            username="user",
            password="pass",
        )
        A1000Session(settings)._open()

        assert "token" not in sdk.constructed[0]
        assert sdk.constructed[0]["username"] == "user"


class TestConnectionParameters:
    """Host, certificate verification and proxy reach the SDK either way."""

    @pytest.mark.parametrize(
        "auth",
        [
            {"token": "tok"},
            {"username": "user", "password": "pass"},
        ],
    )
    def test_host_and_verify_are_passed(self, tmp_path, sdk, auth):
        settings = _settings(tmp_path, verify_ssl=False, **auth)
        A1000Session(settings)._open()

        kwargs = sdk.constructed[0]
        assert kwargs["host"] == "https://a1000.invalid"
        assert kwargs["verify"] is False
        assert kwargs["proxies"] is None

    def test_proxy_is_sent_for_both_schemes(self, tmp_path, sdk):
        settings = _settings(tmp_path, token="tok", proxy="http://proxy.invalid:3128")

        A1000Session(settings)._open()

        assert sdk.constructed[0]["proxies"] == {
            "http": "http://proxy.invalid:3128",
            "https": "http://proxy.invalid:3128",
        }


class TestRequestTimeoutIsApplied:
    """Without the wrapper every SDK call is unbounded — the hang it prevents."""

    def _sent_timeouts(self, client: Any) -> list[Any]:
        """Record the timeout every request goes out with, sending none."""
        sent: list[Any] = []

        def send(*_args: Any, **kwargs: Any) -> MagicMock:
            sent.append(kwargs.get("timeout"))
            return MagicMock()

        client._session.send = send
        return sent

    @pytest.mark.parametrize(
        "auth",
        [
            {"token": "tok"},
            {"username": "user", "password": "pass"},
        ],
    )
    def test_every_verb_gets_the_configured_timeout(self, tmp_path, sdk, auth):
        settings = _settings(tmp_path, **auth)
        session = A1000Session(settings)
        session._open()

        sent = self._sent_timeouts(session.client)
        for verb in ("get", "post", "put", "delete"):
            getattr(session.client._session, verb)("https://a1000.invalid/api/")

        assert sent == [17, 17, 17, 17], "SDK requests went out with no timeout"

    def test_a_call_that_asks_for_its_own_timeout_keeps_it(self, tmp_path, sdk):
        settings = _settings(tmp_path, token="tok")
        session = A1000Session(settings)
        session._open()

        sent = self._sent_timeouts(session.client)
        session.client._session.get("https://a1000.invalid/api/", timeout=5)

        assert sent == [5]

    def test_the_wrapped_client_is_the_one_the_session_keeps(self, tmp_path, sdk):
        settings = _settings(tmp_path, token="tok")
        session = A1000Session(settings)
        session._open()

        assert isinstance(session.client, _RecordingA1000)


class TestOpenReplacesTheClient:
    def test_a_second_open_replaces_the_first_client(self, tmp_path, sdk):
        settings = _settings(tmp_path, token="tok")
        session = A1000Session(settings)
        session._open()
        first = session.client
        session._open()

        assert len(sdk.constructed) == 2
        assert session.client is not first
