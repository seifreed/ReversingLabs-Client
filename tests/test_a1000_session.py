"""The session is the connection: services share one, or pay for one each.

Nothing counted connections while the connection was a base class, so
"this object holds every area" was the only way to be sure a script did
not authenticate once per area. These count them at ``A1000Session._open``
— the single call that constructs an SDK client, and under
username/password auth a token POST to the appliance.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rl_cli.config import Settings
from rl_cli.services.a1000 import (
    A1000MetadataService,
    A1000ReportService,
    A1000SampleService,
    A1000Service,
    A1000Session,
    A1000YaraService,
    upload_and_get_report,
)
from rl_cli.services.a1000 import session as session_module

SHA1 = "da39a3ee5e6b4b0d3255bfef95601890afd80709"


@pytest.fixture
def session(a1000_session: A1000Session) -> A1000Session:
    """The shared sandboxed session, under the name this suite reads by."""
    return a1000_session


class TestOneSessionIsOneConnection:
    def test_three_areas_authenticate_once(self, session, a1000_connections):
        samples = session.service(A1000SampleService)
        yara = session.service(A1000YaraService)
        metadata = session.service(A1000MetadataService)

        samples.list_samples()
        yara.list_yara_rulesets()
        metadata.get_user_tags("e" * 64)

        assert len(a1000_connections) == 1
        assert samples.client is yara.client is metadata.client

    def test_services_with_sessions_of_their_own_pay_for_one_each(
        self, tmp_path, a1000_connections
    ):
        """The cost the session exists to avoid, so it stays visible."""
        settings = Settings(cache_dir=tmp_path / "cache", config_dir=tmp_path / "config")

        A1000Session(settings).service(A1000SampleService).list_samples()
        A1000Session(settings).service(A1000YaraService).list_yara_rulesets()

        assert len(a1000_connections) == 2

    def test_a_connection_opened_by_one_service_serves_the_next(self, session, a1000_connections):
        """The second service must not connect: the session already has one."""
        session.service(A1000SampleService).list_samples()
        assert len(a1000_connections) == 1

        session.service(A1000ReportService).get_summary_report_v2(SHA1)
        assert len(a1000_connections) == 1

    def test_a_session_that_is_already_connected_does_not_reconnect(
        self, session, a1000_connections
    ):
        """``connect()`` is public, and it is the same claim as the refusal one.

        Nothing but the failing path was memoised, so asking a working
        session to connect again built a fresh SDK client — under
        username/password auth, another token POST — and quietly replaced
        the one its services were holding.
        """
        session.connect()
        client = session.client

        session.connect()
        session.connect()

        assert len(a1000_connections) == 1
        assert session.client is client

    def test_the_workflow_spanning_two_areas_authenticates_once(
        self, session, a1000_connections, tmp_path
    ):
        sample = tmp_path / "sample.bin"
        sample.write_bytes(b"x")

        report = upload_and_get_report(
            session.service(A1000SampleService),
            session.service(A1000ReportService),
            sample,
            "summary",
        )

        assert report == {}
        assert len(a1000_connections) == 1


class TestAnApplianceThatDidNotAnswerIsNotDialledAgain:
    """The same claim on the path the stub connection cannot reach.

    Every attempt waits out the connect timeout — 300 s by default — so a
    session that forgot a refusal cost one of those per thing that wanted
    the appliance, all to be told the same thing again.
    """

    def test_two_services_over_one_session_attempt_once(self, session, a1000_failed_connections):
        metadata = session.service(A1000MetadataService, MagicMock())
        reports = session.service(A1000ReportService, MagicMock())

        assert metadata.test_connection() is False
        assert reports.get_summary_report_v2(SHA1) is None

        assert len(a1000_failed_connections) == 1

    def test_each_caller_still_reports_the_failure_as_its_own(
        self, session, a1000_failed_connections
    ):
        """Not retrying must not mean the second caller silently succeeds."""
        output = MagicMock()
        service = session.service(A1000ReportService, output)

        assert service.connect() is False
        assert service.connect() is False

        assert len(a1000_failed_connections) == 1
        assert output.error.call_count == 2

    def test_a_refused_connection_is_told_apart_from_an_unused_one(
        self, session, a1000_failed_connections
    ):
        assert session.client is None
        assert session.failure is None

        session.service(A1000MetadataService, MagicMock()).connect()

        assert session.client is None
        assert isinstance(session.failure, ConnectionError)


class TestDisconnectLeavesNoLiveClient:
    def _connected(self, session: A1000Session) -> list[A1000Service]:
        """Three areas over one session, connected by the first call made."""
        samples = session.service(A1000SampleService)
        services: list[A1000Service] = [
            samples,
            session.service(A1000ReportService),
            session.service(A1000YaraService),
        ]
        samples.list_samples()
        return services

    def test_disconnecting_one_service_ends_it_for_all(self, session, a1000_connections):
        services = self._connected(session)
        assert all(service.client is not None for service in services)

        services[0].disconnect()

        assert session.client is None
        assert all(service.client is None for service in services)

    def test_closing_the_session_leaves_no_live_client_held_anywhere(
        self, session, a1000_connections
    ):
        services = self._connected(session)

        session.close()

        for service in services:
            assert not [
                held for held in vars(service).values() if getattr(held, "client", None) is not None
            ]


class TestTheCredentialASessionAuthenticatesWith:
    """A config half-filled from the example still has a real credential in it.

    ``check-access`` reads the placeholder rule and reports
    username/password for exactly that config, so a session picking by
    bare truthiness authenticates with ``your_a1000_api_token_here``,
    fails every command in the run, and disagrees with the verdict the
    probe cached for the next 24 hours.
    """

    def _opened_with(self, session, monkeypatch) -> dict[str, object]:
        """The keyword arguments the SDK client was constructed with."""
        captured: dict[str, object] = {}

        def a1000(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(session_module, "A1000", a1000)
        session.connect()
        return captured

    def test_a_placeholder_token_does_not_outrank_credentials_the_user_set(
        self, session, monkeypatch
    ):
        session.a1000_settings.token = "your_a1000_api_token_here"
        session.a1000_settings.username = "analyst"
        session.a1000_settings.password = "s3cret"

        opened = self._opened_with(session, monkeypatch)

        assert opened["username"] == "analyst"
        assert opened["password"] == "s3cret"
        assert "token" not in opened

    def test_a_token_the_user_set_is_still_what_authenticates(self, session, monkeypatch):
        session.a1000_settings.token = "t0ken"
        session.a1000_settings.username = "analyst"
        session.a1000_settings.password = "s3cret"

        opened = self._opened_with(session, monkeypatch)

        assert opened["token"] == "t0ken"
        assert "username" not in opened

    @pytest.mark.parametrize(
        ("username", "password"),
        [
            ("your_username", "s3cret"),
            ("analyst", "your_password"),
            ("your_username", "your_password"),
        ],
    )
    def test_no_stand_in_reaches_the_fallback_either(
        self, session, monkeypatch, username, password
    ):
        """The token is not the only credential a half-filled example carries.

        The fallback branch was passed straight through, so a config whose
        username and password are still the example's own — which the probe
        reports as "No A1000 credentials configured" — authenticated with
        ``your_username``.
        """
        session.a1000_settings.token = None
        session.a1000_settings.username = username
        session.a1000_settings.password = password

        opened = self._opened_with(session, monkeypatch)

        assert "your_username" not in opened.values()
        assert "your_password" not in opened.values()

    @pytest.mark.parametrize("padded", ["your_a1000_api_token_here ", " your_a1000_api_token_here"])
    def test_a_stand_in_padded_with_whitespace_is_not_a_token_either(
        self, session, monkeypatch, padded
    ):
        """One keystroke inside the example's quotes must not promote it."""
        session.a1000_settings.token = padded
        session.a1000_settings.username = "analyst"
        session.a1000_settings.password = "s3cret"

        opened = self._opened_with(session, monkeypatch)

        assert opened["username"] == "analyst"
        assert "token" not in opened


class TestServicesBuiltFromASession:
    def test_service_returns_the_requested_class(self, session):
        assert isinstance(session.service(A1000YaraService), A1000YaraService)

    def test_a_service_reports_through_the_output_it_was_built_with(self, session):
        output = MagicMock()
        assert session.service(A1000SampleService, output).output is output

    def test_a_service_reads_the_settings_of_the_session_it_speaks_over(self, session):
        """It cannot be told about one appliance and connected to another."""
        service = session.service(A1000SampleService)

        assert service.settings is session.settings
        assert service.a1000_settings is session.a1000_settings
