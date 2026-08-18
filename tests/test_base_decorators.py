"""Tests for the machinery every service wrapper is written on top of.

These verify the boilerplate-removal contract that the rest of the
service layer relies on: the ``@with_client``/``@safe_call`` decorators'
connect-on-demand, exception-to-default mapping and error-context format,
``BaseService.handle_error``'s advice, and the HTTP bounding, response
reading and poll spacing in ``rl_cli.services.http``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests
from ReversingLabs.SDK.helper import ForbiddenError, TooManyRequestsError, UnauthorizedError

from rl_cli.config import Settings
from rl_cli.services.base import BaseService
from rl_cli.services.decorators import safe_call, with_client
from rl_cli.services.http import (
    PollBackoff,
    PollState,
    apply_request_timeout,
    default_post_timeout,
    json_on,
    poll_until,
    succeeded,
)
from tests.conftest import sdk_response


class _FakeService(BaseService):
    """Minimal concrete service used to exercise the decorators.

    Declares the connection ``with_client`` manages: it belongs to the
    services that have one, not to ``BaseService``.
    """

    client: Any = None

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.connect_attempts = 0
        self.next_connect_returns = True

    def connect(self) -> bool:
        self.connect_attempts += 1
        if self.next_connect_returns:
            self.client = MagicMock()
        return self.next_connect_returns

    def disconnect(self) -> None:
        self.client = None

    @with_client(default=None)
    def fetch(self, key: str) -> dict[str, Any] | None:
        # The stand-in client is a MagicMock, so what it answers is only as
        # typed as the wrapper says it is - which is the point being tested.
        record: dict[str, Any] = self.client.do(key)
        return record

    @with_client(default=False)
    def commit(self, key: str) -> bool:
        committed: bool = self.client.commit(key)
        return committed

    @safe_call(default=None)
    def maybe_explode(self, value: int) -> int | None:
        if value < 0:
            raise ValueError("negative")
        return value


@pytest.fixture
def service() -> _FakeService:
    return _FakeService(Settings())


class TestWithClientConnectsOnDemand:
    def test_calls_connect_when_client_missing(self, service: _FakeService) -> None:
        service.fetch("x")
        assert service.connect_attempts == 1
        service.client.do.assert_called_once_with("x")

    def test_does_not_reconnect_when_client_present(self, service: _FakeService) -> None:
        service.client = MagicMock()  # already connected
        service.fetch("x")
        assert service.connect_attempts == 0

    def test_returns_default_when_connect_fails(self, service: _FakeService) -> None:
        service.next_connect_returns = False
        assert service.fetch("x") is None
        assert service.commit("x") is False

    def test_a_failed_connect_does_not_go_on_to_call_the_body(self, service: _FakeService) -> None:
        """``connect`` has already said why; the body must not say it again.

        Dropping the ``and not self.connect()`` short-circuit — so a
        failed connect fell through into the wrapped method — left the
        whole suite green, because the body then raised ``AttributeError``
        on the ``None`` client and the ``except`` answered the same
        default the guard would have. What changed was the reporting:
        every unreachable appliance grew a second, useless line under the
        real explanation, ``fetch('x'): 'NoneType' object has no attribute
        'do'``. The default is only half the contract, so the silence is
        asserted with it.
        """
        service.output = MagicMock()
        service.next_connect_returns = False

        assert service.fetch("x") is None

        assert service.connect_attempts == 1
        assert service.output.error.call_count == 0, "the failed connect was reported twice"


class TestWithClientCatchesExceptions:
    def test_returns_default_on_body_exception(self, service: _FakeService) -> None:
        service.client = MagicMock()
        service.client.do.side_effect = RuntimeError("boom")
        assert service.fetch("hash-abc") is None

    def test_handle_error_receives_method_name_and_first_arg(self, service: _FakeService) -> None:
        # Spy on handle_error to inspect the context string.
        service.handle_error = MagicMock()  # type: ignore[method-assign]
        service.client = MagicMock()
        service.client.do.side_effect = RuntimeError("boom")

        service.fetch("hash-abc")

        assert service.handle_error.call_count == 1
        _, ctx = service.handle_error.call_args[0]
        assert ctx == "fetch('hash-abc')"


class TestSafeCall:
    def test_returns_value_on_success(self, service: _FakeService) -> None:
        assert service.maybe_explode(7) == 7

    def test_returns_default_on_exception(self, service: _FakeService) -> None:
        assert service.maybe_explode(-1) is None

    def test_does_not_invoke_connect(self, service: _FakeService) -> None:
        service.maybe_explode(7)
        assert service.connect_attempts == 0


class TestBothDecoratorsAskTheSameKindOfQuestion:
    """``safe_call`` used to name a class where ``with_client`` named a shape.

    The nominal check made ``decorators`` depend on ``base`` — and so on
    ``requests`` and the SDK's error map — to satisfy one ``isinstance``,
    and it meant a service could satisfy the connection decorator and be
    refused by its sibling. Both now ask for the one method they use.

    They ask it of mypy, not of the interpreter. The ``isinstance``
    against a ``runtime_checkable`` Protocol that used to open every
    decorated call — some eighty of them — ran on every invocation to
    catch a mistake that can only be made while writing one, and this
    repo's own rule is to validate at system boundaries and trust
    internal code. The type variable the decorators bind ``self`` to is
    the same check, made once, where the mistake is: a class with no
    ``handle_error`` is a ``[type-var]`` error at the decoration, so
    there is no runtime ``TypeError`` left to assert on here.
    """

    class _ReportsWithoutABaseClass:
        """Everything ``safe_call`` needs, and no service in its ancestry."""

        def __init__(self) -> None:
            self.reported: list[str] = []

        def handle_error(self, error: Exception, context: str = "") -> None:
            self.reported.append(f"{error} ({context})")

        @safe_call(default="fallback")
        def explode(self, value: str) -> str:
            raise ValueError(f"no {value}")

    def test_it_accepts_anything_that_can_report(self) -> None:
        subject = self._ReportsWithoutABaseClass()

        assert not isinstance(subject, BaseService)
        assert subject.explode("dice") == "fallback"
        assert subject.reported == ["no dice (explode('dice'))"]


class TestHandleErrorTellsTheUserWhatToDo:
    """Every service reports through ``handle_error``, and it used to print
    ``Error in <Class> (<method>('<hash>')): <SDK sentence>``: our class
    name and a Python repr in front of a sentence that never named the
    profile, the config file, or anything to do next.
    """

    def _reported(self, error: Exception, tmp_path: Any = None) -> str:
        settings = Settings()
        settings.profile = "prod"
        if tmp_path is not None:
            settings.config_file = tmp_path / "appliances.yaml"
        service = _FakeService(settings)
        service.output = MagicMock()
        service.handle_error(error, "get_report('e3b0c442')")
        return str(service.output.error.call_args[0][0])

    def test_rejected_credentials_name_the_profile_and_config_file(self, tmp_path: Any) -> None:
        message = self._reported(UnauthorizedError(None), tmp_path)

        assert "prod" in message
        assert str(tmp_path / "appliances.yaml") in message
        # The SDK's own sentence is kept, the internal locator is not.
        assert "The provided credentials are invalid" in message
        assert "Error in" not in message

    def test_permission_denied_points_at_the_account(self, tmp_path: Any) -> None:
        message = self._reported(ForbiddenError(None), tmp_path)

        assert "Permission denied" in message
        assert "prod" in message

    def test_rate_limiting_says_to_retry(self) -> None:
        message = self._reported(TooManyRequestsError(None))

        assert "Rate limited" in message
        assert "retry" in message

    def test_a_tls_failure_mentions_the_setting_that_governs_it(self) -> None:
        error = requests.exceptions.SSLError(
            "HTTPSConnectionPool(host='a1000.invalid', port=443): SSLCertVerificationError"
        )

        message = self._reported(error)

        assert "verify_ssl" in message
        assert "SSLCertVerificationError" in message

    def test_an_unrecognised_failure_keeps_its_detail_and_locator(self) -> None:
        message = self._reported(RuntimeError("HTTP 507: Insufficient Storage"))

        assert "HTTP 507: Insufficient Storage" in message
        assert "get_report('e3b0c442')" in message
        assert "_FakeService" not in message


class TestPollBackoff:
    """A poll loop that re-issues every ``interval`` until its timeout turns
    one rate-limited appliance into sixty identical requests and sixty
    identical error lines.
    """

    def test_it_gives_up_after_three_consecutive_failures(self) -> None:
        backoff = PollBackoff(5)

        assert [backoff.keep_going(False) for _ in range(3)] == [True, True, False]

    def test_an_answer_resets_the_count(self) -> None:
        backoff = PollBackoff(5)

        backoff.keep_going(False)
        backoff.keep_going(False)
        backoff.keep_going(True)

        assert backoff.keep_going(False) is True

    def test_an_answer_resets_the_spacing_as_well_as_the_count(self, monkeypatch: Any) -> None:
        """ "A poll that answers resets both" — only the count was asserted.

        Dropping ``self._interval = self._base`` from the answering branch
        left the suite green: an appliance that failed twice and then
        recovered went on being polled at 20 s instead of returning to its
        5 s interval, so a wait sized for its timeout ran out of time
        having asked a third as often. The count reset is checked above;
        this is the other half of the same sentence.
        """
        slept: list[float] = []
        monkeypatch.setattr("rl_cli.services.http.time.sleep", slept.append)
        backoff = PollBackoff(5)

        backoff.keep_going(False)
        backoff.keep_going(False)
        backoff.keep_going(True)
        backoff.sleep()

        assert slept == [5]

    def test_failures_space_the_retries_out_up_to_a_cap(self, monkeypatch: Any) -> None:
        slept: list[float] = []
        monkeypatch.setattr("rl_cli.services.http.time.sleep", slept.append)
        backoff = PollBackoff(5, max_interval=20, give_up_after=99)

        for _ in range(4):
            backoff.keep_going(False)
            backoff.sleep()

        assert slept == [10, 20, 20, 20]

    def test_an_answering_appliance_is_polled_at_its_interval(self, monkeypatch: Any) -> None:
        slept: list[float] = []
        monkeypatch.setattr("rl_cli.services.http.time.sleep", slept.append)
        backoff = PollBackoff(5)

        for _ in range(3):
            backoff.keep_going(True)
            backoff.sleep()

        assert slept == [5, 5, 5]


class TestJsonOnHelper:
    def _resp(self, status: int, payload: Any | None = None) -> MagicMock:
        r = MagicMock()
        r.status_code = status
        r.json.return_value = payload if payload is not None else {}
        return r

    def test_returns_json_on_default_status_200(self) -> None:
        assert json_on(self._resp(200, {"ok": True})) == {"ok": True}

    def test_raises_with_server_message_on_error_status(self) -> None:
        resp = self._resp(428, {"message": "Appliance is not configured for local retro."})
        with pytest.raises(RuntimeError, match=r"428.*not configured for local retro"):
            json_on(resp)

    def test_the_most_specific_explanation_the_body_carries_is_the_one_shown(self) -> None:
        """A body spelling several of the four keys has them ranked, not picked.

        ``_response_message`` walks ("message", "error", "detail",
        "description") in that order because that is roughly most to least
        specific, but no test ever handed it a body carrying two of them —
        so reversing the tuple, and reporting the generic
        ``description`` over the sentence the appliance wrote about this
        call, left the suite green.
        """
        body = {"description": "Bad request", "message": "Ruleset 'prod' is locked"}
        resp = self._resp(400, body)

        with pytest.raises(RuntimeError, match=r"400: Ruleset 'prod' is locked"):
            json_on(resp)

    def test_raises_with_body_text_when_message_key_absent(self) -> None:
        resp = self._resp(503, {"unexpected": "shape"})
        resp.text = "Service Unavailable"
        with pytest.raises(RuntimeError, match="503: Service Unavailable"):
            json_on(resp)

    def test_an_empty_string_field_is_read_past_to_the_next_real_one(self) -> None:
        """An empty ``message`` is not a message; the walk reads on rather
        than reporting the blank as the explanation."""
        resp = self._resp(400, {"message": "", "error": "Ruleset is locked"})
        with pytest.raises(RuntimeError, match=r"400: Ruleset is locked"):
            json_on(resp)


class TestTheSuccessGateNeedsARealResponse:
    """``BaseService.validate_response`` passed anything that was not one.

    ``None`` was a failure and a ``status_code`` was range-checked, but
    *everything else* — a dict, a string, an SDK object that answered
    something other than a response — returned ``True``. It was the
    success gate at some twenty wrappers, so a method that returned the
    wrong kind of thing entirely was reported to the analyst as a
    success, and ``run_step``'s "None means failure" contract had nothing
    underneath it.
    """

    @pytest.mark.parametrize(
        "not_a_response",
        [{"code": 200, "message": "ok"}, "200 OK", object(), None],
        ids=["dict", "string", "object", "none"],
    )
    def test_a_value_that_is_not_a_response_is_never_a_success(self, not_a_response: Any) -> None:
        with pytest.raises(RuntimeError, match="Expected a response"):
            succeeded(not_a_response)

    def test_a_status_code_that_is_not_a_number_is_not_a_range_to_check(self) -> None:
        """An unconfigured mock answers a mock, and ``200 <= mock`` raises."""
        with pytest.raises(RuntimeError, match="Expected a response"):
            succeeded(MagicMock())

    @pytest.mark.parametrize("status", [True, False], ids=["true", "false"])
    def test_a_boolean_status_is_a_mangled_response_not_an_http_status(self, status: bool) -> None:
        """``bool`` is a subclass of ``int``, so it slips past the type test.

        Every case above fails ``isinstance(status, int)``; a bool is the
        one value that passes it and still is not a status. Dropping the
        ``isinstance(status, bool)`` clause left the suite green while
        ``succeeded`` refused with ``HTTP True: no details returned`` — a
        sentence about an appliance that never said it, in place of the
        one that names the mangled object the wrapper actually returned.
        """
        response = MagicMock()
        response.status_code = status

        with pytest.raises(RuntimeError, match="Expected a response"):
            succeeded(response)

    def test_the_success_range_ends_before_300(self) -> None:
        """2xx is acceptance; 300 is a redirect requests did not follow, so it
        is named through the raise rather than read as a success."""
        assert succeeded(MagicMock(status_code=200)) is True
        assert succeeded(MagicMock(status_code=299)) is True

        redirect = MagicMock(status_code=300)
        redirect.json.return_value = {}
        redirect.text = ""
        with pytest.raises(RuntimeError, match="HTTP 300"):
            succeeded(redirect)

    def test_the_service_layer_has_no_second_gate_left(self) -> None:
        """The deprecated ``BaseService.validate_response`` bridge is gone.

        It survived its own replacement as a one-line forward to
        ``succeeded`` while five wrappers still called it. A service that
        grows it back grows back the disagreement this class documents —
        two answers to "did the appliance accept it" — so its absence is
        asserted rather than assumed.
        """
        assert not hasattr(_FakeService(Settings()), "validate_response")

    def test_a_refusal_carries_the_appliances_own_explanation(self) -> None:
        refusal = sdk_response(500, {"message": "Storage is full"})

        with pytest.raises(RuntimeError, match="HTTP 500: Storage is full"):
            succeeded(refusal)

    @pytest.mark.parametrize("status", [100, 302, 304], ids=["continue", "found", "not-modified"])
    def test_a_status_that_is_neither_names_itself_here(self, status: int) -> None:
        """A redirect used to be handed back as ``False`` "for the caller".

        Not one of the fourteen callers took it: ``return
        succeeded(response)`` turned it into a ``False`` with no message
        emitted anywhere, so ``a1000 delete`` printed "Failed to delete
        sample" and nothing else — the information-free failure this
        function exists to abolish. A status with no explanation to carry
        still has itself to say.
        """
        with pytest.raises(RuntimeError, match=f"HTTP {status}"):
            succeeded(sdk_response(status))

    def test_the_only_answer_left_is_acceptance(self) -> None:
        """Every non-2xx leaves through the raise, so ``False`` is unreachable.

        Asserted rather than assumed: a caller writing ``if not
        succeeded(...)`` around a branch is writing dead code, and the
        branches that did are what this change removed.
        """
        assert succeeded(sdk_response(204)) is True


class TestPollUntil:
    """One loop for the three waits that each used to write their own."""

    def _immediate(self, answers: list[Any]) -> Callable[[], Any]:
        replies = iter(answers)
        return lambda: next(replies)

    def test_it_answers_with_whatever_the_predicate_answered(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("rl_cli.services.http.time.sleep", lambda _: None)

        answer = poll_until(
            self._immediate([PollState.PENDING, PollState.PENDING, {"status": "done"}]),
            timeout=60,
            backoff=PollBackoff(1),
        )

        assert answer == {"status": "done"}

    def test_running_out_of_time_is_told_apart_from_giving_up(self) -> None:
        assert (
            poll_until(lambda: PollState.PENDING, timeout=0, backoff=PollBackoff(0))
            is PollState.TIMED_OUT
        )

    def test_the_deadline_is_exclusive_so_no_poll_is_made_once_time_reaches_it(
        self, monkeypatch: Any
    ) -> None:
        """The instant the clock reaches the deadline the wait is over; it is
        not entitled to one more poll at ``time == deadline``."""
        ticks = iter([100.0, 105.0, 105.0, 105.0])
        monkeypatch.setattr("rl_cli.services.http.time.time", lambda: next(ticks))
        monkeypatch.setattr("rl_cli.services.http.time.sleep", lambda _: None)
        polls: list[bool] = []

        def poll() -> PollState:
            polls.append(True)
            return PollState.PENDING

        result = poll_until(poll, timeout=5, backoff=PollBackoff(1))

        assert result is PollState.TIMED_OUT
        assert polls == [], "a poll fired at the exact deadline the loop should have stopped at"

    def test_the_pacing_cannot_be_stated_twice(self) -> None:
        """``interval`` was a second way to say it, and the ignored one.

        It paced the loop only when no ``backoff`` was passed, so the one
        caller that passed both handed the same number over twice and a
        caller passing two different numbers would have had one of them
        silently dropped.
        """
        with pytest.raises(TypeError):
            poll_until(  # type: ignore[call-arg]
                lambda: PollState.PENDING, timeout=0, interval=5, backoff=PollBackoff(1)
            )

    def test_it_stops_asking_an_appliance_that_keeps_not_answering(self, monkeypatch: Any) -> None:
        """Only the loop with a backoff had this, and it is now every loop's."""
        polls = 0

        def unanswered() -> PollState:
            nonlocal polls
            polls += 1
            return PollState.UNANSWERED

        monkeypatch.setattr("rl_cli.services.http.time.sleep", lambda _: None)

        answer = poll_until(unanswered, timeout=300, backoff=PollBackoff(5))

        assert answer is PollState.ABANDONED
        assert polls == 3, "a poll that failed three times was asked again anyway"


def _recording_send(sent: list[Any]) -> Callable[..., MagicMock]:
    """A ``Session.send`` that records the timeout it was called with."""

    def send(*_args: Any, **kwargs: Any) -> MagicMock:
        sent.append(kwargs.get("timeout"))
        return MagicMock()

    return send


class TestApplyRequestTimeout:
    """The SDKs never pass `timeout` to requests, so we default it ourselves."""

    def _client(self) -> Any:
        client = MagicMock()
        client._session = requests.Session()
        return client

    def test_defaults_timeout_on_every_verb(self) -> None:
        client = apply_request_timeout(self._client(), 42)
        sent: list[Any] = []
        client._session.send = _recording_send(sent)

        for verb in ("get", "post", "put", "delete"):
            getattr(client._session, verb)("https://example.invalid/x")

        assert sent == [42, 42, 42, 42]

    def test_explicit_timeout_wins(self) -> None:
        client = apply_request_timeout(self._client(), 42)
        sent: list[Any] = []
        client._session.send = _recording_send(sent)

        client._session.get("https://example.invalid/x", timeout=5)

        assert sent == [5]

    def test_client_without_session_is_left_alone(self) -> None:
        client = object()
        assert apply_request_timeout(client, 42) is client


class TestTokenFetchIsBounded:
    """The token POST runs before the session exists, so it needs its own bound.

    ``A1000.__init__`` trades username/password for a token with a bare
    ``requests.post`` carrying no ``timeout``, so an appliance that
    accepts the connection and then stops answering used to hang every
    username/password invocation forever. No network is touched here:
    ``requests.post`` is replaced outright.
    """

    def _settings(self) -> Settings:
        settings = Settings()
        settings.a1000.host = "https://a1000.invalid"
        settings.a1000.token = None
        settings.a1000.username = "user"
        settings.a1000.password = "pass"
        settings.a1000.proxy = None
        settings.a1000.timeout = 17
        return settings

    def test_connect_bounds_the_token_post(self, monkeypatch: Any) -> None:
        from rl_cli.services.a1000 import A1000Service, A1000Session

        recorded: list[Any] = []

        def fake_post(*args: Any, **kwargs: Any) -> Any:
            recorded.append(kwargs.get("timeout"))
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {"token": "t"}
            return response

        monkeypatch.setattr(requests, "post", fake_post)

        service = A1000Session(self._settings()).service(A1000Service)
        service.output = MagicMock()
        assert service.connect() is True

        assert recorded == [17], "the SDK's token POST was issued without a timeout"

    def test_the_patch_is_undone(self) -> None:
        original = requests.post
        with default_post_timeout(1):
            assert requests.post is not original
        assert requests.post is original

    def test_explicit_timeout_wins(self) -> None:
        recorded: list[Any] = []
        original = requests.post
        try:
            requests.post = lambda *a, **kw: recorded.append(kw.get("timeout"))  # type: ignore[assignment]
            with default_post_timeout(42):
                requests.post("https://example.invalid", timeout=5)
        finally:
            requests.post = original
        assert recorded == [5]
