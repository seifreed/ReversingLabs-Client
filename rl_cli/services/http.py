"""Bounding, reading and pacing the HTTP the SDKs make on our behalf.

Neither ReversingLabs SDK bounds a request or reports a status it does
not recognise, so the timeouts, the response reading and the poll spacing
are ours to impose. Nothing here knows about a service.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Any

import requests


class PollState(Enum):
    """What one poll established, when what it established was not the answer.

    ``PENDING`` and ``UNANSWERED`` are what a predicate hands
    :func:`poll_until`; they are distinct because only ``UNANSWERED``
    slows the polling down. ``TIMED_OUT`` and ``ABANDONED`` are what
    :func:`poll_until` returns in place of an answer, so a caller can
    report running out of time differently from giving up.
    """

    PENDING = "pending"
    UNANSWERED = "unanswered"
    TIMED_OUT = "timed_out"
    ABANDONED = "abandoned"


class PollBackoff:
    """Spacing and give-up rule for a service's poll loop.

    A fixed-interval loop against a rate-limited appliance re-issues
    ``timeout / interval`` identical requests and reports that many
    identical errors. This doubles the interval after each failure, up to
    ``max_interval``, and stops after ``give_up_after`` consecutive ones.
    A poll that answers resets both.
    """

    def __init__(
        self, interval: float, *, max_interval: float = 60.0, give_up_after: int = 3
    ) -> None:
        self._base = self._interval = float(interval)
        self._max_interval = max_interval
        self._give_up_after = give_up_after
        self._failures = 0

    def keep_going(self, answered: bool) -> bool:
        """Record one poll; ``False`` once the API has failed too often."""
        if answered:
            self._failures = 0
            self._interval = self._base
            return True
        self._failures += 1
        self._interval = min(self._interval * 2, self._max_interval)
        return self._failures < self._give_up_after

    def sleep(self) -> None:
        """Wait the current interval before the next poll."""
        time.sleep(self._interval)


def poll_until[T](
    poll: Callable[[], T | PollState],
    *,
    timeout: float,
    backoff: PollBackoff,
    on_wait: Callable[[], None] | None = None,
) -> T | PollState:
    """Ask ``poll`` until it answers, the deadline passes, or it gives up.

    ``poll`` answers ``PENDING`` to be asked again, ``UNANSWERED`` when
    the appliance did not answer at all — the distinction ``backoff``
    paces itself by — or any other value, which is the answer and ends
    the wait. ``on_wait`` runs after each sleep, for the caller that is
    drawing a spinner over it.

    Every A1000 wait goes through here, so the pacing is the ``backoff``
    and nothing else: a wait cannot be written without one.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        answer = poll()
        if answer is not PollState.PENDING and answer is not PollState.UNANSWERED:
            return answer
        if not backoff.keep_going(answer is PollState.PENDING):
            return PollState.ABANDONED
        backoff.sleep()
        if on_wait is not None:
            on_wait()
    return PollState.TIMED_OUT


def apply_request_timeout(sdk_client: Any, timeout: int) -> Any:
    """Give an SDK client's ``requests.Session`` a default timeout.

    Neither the A1000 nor the TitaniumCloud SDK ever passes ``timeout`` to
    requests, so an appliance that accepts a connection and then stops
    responding hangs the CLI forever. Every session verb funnels through
    ``Session.request``, so defaulting it there covers every SDK call that
    goes through the session.

    It cannot cover ``A1000.__init__``, which trades username and password
    for a token with a bare ``requests.post`` before ``_session`` exists;
    :func:`default_post_timeout` bounds that one.

    Returns the client for chaining.
    """
    session = getattr(sdk_client, "_session", None)
    if session is None:
        return sdk_client

    original = session.request

    @functools.wraps(original)
    def request(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", timeout)
        return original(*args, **kwargs)

    session.request = request
    return sdk_client


@contextmanager
def default_post_timeout(timeout: int) -> Iterator[None]:
    """Bound bare ``requests.post`` calls made inside the block.

    Wrap SDK client construction in this. With username/password auth
    ``A1000.__init__`` fetches a token through the ``requests`` module
    rather than a session and with no ``timeout``, so an appliance that
    completes the TCP handshake and then goes quiet hangs every invocation
    forever.

    Patching the module attribute is the only seam the SDK leaves; it is
    restored on the way out, and is process-wide while held — fine for a
    single-threaded CLI, not for a library used from threads.
    """
    original = requests.post

    @functools.wraps(original)
    def post(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", timeout)
        return original(*args, **kwargs)

    requests.post = post
    try:
        yield
    finally:
        requests.post = original


def _response_message(response: Any) -> str:
    """Best-effort extraction of the server's explanation from a response."""
    try:
        body = response.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        for key in ("message", "error", "detail", "description"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
    text = (getattr(response, "text", "") or "").strip()
    return text[:200] if text else "no details returned"


def succeeded(response: Any) -> bool:
    """Whether the appliance accepted the call; it raises when it did not.

    The one status check in the service layer, and it answers ``True`` or
    raises — never ``False``, which a caller can forward without the
    server's explanation.

    An ``int`` ``status_code`` is required rather than looked for, so that
    a dict, a string or a mangled SDK object cannot be read as a success.
    Only 2xx is an acceptance; anything else — a 1xx, or a 3xx
    ``requests`` did not follow — is named through the raise, so
    ``HTTP 302: no details returned`` reaches the analyst.
    """
    status = getattr(response, "status_code", None)
    if not isinstance(status, int) or isinstance(status, bool):
        raise RuntimeError(f"Expected a response from the appliance, got {type(response).__name__}")
    if not 200 <= status < 300:
        raise RuntimeError(f"HTTP {status}: {_response_message(response)}")
    return True


def json_on(response: Any) -> Any:
    """Return ``response.json()`` once the appliance has accepted the call.

    The ReversingLabs SDK raises only for the statuses in its
    ``RESPONSE_CODE_ERROR_MAP``, which omits several the A1000 does send
    (428 for an unconfigured feature, for example), so :func:`succeeded`
    raises for those instead of letting them collapse into ``None``.

    It answers ``None`` only for a body that literally states ``null``:
    every status that is not an acceptance leaves through the raise, so a
    caller reading ``None`` as "we could not reach it" would be reading a
    body the appliance meant.
    """
    succeeded(response)
    return response.json()
