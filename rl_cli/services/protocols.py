"""What a service needs of its caller, named as protocols.

Structural types only: nothing here imports a service, a client or a
renderer, so the layer that reports to the user and the layer that
answers ``with_client`` can both be depended upon without dragging the
other in.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol


class Spinner(Protocol):
    """The progress widget a service polls under, reduced to what it uses.

    Spelled out rather than typed as a bare context manager so that mypy
    holds every stand-in — :class:`_SilentSpinner` included — to the
    members a long poll will reach for.
    """

    @property
    def task_ids(self) -> Sequence[Any]: ...

    def advance(self, task_id: Any, advance: float = ...) -> None: ...


class Notifier(Protocol):
    """The reporting surface a service needs from its caller.

    Services only ever tell the user what happened; they must not depend
    on *how* it is shown. ``RichOutput`` satisfies this structurally, and
    so can a silent stand-in that draws nothing at all.
    """

    def info(self, message: str) -> None: ...

    def warning(self, message: str) -> None: ...

    def error(self, message: str) -> None: ...

    def progress_spinner(self, message: str = ...) -> AbstractContextManager[Spinner]: ...


class _SilentSpinner:
    """A spinner that draws nothing, for a service nobody is watching."""

    task_ids: Sequence[Any] = (0,)

    def advance(self, task_id: Any, advance: float = 1) -> None:
        return None


class NullNotifier:
    """Reports nothing, and is what a service falls back to.

    Silence is the default so that constructing a service never pulls Rich
    into ``rl_cli.services``: the CLI is the layer that knows a terminal
    exists, and it passes its ``RichOutput`` to everything it builds.
    """

    def info(self, message: str) -> None:
        return None

    def warning(self, message: str) -> None:
        return None

    def error(self, message: str) -> None:
        return None

    @contextmanager
    def progress_spinner(self, message: str = "") -> Iterator[Spinner]:
        yield _SilentSpinner()


class Reporting(Protocol):
    """What :func:`~rl_cli.services.decorators.safe_call` needs of the service.

    One method, named structurally, which is what leaves ``decorators``
    depending on this module alone rather than on ``base`` and through it
    ``requests`` and the SDK's error map. Deliberately not
    ``runtime_checkable``: the decorators' type variables ask this
    statically, once, instead of on every decorated call.
    """

    def handle_error(self, error: Exception, context: str = "") -> None: ...


class ConnectedService(Reporting, Protocol):
    """What :func:`~rl_cli.services.decorators.with_client` needs of the service.

    A service is not a connection, so the connection lifecycle is not on
    ``BaseService``: TitaniumCloud builds an API handle per call and has
    none to manage, and the A1000 services share a connection that belongs
    to their :class:`~rl_cli.services.a1000.session.A1000Session`. Naming
    what the decorator actually uses is what lets it stay in one place
    without the base class carrying a lifecycle only half its subclasses
    have.
    """

    # Read-only, because reading is all the decorator does and because
    # ``A1000Service.client`` is deliberately a property: declared writable
    # here, this protocol is satisfied by no A1000 service at all.
    #
    # The SDK ships no type stubs, so ``Any`` is the truthful annotation.
    @property
    def client(self) -> Any: ...

    def connect(self) -> bool: ...
