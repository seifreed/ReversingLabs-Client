"""The two decorators every service wrapper is written on top of.

They are what keeps a wrapper down to the call it makes: connect if the
connection is not open yet, report whatever went wrong through the
service's own ``handle_error``, and answer the caller with a default
instead of an exception.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec, TypeVar, cast

from rl_cli.services.protocols import ConnectedService, Reporting

P = ParamSpec("P")
R = TypeVar("R")

# The service the decorated method belongs to, bound to the protocol the
# decorator actually uses. These bounds are the whole check: a class with
# no ``handle_error`` is refused by mypy where the decorator is written,
# so neither decorator needs a runtime ``isinstance`` on each of some
# eighty decorated calls.
ReportingSelf = TypeVar("ReportingSelf", bound=Reporting)
ConnectedSelf = TypeVar("ConnectedSelf", bound=ConnectedService)

# ``default`` stays ``Any``, deliberately. Typing it as the wrapped
# method's return type is not expressible through a two-stage decorator:
# mypy solves the type variable from ``with_client(default=None)``, before
# the method is in sight, so every ``T | None`` wrapper would be told its
# default must be ``None`` and its return type therefore ``None`` too. The
# cost is the ``cast`` below: a method annotated ``-> bool`` and decorated
# ``@safe_call(default=None)`` returns ``None`` with mypy silent, so the
# pairing is a review item, not a checked one.


def with_client(
    *, default: Any = None
) -> Callable[
    [Callable[Concatenate[ConnectedSelf, P], R]], Callable[Concatenate[ConnectedSelf, P], R]
]:
    """Run the wrapped service method with the SDK client guaranteed to exist.

    On connect failure or uncaught exception, log via ``handle_error`` and
    return ``default``. Use ``default=None`` for ``T | None`` returns,
    ``default=False`` for ``bool`` returns, etc. Keep it immutable — the
    one object given here is handed to every failing call.
    """

    def decorator(
        func: Callable[Concatenate[ConnectedSelf, P], R],
    ) -> Callable[Concatenate[ConnectedSelf, P], R]:
        @functools.wraps(func)
        def wrapper(self: ConnectedSelf, /, *args: P.args, **kwargs: P.kwargs) -> R:
            try:
                if not self.client and not self.connect():
                    return cast(R, default)
                return func(self, *args, **kwargs)
            except Exception as exc:
                self.handle_error(exc, _build_error_context(func, args))
                return cast(R, default)

        return wrapper

    return decorator


def safe_call(
    *, default: Any = None
) -> Callable[
    [Callable[Concatenate[ReportingSelf, P], R]], Callable[Concatenate[ReportingSelf, P], R]
]:
    """Wrap a service method's exceptions into ``handle_error`` + ``default``.

    Companion to :func:`with_client` for services that build their own SDK
    handles per call (e.g. ``TitaniumCloudService`` instantiates a fresh
    ``ticloud.<API>`` object on each method) and therefore have no shared
    ``self.client`` to manage.
    """

    def decorator(
        func: Callable[Concatenate[ReportingSelf, P], R],
    ) -> Callable[Concatenate[ReportingSelf, P], R]:
        @functools.wraps(func)
        def wrapper(self: ReportingSelf, /, *args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(self, *args, **kwargs)
            except Exception as exc:
                self.handle_error(exc, _build_error_context(func, args))
                return cast(R, default)

        return wrapper

    return decorator


def _build_error_context(func: Callable[..., Any], args: tuple[Any, ...]) -> str:
    """Render an error context like ``method_name`` or ``method_name(arg0)``."""
    if args:
        return f"{func.__name__}({args[0]!r})"
    return func.__name__
