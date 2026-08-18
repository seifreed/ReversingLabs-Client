"""Keeping a per-call fact about an answer from outliving that call."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

_Service = TypeVar("_Service", bound=type[Any])

# Where a covered class, and each entry point of one, records the
# attribute it clears. Written by the decorator and read by
# :func:`cleared_attribute`, so that "is this covered?" is a question with
# an answer rather than an assumption a reader has to make from the
# presence of a decorator line.
_CLEARS = "__cleared_per_call__"


def cleared_per_call(attribute: str) -> Callable[[_Service], _Service]:
    """Clear ``attribute`` before every public method of the class runs.

    The attribute it names carries a fact about the call that has just
    finished — the appliance is still holding records this answer does not
    carry — so a caller reading it after a call that measured no paging at
    all must not be handed the previous call's ``True``: that reports a
    whole answer as a partial one, under a remedy that would fetch nothing.

    Clearing here rather than in each method is what makes the rule hold
    for a method added later: an entry point cannot forget a reset it does
    not perform. The one helper that measured the fact is then the only
    thing that sets it, and it does so after its last request. One
    mechanism for both halves of this package, because a second would be a
    convention a reader learns from one subpackage and is wrong about in
    the other.

    Only the methods the class itself defines are wrapped: those are its
    lookups. An inherited ``connect`` or ``test_connection`` says nothing
    about any answer, and clearing on one would let a liveness probe
    between a lookup and its caller's read erase a verdict that was true.
    The cost of that line is a public method a *mixin* supplies, which is
    inherited and so left alone even when it is a lookup; nothing here can
    tell those two inherited shapes apart, so the invariant tests pin the
    inherited public surface of every covered service instead.
    """

    def decorate(service: _Service) -> _Service:
        for name, member in list(vars(service).items()):
            if name.startswith("_"):
                continue
            entry_point = _cleared_member(member, attribute)
            if entry_point is not None:
                setattr(service, name, entry_point)
        setattr(service, _CLEARS, attribute)
        return service

    return decorate


def cleared_attribute(target: Any) -> str | None:
    """The attribute ``target`` clears on entry, or ``None`` if nothing does.

    Takes a class or one of its entry points, a property among them.
    Reads what the object itself carries and not what it inherits: a
    subclass of a covered service has its own methods to wrap and is
    covered once it is decorated too — until then it is a service holding
    a verdict nothing clears, which is what the invariant tests report it
    as.
    """
    if isinstance(target, property):
        return None if target.fget is None else cleared_attribute(target.fget)
    if isinstance(target, type):
        return vars(target).get(_CLEARS)
    marker: str | None = getattr(target, _CLEARS, None)
    return marker


def _cleared_member(member: Any, attribute: str) -> Any | None:
    """``member`` with the reset in front of each way in, or ``None`` for none.

    A plain method and a property are both reached with an instance in
    hand, so both are entry points and both are wrapped — a property read
    after a call that measured no paging must not answer out of the
    previous call's state any more than a method may.

    A ``staticmethod`` or ``classmethod`` is left alone, and that is the
    one deliberate gap: it is handed no instance, so it can neither set the
    verdict nor read one. ``A1000SampleService.build_search_query`` is the
    shape — local knowledge about the query language, answering the same
    string for every appliance.
    """
    if inspect.isfunction(member):
        return _clearing(member, attribute)
    if isinstance(member, property):
        return property(
            None if member.fget is None else _clearing(member.fget, attribute),
            None if member.fset is None else _clearing(member.fset, attribute),
            None if member.fdel is None else _clearing(member.fdel, attribute),
            member.__doc__,
        )
    return None


def _clearing(method: Callable[..., Any], attribute: str) -> Callable[..., Any]:
    """``method``, with ``attribute`` cleared on the way in."""

    @functools.wraps(method)
    def entry_point(self: Any, *args: Any, **kwargs: Any) -> Any:
        setattr(self, attribute, False)
        return method(self, *args, **kwargs)

    # After ``functools.wraps``, which copies the wrapped function's own
    # ``__dict__`` over this one's.
    setattr(entry_point, _CLEARS, attribute)
    return entry_point
