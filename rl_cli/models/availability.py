"""What a probe run of the ReversingLabs services is, and what it reports.

Here rather than in ``services`` because the statuses are a rule both
sides of a boundary agree on: the prober answers in them and the renderer
colours by them, and ``render`` may not import ``services``.

``TypedDict`` rather than dataclasses because this shape is also the
on-disk cache format: a cached answer is loaded and handed straight to the
same call sites, extra keys from older versions and all. Being a plain
dict at runtime is what lets those keys ride along untouched, while the
annotation still makes the reads checked rather than guesses.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any, TypedDict


class ServiceStatus(Enum):
    """Service availability status."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class ProbeGrade(Enum):
    """How a probe outcome reads: good news, a caveat, or bad news.

    Beside the statuses because grading one is a decision about the
    measurement, and the same decision wherever it is read: a renderer
    picks a colour from a grade rather than deciding for itself which
    status is bad news. None of the three is a failed run — `check-access`
    reports what it measured and exits 0 — so nothing that draws a grade
    has an exit status to choose (``tests/test_formatters.py``).
    """

    GOOD = "good"
    CAVEAT = "caveat"
    BAD = "bad"


_GRADES: dict[str, ProbeGrade] = {
    ServiceStatus.AVAILABLE.value: ProbeGrade.GOOD,
    ServiceStatus.UNAVAILABLE.value: ProbeGrade.CAVEAT,
}


def grade_of(status: str) -> ProbeGrade:
    """How a probe reporting ``status`` reads.

    Anything but the two graded above is bad news, ``ServiceStatus.ERROR``
    and a word this version does not know alike: a status nothing here
    recognises is not something to announce as a success.
    """
    return _GRADES.get(status, ProbeGrade.BAD)


class ServiceProbe(TypedDict):
    """What one probe measured about one service."""

    status: str
    message: str
    credentials_configured: bool
    api_accessible: bool


class AvailabilitySummary(TypedDict):
    """How many of the probed services answered."""

    services_available: int
    services_total: int


class Availability(TypedDict):
    """A whole probe run: what the CLI renders, and what the cache stores."""

    timestamp: str
    titanium_cloud: ServiceProbe
    a1000: ServiceProbe
    summary: AvailabilitySummary


# The services a probe run covers, as ``(label, key)`` in the order they
# are reported. One tuple for the document and the terminal lines alike, so
# that a service cannot reach one and not the other.
SERVICES: tuple[tuple[str, str], ...] = (
    ("TitaniumCloud", "titanium_cloud"),
    ("A1000", "a1000"),
)


def availability_block(availability: Mapping[str, Any]) -> dict[str, Any]:
    """The api_availability section of `config show`: status per service."""
    return {
        key: {"status": availability[key]["status"], "message": availability[key]["message"]}
        for _, key in SERVICES
    }


def availability_payload(availability: Mapping[str, Any]) -> dict[str, Any]:
    """What `check-access` puts on stdout: `show -a`'s block plus the panel's facts."""
    return {
        **availability_block(availability),
        "timestamp": availability["timestamp"],
        "summary": availability["summary"],
    }
