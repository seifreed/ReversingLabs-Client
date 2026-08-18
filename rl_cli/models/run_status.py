"""Whether this invocation failed — the fact its exit status is taken from."""

from __future__ import annotations


class RunStatus:
    """One invocation's answer to "did anything fail?".

    Whether the process failed is not a rendering decision, so the flag is
    not kept by the renderer: ``RichOutput`` holds one of these and reports
    into it, and ``cli/main.py`` reads it for the exit status. Which
    reporting method fails the run is fixed — ``error`` does, ``problem``
    does not — rather than negotiated per call site.

    Deliberately one-way. A run that failed does not become a run that
    succeeded because something printed afterwards, which is what an
    assignable flag allows.
    """

    def __init__(self) -> None:
        self._failed = False

    @property
    def failed(self) -> bool:
        """Whether anything has reported a failure during this run."""
        return self._failed

    def fail(self) -> None:
        """Record that this run failed."""
        self._failed = True
