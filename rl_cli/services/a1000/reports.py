"""Analysis, PDF and dynamic-analysis reports."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, NewType

from rl_cli.models.validators import HashType, normalize_hash
from rl_cli.services.a1000.service import A1000_SHORT_HASH_TYPES, A1000Service
from rl_cli.services.decorators import with_client
from rl_cli.services.http import PollBackoff, PollState, json_on, poll_until, succeeded

# Report-creation status values returned by the PDF and dynamic-analysis
# status endpoints, which answer 200 whether or not the report is ready.
_REPORT_PENDING = 0
_REPORT_READY = 2

# ``xml`` never produced XML: it selects the TitaniumCore endpoint, which
# answers JSON. Kept as a legacy spelling so existing scripts keep working.
_REPORT_FORMAT_ALIASES = {"xml": "titanium"}


# The two formats the A1000 dynamic-analysis endpoints build. The CLI
# offers the same pair as a click.Choice; the service states it too
# because the service is also the library API.
_DYNAMIC_REPORT_FORMATS = ("pdf", "html")

# The two spellings of a sample a report fetch may be handed. Distinct
# types rather than two ``str`` parameters, because a report method takes
# one or the other and both are hexadecimal strings: transposed, they
# type-check and reach the appliance as a lookup about something else.
_TypedHash = NewType("_TypedHash", str)
_Digest = NewType("_Digest", str)


def _canonical_report_format(report_format: str) -> str:
    """``report_format`` with its legacy spelling resolved."""
    return _REPORT_FORMAT_ALIASES.get(report_format, report_format)


def _raised_by_the_call_itself(error: BaseException) -> bool:
    """Whether ``error`` came from making a call rather than from running it.

    A call the installed SDK no longer accepts raises in the frame that
    makes it, so its deepest frame belongs to this module; a failure inside
    the SDK always has a frame of its own underneath. Read from the frames
    rather than from the message, which is one CPython release's wording
    and matches unrelated ``TypeError``\\ s besides.

    So every ``fetch`` handed to
    :meth:`A1000ReportService._analysis_backed_report` must stay a bare SDK
    call spelled in this file. Wrap one in a helper elsewhere and its
    signature errors start reading as the appliance's answer.
    """
    frame = error.__traceback__
    while frame is not None and frame.tb_next is not None:
        frame = frame.tb_next
    return frame is not None and frame.tb_frame.f_globals.get("__name__") == __name__


class _ReportVariant(NamedTuple):
    """One report format :meth:`A1000ReportService.get_report` can fetch.

    ``binary`` is what that fetch answers — bytes rather than a parsed
    structure — and lives beside the fetch rather than in a set of its own
    so that a format cannot be dispatched by one and unknown to the other:
    a caller told to expect a structure writes the repr of a PDF into its
    report.

    ``fetch`` is handed the service, the hash as the analyst typed it and
    the normalized digest. Both, because the PDF endpoint has its own,
    narrower guard and reports what it refused in the analyst's words —
    and each under a type of its own, so a variant reaching for the wrong
    one is a type error rather than a report about another sample.
    """

    binary: bool
    fetch: Callable[[A1000ReportService, _TypedHash, _Digest], Any]


class A1000ReportService(A1000Service):
    """Service for fetching and creating reports about a known sample."""

    def _report_ready(self, check: Callable[[], Any]) -> bool | PollState:
        """Whether the report is built, from one call to a status endpoint.

        The endpoint answers 200 throughout; readiness is carried in the
        body's ``status`` field (0 = being created, 2 = ready).

        Only an explicit "pending" keeps polling: any other value,
        including a body with no status at all, ends the wait rather than
        blocking for the full timeout on a response shape we do not
        understand.
        """
        body = json_on(check())
        entries = body if isinstance(body, dict) else {}
        status = entries.get("status")
        if status == _REPORT_READY:
            return True
        if status == _REPORT_PENDING:
            return PollState.PENDING
        message = entries.get("status_message") or entries.get("message")
        self.output.error(message or f"Unexpected report status: {status!r}")
        return False

    def _wait_for_report(
        self, check: Callable[[], Any], *, timeout: int = 300, interval: int = 5
    ) -> bool:
        """Poll a report-creation status endpoint until the report is ready.

        Report generation regularly takes 45s or more, so there is no
        sleep short enough to substitute for actually asking.
        """
        ready = poll_until(
            lambda: self._report_ready(check), timeout=timeout, backoff=PollBackoff(interval)
        )
        if ready is PollState.TIMED_OUT:
            self.output.warning(f"Report was not ready within {timeout} seconds")
            return False
        return ready is True

    def _valid_report_format(self, report_format: str) -> str | None:
        """The dynamic-report format to send, or ``None`` after saying why.

        All three dynamic-analysis methods go through this, including the
        one that writes a file to disk.

        The format, not a verdict, for the reason :meth:`_valid_sha1` gives:
        a caller forwarding its own argument sends a value no guard has
        looked at.
        """
        wanted = report_format.strip().lower()
        if wanted not in _DYNAMIC_REPORT_FORMATS:
            self.output.error(
                f"Invalid report format '{report_format}'. Expected "
                f"{' or '.join(_DYNAMIC_REPORT_FORMATS)}."
            )
            return None
        return wanted

    def _valid_sha1(self, hash_value: str) -> str | None:
        """The SHA1 to send, or ``None`` after saying why.

        The dynamic-analysis endpoints take SHA1 and nothing else. The SDK
        enforces it too, but answers a bare ``('sha1',)``, so the three
        commands over these endpoints say it themselves, and say the same
        thing.

        The digest, not a verdict: the guard judges the stripped,
        lowercased hash, so a caller forwarding its own argument would send
        the appliance a value no guard had looked at — a hash pasted with
        surrounding whitespace fails inside the SDK as "not a valid
        hexadecimal value".
        """
        hash_type = self._valid_hash(hash_value)
        if not hash_type:
            return None
        if hash_type != HashType.SHA1:
            self.output.error("Dynamic reports require SHA1 hash")
            return None
        return normalize_hash(hash_value)

    def _analysis_backed_report(self, fetch: Callable[[], Any]) -> Any | None:
        """One report fetch the SDK gates on an analysis-status read.

        Both gated endpoints count that page with
        ``len(status.json().get("results"))``, so a 200 carrying no
        ``results`` — a proxy interstitial, a rewritten error page, a
        future API version — raises ``TypeError`` inside the SDK, where
        this repo's envelope guards cannot reach it. It is named here as
        the envelope failure it is.

        A ``TypeError`` from calling the SDK wrongly is ours and must keep
        its own message: this SDK deletes public API in minor releases,
        which pyproject pins ``<3`` against, and a signature change dressed
        as a server-side envelope problem sends the analyst to look at the
        appliance.

        Only the SDK call is wrapped, and a ``TypeError`` from reading the
        body is left alone: a pre-flight status read to tell them apart
        would double this command's requests.
        """
        try:
            response = fetch()
        except TypeError as error:
            if _raised_by_the_call_itself(error):
                raise
            self.output.error("Analysis-status response carried no results list")
            return None
        return json_on(response)

    @staticmethod
    def report_is_binary(report_format: str) -> bool:
        """Whether :meth:`get_report` answers bytes instead of a structure.

        Read off the same table :meth:`get_report` dispatches on, so a
        format cannot be fetched by one and unknown to the other. A format
        neither knows is not binary, and :meth:`get_report` refuses it.
        """
        variant = _REPORT_VARIANTS.get(_canonical_report_format(report_format))
        return variant is not None and variant.binary

    def _pdf_report(self, hash_value: _TypedHash) -> bytes | None:
        """Create the PDF report on the appliance, wait for it, download it.

        The PDF endpoint is the one report format that refuses SHA512, so
        the narrower guard lives here rather than on :meth:`get_report`:
        json and titanium take a SHA512 happily, and rejecting it for all
        three would refuse input the appliance accepts. It is therefore the
        one variant handed the typed hash, so that its refusal names what
        the analyst wrote.
        """
        digest = self._supported_hash(hash_value, A1000_SHORT_HASH_TYPES)
        if digest is None:
            return None
        succeeded(self.client.create_pdf_report(digest))
        if not self._wait_for_report(lambda: self.client.check_pdf_report_creation(digest)):
            return None
        response = self.client.download_pdf_report(digest)
        succeeded(response)
        return response.content

    def _titanium_report(self, digest: _Digest) -> dict[str, Any] | None:
        """The TitaniumCore report, as ``get_report`` fetches it.

        Not through :meth:`get_titanium_core_report_v2`: that one is an
        entry point of its own, and calling it from inside another would
        guard the hash twice and wrap one failure in two ``with_client``
        handlers.
        """
        return json_on(self.client.get_titanium_core_report_v2(digest))

    def _detailed_report(self, digest: _Digest) -> Any | None:
        """The detailed analysis report, which the SDK gates on a status read."""
        return self._analysis_backed_report(
            lambda: self.client.get_detailed_report_v2(
                sample_hashes=digest,
                include_networkthreatintelligence=False,
                skip_reanalysis=True,
            )
        )

    @with_client(default=None)
    def get_report(self, hash_value: str, report_format: str = "json") -> Any | None:
        """Fetch the analysis report in ``json``, ``titanium`` or ``pdf`` format.

        ``xml`` is accepted as a legacy spelling of ``titanium``: it never
        produced XML, it selects the TitaniumCore endpoint, which answers
        JSON. For ``pdf`` the report is created on demand, waited for, and
        then downloaded.

        Which endpoint answers and whether it answers bytes are one fact,
        held in :data:`_REPORT_VARIANTS`, so a format this dispatches
        cannot be one :meth:`report_is_binary` has never heard of.
        """
        digest = self._supported_hash(hash_value)
        if digest is None:
            return None

        wanted = _canonical_report_format(report_format)
        variant = _REPORT_VARIANTS.get(wanted)
        if variant is None:
            self.output.error(f"Unsupported report format: {wanted}")
            return None
        return variant.fetch(self, _TypedHash(hash_value), _Digest(digest))

    @with_client(default=None)
    def get_summary_report_v2(self, hash_value: str) -> dict[str, Any] | None:
        digest = self._supported_hash(hash_value)
        if digest is None:
            return None
        return self._analysis_backed_report(lambda: self.client.get_summary_report_v2(digest))

    @with_client(default=None)
    def get_titanium_core_report_v2(self, hash_value: str) -> dict[str, Any] | None:
        digest = self._supported_hash(hash_value)
        if digest is None:
            return None
        return json_on(self.client.get_titanium_core_report_v2(digest))

    # ---------- Dynamic analysis report ----------
    @with_client(default=None)
    def create_dynamic_report(
        self, hash_value: str, report_format: str = "pdf"
    ) -> dict[str, Any] | None:
        """Initiate creation of a dynamic-analysis report (SHA1 only)."""
        wanted = self._valid_report_format(report_format)
        if wanted is None:
            return None
        digest = self._valid_sha1(hash_value)
        if digest is None:
            return None
        return json_on(
            self.client.create_dynamic_analysis_report(digest, wanted),
        )

    @with_client(default=None)
    def download_dynamic_report(self, hash_value: str, report_format: str = "pdf") -> bytes | None:
        wanted = self._valid_report_format(report_format)
        if wanted is None:
            return None
        digest = self._valid_sha1(hash_value)
        if digest is None:
            return None

        # Downloading before the appliance has finished building the
        # report just 404s, so wait for it the same way the PDF flow does.
        if not self._wait_for_report(
            lambda: self.client.check_dynamic_analysis_report_status(digest, wanted)
        ):
            return None

        response = self.client.download_dynamic_analysis_report(digest, report_format=wanted)
        succeeded(response)
        return response.content

    @with_client(default=None)
    def check_dynamic_analysis_status(
        self, hash_value: str, report_format: str = "pdf"
    ) -> dict[str, Any] | None:
        wanted = self._valid_report_format(report_format)
        if wanted is None:
            return None
        digest = self._valid_sha1(hash_value)
        if digest is None:
            return None
        return json_on(self.client.check_dynamic_analysis_report_status(digest, wanted))


# Every report format ``get_report`` fetches, and what each one answers.
# Written here once: the dispatch and :meth:`A1000ReportService.report_is_binary`
# read this same table, so a format added to one is a format the other knows.
_REPORT_VARIANTS: dict[str, _ReportVariant] = {
    "json": _ReportVariant(
        binary=False,
        fetch=lambda service, _typed, digest: service._detailed_report(digest),
    ),
    "titanium": _ReportVariant(
        binary=False,
        fetch=lambda service, _typed, digest: service._titanium_report(digest),
    ),
    "pdf": _ReportVariant(
        binary=True,
        fetch=lambda service, typed, _digest: service._pdf_report(typed),
    ),
}
