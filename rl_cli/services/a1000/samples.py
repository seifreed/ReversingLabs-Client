"""Sample submission, retrieval, listing and bulk operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

from rl_cli.models.payload import ReanalysisOutcome
from rl_cli.models.shapes import dict_rows, list_section, mapping
from rl_cli.services.a1000.service import A1000_HASH_TYPES, A1000Service, list_from_envelope
from rl_cli.services.a1000.session import A1000Session
from rl_cli.services.decorators import with_client
from rl_cli.services.hashes import normalized_batch
from rl_cli.services.http import PollBackoff, PollState, json_on, poll_until, succeeded
from rl_cli.services.partial_answers import cleared_per_call
from rl_cli.services.protocols import Notifier
from rl_cli.services.walks import fetch_search_page, numbered_search
from rl_cli.storage.archives import unpack_private
from rl_cli.storage.files import write_private_bytes

_MAX_RECORDS_PER_PAGE = 100

# How many Advanced Search pages one walk may fetch. Every page is a POST
# to the appliance, and neither the endpoint's own "another page follows"
# nor the caller's ``limit`` bounds what a search may spend.
#
# 1000 is exactly what the largest ``--limit`` this CLI accepts needs —
# ``_shared_inputs.MAX_LIMIT`` is 100000 and a v3 page holds 100 — so the
# budget stops a walk that has lost the plot without ever capping an answer
# the analyst was allowed to ask for. Held here rather than read off
# ``MAX_LIMIT``: this service must not depend on the command layer, and a
# test pins the two in step.
_MAX_SEARCH_PAGES = 1_000

# Flag shortcuts for the A1000 search DSL, most-preferred first: this
# order *is* the precedence :func:`build_search_query` resolves by, and
# what :func:`unused_search_inputs` reports as dropped is the rest of it.
# Keep it the one place the precedence is written down, or the report of
# what was ignored can name the shortcut that in fact ran.
_CLASSIFICATION_QUERIES = {
    "malicious": "classification:malicious",
    "clean": "classification:clean",
}
_ANY_SAMPLE_QUERY = "available:true"


def _records_per_page(limit: int) -> int:
    """Clamp ``limit`` to the Advanced Search v3 page-size range (1-100)."""
    return max(1, min(limit, _MAX_RECORDS_PER_PAGE))


def _requested_shortcuts(*, malicious: bool, clean: bool) -> list[str]:
    """The classification shortcuts the caller set, most-preferred first."""
    wanted = {"malicious": malicious, "clean": clean}
    return [name for name in _CLASSIFICATION_QUERIES if wanted[name]]


def unused_search_inputs(
    query: str | None, *, malicious: bool = False, clean: bool = False
) -> list[str]:
    """The classification shortcuts :meth:`build_search_query` resolved away.

    Resolving a conflict is not the same as being asked for one thing: a
    caller that sets both shortcuts from variables, or adds ``clean`` to a
    saved ``query``, searches for something other than what it asked for
    and cannot see that in the query that was posted.

    Everything the winner left over, so the answer follows the precedence
    rather than restating it: an explicit ``query`` beats every shortcut,
    and otherwise the first shortcut asked for is the one that runs.

    The names of the inputs, not of any option spelling them: what to call
    them belongs to the caller that offers them.
    """
    requested = _requested_shortcuts(malicious=malicious, clean=clean)
    return requested if query else requested[1:]


class ReanalysisBatch(NamedTuple):
    """A batch-reanalyze answer, graded once for everyone who reads it.

    The appliance answers 200 with a per-sample verdict inside, so how many
    samples were taken is a fact about the answer rather than about the
    sentence a caller writes over it or the table a renderer draws under
    it. Grade it here and nowhere else, or the sentence and the rows can
    disagree about the same entries.

    ``entries`` is the answer exactly as it came — what ``-o json`` emits
    and what the table draws — and ``outcomes`` is that same list graded,
    entry for entry, in order. Nothing here is escaped or shortened: these
    are appliance strings, as everywhere else below :mod:`rl_cli.models`.
    """

    entries: list[Any]
    outcomes: tuple[ReanalysisOutcome, ...]

    @classmethod
    def of(cls, entries: list[Any]) -> ReanalysisBatch:
        """Grade every entry of ``entries``, in the one reading of an entry there is."""
        return cls(
            entries=entries, outcomes=tuple(ReanalysisOutcome.of(entry) for entry in entries)
        )

    @property
    def answered(self) -> int:
        """How many samples the appliance said anything at all about."""
        return len(self.outcomes)

    @property
    def accepted(self) -> int:
        """How many of them it took."""
        return sum(1 for outcome in self.outcomes if outcome.accepted)

    @property
    def refused(self) -> int:
        """How many it answered for and turned down."""
        return self.answered - self.accepted


@cleared_per_call("pages_left_unfetched")
class A1000SampleService(A1000Service):
    """Service for submitting, fetching, listing and removing samples."""

    def __init__(self, session: A1000Session, output: Notifier | None = None):
        super().__init__(session, output)
        # Whether the last Advanced Search left hits behind, for a caller
        # that can offer to fetch them. Both search paths measure it — a
        # single page from ``more_pages``, an aggregated walk from stopping
        # at its cap — because an answer cut short at the cap is no less
        # partial than one cut short at a page boundary. A fact about the
        # answer, not advice about what to do with it: the remedy is the
        # caller's sentence to write.
        #
        # It speaks for the call that just finished and for no earlier one,
        # which is why the class decorator clears it at every entry point
        # and only :meth:`_one_search_page` and :meth:`_aggregated_search`
        # set it.
        self.pages_left_unfetched = False

    # ---------- File submission ----------
    @with_client(default=None)
    def upload_file(self, file_path: Path, comment: str | None = None) -> dict[str, Any] | None:
        """Upload ``file_path`` to A1000 for analysis.

        Adds a top-level ``task_id`` (the SHA1 from the SDK response) to the
        returned payload to make the upload-then-poll workflow ergonomic.
        """
        if not file_path.exists():
            self.output.error(f"File not found: {file_path}")
            return None

        response = self.client.submit_file_from_path(
            file_path=str(file_path), comment=comment if comment else None
        )
        try:
            result = json_on(response)
        except ValueError:
            # Only a 2xx reaches here — ``json_on`` raises for every other
            # status — and a 2xx is A1000 saying it took the file, which an
            # unreadable body does not take back. No ``task_id`` in the
            # result, so ``--wait`` says why it cannot wait rather than
            # pretending it did.
            self.output.warning("Upload accepted, but its response body could not be read")
            return {"status": "accepted", "file_name": file_path.name}
        if isinstance(result, dict):
            task_id = _extract_hash_from_upload(result)
            if task_id:
                result["task_id"] = task_id
        return result

    @with_client(default=None)
    def _analysis_status_page(self, task_id: str) -> dict[str, Any] | None:
        """The whole analysis-status body for ``task_id``, or ``None`` if unread.

        ``None`` means one thing here, and it is what the poll loop counts
        against its give-up rule: the appliance did not answer. Every 200 —
        including one carrying an empty ``results`` — comes back as the
        body it is, so :meth:`_analysis_of` can tell "not yet" from "no
        answer".
        """
        body = json_on(self.client.file_analysis_status([task_id]))
        return body if isinstance(body, dict) else None

    def _analysis_of(self, task_id: str) -> dict[str, Any] | PollState:
        """One poll of the analysis-status endpoint, as a wait can read it.

        An appliance that did not answer is told apart from one that
        answered "not yet", because only the first should slow the polling
        down and count against the give-up rule. A hash the appliance has
        not registered yet is answered 200 with ``{"results": []}``, so the
        whole body is read here rather than through a wrapper that unwraps
        the single entry and cannot tell that page from a call that never
        landed: a body we could read is an answer, even when the answer is
        "nothing yet".
        """
        page = self._analysis_status_page(task_id)
        if page is None:
            return PollState.UNANSWERED
        status = _first_status(page)
        if status is not None and status.get("status") == "processed":
            return status
        return PollState.PENDING

    def wait_for_analysis(
        self, task_id: str, timeout: int = 300, interval: int = 5
    ) -> dict[str, Any] | None:
        """Poll until A1000 holds an analysis for ``task_id``, or ``timeout``.

        The endpoint knows two answers about a hash — "processed" and
        "not_found" — so what it establishes is that an analysis of this
        file exists, not that the submission that started this wait is the
        one that produced it: for a file the appliance already held, the
        first poll answers "processed" from the earlier analysis. Callers
        must not report the return value as "this submission finished".

        Running out of time answers ``None`` through ``error``, not a
        warning: ``--wait`` is a promise, and a caller that exits 0 on an
        unfinished analysis will go on to report it as a finished one.
        """
        with self.output.progress_spinner(f"Waiting for analysis {task_id}...") as progress:
            task = progress.task_ids[0]
            answer = poll_until(
                lambda: self._analysis_of(task_id),
                timeout=timeout,
                backoff=PollBackoff(interval),
                on_wait=lambda: progress.advance(task),
            )
        if isinstance(answer, PollState):
            self.output.error(
                "Giving up: the last status checks all failed"
                if answer is PollState.ABANDONED
                else f"Analysis did not complete within {timeout} seconds"
            )
            return None
        return answer

    @with_client(default=False)
    def delete_sample(self, hash_value: str) -> bool:
        # The guard judges the stripped, lowercased hash, so it is the
        # returned value that goes to the SDK, never the raw argument.
        sample = self._supported_hash(hash_value)
        if sample is None:
            return False
        return succeeded(self.client.delete_samples(sample))

    @with_client(default=None)
    def reanalyze_sample(self, hash_value: str) -> dict[str, Any] | None:
        sample = self._supported_hash(hash_value)
        if sample is None:
            return None
        return json_on(
            self.client.reanalyze_samples_v2(sample, titanium_core=True, titanium_cloud=True)
        )

    # ---------- Sample download ----------
    @with_client(default=False)
    def download_sample(self, hash_value: str, output_path: Path) -> bool:
        sample = self._supported_hash(hash_value)
        if sample is None:
            return False
        response = self.client.download_sample(sample)
        succeeded(response)
        # Not streamed, and not fixable from here: the SDK's __get_request
        # never passes stream=True, so requests has already read the whole
        # sample into memory before this line runs. What this does own is
        # that a failed or interrupted write leaves no truncated file.
        write_private_bytes(output_path, response.content)
        return True

    @with_client(default=None)
    def list_extracted_files(self, hash_value: str) -> list[dict[str, Any]] | None:
        """The files the appliance extracted, or ``None`` if it did not say.

        The endpoint answers ``{"count": n, "results": [...]}``. An
        envelope spelled any other way is a failed listing rather than an
        empty one — the line :func:`list_from_envelope` draws for every
        A1000 listing.
        """
        sample = self._supported_hash(hash_value)
        if sample is None:
            return None
        data = json_on(self.client.list_extracted_files_v2(sample))
        if data is None:
            return None
        results = list_from_envelope(data, "results")
        if results is None:
            self.output.error("Extracted-files response carried no results list")
        return results

    @with_client(default=False)
    def download_extracted_files(self, hash_value: str, output_dir: Path) -> bool:
        sample = self._supported_hash(hash_value)
        if sample is None:
            return False
        response = self.client.download_extracted_files(sample)
        succeeded(response)
        unpack_private(response.content, output_dir, self.output.warning)
        return True

    # ---------- Listings & search ----------
    def list_samples(self, limit: int = 100) -> list[dict[str, Any]] | None:
        """Every available sample, as the whole records the endpoint returned.

        ``list`` and ``search`` read the same endpoint and hand back the
        same whole records. Do not project them down to the panel's columns
        here: that would truncate the filenames ``-o json`` and ``-o sarif``
        emit, and the renderer shortens what it cannot fit.
        """
        return self.advanced_search(_ANY_SAMPLE_QUERY, limit)

    @staticmethod
    def build_search_query(
        query: str | None, *, malicious: bool = False, clean: bool = False
    ) -> str:
        """Turn the classification shortcuts into an A1000 search query.

        An explicit ``query`` always wins; otherwise the first shortcut
        :data:`_CLASSIFICATION_QUERIES` names is the one that runs, and
        with neither flag every available sample matches.
        """
        if query:
            return query
        requested = _requested_shortcuts(malicious=malicious, clean=clean)
        if not requested:
            return _ANY_SAMPLE_QUERY
        return _CLASSIFICATION_QUERIES[requested[0]]

    @with_client(default=None)
    def advanced_search(
        self, query: str, limit: int = 100, page: int = 1
    ) -> list[dict[str, Any]] | None:
        # A v3 page holds at most 100 records, so anything larger has to be
        # walked page by page.
        if limit < 1:
            self.output.error("Invalid search limit: expected a positive integer")
            return None
        if page < 1:
            self.output.error("Invalid search page: expected a positive integer")
            return None
        if limit > _MAX_RECORDS_PER_PAGE:
            if page != 1:
                self.output.warning(
                    f"--limit above {_MAX_RECORDS_PER_PAGE} cannot be combined with --page; "
                    f"returning at most {_MAX_RECORDS_PER_PAGE} records"
                )
            else:
                return self._aggregated_search(query, limit)
        return self._one_search_page(query, limit, page)

    def _aggregated_search(self, query: str, limit: int) -> list[dict[str, Any]] | None:
        """Walk the pages up to ``limit``, within a budget, saying what was left.

        The walk itself, and what each of its stops is for, is
        :func:`~rl_cli.services.walks.numbered_walk`. Every page is asked
        for at the largest size the endpoint serves, because every one of
        them is a POST: the SDK's own default of 20 makes ``--limit 1000``
        cost 50 requests where the maximum needs 10.

        With :meth:`_one_search_page`, one of the two places
        :attr:`pages_left_unfetched` is set, and it is set after the last
        request.
        """
        walked = numbered_search(
            self.client.advanced_search_v3,
            query,
            limit=limit,
            records_per_page=_MAX_RECORDS_PER_PAGE,
            max_pages=_MAX_SEARCH_PAGES,
            report=self.output.error,
        )
        if walked is None:
            return None
        self.pages_left_unfetched = walked.pages_left_unfetched
        return walked.records

    def _one_search_page(self, query: str, limit: int, page: int) -> list[dict[str, Any]] | None:
        """One Advanced Search page, saying so when the appliance held more.

        The v3 envelope states ``more_pages`` and ``next_page`` beside the
        entries — the SDK's own aggregator loops on exactly those two
        (``ReversingLabs/SDK/a1000.py``) — so reading only ``entries``
        would turn a query matching thousands into a silent "Found 100
        samples".

        What is said here is the fact that was measured and nothing about
        what to type next: naming ``--limit`` would make this service
        unusable to a caller that has no such option.
        :attr:`pages_left_unfetched` is how the caller learns of it in
        order to offer its own remedy.
        """
        found = fetch_search_page(
            self.client.advanced_search_v3,
            query,
            page,
            _records_per_page(limit),
            self.output.error,
        )
        if found is None:
            return None
        self.pages_left_unfetched = found.more_pages
        if found.more_pages:
            self.output.warning("More samples matched than one page holds; showing this page only.")
        return found.entries

    # ---------- Batch ----------
    @with_client(default=None)
    def batch_delete_samples(self, hashes: list[str]) -> int | None:
        """Remove several samples through the bulk endpoint.

        The whole list goes to ``delete_samples`` in one call because the
        SDK picks the endpoint from the argument type — a string uses the
        single-sample DELETE, a list uses the bulk v2 endpoint — and this
        appliance answers 403 to the single endpoint for a token the bulk
        one accepts.

        Bulk removal is asynchronous, so the task is polled to completion.
        The appliance's own completion message warns that finishing does
        not guarantee every sample was removed, so the count returned is
        what it accepted, not what it confirmed — and acceptance is decided
        by the 202, not by the poll that follows.

        ``None`` is "we never got to ask": a batch this endpoint cannot
        take, or a call that did not land. A number is what the appliance
        was asked to remove. Keep the two distinct — collapsed into a
        single ``0``, a removal never heard of reads like a removal that
        accepted nothing.
        """
        batch = normalized_batch(hashes, A1000_HASH_TYPES, self.output.error)
        if batch is None:
            # Reported by the guard, and nothing was sent: not a count.
            return None
        response = self.client.delete_samples(batch)
        succeeded(response)

        try:
            body = response.json()
        except ValueError:
            # The 202 is the acceptance, and a body we cannot read does not
            # take it back, so this must not fall through to the ``None``.
            # With no task id there is nothing to poll, and the warning is
            # all the confirmation there is.
            self.output.warning(
                "Removal accepted, but its response body could not be read; "
                "the removal task cannot be followed. "
                "Re-check the samples before assuming they survived."
            )
            return len(hashes)
        task_id = body.get("id") if isinstance(body, dict) else None
        if task_id:
            self._wait_for_removal(str(task_id))
        return len(hashes)

    def _removal_finished(self, task_id: str) -> bool | PollState:
        """One poll of a bulk-removal task; it answers 202 until it finishes.

        Any other status ends the wait — ``succeeded`` raises for anything
        that is not an acceptance — so a status this endpoint has no
        meaning for is reported rather than waited out.
        """
        response = self.client.check_sample_removal_status_v2(task_id)
        if response.status_code == 202:
            return PollState.PENDING
        return succeeded(response)

    def _wait_for_removal(self, task_id: str, *, timeout: int = 300, interval: int = 5) -> None:
        """Wait out an accepted bulk-removal task, and never un-accept it.

        Nothing observed here can withdraw the 202 that got us this far: a
        poll that errors, times out or is interrupted leaves a removal
        running on the appliance that will finish. So every exit from here
        is a warning naming the task, never an error.
        """
        reason = f"did not finish within {timeout} seconds"
        try:
            finished = poll_until(
                lambda: self._removal_finished(task_id),
                timeout=timeout,
                backoff=PollBackoff(interval),
            )
            if finished is True:
                return
        except KeyboardInterrupt:
            # Named, then re-raised: Ctrl-C must mention the removal task
            # still running on the appliance, and must still abort.
            self._warn_removal_running(task_id, "was interrupted")
            raise
        except Exception as exc:
            # ``str(exc)`` before the fallback, because an exception object
            # is always truthy: ``exc or type(exc).__name__`` renders an
            # empty ``()`` for the SDK errors that carry no message.
            reason = f"could not be confirmed ({str(exc) or type(exc).__name__})"
        self._warn_removal_running(task_id, reason)

    def _warn_removal_running(self, task_id: str, reason: str) -> None:
        self.output.warning(
            f"Removal task {task_id} {reason}; it is still running on the appliance. "
            "Re-check the samples before assuming they survived."
        )

    @with_client(default=None)
    def batch_reanalyze_samples(
        self, hashes: list[str], *, titanium_core: bool = True, titanium_cloud: bool = True
    ) -> list[dict[str, Any]] | None:
        """Submit a batch for reanalysis and hand back what the appliance answered.

        The answer as it came, because it is what ``-o json`` emits and
        what the table draws; :class:`ReanalysisBatch` is how a caller reads
        what the appliance made of it. ``None`` stays what it is everywhere
        else: a submission that never landed.
        """
        batch = normalized_batch(hashes, A1000_HASH_TYPES, self.output.error)
        if batch is None:
            return None
        response = self.client.reanalyze_samples_v2(
            batch, titanium_core=titanium_core, titanium_cloud=titanium_cloud
        )
        result_data = json_on(response)
        if isinstance(result_data, list):
            return result_data
        # An envelope we cannot read is a failed submission, the line every
        # A1000 listing draws: a body carrying no per-sample answers must
        # not be counted as one sample accepted out of a batch of five.
        results = list_from_envelope(result_data, "results")
        if results is None:
            self.output.error("Reanalysis response carried no results list")
        return results


# ---------- Module-level helpers ----------


def _first_status(page: dict[str, Any]) -> dict[str, Any] | None:
    """The one per-sample entry in an analysis-status body, if it holds one.

    The endpoint is a bulk one and we ask about a single hash, so the first
    entry is the answer. ``None`` says the array is empty or was not an
    array, which is what a 200 carries for a hash the appliance has not
    registered yet.
    """
    entries = dict_rows(list_section(page, "results") or [])
    return entries[0] if entries else None


def _extract_hash_from_upload(upload_result: Any) -> str | None:
    """Pull the SHA256 (preferred) or SHA1 from an upload response.

    A1000 answers a submission with {"code": 201, "detail": {"sha1": ...}}.
    The nested {"rl": {"sample": ...}} form comes from other SDK upload
    endpoints, so both are accepted.
    """
    if not isinstance(upload_result, dict):
        return None

    for sample in (
        upload_result.get("detail"),
        # ``mapping`` and not ``(x or {})``: the latter raises
        # AttributeError for an ``rl`` stated as a list or a string, which
        # ``with_client`` would swallow into a failed upload the appliance
        # had in fact accepted.
        mapping(upload_result.get("rl")).get("sample"),
    ):
        if isinstance(sample, dict):
            digest = sample.get("sha256") or sample.get("sha1")
            if digest:
                return str(digest)
    return None
