"""File reputation, analysis, search and sample download."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

from ReversingLabs.SDK import ticloud

from rl_cli.config import Settings
from rl_cli.models.payload import unwrap_envelope
from rl_cli.models.validators import HashType, validate_hash
from rl_cli.services.decorators import safe_call
from rl_cli.services.hashes import normalized_batch
from rl_cli.services.http import json_on, succeeded
from rl_cli.services.partial_answers import cleared_per_call
from rl_cli.services.protocols import Notifier
from rl_cli.services.titanium_cloud.api import TitaniumCloudApi
from rl_cli.services.walks import fetch_search_page, numbered_search
from rl_cli.storage.files import write_private_bytes

_MAX_RECORDS_PER_PAGE = 10000

# How many pages one aggregated search may fetch, whatever it was asked
# for. Every page is a metered request against a paid API, and neither
# ``--limit`` nor an endpoint that answers "no entries, more pages follow"
# bounds what a search may spend, so the bound is the walk's own.
#
# Ten because a page holds 10000 records, so the budget is 100000 of them:
# larger than any answer this CLI renders or an analyst reads in one go,
# and small enough that the worst a typo costs is ten requests. An answer
# the budget cut short says so through ``pages_left_unfetched``, the same
# way one cut short at a page boundary does.
_MAX_SEARCH_PAGES = 10

# Every TitaniumCloud hash endpoint this service wraps takes these three
# and no others (SDK ticloud.py:237, 430, 492).
_SUPPORTED_HASH_TYPES = (HashType.MD5, HashType.SHA1, HashType.SHA256)


def _records_per_page(limit: int) -> int:
    """``limit`` as a page size this endpoint takes.

    Only the bottom is bounded: the SDK answers a page size of 0 with
    ``WrongInputError``, which shows the user its internal guard text where
    they asked for a search. Nothing is cut off the top, because a limit
    above one page's worth is walked rather than trimmed to fit.
    """
    return max(1, limit)


@cleared_per_call("pages_left_unfetched")
class TitaniumCloudService(TitaniumCloudApi):
    """Everything TitaniumCloud is asked about a file: a hash, or a sample.

    Reputation and AV scanners, analysis, Advanced Search, upload and
    sample download, over six of the ten ``ticloud`` API families. The other
    four — domain, IP, URL and URI-index intelligence — are
    :class:`~rl_cli.services.titanium_cloud.network.TitaniumCloudNetworkService`,
    a peer of this class and not something it holds.

    Neither builds the other's SDK handles, which is what keeps an
    ``ip-files`` lookup from carrying credentials for six families it never
    calls.
    """

    def __init__(self, settings: Settings, output: Notifier | None = None):
        super().__init__(settings, output)
        # Whether the last Advanced Search left hits behind, for a caller
        # that can offer to fetch them. Both search paths measure it from
        # the last page they read — its envelope's ``more_pages`` — and the
        # walk adds what it fetched and had to trim off ``limit``, because
        # a record cut off the end of the answer is as missing as a page
        # never asked for. A fact about the answer, not advice about what
        # to do with it.
        #
        # It speaks for the call that just finished and for no earlier one,
        # which is why the class decorator clears it at every entry point
        # and only :meth:`_one_search_page` and :meth:`_aggregated_search`
        # set it.
        self.pages_left_unfetched = False

    # Each ticloud API class talks to a different endpoint family. Cache
    # them so repeated calls reuse the same handle.
    @cached_property
    def _file_reputation(self) -> Any:
        return self._api(ticloud.FileReputation)

    @cached_property
    def _av_scanners(self) -> Any:
        return self._api(ticloud.AVScanners)

    @cached_property
    def _file_upload(self) -> Any:
        return self._api(ticloud.FileUpload)

    @cached_property
    def _file_analysis(self) -> Any:
        return self._api(ticloud.FileAnalysis)

    @cached_property
    def _advanced_search(self) -> Any:
        return self._api(ticloud.AdvancedSearch)

    @cached_property
    def _file_download(self) -> Any:
        return self._api(ticloud.FileDownload)

    def supported_hash(self, hash_value: str) -> str | None:
        """The hash as the endpoints want it, or ``None`` after saying why.

        ``validate_hash`` also accepts SHA512, which the A1000 takes and
        TitaniumCloud does not, so the refusal is spelled out here rather
        than left to the SDK's internal guard text. Callers must send this
        normalised value on: it is what was judged, and a hash pasted with
        whitespace or in uppercase is not.

        Public because ``ticloud download`` has to know the answer before it
        creates a directory or announces live malware.
        """
        normalised = hash_value.strip().lower()
        if validate_hash(normalised) not in _SUPPORTED_HASH_TYPES:
            self.output.error(
                f"Invalid hash format: {hash_value} (TitaniumCloud accepts MD5, SHA1 or SHA256)"
            )
            return None
        return normalised

    def _supported_batch(self, hashes: list[str]) -> list[str] | None:
        """A whole batch as the bulk endpoints want it, or ``None`` after saying why.

        Two rules, both the endpoint's. Each entry has to be one of the
        three types these APIs take, named by its position — the shared
        :func:`normalized_batch`. The other is this family's alone: the
        batch has to be all of one type, because the bulk query sends a
        single ``hash_type`` taken from the first entry (SDK
        ticloud.py:258), so a mixed list is asked for under the wrong type
        rather than refused.
        """
        batch = normalized_batch(
            hashes,
            _SUPPORTED_HASH_TYPES,
            lambda reason: self.output.error(
                f"{reason} (TitaniumCloud accepts MD5, SHA1 or SHA256)"
            ),
        )
        if batch is None:
            return None
        first_type: HashType | None = None
        for position, hash_value in enumerate(batch, start=1):
            hash_type = validate_hash(hash_value)
            if first_type is None:
                first_type = hash_type
            elif hash_type is not first_type:
                self.output.error(
                    f"Mixed hash types in one batch: entry {position} is "
                    f"{hash_type.value if hash_type else 'unknown'}, entry 1 is "
                    f"{first_type.value}. The bulk endpoint queries one hash type at a time."
                )
                return None
        return batch

    @safe_call(default=None)
    def get_file_reputation(self, hash_value: str) -> dict[str, Any] | None:
        hash_input = self.supported_hash(hash_value)
        if not hash_input:
            return None
        response = self._file_reputation.get_file_reputation(
            hash_input=hash_input, extended_results=True, show_hashes_in_results=True
        )
        return json_on(response)

    @safe_call(default=None)
    def get_bulk_file_reputation(self, hashes: list[str]) -> list[dict[str, Any]] | None:
        """Grade a whole batch in the one request the bulk endpoint is for.

        ``FileReputation.get_file_reputation`` takes a list as readily as a
        string and posts it to ``malware_presence/bulk_query`` (SDK
        ticloud.py:252-267); asking about 500 hashes one at a time is 500
        metered round-trips for an answer the API will give in one.

        The two endpoints do not answer in the same shape: the single
        query nests one record under ``rl.malware_presence`` and the bulk
        query lists its records under ``rl.entries``, which is why this
        wrapper returns a list where its single-hash sibling returns the
        envelope. :meth:`get_reputation_records` reconciles them for a
        caller that just wants the records.

        ``entries`` is the one envelope key in this module with no
        corroboration in the vendored SDK, so it is read as ``required``: a
        missing key is far more likely to mean this is reading the wrong key
        than that the API holds nothing for any of the hashes.
        """
        batch = self._supported_batch(hashes)
        if batch is None:
            return None
        response = self._file_reputation.get_file_reputation(
            hash_input=batch, extended_results=True, show_hashes_in_results=True
        )
        return self._rl_list(json_on(response), "entries", required=True)

    def get_reputation_records(self, hashes: list[str]) -> list[dict[str, Any]] | None:
        """Grade ``hashes`` and answer one flat record per hash, however many there are.

        One hash goes to the single endpoint and several go to the bulk one
        — a batch is one metered request rather than N — but the two answer
        in different shapes, so handing those on unchanged would answer the
        same question at two fidelities: a verdict for one hash, a table of
        whatever keys the bulk entries carried for two.

        Unwrapping here is the same ``unwrap_envelope`` the rich renderer
        and the SARIF exporter apply to a single answer, done once so that
        callers grade the sample rather than the wrapper it arrived in.
        ``get_file_reputation`` keeps returning the whole envelope: it is the
        documented library call for one hash, and the response body is what
        it promises.
        """
        if len(hashes) == 1:
            payload = self.get_file_reputation(hashes[0])
            if not payload:
                return None
            record = unwrap_envelope(payload)
            # An envelope with nothing in it is an answer we could not
            # read, not a sample with no reputation: the endpoint grades an
            # unseen hash "UNKNOWN" rather than answering empty.
            return [record] if record else None
        return self.get_bulk_file_reputation(hashes)

    @safe_call(default=None)
    def get_av_scanners(self, hash_value: str) -> dict[str, Any] | None:
        hash_input = self.supported_hash(hash_value)
        if not hash_input:
            return None
        response = self._av_scanners.get_scan_results(
            hash_input=hash_input, historical_results=False
        )
        return json_on(response)

    @safe_call(default=None)
    def upload_file(self, file_path: Path) -> dict[str, Any] | None:
        if not file_path.exists():
            self.output.error(f"File not found: {file_path}")
            return None
        response = self._file_upload.upload_sample_from_path(
            file_path=str(file_path), sample_name=file_path.name
        )
        # The SPEX upload interface speaks XML and answers an accepted
        # upload with an empty body (SDK ticloud.py:2339, 2469-2499), so the
        # status is the whole of the answer and this must not be read as
        # JSON.
        succeeded(response)
        return {"file_name": file_path.name, "status": "queued for analysis"}

    @safe_call(default=None)
    def search_samples(self, query: str, limit: int = 100) -> list[dict[str, Any]] | None:
        # If the input is itself a hash, return the file reputation wrapped
        # in a list so callers always receive list-shaped data — and
        # unwrapped, because every other element this method returns is a
        # flat record. An ``{"rl": {"malware_presence": ...}}`` envelope
        # here grades a known-malicious sample "none" in SARIF and renders a
        # one-column "Rl" table under -o rich.
        if validate_hash(query):
            self.output.info("Detected hash input, fetching file reputation instead...")
            result = self.get_file_reputation(query)
            return [unwrap_envelope(result)] if result else None

        # A page holds at most 10000 records, so anything larger has to be
        # walked page by page rather than clamped to one page's worth.
        if limit > _MAX_RECORDS_PER_PAGE:
            return self._aggregated_search(query, limit)
        return self._one_search_page(query, limit)

    def _aggregated_search(self, query: str, limit: int) -> list[dict[str, Any]] | None:
        """Walk the pages up to ``limit``, within a budget, saying what was left.

        The walk itself, and what each of its stops is for, is
        :func:`~rl_cli.services.walks.numbered_walk`; every page is asked
        for at the largest size the endpoint serves, because every one of
        them is a metered request.

        With :meth:`_one_search_page`, one of the two places
        :attr:`pages_left_unfetched` is set, and it is set after the last
        request.
        """
        walked = numbered_search(
            self._advanced_search.search,
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

    def _one_search_page(self, query: str, limit: int) -> list[dict[str, Any]] | None:
        """One Advanced Search page, saying so when the corpus held more.

        The envelope states ``more_pages`` and ``next_page`` beside the
        entries — the SDK's own aggregator loops on exactly those two (SDK
        ticloud.py:1233-1234) — so reading only ``entries`` would turn a
        query matching thousands into a silent "Found 100 samples".

        What is said here is the fact that was measured and nothing about
        what to type next: naming ``--limit`` would make this service
        unusable to a caller that has no such option.
        :attr:`pages_left_unfetched` is how the caller learns of it in order
        to offer its own remedy.
        """
        found = fetch_search_page(
            self._advanced_search.search,
            query,
            1,
            _records_per_page(limit),
            self.output.error,
        )
        if found is None:
            return None
        self.pages_left_unfetched = found.more_pages
        if found.more_pages:
            self.output.warning("More samples matched than one page holds; showing this page only.")
        return found.entries

    @safe_call(default=None)
    def get_file_analysis(self, hash_value: str) -> dict[str, Any] | None:
        hash_input = self.supported_hash(hash_value)
        if not hash_input:
            return None
        response = self._file_analysis.get_analysis_results(hash_input=hash_input)
        return json_on(response)

    # ---------- Sample download ----------
    @safe_call(default=None)
    def get_download_status(self, hash_value: str) -> dict[str, Any] | None:
        """Whether TitaniumCloud holds the sample bytes for ``hash_value``."""
        hash_input = self.supported_hash(hash_value)
        if not hash_input:
            return None
        return json_on(self._file_download.get_download_status(hash_input))

    # ``default=False`` rather than ``None``: this one answers whether the
    # sample was written, and the decorator's contract is that the default
    # is a value of the return type. ``run_step`` reads both the same way.
    @safe_call(default=False)
    def download_sample(self, hash_value: str, output_path: Path) -> bool:
        """Write the sample TitaniumCloud holds for ``hash_value`` to ``output_path``.

        The bytes are live malware, so they go through
        ``write_private_bytes``: 0600 from the moment the file exists, a
        symlink at the destination refused rather than followed, and an
        interrupted download leaving nothing rather than a truncated sample
        indistinguishable from a whole one.

        Not streamed, and not fixable from here: the SDK's request helper
        never passes ``stream=True``.
        """
        hash_input = self.supported_hash(hash_value)
        if not hash_input:
            return False
        response = self._file_download.download_sample(hash_input)
        # The body of a successful download is the sample, not JSON, so
        # ``json_on`` cannot stand in front of this call the way it does for
        # every other wrapper here. ``succeeded`` is the whole failure path
        # instead, and ``safe_call`` reports what it raises.
        succeeded(response)
        write_private_bytes(output_path, response.content)
        return True
