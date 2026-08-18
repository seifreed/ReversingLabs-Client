"""Readers for the response shapes the ReversingLabs APIs return.

A1000 v2, Advanced Search v3 and TitaniumCloud disagree about where a
verdict and a threat name live, and about their spelling and case. These
helpers know every shape once, so the renderers and the SARIF exporter do
not each carry their own copy of the fallback chain.

Nothing here escapes or shortens a value: what is handed on is the
appliance's own bytes, because these strings reach a terminal, a SARIF log
and ``-o json`` alike, and the layer that draws one knows what "unsafe"
means for it. Where a reader below calls :func:`~rl_cli.text.sanitize` or
:func:`~rl_cli.text.says_nothing`, it is normalising a value in order to
*test* it and hands on the untouched original — so do not delete such a
call to "restore consistency", and do not let its result become the value
returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, overload

from rl_cli.models.shapes import count, dict_rows, list_section, mapping, stated_count
from rl_cli.text import sanitize, says_nothing

# YARA match entries report the verdict as a TitaniumCore numeric code.
_CLASSIFICATION_CODES = {0: "unknown", 1: "goodware", 2: "suspicious", 3: "malicious"}

# Every verdict spelling the APIs use, folded onto the four severities the
# renderers and the SARIF exporter actually distinguish. Matching is exact:
# "unknown" contains "known", so folding by substring would grade a
# no-opinion verdict as clean.
_SEVERITIES = {
    "malicious": "malicious",
    "suspicious": "suspicious",
    "clean": "known",
    "goodware": "known",
    "known": "known",
}

# Every spelling that names a verdict at all, which is what ``status`` is
# read against below. "unknown" is one of them: it is a stated no-opinion,
# not a processing state.
_VERDICT_WORDS = frozenset(_SEVERITIES) | {"unknown"}


# Worst first: how two verdicts that disagree are compared, and how a
# listing of them is ordered. A stated "unknown" outranks a stated "known"
# because "we have no opinion" is the less reassuring of the two.
#
# The one table. ``formatters/severity.py`` keeps the colours and reads its
# ranks from here; it may not own them, because the collision rule below
# compares two severities and a payload reader importing the renderers
# would drag Rich into `-o json`.
_SEVERITY_RANKS = {"malicious": 0, "suspicious": 1, "unknown": 2, "known": 3}

# The singular spellings of a sample's own name, least specific last, read
# after ``proposed_filename`` and ``file_names``. ``full_path`` names an
# extracted file inside the container it was carved from.
_NAME_KEYS = ("file_name", "filename", "full_path")

# Every key that *spells* a verdict, a threat name or a threat level. They
# move as one block when two records disagree: a verdict taken from one
# side beside a family name taken from the other is an incoherent record —
# "goodware", "Win32.Trojan.Conti".
#
# ``ticore`` and ``analysis`` must stay out of this set. They are the
# TitaniumCore and RLDATA *documents*, which state a verdict among an
# indicator list, a signature list and an ATT&CK matrix, so moving the
# block would take a whole document with it. ``SampleVerdict.of`` reads a
# buried verdict only after the stated one that wins the collision, so a
# document can stay in the record without overruling the winner.
_VERDICT_KEYS = frozenset(
    {
        "classification",
        "classification_result",
        "status",
        "threat_family",
        "threat_level",
        "threat_name",
        "threat_status",
        "untokenized_threat_name",
    }
)

# Every key a verdict can be *reached* through, which is the wider set: the
# two documents bury one. A half whose only verdict sits inside its
# document still states one, so the merge below must weigh it rather than
# skip the comparison.
_VERDICT_BEARING_KEYS = _VERDICT_KEYS | {"analysis", "ticore"}


def severity_rank(severity: str) -> int:
    """Where ``severity`` sorts, worst first.

    A severity this table has never heard of grades as no opinion rather
    than as clean.
    """
    return _SEVERITY_RANKS.get(severity, _SEVERITY_RANKS["unknown"])


@dataclass(frozen=True, slots=True)
class SearchPage:
    """One page of an Advanced Search answer, and whether it was the last.

    The flag travels with the page so that a caller cannot report one page
    of thousands as the whole answer. The SDK's own aggregator loops on
    ``more_pages`` and ``next_page`` (``ReversingLabs/SDK/a1000.py``).
    """

    entries: list[dict[str, Any]]
    more_pages: bool


def search_page(body: Any) -> SearchPage | None:
    """The Advanced Search page inside ``body``, or ``None`` if unreadable.

    Both Advanced Search endpoints — TitaniumCloud's and the A1000's —
    answer ``rl.web_search_api``, so the shape is read once here.

    ``None`` means the envelope was not there: an answer we cannot parse is
    a failed search, not a search that matched nothing, and the two exit
    differently.

    Both spellings of "there is more" count, because the appliance is not
    obliged to agree with itself: either ``more_pages`` or ``next_page``
    stated means this page is not everything.
    """
    envelope = mapping(mapping(body).get("rl")).get("web_search_api")
    if not isinstance(envelope, dict):
        return None
    return SearchPage(
        entries=list_section(envelope, "entries") or [],
        more_pages=bool(envelope.get("more_pages") or envelope.get("next_page")),
    )


def _merged_record(entry: dict[str, Any], carried: dict[str, Any]) -> dict[str, Any]:
    """One record out of an entry and the record it carries under ``sample``.

    The carried record wins key by key: it is the more specific of the two,
    and the entry around it exists to name the file, not to describe it.

    The collision rule, when both sides state a verdict of their own: **the
    more severe verdict wins, and it brings its whole block with it** — the
    threat name, the threat level and every spelling of the verdict move
    together, so that no reader can pair one side's verdict with the
    other's family name. Severity decides it, not the merge order; the
    carried record only breaks ties.

    Which block moves is :data:`_VERDICT_KEYS` — the verdict's *spellings*
    alone — whenever the winner spells its verdict at all, so the loser's
    TitaniumCore and RLDATA documents survive the collision. A winner that
    states its verdict only inside its own document moves the wider
    :data:`_VERDICT_BEARING_KEYS` instead: there the document *is* the
    winning verdict, and moving only the spellings would leave the loser's
    document for ``_titaniumcore_classification`` to read.
    """
    merged = entry | carried
    if not (_VERDICT_BEARING_KEYS & entry.keys() and _VERDICT_BEARING_KEYS & carried.keys()):
        return merged

    stated, inner = SampleVerdict.of(entry), SampleVerdict.of(carried)
    if not (stated.classification and inner.classification):
        return merged

    worse = entry if severity_rank(stated.severity) < severity_rank(inner.severity) else carried
    moving = _VERDICT_KEYS if _VERDICT_KEYS & worse.keys() else _VERDICT_BEARING_KEYS
    return {key: value for key, value in merged.items() if key not in moving} | {
        key: value for key, value in worse.items() if key in moving
    }


@overload
def unwrap_envelope(payload: dict[str, Any]) -> dict[str, Any]: ...


@overload
def unwrap_envelope(payload: Any) -> Any: ...


def unwrap_envelope(payload: Any) -> Any:
    """Return the sample record carried by a response envelope.

    Handles the paginated ``{"results": [...]}`` form (first record), the
    A1000 extracted-file entry that carries its record under ``sample``, and
    TitaniumCloud's ``{"rl": {"malware_presence": {...}}}`` nesting; a
    payload that is already a record comes back unchanged.

    This must stay the one reader of the extracted-file shape — the terminal
    table goes through it too — or the screen and the SARIF log disagree
    about every verdict the other's half of the shape carried.

    Anything that is not a dict is handed straight back, and the overloads
    say so: a dict in is a dict out. Keep the parameter ``Any``, because
    ``sarif._iter_findings`` maps this over the items of a JSON array, which
    may hold a bare string — that entry has to survive as far as
    ``_result_from`` rather than being dropped out of the log.
    """
    if not isinstance(payload, dict):
        return payload

    results = dict_rows(list_section(payload, "results") or [])
    if results:
        return results[0]

    # An extracted-file entry names the file at the top level and states
    # its metadata — usually including the verdict — one level down under
    # "sample", so both halves are kept.
    #
    # The isinstance checks read "is the record there at all", not "is it
    # non-empty": an empty ``malware_presence`` is TitaniumCloud's answer
    # for a hash it has never seen, which is an answer and not a failure to
    # read one.
    carried = payload.get("sample")
    if isinstance(carried, dict):
        return _merged_record(
            {key: value for key, value in payload.items() if key != "sample"}, carried
        )

    nested = payload.get("rl")
    if isinstance(nested, dict):
        for key in ("malware_presence", "sample"):
            record = nested.get(key)
            if isinstance(record, dict):
                return record
        # Every TitaniumCloud response nests under "rl"; the URL report
        # (TCA-0403) spells none of the record keys above, so an
        # unrecognised "rl" body has to come back unwrapped or a URL
        # verdict can never be read at all.
        return nested
    return payload


def ticore_record(payload: dict[str, Any]) -> dict[str, Any]:
    """A TitaniumCore document as the flat record the readers here read.

    ``/samples/{hash}/ticore/`` answers the document itself and an A1000
    detailed report carries it under ``ticore``; both are accepted, so a
    renderer does not depend on which endpoint produced the payload. The
    document goes back under that key, where
    :func:`_titaniumcore_classification` reads the verdict it buries.
    """
    document = unwrap_envelope(payload)
    nested = document.get("ticore")
    if isinstance(nested, dict):
        document = nested

    file_meta = mapping(mapping(document.get("info")).get("file"))
    classification = mapping(document.get("classification"))
    return {
        "md5": document.get("md5"),
        "sha1": document.get("sha1"),
        "sha256": document.get("sha256"),
        "file_type": file_meta.get("file_type"),
        "file_size": file_meta.get("size"),
        "proposed_filename": file_meta.get("proposed_filename") or file_meta.get("name"),
        "classification": classification.get("classification"),
        "classification_result": classification.get("result"),
        "ticore": document,
        "tags": {"ticore": list_section(document, "tags") or []},
    }


def _decoded(value: Any) -> Any:
    """A TitaniumCore numeric verdict as its name; anything else unchanged."""
    if isinstance(value, int) and not isinstance(value, bool):
        return _CLASSIFICATION_CODES.get(value)
    return value


def _stated_verdict(value: Any) -> Any:
    """``status`` when it names a verdict, and nothing when it names a state.

    On TitaniumCloud ``malware_presence.status`` is the verdict itself —
    MALICIOUS, SUSPICIOUS, KNOWN, UNKNOWN. On the A1000 the same key is the
    *processing* state: "processed", "accepted", "queued", "not_found".

    Recognising verdicts rather than listing states is deliberate: an
    unlisted processing state would go on grading samples, and a new state
    is far likelier than a new verdict.
    """
    return value if str(value).strip().lower() in _VERDICT_WORDS else None


def _titaniumcore_classification(payload: dict[str, Any]) -> dict[str, Any]:
    """The TitaniumCore verdict block a record buries, wherever it sits.

    Two shapes hide the same block, and neither states it at the record's
    top level. TCA-0104 (RLDATA) answers ``rl.sample`` with the verdict down
    at ``analysis.entries[0].tc_report.classification`` — the SDK reads the
    file type through the same path (ticloud.py:671). The A1000 carries the
    whole TitaniumCore document under ``ticore``.
    """
    block = mapping(mapping(payload.get("ticore")).get("classification"))
    if block:
        return block

    entries = dict_rows(list_section(mapping(payload.get("analysis")), "entries") or [])
    entry = entries[0] if entries else {}
    return mapping(mapping(entry.get("tc_report")).get("classification"))


@dataclass(frozen=True, slots=True)
class SampleVerdict:
    """Everything one payload says about a sample's verdict, read in one walk.

    ``of`` visits each key once and hands all four answers back together, so
    that a renderer needing the colour, the text and a sort key costs one
    walk rather than one per question.
    """

    classification: str | None
    severity: str
    threat_name: str | None
    threat_level: Any

    @classmethod
    def of(cls, payload: dict[str, Any]) -> SampleVerdict:
        """Read ``payload``'s verdict, touching each key it holds once."""
        stated = payload.get("classification")
        # The nested form carries both the verdict and the family fragment,
        # so it is read once here and consulted twice below.
        nested = stated if isinstance(stated, dict) else None
        buried = _titaniumcore_classification(payload)

        # An explicit threat_status outranks status: status is the verdict
        # on one API and the processing state on the other, and a field
        # that means two things cannot overrule one that means one.
        verdict = _decoded(nested.get("classification") if nested else stated) or (
            payload.get("threat_status")
            or _stated_verdict(payload.get("status"))
            or _decoded(buried.get("classification"))
        )
        classification = str(verdict) if verdict else None

        name = (
            payload.get("threat_name")
            or payload.get("classification_result")
            or payload.get("untokenized_threat_name")
            or payload.get("threat_family")
            or (nested.get("result") if nested else None)
            or buried.get("result")
            or (nested.get("family_name") if nested else None)
        )

        # Explicitly against None: threat_level 0 is the lowest level, not a
        # missing one, and "or" reads it as absent.
        level = payload.get("threat_level")
        if level is None and nested:
            level = nested.get("factor")
        if level is None:
            level = buried.get("factor")

        return cls(
            classification=classification,
            severity=_SEVERITIES.get((classification or "").strip().lower(), "unknown"),
            threat_name=str(name) if name else None,
            threat_level=level,
        )


def _stated_text(value: Any) -> str | None:
    """A field the payload states, as text; ``None`` when it states nothing."""
    return str(value) if value else None


@dataclass(frozen=True, slots=True)
class SampleFacts:
    """What one payload says a sample is called, hashed, typed and sized.

    Every spelling in one walk, for the same reason ``SampleVerdict`` reads
    a verdict in one: five renderers and the SARIF exporter draw from this,
    and a fallback chain per caller drifts.
    """

    name: str | None
    file_type: str | None
    size: Any
    md5: str | None
    sha1: str | None
    sha256: str | None
    aliases: tuple[str, ...] = ()

    @property
    def digest(self) -> str | None:
        """The strongest digest stated, which is how a nameless sample is named.

        Strongest first, because a listing shows one hash column and falling
        back to MD5 while a SHA256 sits in the record would name the same
        sample two ways in two tables.
        """
        return self.sha256 or self.sha1 or self.md5

    @classmethod
    def of(cls, payload: dict[str, Any]) -> SampleFacts:
        """Read what ``payload`` states about the sample's identity."""
        aliases = tuple(str(alias) for alias in list_section(payload, "aliases") or [] if alias)
        # A search entry states every name the appliance saw the sample
        # under, and no singular key at all; the first of them is the one
        # the panel and the log name it by.
        stated_names = [name for name in list_section(payload, "file_names") or [] if name]
        stated = payload.get("proposed_filename") or (stated_names[0] if stated_names else None)
        for key in _NAME_KEYS:
            stated = stated or payload.get(key)

        # Explicitly against None: a zero-byte file is a size, and ``or``
        # would read it as absent and fall through to the other service's
        # spelling.
        size = payload.get("file_size")
        if size is None:
            size = payload.get("sample_size")

        return cls(
            # An appliance that knows the sample only by its aliases still
            # names it.
            name=_stated_text(stated) or (aliases[0] if aliases else None),
            # File Analysis (TCA-0104) spells the type ``sample_type``, not
            # ``file_type``.
            file_type=_stated_text(payload.get("file_type") or payload.get("sample_type")),
            size=size,
            md5=_stated_text(payload.get("md5")),
            sha1=_stated_text(payload.get("sha1")),
            sha256=_stated_text(payload.get("sha256")),
            aliases=aliases,
        )


# Every spelling of "how many engines flagged it": the A1000 says
# scanner_match_count and TitaniumCloud says scanner_match.
_MATCH_COUNT_KEYS = ("scanner_match_count", "scanner_match")

# Above this share of the scanners the ratio grades as a consensus rather
# than as a handful of noisy signatures. See
# ``ScannerConsensus.consensus_severity``.
_MAJORITY_OF_SCANNERS = 0.5


def _stated_count(source: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    """The first of ``keys`` the mapping states as a number, as an integer.

    ``None`` means the mapping stated none of them, which is not the same
    fact as stating zero.

    A key stated as something that is not a finite number states no number
    either, which is why this reads through ``stated_count`` rather than the
    lenient ``count``: ``""`` is an ordinary API idiom for an unset numeric
    field, and reading it as zero invents a count the record never gave. A
    stated zero is a number and stays one.
    """
    for key in keys:
        stated = stated_count(source.get(key))
        if stated is not None:
            return stated
    return None


@dataclass(frozen=True, slots=True)
class ScannerDetection:
    """One AV engine that flagged a sample, and what it called it.

    ``nextgen`` tells an ML confidence verdict apart from a signature hit;
    the layer that draws it decides how to say so.

    ``name`` is ``None`` for an engine the array did not name: what a report
    puts in that cell is a word the report chooses, and nothing here states
    a value the payload did not.
    """

    name: str | None
    result: str
    nextgen: bool


def _flagged_anything(result: str) -> bool:
    """Whether an engine's result names anything an analyst could read.

    An engine that answered with nothing readable flagged nothing, however
    many bytes it took to say so.

    What "nothing" covers is :func:`~rl_cli.text.says_nothing`'s to say,
    and deliberately not ``sanitize``'s: that names the characters which
    are *unsafe* to print, a smaller set than the ones that are
    *invisible*, so a zero-width space or a BOM would count as a hit.
    """
    return not says_nothing(result)


def _scanner_section(
    scanners: dict[str, Any], section: str, *, nextgen: bool
) -> tuple[int, tuple[ScannerDetection, ...]]:
    """How many engines one scanner array holds, and which of them flagged it.

    A scanner entry states neither field reliably: ``result`` arrives as a
    number and ``name`` is missing outright, one entry of many, so an engine
    that stated nothing readable is counted but is not a detection — see
    :func:`_flagged_anything` for what "nothing" covers.
    """
    entries = dict_rows(list_section(scanners, section) or [])
    detections = []
    for entry in entries:
        result = str(entry.get("result") or "")
        if _flagged_anything(result):
            detections.append(ScannerDetection(_stated_text(entry.get("name")), result, nextgen))
    return len(entries), tuple(detections)


@dataclass(frozen=True, slots=True)
class ScannerConsensus:
    """How many AV engines flagged a sample, and which of them did.

    Either half of the ratio is ``None`` when nothing states it and no
    scanner array was there to count: a number we were not told is not a
    zero.

    ``nextgen_*`` is counted apart from the ratio because a next-gen engine
    answers with an ML confidence verdict rather than a signature hit;
    folding those in overstates the ratio.
    """

    detected: int | None
    total: int | None
    percent: Any
    nextgen_detected: int
    nextgen_total: int
    detections: tuple[ScannerDetection, ...]

    @property
    def consensus_severity(self) -> str:
        """How the ratio grades, as one of the severities the renderers paint.

        Whether thirteen of seventeen engines is a consensus or a handful of
        noisy signatures is a judgement about the data, so it is made once
        here rather than inside whichever report happens to draw the ratio.

        Both halves must be known before it grades either way: ``known``
        needs a stated zero and ``malicious`` needs a stated total to be a
        majority of, so an unknown half grades in the middle. A listed
        detection does too, whatever the summary beside it claims — the
        reassuring half of a self-contradicting report is the one an analyst
        reads first.
        """
        if (
            self.detected is not None
            and self.total is not None
            and self.detected > self.total * _MAJORITY_OF_SCANNERS
        ):
            return "malicious"
        if self.detected == 0 and not self.detections:
            return "known"
        return "suspicious"

    @property
    def unreported(self) -> bool:
        """Whether no engine ran at all, so a panel has nothing to head.

        A signature ratio of ``0/0`` beside next-gen verdicts is a record
        whose only array was the next-gen one, and that panel still has
        those verdicts to draw.
        """
        return not self.nextgen_total and (
            self.total == 0 or (self.total is None and self.detected is None)
        )

    @classmethod
    def of(cls, payload: dict[str, Any]) -> ScannerConsensus:
        """Count ``payload``'s scanner arrays and reconcile the stated ratio.

        The ratio counts signature engines alone, so that one record reports
        one ratio whichever response shape carried it: the summary the
        appliance states excludes next-gen verdicts, and counting the arrays
        differently would answer 5/15 where the summary says 2/10.

        Where the stated ratio and the count disagree the appliance's own
        ratio wins — it is the one its UI shows — but it never falls under
        the count of signature detections listed beneath it, which that
        summary is the count of.
        """
        scanners = payload.get("av_scanners")
        if isinstance(scanners, list):
            scanners = scanners[0] if scanners else {}
        arrays = mapping(scanners)

        signature_total, signature_hits = _scanner_section(
            arrays, "regular_scanners", nextgen=False
        )
        nextgen_total, nextgen_hits = _scanner_section(arrays, "nextgen_scanners", nextgen=True)
        engines = signature_total + nextgen_total

        # The A1000 states its ratio in av_scanners_summary; a TitaniumCloud
        # malware_presence record states the same three fields on itself.
        summary = mapping(payload.get("av_scanners_summary")) or payload
        stated_total = _stated_count(summary, ("scanner_count",))
        stated_detected = _stated_count(summary, _MATCH_COUNT_KEYS)

        detected = max(
            len(signature_hits) if stated_detected is None else stated_detected,
            len(signature_hits),
        )
        total = max(signature_total if stated_total is None else stated_total, detected)

        # Each half is reported only where something states it: a match
        # count is not an engine count, and the renderers draw the half they
        # were given rather than a half invented here.
        return cls(
            detected=None if stated_detected is None and not engines else detected,
            total=None if stated_total is None and not engines else total,
            percent=summary.get("scanner_percent"),
            nextgen_detected=len(nextgen_hits),
            nextgen_total=nextgen_total,
            detections=signature_hits + nextgen_hits,
        )


# An HTTP status at or above this is the appliance rejecting a submission,
# not queueing it. It is stated per engine under ``analysis``, and — for a
# submission the appliance turned down before any engine saw it — on the
# entry itself, which is the shape the upload reader documents:
# ``{"code": 201, "detail": {...}}``.
_REJECTED_FROM = 400

# The spellings of a ``status`` field that mean the sample was taken.
# Anything else stated there is the appliance naming a refusal — "Sample
# not found", "Not authorized" — so an unrecognised word counts as one
# rather than as an acceptance.
_QUEUED_STATUSES = frozenset(
    {"queued", "submitted", "accepted", "pending", "processing", "in progress", "started"}
)


def _stated_refusal(entry: dict[str, Any]) -> str | None:
    """The refusal the entry names for itself, or ``None`` if it names none.

    Two fields can carry it. ``code`` is the entry's own HTTP status, the
    shape a submission is answered with (``{"code": 201, "detail": {...}}``),
    and a rejection carries no ``analysis`` under it to read. ``status`` is
    the appliance stating the outcome in words.
    """
    code = count(entry.get("code"))
    if code >= _REJECTED_FROM:
        return f"{code}: {entry.get('message') or 'rejected'}"
    stated = entry.get("status")
    if stated in (None, ""):
        return None
    # ``sanitize`` normalises the word for the membership test only -- a
    # stray control character in "queued" must not make it a refusal.
    if sanitize(stated).strip().casefold().replace("_", " ").replace("-", " ") in _QUEUED_STATUSES:
        return None
    return str(stated)


@dataclass(frozen=True, slots=True)
class ReanalysisOutcome:
    """What the appliance made of one reanalysis request.

    ``accepted`` is what a caller's headline count and exit status read; the
    reasons beside it are what a renderer draws over that count. Both come
    out of one reading, so that a command cannot announce three successes
    above a table refusing all three.

    ``stated`` is the appliance's own word for an outcome it *did* take, so
    it is ``None`` whenever ``refusal`` is set — a refusal is reported once,
    as the refusal.

    ``refusal`` carries the appliance turning a submission down and nothing
    else. An entry this module could not read states no refusal: it is
    ``accepted=False`` with no reason, because attributing our parse failure
    to the appliance renders "Refused:" over words it never said.
    """

    accepted: bool
    refusal: str | None = None
    stated: str | None = None
    queued: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()

    @classmethod
    def of(cls, entry: Any) -> ReanalysisOutcome:
        """Read one entry of a batch-reanalyze answer.

        An engine answers with a code and a message as well as its name, so
        its code has to be read too or a 404 "Sample not found" counts as
        queued. An entry that refuses nothing was taken, which is the
        "Submitted" a renderer draws for it.

        ``entry`` is ``Any`` because the caller walks the answer as it came,
        and something that is not a record cannot be read as an acceptance.
        A renderer never sees one — ``dict_rows`` drops it before the table
        — but the count walks every entry there is.
        """
        if not isinstance(entry, dict):
            return cls(accepted=False)

        queued: list[str] = []
        failed: list[str] = []
        # ``analysis`` is only iterable when the appliance stated it as the
        # list of engines; a scalar there would raise out of the count.
        for engine in dict_rows(list_section(entry, "analysis") or []):
            name = str(engine.get("name") or "engine")
            code = count(engine.get("code"))
            if code >= _REJECTED_FROM:
                failed.append(f"{name} ({code}): {engine.get('message') or 'rejected'}")
            else:
                queued.append(name)

        refusal = _stated_refusal(entry)
        stated = entry.get("status")
        return cls(
            accepted=refusal is None and (bool(queued) or not failed),
            refusal=refusal,
            stated=None if refusal is not None or stated in (None, "") else str(stated),
            queued=tuple(queued),
            failed=tuple(failed),
        )
