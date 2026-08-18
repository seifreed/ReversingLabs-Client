"""SARIF 2.1.0 export for malware-analysis results.

Converts the JSON payloads the ReversingLabs services return into a
SARIF 2.1.0 log (https://docs.oasis-open.org/sarif/sarif/v2.1.0/) so
CLI output can be ingested by SARIF-aware tooling such as GitHub Code
Scanning. Classification maps to SARIF levels: malicious → ``error``,
suspicious → ``warning``, clean/known/goodware → ``note``, and a verdict
that is absent or unrecognised → ``none``.

Two fields of a result are ours to choose rather than the payload's, and
both are chosen against the payload: the ``ruleId`` is one of the five
fixed ids in :data:`_RULES`, never a value out of the feed, and every
``artifactLocation.uri`` is a percent-encoded ``sample://`` URI that
cannot resolve to a path in the repository being scanned. See
:data:`_RULES` and :func:`_artifact_uri` for why each must stay that way.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

from rl_cli._version import __version__
from rl_cli.models.payload import SampleFacts, SampleVerdict, unwrap_envelope
from rl_cli.text import sanitize

SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = (
    "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json"
)
_INFORMATION_URI = "https://github.com/seifreed/ReversingLabs-Client"

# The scheme every location in this log carries. A sample is not a file in
# the repository being scanned, and saying so in the URI is what keeps it
# from being read as one — see :func:`_artifact_uri`.
_SAMPLE_SCHEME = "sample://"
_UNIDENTIFIED_URI = f"{_SAMPLE_SCHEME}unidentified"

# The SARIF 2.1.0 result level each severity reports as. It is this file's
# vocabulary, not the payload reader's: a severity is what the services
# said about a sample, while "error" and "warning" are what one consumer
# of this log does about it.
_SARIF_LEVELS = {"malicious": "error", "suspicious": "warning", "known": "note"}

# The whole rule catalogue this tool can ever declare: one rule per
# severity, plus the one for a finding we could not read as a record.
#
# A threat name must never become a rule id: §3.27.5 wants a "stable,
# opaque identifier", and a family name fails that three ways over. It is
# attacker-authored — it reaches here verbatim from the vendor feed, tabs,
# newlines and bidi overrides included, and would land in GitHub's rule
# catalogue. It is unbounded: one rule per malware family ever seen. And it
# is not stable: when the vendor renames a family, GitHub closes every alert
# under the old id and opens new ones for the same samples. The family name
# belongs in ``message.text`` and ``properties.threatName`` instead.
_RULES: dict[str, dict[str, Any]] = {
    "rl-cli/malicious": {
        "name": "MaliciousSample",
        "shortDescription": {"text": "ReversingLabs classifies this sample as malicious."},
    },
    "rl-cli/suspicious": {
        "name": "SuspiciousSample",
        "shortDescription": {"text": "ReversingLabs classifies this sample as suspicious."},
    },
    "rl-cli/known": {
        "name": "KnownSample",
        "shortDescription": {"text": "ReversingLabs classifies this sample as known-good."},
    },
    "rl-cli/unknown": {
        "name": "UnknownSample",
        "shortDescription": {"text": "ReversingLabs states no verdict for this sample."},
    },
    "rl-cli/unreadable": {
        "name": "UnreadableFinding",
        "shortDescription": {"text": "This finding could not be read as a sample record."},
    },
}
_UNREADABLE_RULE_ID = "rl-cli/unreadable"


def sarif_level_of(severity: str) -> str:
    """The SARIF level ``severity`` reports as; "none" when unrecognised."""
    return _SARIF_LEVELS.get(severity, "none")


def sarif_rule_of(severity: str) -> str:
    """The stable rule id ``severity`` reports under.

    Four ids, fixed at import time, so GitHub's rule catalogue is four
    entries whatever the feed calls this week's family — and so an alert
    survives the family being renamed.

    A severity this table has never heard of reports as no opinion rather
    than as clean, which is the treatment ``severity_rank`` already gives
    it, and is the level :func:`sarif_level_of` grades it at anyway.
    """
    rule_id = f"rl-cli/{severity}"
    return rule_id if rule_id in _RULES else "rl-cli/unknown"


def to_sarif(data: Any, tool_name: str = "rl-cli") -> dict[str, Any]:
    """Build a SARIF 2.1.0 log dict from a service response payload."""
    results = [_result_from(item) for item in _iter_findings(data)]
    rule_ids = sorted({result["ruleId"] for result in results})
    return {
        "$schema": SARIF_SCHEMA,
        "version": SARIF_VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": tool_name,
                        "version": __version__,
                        "informationUri": _INFORMATION_URI,
                        "rules": [{"id": rule_id, **_RULES[rule_id]} for rule_id in rule_ids],
                    }
                },
                "results": results,
            }
        ],
    }


# The keys a paginated listing envelope carries and nothing else: the
# count, the cursors, and the page itself. A dict built from only these is
# the envelope with no page, which the appliance answers a listing that
# found nothing with -- see ``a1000.service`` for the same leniency.
_PAGINATION_KEYS = frozenset({"count", "next", "previous", "next_page", "results"})


def _iter_findings(data: Any) -> list[Any]:
    """Flatten known response envelopes into a list of finding payloads.

    Nothing found must produce no results. A phantom finding for an empty
    payload turns "0 detections" into one alert in whatever consumes the
    log, and for a paginated envelope it presents the
    ``{"count": 0, "next": null}`` wrapper itself as the detection.
    """
    if data is None:
        return []
    if isinstance(data, list):
        # A list item can be an envelope too — a hash search answers with
        # the reputation record still wrapped in {"rl": {...}} — and a
        # wrapper read as a record grades a known-malicious sample "none".
        return [unwrap_envelope(item) for item in data]
    if isinstance(data, dict):
        # An envelope is authoritative about its own emptiness. Its items
        # are unwrapped for the same reason the list branch above unwraps
        # its own: a paginated page of extracted files holds each record
        # under "sample", and the wrapper grades as no verdict.
        if isinstance(data.get("results"), list):
            return [unwrap_envelope(item) for item in data["results"]]
        # A paginated envelope that carries no ``results`` key at all is an
        # empty page, not a detection: the appliance answers a listing that
        # found nothing with ``{"count": 0, "next": null}``, and grading
        # that wrapper as a record turns 0 detections into one phantom
        # alert -- the exact case this function's contract rules out.
        if data and set(data) <= _PAGINATION_KEYS:
            return []
        # ``a1000 titanium-report`` answers the TitaniumCore document under
        # "ticore" and nothing else. The verdict is graded through the
        # wrapper either way — SampleVerdict knows the shape — but the hash
        # and the file name only exist inside the document, so a log built
        # on the wrapper names no sample at all. Gated on the document being
        # the whole answer: a detailed report carries its own record around
        # a "ticore" section, and descending into it would drop the record.
        document = data.get("ticore")
        if isinstance(document, dict) and set(data) == {"ticore"}:
            data = document

        # Anything else may still be wrapped — TitaniumCloud buries the
        # verdict under {"rl": {"malware_presence": {...}}}, and reading
        # the wrapper instead of the record grades a known-malicious
        # sample as level "none".
        record = unwrap_envelope(data)
        return [record] if record else []
    return [data]


def _result_from(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        # "We could not read this" is a non-alert, not a reassurance: it
        # must not be "note", the level ``known``/``goodware`` maps to, or
        # an unparseable entry and a clean sample read identically in the CI
        # log. ``none`` is what ``sarif_level_of`` answers for anything
        # unrecognised.
        #
        # Sanitised like every other message this file writes: the entry we
        # could not read is the whole alert body here, and a list of bare
        # strings — what ``ticloud uri-index`` answers — is as
        # attacker-authored as a file name. See :func:`_message`.
        return _located(
            {
                "ruleId": _UNREADABLE_RULE_ID,
                "level": "none",
                "message": {"text": sanitize(str(item))},
            },
            _UNIDENTIFIED_URI,
        )

    # One walk of the payload for the id, the level and the message.
    # Advanced Search v3 capitalises the verdict; the rule id and the
    # message carry the lower-case spelling.
    verdict = SampleVerdict.of(item)
    classification = verdict.classification.lower() if verdict.classification else None
    threat_name = verdict.threat_name
    # One reading of what the sample is called and hashed as, for the two
    # places below that name it. The spellings live in the payload reader:
    # the terminal and this log naming the same file differently is how a
    # Conti DLL becomes one sample on screen and another in CI.
    facts = SampleFacts.of(item)

    properties = _json_safe(item)
    if threat_name and isinstance(properties, dict):
        # The family name, in the one place a SARIF consumer reads free text
        # about a result: it is not the rule id, and it is not dropped — an
        # analyst reading the alert needs to know the thing is Conti.
        properties["threatName"] = threat_name

    return _located(
        {
            "ruleId": sarif_rule_of(verdict.severity),
            "level": sarif_level_of(verdict.severity),
            "message": {"text": _message(facts, classification, threat_name)},
            "properties": properties,
        },
        _artifact_uri(facts),
    )


def _located(result: dict[str, Any], uri: str) -> dict[str, Any]:
    """``result`` with the one location every result must carry.

    Every result, not only the ones we can name: GitHub Code Scanning drops
    a result with no ``locations`` outright, so a record carrying neither a
    file name nor a digest would be exported and then silently thrown away
    at the far end.
    """
    result["locations"] = [{"physicalLocation": {"artifactLocation": {"uri": uri}}}]
    return result


def _message(facts: SampleFacts, classification: str | None, threat_name: str | None) -> str:
    """The one line a human reads about this finding, in their code host's UI.

    Sanitised for the same reason every rendered string in this tree is,
    and against the same threat: the sample's file name and its threat
    name are attacker-chosen, and a right-to-left override in either one
    reverses the text after it: a name written as U+202E followed by
    ``gpj.exe`` reads as ``exe.jpg`` in GitHub's alert body — the file
    announced as the thing it is pretending to be, in the sentence the
    analyst is meant to trust. (Spelled by codepoint rather than written
    out, because a source file carrying the character is the Trojan
    Source hazard bandit's B613 exists to refuse.)

    Only the message is scrubbed. ``properties`` carries the record as it
    arrived, because that is what a consumer parses rather than reads.
    """
    subject = facts.name or facts.digest
    parts = [sanitize(classification) if classification else "result"]
    if threat_name:
        parts.append(sanitize(threat_name))
    if subject:
        parts.append(f"({sanitize(subject)})")
    return " ".join(parts)


def _artifact_uri(facts: SampleFacts) -> str:
    """What the log points at: the sample's name, or its hash when it has none.

    Under ``sample://`` and percent-encoded, always. The name is the
    sample's own file name, which is attacker-chosen, and writing it into
    the log verbatim is wrong two ways over:

    * It is not a URI reference. RFC 3986 has no place for a backslash or a
      literal newline, so ``..\\..\\..\\etc\\passwd`` produces a document a
      strict consumer is entitled to reject.
    * Where it *is* a valid relative reference it resolves against the
      repository. A sample named ``src/app.py`` puts a malware alert on that
      source file in GitHub Code Scanning — the CI equivalent of pointing at
      the wrong suspect, and a sample name is the cheapest thing in the
      world for an attacker to choose.

    ``quote(safe="")`` leaves only RFC 3986's unreserved set, so a
    separator, a control character or a bidi override becomes a
    percent-escape here rather than a path segment at the far end. The
    scheme is what makes it unresolvable against the repository at all.
    """
    subject = facts.name or facts.digest
    return f"{_SAMPLE_SCHEME}{quote(subject, safe='')}" if subject else _UNIDENTIFIED_URI


def _json_safe(data: Any) -> Any:
    """Reduce a payload to values a strict JSON parser will accept.

    ``parse_constant`` is what makes it strict: ``requests``' ``.json()``
    accepts NaN and Infinity on input and ``json.dumps`` writes them back
    out bare, so a single non-finite score otherwise produces a .sarif file
    that GitHub Code Scanning and jq reject outright — a pipeline that fails
    with no bad sample to point at. Null is what the TOON encoder writes for
    the same values.
    """
    return json.loads(json.dumps(data, default=str), parse_constant=lambda _: None)
