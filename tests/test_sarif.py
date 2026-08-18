"""Tests for the SARIF 2.1.0 exporter."""

import json
from typing import Any, ClassVar

import pytest

from rl_cli.models.payload import SampleVerdict
from rl_cli.render.sarif import SARIF_SCHEMA, SARIF_VERSION, sarif_rule_of, to_sarif

SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def only_result(payload) -> Any:
    (result,) = to_sarif(payload)["runs"][0]["results"]
    return result


def uri_of(result) -> str:
    return str(result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"])


class TestEnvelope:
    def test_schema_and_version(self):
        log = to_sarif({})
        assert log["$schema"] == SARIF_SCHEMA
        assert log["version"] == SARIF_VERSION
        assert len(log["runs"]) == 1

    def test_tool_driver_metadata(self):
        driver = to_sarif({})["runs"][0]["tool"]["driver"]
        assert driver["name"] == "rl-cli"
        assert driver["informationUri"].startswith("https://")
        assert isinstance(driver["rules"], list)

    def test_output_is_json_serializable(self):
        payload = {"sha256": SHA256, "classification": "malicious", "nested": {"x": object()}}
        assert json.loads(json.dumps(to_sarif(payload)))


class TestLevelMapping:
    def test_malicious_maps_to_error(self):
        result = to_sarif({"classification": "malicious", "sha256": SHA256})
        assert result["runs"][0]["results"][0]["level"] == "error"

    def test_suspicious_maps_to_warning(self):
        result = to_sarif({"classification": "suspicious"})
        assert result["runs"][0]["results"][0]["level"] == "warning"

    def test_clean_maps_to_note(self):
        result = to_sarif({"threat_status": "clean"})
        assert result["runs"][0]["results"][0]["level"] == "note"

    def test_unknown_maps_to_none(self):
        result = to_sarif({"classification": "weird"})
        assert result["runs"][0]["results"][0]["level"] == "none"

    def test_classification_dict_shape(self):
        payload = {"classification": {"classification": "malicious", "family_name": "Evil"}}
        sarif_result = to_sarif(payload)["runs"][0]["results"][0]
        assert sarif_result["level"] == "error"
        assert sarif_result["ruleId"] == "rl-cli/malicious"


class TestResults:
    def test_the_threat_name_is_reported_and_registered_under_the_severity_rule(self):
        log = to_sarif({"classification": "malicious", "threat_name": "Trojan.Generic"})
        run = log["runs"][0]
        (result,) = run["results"]

        assert result["ruleId"] == "rl-cli/malicious"
        assert result["properties"]["threatName"] == "Trojan.Generic"
        assert "Trojan.Generic" in result["message"]["text"]
        assert [rule["id"] for rule in run["tool"]["driver"]["rules"]] == ["rl-cli/malicious"]

    def test_hash_becomes_artifact_location(self):
        assert uri_of(only_result({"sha256": SHA256})) == f"sample://{SHA256}"

    def test_file_name_preferred_over_hash(self):
        result = only_result({"file_name": "evil.exe", "sha256": SHA256})
        assert uri_of(result) == "sample://evil.exe"

    def test_results_envelope_is_flattened(self):
        payload = {"results": [{"classification": "malicious"}, {"classification": "clean"}]}
        results = to_sarif(payload)["runs"][0]["results"]
        assert [r["level"] for r in results] == ["error", "note"]

    def test_list_input_yields_one_result_each(self):
        results = to_sarif([{"classification": "clean"}, "raw text"])["runs"][0]["results"]
        assert len(results) == 2
        assert results[1]["message"]["text"] == "raw text"

    def test_a_payload_that_is_not_a_record_is_still_reported(self):
        """A body that parsed to a bare scalar is one finding, not none."""
        results = to_sarif("raw text")["runs"][0]["results"]
        assert [r["message"]["text"] for r in results] == ["raw text"]

    def test_payload_preserved_in_properties(self):
        payload = {"classification": "malicious", "extra": {"depth": 1}}
        result = to_sarif(payload)["runs"][0]["results"][0]
        assert result["properties"]["extra"] == {"depth": 1}


class TestNothingFoundProducesNoResults:
    """A phantom finding turns "0 detections" into one alert downstream."""

    def _results(self, payload):
        return to_sarif(payload)["runs"][0]["results"]

    def test_empty_list(self):
        assert self._results([]) == []

    def test_empty_results_envelope(self):
        """The wrapper itself was being reported as the detection."""
        assert self._results({"count": 0, "next": None, "previous": None, "results": []}) == []

    def test_empty_page_that_omits_the_results_key(self):
        """A found-nothing listing answers ``{"count": 0, "next": null}``.

        With no ``results`` key the wrapper slipped past the empty-page
        guard and was graded as one detection — 0 detections read as a
        phantom alert downstream. A dict of pure pagination metadata is the
        empty page it looks like.
        """
        assert self._results({"count": 0, "next": None}) == []

    def test_a_record_that_merely_carries_a_count_is_not_an_empty_page(self):
        """The empty-page guard keys on the whole shape, not a lone field."""
        results = self._results({"count": 2, "classification": "malicious", "sha256": "a" * 64})
        assert len(results) == 1

    def test_empty_dict(self):
        assert self._results({}) == []

    def test_none(self):
        assert self._results(None) == []

    def test_no_rules_are_declared_either(self):
        driver = to_sarif({"results": []})["runs"][0]["tool"]["driver"]
        assert driver["rules"] == []

    def test_a_populated_envelope_is_still_flattened(self):
        results = self._results(
            {
                "count": 2,
                "results": [
                    {"classification": "malicious", "sha256": "a" * 64},
                    {"classification": "clean", "sha256": "b" * 64},
                ],
            }
        )
        assert [r["level"] for r in results] == ["error", "note"]


class TestFamilyNamesFromRealApiShapes:
    """v2 records and search entries carry no threat_name field.

    Each spelling still has to be *found* — that is the payload reader's
    job and it has not changed. What changed is where it is put: in
    ``properties.threatName`` and the message, never in the rule id.
    """

    def _names(self, payload):
        run = to_sarif(payload)["runs"][0]
        return [r["properties"].get("threatName") for r in run["results"]], [
            r["id"] for r in run["tool"]["driver"]["rules"]
        ]

    def test_classification_result_is_the_family_name(self):
        names, rules = self._names(
            [{"classification": "malicious", "classification_result": "Win64.Malware.Heuristic"}]
        )
        assert names == ["Win64.Malware.Heuristic"]
        assert rules == ["rl-cli/malicious"]

    def test_untokenized_threat_name_is_used_too(self):
        names, _ = self._names(
            [{"classification": "Malicious", "untokenized_threat_name": "Linux.Ransomware.Oof"}]
        )
        assert names == ["Linux.Ransomware.Oof"]

    def test_distinct_families_share_one_rule(self):
        """The catalogue used to grow one entry per family ever seen."""
        names, rules = self._names(
            [
                {"classification": "malicious", "classification_result": "Win64.Malware.Heuristic"},
                {"classification": "malicious", "classification_result": "Linux.Ransomware.Oof"},
            ]
        )
        assert names == ["Win64.Malware.Heuristic", "Linux.Ransomware.Oof"]
        assert rules == ["rl-cli/malicious"]

    def test_a_verdict_with_no_family_still_grades(self):
        names, rules = self._names([{"classification": "malicious"}])
        assert names == [None]
        assert rules == ["rl-cli/malicious"]


class TestTitaniumCloudEnvelope:
    """A verdict buried under {"rl": {"malware_presence": ...}} must still grade."""

    def test_malware_presence_is_graded_not_ignored(self):
        log = to_sarif(
            {
                "rl": {
                    "malware_presence": {
                        "sha1": "da39a3ee5e6b4b0d3255bfef95601890afd80709",
                        "status": "MALICIOUS",
                        "threat_name": "Win32.Trojan.Emotet",
                    }
                }
            }
        )
        (result,) = log["runs"][0]["results"]
        assert result["level"] == "error"
        assert "Emotet" in result["message"]["text"]

    def test_an_envelope_inside_a_list_is_unwrapped_too(self):
        """``ticloud search <hash>`` answers with a one-item list holding
        the reputation envelope; the list branch read it as a record."""
        log = to_sarif(
            [{"rl": {"malware_presence": {"status": "MALICIOUS", "threat_name": "Win32.Evil"}}}]
        )
        (result,) = log["runs"][0]["results"]
        assert result["level"] == "error"
        assert result["properties"]["threatName"] == "Win32.Evil"

    def test_a_url_report_verdict_is_graded(self):
        """TCA-0403 nests the URL report under "rl" and spells none of the
        record keys, so its verdict could never be read."""
        log = to_sarif(
            {
                "rl": {
                    "classification": "malicious",
                    "requested_url": "http://evil.tld/",
                    "analysis": {"statistics": {"malicious": 3}},
                }
            }
        )
        assert log["runs"][0]["results"][0]["level"] == "error"


class TestA1000EnvelopesThatGradedAsNothing:
    """Two A1000 shapes the log used to report at level "none".

    "none" is a non-alert in GitHub Code Scanning, so each of these
    shipped a named piece of malware to CI as no finding at all while the
    terminal painted it red.
    """

    TICORE: ClassVar[dict[str, Any]] = {
        "ticore": {
            "sha256": SHA256,
            "classification": {"classification": 3, "factor": 5, "result": "Win32.Trojan.Emotet"},
        }
    }
    EXTRACTED: ClassVar[dict[str, Any]] = {
        "filename": "stage2.dll",
        "sample": {
            "sha256": SHA256,
            "classification": "malicious",
            "classification_result": "Win64.Ransomware.Conti",
        },
    }

    def _result(self, payload):
        (result,) = to_sarif(payload)["runs"][0]["results"]
        return result

    def test_the_titaniumcore_document_grades_and_names_its_sample(self):
        result = self._result(self.TICORE)
        assert result["level"] == "error"
        assert result["properties"]["threatName"] == "Win32.Trojan.Emotet"
        assert uri_of(result) == f"sample://{SHA256}"

    def test_a_record_around_a_ticore_section_is_not_descended_into(self):
        """A detailed report grades itself; its document is one section."""
        result = self._result({"classification": "suspicious", **self.TICORE})
        assert result["level"] == "warning"
        assert result["properties"]["classification"] == "suspicious"

    def test_an_extracted_file_grades_and_keeps_the_name_it_was_found_under(self):
        result = self._result([self.EXTRACTED])
        assert result["level"] == "error"
        assert result["properties"]["threatName"] == "Win64.Ransomware.Conti"
        assert uri_of(result) == "sample://stage2.dll"

    def test_a_page_of_extracted_files_is_unwrapped_entry_by_entry(self):
        """The listing endpoint pages, and a page item is an entry too."""
        results = to_sarif({"count": 1, "results": [self.EXTRACTED]})["runs"][0]["results"]
        assert [r["level"] for r in results] == ["error"]

    def test_the_path_inside_the_container_names_an_unnamed_entry(self):
        result = self._result(
            [{"full_path": "arc/stage2.dll", "sample": {"threat_status": "clean"}}]
        )
        # The separator inside a container path is escaped like any other:
        # nothing in a sample's own name may become a path segment here.
        assert uri_of(result) == "sample://arc%2Fstage2.dll"


class TestSearchEntryFileNames:
    """An A1000 search entry spells the name ``file_names`` — a list."""

    ENTRY: ClassVar[dict[str, Any]] = {
        "file_names": ["invoice.exe"],
        "sha256": SHA256,
        "classification": "malicious",
    }

    def test_the_list_spelling_becomes_the_artifact_uri(self):
        assert uri_of(only_result(self.ENTRY)) == "sample://invoice.exe"

    def test_the_list_spelling_reaches_the_message_too(self):
        result = to_sarif(self.ENTRY)["runs"][0]["results"][0]
        assert "(invoice.exe)" in result["message"]["text"]

    def test_the_singular_spelling_still_wins_when_it_is_the_only_one(self):
        result = only_result({"file_name": "a.exe", "sha256": SHA256})
        assert uri_of(result) == "sample://a.exe"

    def test_an_empty_name_list_falls_through_to_the_hash(self):
        result = only_result({"file_names": [], "sha256": SHA256})
        assert uri_of(result) == f"sample://{SHA256}"


# Every verdict word the payload reader folds onto a severity, plus one it
# has never heard of. Spelled here rather than imported so that a word
# quietly dropped from the reader's table is a failure here too.
_VERDICTS = ("malicious", "suspicious", "clean", "goodware", "known", "unknown", "eldritch")


class TestARuleIdIsStableAndOursAlone:
    """The rule id used to be the threat name, straight out of the feed.

    SARIF §3.27.5 wants a stable opaque identifier, and GitHub Code
    Scanning keys its rule catalogue and its open alerts on it. A vendor
    family name is none of those things: it is attacker-influenced free
    text, there is one of them per family ever seen, and when the vendor
    renames a family every alert under the old id closes and reopens as a
    new one for the very same sample.
    """

    HOSTILE_NAME = "  ‮exe.dcoips\t\n  "

    def test_control_characters_and_bidi_overrides_cannot_reach_the_rule_id(self):
        log = to_sarif({"classification": "malicious", "threat_name": self.HOSTILE_NAME})
        run = log["runs"][0]
        rule_ids = [rule["id"] for rule in run["tool"]["driver"]["rules"]]

        assert run["results"][0]["ruleId"] == "rl-cli/malicious"
        assert rule_ids == ["rl-cli/malicious"]
        for spelling in [*rule_ids, run["results"][0]["ruleId"]]:
            assert "‮" not in spelling
            assert not any(character in spelling for character in "\t\n")
            assert spelling == spelling.strip()

    @pytest.mark.parametrize(
        ("payload", "rule_id"),
        [
            ({"classification": "malicious"}, "rl-cli/malicious"),
            ({"classification": "suspicious"}, "rl-cli/suspicious"),
            ({"threat_status": "goodware"}, "rl-cli/known"),
            ({"classification": "weird"}, "rl-cli/unknown"),
            ({"sha256": SHA256}, "rl-cli/unknown"),
        ],
    )
    def test_the_catalogue_is_the_four_severities(self, payload, rule_id):
        assert only_result(payload)["ruleId"] == rule_id

    def test_every_severity_the_payload_reader_can_state_has_a_rule(self):
        """The rule table is indexed, not looked up with a fallback."""
        severities = {SampleVerdict.of({"classification": word}).severity for word in _VERDICTS}

        assert {sarif_rule_of(severity) for severity in severities} == {
            "rl-cli/malicious",
            "rl-cli/suspicious",
            "rl-cli/known",
            "rl-cli/unknown",
        }

    def test_a_declared_rule_describes_itself(self):
        """A four-entry catalogue is worth naming; an unbounded one was not."""
        (rule,) = to_sarif({"classification": "malicious"})["runs"][0]["tool"]["driver"]["rules"]

        assert rule["id"] == "rl-cli/malicious"
        assert rule["name"] == "MaliciousSample"
        assert rule["shortDescription"]["text"]


class TestAnArtifactUriCannotAddressTheRepository:
    """The uri was the sample's own file name, which the attacker picks.

    Two separate defects, one per half of that sentence: a name that is
    not a URI reference at all produced a document a strict consumer may
    reject, and a name that *is* a valid relative reference resolved
    against the repository being scanned.
    """

    def test_a_sample_named_like_a_source_file_cannot_alert_on_that_file(self):
        result = only_result(
            {"file_name": "src/app.py", "classification": "malicious", "sha256": SHA256}
        )
        uri = uri_of(result)

        assert uri == "sample://src%2Fapp.py"
        # The two properties that matter downstream: it is not relative,
        # and it holds no separator GitHub could split into a path.
        assert uri.startswith("sample://")
        assert "/" not in uri.removeprefix("sample://")

    def test_a_traversal_name_is_escaped_rather_than_written_out(self):
        result = only_result({"file_name": "..\\..\\..\\etc\\passwd\nsecond line"})
        uri = uri_of(result)

        assert uri == "sample://..%5C..%5C..%5Cetc%5Cpasswd%0Asecond%20line"
        assert not any(character in uri for character in "\\\n \t")

    def test_a_result_with_nothing_to_name_still_carries_a_location(self):
        """GitHub Code Scanning drops a result that has no ``locations``."""
        result = only_result({"classification": "malicious"})

        assert uri_of(result) == "sample://unidentified"

    def test_every_result_of_a_mixed_log_is_located(self):
        results = to_sarif(
            [
                {"file_name": "a.exe", "classification": "malicious"},
                {"sha256": SHA256},
                {"classification": "clean"},
                "unparseable",
            ]
        )["runs"][0]["results"]

        assert all(result["locations"] for result in results)
        assert all(uri_of(result).startswith("sample://") for result in results)


class TestAnUnreadableFindingIsNotGradedClean:
    """ "We could not read this" and "this is clean" were the same value.

    A bare string in a results array — the case ``shapes.dict_rows``
    exists to defend against — came out as ``"level": "note"``, which is
    what ``known``/``goodware`` maps to. In a CI log the two are then
    indistinguishable.
    """

    def test_a_non_record_finding_grades_as_none(self):
        assert only_result("raw text")["level"] == "none"

    def test_it_does_not_share_a_level_with_a_clean_sample(self):
        clean = only_result({"threat_status": "clean"})

        assert clean["level"] == "note"
        assert only_result("raw text")["level"] != clean["level"]

    @pytest.mark.parametrize(
        "item",
        ["\u202egpj.exe", "\x1b[2Jcleared", "\x1b]0;pwned\x07"],
        ids=["bidi", "escape", "title"],
    )
    def test_the_entry_we_could_not_read_is_still_sanitised(self, item):
        """It is the whole alert body, and it was the one message that was not.

        ``ticloud uri-index`` answers a JSON array of SHA-1 strings, so any
        listing can carry a stray one — and this branch wrote it into
        ``message.text`` verbatim, bidi override and ESC included, in the
        sentence the analyst reads in their code host. Every other message
        this exporter writes goes through ``sanitize``.
        """
        text = only_result(item)["message"]["text"]

        assert "\x1b" not in text
        assert "\u202e" not in text

    def test_the_entry_is_still_named_in_the_message(self):
        assert "gpj.exe" in only_result("\u202egpj.exe")["message"]["text"]

    def test_it_says_so_in_its_rule_rather_than_claiming_a_verdict(self):
        log = to_sarif(["raw text"])
        (result,) = log["runs"][0]["results"]
        (rule,) = log["runs"][0]["tool"]["driver"]["rules"]

        assert result["ruleId"] == "rl-cli/unreadable"
        assert rule["id"] == "rl-cli/unreadable"
        assert "could not be read" in rule["shortDescription"]["text"]


# What a SARIF 2.1.0 consumer is entitled to assume of every result, held
# over the payload shapes this exporter actually meets. GitHub Code
# Scanning refuses a log whose result names a rule the driver never
# declared, and drops a result carrying no locations — both of which are
# silent at the point the file is written and loud a build later.
_SHAPES: dict[str, Any] = {
    "a1000-record": {"file_name": "src/app.py", "classification": "malicious", "sha256": SHA256},
    "search-entry": {"file_names": ["invoice.exe"], "classification": "Suspicious"},
    "ticloud-envelope": {"rl": {"malware_presence": {"status": "KNOWN", "sha1": "a" * 40}}},
    "extracted-file": {"filename": "s.dll", "sample": {"classification": "malicious"}},
    "hostile-name": {"file_name": "..\\..\\x\ny", "threat_name": "  ‮exe.dcoips\t\n  "},
    "nothing-stated": {"count": 1, "results": [{"id": 4}]},
    "not-a-record": "raw text",
}


class TestTheDocumentSatisfiesTheSarifContract:
    @pytest.mark.parametrize("payload", _SHAPES.values(), ids=list(_SHAPES))
    def test_every_result_is_well_formed(self, payload):
        run = to_sarif(payload)["runs"][0]
        declared = {rule["id"] for rule in run["tool"]["driver"]["rules"]}

        assert run["results"], "a shape that produced no finding at all"
        for result in run["results"]:
            assert result["ruleId"] in declared
            assert result["level"] in {"none", "note", "warning", "error"}
            assert isinstance(result["message"]["text"], str)
            uri = uri_of(result)
            assert uri.isascii() and uri.isprintable()
            assert uri.startswith("sample://")

    @pytest.mark.parametrize("payload", _SHAPES.values(), ids=list(_SHAPES))
    def test_the_whole_log_survives_a_strict_json_round_trip(self, payload):
        assert json.loads(json.dumps(to_sarif(payload)))


class TestTheAlertBodyCannotBeReversedByTheSample:
    """A right-to-left override in a name reverses the text after it.

    ``message.text`` is the sentence a human reads in their code host's
    alert body, and both halves it interpolates -- the sample's file name
    and its threat name -- are attacker-chosen. A name written as U+202E
    followed by ``gpj.exe`` reads as ``exe.jpg`` there: the file announced
    as the thing it is pretending to be, in the line the analyst is meant
    to trust. Every other rendered string in this tree goes through
    ``sanitize`` for this reason; the SARIF message did not.

    The character is written as an escape, never literally: a source file
    that carries one is itself the Trojan Source hazard.
    """

    BIDI = "\u202e"

    def _finding(self, **record):
        return to_sarif([{"sha256": "a" * 64, "classification": "malicious", **record}])["runs"][0][
            "results"
        ][0]

    def test_a_reversed_file_name_does_not_reverse_the_message(self):
        message = self._finding(file_name=f"{self.BIDI}gpj.exe")["message"]["text"]

        assert self.BIDI not in message
        assert "gpj.exe" in message

    def test_a_reversed_threat_name_does_not_reverse_the_message(self):
        message = self._finding(threat_name=f"{self.BIDI}exe.dcoips")["message"]["text"]

        assert self.BIDI not in message

    def test_an_escape_sequence_cannot_reach_the_message(self):
        message = self._finding(file_name="\x1b[2Jinvoice.pdf")["message"]["text"]

        assert "\x1b" not in message

    def test_the_payload_still_carries_the_record_as_it_arrived(self):
        """``properties`` is parsed, not read, so it is not scrubbed."""
        finding = self._finding(threat_name=f"{self.BIDI}exe.dcoips")

        assert finding["properties"]["threatName"] == f"{self.BIDI}exe.dcoips"
