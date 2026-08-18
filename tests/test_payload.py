"""Tests for the shared payload readers in rl_cli.models.payload."""

import io
from typing import Any, ClassVar, cast

import pytest
from rich.console import Console

from rl_cli.models.payload import (
    ReanalysisOutcome,
    SampleVerdict,
    ScannerConsensus,
    search_page,
    unwrap_envelope,
)
from rl_cli.models.shapes import count, stated_count
from rl_cli.render.formatters.a1000_listings import (
    print_extracted_files_table,
    print_samples_panels,
    print_search_results_table,
)
from rl_cli.render.formatters.a1000_operations import print_reanalyze_results_table
from rl_cli.render.formatters.a1000_sample_report import print_report_summary, print_titanium_report
from rl_cli.render.formatters.severity import colour_of
from rl_cli.render.formatters.ticloud_report import print_file_analysis
from rl_cli.render.sarif import sarif_level_of, sarif_rule_of, to_sarif


# The production code reads a verdict once through SampleVerdict; these
# name the four answers so the assertions below stay readable.
def classification_of(payload):
    return SampleVerdict.of(payload).classification


def severity_of(payload):
    return SampleVerdict.of(payload).severity


def threat_name_of(payload):
    return SampleVerdict.of(payload).threat_name


def threat_level_of(payload):
    return SampleVerdict.of(payload).threat_level


@pytest.fixture
def console() -> Console:
    """A Console wired to an in-memory buffer so tests stay quiet."""
    return Console(file=io.StringIO(), width=120)


class TestUnwrapEnvelope:
    def test_paginated_results_envelope_yields_first_record(self):
        assert unwrap_envelope({"count": 2, "results": [{"sha1": "a"}, {"sha1": "b"}]}) == {
            "sha1": "a"
        }

    def test_titaniumcloud_malware_presence_is_read_through(self):
        payload = {"rl": {"malware_presence": {"status": "MALICIOUS"}}}
        assert unwrap_envelope(payload) == {"status": "MALICIOUS"}

    def test_rl_sample_is_read_through(self):
        assert unwrap_envelope({"rl": {"sample": {"sha1": "a"}}}) == {"sha1": "a"}

    def test_a_plain_record_is_returned_unchanged(self):
        record = {"sha1": "a"}
        assert unwrap_envelope(record) is record

    def test_an_unrecognised_rl_body_is_the_record_itself(self):
        """The TCA-0403 URL report nests everything under "rl" without a
        malware_presence or sample block, so leaving it wrapped meant a
        URL verdict could never be read."""
        report = {"classification": "malicious", "requested_url": "http://evil.tld/"}
        assert unwrap_envelope({"rl": report}) == report

    def test_empty_and_malformed_envelopes_are_left_alone(self):
        assert unwrap_envelope({"results": []}) == {"results": []}
        assert unwrap_envelope({"results": ["bare-string"]}) == {"results": ["bare-string"]}
        assert unwrap_envelope({"rl": "not-a-dict"}) == {"rl": "not-a-dict"}

    @pytest.mark.parametrize("payload", ["bare-string", None, 7, ["a"], True])
    def test_a_payload_that_is_not_a_record_is_handed_straight_back(self, payload):
        """``sarif._iter_findings`` maps this over the items of a JSON array.

        An array may hold a bare string, so the guard is what stands between
        an unreadable entry and an ``AttributeError`` — and the entry has to
        survive as far as ``_result_from``, which reports "we could not read
        this" rather than dropping it out of the log. Nothing exercised this
        path while the parameter was annotated ``dict[str, Any]``, which also
        left mypy reading the guard as unreachable.
        """
        assert unwrap_envelope(payload) is payload

    def test_an_extracted_file_entry_is_read_through_to_its_record(self):
        """``a1000 extracted`` states the verdict one level down under
        "sample"; the entry read as the record graded a Conti DLL nothing."""
        entry = {
            "filename": "stage2.dll",
            "sample": {"sha256": "c" * 64, "classification": "malicious"},
        }
        assert unwrap_envelope(entry)["classification"] == "malicious"

    def test_the_extracted_entry_keeps_the_name_only_it_carries(self):
        """The record grades the file and only the entry says which file."""
        entry = {"filename": "stage2.dll", "full_path": "a/stage2.dll", "sample": {"sha1": "a"}}
        assert unwrap_envelope(entry) == {
            "filename": "stage2.dll",
            "full_path": "a/stage2.dll",
            "sha1": "a",
        }

    def test_the_record_outranks_the_entry_around_it(self):
        entry = {"sha1": "outer", "sample": {"sha1": "inner"}}
        assert unwrap_envelope(entry)["sha1"] == "inner"

    def test_the_entrys_own_verdict_is_not_lost_to_the_record(self):
        """An entry can grade the file itself and carry only hashes below.

        The merge kept that verdict, so the SARIF log read it; the screen
        unwrapped the same entry its own way and read the record alone, so
        this DLL was "unknown" on screen and "error" in the log.
        """
        entry = {
            "filename": "payload.dll",
            "classification": "malicious",
            "threat_name": "Win32.Trojan.Conti",
            "sample": {"sha1": "b" * 40},
        }
        record = unwrap_envelope(entry)
        assert record["classification"] == "malicious"
        assert record["threat_name"] == "Win32.Trojan.Conti"
        assert record["sha1"] == "b" * 40

    def test_a_non_dict_sample_field_is_left_alone(self):
        assert unwrap_envelope({"sample": "not-a-dict"}) == {"sample": "not-a-dict"}


class TestTheWorseVerdictWinsACollision:
    """The stated rule when both halves of an entry grade the sample.

    A key-by-key merge resolved a disagreement toward whichever half spelled
    its verdict in the key ``SampleVerdict`` reads first, so an entry stating
    ``malicious`` beside a Conti threat name came back ``goodware``.
    Resolving toward the reassuring value is the wrong default for a malware
    tool, so the more severe verdict wins and the carried record — the more
    specific of the two — only breaks ties.
    """

    def test_the_entrys_worse_verdict_wins(self):
        entry = {"classification": "malicious", "sample": {"classification": "goodware"}}
        assert severity_of(unwrap_envelope(entry)) == "malicious"

    def test_the_records_worse_verdict_wins(self):
        entry = {"classification": "goodware", "sample": {"classification": "malicious"}}
        assert severity_of(unwrap_envelope(entry)) == "malicious"

    def test_the_winner_wins_across_spellings_of_the_verdict(self):
        """The two halves need not disagree in the same key to disagree.

        ``classification`` is read before ``threat_status``, so a merge that
        left both in place graded by spelling rather than by severity.
        """
        entry = {"classification": "goodware", "sample": {"threat_status": "malicious"}}
        assert severity_of(unwrap_envelope(entry)) == "malicious"

        entry = {"threat_status": "malicious", "sample": {"classification": "goodware"}}
        assert severity_of(unwrap_envelope(entry)) == "malicious"

    def test_the_whole_verdict_block_moves_with_the_winner(self):
        """A verdict beside the loser's family name is an incoherent record."""
        entry = {
            "classification": "goodware",
            "threat_name": "Win32.Goodware.Nothing",
            "threat_level": 0,
            "sample": {
                "classification": "malicious",
                "classification_result": "Win64.Ransomware.Conti",
                "threat_level": 5,
            },
        }
        record = unwrap_envelope(entry)
        assert classification_of(record) == "malicious"
        assert threat_name_of(record) == "Win64.Ransomware.Conti"
        assert threat_level_of(record) == 5

    def test_a_tie_goes_to_the_carried_record(self):
        entry = {
            "classification": "malicious",
            "threat_name": "Win32.Trojan.Generic",
            "sample": {"classification": "malicious", "threat_name": "Win64.Ransomware.Conti"},
        }
        assert threat_name_of(unwrap_envelope(entry)) == "Win64.Ransomware.Conti"

    def test_a_half_that_states_no_verdict_takes_nothing_from_the_other(self):
        """Only a stated verdict collides; a name alone does not.

        The entry names a threat and grades nothing, so the record's verdict
        stands and the name it does not carry is still read off the entry.
        """
        entry = {"threat_name": "Win32.Trojan.Conti", "sample": {"classification": "suspicious"}}
        record = unwrap_envelope(entry)
        assert classification_of(record) == "suspicious"
        assert threat_name_of(record) == "Win32.Trojan.Conti"

    def test_no_opinion_outranks_a_stated_clean(self):
        """ "We have no opinion" is the less reassuring of the two."""
        entry = {"classification": "unknown", "sample": {"classification": "goodware"}}
        assert severity_of(unwrap_envelope(entry)) == "unknown"


class TestTheLosersEvidenceSurvivesTheCollision:
    """The verdict block that moved took the whole analysis with it.

    ``ticore`` and ``analysis`` are not spellings of a verdict — they are
    the TitaniumCore and RLDATA *documents*, which state one among a story,
    an indicator list, a signature list and an ATT&CK matrix. Listing them
    among the keys that move with the winning verdict meant the loser's
    document was dropped outright, and only ever when the two halves
    disagreed: an entry graded malicious around a sample graded goodware
    lost every piece of evidence the sample carried. The worse the verdict,
    the less the report showed.
    """

    TICORE: ClassVar[dict[str, Any]] = {
        "story": "The file encrypts documents and deletes shadow copies.",
        "indicators": [{"description": "Deletes volume shadow copies", "priority": 10}],
        "attack": [
            {
                "tactics": [
                    {
                        "name": "Impact",
                        "techniques": [{"id": "T1490", "name": "Inhibit System Recovery"}],
                    }
                ]
            }
        ],
    }
    ENTRY: ClassVar[dict[str, Any]] = {
        "filename": "payload.dll",
        "threat_status": "malicious",
        "sample": {"sha256": "b" * 64, "classification": "goodware", "ticore": TICORE},
    }

    def test_the_document_survives_a_verdict_it_lost(self):
        assert unwrap_envelope(self.ENTRY)["ticore"] == self.TICORE

    def test_the_worse_verdict_still_wins(self):
        assert severity_of(unwrap_envelope(self.ENTRY)) == "malicious"

    def test_the_report_still_renders_the_whole_analysis(self, console):
        print_report_summary(unwrap_envelope(self.ENTRY), console=console)
        rendered = console.file.getvalue()
        assert "encrypts documents and deletes shadow copies" in rendered
        assert "Deletes volume shadow copies" in rendered
        assert "T1490" in rendered

    def test_an_rldata_analysis_block_survives_the_same_way(self):
        entry = {
            "threat_status": "malicious",
            "sample": {
                "classification": "goodware",
                "analysis": {"entries": [{"tc_report": {"info": {"file": {"size": 4096}}}}]},
            },
        }
        assert "analysis" in unwrap_envelope(entry)

    def test_a_verdict_buried_in_the_losers_document_does_not_overrule_the_winner(self):
        """The document stays; the verdict it states is still the loser's."""
        entry = {
            "classification": "malicious",
            "sample": {"classification": "goodware", "ticore": {"classification": {"factor": 1}}},
        }
        assert severity_of(unwrap_envelope(entry)) == "malicious"

    def test_a_half_whose_only_verdict_is_buried_still_collides(self):
        """A document is where an A1000 record states its verdict, so a half
        carrying nothing but one still has an opinion to weigh."""
        entry = {
            "ticore": {"classification": {"classification": 3, "result": "Win32.Ransom.Conti"}},
            "sample": {"classification": "goodware"},
        }
        assert severity_of(unwrap_envelope(entry)) == "malicious"

    def test_the_winners_own_document_survives_when_both_halves_carry_one(self):
        """The winning verdict cannot be the one the rebuild discards.

        The entry wins on severity through its document, so it contributed
        no verdict spelling to the rebuild while the loser's was stripped —
        over a merge that had already replaced the entry's document with the
        carried one. The Conti DLL came back with a goodware document and no
        verdict spelling at all, graded goodware on screen and in SARIF.
        """
        entry = {
            "file_name": "stage2.dll",
            "ticore": {"classification": {"classification": 3, "result": "Win64.Ransomware.Conti"}},
        }
        carried = {
            "classification": "goodware",
            "ticore": {"classification": {"classification": 1}},
        }
        record = unwrap_envelope({**entry, "sample": carried})

        assert severity_of(record) == "malicious"
        assert threat_name_of(record) == "Win64.Ransomware.Conti"
        assert record["file_name"] == "stage2.dll"

    @pytest.mark.parametrize("side", ["entry", "carried"])
    def test_the_loser_does_not_keep_a_document_the_winner_spells_elsewhere(self, side):
        """The two halves can bury their verdicts under different keys.

        ``_titaniumcore_classification`` reads ``ticore`` before
        ``analysis``, so a winner stating its verdict in ``analysis`` beside
        a loser's ``ticore`` was graded by the loser: the rebuild moved the
        wider bearing set in but stripped only the narrow one, leaving the
        loser's document in place. Both directions, because which half is
        the entry is not the appliance's business.
        """
        malicious = {
            "analysis": {
                "entries": [
                    {
                        "tc_report": {
                            "classification": {
                                "classification": 3,
                                "result": "Win64.Ransomware.Conti",
                            }
                        }
                    }
                ]
            }
        }
        clean = {"ticore": {"classification": {"classification": 1}}}
        entry = {"file_name": "stage2.dll"} | (malicious if side == "entry" else clean)
        carried = clean if side == "entry" else malicious

        record = unwrap_envelope({**entry, "sample": carried})

        assert severity_of(record) == "malicious"
        assert threat_name_of(record) == "Win64.Ransomware.Conti"


class TestClassificationOf:
    def test_bare_string(self):
        assert classification_of({"classification": "malicious"}) == "malicious"

    def test_case_is_preserved_for_display(self):
        assert classification_of({"classification": "Malicious"}) == "Malicious"

    def test_nested_dict_shape(self):
        assert classification_of({"classification": {"classification": "clean"}}) == "clean"

    def test_titaniumcore_numeric_code_is_decoded(self):
        assert classification_of({"classification": 3}) == "malicious"
        assert classification_of({"classification": 0}) == "unknown"

    @pytest.mark.parametrize("stated", [True, False], ids=["true", "false"])
    def test_a_boolean_is_not_a_titaniumcore_code(self, stated):
        """``bool`` is a subclass of ``int``, and ``{True: ...}`` is ``{1: ...}``.

        So dropping the ``not isinstance(value, bool)`` guard graded a
        payload spelling ``"classification": true`` as **goodware** — the
        code 1 sits at — and ``false`` as "unknown". The suite stayed
        green through it. A field spelling a boolean where the appliance
        spells a code is a payload we cannot read, and "we have no
        opinion" is the answer for that; reading it as a verdict is how a
        malformed or spoofed record comes back clean.
        """
        assert severity_of({"classification": stated}) == "unknown"

    def test_status_and_threat_status_aliases(self):
        assert classification_of({"status": "KNOWN"}) == "KNOWN"
        assert classification_of({"threat_status": "clean"}) == "clean"

    def test_no_verdict_is_none(self):
        assert classification_of({"sha1": "a"}) is None


class TestThreatNameOf:
    def test_threat_name_wins(self):
        assert threat_name_of({"threat_name": "Trojan.Generic"}) == "Trojan.Generic"

    def test_family_name_from_the_classification_block(self):
        assert threat_name_of({"classification": {"family_name": "Emotet"}}) == "Emotet"

    def test_classification_result_outranks_the_family_fragment(self):
        """An A1000 v2 record carries both; only the dotted name is complete."""
        assert threat_name_of(
            {
                "classification": {"family_name": "Emotet"},
                "classification_result": "Win64.Trojan.Emotet",
            }
        ) == ("Win64.Trojan.Emotet")

    def test_advanced_search_v3_fallbacks(self):
        assert threat_name_of({"classification_result": "Win64.Malware.Heuristic"}) == (
            "Win64.Malware.Heuristic"
        )
        assert threat_name_of({"untokenized_threat_name": "Linux.Ransomware.Oof"}) == (
            "Linux.Ransomware.Oof"
        )
        assert threat_name_of({"threat_family": "Oof"}) == "Oof"

    def test_no_name_is_none(self):
        assert threat_name_of({"classification": "malicious"}) is None


class TestFileAnalysisRecord:
    """TCA-0104 (RLDATA) buries the verdict in the first analysis entry."""

    RECORD: ClassVar[dict[str, Any]] = {
        "sha1": "a" * 40,
        "sample_type": "PE/Exe",
        "sample_size": 4096,
        "analysis": {
            "entries": [
                {
                    "tc_report": {
                        "classification": {
                            "classification": 3,
                            "factor": 5,
                            "result": "Win32.Trojan.Emotet",
                        }
                    }
                }
            ]
        },
    }

    def test_the_tc_report_code_is_the_verdict(self):
        assert classification_of(self.RECORD) == "malicious"
        assert severity_of(self.RECORD) == "malicious"

    def test_the_tc_report_result_is_the_threat_name(self):
        assert threat_name_of(self.RECORD) == "Win32.Trojan.Emotet"

    def test_a_record_carrying_its_own_verdict_still_wins(self):
        record = dict(self.RECORD, status="KNOWN", threat_name="Named.Elsewhere")
        assert classification_of(record) == "KNOWN"
        assert threat_name_of(record) == "Named.Elsewhere"

    @pytest.mark.parametrize(
        "record",
        [
            {"analysis": "not-a-dict"},
            {"analysis": {"entries": []}},
            {"analysis": {"entries": ["bare-string"]}},
            {"analysis": {"entries": [{"tc_report": None}]}},
            {"analysis": {"entries": [{"tc_report": {"classification": 3}}]}},
        ],
    )
    def test_malformed_analysis_blocks_are_survived(self, record):
        assert severity_of(record) == "unknown"
        assert threat_name_of(record) is None


class TestTitaniumCoreDocument:
    """``a1000 titanium-report`` answers the TitaniumCore document itself.

    The verdict is the numeric code inside a "classification" block, and
    the appliance sends the document either bare or one level down under
    "ticore". The renderer reads both spellings; the model read neither,
    so an Emotet sample the terminal painted red exported as level "none".
    """

    DOCUMENT: ClassVar[dict[str, Any]] = {
        "sha256": "e" * 64,
        "info": {"file": {"file_type": "PE+ executable", "size": 4096}},
        "classification": {"classification": 3, "factor": 5, "result": "Win32.Trojan.Emotet"},
    }
    WRAPPED: ClassVar[dict[str, Any]] = {"ticore": DOCUMENT}

    @pytest.mark.parametrize("payload", [DOCUMENT, WRAPPED], ids=["bare", "under-ticore"])
    def test_the_document_grades_malicious_either_way(self, payload):
        assert classification_of(payload) == "malicious"
        assert severity_of(payload) == "malicious"

    @pytest.mark.parametrize("payload", [DOCUMENT, WRAPPED], ids=["bare", "under-ticore"])
    def test_the_result_is_the_threat_name(self, payload):
        assert threat_name_of(payload) == "Win32.Trojan.Emotet"

    @pytest.mark.parametrize("payload", [DOCUMENT, WRAPPED], ids=["bare", "under-ticore"])
    def test_the_factor_is_the_threat_level(self, payload):
        assert threat_level_of(payload) == 5

    def test_the_record_around_a_ticore_section_still_states_its_own_verdict(self):
        """The detailed report carries the document as one section of a
        record that grades itself; the section must not overrule it."""
        report = {"classification": "suspicious", "ticore": self.DOCUMENT}
        assert severity_of(report) == "suspicious"

    @pytest.mark.parametrize(
        "payload",
        [
            {"ticore": "not-a-dict"},
            {"ticore": {}},
            {"ticore": {"classification": 3}},
            {"ticore": {"classification": {}}},
        ],
    )
    def test_malformed_documents_are_survived(self, payload):
        assert severity_of(payload) == "unknown"
        assert threat_name_of(payload) is None


# Processing states the A1000 spells in the same "status" field that
# carries TitaniumCloud's verdict. None of them is an opinion about the
# sample.
_PROCESSING_STATES = ["processed", "accepted", "queued", "not_found", "in_progress", "Submitted"]


class TestProcessingStateIsNotAVerdict:
    """``status`` means the verdict on one API and the state on the other.

    Reading the A1000's processing state as a verdict graded a Conti
    sample "processed" — which folds to no severity and exports as SARIF
    "none" — while ``threat_status`` beside it said malicious.
    """

    @pytest.mark.parametrize("state", _PROCESSING_STATES)
    def test_a_state_never_becomes_a_verdict(self, state):
        assert classification_of({"status": state}) is None
        assert severity_of({"status": state}) == "unknown"

    @pytest.mark.parametrize("state", _PROCESSING_STATES)
    def test_an_explicit_threat_status_outranks_the_state(self, state):
        record = {
            "status": state,
            "threat_status": "malicious",
            "threat_name": "Win32.Ransomware.Conti",
        }
        assert classification_of(record) == "malicious"
        assert severity_of(record) == "malicious"
        assert to_sarif(record)["runs"][0]["results"][0]["level"] == "error"

    def test_an_explicit_threat_status_outranks_a_status_verdict_too(self):
        """A field that means two things cannot overrule one that means one."""
        assert classification_of({"status": "KNOWN", "threat_status": "malicious"}) == "malicious"

    def test_a_state_leaves_the_titaniumcore_block_to_grade(self):
        record = dict(TestFileAnalysisRecord.RECORD, status="processed")
        assert severity_of(record) == "malicious"

    def test_the_upload_receipt_claims_no_verdict_at_all(self):
        """``a1000 upload``'s fallback receipt: a state, and nothing else."""
        assert SampleVerdict.of({"status": "accepted"}) == SampleVerdict(
            classification=None, severity="unknown", threat_name=None, threat_level=None
        )

    @pytest.mark.parametrize(
        "verdict", ["MALICIOUS", "suspicious", "KNOWN", "goodware", "clean", "unknown"]
    )
    def test_a_status_that_names_a_verdict_is_still_read_as_one(self, verdict):
        """TitaniumCloud's malware_presence.status is the verdict itself."""
        assert classification_of({"status": verdict}) == verdict


# --- Malicious must never render reassuring -----------------------------------
#
# The shapes below all say "this sample is malware", each in the spelling
# one endpoint uses. The invariant is not that each grades to some
# particular string but that none of them can come out as a clean or a
# no-opinion verdict: an unparsed payload that reads reassuring is the one
# failure this tool cannot afford.

_MALICIOUS_SHAPES = {
    "ticloud-malware-presence": {
        "rl": {"malware_presence": {"status": "MALICIOUS", "threat_name": "Win32.Trojan.Emotet"}}
    },
    "ticloud-rldata": {
        "rl": {
            "sample": {
                "sha1": "a" * 40,
                "analysis": {
                    "entries": [
                        {
                            "tc_report": {
                                "classification": {
                                    "classification": 3,
                                    "result": "Win32.Trojan.Emotet",
                                }
                            }
                        }
                    ]
                },
            }
        }
    },
    "ticloud-url-report": {"rl": {"classification": "malicious", "requested_url": "http://evil/"}},
    "advanced-search-entry": {
        "classification": "MALICIOUS",
        "classification_result": "Win64.Malware.Heuristic",
        "sha256": "b" * 64,
    },
    "a1000-detailed-report": {
        "classification": "malicious",
        "classification_result": "Win32.Trojan.Emotet",
        "threat_level": 5,
    },
    "a1000-paginated-report": {
        "count": 1,
        "results": [{"threat_status": "malicious", "threat_name": "Win32.Trojan.Emotet"}],
    },
    "a1000-titanium-report": TestTitaniumCoreDocument.WRAPPED,
    "a1000-titanium-document": TestTitaniumCoreDocument.DOCUMENT,
    "a1000-extracted-entry": {
        "filename": "stage2.dll",
        "sample": {
            "sha256": "c" * 64,
            "classification": "malicious",
            "classification_result": "Win64.Ransomware.Conti",
        },
    },
    "a1000-processed-sample": {
        "status": "processed",
        "threat_status": "malicious",
        "threat_name": "Win32.Ransomware.Conti",
    },
    "a1000-classification-block": {
        "classification": {"classification": 3, "family_name": "Emotet"}
    },
    "yara-match-code": {"classification": 3, "sha256": "d" * 64},
}


@pytest.mark.parametrize("payload", _MALICIOUS_SHAPES.values(), ids=list(_MALICIOUS_SHAPES))
def test_no_malicious_shape_can_grade_reassuring(payload):
    assert severity_of(unwrap_envelope(payload)) == "malicious"
    for result in to_sarif(payload)["runs"][0]["results"]:
        assert result["level"] == "error"


# Every way an A1000 extracted-file entry can place a verdict: on the entry,
# in the record it carries under "sample", on both halves at once (agreeing
# or not), and on neither. Each row is the one grading both the screen and
# the log have to reach.
#
# id -> (entry, severity, the verdict as displayed, the log's rule id)
_EXTRACTED_PLACEMENTS: dict[str, tuple[dict[str, Any], str, str, str]] = {
    "verdict-in-the-carried-record": (
        {
            "filename": "stage2.dll",
            "sample": {
                "sha256": "c" * 64,
                "classification": "malicious",
                "classification_result": "Win64.Ransomware.Conti",
            },
        },
        "malicious",
        "malicious",
        "Win64.Ransomware.Conti",
    ),
    "verdict-on-the-entry": (
        {
            "filename": "payload.dll",
            "classification": "malicious",
            "threat_name": "Win32.Trojan.Conti",
            "sample": {"sha1": "b" * 40, "sha256": "c" * 64},
        },
        "malicious",
        "malicious",
        "Win32.Trojan.Conti",
    ),
    "verdict-on-the-entry-in-caps": (
        {
            "filename": "loader.exe",
            "classification": "MALICIOUS",
            "sample": {"sha256": "e" * 64},
        },
        "malicious",
        "MALICIOUS",
        "malicious",
    ),
    "the-entry-grades-it-worse-than-the-record": (
        {
            "filename": "payload.dll",
            "classification": "malicious",
            "threat_name": "Win32.Trojan.Conti",
            "sample": {"sha256": "c" * 64, "classification": "goodware"},
        },
        "malicious",
        "malicious",
        "Win32.Trojan.Conti",
    ),
    "the-record-grades-it-worse-than-the-entry": (
        {
            "filename": "payload.dll",
            "classification": "goodware",
            "sample": {
                "sha256": "c" * 64,
                "classification": "malicious",
                "classification_result": "Win64.Ransomware.Conti",
            },
        },
        "malicious",
        "malicious",
        "Win64.Ransomware.Conti",
    ),
    "neither-half-states-one": (
        {"filename": "readme.txt", "sample": {"sha256": "d" * 64}},
        "unknown",
        "unknown",
        "result",
    ),
}


class TestTheScreenAndTheLogAgreeOnTheShapesThatDisagreed:
    """The terminal and the SARIF log grade one payload one way.

    Both defects were the two halves of a single run disagreeing: first a
    red terminal beside a "none" log, then — once the log learned the
    extracted-file shape — an "unknown" terminal beside an "error" log. The
    terminal is what the analyst reads and the log is what CI reads, so
    either way round a Conti DLL ships onward as something it is not.

    So the invariant is tested as a property over shapes rather than as the
    examples that broke: for every placement of a verdict, the colour on
    screen and the level in the log name the same severity, and both name
    the file. The severity→colour and severity→level tables are pinned
    literally further down, in ``test_every_site_presents_the_same_severity``.
    """

    def _sarif(self, payload):
        return to_sarif(payload)["runs"][0]["results"][0]

    def test_the_titanium_report_reads_the_same_on_screen_and_in_the_log(self, ansi):
        payload = TestTitaniumCoreDocument.WRAPPED
        screen = ansi(lambda c: print_titanium_report(payload, console=c))

        assert f"{_ANSI['red']}malicious" in screen
        assert "Win32.Trojan.Emotet" in screen

        result = self._sarif(payload)
        assert result["level"] == "error"
        # The family name is a property of the result, not its rule id —
        # see tests/test_sarif.py::TestARuleIdIsStableAndOursAlone.
        assert result["properties"]["threatName"] == "Win32.Trojan.Emotet"

    @pytest.mark.parametrize(
        ("entry", "severity", "displayed", "named"),
        _EXTRACTED_PLACEMENTS.values(),
        ids=list(_EXTRACTED_PLACEMENTS),
    )
    def test_an_extracted_file_reads_the_same_on_screen_and_in_the_log(
        self, ansi, entry: dict[str, Any], severity: str, displayed: str, named: str
    ) -> None:
        assert severity_of(unwrap_envelope(entry)) == severity

        screen = ansi(lambda c: print_extracted_files_table([entry], console=c))
        assert f"{_ANSI[colour_of(severity)]}{displayed}" in screen
        assert entry["filename"] in screen

        result = self._sarif([entry])
        assert result["level"] == sarif_level_of(severity)
        # The rule id is the severity's, not the family's — the family
        # name is free text and belongs in the message. See
        # tests/test_sarif.py::TestARuleIdIsStableAndOursAlone.
        assert result["ruleId"] == sarif_rule_of(severity)
        assert named in result["message"]["text"]
        location = result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        assert location == f"sample://{entry['filename']}"


class TestSeverityOf:
    def test_the_four_severities(self):
        assert severity_of({"classification": "malicious"}) == "malicious"
        assert severity_of({"classification": "suspicious"}) == "suspicious"
        assert severity_of({"classification": "clean"}) == "known"
        assert severity_of({"classification": "unknown"}) == "unknown"

    def test_case_and_padding_are_folded(self):
        assert severity_of({"classification": "Malicious"}) == "malicious"
        assert severity_of({"status": " KNOWN "}) == "known"

    def test_an_absent_or_unrecognised_verdict_is_unknown_not_known(self):
        assert severity_of({"sha1": "a"}) == "unknown"
        assert severity_of({"classification": "weird"}) == "unknown"
        assert severity_of({"classification": "unknown"}) == "unknown"


# --- One severity, four presentations ----------------------------------------
#
# The verdict→severity fold lives here, so the check that every site
# presents the same severity consistently lives here too rather than in one
# of the three presentation modules' test files.

_ANSI = {"red": "\x1b[31m", "green": "\x1b[32m", "yellow": "\x1b[33m", "white": "\x1b[37m"}

# verdict -> (severity, Rich colour, TitaniumCloud dot, SARIF level)
_AGREEMENT = [
    ("malicious", "malicious", "red", "🔴", "error"),
    ("suspicious", "suspicious", "yellow", "🟡", "warning"),
    ("clean", "known", "green", "🟢", "note"),
    ("goodware", "known", "green", "🟢", "note"),
    ("known", "known", "green", "🟢", "note"),
    ("unknown", "unknown", "white", "⚪", "none"),
    (None, "unknown", "white", "⚪", "none"),
]
_AGREEMENT_IDS = [verdict or "absent" for verdict, *_ in _AGREEMENT]


def _payload(verdict: str | None) -> dict[str, Any]:
    return {"sha256": "a" * 64} if verdict is None else {"sha256": "a" * 64, "status": verdict}


@pytest.mark.parametrize(
    ("verdict", "severity", "colour", "dot", "level"), _AGREEMENT, ids=_AGREEMENT_IDS
)
def test_every_site_presents_the_same_severity(
    ansi, verdict: str | None, severity: str, colour: str, dot: str, level: str
) -> None:
    payload = _payload(verdict)
    wrong_colours = [code for name, code in _ANSI.items() if name != colour]

    assert severity_of(payload) == severity

    # Each site stands in its own placeholder for a verdict that is absent.
    # The samples panel spells that placeholder "N/A", which it also uses for
    # every other missing field, so only its positive colour is checkable.
    for render, displayed, unique in (
        (lambda c: print_samples_panels([payload], console=c), verdict or "N/A", bool(verdict)),
        (lambda c: print_search_results_table([payload], console=c), verdict or "unknown", True),
        (lambda c: print_report_summary(payload, console=c), verdict or "N/A", True),
    ):
        output = ansi(render)
        assert f"{_ANSI[colour]}{displayed}" in output
        # Nothing may reach the verdict but its own colour — a cell left
        # unmarked inherits its column's style, which painted red.
        assert not unique or not any(f"{other}{displayed}" in output for other in wrong_colours)

    assert dot in ansi(lambda c: print_file_analysis(payload, console=c))

    assert to_sarif(payload)["runs"][0]["results"][0]["level"] == level


class TestThreatLevelOf:
    """The panel that read only the flat key printed N/A for every sample."""

    def test_the_flat_key_wins(self):
        assert threat_level_of({"threat_level": 4}) == 4

    def test_the_tc_report_factor_is_the_fallback(self):
        assert threat_level_of(TestFileAnalysisRecord.RECORD) == 5

    def test_a_record_with_neither_is_none(self):
        assert threat_level_of({"sha1": "a"}) is None

    def test_a_zero_level_is_not_mistaken_for_absent(self):
        assert threat_level_of({"threat_level": 0}) == 0


class _CountingPayload(dict[str, Any]):
    """A payload that records every key looked up through it.

    ``SampleVerdict`` exists because four readers each re-walked the same
    record for one answer each; nothing but a count keeps that from
    creeping back in the next renderer.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__(data)
        self.reads: list[str] = []

    def get(self, key: Any, default: Any = None) -> Any:
        self.reads.append(key)
        return super().get(key, default)

    def __getitem__(self, key: Any) -> Any:
        self.reads.append(key)
        return super().__getitem__(key)

    def walks(self) -> int:
        """How many times the verdict was read out of this payload.

        ``SampleVerdict.of`` reads ``classification`` exactly once and is
        the only thing that reads it, so counting that key counts walks.
        """
        return self.reads.count("classification")


class TestOneWalkPerRow:
    """A row costs one walk of its payload, not one per field asked for.

    ``print_extracted_files_table`` used to call ``severity_of`` from
    inside its sort key and ``severity_of`` plus ``classification_of``
    again while drawing the row — three walks per row per comparison, up
    to nine for a row that also carried a threat name.
    """

    RECORD: ClassVar[dict[str, Any]] = {
        "classification": "malicious",
        "classification_result": "Win64.Ransomware.Conti",
        "threat_level": 5,
        "riskscore": 10,
        "sha256": "c" * 64,
        "type_display": "PE+ executable",
        "file_size": 4096,
    }

    def test_the_verdict_reads_each_key_at_most_once(self):
        payload = _CountingPayload(self.RECORD)
        verdict = SampleVerdict.of(payload)

        assert verdict.classification == "malicious"
        assert verdict.severity == "malicious"
        assert verdict.threat_name == "Win64.Ransomware.Conti"
        assert verdict.threat_level == 5
        assert sorted(payload.reads) == sorted(set(payload.reads))

    def test_an_extracted_file_row_costs_one_walk(self, console):
        entries = [
            _CountingPayload({"filename": f"stage{index}.dll"} | self.RECORD) for index in range(5)
        ]
        # ``list`` is invariant, and a ``list[_CountingPayload]`` is not a
        # ``list[dict[str, Any]]`` to mypy even though the renderer only reads
        # it. Cast rather than copy: the entries handed over are the ones the
        # walk count is read from.
        print_extracted_files_table(cast("list[dict[str, Any]]", entries), console=console)
        # One each, including the sort: the rows are graded before they are
        # ordered, not while they are being compared.
        assert [entry.walks() for entry in entries] == [1] * len(entries)

    def test_a_nested_extracted_row_is_graded_once(self, console, monkeypatch):
        """An entry that carries its record costs the same single grading.

        The counting payload cannot see this one: ``unwrap_envelope`` reads
        the two halves of an extracted-file entry into one record before
        anything grades it, so the reads land on the record rather than on
        either half. What is worth pinning is the count of gradings — the
        rule that resolves a verdict collision grades both halves, and a
        row whose halves cannot disagree must not pay for that comparison.
        """
        gradings = []
        grade = SampleVerdict.of

        def counted(payload):
            gradings.append(payload)
            return grade(payload)

        monkeypatch.setattr(SampleVerdict, "of", counted)
        print_extracted_files_table(
            [{"filename": "stage2.dll", "sample": dict(self.RECORD)}], console=console
        )
        assert len(gradings) == 1

    def test_a_search_row_costs_one_walk(self, console):
        entry = _CountingPayload(self.RECORD)
        print_search_results_table([entry], console=console)
        assert entry.walks() == 1

    def test_a_sample_panel_costs_one_walk(self, console):
        entry = _CountingPayload(self.RECORD)
        print_samples_panels([entry], console=console)
        assert entry.walks() == 1

    def test_a_sarif_result_costs_one_walk(self):
        entry = _CountingPayload(self.RECORD)
        to_sarif([entry])
        assert entry.walks() == 1

    def test_the_counter_can_see_a_second_walk(self):
        """The meter above means nothing unless it can see repetition.

        Three separate reader calls are three walks — which is what each of
        the renderers above used to do per row.
        """
        payload = _CountingPayload(self.RECORD)
        classification_of(payload)
        severity_of(payload)
        threat_name_of(payload)
        assert payload.walks() == 3

    def test_the_free_readers_still_answer_the_same(self):
        """They are public and tested; they now stand on the same walk."""
        assert classification_of(self.RECORD) == "malicious"
        assert severity_of(self.RECORD) == "malicious"
        assert threat_name_of(self.RECORD) == "Win64.Ransomware.Conti"
        assert threat_level_of(self.RECORD) == 5


class TestAHalfOfTheRatioNothingStatesIsNone:
    """The ratio never states a number the record did not.

    Each half is read from what the payload says: a summary stating only
    ``scanner_count`` knows no match count, and a ``malware_presence``
    stating only ``scanner_match`` knows no engine count. Filling either in
    from the other reports "0/17" — no engine flagged this file — or
    "13/13" — every engine that ran did — as if the record had said so.
    """

    def test_a_stated_engine_count_alone_leaves_the_matches_unknown(self):
        consensus = ScannerConsensus.of({"av_scanners_summary": {"scanner_count": 17}})
        assert consensus.detected is None
        assert consensus.total == 17

    def test_a_stated_match_count_alone_leaves_the_engines_unknown(self):
        consensus = ScannerConsensus.of({"status": "MALICIOUS", "scanner_match": 13})
        assert consensus.detected == 13
        assert consensus.total is None

    def test_a_counted_array_states_both_halves(self):
        """An array that was there to count is a stated engine count."""
        consensus = ScannerConsensus.of(
            {
                "av_scanners": [
                    {
                        "regular_scanners": [
                            {"name": "AV1", "result": "Win32.Conti"},
                            {"name": "AV2", "result": ""},
                        ]
                    }
                ]
            }
        )
        assert (consensus.detected, consensus.total) == (1, 2)


class TestTheRatioCountsSignatureEnginesOnly:
    """One sample reports one ratio, whichever response shape carried it.

    A next-gen engine answers with an ML confidence verdict, and folding
    those into the ratio answered 5/15 for the arrays and 2/10 for the
    summary beside them — the same record, differing only in which shape
    arrived — with the ML verdicts counted twice over: once inside the ratio
    and again on the "Next-Gen (ML)" line under it.
    """

    ARRAYS: ClassVar[list[dict[str, Any]]] = [
        {
            "regular_scanners": [
                {"name": f"AV{index}", "result": "Trojan.X" if index < 2 else ""}
                for index in range(10)
            ],
            "nextgen_scanners": [
                {"name": f"NG{index}", "result": "malicious" if index < 3 else ""}
                for index in range(5)
            ],
        }
    ]

    def test_the_arrays_alone_report_the_signature_ratio(self):
        consensus = ScannerConsensus.of({"av_scanners": self.ARRAYS})
        assert (consensus.detected, consensus.total) == (2, 10)
        assert (consensus.nextgen_detected, consensus.nextgen_total) == (3, 5)

    def test_a_summary_of_the_same_numbers_reports_the_same_ratio(self):
        """The invariant the fold broke: the shape must not move the ratio."""
        counted = ScannerConsensus.of({"av_scanners": self.ARRAYS})
        stated = ScannerConsensus.of(
            {
                "av_scanners": self.ARRAYS,
                "av_scanners_summary": {"scanner_count": 10, "scanner_match_count": 2},
            }
        )
        assert (counted.detected, counted.total) == (stated.detected, stated.total)

    def test_an_ml_verdict_is_not_a_signature_hit_in_either_half(self):
        """Three ML verdicts and no signature array is a 0/0 signature ratio."""
        consensus = ScannerConsensus.of(
            {"av_scanners": [{"nextgen_scanners": [{"name": "NG1", "result": "malicious"}]}]}
        )
        assert (consensus.detected, consensus.total) == (0, 0)
        assert consensus.nextgen_detected == 1
        # Engines ran, so the panel the ratio heads still has a verdict to
        # draw: ``total == 0`` alone would drop it and the detection with it.
        assert not consensus.unreported

    def test_no_engine_at_all_is_unreported(self):
        assert ScannerConsensus.of({"scanner_count": 0}).unreported
        assert ScannerConsensus.of({}).unreported


class TestAnUnreadableEngineCountStatesNoCount:
    """A number we cannot read is not a zero, and zero collapses the ratio.

    ``total`` never falls under the matches beside it, so an engine count
    read as zero comes back *as* the match count: thirteen detections
    reported as "13/13", every engine that ran flagged this file, on a
    record that never said how many ran. ``""`` for an unset numeric field
    is an ordinary API idiom, and ``Infinity`` reaches a payload because
    ``requests``' ``.json()`` accepts it.
    """

    @staticmethod
    def _ratio(stated_engines):
        consensus = ScannerConsensus.of(
            {"status": "MALICIOUS", "scanner_count": stated_engines, "scanner_match": 13}
        )
        return consensus.detected, consensus.total

    @pytest.mark.parametrize(
        "stated_engines",
        ["", "forty", float("inf"), float("nan"), None],
        ids=["empty", "word", "infinity", "nan", "unstated"],
    )
    def test_an_unreadable_engine_count_leaves_the_engines_unknown(self, stated_engines):
        assert self._ratio(stated_engines) == (13, None)

    def test_a_stated_zero_is_still_a_number(self):
        """The strict read must not turn a genuine zero into "unstated".

        A record stating zero engines beside zero matches is stating
        ``0/0``; reporting either half as unknown loses a fact it gave us.
        """
        consensus = ScannerConsensus.of({"scanner_count": 0, "scanner_match": 0})
        assert (consensus.detected, consensus.total) == (0, 0)

    def test_a_stated_zero_never_falls_under_the_matches_beside_it(self):
        """A stated number enters the ratio, and the ratio's own rule clamps
        a self-contradictory zero up to the matches it sits beside."""
        assert self._ratio(0) == (13, 13)

    def test_a_stated_zero_match_count_stays_zero(self):
        """Nothing detected is what a clean sample's summary states."""
        consensus = ScannerConsensus.of(
            {"av_scanners_summary": {"scanner_count": 17, "scanner_match": 0}}
        )
        assert (consensus.detected, consensus.total) == (0, 17)


class TestANonFiniteNumberDoesNotAbortTheReport:
    """``int(float("inf"))`` raises ``OverflowError``, which ``count`` let out.

    ``requests``' ``.json()`` accepts a bare ``Infinity``, so a hostile
    record can carry one anywhere a count is read. The analyst got the
    success line, half a report with the AV consensus panel silently
    missing, then "✗ Unexpected error: cannot convert float infinity to
    integer" naming no field, and exit 1.
    """

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
    def test_a_non_finite_count_reads_as_zero(self, value):
        assert count(value) == 0


class TestStatedCountReportsOnlyANumberItWasTold:
    """Strict where ``count`` is lenient: no number becomes ``None``, not zero."""

    def test_a_bool_states_no_count(self):
        # bool is an int subclass, so float(True) is 1.0; the report must not
        # read a flag as the number one.
        assert stated_count(True) is None
        assert stated_count(False) is None

    @pytest.mark.parametrize("value", ["", "word", "Infinity", "NaN", None])
    def test_a_non_number_states_no_count(self, value):
        assert stated_count(value) is None

    def test_an_integer_past_what_a_float_holds_states_no_count(self):
        """``.json()`` decodes an arbitrary-precision int; ``float`` overflows.

        ``count`` caught this and ``stated_count`` did not, so a ~309-digit
        scanner count took the whole report down with an ``OverflowError``
        instead of reading as the unstated count it is.
        """
        assert stated_count(int("9" * 400)) is None

    def test_a_real_number_is_carried(self):
        assert stated_count("5") == 5
        assert stated_count(7) == 7

    def test_a_file_analysis_still_draws_what_follows_the_ratio(self, console):
        """The consensus panel is where it raised, mid-report."""
        payload = {
            "sha1": "a" * 40,
            "status": "MALICIOUS",
            "scanner_match": float("inf"),
            "sources": [{"domain": "conti.example"}],
        }
        print_file_analysis(payload, console=console)
        assert "conti.example" in console.file.getvalue()

    def test_a_report_summary_still_draws(self, console):
        payload = {"sha1": "a" * 40, "classification": "malicious", "scanner_match": float("inf")}
        print_report_summary(payload, console=console)
        assert "malicious" in console.file.getvalue().lower()

    def test_an_extracted_files_table_still_sorts_and_draws(self, console):
        """The sort key is ``-count(riskscore)``, so the row is read for it."""
        print_extracted_files_table(
            [{"filename": "stage2.dll", "riskscore": float("inf")}, {"filename": "b.dll"}],
            console=console,
        )
        assert "stage2.dll" in console.file.getvalue()

    def test_a_reanalyze_results_table_still_draws(self, console):
        """The entry's own HTTP status is read through ``count``."""
        drawn = print_reanalyze_results_table(
            [{"detail": {"sha1": "a" * 40}, "code": float("inf")}], console=console
        )
        assert drawn == 1
        assert "Reanalysis Results" in console.file.getvalue()


class TestAnEntryWeCouldNotReadRefusedNothing:
    """ "Refused: unreadable entry" put our parse failure in the appliance's mouth.

    ``refusal`` is the appliance turning a submission down, which is why the
    renderer draws it as "Refused:". A non-record entry is ours to fail on,
    and the count above the table needs one fact from it: it was not
    accepted.
    """

    @pytest.mark.parametrize("entry", ["nonsense", None, 7, ["a"]])
    def test_a_non_record_is_not_accepted_and_names_no_refusal(self, entry):
        outcome = ReanalysisOutcome.of(entry)
        assert outcome.accepted is False
        assert outcome.refusal is None

    def test_a_refusal_the_appliance_stated_is_still_carried(self):
        outcome = ReanalysisOutcome.of({"code": 404, "message": "Sample not found"})
        assert outcome.refusal == "404: Sample not found"


class TestTheReanalysisBoundariesAndLogicAreExact:
    """Mutation-pinned: 400 is the first rejecting code, a failed engine
    keeps its own message, acceptance reads queued-or-not-failed under no
    refusal, and a stated word is dropped whenever a refusal is set."""

    def test_the_first_rejecting_code_is_400_not_401(self):
        """``code >= 400`` refuses; 399 is not a rejection."""
        assert ReanalysisOutcome.of({"code": 400}).refusal == "400: rejected"
        assert ReanalysisOutcome.of({"code": 400}).accepted is False
        assert ReanalysisOutcome.of({"code": 399}).refusal is None
        assert ReanalysisOutcome.of({"code": 399}).accepted is True

    def test_a_per_engine_code_of_400_is_a_failure_not_a_queue(self):
        at = ReanalysisOutcome.of({"analysis": [{"name": "AV", "code": 400, "message": "quota"}]})
        assert at.failed == ("AV (400): quota",)
        assert at.queued == ()

        under = ReanalysisOutcome.of({"analysis": [{"name": "AV", "code": 399}]})
        assert under.queued == ("AV",)
        assert under.failed == ()

    def test_a_failed_engine_carries_its_own_message_not_the_fallback(self):
        """The fallback word is for an engine that stated none; a stated
        message is the one drawn."""
        outcome = ReanalysisOutcome.of(
            {"analysis": [{"name": "AV", "code": 404, "message": "Sample not found"}]}
        )
        assert outcome.failed == ("AV (404): Sample not found",)

    def test_some_queued_beside_some_failed_is_still_accepted(self):
        """One engine taking the sample is an acceptance even if another
        refused it, as long as the appliance stated no refusal of its own."""
        outcome = ReanalysisOutcome.of(
            {"analysis": [{"name": "q", "code": 201}, {"name": "f", "code": 404}]}
        )
        assert outcome.accepted is True
        assert outcome.queued == ("q",)

    def test_a_stated_refusal_is_never_accepted_even_with_a_queued_engine(self):
        """The entry's own refusal outranks a queued engine under it."""
        outcome = ReanalysisOutcome.of({"code": 400, "analysis": [{"name": "q", "code": 201}]})
        assert outcome.accepted is False

    def test_a_stated_word_is_dropped_whenever_a_refusal_is_set(self):
        """A refusal is reported once, as the refusal; the appliance's status
        word for it is not carried alongside."""
        outcome = ReanalysisOutcome.of({"code": 400, "status": "REFUSED"})
        assert outcome.refusal == "400: rejected"
        assert outcome.stated is None


class TestNothingInThisModuleIsEscaped:
    """The stated contract, which one reader broke: ``_scanner_section``.

    ``ScannerDetection`` reached a renderer already sanitized while the
    module said nothing here is, so the renderer escaped what was escaped
    and a maintainer restoring consistency by deleting the odd ``sanitize``
    would have put ESC back on a terminal with the docstring's blessing.
    These values reach a terminal, a SARIF log and ``-o json`` alike, and
    the layer that draws one is the layer that makes it safe.
    """

    HOSTILE = "\x1b[2JWin32.Conti"

    def _detection(self, entry):
        arrays = {"av_scanners": [{"regular_scanners": [entry]}]}
        return ScannerConsensus.of(arrays).detections[0]

    def test_a_scanner_name_is_carried_as_the_appliance_stated_it(self):
        assert self._detection({"name": self.HOSTILE, "result": "x"}).name == self.HOSTILE

    def test_a_scanner_result_is_carried_as_the_appliance_stated_it(self):
        assert self._detection({"name": "AV1", "result": self.HOSTILE}).result == self.HOSTILE

    def test_an_unnamed_engine_states_no_name_for_the_renderer_to_choose(self):
        """A word like "Unknown" is one a report says, not one a payload states."""
        assert self._detection({"result": "Win32.Conti"}).name is None

    def test_an_engine_that_stated_no_result_is_still_not_a_detection(self):
        consensus = ScannerConsensus.of(
            {"av_scanners": [{"regular_scanners": [{"name": "AV1", "result": ""}]}]}
        )
        assert consensus.detections == ()
        assert (consensus.detected, consensus.total) == (0, 1)


class TestAnEngineThatSaidNothingReadableFlaggedNothing:
    """A result made only of control characters is not a detection.

    ``result`` is carried as the appliance stated it, so the emptiness test
    beside it was ``str(result).strip()`` — and ``.strip()`` removes only
    whitespace. An engine answering ``"\\u202e"`` or ``"\\x00"`` therefore
    counted as a hit, and a record stating ``scanner_count: 17,
    scanner_match_count: 0`` came back ``1/17`` with a detection in it whose
    text no analyst can read. This is a malware tool reporting a detection
    that does not exist.

    The question the test is asking is about the data — does this string say
    anything at all — and not about how to print it: what the module hands
    back is still the appliance's own bytes, which
    :class:`TestNothingInThisModuleIsEscaped` above pins.
    """

    # Drawn from what is invisible on a screen, NOT from what ``sanitize``
    # strips. The first spelling of this list was the latter, so it agreed
    # with the check by construction and went on passing while every
    # zero-width and format character below still counted as a hit.
    #
    # An escape sequence does not belong here: ``"\x1b[2J"`` leaves "[2J"
    # behind, which is text the appliance stated and an analyst can read.
    UNREADABLE: ClassVar[tuple[str, ...]] = (
        "‮",  # RLO — a bidi override
        "‏",  # RLM — a bidi mark, which sanitize does not strip
        "‎",  # LRM
        "​",  # zero-width space
        "﻿",  # BOM / zero-width no-break space
        "⁠",  # word joiner
        "­",  # soft hyphen
        "؜",  # Arabic letter mark
        "᠎",  # Mongolian vowel separator
        "️",  # variation selector, a combining mark with no base
        "́",  # combining acute, likewise
        "\x00",
        "\x0b\x0c",
        "\x7f\x9f",
        "⁦⁩",
        "   ",
    )

    # Threat names an engine really does answer. The check must not grow so
    # broad that one of these stops counting — a malware tool that drops a
    # real detection is worse than one that invents a fake.
    READABLE: ClassVar[tuple[str, ...]] = (
        "Win32.Conti",
        "木马",
        "Троян",
        "מזיק",
        "Troján",
        "\x85Trojan\x85",
        "[2J",
        "0",
    )

    @staticmethod
    def _consensus(result: str) -> ScannerConsensus:
        return ScannerConsensus.of(
            {
                "av_scanners_summary": {"scanner_count": 17, "scanner_match_count": 0},
                "av_scanners": [{"regular_scanners": [{"name": "AV1", "result": result}]}],
            }
        )

    @pytest.mark.parametrize("result", UNREADABLE)
    def test_an_unreadable_result_is_not_listed_as_a_detection(self, result):
        assert self._consensus(result).detections == ()

    @pytest.mark.parametrize("result", UNREADABLE)
    def test_an_unreadable_result_does_not_raise_the_ratio(self, result):
        consensus = self._consensus(result)
        assert (consensus.detected, consensus.total) == (0, 17)

    @pytest.mark.parametrize("result", UNREADABLE)
    def test_an_unreadable_result_leaves_the_clean_grade_alone(self, result):
        """Green is the report saying "no engine flagged this file"."""
        assert self._consensus(result).consensus_severity == "known"

    def test_a_readable_result_wearing_a_control_character_is_still_a_detection(self):
        """The characters are noise around the name, not the whole of it."""
        consensus = self._consensus("\x1b[2JWin32.Conti‮")
        assert (consensus.detected, consensus.total) == (1, 17)
        assert consensus.detections[0].result == "\x1b[2JWin32.Conti‮"

    @pytest.mark.parametrize("result", READABLE)
    def test_a_result_an_analyst_could_read_is_still_a_detection(self, result):
        """The other direction, which matters more.

        Widening what counts as "nothing readable" until a real threat name
        falls into it would make this tool under-report a detection, which
        is the worse of the two failures.
        """
        consensus = self._consensus(result)
        assert (consensus.detected, consensus.total) == (1, 17)
        assert consensus.consensus_severity == "suspicious"


class TestTheRatioGradesItselfOnTheModel:
    """Whether a ratio is a consensus is a judgement about the data.

    It lived in one of the two renderers, so the other painted nothing at
    all: the same 13 of 17 was red in ``a1000 report`` and white in
    ``ticloud file-analysis``.
    """

    @staticmethod
    def _consensus(detected, total):
        return ScannerConsensus.of({"scanner_match": detected, "scanner_count": total})

    def test_a_majority_of_the_engines_is_a_malicious_consensus(self):
        assert self._consensus(9, 14).consensus_severity == "malicious"

    def test_a_minority_is_no_consensus_either_way(self):
        assert self._consensus(3, 14).consensus_severity == "suspicious"

    def test_a_stated_zero_with_nothing_listed_is_the_clean_answer(self):
        assert self._consensus(0, 14).consensus_severity == "known"

    def test_a_half_nothing_states_is_never_graded_clean(self):
        """Green over an unknown numerator reports a sample clean over a
        count nobody gave."""
        assert ScannerConsensus.of({"scanner_count": 17}).consensus_severity != "known"

    def test_a_listed_detection_outranks_a_summary_that_states_none(self):
        """A detections table under a green ratio is the report contradicting
        itself, and the reassuring half is the one read first."""
        consensus = ScannerConsensus.of(
            {
                "av_scanners_summary": {"scanner_count": 40, "scanner_match_count": 0},
                "av_scanners": [{"nextgen_scanners": [{"name": "NG", "result": "malicious"}]}],
            }
        )
        assert consensus.consensus_severity != "known"


class TestTheConsensusBoundariesAreExact:
    """Mutation-pinned: the majority cutoff, the "engines ran" test and the
    two independent unreported conditions each read the value they name."""

    def test_exactly_half_detected_is_not_yet_a_majority(self):
        """``detected > total/2`` is a strict majority: 5 of 10 is suspicious."""
        half = ScannerConsensus.of(
            {"av_scanners_summary": {"scanner_count": 10, "scanner_match_count": 5}}
        )
        assert half.consensus_severity == "suspicious"

        over = ScannerConsensus.of(
            {"av_scanners_summary": {"scanner_count": 10, "scanner_match_count": 6}}
        )
        assert over.consensus_severity == "malicious"

    def test_a_stated_match_count_alone_is_not_an_unreported_record(self):
        """13 detections with no stated total still ran engines, so the panel
        it heads is drawn, not dropped as unreported."""
        consensus = ScannerConsensus.of({"status": "MALICIOUS", "scanner_match": 13})
        assert consensus.detected == 13
        assert not consensus.unreported

    def test_engines_that_ran_keep_a_ratio_even_when_the_two_halves_are_equal(self):
        """The engine count is the two arrays summed; a difference of them
        would read equal halves as no engines and blank the ratio."""
        arrays = [
            {
                "regular_scanners": [
                    {"name": f"AV{index}", "result": "X" if index < 1 else ""} for index in range(3)
                ],
                "nextgen_scanners": [
                    {"name": f"NG{index}", "result": "malicious" if index < 2 else ""}
                    for index in range(3)
                ],
            }
        ]
        consensus = ScannerConsensus.of({"av_scanners": arrays})
        assert (consensus.detected, consensus.total) == (1, 3)


class TestSearchPageReadsEitherSpellingOfMore:
    """Mutation-pinned: the entries list is carried through, and either
    ``more_pages`` or ``next_page`` on its own says the page is not the last."""

    @staticmethod
    def _api(body):
        return {"rl": {"web_search_api": body}}

    def test_the_entries_list_is_carried(self):
        page = search_page(self._api({"entries": [{"sha1": "a"}, {"sha1": "b"}]}))
        assert page is not None
        assert page.entries == [{"sha1": "a"}, {"sha1": "b"}]

    def test_more_pages_alone_says_there_is_more(self):
        page = search_page(self._api({"more_pages": True}))
        assert page is not None and page.more_pages is True

    def test_next_page_alone_says_there_is_more(self):
        page = search_page(self._api({"next_page": "/page/2"}))
        assert page is not None and page.more_pages is True

    def test_neither_spelling_is_the_last_page(self):
        page = search_page(self._api({"entries": []}))
        assert page is not None and page.more_pages is False
