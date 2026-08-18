"""Rendering tests for the rl_cli.render.formatters package.

These drive the renderers over the payload shapes the A1000 and
TitaniumCloud endpoints really answer with and assert on what reaches the
console: the value a panel reads out of a nested record, the width a cell
is cut to, the colour a verdict is painted, the order rows are ranked in.
That is the contract the CLI commands rely on since the presentation layer
was extracted out of ``cli/commands/a1000.py`` — the commands assert on
their own output, not on these functions.

A few tests here still only render and assert nothing: they guard payloads
whose regression was a crash — an explicit ``null`` where a string
belonged, a bare string where a record belonged, a field of the wrong type
— and each names that in its docstring.
"""

from __future__ import annotations

import ast
import inspect
import io
import re
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from rich.console import Console
from rich.table import Table

from rl_cli.cli.commands import config as config_commands
from rl_cli.cli.context import display
from rl_cli.models.payload import ReanalysisOutcome, SampleFacts
from rl_cli.render.formatters import (
    print_analysis_status,
    print_extracted_files_table,
    print_file_analysis,
    print_reanalyze_results_table,
    print_report_summary,
    print_samples_panels,
    print_search_results_table,
    print_summary_panel,
    print_titanium_report,
    print_yara_content,
    print_yara_rulesets_table,
)
from rl_cli.render.formatters.config_report import (
    print_availability_panel,
    print_profile_names,
)
from rl_cli.render.formatters.panels import add_capped_rows, print_file_information
from rl_cli.render.formatters.severity import colour_of, rank_of, style_of
from rl_cli.render.output import OutputFormat, OutputFormatter, RichOutput
from rl_cli.render.sarif import sarif_level_of, sarif_rule_of, to_sarif
from rl_cli.services.a1000.samples import ReanalysisBatch
from rl_cli.text import DIGEST_CELL_WIDTH, digest_cell
from tests.conftest import console_text

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "rl_cli"


@pytest.fixture
def console() -> Console:
    """A Console wired to an in-memory buffer so tests stay quiet."""
    return Console(file=io.StringIO(), width=120, record=True)


# ---------- Sanity payloads ----------

_FULL_REPORT: dict[str, Any] = {
    "results": [
        {
            "md5": "d41d8cd98f00b204e9800998ecf8427e",
            "sha1": "da39a3ee5e6b4b0d3255bfef95601890afd80709",
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "file_type": "PE+ executable (GUI) x86-64",
            "file_size": 12345,
            "category": "Trojan",
            "riskscore": 9,
            "imphash": "abc123",
            "local_first_seen": "2024-01-01T12:34:56Z",
            "classification": {
                "classification": "malicious",
                "family_name": "Emotet",
                "platform": "Windows",
            },
            "threat_name": "Trojan.Generic",
            "threat_level": "high",
            "trust_factor": "low",
            "av_scanners": [
                {
                    "regular_scanners": [
                        {"name": "AV1", "result": "Trojan.Win32.Generic"},
                        {"name": "AV2", "result": ""},
                    ],
                    "nextgen_scanners": [
                        {"name": "NG1", "result": "Trojan/Generic"},
                    ],
                }
            ],
            "ticore": {
                "behaviour": [{"name": "writes registry"}, "drops file"],
                "indicators": [
                    {"description": "self-deletes", "category": "evasion"},
                    "anti-debug",
                ],
                "signatures": [
                    {"name": "PackerSig", "severity": "high"},
                    "GenericSig",
                ],
            },
            "tags": {
                "ticore": [
                    "capability-network",
                    "indicator-evasion",
                    "protection-aslr",
                    "language-rust",
                ]
            },
        }
    ]
}


# ---------- Tests ----------


_NESTED_CLASSIFICATION = {
    "classification": {"classification": "malicious", "family_name": "Emotet"},
    "riskscore": 10,
    "file_type": "PE32",
}


def test_print_analysis_status_reads_the_nested_classification(console: Console) -> None:
    """A v2 record nests the verdict; the panel must not print the dict repr."""
    print_analysis_status(_NESTED_CLASSIFICATION, console=console)
    output = console_text(console)
    assert "Classification: malicious" in output
    assert "family_name" not in output


def test_print_analysis_status_states_where_and_when_it_was_seen(console: Console) -> None:
    """The provenance rows are why this panel is not the summary one."""
    print_analysis_status(
        {
            "classification": "malicious",
            "riskscore": 7,
            "first_seen": "2024-01-01",
            "last_seen": "2024-02-01",
            "data_source": "TitaniumCore",
        },
        console=console,
    )
    output = console_text(console)
    assert "Risk Score: 7" in output
    assert "First Seen: 2024-01-01" in output
    assert "Last Seen: 2024-02-01" in output
    assert "Data Source: TitaniumCore" in output


def test_print_summary_panel_reads_the_nested_classification(console: Console) -> None:
    """Same record through the summary panel, envelope included."""
    print_summary_panel({"results": [_NESTED_CLASSIFICATION]}, console=console)
    output = console_text(console)
    assert "Classification: malicious" in output
    assert "Threat Name: Emotet" in output
    assert "family_name" not in output


def test_print_summary_panel_reads_a_record_handed_over_without_its_envelope(
    console: Console,
) -> None:
    """A zero risk score is a score: ``.get(key, "N/A")`` reported it as no answer."""
    print_summary_panel(
        {"classification": "clean", "threat_name": "-", "riskscore": 0, "file_type": "ELF"},
        console=console,
    )
    output = console_text(console)
    assert "Classification: clean" in output
    assert "Risk Score: 0" in output
    assert "File Type: ELF" in output


def test_print_report_summary_full_payload(console: Console) -> None:
    print_report_summary(_FULL_REPORT, console=console)


def test_print_report_summary_minimal_payload(console: Console) -> None:
    """Most upstream fields missing; renderer must not raise."""
    print_report_summary({"results": [{}]}, console=console)


def test_print_report_summary_classification_as_string(console: Console) -> None:
    """SDK sometimes returns the classification as a plain string."""
    print_report_summary({"results": [{"classification": "clean"}]}, console=console)


def test_extracted_table_reads_an_entry_that_states_its_own_metadata(
    console: Console,
) -> None:
    """Not every entry nests the metadata under "sample"; the flat shape reads too."""
    print_extracted_files_table(
        [
            {
                "filename": "evil.dll",
                "sha256": "a" * 64,
                "file_type": "PE32",
                "file_size": 4096,
            }
        ],
        console=console,
    )
    output = console_text(console)
    assert "evil.dll" in output
    assert "PE32" in output
    assert "4,096 bytes" in output


def test_search_table_reads_both_spellings_of_type_and_size(console: Console) -> None:
    """``file_type``/``file_size`` on one entry, ``sample_type``/``sample_size`` on the next."""
    print_search_results_table(
        [
            {
                "sha256": "a" * 64,
                "file_type": "PE32",
                "classification": "malicious",
                "threat_name": "Emotet",
                "file_size": 1024,
            },
            {
                "sha256": "b" * 64,
                "sample_type": "ELF",
                "classification": {"classification": "suspicious"},
                "threat_name": None,
                "sample_size": 2048,
            },
        ],
        console=console,
    )
    output = console_text(console)
    assert "PE32" in output and "1,024" in output
    assert "ELF" in output and "2,048" in output
    # The nested verdict is read the same way here as in the panels, and
    # the entry that states no threat name gets the placeholder.
    assert "suspicious" in output
    assert "Emotet" in output


def test_search_table_renders_advanced_search_v3_entries(console: Console) -> None:
    """v3 entries capitalise classification and carry no threat_name/sha256."""
    print_search_results_table(
        [
            {
                "sha256": "a" * 64,
                "sample_type": "PE+/Exe",
                "classification": "Malicious",
                "classification_result": "Win64.Malware.Heuristic",
                "sample_size": 3627912,
            },
            {
                "sha1": "b" * 40,
                "sample_type": "PE+/Exe",
                "classification": "Malicious",
                "untokenized_threat_name": "Binary.Ransomware.Generic",
                "sample_size": 14333199,
            },
        ],
        console=console,
    )
    output = console_text(console)
    # Both names are longer than the column, and the cut is now marked.
    assert "Win64.Malware.Heu..." in output
    assert "Binary.Ransomware..." in output
    # The digest is cut to the same width as its neighbours, ellipsis
    # included: it used to spend the ellipsis on top of the width and draw
    # three characters wider than the column beside it.
    assert "b" * (DIGEST_CELL_WIDTH - 3) + "..." in output
    assert "b" * (DIGEST_CELL_WIDTH - 2) not in output
    assert "N/A..." not in output


def test_a_truncated_threat_name_says_so(console: Console) -> None:
    """The SHA256 column three lines above suffixes its own truncation, which
    trains the reader that an unsuffixed cell is complete."""
    print_search_results_table(
        [{"sha256": "a" * 64, "threat_name": "Win64.Ransomware.ContiV3"}], console=console
    )
    output = console_text(console)
    assert "Win64.Ransomware.Con" not in output
    assert "Win64.Ransomware..." in output


def test_an_absent_size_is_not_reported_as_a_zero_byte_file(console: Console) -> None:
    """A bare 0 under a unit-less header reads as a real, empty file."""
    print_search_results_table([{"sha256": "a" * 64}], console=console)
    output = console_text(console)
    assert "Size (bytes)" in output
    assert "0" not in output.replace("Size (bytes)", "")


def test_a_ruleset_that_states_no_status_is_read_from_enabled(console: Console) -> None:
    """The appliance reports ``status``; the older shape reports ``enabled``."""
    print_yara_rulesets_table(
        [
            {"name": "Apt29", "enabled": True, "rule_count": 12, "modified": "2024-01-01"},
            {"name": "Other", "enabled": False, "rule_count": 0, "modified": "N/A"},
        ],
        console=console,
    )
    output = console_text(console)
    assert "active" in output
    assert "inactive" in output
    # ``modified`` stands in for the "Last Matched" column a ruleset that
    # has never matched does not carry.
    assert "2024-01-01" in output


def test_a_reanalysis_entry_keeps_the_word_the_appliance_used(console: Console) -> None:
    """A stated outcome is not restated as "Submitted"."""
    drawn = print_reanalyze_results_table(
        [{"hash": "a" * 64, "status": "Submitted"}], console=console
    )

    assert drawn == 1
    output = console_text(console)
    assert "Submitted" in output
    assert "aaaaaaaa" in output


def test_tables_count_the_rows_they_drew_not_the_cap(console: Console) -> None:
    """Non-mapping entries are skipped, so the drawn count is short of the cap.

    The reanalysis table is not among these: it draws every entry the batch
    above it was counted from, unreadable ones included, which
    :class:`TestTheTableAccountsForEveryEntryTheCountDid` pins.
    """
    rows = cast(
        list[dict[str, Any]],
        [{"sha256": f"{index:064x}"} for index in range(12)] + ["bare-string"] * 10,
    )

    assert print_search_results_table(rows, max_rows=20, console=console) == 12


@pytest.mark.parametrize(
    ("draw", "cap"),
    [
        (print_extracted_files_table, 20),
        (print_search_results_table, 20),
        (print_samples_panels, 10),
        (print_yara_rulesets_table, 20),
        (print_reanalyze_results_table, 10),
    ],
    ids=["extracted", "search", "panels", "rulesets", "reanalyze"],
)
def test_a_listing_asked_for_no_cap_stops_at_its_own(draw, cap, console: Console) -> None:
    """Each renderer's default is the cap a caller that binds none gets.

    The numbers are written out here rather than read back off the
    constants they pin: a test that imports the limit it asserts passes
    whatever the limit is changed to, which is how five caps came to be
    numbers that no run of the CLI could reach.
    """
    rows = [{"sha256": f"{index:064x}"} for index in range(cap + 5)]

    assert draw(rows, console=console) == cap


@pytest.mark.parametrize(
    "render",
    [
        lambda rows, c: print_search_results_table(rows, max_rows=10, console=c),
        lambda rows, c: print_reanalyze_results_table(rows, max_rows=10, console=c),
    ],
    ids=["search", "reanalyze"],
)
def test_tables_count_stops_at_the_cap(render, console: Console) -> None:
    rows = [{"sha256": f"{index:064x}"} for index in range(15)]

    assert render(rows, console) == 10


def test_print_samples_panels_returns_count(console: Console) -> None:
    rendered = print_samples_panels(
        [
            {
                "file_name": "x.exe",
                "sha256": "a" * 64,
                "sha1": "b" * 40,
                "md5": "c" * 32,
                "threat_status": "malicious",
                "threat_name": "Emotet",
                "file_type": "PE32",
                "file_size": 1024,
            },
            {"file_name": "y.exe", "threat_status": "clean"},
            {"file_name": "z.exe", "threat_status": "unknown"},
        ],
        max_panels=2,
        console=console,
    )
    assert rendered == 2


def test_samples_panels_read_search_entries_and_shorten_them(console: Console) -> None:
    """``list`` hands over whole entries now; the panel does the shortening.

    Shortened through ``_clip``, so the cut is marked: the panel used to
    slice bare, which is what makes two long names that differ only past
    the cap render as the same string.
    """
    print_samples_panels(
        [
            {
                "file_names": ["x" * 60 + ".exe"],
                "sample_type": "PE32 " + "y" * 40,
                "sample_size": 7,
                "classification": "Malicious",
            }
        ],
        console=console,
    )
    text = console.export_text()
    assert "x" * 27 + "..." in text and "x" * 28 not in text
    assert "PE32 " + "y" * 12 + "..." in text and "y" * 13 not in text
    assert "7 bytes" in text


GREEN, RED, YELLOW, WHITE = "\x1b[32m", "\x1b[31m", "\x1b[33m", "\x1b[37m"


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ({"classification": "malicious"}, f"{RED}malicious"),
        ({"classification": "Malicious"}, f"{RED}Malicious"),
        ({"classification": "suspicious"}, f"{YELLOW}suspicious"),
        ({"classification": "unknown"}, f"{WHITE}unknown"),
        ({"file_name": "x.exe"}, f"{WHITE}N/A"),
        ({"classification": "goodware"}, f"{GREEN}goodware"),
        ({"classification": "clean"}, f"{GREEN}clean"),
    ],
    ids=["malicious", "capitalised", "suspicious", "unknown", "absent", "goodware", "clean"],
)
def test_samples_panels_colour_the_verdict(ansi, sample: dict[str, Any], expected: str) -> None:
    """Green must mean "known clean" — never "suspicious" and never "no idea"."""
    output = ansi(lambda c: print_samples_panels([sample], console=c))
    assert expected in output
    assert (GREEN in output) == expected.startswith(GREEN)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"classification": "malicious"}, f"{RED}malicious"),
        ({"classification": "suspicious"}, f"{YELLOW}suspicious"),
        ({"classification": "goodware"}, f"{GREEN}goodware"),
        ({"classification": "clean"}, f"{GREEN}clean"),
    ],
    ids=["malicious", "suspicious", "goodware", "clean"],
)
def test_search_table_colours_the_verdict(ansi, result: dict[str, Any], expected: str) -> None:
    output = ansi(lambda c: print_search_results_table([result], console=c))
    assert expected in output


@pytest.mark.parametrize("result", [{"classification": "unknown"}, {"sha256": "a" * 64}])
def test_search_table_never_paints_a_missing_verdict_green(ansi, result: dict[str, Any]) -> None:
    output = ansi(lambda c: print_search_results_table([result], console=c))
    assert f"{GREEN}unknown" not in output


def test_crypto_tags_are_rendered(console: Console) -> None:
    """crypto-* tags have no panel of their own; they belong in "other"."""
    print_report_summary(
        {"results": [{"tags": {"ticore": ["crypto-aes", "capability-network"]}}]},
        console=console,
    )
    assert "crypto-aes" in console_text(console)


class TestPrintTitaniumReport:
    """The /ticore/ endpoint answers the bare TitaniumCore document."""

    # Section names are the SDK's ticore_fields; the verdict is the numeric
    # TitaniumCore code and the threat name is classification.result.
    PAYLOAD: ClassVar[dict[str, Any]] = {
        "md5": "d41d8cd98f00b204e9800998ecf8427e",
        "sha1": "d" * 40,
        "sha256": "e" * 64,
        "info": {"file": {"file_type": "PE+", "file_subtype": "Exe", "size": 4096}},
        "classification": {
            "classification": 3,
            "factor": 5,
            "result": "Win32.Trojan.Emotet",
        },
        # A section the appliance states as an object, not a list.
        "behaviour": {"summary": "does things"},
        "indicators": [{"description": "Contains a suspicious section name", "priority": 5}],
        "tags": ["capability-network", "protection-aslr"],
        "story": "The file is a Windows executable.",
    }

    def _render(self, payload: dict[str, Any]) -> str:
        console = Console(file=io.StringIO(), width=120)
        print_titanium_report(payload, console=console)
        return console_text(console)

    def test_the_document_is_rendered_not_hinted_at(self):
        rendered = self._render(self.PAYLOAD)
        assert "e" * 64 in rendered
        assert "PE+" in rendered
        assert "4,096 bytes" in rendered
        assert "malicious" in rendered
        assert "Win32.Trojan.Emotet" in rendered
        assert "Contains a suspicious section name" in rendered
        assert "network" in rendered
        assert "ASLR" in rendered
        assert "--json" in rendered

    def test_the_numeric_verdict_is_coloured_by_severity(self):
        console = Console(file=io.StringIO(), width=120, force_terminal=True, no_color=False)
        print_titanium_report(self.PAYLOAD, console=console)
        assert f"{RED}malicious" in console_text(console)

    def test_the_threat_level_comes_from_the_classification_factor(self):
        assert "Threat Level" in self._render(self.PAYLOAD)
        assert "5" in self._render(self.PAYLOAD)

    def test_the_same_document_nested_under_ticore_renders_alike(self):
        assert self._render({"ticore": self.PAYLOAD}) == self._render(self.PAYLOAD)

    def test_an_object_valued_behaviour_section_is_rendered(self):
        """``behaviour`` is an object in the real document, so gating the
        panel on the list spelling skipped it silently every time."""
        rendered = self._render(self.PAYLOAD)
        assert "Behavioral Analysis" in rendered
        assert "does things" in rendered

    def test_the_story_is_rendered(self):
        """TitaniumCore's plain-English narrative — nothing showed it."""
        assert "The file is a Windows executable." in self._render(self.PAYLOAD)

    def test_the_attack_matrix_is_rendered(self):
        rendered = self._render(
            dict(
                self.PAYLOAD,
                attack=[
                    {
                        "matrix": "Enterprise",
                        "tactics": [
                            {
                                "id": "TA0040",
                                "name": "Impact",
                                "techniques": [
                                    {"id": "T1486", "name": "Data Encrypted for Impact"}
                                ],
                            }
                        ],
                    }
                ],
            )
        )
        assert "MITRE ATT&CK" in rendered
        assert "Impact" in rendered
        assert "T1486" in rendered

    def test_signatures_are_read_from_the_certificate_section(self):
        """ "signatures" is not a TitaniumCore section, so the panel gated on
        ticore["signatures"] could never fire."""
        rendered = self._render(
            dict(self.PAYLOAD, certificate={"signatures": [{"identifier": "Acme Corp"}]})
        )
        assert "Matched Signatures" in rendered
        assert "Acme Corp" in rendered


class TestPrintFileAnalysis:
    """TitaniumCloud nests the reputation under rl.malware_presence."""

    def _render(self, payload: dict[str, Any]) -> str:
        console = Console(file=io.StringIO(), force_terminal=False, width=100)
        print_file_analysis(payload, console=console)
        return console_text(console)

    def test_the_threat_name_is_rendered(self):
        rendered = self._render({"status": "malicious", "sha256": "abc", "threat_name": "Evil"})
        assert "Evil" in rendered
        assert "abc" in rendered

    def test_nested_titaniumcloud_payload_is_read_through(self):
        rendered = self._render(
            {
                "rl": {
                    "malware_presence": {
                        "status": "MALICIOUS",
                        "threat_name": "Win32.Trojan.X",
                        "sha1": "a" * 40,
                    }
                }
            }
        )
        assert "🔴" in rendered
        assert "Win32.Trojan.X" in rendered
        assert "a" * 40 in rendered

    def test_uppercase_status_is_recognised(self):
        assert "🟢" in self._render({"rl": {"malware_presence": {"status": "KNOWN"}}})

    def test_flat_lowercase_payload_still_works(self):
        assert "🔴" in self._render({"status": "malicious", "sha256": "c" * 64})

    def test_the_file_analysis_record_is_read_where_its_fields_live(self):
        """RLDATA spells the type and size sample_*, and hides the verdict
        in the first analysis entry: the panel used to hold only hashes,
        under a white dot, with no Threat Assessment at all."""
        rendered = self._render(
            {
                "rl": {
                    "sample": {
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
                }
            }
        )
        assert "🔴" in rendered
        assert "PE/Exe" in rendered
        # Every A1000 table prints a size with its unit; this one printed a
        # bare 3627912.
        assert "4,096 bytes" in rendered
        assert "Threat Assessment" in rendered
        assert "Win32.Trojan.Emotet" in rendered
        # The factor is right there in the tc_report the verdict came from.
        assert "Level: 5" in rendered

    def test_the_permanently_empty_score_row_is_gone(self):
        """ "threat_score" is not a field of malware_presence, so the row it
        filled was structurally always N/A."""
        rendered = self._render(
            {"rl": {"malware_presence": {"status": "MALICIOUS", "threat_name": "X"}}}
        )
        assert "Score" not in rendered

    def test_the_av_consensus_is_reported(self):
        """31 of 37 engines agreeing is the whole value of TCA-0101."""
        rendered = self._render(
            {
                "rl": {
                    "malware_presence": {
                        "status": "MALICIOUS",
                        "threat_name": "Win32.Trojan.X",
                        "threat_level": 5,
                        "scanner_count": 37,
                        "scanner_match": 31,
                        "scanner_percent": 83.78,
                        "reason": "antivirus",
                        "trust_factor": 5,
                        "first_seen": "2024-01-01T00:00:00",
                        "last_seen": "2024-06-01T00:00:00",
                    }
                }
            }
        )
        assert "31/37" in rendered
        assert "83.78" in rendered
        assert "Reason: antivirus" in rendered
        assert "Trust Factor: 5" in rendered
        assert "2024-01-01T00:00:00" in rendered

    def test_the_download_sources_are_rendered(self):
        """The download URL and domain are first-class IOCs, and were dropped."""
        rendered = self._render(
            {
                "rl": {
                    "sample": {
                        "sha1": "a" * 40,
                        "sources": [
                            {
                                "url": "http://evil.tld/payload.exe",
                                "domain": "evil.tld",
                                "record_time": "2024-05-05T00:00:00",
                            }
                        ],
                    }
                }
            }
        )
        assert "Sources" in rendered
        assert "evil.tld" in rendered
        assert "payload.exe" in rendered

    def test_unknown_status_is_not_rendered_as_clean(self):
        """Green for "no idea" reads as a clean verdict."""
        rendered = self._render({"unrecognised": "shape"})
        assert "🟢" not in rendered
        assert "⚪" in rendered


# --- Null-tolerance regression -----------------------------------------------

_NULLABLE_FIELDS = [
    "sha256",
    "sha1",
    "md5",
    "file_type",
    "sample_type",
    "threat_name",
    "file_size",
    "sample_size",
    "classification",
    "threat_status",
    "file_name",
    "filename",
    "name",
    "enabled",
    "rule_count",
    "modified",
    "riskscore",
    "first_seen",
    "last_seen",
    "data_source",
    "risk_score",
    "hash",
    "status",
    "threat_level",
    "trust_factor",
    "av_scanners",
    "ticore",
    "tags",
    "category",
    "local_first_seen",
    "imphash",
]

ALL_NONE = dict.fromkeys(_NULLABLE_FIELDS)


@pytest.mark.parametrize(
    "render",
    [
        lambda c: print_analysis_status(ALL_NONE, console=c),
        lambda c: print_summary_panel(ALL_NONE, console=c),
        lambda c: print_report_summary(ALL_NONE, console=c),
        lambda c: print_titanium_report(ALL_NONE, console=c),
        lambda c: print_extracted_files_table([ALL_NONE], console=c),
        lambda c: print_search_results_table([ALL_NONE], console=c),
        lambda c: print_yara_rulesets_table([ALL_NONE], console=c),
        lambda c: print_reanalyze_results_table([ALL_NONE], console=c),
        lambda c: print_samples_panels([ALL_NONE], console=c),
    ],
    ids=[
        "analysis_status",
        "summary_panel",
        "report_summary",
        "titanium_report",
        "extracted_files",
        "search_results",
        "yara_rulesets",
        "reanalyze_results",
        "samples_panels",
    ],
)
def test_formatters_survive_explicit_nulls(render, console: Console) -> None:
    """The API returns keys with null values; slicing/formatting them must not crash."""
    render(console)


def _rows_with_a_bare_string(row: dict[str, Any]) -> list[dict[str, Any]]:
    """A result array as the appliance can send it: one row is not a mapping.

    The renderers take ``list[dict[str, Any]]`` because that is what the
    documented JSON is; these tests exist because what arrives is not. The
    cast says so once, here, instead of at each call below.
    """
    return cast(list[dict[str, Any]], ["bare-string", row])


@pytest.mark.parametrize(
    "render",
    [
        lambda c: print_extracted_files_table(
            _rows_with_a_bare_string({"filename": "a"}), console=c
        ),
        lambda c: print_search_results_table(
            _rows_with_a_bare_string({"sha256": "a" * 64}), console=c
        ),
        lambda c: print_yara_rulesets_table(_rows_with_a_bare_string({"name": "r"}), console=c),
        lambda c: print_samples_panels(_rows_with_a_bare_string({"file_name": "x"}), console=c),
    ],
    ids=["extracted", "search", "yara", "samples"],
)
def test_table_formatters_skip_non_mapping_rows(render, console: Console) -> None:
    """Result arrays can carry bare strings; those rows are skipped, not fatal.

    The reanalysis table draws them instead of skipping them — its count is
    the one the exit status is read from — so it is pinned separately, by
    :class:`TestTheTableAccountsForEveryEntryTheCountDid`.
    """
    render(console)


def test_extracted_table_reads_metadata_from_the_sample_block(console: Console) -> None:
    """Only filename sits at the top level; the rest lives under "sample"."""
    print_extracted_files_table(
        [
            {
                "filename": "beta.txt",
                "full_path": "beta.txt",
                "sample": {
                    "sha256": "b" * 64,
                    "type_display": "Text/None",
                    "file_size": 38,
                },
            }
        ],
        console=console,
    )
    output = console_text(console)
    assert "bbbbbbbb" in output
    assert "Text/None" in output
    assert "38 bytes" in output
    assert "N/A" not in output


# --- Hostile-input regression ------------------------------------------------
#
# Every string these renderers print - a file name, a threat name, a tag, a
# YARA rule - was written by whoever built the sample. The fixtures below are
# the shapes a fuzzing sweep drove them with.

# An OSC that retitles the window, a CSI that clears the screen, and a
# right-to-left override that makes a dropper read as "invoice.jpg".
HOSTILE = "\x1b]0;rl-cli owned\x07\x1b[2Jinvoice\u202egpj.exe"


def _plain(render) -> str:
    """Render through a console that emits no styling of its own.

    Any ESC left in the output therefore came from the payload.
    """
    console = Console(file=io.StringIO(), width=200)
    render(console)
    return console_text(console)


@pytest.mark.parametrize(
    "render",
    [
        lambda c: print_samples_panels([{"file_names": [HOSTILE]}], console=c),
        lambda c: print_search_results_table([{"threat_name": HOSTILE}], console=c),
        lambda c: print_extracted_files_table([{"filename": HOSTILE}], console=c),
        lambda c: print_yara_rulesets_table([{"name": HOSTILE}], console=c),
        lambda c: print_report_summary({"tags": {"ticore": [f"capability-{HOSTILE}"]}}, console=c),
        lambda c: print_file_analysis({"file_name": HOSTILE, "status": "malicious"}, console=c),
        lambda c: print_yara_content(f'rule X {{ strings: $a = "{HOSTILE}" }}', console=c),
    ],
    ids=["samples", "search", "extracted", "yara_rulesets", "tags", "ticloud", "yara_content"],
)
def test_no_escape_sequence_reaches_the_terminal(render) -> None:
    """An escape sequence in a sample's own metadata must not drive the terminal."""
    output = _plain(render)
    assert "\x1b" not in output
    assert "\u202e" not in output


def test_a_bracket_class_in_a_threat_name_is_still_visible() -> None:
    """Rich reads "[a-z0-9]" as a style tag and drops it; the analyst needs it."""
    output = _plain(
        lambda c: print_search_results_table([{"threat_name": "W32.[a-z0-9]x"}], console=c)
    )
    assert "W32.[a-z0-9]x" in output


@pytest.mark.parametrize(
    ("render", "expected"),
    [
        (
            lambda c: print_extracted_files_table([{"filename": "C:/x/[a-z].dll"}], console=c),
            "[a-z].dll",
        ),
        (
            lambda c: print_yara_rulesets_table([{"name": "rules[abc].yar"}], console=c),
            "rules[abc]",
        ),
        (
            lambda c: print_samples_panels([{"file_name": "inv[bold red]oice.pdf"}], console=c),
            "inv[bold red]oice.pdf",
        ),
        (
            lambda c: print_report_summary(
                {"av_scanners": [{"regular_scanners": [{"name": "AV", "result": "Trojan[a-z]G"}]}]},
                console=c,
            ),
            "Trojan[a-z]G",
        ),
        (
            lambda c: print_report_summary(
                {"tags": {"ticore": ["capability-x[bold]y"]}}, console=c
            ),
            "x[bold]y",
        ),
        (
            lambda c: print_file_analysis({"file_name": "a[bold]b.exe"}, console=c),
            "a[bold]b.exe",
        ),
    ],
    ids=["extracted", "yara_rulesets", "samples", "av_detection", "tags", "ticloud"],
)
def test_markup_in_a_value_is_shown_not_swallowed(render, expected: str) -> None:
    assert expected in _plain(render)


@pytest.mark.parametrize(
    "render",
    [
        lambda c: print_search_results_table([{"threat_name": "evil[/]x"}], console=c),
        lambda c: print_samples_panels([{"file_names": ["a[/cyan]b"]}], console=c),
        lambda c: print_extracted_files_table([{"filename": "a[/cyan]b"}], console=c),
        lambda c: print_file_analysis({"file_name": "a[/cyan]b"}, console=c),
        lambda c: print_report_summary({"tags": {"ticore": ["capability-net[/]work"]}}, console=c),
    ],
    ids=["search", "samples", "extracted", "ticloud", "tags"],
)
def test_an_unbalanced_closing_tag_does_not_abort_the_report(render) -> None:
    """MarkupError reaches main.py as "Unexpected error" - after half a report."""
    assert "[/" in _plain(render)


def test_indicator_priority_and_signature_severity_are_shown() -> None:
    """Both were wrapped in brackets Rich read as a style tag, and deleted."""
    output = _plain(
        lambda c: print_report_summary(
            {
                "ticore": {
                    "indicators": [{"description": "self-deletes", "priority": 7}],
                    "certificate": {"signatures": [{"name": "PackerSig", "severity": "high"}]},
                }
            },
            console=c,
        )
    )
    assert "[7] self-deletes" in output
    assert "[high] PackerSig" in output


def test_an_indicator_description_cannot_recolour_its_line(ansi) -> None:
    """A description that names a colour painted the line the verdict's red."""
    output = ansi(
        lambda c: print_report_summary(
            {"ticore": {"indicators": [{"description": "[red]d", "priority": 1}]}}, console=c
        )
    )
    assert "[1] [red]d" in output
    assert RED not in output


class TestIndicatorPanelRanking:
    """The panel shows ten of however many TitaniumCore reported."""

    # Payload order, with the two that matter last — as the appliance
    # states them.
    INDICATORS = [
        {"description": f"filler {index}", "priority": 1, "category": 22} for index in range(10)
    ] + [
        {"description": "Deletes Volume Shadow Copies (anti-recovery)", "priority": 10},
        {"description": "Encrypts files on all mounted drives", "priority": 10},
    ]

    def _render(self) -> str:
        return _plain(
            lambda c: print_report_summary({"ticore": {"indicators": self.INDICATORS}}, console=c)
        )

    def test_the_worst_indicators_are_not_sliced_off_the_end(self):
        output = self._render()
        assert "Deletes Volume Shadow Copies (anti-recovery)" in output
        assert "Encrypts files on all mounted drives" in output

    def test_filler_is_dropped_to_make_room(self):
        assert self._render().count("filler ") == 8

    def test_the_slice_says_how_much_it_hid(self):
        """The AV table already emits one; this panel cut in silence."""
        assert "and 2 more indicators" in self._render()

    def test_the_numeric_category_code_is_not_offered_as_a_label(self):
        """It rendered as "• [22] …", which says nothing at all."""
        assert "[22]" not in self._render()
        assert "[10] Deletes Volume Shadow Copies" in self._render()


@pytest.mark.parametrize(
    ("render", "expected"),
    [
        (lambda c: print_report_summary({"file_size": "1024"}, console=c), "1,024 bytes"),
        (
            lambda c: print_extracted_files_table([{"file_size": "4096"}], console=c),
            "4,096 bytes",
        ),
        (lambda c: print_search_results_table([{"file_size": "1024"}], console=c), "1,024"),
        (lambda c: print_samples_panels([{"file_size": "1024"}], console=c), "1,024 bytes"),
        (lambda c: print_samples_panels([{"file_size": float("inf")}], console=c), "N/A"),
        (lambda c: print_samples_panels([{"file_size": "not-a-size"}], console=c), "N/A"),
    ],
    ids=["report", "extracted", "search", "samples", "infinite", "unparsable"],
)
def test_a_non_integer_size_is_formatted_not_fatal(render, expected: str) -> None:
    """A JSON payload states a size as a string; ``f"{size:,}"`` raises on one."""
    assert expected in _plain(render)


@pytest.mark.parametrize(
    "av_scanners",
    [
        [{"nextgen_scanners": [{"result": "x"}]}],
        [{"regular_scanners": [{"name": "n", "result": 5}]}],
        [{"regular_scanners": ["x"]}],
        "none",
        {"regular_scanners": None},
    ],
    ids=["no_name", "numeric_result", "bare_string_row", "not_a_mapping", "null_section"],
)
def test_a_malformed_scanner_entry_does_not_kill_the_report(av_scanners) -> None:
    """One nameless scanner used to take the whole ``a1000 report`` with it."""
    # File Metadata rather than File Information: this payload states none
    # of the fields that identify a sample, and the panel for those is
    # skipped rather than drawn as a box of N/A.
    assert "File Metadata" in _plain(
        lambda c: print_report_summary({"av_scanners": av_scanners}, console=c)
    )


def test_a_nameless_scanner_still_reports_its_detection() -> None:
    output = _plain(
        lambda c: print_report_summary(
            {"av_scanners": [{"nextgen_scanners": [{"result": "Trojan.X"}]}]}, console=c
        )
    )
    assert "Trojan.X" in output
    assert "1/1" in output


@pytest.mark.parametrize(
    "render",
    [
        lambda c: print_search_results_table([{"file_type": 7}], console=c),
        lambda c: print_search_results_table([{"file_type": ["a"]}], console=c),
        lambda c: print_search_results_table([{"sha256": 12345678901234567890}], console=c),
        lambda c: print_report_summary({"file_type": 7}, console=c),
        lambda c: print_report_summary({"local_first_seen": 1700000000}, console=c),
        lambda c: print_report_summary({"ticore": ["not-a-mapping"]}, console=c),
        lambda c: print_report_summary({"tags": ["not-a-mapping"]}, console=c),
        lambda c: print_report_summary({"tags": {"ticore": [7]}}, console=c),
        lambda c: print_extracted_files_table([{"filename": 123}], console=c),
        lambda c: print_extracted_files_table([{"filename": {"a": 1}}], console=c),
        lambda c: print_yara_rulesets_table([{"name": 5}], console=c),
        lambda c: print_yara_rulesets_table([{"last_matched": 17}], console=c),
        lambda c: print_yara_rulesets_table([{"malicious_match_count": "x"}], console=c),
        lambda c: print_reanalyze_results_table(
            [{"detail": "flat", "analysis": "none"}], console=c
        ),
        lambda c: print_file_analysis({"file_name": ["a"], "threat_level": {"x": 1}}, console=c),
    ],
    ids=[
        "search_numeric_type",
        "search_list_type",
        "search_numeric_digest",
        "report_numeric_type",
        "report_numeric_first_seen",
        "report_list_ticore",
        "report_list_tags",
        "report_numeric_tag",
        "extracted_numeric_name",
        "extracted_mapping_name",
        "yara_numeric_name",
        "yara_numeric_last_matched",
        "yara_unparsable_count",
        "reanalyze_scalar_sections",
        "ticloud_non_string_fields",
    ],
)
def test_a_field_of_the_wrong_type_renders_rather_than_raising(render) -> None:
    """These come out of the archive the attacker built; none of them is a str."""
    render(Console(file=io.StringIO(), width=200))


class TestThreatPanelKeySpelling:
    """The panel must not hinge on the literal "threat_name" key."""

    def test_classification_result_still_renders_the_panel(self):
        console = Console(width=100, force_terminal=False)
        with console.capture() as capture:
            print_file_analysis(
                {"status": "MALICIOUS", "classification_result": "Win32.Trojan.Emotet"},
                console=console,
            )
        assert "Threat Assessment" in capture.get()
        assert "Win32.Trojan.Emotet" in capture.get()


# --- Fields the report asked the appliance for and never rendered -------------


class TestReportRendersWhatItRequested:
    """``a1000 report`` surfaced 14 of the 32 fields in the SDK's fields_v2."""

    RECORD: ClassVar[dict[str, Any]] = {
        "sha256": "a" * 64,
        "file_type": "PE+ executable",
        "classification": "malicious",
        "classification_result": "Win64.Ransomware.Conti",
        "proposed_filename": "invoice_2024.exe",
        "aliases": ["setup.exe", "readme.scr"],
        "classification_reason": "Antivirus",
        "classification_origin": "ReversingLabs",
        "classification_source": "cloud",
        "extracted_file_count": 17,
        "local_last_seen": "2024-06-01T09:00:00Z",
        "networkthreatintelligence": {
            "ip": [{"ipv4": "203.0.113.5", "classification": "malicious"}],
            "url": [{"url": "http://evil.tld/gate.php", "classification": "suspicious"}],
        },
        "domainthreatintelligence": {"domain": [{"domain": "evil.tld", "classification": 3}]},
    }

    def _render(self) -> str:
        return _plain(lambda c: print_report_summary(self.RECORD, console=c))

    def test_the_report_finally_shows_a_filename(self):
        assert "File Name: invoice_2024.exe" in self._render()

    def test_the_other_names_the_sample_was_seen_under_are_listed(self):
        assert "Aliases: setup.exe, readme.scr" in self._render()

    def test_why_it_is_malicious_is_stated(self):
        rendered = self._render()
        assert "Reason: Antivirus" in rendered
        assert "Origin: ReversingLabs" in rendered
        assert "Source: cloud" in rendered

    def test_the_c2_indicators_are_rendered(self):
        rendered = self._render()
        assert "Network Indicators" in rendered
        assert "203.0.113.5" in rendered
        assert "http://evil.tld/gate.php" in rendered
        assert "evil.tld" in rendered

    def test_the_extracted_file_count_is_shown(self):
        assert "Extracted Files: 17" in self._render()

    def test_last_seen_is_shown_next_to_first_seen(self):
        assert "Last Seen: 2024-06-01T09:00:00" in self._render()

    def test_an_ioc_carries_its_own_verdict_not_the_samples(self, ansi):
        """A clean sample can still have talked to a malicious host."""
        output = ansi(
            lambda c: print_report_summary(
                {
                    "classification": "goodware",
                    "networkthreatintelligence": {
                        "ip": [{"ipv4": "203.0.113.5", "classification": "malicious"}]
                    },
                },
                console=c,
            )
        )
        assert f"{RED}malicious" in output
        assert "203.0.113.5" in output

    def test_an_intel_entry_without_an_indicator_is_skipped(self, ansi):
        """An entry carrying no address key contributes no row."""
        output = ansi(
            lambda c: print_report_summary(
                {
                    "classification": "goodware",
                    "networkthreatintelligence": {
                        "ip": [{"classification": "malicious"}, {"ipv4": "203.0.113.9"}]
                    },
                },
                console=c,
            )
        )
        assert "203.0.113.9" in output


def test_bullet_items_drops_members_that_would_draw_nothing() -> None:
    """A section's empty members are not bullets; its scalars and lists are."""
    from rl_cli.render.formatters.ticore_panels import bullet_items

    assert bullet_items({"empty": None, "blank": "", "obj": {}, "real": "x"}) == ["real: x"]


def test_a_macho_sample_gets_a_platform_row() -> None:
    """TitaniumCore spells it "MachO"; the hyphenated test never matched."""
    assert "Platform: macOS" in _plain(
        lambda c: print_report_summary({"file_type": "MachO64 Executable"}, console=c)
    )


def test_the_appliances_own_scanner_ratio_is_authoritative() -> None:
    """Summing the two scanner arrays reported 13/16 where the A1000 UI —
    and av_scanners_summary, sitting unread in the same payload — said 11/14."""
    output = _plain(
        lambda c: print_report_summary(
            {
                "av_scanners_summary": {"scanner_count": 14, "scanner_match_count": 11},
                "av_scanners": [
                    {
                        "regular_scanners": [
                            {"name": f"AV{i}", "result": "Trojan.X"} for i in range(11)
                        ]
                        + [{"name": "AVClean", "result": ""}],
                        "nextgen_scanners": [
                            {"name": "NG1", "result": "malicious"},
                            {"name": "NG2", "result": "malicious"},
                        ],
                    }
                ],
            },
            console=c,
        )
    )
    assert "11/14" in output
    assert "13/16" not in output


def test_next_gen_verdicts_are_reported_separately() -> None:
    """An ML confidence verdict is not a signature hit."""
    output = _plain(
        lambda c: print_report_summary(
            {
                "av_scanners": [
                    {
                        "regular_scanners": [{"name": "AV1", "result": "Trojan.X"}],
                        "nextgen_scanners": [{"name": "NG1", "result": "malicious"}],
                    }
                ]
            },
            console=c,
        )
    )
    assert "Next-Gen (ML): 1/1" in output


class TestOneRecordReportsOneRatio:
    """The response shape must not move the ratio the report draws.

    Folding the next-gen verdicts in drew "5/15" from the arrays and "2/10"
    from a summary stating the same signature numbers, with those three ML
    verdicts inside the ratio and again on the line below it.
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

    def _render(self, payload: dict[str, Any]) -> str:
        return _plain(lambda c: print_report_summary(payload, console=c))

    def test_the_arrays_alone_draw_the_signature_ratio(self):
        output = self._render({"av_scanners": self.ARRAYS})
        assert "Detection Ratio: 2/10" in output
        assert "5/15" not in output
        assert "Next-Gen (ML): 3/5" in output

    def test_a_summary_of_the_same_numbers_draws_the_same_ratio(self):
        summarised = self._render(
            {
                "av_scanners": self.ARRAYS,
                "av_scanners_summary": {"scanner_count": 10, "scanner_match_count": 2},
            }
        )
        assert "Detection Ratio: 2/10" in summarised
        assert "Next-Gen (ML): 3/5" in summarised

    def test_a_next_gen_array_alone_keeps_its_panel(self):
        """A 0/0 signature ratio is not "no engine ran": the ML ones did."""
        output = self._render(
            {"av_scanners": [{"nextgen_scanners": [{"name": "NG1", "result": "Trojan.X"}]}]}
        )
        assert "Next-Gen (ML): 1/1" in output
        assert "Trojan.X" in output


class TestATruncatedTimestampKeepsItsZone:
    """Cutting the microseconds off must not cut the zone off with them.

    ``sanitize(seen)[:19]`` left the wall clock and dropped the ``Z`` or the
    ``+05:30``, so a UTC report read as a local one to anyone building a
    timeline out of it.
    """

    def _seen(self, stated: str) -> str:
        return _plain(lambda c: print_report_summary({"local_first_seen": stated}, console=c))

    def test_a_utc_timestamp_still_says_utc(self):
        assert "2024-01-01T00:00:00Z" in self._seen("2024-01-01T00:00:00.123456Z")

    def test_an_offset_survives_the_truncation(self):
        assert "2024-01-02T03:04:05+05:30" in self._seen("2024-01-02T03:04:05.999+05:30")

    def test_the_sub_second_digits_are_still_dropped(self):
        assert "00:00:00.123456" not in self._seen("2024-01-01T00:00:00.123456Z")

    def test_a_timestamp_naming_no_zone_gains_none(self):
        rendered = self._seen("2024-01-01T00:00:00.123456")
        assert "2024-01-01T00:00:00" in rendered
        assert "2024-01-01T00:00:00Z" not in rendered


def test_the_detection_table_does_not_claim_a_ranking_it_has_not_got() -> None:
    output = _plain(
        lambda c: print_report_summary(
            {"av_scanners": [{"regular_scanners": [{"name": "AV1", "result": "T"}]}]}, console=c
        )
    )
    assert "Top Detections" not in output
    assert "AV Detections" in output


def test_the_status_panel_names_the_malware() -> None:
    """A verdict that never names the threat is half an answer."""
    assert "Threat Name: Win64.Ransomware.Conti" in _plain(
        lambda c: print_analysis_status(
            {"classification": "malicious", "classification_result": "Win64.Ransomware.Conti"},
            console=c,
        )
    )


class TestExtractedFilesShowTheirVerdict:
    """The command for finding the payload inside a dropper."""

    FILES: ClassVar[list[dict[str, Any]]] = [
        {"filename": "icon.ico", "sample": {"sha256": "c" * 64, "classification": "goodware"}},
        {"filename": "readme.txt", "sample": {"sha256": "d" * 64}},
        {
            "filename": "payload.dll",
            "sample": {
                "sha256": "e" * 64,
                "classification": "malicious",
                "classification_result": "Win64.Ransomware.Conti",
                "riskscore": 10,
            },
        },
    ]

    def _render(self) -> str:
        return _plain(lambda c: print_extracted_files_table(self.FILES, console=c))

    def test_the_verdict_is_a_column_now(self):
        rendered = self._render()
        assert "Classification" in rendered
        assert "malicious" in rendered
        assert "goodware" in rendered

    def test_the_malicious_file_sorts_to_the_top(self):
        rendered = self._render()
        assert rendered.index("payload.dll") < rendered.index("icon.ico")
        assert rendered.index("payload.dll") < rendered.index("readme.txt")

    def test_a_clean_extracted_file_is_not_painted_red(self, ansi):
        output = ansi(lambda c: print_extracted_files_table(self.FILES, console=c))
        assert f"{GREEN}goodware" in output
        assert f"{RED}malicious" in output


class TestReanalysisReportsRejections:
    """A batch of 50 unknown hashes came back as 50 successes."""

    REJECTED: ClassVar[dict[str, Any]] = {
        "detail": {"sha256": "a" * 64},
        "analysis": [
            {"name": "cloud", "code": 404, "message": "Sample not found on the appliance"},
            {"name": "core", "code": 403, "message": "User is not authorized"},
        ],
    }
    ACCEPTED: ClassVar[dict[str, Any]] = {
        "detail": {"sha256": "b" * 64},
        "analysis": [{"name": "cloud", "code": 201}, {"name": "core", "code": 201}],
    }

    def test_a_rejected_sample_is_not_reported_as_queued(self):
        output = _plain(lambda c: print_reanalyze_results_table([self.REJECTED], console=c))
        assert "Queued" not in output
        assert "Failed" in output

    def test_the_engines_own_message_is_shown(self):
        output = _plain(lambda c: print_reanalyze_results_table([self.REJECTED], console=c))
        assert "Sample not found on the appliance" in output
        assert "User is not authorized" in output
        assert "404" in output

    def test_an_accepted_sample_still_reads_as_queued(self):
        output = _plain(lambda c: print_reanalyze_results_table([self.ACCEPTED], console=c))
        assert "Queued: cloud, core" in output
        assert "Failed" not in output

    def test_a_partial_batch_reports_both(self):
        mixed = {
            "detail": {"sha256": "c" * 64},
            "analysis": [
                {"name": "cloud", "code": 201},
                {"name": "core", "code": 404, "message": "not found"},
            ],
        }
        output = _plain(lambda c: print_reanalyze_results_table([mixed], console=c))
        assert "Queued: cloud" in output
        assert "core (404): not found" in output

    @pytest.mark.parametrize(
        "entry, accepted",
        [
            (REJECTED, False),
            (ACCEPTED, True),
            ({"code": 404, "message": "Sample not found"}, False),
            ({"status": "Sample not found", "analysis": []}, False),
            ({"status": "queued", "analysis": []}, True),
            ({"detail": {"sha256": "d" * 64}}, True),
        ],
        ids=["engines-refused", "engines-queued", "code", "stated", "queued", "silent"],
    )
    def test_the_cell_states_the_verdict_the_count_above_it_read(self, entry, accepted):
        """The Status column draws ``accepted``; it does not grade the row again.

        The count over the table and the exit status are that same flag, so
        a cell that decided for itself is how a green "Reanalysis started
        for 5 samples" came to sit over five refused rows.
        """
        assert ReanalysisOutcome.of(entry).accepted is accepted, "the fixture assumes the grading"

        output = _plain(lambda c: print_reanalyze_results_table([entry], console=c))

        assert ("Refused" in output) is not accepted, output


class TestTheTableAccountsForEveryEntryTheCountDid:
    """A row the table could not draw was counted as refused and then dropped.

    ``["oops"]`` is one sample the appliance answered about: the headline
    says "Reanalysis refused for all 1 samples" and the run exits 1, over an
    empty table. The two numbers come from one grading —
    ``ReanalysisOutcome.of`` reads a non-record as "not accepted" precisely
    so the count can have it — and a table that then draws nothing leaves
    the analyst reading a failure with no row to attribute it to.

    So every entry gets a row. The hash cell is the mark for "the payload
    did not say", and the Status cell is "Refused" with no reason after it:
    the parse failure is ours, and naming it as the appliance's would put
    words in its mouth.
    """

    UNREADABLE: ClassVar[list[Any]] = ["oops", None, 7, ["nested"]]

    @pytest.mark.parametrize("entry", UNREADABLE, ids=["string", "null", "number", "list"])
    def test_an_entry_the_table_cannot_draw_is_still_drawn(self, entry):
        console = Console(file=io.StringIO(), width=200)

        drawn = print_reanalyze_results_table([entry], console=console)

        assert drawn == 1
        assert "Refused" in console_text(console)

    def test_the_drawn_count_is_every_entry_the_batch_was_graded_from(self):
        """What the count over the table is a count of."""
        entries = cast(list[Any], [{"detail": {"sha256": "a" * 64}}, *self.UNREADABLE])

        assert print_reanalyze_results_table(entries, console=Console(file=io.StringIO())) == len(
            entries
        )

    def test_the_batch_grades_the_entries_this_table_draws_in_the_order_it_draws_them(self):
        """What keeps the sentence above the table and the rows under it in step.

        They are two invocations of one rule — ``ReanalysisOutcome.of`` —
        rather than one grading shared: the command grades the answer into
        a ``ReanalysisBatch`` for its count and hands this function the raw
        list, which grades it again, entry by entry. That holds only while
        the batch grades exactly the entries it was given, in the order it
        was given them; a batch that filtered or reordered them first would
        move the count and leave these rows where they were, with nothing
        on this side able to see it. So the alignment is asserted here,
        where both halves are in view.
        """
        entries: list[Any] = [
            {"detail": {"sha256": "a" * 64}, "analysis": [{"name": "core", "code": 201}]},
            {"detail": {"sha256": "b" * 64}, "analysis": [{"name": "core", "code": 404}]},
            *self.UNREADABLE,
        ]
        batch = ReanalysisBatch.of(entries)

        assert batch.entries == entries
        assert batch.outcomes == tuple(ReanalysisOutcome.of(entry) for entry in entries)

        console = Console(file=io.StringIO(), width=200)
        assert print_reanalyze_results_table(entries, console=console) == batch.answered
        assert console_text(console).count("Refused") == batch.refused

    def test_an_unreadable_entry_states_no_refusal_of_its_own(self):
        """ "Refused: oops" is our parse failure quoted as the appliance's word."""
        output = _plain(lambda c: print_reanalyze_results_table(["oops"], console=c))

        assert "Refused" in output
        assert "oops" not in output

    @pytest.mark.parametrize(
        "answer",
        [{"detail": "nope"}, "oops", 7, None],
        ids=["mapping", "string", "number", "null"],
    )
    def test_an_answer_that_is_no_list_of_entries_draws_an_empty_table(self, answer):
        """ "Every entry the caller was given" is no entries when it was given none.

        The appliance answers this endpoint with a list, and an error body
        is a mapping — ``{"detail": "nope"}`` — so the slice that takes the
        first ``max_rows`` entries raised ``KeyError: slice(None, 10,
        None)`` out of a renderer nothing wraps in a ``try``. A string is
        worse than a raise: it slices and iterates, so a five-character
        body draws five rows about samples that were never mentioned.

        The empty table is the whole answer: the function is exported for
        any caller with a reanalysis answer to draw, and one that could not
        be read holds no entries to attribute a row to.
        """
        console = Console(file=io.StringIO(), width=200)

        assert print_reanalyze_results_table(cast(list[Any], answer), console=console) == 0
        # The table is still drawn, headers and all: the caller asked for
        # the rendering, and an empty one is what "no entries" looks like.
        assert "Hash" in console_text(console)


class TestYaraRulesetRanking:
    """Goodware hits dominated the only count the table showed."""

    RULESETS: ClassVar[list[dict[str, Any]]] = [
        {"name": "packer", "status": "active", "goodware_match_count": 48213},
        {"name": "conti", "status": "active", "malicious_match_count": 12},
        {"name": "broken", "status": "error", "error_message": "syntax error at line 4"},
    ]

    def _render(self) -> str:
        return _plain(lambda c: print_yara_rulesets_table(self.RULESETS, console=c))

    def test_the_malicious_count_gets_its_own_column(self):
        rendered = self._render()
        assert "Malicious" in rendered
        assert "48,213" in rendered
        assert "12" in rendered

    def test_a_broken_ruleset_says_why(self):
        assert "syntax error at line 4" in self._render()


# The SGR parameters that set a foreground or background colour. Rich's
# ``no_color`` suppresses exactly these and leaves bold, dim and italic
# alone, so "no escapes at all" is the wrong thing to assert.
_COLOUR_PARAMETERS = frozenset(
    [str(code) for code in (*range(30, 39), *range(40, 49), *range(90, 98), *range(100, 108))]
)


def _colour_escapes(rendered: str) -> list[str]:
    """The SGR sequences in ``rendered`` that paint something a colour."""
    return [
        sequence
        for sequence in re.findall(r"\x1b\[([0-9;]*)m", rendered)
        if _COLOUR_PARAMETERS & set(sequence.split(";"))
    ]


class TestTheCliConsoleReachesTheRenderers:
    """``color: false`` was read, acted on, and then thrown away.

    ``cli/main.py`` builds both consoles with ``no_color`` from the
    settings, but ``display`` called its renderer with the data alone, so
    every A1000 renderer fell back to a ``Console()`` of its own — one that
    reads the terminal instead of the configuration. Only the TitaniumCloud
    panel, which bound the console at its call site, honoured the setting.
    """

    RECORD: ClassVar[dict[str, Any]] = {"classification": "malicious", "threat_name": "Trojan.X"}

    @staticmethod
    def _rendered(renderer, *, no_color: bool) -> str:
        buffer = io.StringIO()
        # force_terminal, or Rich emits no styling into a StringIO at all
        # and the colour assertions would hold however broken the wiring is.
        console = Console(file=buffer, width=100, force_terminal=True, no_color=no_color)
        display(
            OutputFormatter(OutputFormat.RICH, RichOutput(console)),
            renderer,
            TestTheCliConsoleReachesTheRenderers.RECORD,
        )
        return buffer.getvalue()

    def test_the_renderer_draws_on_the_console_it_was_given(self):
        assert "Analysis Status" in self._rendered(print_analysis_status, no_color=False)

    def test_a_colour_console_still_emits_colour(self):
        assert _colour_escapes(self._rendered(print_analysis_status, no_color=False))

    def test_a_no_colour_console_emits_none(self):
        assert _colour_escapes(self._rendered(print_analysis_status, no_color=True)) == []

    @pytest.mark.parametrize(
        "renderer",
        [
            print_analysis_status,
            print_summary_panel,
            print_report_summary,
            print_titanium_report,
            print_file_analysis,
        ],
    )
    def test_no_renderer_reaches_for_a_console_of_its_own(self, renderer):
        rendered = self._rendered(renderer, no_color=True)
        assert rendered.strip(), f"{renderer.__name__} drew nothing on the CLI's console"
        assert _colour_escapes(rendered) == []
        assert _colour_escapes(self._rendered(renderer, no_color=False))


class TestSeverityPresentationIsTotal:
    """One table, and no renderer raises over a severity it has not met.

    Three modules each kept their own map keyed by the same four verdicts
    and looked them up with ``[]``; adding a fifth severity to payload.py
    would have raised KeyError out of the report renderer and the SARIF
    exporter.
    """

    @pytest.mark.parametrize(
        ("severity", "colour", "icon", "rank", "level"),
        [
            ("malicious", "red", "🔴", 0, "error"),
            ("suspicious", "yellow", "🟡", 1, "warning"),
            ("unknown", "white", "⚪", 2, "none"),
            ("known", "green", "🟢", 3, "note"),
        ],
    )
    def test_every_known_severity_keeps_its_presentation(self, severity, colour, icon, rank, level):
        assert colour_of(severity) == colour
        assert style_of(severity) == (colour, icon)
        assert rank_of(severity) == rank
        assert sarif_level_of(severity) == level
        # The rule id is the fifth lookup keyed by the same four verdicts,
        # and it was the one this class did not name — see the fallback
        # parametrize below, which it was also missing from.
        assert sarif_rule_of(severity) == f"rl-cli/{severity}"

    def test_the_worst_verdict_sorts_first(self):
        severities = ["known", "malicious", "unknown", "suspicious"]
        assert sorted(severities, key=rank_of) == [
            "malicious",
            "suspicious",
            "unknown",
            "known",
        ]

    # ``sarif_rule_of`` was absent from this list, and it is the lookup
    # where falling back wrongly is worst: pointing an unrecognised
    # severity at ``rl-cli/known`` files the alert under the rule that
    # says "ReversingLabs classifies this sample as known-good", which is
    # a reassurance rather than the absence of an opinion. The suite was
    # green with that fallback in place.
    @pytest.mark.parametrize(
        "lookup", [colour_of, style_of, rank_of, sarif_level_of, sarif_rule_of]
    )
    def test_an_unheard_of_severity_falls_back_instead_of_raising(self, lookup):
        assert lookup("catastrophic") == lookup("unknown")


class TestThreatLevelZeroIsALevel:
    """``or`` read the appliance's lowest threat level as a missing one."""

    def test_a_zero_threat_level_is_reported_as_zero(self):
        rendered = _plain(
            lambda c: print_report_summary(
                {"classification": "malicious", "threat_level": 0, "ticore": {}}, console=c
            )
        )
        assert "Threat Level: 0" in rendered

    def test_an_absent_threat_level_still_falls_back_to_the_ticore_factor(self):
        rendered = _plain(
            lambda c: print_report_summary(
                {"classification": "malicious", "ticore": {"classification": {"factor": 4}}},
                console=c,
            )
        )
        assert "Threat Level: 4" in rendered


class TestTitaniumReportReadsOddShapes:
    """``(info.get("file") or {})`` still handed a list to ``.get``."""

    def test_a_list_valued_file_section_renders_rather_than_raising(self):
        rendered = _plain(
            lambda c: print_titanium_report(
                {"info": {"file": ["not", "a", "mapping"]}, "sha1": "a" * 40}, console=c
            )
        )
        assert "a" * 40 in rendered

    def test_a_list_valued_classification_renders_rather_than_raising(self):
        rendered = _plain(
            lambda c: print_titanium_report(
                {"classification": ["nope"], "md5": "d" * 32}, console=c
            )
        )
        assert "d" * 32 in rendered


# --- Caps: a listing that was cut must say so --------------------------------


class TestListingsAreBounded:
    """An archive states how many files it carries; the table did not.

    ``print_extracted_files_table`` and ``print_yara_rulesets_table`` were
    the only tables in the package with no row cap, so a dropper carrying
    four thousand extracted files rendered four thousand rows. They then
    grew an in-table "... and N more" tail while every sibling listing
    returned its drawn count for the command to announce, so the CLI had
    two vocabularies for one fact and only one of them named ``-o json``.
    """

    def test_the_extracted_files_table_caps_and_says_how_many_it_drew(
        self, console: Console
    ) -> None:
        files = [
            {"filename": f"stage{index}.bin", "sample": {"classification": "goodware"}}
            for index in range(30)
        ]
        assert print_extracted_files_table(files, max_rows=5, console=console) == 5
        assert "stage4.bin" in console_text(console)

    def test_the_extracted_files_cap_keeps_the_worst_verdicts(self, console: Console) -> None:
        """The cap must not drop the payload the command exists to find."""
        files = [
            {"filename": f"clean{index}.bin", "sample": {"classification": "goodware"}}
            for index in range(30)
        ]
        files.append({"filename": "conti.dll", "sample": {"classification": "malicious"}})
        assert print_extracted_files_table(files, max_rows=3, console=console) == 3
        assert "conti.dll" in console_text(console)

    def test_an_uncut_extracted_listing_grows_no_tail(self, console: Console) -> None:
        assert print_extracted_files_table([{"filename": "only.bin"}], console=console) == 1
        assert "more" not in console_text(console)

    def test_the_yara_rulesets_table_caps_and_says_how_many_it_drew(self, console: Console) -> None:
        rulesets = [{"name": f"ruleset-{index}"} for index in range(25)]
        assert print_yara_rulesets_table(rulesets, max_rows=4, console=console) == 4
        assert "ruleset-3" in console_text(console)

    def test_an_uncut_ruleset_listing_grows_no_tail(self, console: Console) -> None:
        assert print_yara_rulesets_table([{"name": "only"}], console=console) == 1
        assert "more" not in console_text(console)

    @pytest.mark.parametrize("render", [print_extracted_files_table, print_yara_rulesets_table])
    def test_the_default_cap_is_the_one_the_siblings_use(self, render, console: Console) -> None:
        drawn = render(
            [{"filename": f"r{index}", "name": f"r{index}"} for index in range(25)], console=console
        )
        assert drawn == 20

    @pytest.mark.parametrize(
        "render",
        [print_extracted_files_table, print_yara_rulesets_table, print_search_results_table],
    )
    def test_the_drawn_count_excludes_rows_the_table_skipped(
        self, render, console: Console
    ) -> None:
        """A result array carries entries no table can render; the note must not count them."""
        rows: list[Any] = [{"filename": "a.bin", "name": "a", "sha256": "a" * 64}, "skipped", None]
        assert render(rows, max_rows=10, console=console) == 1


# --- The File Information panel both reports share ----------------------------


class TestSampleFacts:
    """One record, two services' spellings, one panel.

    The A1000 report and the TitaniumCloud report each carried their own
    copy of this panel; the normalising lives in ``SampleFacts.of`` now.
    """

    def test_the_titaniumcloud_sample_spelling_is_read(self):
        facts = SampleFacts.of({"sample_type": "PE/Exe", "sample_size": 4096, "file_name": "a.exe"})
        assert facts.file_type == "PE/Exe"
        assert facts.size == 4096
        assert facts.name == "a.exe"

    def test_the_a1000_spelling_wins_where_both_are_stated(self):
        facts = SampleFacts.of(
            {"file_type": "ELF", "sample_type": "PE/Exe", "file_size": 1, "sample_size": 2}
        )
        assert facts.file_type == "ELF"
        assert facts.size == 1

    def test_the_appliances_own_choice_of_name_outranks_the_list_it_saw(self):
        """``proposed_filename`` is the appliance's answer; ``file_names`` is
        every name the sample ever arrived under.

        The two are read in that order on purpose and a search entry can
        carry both, but nothing pinned it: swapping them left the suite
        green while the panel and the SARIF ``artifactLocation`` named the
        sample by whichever alias happened to be first in the list.
        """
        facts = SampleFacts.of(
            {"proposed_filename": "conti.dll", "file_names": ["invoice.pdf", "conti.dll"]}
        )
        assert facts.name == "conti.dll"

    def test_the_bare_name_outranks_the_path_it_was_carved_from(self):
        """``_NAME_KEYS`` runs least specific last, and an extracted-file
        entry states both halves of the pair.

        Reversing the tuple left the suite green while every extracted row
        and every SARIF location named a Conti DLL by the path inside the
        container it came out of rather than by the file, so the same
        sample read two ways depending on which spelling its entry held.
        """
        facts = SampleFacts.of({"file_name": "stage2.dll", "full_path": "dropper/stage2.dll"})
        assert facts.name == "stage2.dll"

    @pytest.mark.parametrize(
        "record",
        [
            {"proposed_filename": "a.exe"},
            {"file_names": ["a.exe", "b.exe"]},
            {"file_name": "a.exe"},
            {"filename": "a.exe"},
            {"full_path": "a.exe"},
            {"aliases": ["a.exe"]},
        ],
        ids=["proposed", "list", "file_name", "filename", "full_path", "alias"],
    )
    def test_every_spelling_of_a_name_names_the_same_sample(self, record):
        """Five renderers read five subsets of these, so one file had two names."""
        assert SampleFacts.of(record).name == "a.exe"

    def test_the_appliance_choice_of_name_outranks_the_path_it_was_carved_from(self):
        record = {"proposed_filename": "a.exe", "full_path": "drop/a.tmp"}
        assert SampleFacts.of(record).name == "a.exe"

    def test_the_strongest_digest_names_a_sample_with_no_name(self):
        assert SampleFacts.of({"md5": "c" * 32, "sha256": "a" * 64}).digest == "a" * 64
        assert SampleFacts.of({"md5": "c" * 32}).digest == "c" * 32
        assert SampleFacts.of({}).digest is None

    def test_the_first_alias_names_a_sample_with_no_proposed_filename(self):
        facts = SampleFacts.of({"aliases": ["setup.exe", "readme.scr"]})
        assert facts.name == "setup.exe"
        assert facts.aliases == ("setup.exe", "readme.scr")

    def test_a_zero_size_is_not_mistaken_for_a_missing_one(self):
        assert SampleFacts.of({"file_size": 0}).size == 0

    def test_a_malformed_aliases_section_renders_as_none(self):
        assert SampleFacts.of({"aliases": "not-a-list"}).aliases == ()

    def test_the_alias_list_is_capped_with_a_count(self, console: Console) -> None:
        print_file_information(
            SampleFacts.of({"aliases": [f"name{index}.exe" for index in range(9)]}), console=console
        )
        assert "(and 4 more)" in console_text(console)

    def test_both_reports_state_the_same_fields_in_the_same_order(self, console: Console) -> None:
        record = {
            "file_name": "a.exe",
            "md5": "c" * 32,
            "sha1": "b" * 40,
            "sha256": "a" * 64,
            "sample_type": "PE/Exe",
            "sample_size": 4096,
        }
        print_file_information(SampleFacts.of(record), console=console)
        output = console_text(console)
        labels = ["File Name", "MD5", "SHA1", "SHA256", "File Type", "File Size"]
        assert [label for label in labels if f"{label}:" in output] == labels
        assert sorted(labels, key=output.index) == labels

    def test_a_field_the_service_never_states_is_left_out_not_shown_as_na(
        self, console: Console
    ) -> None:
        """A reputation record is a hash and a verdict; four N/A rows say less."""
        print_file_information(SampleFacts.of({"sha1": "b" * 40}), console=console)
        output = console_text(console)
        assert "b" * 40 in output
        assert "N/A" not in output

    def test_a_record_identifying_nothing_gets_no_panel(self, console: Console) -> None:
        print_file_information(SampleFacts.of({}), console=console)
        assert console_text(console).strip() == ""


class TestAZeroByteFileIsTheSameSizeEverywhere:
    """An empty file is a size, and every renderer has to draw the same one.

    The identity reader guards it against ``None``; the sample panel used
    ``or`` and fell through to the *other* service's spelling, so one
    record drew "0 bytes" in the File Information panel and 4,096 — a size
    the file does not have — in the panel the listing drew beside it.
    """

    RECORD: ClassVar[dict[str, Any]] = {
        "file_name": "empty.bin",
        "sha256": "a" * 64,
        "file_size": 0,
        "sample_size": 4096,
    }

    def test_the_sample_panel_draws_the_zero_not_the_other_spelling(self):
        rendered = _plain(lambda c: print_samples_panels([self.RECORD], console=c))
        assert "Size: 0 bytes" in rendered
        assert "4,096" not in rendered

    def test_the_file_information_panel_draws_the_same_zero(self, console: Console) -> None:
        print_file_information(SampleFacts.of(self.RECORD), console=console)
        output = console_text(console)
        assert "0 bytes" in output
        assert "4,096" not in output

    def test_the_search_table_draws_the_same_zero(self):
        rendered = _plain(lambda c: print_search_results_table([self.RECORD], console=c))
        assert "4,096" not in rendered

    def test_an_absent_size_is_still_absent(self):
        """The guard must not turn "no size stated" into a zero-byte file."""
        assert SampleFacts.of({"sample_size": 4096}).size == 4096
        assert SampleFacts.of({}).size is None


class TestADigestIsCutToOneWidth:
    """Three clippers for one column, two of which overspent the width.

    ``clip`` reserves the ellipsis inside ``width`` — the rule that lets a
    reader take an unsuffixed cell for a complete one — while the search
    table and the reanalysis table sliced by hand and added the ellipsis on
    top, drawing 19 characters where their sibling column drew 16.
    """

    def test_a_cut_digest_fits_the_column_it_is_cut_for(self):
        assert len(digest_cell("a" * 64)) == DIGEST_CELL_WIDTH

    def test_a_digest_that_fits_is_left_unmarked(self):
        assert digest_cell("a" * DIGEST_CELL_WIDTH) == "a" * DIGEST_CELL_WIDTH

    @pytest.mark.parametrize(
        "render",
        [
            lambda c: print_search_results_table([{"sha256": "a" * 64}], console=c),
            lambda c: print_reanalyze_results_table([{"hash": "a" * 64}], console=c),
        ],
        ids=["search", "reanalyze"],
    )
    def test_every_listing_cuts_a_digest_at_the_same_width(self, render):
        rendered = _plain(render)
        assert "a" * (DIGEST_CELL_WIDTH - 3) + "..." in rendered
        assert "a" * (DIGEST_CELL_WIDTH - 2) not in rendered


# --- The report must not get quieter as the sample gets worse -----------------
#
# Each class below is one finding from an audit that rendered a real
# payload and read the screen back. Every one of them made the report
# *less* alarming exactly where the sample was dangerous, which is the
# worst thing this tool can do.


class TestTheRatioNeverUnderreportsTheDetections:
    """A green "0/17" printed three lines above a table of 13 detections.

    ``av_scanners_summary`` was read with the gate on one key and the value
    taken from another, so a summary stating only ``scanner_count`` had
    ``count(None)`` — zero — overwrite the thirteen detections counted off
    the scanner arrays, and the ratio panel painted that zero green.
    """

    DETECTIONS: ClassVar[list[dict[str, Any]]] = [
        {"name": f"AV{index}", "result": "Win32.Ransom.Conti"} for index in range(13)
    ]

    def _render(self, summary: dict[str, Any]) -> str:
        return _plain(
            lambda c: print_report_summary(
                {
                    "classification": "malicious",
                    "av_scanners_summary": summary,
                    "av_scanners": [{"regular_scanners": self.DETECTIONS}],
                },
                console=c,
            )
        )

    @pytest.mark.parametrize(
        "summary",
        [
            {"scanner_count": 17},
            {"scanner_count": 17, "scanner_match_count": None},
            {"scanner_count": 17, "scanner_match_count": 0},
        ],
        ids=["omitted", "explicit_null", "stated_zero"],
    )
    def test_a_summary_that_states_no_hits_does_not_erase_the_counted_ones(self, summary):
        rendered = self._render(summary)
        assert "13/17" in rendered
        assert "0/17" not in rendered

    def test_the_titaniumcloud_spelling_of_the_match_count_is_read(self):
        """``scanner_match`` is the same number under the other API's name."""
        assert "13/17" in self._render({"scanner_count": 17, "scanner_match": 13})

    def test_a_detections_table_is_never_headed_by_a_green_ratio(self, ansi):
        """Even where every hit is a next-gen verdict the summary leaves out."""
        output = ansi(
            lambda c: print_report_summary(
                {
                    "av_scanners_summary": {"scanner_count": 40, "scanner_match_count": 0},
                    "av_scanners": [{"nextgen_scanners": self.DETECTIONS}],
                },
                console=c,
            )
        )
        assert "AV Detections" in output
        assert GREEN not in output


class TestTheRatioIsColouredByHowManyAgree:
    """Green means "no engine flagged this file", and nothing else.

    The two reassuring branches of the colour ladder were the only ones the
    suite never executed — and they are the code that decides whether a
    report looks alarming.
    """

    def _scanners(self, hits: int, total: int) -> dict[str, Any]:
        rows: list[dict[str, Any]] = [
            {"name": f"AV{index}", "result": "Win32.Ransom.Conti"} for index in range(hits)
        ]
        rows += [{"name": f"AVClean{index}", "result": ""} for index in range(total - hits)]
        return {"av_scanners": [{"regular_scanners": rows}]}

    @pytest.mark.parametrize(
        ("hits", "total", "colour"),
        [(9, 14, RED), (3, 14, YELLOW), (0, 14, GREEN)],
        ids=["a_majority_is_red", "a_minority_is_yellow", "none_is_green"],
    )
    def test_the_ratio_wears_the_colour_its_count_earns(self, ansi, hits, total, colour):
        output = ansi(lambda c: print_report_summary(self._scanners(hits, total), console=c))
        assert f"{colour}{hits}/{total}" in output

    def test_a_file_nothing_flagged_draws_no_detections_table(self):
        assert "AV Detections" not in _plain(
            lambda c: print_report_summary(self._scanners(0, 14), console=c)
        )


class TestNeitherHalfOfTheRatioIsInvented:
    """A green "0/17" over a stated engine count, and a red "13/13" over none.

    ``ScannerConsensus`` reports a half nothing states as ``None`` so that a
    number we were not told is not a zero. Defaulting the numerator painted
    "0/17" green — this tool calling a file clean over a count it never
    received — and defaulting the denominator to the match count asserted
    that every engine which ran flagged the file. Both renderers read the
    same object, so both say "?" about the same half.
    """

    STATED_ENGINES: ClassVar[dict[str, Any]] = {
        "sha256": "a" * 64,
        "av_scanners_summary": {"scanner_count": 17},
    }
    STATED_MATCHES: ClassVar[dict[str, Any]] = {"status": "MALICIOUS", "scanner_match": 13}

    def test_an_engine_count_alone_reports_no_detections_it_was_not_told_of(self):
        rendered = _plain(lambda c: print_report_summary(self.STATED_ENGINES, console=c))
        assert "?/17" in rendered
        assert "0/17" not in rendered

    def test_an_unknown_numerator_is_not_painted_clean(self, ansi):
        """Green is the report saying "no engine flagged this file"."""
        output = ansi(lambda c: print_report_summary(self.STATED_ENGINES, console=c))
        assert "?/17" in output
        assert GREEN not in output

    def test_a_match_count_alone_is_not_also_the_engine_count(self):
        rendered = _plain(lambda c: print_report_summary(self.STATED_MATCHES, console=c))
        assert "13/?" in rendered
        assert "13/13" not in rendered

    def test_the_titaniumcloud_sibling_says_the_same_of_the_same_record(self):
        rendered = _plain(
            lambda c: print_file_analysis(
                {"rl": {"malware_presence": self.STATED_MATCHES}}, console=c
            )
        )
        assert "13/?" in rendered
        assert "13/13" not in rendered


class TestACutPanelSaysHowMuchItHid:
    """Four panels sliced their list and printed no tail at all.

    Their siblings in the same file — Indicators, Additional Tags — and the
    ATT&CK table all say how much they hid, so a panel that stops at ten
    with nothing under it reads as the whole answer. The two capabilities
    an analyst opens the panel for were the ones cut.
    """

    PAYLOAD: ClassVar[dict[str, Any]] = {
        "ticore": {
            "behaviour": [f"filler behaviour {index}" for index in range(11)]
            + ["encrypts every mounted drive", "deletes the backups"],
            "certificate": {
                "signatures": [{"name": f"filler sig {index}"} for index in range(11)]
                + [{"name": "REVOKED code-signing certificate", "severity": "high"}]
            },
        },
        "tags": {
            "ticore": [f"capability-filler-{index}" for index in range(11)]
            + ["capability-ransomware-encryption", "capability-keylogging"]
            + [f"indicator-filler-{index}" for index in range(9)]
            + ["indicator-anti-vm"]
        },
    }

    def _render(self) -> str:
        return _plain(lambda c: print_report_summary(self.PAYLOAD, console=c))

    @pytest.mark.parametrize(
        "tail",
        ["... and 3 more", "... and 2 more signatures", "... and 5 more", "... and 2 more"],
        ids=["behaviour", "signatures", "capabilities", "indicator_types"],
    )
    def test_every_capped_panel_states_what_it_left_out(self, tail):
        assert tail in self._render()

    def test_an_uncut_panel_grows_no_tail(self):
        rendered = _plain(
            lambda c: print_report_summary(
                {"ticore": {"behaviour": ["writes to the registry"]}}, console=c
            )
        )
        assert "Behavioral Analysis" in rendered
        assert "more" not in rendered

    def test_the_attack_table_states_what_it_left_out(self):
        """16 techniques into 15 rows: the tail existed and was never run."""
        rendered = _plain(
            lambda c: print_report_summary(
                {
                    "ticore": {
                        "attack": [
                            {
                                "tactics": [
                                    {
                                        "name": "Impact",
                                        "techniques": [
                                            {"id": f"T{1400 + index}", "name": f"technique {index}"}
                                            for index in range(16)
                                        ],
                                    }
                                ]
                            }
                        ]
                    }
                },
                console=c,
            )
        )
        assert "and 1 more techniques" in rendered

    def test_the_additional_tags_panel_states_what_it_left_out(self):
        rendered = _plain(
            lambda c: print_report_summary(
                {"tags": {"ticore": [f"language-{index}" for index in range(14)]}}, console=c
            )
        )
        assert "... and 4 more" in rendered

    def test_the_sources_table_states_what_it_left_out(self):
        rendered = _plain(
            lambda c: print_file_analysis(
                {
                    "rl": {
                        "sample": {
                            "sha1": "a" * 40,
                            "sources": [{"domain": f"host{index}.evil.tld"} for index in range(13)],
                        }
                    }
                },
                console=c,
            )
        )
        assert "and 3 more sources" in rendered

    def test_a_bare_string_signature_still_draws_a_bullet(self):
        """TitaniumCore states a signature as a string as readily as a dict."""
        assert "• Acme Corp" in _plain(
            lambda c: print_report_summary(
                {"ticore": {"certificate": {"signatures": ["Acme Corp"]}}}, console=c
            )
        )

    def test_a_list_valued_behaviour_member_is_one_bullet_each(self):
        """``behaviour`` is an object whose members are the entry lists."""
        rendered = _plain(
            lambda c: print_report_summary(
                {"ticore": {"behaviour": {"network": ["resolves evil.tld", "opens a socket"]}}},
                console=c,
            )
        )
        assert "• resolves evil.tld" in rendered
        assert "• opens a socket" in rendered


class TestTheCapKeepsTheWorstRow:
    """The caps took API order, so the one malicious row was the one cut.

    ``print_extracted_files_table`` already sorts worst verdict first; these
    two siblings drew whatever the appliance happened to list first. A page
    of known CDN domains with the C2 stated last hid the C2 inside "and 6
    more", and a search page of goodware with one Conti sample last showed
    twenty green rows while ``-o sarif`` over the same list emitted an
    ``error`` — the screen saying clean and the CI gate saying otherwise
    about one answer.
    """

    IOCS: ClassVar[dict[str, Any]] = {
        "domainthreatintelligence": {
            "domain": [
                {"domain": f"cdn{index}.example.com", "classification": "known"}
                for index in range(20)
            ]
            + [{"domain": "conti-c2.onion.pet", "classification": "malicious"}]
        }
    }
    HITS: ClassVar[list[dict[str, Any]]] = [
        {"sha256": f"{index:064x}", "classification": "goodware"} for index in range(20)
    ] + [
        {
            "sha256": "f" * 64,
            "classification": "malicious",
            "classification_result": "Win64.Ransomware.Conti",
        }
    ]

    def test_the_c2_survives_the_indicator_cap(self):
        rendered = _plain(lambda c: print_report_summary(self.IOCS, console=c))
        assert "conti-c2.onion.pet" in rendered
        assert rendered.index("conti-c2.onion.pet") < rendered.index("cdn0.example.com")

    def test_the_indicator_cap_still_says_how_much_it_hid(self):
        assert "and 6 more indicators" in _plain(
            lambda c: print_report_summary(self.IOCS, console=c)
        )

    def test_indicators_of_equal_severity_keep_the_order_they_arrived_in(self):
        rendered = _plain(lambda c: print_report_summary(self.IOCS, console=c))
        assert rendered.index("cdn0.example.com") < rendered.index("cdn1.example.com")

    def test_a_section_stated_as_something_other_than_a_list_is_skipped(self):
        rendered = _plain(
            lambda c: print_report_summary(
                {"networkthreatintelligence": {"ip": "not-a-list", "url": [{"url": "http://x/"}]}},
                console=c,
            )
        )
        assert "http://x/" in rendered

    def test_the_malicious_hit_survives_the_search_cap(self, console: Console) -> None:
        assert print_search_results_table(self.HITS, max_rows=5, console=console) == 5
        # The threat column is clipped to its own width; the family is what
        # has to survive the cap, not the whole dotted name.
        assert "Win64.Ransomware" in console_text(console)

    def test_the_screen_and_the_sarif_log_agree_on_what_the_page_held(
        self, console: Console
    ) -> None:
        print_search_results_table(self.HITS, console=console)
        assert "malicious" in console_text(console)
        assert "error" in {result["level"] for result in to_sarif(self.HITS)["runs"][0]["results"]}

    def test_hits_of_equal_severity_keep_the_order_the_appliance_ranked_them_in(
        self, console: Console
    ) -> None:
        hits = [{"sha256": str(index) * 64, "classification": "goodware"} for index in range(3)]
        print_search_results_table(hits, console=console)
        rendered = console_text(console)
        assert rendered.index("1" * 8) < rendered.index("2" * 8)


class TestAnExplicitNullIsNotAValue:
    """``.get(key, "N/A")`` fires its default only when the key is absent.

    Both APIs state an unset field as an explicit JSON null, so the panels
    read ``Risk Score: None`` and ``First Seen: None`` — the literal word,
    which reads as an answer rather than as the absence of one.
    """

    def test_the_status_panel_says_n_a_for_a_stated_null(self):
        rendered = _plain(
            lambda c: print_analysis_status(
                {
                    "classification": "malicious",
                    "riskscore": None,
                    "first_seen": None,
                    "last_seen": None,
                    "data_source": None,
                },
                console=c,
            )
        )
        assert "None" not in rendered
        assert "Risk Score: N/A" in rendered
        assert "Data Source: N/A" in rendered

    def test_the_metadata_panel_says_n_a_for_a_stated_null(self):
        rendered = _plain(
            lambda c: print_report_summary(
                {"file_type": "PE32 executable", "category": None, "riskscore": None}, console=c
            )
        )
        assert "None" not in rendered
        assert "Category: N/A" in rendered

    def test_the_summary_panel_says_n_a_for_a_stated_null(self):
        assert "None" not in _plain(
            lambda c: print_summary_panel({"riskscore": None, "file_type": None}, console=c)
        )

    def test_the_threat_panel_says_n_a_for_a_stated_null(self):
        assert "None" not in _plain(
            lambda c: print_report_summary({"trust_factor": None, "threat_level": None}, console=c)
        )

    def test_a_stated_zero_is_still_a_zero(self):
        """The guard must not turn a risk score of 0 into "no answer"."""
        assert "Risk Score: 0" in _plain(
            lambda c: print_analysis_status({"riskscore": 0}, console=c)
        )

    def test_an_elf_sample_gets_its_platform_row(self):
        assert "Platform: Linux/Unix (x86-64)" in _plain(
            lambda c: print_report_summary({"file_type": "ELF 64-bit LSB executable"}, console=c)
        )


class TestAnUnstatedConsensusIsNotAZero:
    """``Detections: 0/37`` printed under a ``Threat: Win32.Conti`` panel.

    ``malware_presence`` need not carry ``scanner_match``; defaulting it to
    zero rendered "no engine flagged this file" about a record whose status
    was MALICIOUS. The same class of defect as the A1000 ratio above.
    """

    def _render(self, presence: dict[str, Any]) -> str:
        return _plain(
            lambda c: print_file_analysis({"rl": {"malware_presence": presence}}, console=c)
        )

    def test_an_unstated_match_count_is_not_reported_as_no_detections(self):
        rendered = self._render(
            {"status": "MALICIOUS", "threat_name": "Win32.Conti", "scanner_count": 37}
        )
        assert "?/37" in rendered
        assert "0/37" not in rendered

    def test_a_stated_zero_is_still_reported_as_zero(self):
        assert "0/37" in self._render({"status": "KNOWN", "scanner_count": 37, "scanner_match": 0})

    def test_a_next_gen_array_is_accounted_for_beside_the_ratio(self):
        """The ratio counts signature engines; the ML verdicts say so here.

        Drawn without this row, a record carrying both arrays showed a number
        the reader could only take for every engine that ran.
        """
        rendered = self._render(
            {
                "status": "MALICIOUS",
                "scanner_count": 10,
                "scanner_match": 2,
                "av_scanners": [
                    {
                        "nextgen_scanners": [
                            {"name": f"NG{index}", "result": "bad"} for index in "12"
                        ]
                    }
                ],
            }
        )
        assert "2/10" in rendered
        assert "Next-Gen (ML)" in rendered
        assert "2/2" in rendered

    def test_a_record_with_no_scanner_arrays_draws_no_next_gen_row(self):
        assert "Next-Gen" not in self._render(
            {"status": "MALICIOUS", "scanner_count": 10, "scanner_match": 2}
        )


class TestAStatedZeroEngineCountDrawsNoRatioAtAll:
    """One renderer drew "?/0" where the other drew nothing.

    ``scanner_count: 0`` says zero engines ran, so a ratio panel about it
    reports nothing the rest of the report does not say better. The A1000
    summary already suppressed it and the TitaniumCloud report did not, so
    one record was answered two ways. Both suppress it now.
    """

    ZERO_ENGINES: ClassVar[dict[str, Any]] = {"status": "KNOWN", "scanner_count": 0}

    def test_the_a1000_summary_draws_no_av_section(self):
        rendered = _plain(
            lambda c: print_report_summary(
                {"classification": "known", "av_scanners_summary": self.ZERO_ENGINES}, console=c
            )
        )
        assert "AV Scanner" not in rendered
        assert "/0" not in rendered

    def test_the_titaniumcloud_report_draws_no_av_section_either(self):
        rendered = _plain(
            lambda c: print_file_analysis(
                {"rl": {"malware_presence": self.ZERO_ENGINES}}, console=c
            )
        )
        assert "AV Scanners" not in rendered
        assert "/0" not in rendered

    @pytest.mark.parametrize(
        "render",
        [
            lambda payload, c: print_report_summary(
                {"classification": "malicious", "av_scanners_summary": payload}, console=c
            ),
            lambda payload, c: print_file_analysis(
                {"rl": {"malware_presence": payload}}, console=c
            ),
        ],
        ids=["a1000", "ticloud"],
    )
    def test_engines_that_did_run_still_draw_their_ratio(self, render):
        """The suppression is about a zero, not about a half-stated ratio."""
        rendered = _plain(lambda c: render({"status": "MALICIOUS", "scanner_match": 13}, c))
        assert "13/?" in rendered


class TestATruncatedPanelListingKeepsTheWorstSample:
    """The cap has to drop the least interesting entries, not arbitrary ones.

    ``print_samples_panels`` took the appliance's order, so a page whose
    one malicious sample came back last was exactly the panel the cap
    dropped -- the same defect its two sibling listings were already
    sorted to avoid.
    """

    def _samples(self):
        clean = [
            {"sha256": f"{i:064x}", "classification": "goodware", "file_name": f"clean{i}.dll"}
            for i in range(12)
        ]
        return [
            *clean,
            {
                "sha256": "f" * 64,
                "classification": "malicious",
                "threat_name": "Win32.Ransom.Conti",
                "file_name": "invoice.exe",
            },
        ]

    def test_the_malicious_sample_survives_the_cap(self, console: Console) -> None:
        drawn = print_samples_panels(self._samples(), max_panels=10, console=console)

        assert drawn == 10
        assert "Win32.Ransom.Conti" in console.export_text()

    def test_the_worst_sample_is_the_first_panel(self, console: Console) -> None:
        print_samples_panels(self._samples(), max_panels=10, console=console)

        rendered = console.export_text()
        assert rendered.index("invoice.exe") < rendered.index("clean0.dll")

    def test_the_count_still_names_the_whole_answer(self, console: Console) -> None:
        """``Sample 1/13``: the cap is not allowed to shrink the total."""
        print_samples_panels(self._samples(), max_panels=10, console=console)

        assert "1/13" in console.export_text()


class TestTheCapTailFitsTheTableItIsAddedTo:
    """One cell per column, whatever the caller declared.

    Rich does not report a row with more cells than the table has columns;
    it appends a blank-headed column, so a one-column table put the count
    under a heading it never declared.
    """

    @staticmethod
    def _capped(*headers: str) -> str:
        table = Table(show_header=True)
        for header in headers:
            table.add_column(header)
        add_capped_rows(
            table, [(f"row {index}",) * len(headers) for index in range(5)], limit=2, noun="sources"
        )
        console = Console(file=io.StringIO(), width=100, record=True)
        console.print(table)
        return console_text(console)

    def test_a_one_column_table_states_the_cut_and_the_count_in_its_one_cell(self):
        rendered = self._capped("Source")

        assert "... and 3 more sources" in " ".join(rendered.split())
        assert rendered.count("Source") == 1, "the tail declared a second column"

    def test_a_wider_table_keeps_the_mark_and_the_count_at_its_two_ends(self):
        rendered = " ".join(self._capped("Source", "Detection", "Version").split())

        assert "│ ... │ │ and 3 more sources │" in rendered


class TestAvailabilityPanel:
    """The head of `check-access`'s report, drawn from the payload it printed.

    ``tests/test_cli_config.py`` owns the payload's shape; the per-service
    lines are drawn by ``render.formatters.config_report`` and graded
    against ``ServiceStatus``, which lives in ``rl_cli.models`` and which
    this package therefore may import.
    """

    def test_the_panel_states_the_count_the_payload_carries(self, console: Console) -> None:
        """Drawn from the payload, so it cannot grade the run differently."""
        payload = {
            "titanium_cloud": {"status": "available", "message": "probe says available"},
            "a1000": {"status": "error", "message": "probe says error"},
            "timestamp": "2026-01-01T00:00:00",
            "summary": {"services_available": 1, "services_total": 2},
        }

        print_availability_panel(payload, console=console)

        rendered = " ".join(console_text(console).split())
        assert "API Availability Status" in rendered
        assert "Services Available: 1/2" in rendered
        assert "Last Checked: 2026-01-01T00:00:00" in rendered


class TestProfileNames:
    """The profile list, whose marker is chatter and whose names are the data."""

    @staticmethod
    def _rendered(names: list[str], current: str) -> tuple[str, str]:
        data, status = io.StringIO(), io.StringIO()
        console = Console(file=data, width=100)
        print_profile_names(
            names,
            output=RichOutput(console, Console(file=status, width=100)),
            current=current,
        )
        return data.getvalue(), status.getvalue()

    def test_the_current_profile_is_marked(self):
        _, status = self._rendered(["default", "staging"], "default")

        assert "- default (current)" in status
        assert "- staging" in status
        assert "staging (current)" not in status

    def test_nothing_is_drawn_on_stdout(self):
        """``display`` puts the bare list there; the marker is chatter."""
        data, _ = self._rendered(["default", "staging"], "default")

        assert data == ""

    def test_it_takes_no_console_and_says_so_at_the_call_site(self):
        """A renderer drawing elsewhere declares that, rather than dropping one.

        Taking a ``console`` and never loading it reads, to anyone changing
        the console policy, like a renderer that honours it. ``display``
        still needs a :class:`RichRenderer`, so the seam is
        ``cli.context.without_console``.
        """
        assert "console" not in inspect.signature(print_profile_names).parameters
        listing = config_commands.list_profiles.callback
        assert listing is not None
        assert "without_console(" in inspect.getsource(listing)


class TestBothReportsDrawTheSameAvConsensus:
    """``a1000 report`` and ``ticloud file-analysis`` read one ``ScannerConsensus``.

    They drew it two ways: the A1000 panel coloured the ratio and dropped
    the share the record stated, the TitaniumCloud panel printed the share
    and painted nothing at all — so the same 13 of 17 arrived red in one
    report and white in the next. Whether a ratio is a consensus is a
    judgement about the data and belongs to the model; what the ratio reads
    as is one helper both panels call.
    """

    RECORD: ClassVar[dict[str, Any]] = {
        "sha256": "a" * 64,
        "status": "MALICIOUS",
        "scanner_count": 17,
        "scanner_match": 13,
        "scanner_percent": 76.47,
    }

    @staticmethod
    def _a1000(console: Console) -> None:
        print_report_summary(TestBothReportsDrawTheSameAvConsensus.RECORD, console=console)

    @staticmethod
    def _ticloud(console: Console) -> None:
        print_file_analysis(
            {"rl": {"malware_presence": TestBothReportsDrawTheSameAvConsensus.RECORD}},
            console=console,
        )

    UNSHARED: ClassVar[dict[str, Any]] = {
        key: value for key, value in RECORD.items() if key != "scanner_percent"
    }

    @staticmethod
    def _a1000_without_the_share(console: Console) -> None:
        print_report_summary(TestBothReportsDrawTheSameAvConsensus.UNSHARED, console=console)

    @staticmethod
    def _ticloud_without_the_share(console: Console) -> None:
        print_file_analysis(
            {"rl": {"malware_presence": TestBothReportsDrawTheSameAvConsensus.UNSHARED}},
            console=console,
        )

    @pytest.mark.parametrize("render", [_a1000, _ticloud], ids=["a1000", "ticloud"])
    def test_both_state_the_share_the_record_carries(self, render):
        """Drawn by both reports or by neither: the A1000 body dropped it."""
        assert "13/17 (76.47%)" in _plain(render)

    @pytest.mark.parametrize(
        "render",
        [_a1000_without_the_share, _ticloud_without_the_share],
        ids=["a1000", "ticloud"],
    )
    def test_neither_computes_a_share_the_record_left_out(self, render):
        """A share is a number the record states, not one this tool divides."""
        rendered = _plain(render)
        assert "13/17" in rendered
        assert "%" not in rendered

    @pytest.mark.parametrize("render", [_a1000, _ticloud], ids=["a1000", "ticloud"])
    def test_both_paint_a_majority_red(self, ansi, render):
        assert f"{RED}13/17" in ansi(render)


class TestAnUnreadableEngineResultIsNoDetectionInEitherReport:
    """One control character from a third-party engine turned 0/17 yellow.

    The engine array carries an entry whose ``result`` is a bidi override
    and nothing else — no readable character in it — beside a summary
    stating ``scanner_match_count: 0``. Counting it as a hit reports
    ``1/17``, paints the ratio the colour of a partial consensus in both
    reports, and draws a detections table under ``a1000 report`` whose one
    row has a blank Detection cell: a detection an analyst can neither read
    nor confirm, on a file no engine flagged.
    """

    # Two that ``sanitize`` strips and three it does not, so this cannot
    # agree with the check by construction the way its first spelling did.
    UNREADABLE: ClassVar[tuple[str, ...]] = ("‮", "\x00", "​", "﻿", "‏")
    UNREADABLE_IDS: ClassVar[list[str]] = ["bidi_override", "nul", "zero_width", "bom", "bidi_mark"]

    @staticmethod
    def _record(result: str) -> dict[str, Any]:
        return {
            "sha256": "a" * 64,
            "av_scanners_summary": {"scanner_count": 17, "scanner_match_count": 0},
            "av_scanners": [{"regular_scanners": [{"name": "AV1", "result": result}]}],
        }

    @classmethod
    def _a1000(cls, result: str):
        return lambda console: print_report_summary(cls._record(result), console=console)

    @classmethod
    def _ticloud(cls, result: str):
        return lambda console: print_file_analysis(
            {"rl": {"malware_presence": cls._record(result)}}, console=console
        )

    @pytest.mark.parametrize("render", ["_a1000", "_ticloud"], ids=["a1000", "ticloud"])
    @pytest.mark.parametrize("result", UNREADABLE, ids=UNREADABLE_IDS)
    def test_neither_report_raises_the_ratio(self, render, result):
        rendered = _plain(getattr(self, render)(result))
        assert "0/17" in rendered
        assert "1/17" not in rendered

    @pytest.mark.parametrize("render", ["_a1000", "_ticloud"], ids=["a1000", "ticloud"])
    @pytest.mark.parametrize("result", UNREADABLE, ids=UNREADABLE_IDS)
    def test_neither_report_takes_the_green_off_it(self, ansi, render, result):
        assert f"{GREEN}0/17" in ansi(getattr(self, render)(result))

    @pytest.mark.parametrize("result", UNREADABLE, ids=UNREADABLE_IDS)
    def test_no_phantom_row_is_drawn_under_the_ratio(self, result):
        assert "AV Detections" not in _plain(self._a1000(result))


class TestOneWordForWhatThePayloadDidNotSay:
    """One spelling of "the payload did not say" per place it is said.

    Within one ``a1000 search`` an absent file type rendered ``Unknown``
    while the absent threat name beside it rendered ``-``, and ``a1000
    report`` wrote ``N/A`` for the same field. A table cell is one of five
    on a line and gets the mark; a panel field has room for the phrase.
    """

    def test_the_search_table_marks_every_absent_cell_the_same_way(self):
        rendered = _plain(lambda c: print_search_results_table([{"riskscore": 1}], console=c))
        assert "Unknown" not in rendered
        assert "N/A" not in rendered

    def test_the_extracted_files_table_marks_them_the_same_way(self):
        rendered = _plain(lambda c: print_extracted_files_table([{"file_size": 10}], console=c))
        assert "Unknown" not in rendered
        assert "N/A" not in rendered

    @pytest.mark.parametrize(
        "entry",
        [{"riskscore": 1}, {"file_size": None}, {"file_size": "not-a-size"}],
        ids=["no_size_key", "explicit_null", "unusable"],
    )
    def test_the_extracted_files_size_column_marks_an_absent_size_the_same_way(self, entry):
        """The Size cell answered ``N/A`` beside five cells answering ``-``."""
        rendered = _plain(lambda c: print_extracted_files_table([entry], console=c))
        assert "N/A" not in rendered
        assert "-" in rendered

    @pytest.mark.parametrize(
        "entry",
        [{"riskscore": 1}, {"file_size": None}, {"file_size": "not-a-size"}],
        ids=["no_size_key", "explicit_null", "unusable"],
    )
    def test_the_search_size_column_marks_an_absent_size_the_same_way(self, entry):
        """A size the payload states as something unusable is still an absent size."""
        rendered = _plain(lambda c: print_search_results_table([entry], console=c))
        assert "N/A" not in rendered
        assert "-" in rendered

    def test_the_ruleset_table_marks_an_unnamed_ruleset_the_same_way(self):
        rendered = _plain(lambda c: print_yara_rulesets_table([{"status": "active"}], console=c))
        assert "N/A" not in rendered

    def test_a_sample_panel_spells_out_every_absent_field(self):
        rendered = _plain(lambda c: print_samples_panels([{"file_size": 1}], console=c))
        assert "File: N/A" in rendered
        assert "Threat: N/A" in rendered
        assert "Status: N/A" in rendered

    def test_a_panel_bullet_with_no_name_spells_it_out_too(self):
        rendered = _plain(
            lambda c: print_report_summary({"ticore": {"behaviour": [{"category": 3}]}}, console=c)
        )
        assert "• N/A" in rendered


class TestAnEmptyStringIsAFieldThePayloadDidNotState:
    """The third spelling of "unset", read as the same answer as the other two.

    Both APIs state a field they have nothing for three ways -- the key
    absent, an explicit JSON null, and ``""`` -- and the panels used to
    answer the first two with ``N/A`` and the third with a bare
    ``Category:`` and nothing after it. That line reads as the report having
    broken off rather than as the appliance having no answer, which is the
    one thing the row exists to say. So all three read alike: ``N/A`` in a
    panel, ``-`` in a cell, and no row at all in File Information.
    """

    @pytest.mark.parametrize(
        ("render", "expected"),
        [
            (lambda c: print_analysis_status({"data_source": ""}, console=c), "Data Source: N/A"),
            (lambda c: print_analysis_status({"riskscore": ""}, console=c), "Risk Score: N/A"),
            (lambda c: print_summary_panel({"file_type": ""}, console=c), "File Type: N/A"),
            (lambda c: print_report_summary({"category": ""}, console=c), "Category: N/A"),
            (lambda c: print_report_summary({"trust_factor": ""}, console=c), "Trust Factor: N/A"),
            (lambda c: print_report_summary({"threat_name": ""}, console=c), "Threat Name: N/A"),
            (
                # An undotted threat name, so the platform this panel reads
                # off a "Platform.Type.Family" string is not what is under
                # test here.
                lambda c: print_report_summary(
                    {"classification": {"platform": ""}, "threat_name": "Conti"}, console=c
                ),
                "Platform: N/A",
            ),
        ],
        ids=["data_source", "risk_score", "file_type", "category", "trust", "threat", "platform"],
    )
    def test_a_panel_field_stated_as_an_empty_string_says_it_has_no_answer(self, render, expected):
        assert expected in _plain(render)

    def test_a_table_cell_stated_as_an_empty_string_takes_the_cell_mark(self):
        """Not ``N/A``: a cell is one of six across a line and gets the mark."""
        rendered = _plain(lambda c: print_search_results_table([{"threat_name": ""}], console=c))
        assert "N/A" not in rendered
        assert "-" in rendered

    def test_file_information_leaves_out_a_row_stated_as_an_empty_string(self):
        """It draws what a record states, and four ``N/A`` rows say less than none."""
        rendered = _plain(
            lambda c: print_file_information(
                SampleFacts.of({"file_name": "", "file_type": "", "sha256": "a" * 64}), console=c
            )
        )
        assert "File Name" not in rendered
        assert "File Type" not in rendered
        assert "SHA256" in rendered


class TestOneSentenceForWhatAListingHid:
    """Every cap in the report says how much it hid, in one sentence.

    The tables named what they left out — "and 6 more indicators" — and the
    panels beside them said "... and 8 more" about nothing in particular, so
    one report stated its truncation two ways.
    """

    @staticmethod
    def _tags(prefix: str, count_: int) -> dict[str, Any]:
        return {"tags": {"ticore": [f"{prefix}{index}" for index in range(count_)]}}

    @pytest.mark.parametrize(
        ("payload", "tail"),
        [
            (_tags("capability-c", 12), "and 4 more capabilities"),
            (_tags("indicator-i", 12), "and 4 more indicator types"),
            (_tags("language-l", 14), "and 4 more tags"),
            (
                {"ticore": {"behaviour": [f"does thing {i}" for i in range(13)]}},
                "and 3 more behaviours",
            ),
        ],
        ids=["capabilities", "indicator_types", "other_tags", "behaviour"],
    )
    def test_a_capped_panel_names_what_it_hid(self, payload, tail):
        assert tail in _plain(lambda c: print_report_summary(payload, console=c))

    def test_an_uncapped_panel_draws_every_item_and_no_tail(self):
        rendered = _plain(lambda c: print_report_summary(self._tags("protection-p", 12), console=c))
        assert "P11" in rendered
        assert "more" not in rendered


def test_a_scanner_entry_is_escaped_by_the_renderer_that_draws_it(ansi) -> None:
    """The model carries the appliance's bytes; this is where they are made safe."""
    output = ansi(
        lambda c: print_report_summary(
            {"av_scanners": [{"regular_scanners": [{"name": HOSTILE, "result": HOSTILE}]}]},
            console=c,
        )
    )
    assert "\x1b[2J" not in output
    assert "\x1b]0;" not in output


def test_an_unnamed_scanner_is_labelled_by_the_renderer() -> None:
    """``models`` states no presentation string; the table marks the cell."""
    rendered = _plain(
        lambda c: print_report_summary(
            {"av_scanners": [{"regular_scanners": [{"result": "Win32.Conti"}]}]}, console=c
        )
    )
    assert "Unknown" not in rendered
    assert "Win32.Conti" in rendered


class TestNoRendererDecidesTheRunStatus:
    """A renderer draws what it is handed; it does not grade it or fail the run.

    Both halves are read off the source rather than exercised, because both
    break silently: ``config_report`` graded probe statuses against
    ``ServiceStatus`` and picked the reporting method from the grade, and
    the choice between ``problem`` and ``error`` is exactly the choice of
    whether this invocation exits non-zero (``render/output.py``). Nothing
    failed, because the layer table allows ``render.formatters`` to read
    both ``render`` and ``models``.
    """

    @staticmethod
    def _sources(root: Path) -> list[tuple[str, ast.Module]]:
        return [
            (str(path.relative_to(_PACKAGE_ROOT)), ast.parse(path.read_text(encoding="utf-8")))
            for path in sorted(root.rglob("*.py"))
        ]

    def test_no_renderer_grades_a_probe_status(self):
        """Which outcome is good news is a fact about the measurement.

        So it is answered once, in ``models``, and read the same way by the
        document ``-o json`` writes and by the lines a terminal gets. A
        renderer that compares statuses itself is a second grading, free to
        drift from the first.
        """
        offenders = [
            module
            for module, tree in self._sources(_PACKAGE_ROOT / "render")
            if any(
                isinstance(node, ast.Name) and node.id == "ServiceStatus" for node in ast.walk(tree)
            )
        ]

        assert not offenders, f"render grades a status instead of drawing a grade: {offenders}"

    def test_no_formatter_reaches_the_method_that_fails_the_run(self):
        """``error`` is the one reporting method that sets the exit status.

        ``problem`` draws the same red line and leaves the run's fate
        alone, which is what a measurement gets. A formatter reaching for
        ``error`` — or for ``RunStatus.fail`` under it — is a rendering
        deciding whether ``rl-cli`` exited 1.
        """
        offenders = [
            module
            for module, tree in self._sources(_PACKAGE_ROOT / "render" / "formatters")
            if any(
                isinstance(node, ast.Attribute) and node.attr in {"error", "fail"}
                for node in ast.walk(tree)
            )
        ]

        assert not offenders, f"a formatter can fail the run: {offenders}"
