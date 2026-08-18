"""Everything the download commands write to the analyst's disk.

``download`` and ``extracted --download-all`` fetch live malware, so what
these pin is the order of events: the hash is checked, the analyst is
warned, and only then is anything written.

``a1000 download`` and ``ticloud download`` are one command spelled for
two APIs, and were two copies of the same six lines until one of them
grew a cleanup the other did not. They go through one helper now, so the
claims about it are parametrized over both groups rather than asserted of
whichever copy someone remembered to test.
"""

import stat
from collections.abc import Callable
from pathlib import Path

import pytest
from rich.progress import Progress

from rl_cli.cli.commands.a1000 import a1000
from rl_cli.cli.commands.ticloud import ticloud
from rl_cli.services.a1000 import A1000SampleService
from rl_cli.services.titanium_cloud import TitaniumCloudService
from tests.cli_support import SHA1, SHA256, flat, invoke, make_context

# The two ``download`` commands, with the service each fetches through.
DOWNLOAD_TWINS = [
    pytest.param(a1000, A1000SampleService, id="a1000"),
    pytest.param(ticloud, TitaniumCloudService, id="ticloud"),
]


@pytest.fixture
def ctx_obj(tmp_path):
    return make_context(tmp_path)


class TestExtractedDownloadAnnouncesWhatItDoes:
    """The listing is one page; the download fetches the whole archive."""

    def test_the_spinner_does_not_claim_the_listings_count(self, ctx_obj, monkeypatch, tmp_path):
        spinners: list[str] = []
        output = ctx_obj.output
        spinner: Callable[[str], Progress] = output.progress_spinner

        def record_spinner(message: str = "Processing...") -> Progress:
            spinners.append(message)
            return spinner(message)

        monkeypatch.setattr(output, "progress_spinner", record_spinner)
        monkeypatch.setattr(
            A1000SampleService,
            "list_extracted_files",
            lambda self, h: [{"sha1": SHA1} for _ in range(100)],
        )
        monkeypatch.setattr(
            A1000SampleService, "download_extracted_files", lambda self, h, directory: True
        )

        result = invoke(
            a1000,
            ["extracted", SHA256, "--download-all", "--output-dir", str(tmp_path)],
            ctx_obj,
        )

        assert result.exit_code == 0, result.output
        assert "Downloading extracted files..." in spinners
        assert not any("100 files" in message for message in spinners), "one page is not the whole"


class TestDownloadsWarnBeforeTheyWrite:
    """The analyst must hear about the malware while the disk is still clean.

    Both commands printed the warning after the download returned, and
    ``--output-dir`` defaults to the working directory, so the first the
    analyst heard of a live sample was that it was already sitting in the
    directory they were standing in. Ordering is what these assert:
    presence alone passed for the whole time it was wrong.
    """

    @staticmethod
    def _recorder(ctx_obj, monkeypatch) -> list[tuple[str, str]]:
        """Record warnings and writes on one timeline, in the order they happen."""
        events: list[tuple[str, str]] = []
        monkeypatch.setattr(
            ctx_obj.output, "warning", lambda message: events.append(("warning", message))
        )
        return events

    def test_download_warns_before_the_sample_is_written(self, ctx_obj, monkeypatch, tmp_path):
        events = self._recorder(ctx_obj, monkeypatch)

        def download(self, hash_value, output_path):
            events.append(("write", str(output_path)))
            output_path.write_bytes(b"MZ")
            return True

        monkeypatch.setattr(A1000SampleService, "download_sample", download)

        result = invoke(a1000, ["download", SHA256, "--output-dir", str(tmp_path)], ctx_obj)

        assert result.exit_code == 0, result.output
        kinds = [kind for kind, _ in events]
        assert "write" in kinds, "the download never ran"
        assert kinds.index("warning") < kinds.index("write"), "warned only after writing malware"
        assert str(tmp_path) in events[0][1], "the warning does not say where the sample lands"

    def test_extracted_download_all_warns_before_the_files_are_written(
        self, ctx_obj, monkeypatch, tmp_path
    ):
        events = self._recorder(ctx_obj, monkeypatch)
        monkeypatch.setattr(
            A1000SampleService, "list_extracted_files", lambda self, h: [{"sha1": SHA1}]
        )

        def download_all(self, hash_value, output_dir):
            events.append(("write", str(output_dir)))
            (output_dir / "extracted.bin").write_bytes(b"MZ")
            return True

        monkeypatch.setattr(A1000SampleService, "download_extracted_files", download_all)

        result = invoke(
            a1000,
            ["extracted", SHA256, "--download-all", "--output-dir", str(tmp_path)],
            ctx_obj,
        )

        assert result.exit_code == 0, result.output
        kinds = [kind for kind, _ in events]
        assert "write" in kinds, "the download never ran"
        assert kinds.index("warning") < kinds.index("write"), "warned only after writing malware"
        assert str(tmp_path) in events[0][1], "the warning does not say where the files land"

    def test_the_warning_does_not_wait_on_the_download_succeeding(
        self, ctx_obj, monkeypatch, tmp_path
    ):
        """Nothing consults a classification, so nothing gates the warning.

        A warning conditional on the transfer is a warning the analyst gets
        after the bytes have landed, which is the bug above from the other
        side.
        """
        events = self._recorder(ctx_obj, monkeypatch)
        monkeypatch.setattr(A1000SampleService, "download_sample", lambda self, h, path: False)

        invoke(a1000, ["download", SHA256, "--output-dir", str(tmp_path)], ctx_obj)

        assert any("malware" in message for _, message in events)


class TestADownloadThatFetchedNothingLeavesNoDirectoryBehind:
    """A refused sample used to leave the ``--output-dir`` it had made standing.

    ``ticloud download`` removed it again; ``a1000 download``, the same
    six lines with another service under them, did not — so an appliance
    that would not hand the sample over left a folder named for an
    investigation that never happened, and ``ls`` said otherwise. Only a
    directory this command created is removed, and only while it is still
    empty: the analyst's own quarantine folder is not this command's to
    delete.
    """

    @pytest.mark.parametrize("group,service", DOWNLOAD_TWINS)
    def test_a_directory_it_created_for_nothing_is_removed_again(
        self, ctx_obj, monkeypatch, tmp_path, group, service
    ):
        monkeypatch.setattr(service, "download_sample", lambda self, h, path: False)
        target = tmp_path / "quarantine"

        result = invoke(group, ["download", SHA256, "--output-dir", str(target)], ctx_obj)

        assert ctx_obj.output.status.failed, result.output
        assert not target.exists(), "a refused download left an empty directory behind"

    @pytest.mark.parametrize("group,service", DOWNLOAD_TWINS)
    def test_a_directory_the_analyst_already_had_is_left_alone(
        self, ctx_obj, monkeypatch, tmp_path, group, service
    ):
        monkeypatch.setattr(service, "download_sample", lambda self, h, path: False)
        target = tmp_path / "quarantine"
        target.mkdir()

        invoke(group, ["download", SHA256, "--output-dir", str(target)], ctx_obj)

        assert target.is_dir(), "a directory the command did not create was removed"

    @pytest.mark.parametrize("group,service", DOWNLOAD_TWINS)
    def test_a_sample_that_arrives_keeps_the_directory_it_landed_in(
        self, ctx_obj, monkeypatch, tmp_path, group, service
    ):
        monkeypatch.setattr(
            service,
            "download_sample",
            lambda self, h, path: bool(path.write_bytes(b"MZ")) or True,
        )
        target = tmp_path / "quarantine"

        result = invoke(group, ["download", SHA256, "--output-dir", str(target)], ctx_obj)

        assert result.exit_code == 0, result.output
        assert (target / f"{SHA256}.malware").read_bytes() == b"MZ"


class TestAnExtractionThatFetchedNothingLeavesNoDirectoryBehind:
    """The third command that writes live malware, and the last to be joined up.

    ``extracted --download-all`` made ``--output-dir`` *and* the
    ``extracted_<hash>/`` under it, warned, fetched — and on a refused
    archive left both standing, so ``ls`` showed a folder named for an
    investigation that never happened. The two ``download`` commands had
    already been given one body over exactly this; this one had not.
    """

    @staticmethod
    def _listing(monkeypatch) -> None:
        monkeypatch.setattr(
            A1000SampleService, "list_extracted_files", lambda self, h: [{"sha1": SHA1}]
        )

    def test_the_directories_it_created_for_nothing_are_removed_again(
        self, ctx_obj, monkeypatch, tmp_path
    ):
        self._listing(monkeypatch)
        monkeypatch.setattr(
            A1000SampleService, "download_extracted_files", lambda self, h, directory: False
        )
        target = tmp_path / "quarantine"

        result = invoke(
            a1000,
            ["extracted", SHA256, "--download-all", "--output-dir", str(target)],
            ctx_obj,
        )

        assert ctx_obj.output.status.failed, result.output
        assert not target.exists(), "a refused archive left its directories behind"

    def test_a_directory_the_analyst_already_had_is_left_alone(
        self, ctx_obj, monkeypatch, tmp_path
    ):
        self._listing(monkeypatch)
        monkeypatch.setattr(
            A1000SampleService, "download_extracted_files", lambda self, h, directory: False
        )
        target = tmp_path / "quarantine"
        target.mkdir()

        invoke(a1000, ["extracted", SHA256, "--download-all", "--output-dir", str(target)], ctx_obj)

        assert target.is_dir(), "a directory the command did not create was removed"
        assert not (target / f"extracted_{SHA256[:16]}").exists()

    def test_an_archive_that_arrives_keeps_the_directory_it_landed_in(
        self, ctx_obj, monkeypatch, tmp_path
    ):
        self._listing(monkeypatch)
        monkeypatch.setattr(
            A1000SampleService,
            "download_extracted_files",
            lambda self, h, directory: bool((directory / "carved.bin").write_bytes(b"MZ")) or True,
        )
        target = tmp_path / "quarantine"

        result = invoke(
            a1000,
            ["extracted", SHA256, "--download-all", "--output-dir", str(target)],
            ctx_obj,
        )

        assert result.exit_code == 0, result.output
        assert (target / f"extracted_{SHA256[:16]}" / "carved.bin").read_bytes() == b"MZ"

    def test_the_directory_it_carves_out_for_the_files_is_owner_only(
        self, ctx_obj, monkeypatch, tmp_path
    ):
        """``mkdir`` took the umask for a directory holding live malware.

        The files inside land 0600 through ``private_writer``, so what a
        0755 directory exposed was the listing — the carved-out names of
        one sample's payloads — which ``archives.private_mkdir`` is the
        module's own answer to.
        """
        self._listing(monkeypatch)
        monkeypatch.setattr(
            A1000SampleService, "download_extracted_files", lambda self, h, directory: True
        )

        result = invoke(
            a1000,
            ["extracted", SHA256, "--download-all", "--output-dir", str(tmp_path)],
            ctx_obj,
        )

        assert result.exit_code == 0, result.output
        extracted_dir = tmp_path / f"extracted_{SHA256[:16]}"
        assert stat.S_IMODE(extracted_dir.stat().st_mode) == 0o700


class TestDownloadAnnouncesAnOverwrite:
    """Replacing the previous download in silence still printed "Sample downloaded"."""

    def test_an_existing_sample_file_is_announced_before_the_write(
        self, ctx_obj, monkeypatch, tmp_path
    ):
        (tmp_path / f"{SHA256}.malware").write_bytes(b"an earlier download")
        monkeypatch.setattr(A1000SampleService, "download_sample", lambda self, h, path: True)

        result = invoke(a1000, ["download", SHA256, "--output-dir", str(tmp_path)], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "already exists and will be replaced" in flat(result)


class TestTheHashIsCheckedBeforeAnythingIsWrittenOrWarnedAbout:
    """``download`` and ``extracted --download-all`` acted on a typo.

    Both created ``--output-dir`` and printed "about to write live
    malware to ..." before the hash was looked at, so a mistyped digest
    left empty directories in the analyst's tree and a warning about a
    sample that was never going to be fetched. The hash is refused first
    now, and nothing else happens.
    """

    NOT_A_HASH = "definitely-not-a-hash"

    def _target(self, tmp_path):
        return tmp_path / "quarantine"

    def test_download_creates_no_directory_for_a_bad_hash(self, ctx_obj, monkeypatch, tmp_path):
        monkeypatch.setattr(
            A1000SampleService,
            "download_sample",
            lambda self, h, path: pytest.fail("a sample was fetched for a hash that is not one"),
        )
        target = self._target(tmp_path)

        result = invoke(a1000, ["download", self.NOT_A_HASH, "--output-dir", str(target)], ctx_obj)

        assert not target.exists(), "an invalid hash left an empty download directory behind"
        assert "live malware" not in flat(result)
        assert "Invalid hash" in flat(result)
        assert ctx_obj.output.status.failed

    def test_extracted_download_all_creates_no_directory_for_a_bad_hash(
        self, ctx_obj, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            A1000SampleService,
            "list_extracted_files",
            lambda self, h: pytest.fail("the appliance was asked about a hash that is not one"),
        )
        target = self._target(tmp_path)

        result = invoke(
            a1000,
            ["extracted", self.NOT_A_HASH, "--download-all", "--output-dir", str(target)],
            ctx_obj,
        )

        assert not target.exists(), "an invalid hash left an empty extraction directory behind"
        assert "may contain malware" not in flat(result)
        assert "Invalid hash" in flat(result)
        assert ctx_obj.output.status.failed

    def test_a_real_hash_still_writes_where_it_was_told(self, ctx_obj, monkeypatch, tmp_path):
        """The guard must not have moved the warning off the happy path."""
        monkeypatch.setattr(A1000SampleService, "download_sample", lambda self, h, path: True)
        target = self._target(tmp_path)

        result = invoke(a1000, ["download", SHA256, "--output-dir", str(target)], ctx_obj)

        assert result.exit_code == 0, result.output
        assert target.is_dir()
        assert "live malware" in flat(result)


class TestExtractedAllIsHiddenNotAdvertised:
    """It does nothing but warn that it does nothing, so it must not be offered."""

    def test_the_help_no_longer_lists_it(self, ctx_obj):
        result = invoke(a1000, ["extracted", "--help"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "--all" not in result.output
        # The one it must still advertise, which is a different flag.
        assert "--download-all" in result.output

    def test_a_script_still_passing_it_is_not_broken(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(
            A1000SampleService, "list_extracted_files", lambda self, h: [{"sha1": SHA1}]
        )

        result = invoke(a1000, ["extracted", SHA256, "--all"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "deprecated no-op" in flat(result)


class TestTheSampleLandsWhereTheAnalystIsStanding:
    """``--output-dir`` follows the working directory, not the import.

    ``default=Path.cwd()`` bound the directory current when the module
    was imported, so a process that moved afterwards wrote live malware
    somewhere the analyst was no longer standing — and the warning that
    follows the download named that other directory as where it now is.
    Passing the callable makes click resolve it per invocation.
    """

    def test_no_command_freezes_its_output_directory_at_import(self):
        """The property, asserted over the real command tree.

        The first version of this test built a probe command out of the
        shared decorator, so it could only see commands that used it —
        and `ticloud download`, which carries its own copy of the option,
        kept writing live malware to the import-time directory while this
        test stayed green. Walk what the CLI actually exposes instead.
        """
        import click

        from rl_cli.cli.main import cli

        frozen = []

        def walk(group, path=""):
            for name, command in getattr(group, "commands", {}).items():
                if isinstance(command, click.Group):
                    walk(command, f"{path}{name} ")
                    continue
                for parameter in command.params:
                    if parameter.name == "output_dir" and not callable(parameter.default):
                        frozen.append(f"{path}{name}")

        walk(cli)
        assert not frozen, f"these resolve --output-dir at import time: {frozen}"

    def test_the_default_is_resolved_per_invocation(self, tmp_path, monkeypatch):
        import click
        from click.testing import CliRunner

        from rl_cli.cli.commands._shared_inputs import output_dir_option

        seen: list[Path] = []

        @click.command()
        @output_dir_option
        def probe(output_dir: Path) -> None:
            seen.append(output_dir)

        monkeypatch.chdir(tmp_path)
        CliRunner().invoke(probe, [])

        assert seen == [Path.cwd()]
        assert seen[0] == tmp_path

    def test_an_explicit_directory_still_wins(self, tmp_path):
        import click
        from click.testing import CliRunner

        from rl_cli.cli.commands._shared_inputs import output_dir_option

        seen: list[Path] = []

        @click.command()
        @output_dir_option
        def probe(output_dir: Path) -> None:
            seen.append(output_dir)

        CliRunner().invoke(probe, ["--output-dir", str(tmp_path / "elsewhere")])
        assert seen == [tmp_path / "elsewhere"]


class TestTheWrittenPathIsTheHashTheApplianceWasAsked:
    """The guard judges a stripped, lowercased hash; the file name did not.

    ``download`` built ``{hash_value}.malware`` and the extraction
    directory out of the raw argument, so a digest pasted with padding or
    in upper case landed under a name that did not match the sample the
    appliance was asked for — and the malware warning named that path.
    """

    def test_the_sample_is_written_under_the_normalized_hash(self, ctx_obj, monkeypatch, tmp_path):
        written: list[Path] = []

        def download(self: A1000SampleService, h: str, path: Path) -> bool:
            written.append(path)
            path.write_bytes(b"MZ")
            return True

        monkeypatch.setattr(A1000SampleService, "download_sample", download)

        result = invoke(
            a1000,
            ["download", f"  {SHA256.upper()}  ", "--output-dir", str(tmp_path)],
            ctx_obj,
        )

        assert result.exit_code == 0, result.output
        assert written == [tmp_path / f"{SHA256}.malware"]
        # Rich wraps the path, so the digest is matched by a fragment: the
        # warning must name the file that was written, not the argument.
        assert SHA256.upper()[:16] not in flat(result), "the warning named a path never written"
        assert SHA256[:16] in flat(result)

    def test_the_normalized_hash_is_what_the_appliance_is_asked_for(
        self, ctx_obj, monkeypatch, tmp_path
    ):
        asked: list[str] = []

        def list_extracted(self: A1000SampleService, h: str) -> list[dict[str, str]]:
            asked.append(h)
            return [{"sha1": SHA1}]

        monkeypatch.setattr(A1000SampleService, "list_extracted_files", list_extracted)
        monkeypatch.setattr(
            A1000SampleService, "download_extracted_files", lambda self, h, directory: True
        )

        result = invoke(
            a1000,
            [
                "extracted",
                f"{SHA256.upper()}\n",
                "--download-all",
                "--output-dir",
                str(tmp_path),
            ],
            ctx_obj,
        )

        assert result.exit_code == 0, result.output
        assert asked == [SHA256]
        assert (tmp_path / f"extracted_{SHA256[:16]}").is_dir()


class TestAFetchThatRaisedLeavesNoDirectoryBehindEither:
    """Cleanup has to happen on every way out, not just the tidy one.

    ``fetch_into`` removed the directories it had just created when the
    fetch *returned* falsy, and not when it raised. Both raising paths are
    reachable: refusing a zip bomb comes out of ``extract_private`` as a
    ``ValueError``, and Ctrl-C during a download raises too. Each left an
    empty ``extracted_<hash>/`` beside an empty ``--output-dir``, which
    reads as an extraction that happened and found nothing.
    """

    def _attempt(self, tmp_path, raising):
        from rl_cli.cli.commands._downloads import fetch_into
        from rl_cli.render.output import RichOutput

        destination = tmp_path / "quarantine" / f"extracted_{SHA256}"
        with pytest.raises(type(raising)):
            fetch_into(
                RichOutput(),
                destination,
                spinner="fetching",
                fetch=lambda: (_ for _ in ()).throw(raising),
                success="done",
                failure="failed",
                warnings=(),
            )
        return destination

    @pytest.mark.parametrize(
        "raising",
        [ValueError("Refusing archive: compressed size exceeds the bound"), KeyboardInterrupt()],
        ids=["refused-archive", "interrupted"],
    )
    def test_the_directories_it_created_are_removed_when_the_fetch_raises(self, tmp_path, raising):
        destination = self._attempt(tmp_path, raising)

        assert not destination.exists()
        assert not destination.parent.exists()

    def test_a_directory_that_took_a_file_is_left_alone(self, tmp_path):
        """``rmdir`` refuses a non-empty directory, so a partial write survives."""
        from rl_cli.cli.commands._downloads import fetch_into
        from rl_cli.render.output import RichOutput

        destination = tmp_path / "quarantine" / f"extracted_{SHA256}"

        def wrote_then_failed():
            (destination / "sample.bin").write_bytes(b"MZ")
            raise ValueError("Refusing archive: too many entries")

        with pytest.raises(ValueError):
            fetch_into(
                RichOutput(),
                destination,
                spinner="fetching",
                fetch=wrote_then_failed,
                success="done",
                failure="failed",
                warnings=(),
            )

        assert (destination / "sample.bin").read_bytes() == b"MZ"
