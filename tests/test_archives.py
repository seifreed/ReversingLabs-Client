"""Tests for ``rl_cli.storage.archives.unpack_private``.

The unpacking a downloaded archive needs is filesystem mechanics — a
staging directory, a private write, an extraction, a promotion — and it is
stated here, against the storage function, rather than only through the
A1000 service that happens to call it. The service's own tests drive the
same path end to end; these pin the promises the function makes to any
caller: nothing lands in the output directory until every member is out,
the staging directory goes whatever way the call exits, and the archive
itself is never one of the files left behind.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from rl_cli.storage.archives import unpack_private


@pytest.fixture
def out(tmp_path: Path) -> Path:
    """An empty download directory, so "nothing was left" can be asserted.

    Its own directory rather than ``tmp_path``, which the suite's home
    fixture already has a directory of its own in.
    """
    directory = tmp_path / "out"
    directory.mkdir()
    return directory


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    """A stored (uncompressed) archive of ``members``."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _staging(output_dir: Path) -> list[Path]:
    """Whatever the call left of its own working directory."""
    return [entry for entry in output_dir.iterdir() if entry.name.startswith(".rl-extract-")]


def _silent(message: str) -> None:
    """A ``warn`` for the cases that must not produce one."""
    raise AssertionError(f"unexpected warning: {message}")


class TestAnArchiveIsUnpackedWhole:
    def test_every_member_lands_under_the_output_directory(self, out):
        unpack_private(_zip_bytes({"a.bin": b"MZ", "sub/b.bin": b"ZM"}), out, _silent)

        assert (out / "a.bin").read_bytes() == b"MZ"
        assert (out / "sub" / "b.bin").read_bytes() == b"ZM"

    @pytest.mark.posix_only
    def test_the_unpacked_files_are_owner_only(self, out):
        unpack_private(_zip_bytes({"sub/a.bin": b"MZ"}), out, _silent)

        assert (out / "sub" / "a.bin").stat().st_mode & 0o777 == 0o600
        assert (out / "sub").stat().st_mode & 0o777 == 0o700

    def test_an_explicit_directory_member_is_made_not_written_to(self, out):
        """A zip may carry a bare ``dir/`` entry, and a member may nest deeply."""
        unpack_private(_zip_bytes({"emptydir/": b"", "deep/nested/file.bin": b"MZ"}), out, _silent)

        assert (out / "emptydir").is_dir()
        assert (out / "deep" / "nested" / "file.bin").read_bytes() == b"MZ"

    def test_a_member_naming_the_output_root_itself_is_handled(self, out):
        """A "./" entry resolves to the output root, whose parents exclude it."""
        unpack_private(_zip_bytes({"./": b"", "a.bin": b"MZ"}), out, _silent)

        assert (out / "a.bin").read_bytes() == b"MZ"

    def test_the_archive_itself_is_not_left_beside_what_came_out_of_it(self, out):
        unpack_private(_zip_bytes({"a.bin": b"MZ"}), out, _silent)

        assert sorted(entry.name for entry in out.iterdir()) == ["a.bin"]

    def test_the_staging_directory_does_not_survive_a_success(self, out):
        unpack_private(_zip_bytes({"a.bin": b"MZ"}), out, _silent)

        assert _staging(out) == []

    def test_an_existing_file_is_announced_before_it_is_replaced(self, out):
        (out / "a.bin").write_bytes(b"the analyst's copy")
        warned: list[str] = []

        unpack_private(_zip_bytes({"a.bin": b"MZ"}), out, warned.append)

        assert warned == [f"{out / 'a.bin'} already exists and will be replaced"]
        assert (out / "a.bin").read_bytes() == b"MZ"

    def test_a_run_does_not_write_the_archive_under_a_name_another_run_shares(self, out):
        """Two runs shared ``extracted_files.zip`` and truncated each other."""
        other = out / "extracted_files.zip"
        other.write_bytes(b"the other run's archive")

        unpack_private(_zip_bytes({"a.bin": b"MZ"}), out, _silent)

        assert other.read_bytes() == b"the other run's archive"


class TestARefusedArchiveLeavesNothing:
    """A member refused half-way must not leave live malware unpacked."""

    def test_a_traversal_member_leaves_the_output_directory_empty(self, out):
        with pytest.raises(ValueError, match="Refusing to extract outside"):
            unpack_private(_zip_bytes({"ok.bin": b"MZ", "../escaped.bin": b"MZ"}), out, _silent)

        assert list(out.iterdir()) == []

    def test_a_body_that_is_not_an_archive_leaves_the_output_directory_empty(self, out):
        with pytest.raises(zipfile.BadZipFile):
            unpack_private(b"not a zip at all", out, _silent)

        assert list(out.iterdir()) == []

    def test_a_bomb_leaves_no_staging_directory(self, out):
        members = {f"{index}.bin": b"" for index in range(10_001)}

        with pytest.raises(ValueError, match="member limit"):
            unpack_private(_zip_bytes(members), out, _silent)

        assert _staging(out) == []
