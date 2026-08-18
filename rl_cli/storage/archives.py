"""Unpack an archive that arrived from somewhere untrusted.

An extracted-files archive comes off an appliance, and an appliance can
be compromised or spoofed. Everything here therefore treats the archive
as hostile input: the member count, declared total and per-member
compression ratio are all judged from the central directory *before* a
byte is decompressed, the running total is re-checked while writing
because those declarations may lie, and no member is allowed to land
outside the directory it was aimed at.

This is filesystem infrastructure with no knowledge of any particular
service, so refusing a zip bomb does not live in the module that knows how
to upload a sample -- and neither does the staging directory the whole
unpack runs in: :func:`unpack_private` is the one entry point a caller
holding a downloaded archive needs, and it takes bytes and a directory.

Writing is left to :mod:`rl_cli.storage.files`, which owns the separate
promise that a file is owner-only and never half-written.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import IO

from rl_cli.storage.files import private_writer, write_private_bytes

# Bounds on a downloaded extracted-files archive. The appliance is meant
# to send the files it carved out of one sample; a compromised or spoofed
# one sends a bomb, and 337 KB expanding to 300 MB fills the analyst's
# disk on a single `extracted --download-all`.
_MAX_ARCHIVE_MEMBERS = 10_000
_MAX_ARCHIVE_BYTES = 2 * 1024**3
# Deflate tops out near 1032:1 on a run of one byte, and the bombs seen
# here reached 932:1; real extracted files — PEs, scripts, resources —
# stay an order of magnitude below this.
_MAX_MEMBER_RATIO = 200
# Below this a wild ratio buys nothing: a 64 KiB member cannot fill a
# disk however well it compressed, and the total cap covers a crowd.
_RATIO_EXEMPT_BYTES = 64 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024


def refuse_bomb(members: list[zipfile.ZipInfo]) -> None:
    """Refuse an archive whose central directory already describes a bomb.

    ``file_size`` and ``compress_size`` are read from the central
    directory, so all three limits are decided before a byte is
    decompressed. They are also attacker-controlled and may lie, which is
    what the running total in :func:`copy_member` is for.
    """
    if len(members) > _MAX_ARCHIVE_MEMBERS:
        raise ValueError(
            f"Refusing archive: {len(members)} members is over the "
            f"{_MAX_ARCHIVE_MEMBERS}-member limit"
        )
    declared = sum(member.file_size for member in members)
    if declared > _MAX_ARCHIVE_BYTES:
        raise ValueError(
            f"Refusing archive: it declares {declared} uncompressed bytes, over the "
            f"{_MAX_ARCHIVE_BYTES}-byte limit"
        )
    for member in members:
        if member.file_size <= _RATIO_EXEMPT_BYTES or not member.compress_size:
            continue
        ratio = member.file_size / member.compress_size
        if ratio > _MAX_MEMBER_RATIO:
            raise ValueError(
                f"Refusing archive: {member.filename} expands {ratio:.0f}:1, over the "
                f"{_MAX_MEMBER_RATIO}:1 compression-ratio limit"
            )


def copy_member(source: IO[bytes], destination: IO[bytes], member: zipfile.ZipInfo) -> int:
    """Copy one member in bounded chunks, refusing to exceed what it declared.

    Chunked because a whole-member ``read()`` costs a 400 MB member 400 MB
    of RSS, which is the amplifier that makes a bomb cheap. Returns the
    bytes written so the caller can keep a running total across members.
    """
    written = 0
    while chunk := source.read(_COPY_CHUNK_BYTES):
        written += len(chunk)
        if written > member.file_size:
            raise ValueError(
                f"Refusing archive: {member.filename} decompressed past the "
                f"{member.file_size} bytes it declared"
            )
        destination.write(chunk)
    return written


def extract_private(archive: zipfile.ZipFile, output_dir: Path) -> None:
    """Unpack ``archive`` so every extracted file is owner-only.

    ``extractall`` is not usable here: it recreates members under the
    default umask, which lands the live malware inside world-readable even
    though the archive itself was written 0600. Members go one at a time
    through the same private-write path, and every destination is resolved
    before anything is written.
    """
    refuse_bomb(archive.infolist())
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_plan(_planned_members(archive, output_dir), archive)


def _planned_members(
    archive: zipfile.ZipFile, output_dir: Path
) -> list[tuple[zipfile.ZipInfo, Path]]:
    """Where every member would land, or nothing at all.

    Separated from the writing so the invariant is structural rather than a
    comment: this either returns a whole plan or raises, and nothing has
    been written when it does. Refusing a member halfway through would leave
    the ones before it — live malware — unpacked on disk.
    """
    root = output_dir.resolve()
    members = []
    files: set[Path] = set()
    directories: set[Path] = set()
    for member in archive.infolist():
        # Zip entries are attacker-controlled; a "../" member must not be
        # able to write outside the requested directory.
        resolved = (output_dir / member.filename).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Refusing to extract outside {output_dir}: {member.filename}")
        members.append((member, resolved))
        (directories if member.is_dir() else files).add(resolved)
        for parent in resolved.parents:
            if parent == root:
                break
            directories.add(parent)

    # "payload" as a file plus "payload/stage2.dll" raises FileExistsError
    # from the mkdir mid-write, with the members before it already on disk.
    clash = files & directories
    if clash:
        name = sorted(clash)[0].relative_to(root)
        raise ValueError(f"Refusing archive: {name} is both a file and a directory member")
    return members


def _write_plan(members: list[tuple[zipfile.ZipInfo, Path]], archive: zipfile.ZipFile) -> None:
    """Write an already-validated plan, still watching the running total.

    The declared sizes :func:`refuse_bomb` judged are attacker-controlled
    and may lie, so the real total is re-checked as it accumulates.
    """
    written = 0
    for member, resolved in members:
        if member.is_dir():
            resolved.mkdir(parents=True, exist_ok=True)
            continue
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, private_writer(resolved) as handle:
            written += copy_member(source, handle, member)
        if written > _MAX_ARCHIVE_BYTES:
            raise ValueError(
                f"Refusing archive: it decompressed past the {_MAX_ARCHIVE_BYTES}-byte limit"
            )


def private_mkdir(path: Path, root: Path) -> None:
    """Create ``path`` and every parent below ``root`` owner-only.

    ``Path.mkdir(parents=True)`` hands its mode to the leaf only, so a
    member at ``a/b/payload`` leaves ``a`` world-readable. The files are
    0600 either way, so what this covers is the carved-out path names.
    """
    if path == root or not path.is_relative_to(root):
        return
    private_mkdir(path.parent, root)
    path.mkdir(mode=0o700, exist_ok=True)


def promote(staging: Path, output_dir: Path, warn: Callable[[str], None]) -> None:
    """Move a fully unpacked tree into ``output_dir``, one rename at a time.

    ``staging`` is a directory inside ``output_dir``, so each member
    arrives by rename on one filesystem: no member is ever half-written.
    The set is not atomic, though, and making it so would need a second
    staging copy of whatever it displaces — more machinery than an
    extraction that has already survived every check is worth. A failure
    part-way (a destination that is a non-empty directory, EPERM) leaves
    the members already moved in place, and the caller's ``rmtree`` takes
    the rest.
    """
    for source in sorted(staging.rglob("*")):
        destination = output_dir / source.relative_to(staging)
        if source.is_dir():
            private_mkdir(destination, output_dir)
        else:
            private_mkdir(destination.parent, output_dir)
            if destination.exists():
                # ``download`` and the report writer both say so before
                # replacing a file the analyst may be working from.
                warn(f"{destination} already exists and will be replaced")
            source.replace(destination)


def unpack_private(archive_bytes: bytes, output_dir: Path, warn: Callable[[str], None]) -> None:
    """Unpack a downloaded archive so ``output_dir`` only ever sees whole files.

    The bytes are written and opened under a private per-run staging
    directory and moved into place only once every member is out, so a
    member refused half-way leaves no live malware in the output directory
    and two concurrent runs cannot truncate one shared archive under each
    other. The staging directory goes whatever way this exits.

    ``warn`` says a file is about to be replaced, and is the caller's:
    which channel that reaches, and whether it is worth saying at all, is
    not something a filesystem module knows.
    """
    staging = Path(tempfile.mkdtemp(dir=output_dir, prefix=".rl-extract-"))
    try:
        archive_path = staging / "extracted_files.zip"
        write_private_bytes(archive_path, archive_bytes)
        unpacked = staging / "files"
        with zipfile.ZipFile(archive_path, "r") as archive:
            extract_private(archive, unpacked)
        archive_path.unlink()
        promote(unpacked, output_dir, warn)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
