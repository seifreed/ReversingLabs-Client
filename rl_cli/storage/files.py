"""Write files that hold secrets or live malware.

Three things have to hold everywhere this project writes: the file is
owner-only from the moment it exists, a symlink sitting at the
destination is never followed, and a write that fails leaves no partial
file behind wearing the name of a complete one.

All three hold for reports as well as samples: a report carries the same
analysis - extracted strings, C2 indicators, customer file names - and a
followed symlink aims a download at an arbitrary file the user can write.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import IO, Any, Literal, overload

# ``O_NOFOLLOW`` and ``fchmod`` are POSIX, absent on Windows. Where they
# are missing the write degrades to what that platform enforces instead --
# ``os.open`` still refuses to create over an existing name via ``O_EXCL``,
# and a file made under the user's profile is ACL-scoped to that user --
# while every POSIX target keeps the umask-proof 0600 and the no-symlink
# open below. Resolved once, so the per-write path holds values, not a
# platform check.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CAN_FCHMOD = hasattr(os, "fchmod")


# ``binary`` decides which of the two file protocols the caller gets, and
# saying so in the overloads is what lets a caller hand the handle to
# something that wants real bytes — the archive extractor does — with no
# suppression needed at the call site.
@overload
def private_writer(
    path: Path, *, binary: Literal[True] = ...
) -> AbstractContextManager[IO[bytes]]: ...


@overload
def private_writer(path: Path, *, binary: Literal[False]) -> AbstractContextManager[IO[str]]: ...


@contextmanager
def private_writer(path: Path, *, binary: bool = True) -> Iterator[IO[Any]]:
    """Open ``path`` 0600, refusing symlinks, replacing it only on success.

    Writes go to a temporary sibling which is renamed into place when the
    block completes; any exception - including KeyboardInterrupt -
    removes it. An interrupted download therefore leaves nothing, rather
    than a truncated sample indistinguishable from a whole one.

    That temporary is named per process and per call and created with
    ``O_EXCL``, so it is always a file this call has just made. A fixed
    ``<name>.part`` is both a security hole and a race: anyone able to
    create files in the download directory pre-creates it as a hard link to
    a file the invoking user owns - which ``O_NOFOLLOW`` does not see, it
    only refuses symlinks - and the write and ``fchmod`` go straight through
    to the victim; and two concurrent writers to one destination share it,
    the second renaming it away under the first.
    """
    partial = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.part")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW
    descriptor = os.open(partial, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb" if binary else "w") as handle:
            if _CAN_FCHMOD:
                os.fchmod(descriptor, 0o600)
            yield handle
        # Path.replace does not follow a symlink at the destination - it
        # replaces the link itself - so this guards nothing about where
        # the bytes land. It stays as policy: detaching the analyst's
        # link and leaving the file it named unwritten is not something
        # to do silently under "Report saved".
        if path.is_symlink():
            raise OSError(f"Refusing to write through a symlink: {path}")
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def write_private_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path``, owner-only and all-or-nothing."""
    with private_writer(path) as handle:
        handle.write(data)


def write_private_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path``, owner-only and all-or-nothing."""
    with private_writer(path, binary=False) as handle:
        handle.write(text)


def save_private_report(
    path: Path, content: Any, *, binary: bool, warn: Callable[[str], None]
) -> None:
    """Write a fetched report to ``path``, owner-only, announcing a replacement.

    Reports carry the whole analysis - extracted strings, C2 indicators,
    embedded resources, the customer's own file names - so they get the
    same 0600, no-symlink, all-or-nothing write as a downloaded sample
    rather than a bare ``write_bytes`` at the umask's mercy. Which of the
    two encodings applies is the caller's ``binary``: the pdf variant is
    already bytes, everything else is a structure this serialises.

    An existing file is announced before it is replaced, on the same terms
    as a downloaded sample: overwriting in silence prints "Report saved to
    keep.json" over the annotated copy the analyst was working from.
    ``warn`` is how it is announced — this module knows the policy, not the
    terminal.

    Raises ``OSError`` if the destination cannot be written. That is the
    destination's problem rather than a CLI bug, and it is the caller
    that knows how to say so; there is no fallback to stdout, because
    half a fallback is worse than a message naming the path.
    """
    if path.exists():
        warn(f"{path} already exists and will be replaced")
    if binary:
        write_private_bytes(path, content)
    else:
        write_private_text(path, json.dumps(content, indent=2))


def read_text_lenient(path: Path) -> str:
    """Read ``path`` as text, falling back to latin-1 for what is not UTF-8.

    Third-party YARA rule files are routinely cp1252 or latin-1, and
    reading one strictly ends the run in a raw ``UnicodeDecodeError``.
    latin-1 decodes any byte sequence, so the file reaches its destination
    as written rather than not at all.

    ``OSError`` is left to the caller: an unreadable path is a different
    fact from an oddly encoded one, and only the caller knows what it was
    trying to do with it.
    """
    try:
        return path.read_text()
    except UnicodeDecodeError:
        return path.read_bytes().decode("latin-1")
