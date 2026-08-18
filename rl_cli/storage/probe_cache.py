"""A probe's answer kept on disk: good for a while, tied to one configuration."""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from rl_cli.storage.files import write_private_text


class ProbeCache:
    """The on-disk memory of a probe run.

    Keeping a JSON file is a different job from talking to an appliance, so
    this owns where the file lives, how long an answer is good for, and
    which configuration the answer belongs to; what a probe measures, and
    which settings identify it, stay with the prober.

    ``identity`` is a callable rather than the values themselves, because
    the settings it is derived from can still be edited after this object is
    built, and a cached answer has to stop matching the moment the host or
    the credential it was measured against changes.

    ``report`` is where trouble with the file goes, or ``None`` for silence.
    A plain callable rather than the services' ``Notifier``, so a filesystem
    helper does not depend on the service layer to say a word — and the
    caller decides whether this run wants to hear it, since nothing here is
    fatal: a cache that cannot be read or written costs a re-probe, not an
    answer.
    """

    def __init__(
        self,
        path: Path,
        identity: Callable[[], Sequence[object]],
        *,
        duration: timedelta,
        report: Callable[[str], None] | None = None,
    ) -> None:
        self.path = path
        self.salt_path = path.parent / "fingerprint_salt"
        self.duration = duration
        self._identity = identity
        self._report_to = report

    def _salt(self) -> str:
        """A random per-install value mixed into the fingerprint.

        Unsalted, the digest is sha256 over the configuration, and every
        part of that except the secrets is low-entropy and largely
        discoverable — with the recipe published right here. That makes the
        cache file an offline oracle: guess a token, hash, compare. The salt
        is generated once, stored 0600 beside the cache, and never leaves
        this machine, so the digest confirms nothing to anyone who does not
        already have it.

        A salt that cannot be stored yields a fresh value each call, which
        costs a re-probe rather than a wrong answer.

        ``ValueError`` as well as ``OSError``: a salt file that is not UTF-8
        makes ``read_text`` raise ``UnicodeDecodeError``, and left to escape
        it is swallowed by the broad handlers in ``load``/``save`` — the
        replacement never written, the cache never loaded or saved, and both
        APIs re-probed on every invocation with nothing said unless
        ``--verbose``. Unlinked rather than overwritten so a file we cannot
        read is gone even if writing the replacement then fails.
        """
        try:
            return self.salt_path.read_text(encoding="utf-8")
        except (OSError, ValueError):
            pass
        salt = secrets.token_hex(16)
        try:
            self._make_dir()
            self.salt_path.unlink(missing_ok=True)
            write_private_text(self.salt_path, salt)
        except OSError as exc:
            self._report(f"Failed to store availability cache salt: {exc}")
        return salt

    def _make_dir(self) -> None:
        """Owner-only: the cache and its salt both live in here."""
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _report(self, message: str) -> None:
        """Say what went wrong with the file, if anyone asked to be told."""
        if self._report_to is not None:
            self._report_to(message)

    def fingerprint(self) -> str:
        """Identify the configuration a cached answer belongs to.

        The whole run lives in a single file, so without this the result
        computed for one configuration is served to another — including
        one pointing at a different appliance, for the full duration.
        Whatever the caller counts as its identity is hashed with the
        salt, so a secret can be part of it without being stored.

        The identity is JSON-encoded rather than joined on a delimiter: a
        password may hold any character, ``|`` included, so ``"|".join``
        let ``["alice", "x|y"]`` and ``["alice|x", "y"]`` hash alike and
        one configuration collect the availability answer measured for
        another. JSON quotes every part, and keeps ``None`` distinct from
        ``""`` where the old ``str(part or "")`` folded the two together.
        """
        identity = json.dumps([self._salt(), *self._identity()])
        return hashlib.sha256(identity.encode()).hexdigest()[:16]

    def load(self) -> dict[str, Any] | None:
        """The stored answer, if there is one this configuration may have.

        An answer is good for ``duration`` counted forwards from the moment
        it was measured, so an age outside ``0 <= age < duration`` is stale
        either way. The upper bound alone lets a timestamp in the future —
        written while the clock was ahead of where it is now, after a
        suspended VM resumes, a backwards NTP correction or a timezone
        fixed after the fact — satisfy the comparison with a negative age
        and pin one answer for as long as the file survives, with only
        ``check-access --force`` to shift it. Re-probing on a clock that
        makes no sense costs one probe and rewrites the timestamp; trusting
        it costs every later warning about an appliance that is down.
        """
        if not self.path.exists():
            return None
        try:
            with self.path.open(encoding="utf-8") as f:
                data = json.load(f)
            if data.get("fingerprint") != self.fingerprint():
                return None
            age = datetime.now() - datetime.fromisoformat(data["timestamp"])
            if timedelta() <= age < self.duration:
                # Whatever a previous version wrote is passed through as it
                # was written: this is the boundary where an unknown key is
                # tolerated, and dropping one throws away an answer the user
                # is still entitled to. The fingerprint is this file's own
                # bookkeeping, not part of what a probe answers; returning it
                # would make a cache hit a different shape from a fresh run.
                return {key: value for key, value in data.items() if key != "fingerprint"}
        except Exception as exc:
            self._report(f"Failed to load availability cache: {exc}")
        return None

    def save(self, data: Mapping[str, Any]) -> None:
        """Store ``data`` for this configuration, or say why it could not be."""
        try:
            self._make_dir()
            # 0600: the fingerprint is derived from the credentials, and
            # the answer names the appliances this machine talks to.
            write_private_text(
                self.path, json.dumps({**data, "fingerprint": self.fingerprint()}, indent=2)
            )
        except Exception as exc:
            self._report(f"Failed to save availability cache: {exc}")

    def clear(self) -> None:
        """Forget the stored answer; the salt outlives it."""
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError as exc:
            self._report(f"Failed to clear availability cache: {exc}")
