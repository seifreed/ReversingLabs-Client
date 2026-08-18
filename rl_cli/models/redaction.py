"""Mask the credentials in a value that is about to be printed.

This is display safety, not configuration parsing: it exists because
``config show`` writes a profile to **stdout** and ``a1000 config-dump``
prints a connection summary, both out of a file deliberately kept at 0600.
It lives apart from :mod:`rl_cli.config.settings` so that a service hiding
a password in a proxy URL does not import the config module, and through it
pydantic-settings and the profile-file machinery.
"""

from __future__ import annotations

from typing import Any

# The fields of a profile that are credentials rather than configuration.
# Hosts and usernames are deliberately not here: they are what a user
# reads `config show` to check, and neither authenticates anything.
_SECRET_FIELDS = ("password", "token")

# How much of a secret `config show` leaves readable, so a user can tell
# which of two tokens is configured — and the length below which even
# that is too much, because four characters of a six-character secret is
# most of it.
_SECRET_TAIL = 4
_SHORTEST_SECRET_WITH_A_VISIBLE_TAIL = 8


def redact_secret(secret: str) -> str:
    """Enough of a secret to recognise it, not enough to use it."""
    if len(secret) < _SHORTEST_SECRET_WITH_A_VISIBLE_TAIL:
        return "***"
    return f"***{secret[-_SECRET_TAIL:]}"


def redact_proxy(url: str) -> str:
    """A proxy URL with its userinfo hidden and its host left readable.

    ``http://user:pass@proxy:8080`` is how a proxy credential is
    conventionally written, so ``proxy`` is a secret field in every way
    that matters even though it is not named like one. The host and the
    user are what someone reads `config show` or `a1000 config-dump` to
    check; the password is what those two commands were copying out of a
    0600 config file onto stdout.
    """
    scheme, separator, rest = url.partition("://")
    # A proxy written without a scheme -- ``user:pass@proxy:8080``, which the
    # ``proxy`` field stores as-is -- puts the whole value in ``scheme`` and
    # leaves ``rest`` empty, so the userinfo below was never found and the
    # password printed in the clear. The schemeless value is the ``rest``.
    if not separator:
        scheme, rest = "", url
    userinfo, at, host = rest.rpartition("@")
    if not at:
        return url
    user, colon, _ = userinfo.partition(":")
    return f"{scheme}{separator}{user}{':***' if colon else ''}@{host}"


def redact_section(values: dict[str, Any]) -> dict[str, Any]:
    """One profile section with every credential in it masked.

    Two kinds of secret live in a section: the ones named like one, and
    the proxy URL, which carries a password in its userinfo without being
    a secret field. Both have to be handled wherever a section is shown.
    """
    redacted = dict(values)
    for field in _SECRET_FIELDS:
        if field in redacted:
            redacted[field] = redact_secret(redacted[field])
    if redacted.get("proxy"):
        redacted["proxy"] = redact_proxy(redacted["proxy"])
    return redacted
