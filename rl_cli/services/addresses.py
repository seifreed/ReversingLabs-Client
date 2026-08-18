"""The address guards the A1000 and TitaniumCloud network lookups share.

Each guard answers the value to send rather than a yes/no: the checks
judge a stripped — and mostly lowercased — subject, so a caller handed a
bool would forward its own raw argument and ``" 8.8.8.8 "``, ``"EVIL.COM"``
and ``"evil.com."`` would reach the API in a form no guard had looked at.

``on_error`` is how the caller says why, the way
:mod:`rl_cli.services.hashes` takes it: the two services report through
different notifiers, so a guard that reached for one of them would be a
guard only one of them could use.
"""

from __future__ import annotations

from collections.abc import Callable

from rl_cli.models.validators import (
    normalize_domain,
    normalize_ip_address,
    validate_url,
    validate_url_or_host,
)


def valid_ip(ip: str, on_error: Callable[[str], None]) -> str | None:
    """The address to send, or ``None`` after saying why.

    What comes back is the canonical spelling, not the typed one: the
    A1000 endpoints make the address a path segment, so an expanded or
    uppercased IPv6 address is a second key for a host the appliance files
    under one, and a scope id's ``%`` reshapes the path outright.
    """
    address = normalize_ip_address(ip)
    if address is None:
        on_error(f"Invalid IP address: {ip}")
    return address


def valid_domain(domain: str, on_error: Callable[[str], None]) -> str | None:
    """The bare domain to send, or ``None`` after saying why.

    A malformed name is answered wrongly by two different routes. The
    A1000's domain endpoint makes its argument an unquoted path segment,
    so a URL, a ``host:port``, an email address or a path there is a
    request about something else. TitaniumCloud's five domain calls POST
    the name in a JSON body instead, where it is simply a key the corpus
    holds nothing under, answered with an empty page and reported as "No
    files found for a..b.com".

    The root dot is the same defect in miniature on both: ``evil.com.``
    and ``evil.com`` name one zone to a resolver and two keys to either
    API, which holds records under only one of them.

    A non-ASCII name is not a typo, and is encoded rather than refused —
    see :func:`~rl_cli.models.validators.normalize_domain`.
    """
    normalised = normalize_domain(domain)
    if normalised is None:
        on_error(
            f"Invalid domain: {domain} (expected a bare domain such as example.com, "
            "not a URL, a host:port or a path)"
        )
    return normalised


def valid_url(url: str, on_error: Callable[[str], None]) -> str | None:
    """For endpoints that name a fetched URL, so a scheme is required."""
    if validate_url(url):
        return url.strip()
    on_error(f"Invalid URL format: {url}")
    return None


def valid_url_or_host(value: str, on_error: Callable[[str], None]) -> str | None:
    """For the lookups that take a bare hostname, a URL or an email alike.

    Stripped and no more: a URL path and the local part of an email
    address are both case-sensitive, so this is the one subject here that
    must reach the endpoint spelled as it was typed.
    """
    if validate_url_or_host(value):
        return value.strip()
    on_error(f"Invalid URL or hostname: {value}")
    return None
