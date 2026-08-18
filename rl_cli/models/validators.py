"""Input validation utilities.

The one fact the three ``normalize_*`` guards below are built on: **the SDK
interpolates these values into a request without quoting them.** A domain,
a ruleset name and an IP address each become a path segment —
``/api/network-threat-intel/domain/{domain}/``,
``/api/yara/publish/ruleset/{ruleset_name}/``,
``/api/network-threat-intel/ip/{ip}/report/`` — and the ruleset name also
becomes a query parameter built by string concatenation
(``ReversingLabs/SDK/a1000.py``). There is no ``quote()`` anywhere in that
chain, so the value decides the shape of the request: ``..`` retargets the
endpoint, ``#`` truncates it, ``&name=`` adds a second parameter, ``%``
introduces a percent-escape.

Each guard therefore returns the value to send rather than a bool, so that
a caller cannot forward its own argument past a check that judged
something else.
"""

import ipaddress
import re
from enum import Enum
from urllib.parse import urlparse

# The ``idna`` package, never the stdlib's ``str.encode("idna")``, which
# implements IDNA **2003** and folds characters IDNA 2008 keeps: it turns
# ``faß.de`` into ``fass.de`` and a final sigma into a medial one. Those
# are separate registrations, so the stdlib codec would report another
# domain's verdict as this one's. ``idna`` is a hard dependency of
# ``requests`` and so already present.
import idna

# One DNS label: letters, digits, underscores and inner hyphens, 1-63
# characters. Underscores are illegal in the strict hostname grammar and
# legal in DNS — ``_dmarc.example.com`` is ordinary, and ``foo_bar.evil.com``
# is ordinary C2 infrastructure — so refusing them would refuse names the
# appliance answers about.
#
# ASCII-only: it must be matched against the punycode spelling, not the
# typed one, or it refuses every IDN. :func:`normalize_domain` encodes
# before it gets here.
#
# ``\Z`` and not ``$``: this is matched per-label, so the ``strip()`` on the
# whole value never reaches a newline sitting at a label boundary, and
# ``$`` matches just before a trailing ``\n`` — so ``normalize_domain`` was
# returning ``good.com\n.evil.com`` with the newline still in it, the
# request-splitting primitive this whole module exists to keep out.
_DOMAIN_LABEL = re.compile(r"^[a-z0-9_](?:[a-z0-9_-]{0,61}[a-z0-9_])?\Z")

# A YARA ruleset name, as the appliance itself defines one. Not a guess:
# the Spectra Intelligence YARA hunting API (TCA-0303) states "between 3
# and 48 characters long and conforming to the following regular
# expression: ^[a-z,A-Z,0-9,_-]*$", and the Spectra Analyze (A1000) YARA
# Hunting page repeats the length and the character set — underscores for
# spaces, "any other special characters should be avoided".
#
# The commas inside the published character class are the documentation
# listing its ranges, not a comma being legal.
#
# ``\Z`` and not ``$`` for the reason :data:`_DOMAIN_LABEL` gives: ``$``
# admits a trailing newline. The ``strip()`` in ``normalize_ruleset_name``
# happens to close the hole today, but the correct anchor does not lean on
# a caller stripping first.
_RULESET_NAME = re.compile(r"^[A-Za-z0-9_-]{3,48}\Z")


class HashType(Enum):
    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    SHA512 = "sha512"


def validate_hash(hash_value: object) -> HashType | None:
    """Identify ``hash_value`` as MD5/SHA1/SHA256/SHA512 or return ``None``.

    Recognising a hash is not the same as an endpoint accepting it: the
    TitaniumCloud hash endpoints take only the first three.

    Takes ``object`` because this is a boundary check: a caller handing it
    the ``None`` a payload had where a hash should be is asking exactly the
    question this answers, so it must not raise.
    """
    if not isinstance(hash_value, str):
        return None
    hash_value = hash_value.lower().strip()
    if re.match(r"^[a-f0-9]{32}$", hash_value):
        return HashType.MD5
    if re.match(r"^[a-f0-9]{40}$", hash_value):
        return HashType.SHA1
    if re.match(r"^[a-f0-9]{64}$", hash_value):
        return HashType.SHA256
    # The A1000's report, classification, download and search endpoints all
    # take a SHA512, so refusing to recognise one here would be stricter
    # than the API. Which endpoint accepts which type is the service's to
    # say, not this function's.
    if re.match(r"^[a-f0-9]{128}$", hash_value):
        return HashType.SHA512
    return None


def normalize_hash(hash_value: object) -> str | None:
    """The hash as the APIs want it — lowercase, unpadded — or ``None``.

    :func:`validate_hash` judges the stripped, lowercased value, so this is
    what a caller must send rather than its own argument.
    """
    if not isinstance(hash_value, str):
        return None
    normalised = hash_value.strip().lower()
    return normalised if validate_hash(normalised) else None


def validate_url(url: str) -> bool:
    """Require a full URL with a scheme.

    Use for endpoints that fetch the URL — submitting ``example.com`` for
    crawling is rejected by the appliance with "Bad request created".
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_url_or_host(value: str) -> bool:
    """Accept a full URL or a bare hostname.

    The network intelligence lookups take either. They also answer 200 with
    an empty report for outright nonsense, so a typo is worth catching here
    even though the appliance will not complain about it.
    """
    value = value.strip()
    if not value or any(character.isspace() for character in value):
        return False
    if validate_url(value):
        return True
    # No scheme: parse as a network location and require a dotted host.
    # urlparse raises "Invalid IPv6 URL" on an unbalanced bracket, and a
    # bracket is a typo like any other — this check must name a typo rather
    # than crash on it.
    try:
        netloc = urlparse(f"//{value}").netloc
    except ValueError:
        return False
    return "." in netloc


def normalize_domain(value: object) -> str | None:
    """The bare domain to look up — lowercased, punycode — or ``None``.

    Stricter than :func:`validate_url_or_host` on purpose, for the reason
    this module states: a scheme, a port, userinfo or a path each reshape
    the request the domain is interpolated into. So: dot-separated
    letters-digits-hyphen labels and nothing else. A trailing root dot is
    dropped, and a final label of digits is an IPv4 address rather than a
    domain — ``ip-report`` is the lookup for that.

    A name with non-ASCII in it is *encoded* rather than refused, since both
    APIs key domains by their punycode spelling and an analyst pastes
    ``münchen.de`` out of a phishing lure. Encoding happens before the
    length and label checks so that they judge the string actually sent.
    """
    if not isinstance(value, str):
        return None
    domain = value.strip().rstrip(".").lower()
    if not domain:
        return None
    # Only a name that actually needs encoding goes through ``idna``: the
    # package implements IDNA 2008 strictly and refuses an underscore
    # label, while ``_dmarc.example.com`` and ``cdn_1.evil_c2.net`` are
    # names ``_DOMAIN_LABEL`` below deliberately accepts.
    if not domain.isascii():
        try:
            # ``uts46`` applies the case and width mapping; without it
            # ``idna`` refuses anything not already normalised.
            # ``transitional=False`` is what keeps ``faß.de`` as
            # ``xn--fa-hia.de`` rather than folding it to ``fass.de``.
            domain = idna.encode(domain, uts46=True, transitional=False).decode("ascii")
        except (idna.IDNAError, UnicodeError):
            return None
    if len(domain) > 253:
        return None
    labels = domain.split(".")
    if len(labels) < 2 or labels[-1].isdigit():
        return None
    if not all(_DOMAIN_LABEL.match(label) for label in labels):
        return None
    return domain


def normalize_ruleset_name(value: object) -> str | None:
    """The YARA ruleset name to send, or ``None``.

    The unquoted interpolation this module states bites a ruleset name in
    three ways: ``../../../api/samples/v2/list/details`` lands a
    cloud-retro POST on the sample-listing endpoint, ``prod#ignored``
    fetches the ruleset ``prod`` under a name the CLI keeps printing, and
    ``a&name=core`` puts two ``name=`` parameters in one query string.

    :data:`_RULESET_NAME` is the appliance's own published rule, which is
    stricter than any of those. Surrounding whitespace is stripped, as it
    is for a hash, and nothing else is changed: ruleset names are
    case-sensitive (``a1000.get_yara_ruleset_contents``).
    """
    if not isinstance(value, str):
        return None
    name = value.strip()
    return name if _RULESET_NAME.match(name) else None


def normalize_ip_address(ip: object) -> str | None:
    """The address as the APIs want it — canonical, unscoped — or ``None``.

    Canonical, because the address becomes an unquoted path segment, so
    ``2001:DB8:0000:0000:0000:0000:0000:0001`` and ``2001:db8::1`` are one
    host and two keys — the same split :func:`normalize_domain` closes.

    Unscoped, because ``%`` in ``fe80::1%eth0`` is the percent-escape
    introducer, and a scope id names an interface on the machine that typed
    it — meaningless to a remote appliance.
    """
    if not isinstance(ip, str):
        return None
    try:
        address = ipaddress.ip_address(ip.strip())
    except ValueError:
        return None
    # ``str`` keeps the scope id an IPv6Address parsed; ``partition``
    # drops it. An IPv4 address never has one.
    return str(address).partition("%")[0]
