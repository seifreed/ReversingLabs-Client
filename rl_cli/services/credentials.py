"""Which credential strings are the user's, and which are the examples' stand-ins.

A leaf module on purpose: everything that picks an authentication method
has to be able to ask, and the services that do are the ones
:mod:`rl_cli.services.availability` imports, so the rule cannot live
there without a cycle.
"""

from __future__ import annotations

# Every credential stand-in config.example.yaml and .env.example ship,
# matched in full rather than by their shared "your_" prefix: the prefix is
# a fair rule for a token or a username, which have shapes, but not for a
# password, where `your_own_passphrase` is a credential the user chose.
#
# Edit this in step with the examples;
# `TestTheExamplesStillShipAStandInForEveryCredential` fails when a `your_*`
# value in either file is not refused here.
PLACEHOLDERS = frozenset(
    {
        "your_ticloud_username",
        "your_ticloud_password",
        "your_a1000_api_token_here",
        "your_username",
        "your_password",
        "your_staging_token",
    }
)


def is_real_credential(value: str | None) -> bool:
    """Whether a credential is one the user actually set.

    Empty, whitespace, and a value still carrying one of the examples'
    stand-ins all mean the same thing: nothing to authenticate with. Gate
    every credential of both services through this — the probe that reports
    whether a service is configured and the session that picks what to
    authenticate with — so that the two answers cannot disagree.

    Only the exact shipped strings are refused, and only after stripping
    the whitespace a stand-in can reach ``Settings`` wearing. A passphrase
    that merely begins ``your_`` is the user's.
    """
    stripped = (value or "").strip()
    return bool(stripped) and stripped.casefold() not in PLACEHOLDERS


def supplied_credential(value: str | None) -> str | None:
    """``value`` if the user set it, and ``None`` if it is a stand-in.

    Build every authentication from this, so that no service sends a
    credential :func:`is_real_credential` has just refused to count.
    ``None`` rather than an error, because a stand-in means the line may as
    well have been left out.
    """
    return value if is_real_credential(value) else None
