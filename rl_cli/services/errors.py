"""Turning a failure from the vendor's SDK into advice the analyst can act on.

The knowledge of *which* exceptions mean what — the SDK's error taxonomy,
the transport's — and of what the user can do about each one lives here so
that it does not live on ``BaseService``, which every service inherits and
which is meant to be nothing but settings and a notifier. This module is
what imports ``requests`` and ``ReversingLabs.SDK.helper``, and what
formats config-file paths and user-facing English.
"""

from __future__ import annotations

import requests
from ReversingLabs.SDK.helper import ForbiddenError, TooManyRequestsError, UnauthorizedError

from rl_cli.config import Settings


def error_advice(error: Exception, settings: Settings) -> str:
    """What the user can do about ``error``; empty when we do not know.

    A pure function of the failure and the profile it happened under, so
    it is testable without a service and reusable by anything that has
    both. The failures an analyst actually hits are named here; anything
    else is reported as it arrived rather than guessed at.
    """
    config_file = settings.config_file or settings.config_dir / "config.yaml"
    where = f"profile '{settings.profile}' in {config_file}"
    if isinstance(error, UnauthorizedError):
        return f"Credentials were rejected. Check {where}"
    if isinstance(error, ForbiddenError):
        return f"Permission denied. The account in {where} may not make this call"
    if isinstance(error, TooManyRequestsError):
        return "Rate limited. Wait a few minutes and retry"
    if isinstance(error, requests.exceptions.SSLError):
        return (
            "TLS verification failed. Trust the appliance's CA, or set "
            f"verify_ssl: false for {where}"
        )
    return ""
