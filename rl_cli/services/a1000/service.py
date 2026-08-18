"""Connection and input validation shared by every A1000 service."""

from __future__ import annotations

from typing import Any

from rl_cli.models.redaction import redact_proxy
from rl_cli.models.validators import HashType, normalize_hash, validate_hash
from rl_cli.services.a1000.session import A1000Session
from rl_cli.services.base import BaseService
from rl_cli.services.decorators import with_client
from rl_cli.services.http import succeeded
from rl_cli.services.protocols import Notifier

# Every hash type the appliance recognises. Individual endpoints take
# less — the tag endpoints, the container lookup and the PDF report refuse
# SHA512 — and pass their own tuple to :meth:`A1000Service._valid_hash`.
A1000_HASH_TYPES = (HashType.MD5, HashType.SHA1, HashType.SHA256, HashType.SHA512)
A1000_SHORT_HASH_TYPES = (HashType.MD5, HashType.SHA1, HashType.SHA256)

# The all-zero SHA1 the SDK's own ``test_connection`` asks Check Status
# about: a well-formed hash no appliance holds, so the cheapest answer the
# endpoint can give is the one a liveness probe wants.
_LIVENESS_SHA1 = "0" * 40


def list_from_envelope(body: Any, key: str, *, required: bool = False) -> list[Any] | None:
    """The list under ``key`` in a response envelope, or ``None`` if unreadable.

    A key that is present but is not a list is an answer we could not
    read, and is reported as the failed lookup it is rather than as a
    lookup that found nothing — as is a body that is not an envelope at
    all. ``models.payload.search_page`` draws the same line for Advanced
    Search.

    A key that is simply absent is an empty page, which is what the SDK's
    own aggregators take it for while walking these very listings.
    Refusing it would make this CLI stricter than the SDK it wraps and turn
    ``{"count": 0, "next": null}`` into a failure.

    ``required`` says the key has to be there, for an endpoint whose
    envelope key this repo cannot corroborate: a missing key is then far
    more likely to mean we are reading the wrong key than that the
    endpoint found nothing. ``TitaniumCloudService._rl_list`` carries the
    same flag with the same default.

    ``{}`` carries no envelope to misread and no records to lose, so it is
    the empty answer it looks like even where the key is required.
    """
    if not isinstance(body, dict):
        return None
    if not body:
        return []
    if key not in body:
        # Absent, but the body carries records under a name we did not
        # expect: that is a spelling we cannot read, not a page with
        # nothing on it. Leniency is only for `{"count": 0, "next": null}`,
        # where nothing was found and nothing was hidden.
        if any(isinstance(value, list) and value for value in body.values()):
            return None
        return None if required else []
    entries = body[key]
    return entries if isinstance(entries, list) else None


class A1000Service(BaseService):
    """Reaches the appliance over an :class:`A1000Session`, and guards its input.

    Every focused A1000 service extends this: they differ in which part of
    the appliance they talk to, not in how they reach it. It is also the
    service the connection-level commands ask for, the connection being all
    they are about.

    The session is the connection, and is a collaborator rather than a base
    class so that several services can be handed the same one — build them
    with :meth:`A1000Session.service` and they authenticate once between
    them. Settings come from the session rather than alongside it: a
    service handed one appliance's settings and another's connection would
    report the first and talk to the second.
    """

    def __init__(self, session: A1000Session, output: Notifier | None = None):
        super().__init__(session.settings, output)
        self.session = session
        self.a1000_settings = session.a1000_settings

    @property
    def client(self) -> Any:
        """The SDK client, which belongs to the session, not to this service.

        Read-only on purpose: every service built over the session sees the
        same connection, so a setter here would let a caller swap the client
        out from under a service's siblings. Write ``session.client`` when
        that is what is meant.
        """
        return self.session.client

    def connect(self) -> bool:
        try:
            self.session.connect()
            return True
        except Exception as e:
            self.handle_error(e, "Failed to connect to A1000")
            return False

    def disconnect(self) -> None:
        """Drop the connection, for every service built over this session.

        Library API, and the counterpart to :meth:`connect`. No command
        calls it: a CLI invocation is one connection that ends with the
        process, so closing early would only cost the next service a second
        token POST. A caller embedding these services outside a CLI has a
        connection to manage and reaches for this or ``session.close()``.
        """
        self.session.close()

    def _valid_hash(
        self, hash_value: str, allowed: tuple[HashType, ...] = A1000_HASH_TYPES
    ) -> HashType | None:
        """The type of ``hash_value``, or ``None`` after saying why.

        ``allowed`` is the endpoint's own rule, because not every A1000
        endpoint takes every hash the appliance recognises: the tag
        endpoints, the container lookup and the PDF report take no SHA512.
        An endpoint stricter than the appliance passes its own tuple, so
        the refusal is named here rather than inside the SDK.
        """
        hash_type = validate_hash(hash_value)
        if not hash_type:
            self.output.error(f"Invalid hash format: {hash_value}")
            return None
        if hash_type not in allowed:
            accepted = ", ".join(allowed_type.value.upper() for allowed_type in allowed)
            self.output.error(
                f"This endpoint does not accept {hash_type.value.upper()}: "
                f"{hash_value} (accepted: {accepted})"
            )
            return None
        return hash_type

    def _supported_hash(
        self, hash_value: str, allowed: tuple[HashType, ...] = A1000_HASH_TYPES
    ) -> str | None:
        """The hash as the endpoint wants it, or ``None`` after saying why.

        The value to send, not the type: the guard judges the stripped,
        lowercased hash, so forwarding the raw argument would send the
        appliance something no guard had looked at.
        """
        if self._valid_hash(hash_value, allowed) is None:
            return None
        return normalize_hash(hash_value)

    # ---------- The connection itself ----------
    @with_client(default=False)
    def test_connection(self) -> bool:
        """Probe the connection. Returns ``False`` on any failure.

        The status decides, not the SDK's silence. The Check Status call is
        the one the SDK's ``test_connection`` makes, but the SDK raises only
        for the statuses in its ``RESPONSE_CODE_ERROR_MAP``, which carries
        no 3xx and neither 407, 410, 422, 428, 501, 504 nor 511 — the very
        answers a captive portal, an authenticating proxy or a dead gateway
        gives. Returning ``True`` because nothing was raised would therefore
        call an unreachable appliance healthy, and the availability probe
        pins that for 24 h, so the status is read through
        :func:`succeeded` as everywhere else here.

        The failure must reach this service's notifier and not be
        swallowed: the availability probe is the only place the user hears
        about it, and under the probe the notifier keeps the reason rather
        than drawing it.
        """
        return succeeded(
            self.client.file_analysis_status(
                sample_hashes=[_LIVENESS_SHA1], sample_status="processed"
            )
        )

    def connection_summary(self) -> dict[str, Any]:
        """Describe how this client is configured to reach the appliance.

        Local knowledge only — our own settings, not anything the appliance
        was asked. Not the SDK's ``configuration_dump()``, which formats the
        same local values but needs a constructed ``A1000`` and so costs a
        token POST under username/password auth for a read that touches
        nothing.

        Credentials must stay absent, and that includes the proxy URL:
        ``http://user:pass@proxy:8080`` is how a proxy credential is
        written, and this goes to stdout.
        """
        proxy = self.a1000_settings.proxy
        return {
            "host": self.a1000_settings.host,
            "auth": "Token" if self.a1000_settings.token else "Username/Password",
            "verify_ssl": self.a1000_settings.verify_ssl,
            "timeout": self.a1000_settings.timeout,
            "proxy": redact_proxy(proxy) if proxy else proxy,
        }
