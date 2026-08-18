"""One connection to an A1000, and the services that speak over it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ReversingLabs.SDK.a1000 import A1000

from rl_cli.config import Settings
from rl_cli.services.credentials import is_real_credential, supplied_credential
from rl_cli.services.http import apply_request_timeout, default_post_timeout
from rl_cli.services.protocols import Notifier

if TYPE_CHECKING:
    from rl_cli.services.a1000.service import A1000Service


class A1000Session:
    """The appliance connection, owned apart from the services that use it.

    A service is not a connection — it is a group of calls made over one.
    Holding the connection here is what lets several focused services share
    it, which matters because under username/password auth constructing the
    SDK client is itself a token POST to the appliance.

    Build the services from the session that is to carry them::

        session = A1000Session(settings)
        samples = session.service(A1000SampleService)
        reports = session.service(A1000ReportService)

    Both hand their calls to the one client. ``session.close()`` ends it
    for both, and so does either service's ``disconnect()`` — library API
    for a caller that outlives one connection, which a CLI invocation does
    not.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.a1000_settings = settings.a1000
        # The SDK ships no type stubs, so ``Any`` is the truthful annotation.
        self.client: Any = None
        # How "the appliance did not answer" is told apart from "nobody has
        # asked yet", which is both ``client`` being ``None``.
        self.failure: Exception | None = None

    def service[S: A1000Service](self, service_cls: type[S], output: Notifier | None = None) -> S:
        """Build ``service_cls`` over this connection, reporting through ``output``.

        The notifier is the caller's, not the connection's: a session draws
        nothing itself, so it takes one per service rather than holding one.
        """
        return service_cls(self, output)

    def connect(self) -> None:
        """Open the connection, attempting it at most once however often asked.

        A session is one invocation's connection, so "this appliance did
        not answer" is a fact for the rest of the run rather than a
        question to put again: the configured timeout applies to connecting
        as well as reading, so a firewall dropping SYNs costs one of those
        per attempt, and the availability probe, its fallback call and the
        command's own service all reach here. The remembered failure is
        re-raised, so a caller still reports it as its own.

        A connection already made is the same fact from the other side:
        re-opening one would spend a second token POST to replace the
        client this session's services are already holding.
        """
        if self.failure is not None:
            raise self.failure
        if self.client is not None:
            return
        try:
            self._open()
        except Exception as exc:
            self.failure = exc
            raise

    def _open(self) -> None:
        """Construct the SDK client, replacing any this session already had.

        The attempt itself; :meth:`connect` is what callers want, since it
        makes at most one of these. Raises whatever the SDK raises, for the
        calling service to report — it is the one the user asked for and
        can say which area could not reach the appliance.
        """
        proxy = self.a1000_settings.proxy
        proxies = {"http": proxy, "https": proxy} if proxy else None
        common = {
            "host": self.a1000_settings.host,
            "verify": self.a1000_settings.verify_ssl,
            "proxies": proxies,
        }
        # Constructing the client is itself a network call under
        # username/password auth: the SDK fetches a token before it has a
        # session to time out, so the bare ``requests.post`` is bounded here.
        with default_post_timeout(self.a1000_settings.timeout):
            # The same rule the availability probe reports by: a config
            # half-filled from config.example.yaml carries a stand-in token
            # beside credentials the user did set, and picking by bare
            # truthiness would authenticate with the stand-in while
            # ``check-access`` reported username/password.
            #
            # The fallback is gated as well as the choice between the two:
            # an example copied without filling anything in leaves a
            # stand-in in all three, which the probe reports as "No A1000
            # credentials configured".
            if is_real_credential(self.a1000_settings.token):
                client = A1000(token=self.a1000_settings.token, **common)
            else:
                client = A1000(
                    username=supplied_credential(self.a1000_settings.username),
                    password=supplied_credential(self.a1000_settings.password),
                    **common,
                )
        self.client = apply_request_timeout(client, self.a1000_settings.timeout)

    def close(self) -> None:
        """Drop the client, for every service built over this session."""
        self.client = None
