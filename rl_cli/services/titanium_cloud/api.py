"""What every TitaniumCloud service has: its credentials, and how it reads an answer."""

from __future__ import annotations

from functools import cached_property
from typing import Any

from ReversingLabs.SDK.helper import WrongInputError

from rl_cli.config import Settings
from rl_cli.services.base import BaseService
from rl_cli.services.credentials import supplied_credential
from rl_cli.services.http import apply_request_timeout
from rl_cli.services.protocols import Notifier


def _dotted(path: tuple[str, ...], depth: int) -> str:
    """``rl`` path down to ``depth``, for naming where an answer went wrong."""
    return ".".join(path[: depth + 1])


class TitaniumCloudApi(BaseService):
    """How a TitaniumCloud service reaches the API, and reads what comes back.

    There is no connection to share — every call builds its own
    ``ticloud.<API>`` handle, which is why these services are decorated with
    ``safe_call`` and not with ``with_client`` — so what they have in common
    is the configuration those handles are built from and the envelope every
    endpoint answers in. That is a base class rather than a collaborator
    like the A1000's :class:`~rl_cli.services.a1000.session.A1000Session`,
    which exists because constructing *that* client is a token POST.

    Which handles a service builds is the split itself: the network lookups
    reach four API families and the file-oriented calls reach six, and no
    method needs one from the other half.
    """

    def __init__(self, settings: Settings, output: Notifier | None = None):
        super().__init__(settings, output)
        self.ti_cloud_settings = settings.titanium_cloud

    # Private on purpose: this dict holds the password and the whole proxy
    # URL, and ``http://user:pass@proxy:8080`` is how a proxy credential is
    # written. A public name is one an output formatter can be handed, and
    # anything that renders it prints the credentials to stdout —
    # ``A1000Service.connection_summary`` redacts for the same reason.
    @cached_property
    def _api_config(self) -> dict[str, Any]:
        # Both credentials go through ``supplied_credential`` — the same
        # rule the availability probe reports by, and the one
        # ``A1000Session`` picks its authentication with. There is no
        # second credential to fall back to here, so a config still
        # carrying ``your_ticloud_username`` is exactly as unconfigured as
        # one with the line left out.
        config: dict[str, Any] = {
            "host": self.ti_cloud_settings.host,
            "username": supplied_credential(self.ti_cloud_settings.username),
            "password": supplied_credential(self.ti_cloud_settings.password),
            "user_agent": self.ti_cloud_settings.user_agent,
            "verify": self.ti_cloud_settings.verify_ssl,
        }
        if self.ti_cloud_settings.proxy:
            config["proxies"] = {
                "http": self.ti_cloud_settings.proxy,
                "https": self.ti_cloud_settings.proxy,
            }
        return config

    def _api(self, api_class: type) -> Any:
        """One SDK handle for one endpoint family, or a refusal to build it.

        Nothing here is worth sending without a credential. The SDK takes
        the pair exactly as given — ``session.auth = (None, None)`` — and
        requests then puts ``Basic Tm9uZTpOb25l`` on the wire, so an
        unconfigured profile would spend a metered request per command to
        be told 401, and hear that its credentials were rejected rather
        than never set.

        Refused for the same state and in the same place as on the A1000,
        whose SDK constructor raises ``WrongInputError`` when it is handed
        neither a token nor a pair; ``safe_call`` reports this the way it
        reports that one. A stand-in counts as no credential, because
        :func:`supplied_credential` has already made it ``None`` — which is
        what ``check-access`` says about the same config.
        """
        config = self._api_config
        if config["username"] is None or config["password"] is None:
            raise WrongInputError(
                "No TitaniumCloud credentials configured: set a username and a password for "
                f"profile '{self.settings.profile}'"
            )
        return apply_request_timeout(api_class(**config), self.ti_cloud_settings.timeout)

    def _rl_list(self, body: Any, *path: str, required: bool = False) -> list[Any] | None:
        """The list at ``rl.<path>`` of a parsed TitaniumCloud answer, or ``None``.

        Every networking endpoint answers one page as a list under the
        ``rl`` envelope, and the SDK's paging variants read exactly these
        keys. The bulk reputation query answers under the same envelope,
        which is why this is here rather than with the network lookups that
        use it most.

        A key that is simply absent is an empty page, which is what the
        SDK's own aggregators take it for. Anything else is an answer we
        could not read, and is reported as the failed lookup it is: no
        envelope at all, a step that is present but is not an object to
        descend into, or a leaf that is present but is not a list. Every
        step is checked, not just the top-level ``rl``, or a body shaped
        ``{"rl": {"uri_index": "n/a"}}`` answers "nothing found".

        ``required`` says the key has to be there: for an endpoint whose
        envelope key this repo cannot corroborate against the SDK, a
        missing key is far more likely to mean we are reading the wrong
        key than that the endpoint found nothing.

        Takes the parsed body rather than the response, so a caller that
        also needs the paging cursor out of the same envelope reads one
        ``json_on`` instead of two: ``requests.Response.json`` re-runs
        ``json.loads`` over the whole page each time.
        """
        node: Any = body.get("rl") if isinstance(body, dict) else None
        if not isinstance(node, dict):
            self.output.error("TitaniumCloud response carried no rl envelope")
            return None

        for depth, step in enumerate(path):
            expected: type = list if depth == len(path) - 1 else dict
            if step not in node:
                if not required:
                    return []
                self.output.error(f"TitaniumCloud response carried no rl.{_dotted(path, depth)}")
                return None
            node = node[step]
            if not isinstance(node, expected):
                self.output.error(
                    f"TitaniumCloud response carried {type(node).__name__} at "
                    f"rl.{_dotted(path, depth)}, not {expected.__name__}"
                )
                return None
        return node
