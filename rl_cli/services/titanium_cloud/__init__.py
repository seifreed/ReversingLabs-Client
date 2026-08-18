"""TitaniumCloud services, split by the area of the API they talk to.

Ask :class:`TitaniumCloudService` about a hash and
:class:`TitaniumCloudNetworkService` about an address. Neither answers to
the other's names, so a caller depends on the calls it actually makes::

    network = TitaniumCloudNetworkService(settings)
    network.get_files_from_ip("8.8.8.8")

There is no session to build them from, because there is no connection to
share: each call constructs its own SDK handle, so what the services have
in common is :class:`TitaniumCloudApi` — the credentials those handles are
built from, and the ``rl`` envelope every endpoint answers in.

Both are published: :mod:`rl_cli.services` re-exports each, and
``README.md`` constructs each by name.
"""

from rl_cli.services.titanium_cloud.api import TitaniumCloudApi
from rl_cli.services.titanium_cloud.network import TitaniumCloudNetworkService
from rl_cli.services.titanium_cloud.service import TitaniumCloudService

__all__ = [
    "TitaniumCloudApi",
    "TitaniumCloudNetworkService",
    "TitaniumCloudService",
]
