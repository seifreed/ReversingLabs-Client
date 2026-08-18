"""A1000 platform CLI commands, one module per area of the appliance.

The split follows what an analyst is doing, which is near the service
layer's seams but not the same line: ``reports`` and ``samples`` each reach
for two services, and ``yara``, ``yara_repos`` and ``yara_retro`` are three
workflows sharing the one ``A1000YaraService``.

Importing the modules is what registers their commands: each decorates with
``@a1000.command()``, so the group is complete once all are imported.
"""

from rl_cli.cli.commands.a1000 import (
    admin,
    metadata,
    network,
    reports,
    samples,
    yara,
    yara_repos,
    yara_retro,
)
from rl_cli.cli.commands.a1000._shared import a1000

__all__ = [
    "a1000",
    "admin",
    "metadata",
    "network",
    "reports",
    "samples",
    "yara",
    "yara_repos",
    "yara_retro",
]
