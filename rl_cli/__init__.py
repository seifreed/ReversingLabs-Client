"""ReversingLabs CLI - A modular command-line interface for ReversingLabs services."""

from rl_cli._version import __version__
from rl_cli.config import Settings, get_settings

__author__ = "Marc Rivero"

# ``get_settings`` beside ``Settings`` because it is the one a library
# consumer wants: ``Settings()`` holds only what it is handed, so it reads no
# config file and points at ReversingLabs' public clouds.
__all__ = ["Settings", "__version__", "get_settings"]
