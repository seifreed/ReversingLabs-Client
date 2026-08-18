"""The package version, in a module that imports nothing.

``rl_cli/__init__.py`` re-exports ``Settings`` from ``rl_cli.config``, and
``rl_cli.config.settings`` needs the version for its default user agent.
Reading it back off the package root is a cycle, held together only by
``__version__`` being assigned above that import. A leaf both ends can
import has no such ordering to get wrong.
"""

__version__ = "0.1.0"
