"""The output formats this CLI speaks, named once for everything that reads them."""

from __future__ import annotations

from enum import Enum


class OutputFormat(Enum):
    """Supported output formats.

    A module of its own, holding nothing but the vocabulary, because two
    layers have to agree on it: the renderer in :mod:`rl_cli.render.output`
    and the config loader, which validates ``output.format`` against it.
    The loader must not import the renderer instead — that would pull in
    Rich, tabulate, the TOON encoder and the SARIF writer to find out that
    "json" is a word.
    """

    RICH = "rich"
    JSON = "json"
    YAML = "yaml"
    TABLE = "table"
    RAW = "raw"
    TOON = "toon"
    SARIF = "sarif"
