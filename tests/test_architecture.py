"""The layering, as a rule that fails rather than a sentence in a docstring.

Every dependency direction this package relies on was written down
somewhere -- ``services/protocols.py`` says no service imports the console,
``models/payload.py`` says it cannot reach the renderers, ``text.py`` says it
imports nothing from the package. All of it was true and none of it was
checked, which is the same shape as the two live defects this suite has
already grown tests for: a ``.gitignore`` asserting "nothing tracked in this
repo matches any of them", and a placeholder check whose tests fed the
constant back into itself. A claim and its enforcement have to live in the
same place.

Read from the import graph rather than by importing the modules, so a cycle
is reported as a cycle instead of as an ``ImportError`` from whichever module
happened to be imported first.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent / "rl_cli"

# What each layer is allowed to import, and nothing else. `render.formatters`
# is called out separately from `render` because it is the top of the drawing
# stack: it may read the rest of `render`, and the rest of `render` may not
# read it back.
# Modules that import nothing from the package at all. Any layer may read
# one, because depending on a leaf cannot make a cycle -- which is the whole
# reason `text` sits at the top level and `_version` is its own module
# rather than a constant on the package root.
_LEAVES = {"text", "_version"}

_ALLOWED = {
    "models": {"models"},
    "storage": {"storage"},
    "config": {"config", "storage", "models"},
    "render": {"render", "models"},
    "render.formatters": {"render.formatters", "render", "models"},
    "services": {"services", "config", "models", "storage"},
    "cli": {
        "cli",
        "config",
        "models",
        "services",
        "render",
        "render.formatters",
        "storage",
        "rl_cli",
    },
    "rl_cli": {"config"},
    **{leaf: set() for leaf in _LEAVES},
}


def _layer(module: str) -> str:
    """Which layer ``rl_cli.a.b.c`` belongs to.

    ``rl_cli/__init__.py`` reads as ``rl_cli.__init__`` and is the root
    package itself, not a layer named ``__init__`` -- every other
    ``__init__`` carries its own package's name one level up.
    """
    parts = module.split(".")
    if len(parts) < 2 or parts[1] == "__init__":
        return "rl_cli"
    if parts[1] == "render" and len(parts) > 2 and parts[2] == "formatters":
        return "render.formatters"
    return parts[1]


def _imports() -> dict[str, set[str]]:
    """Every in-package import, as ``module -> the modules it imports``."""
    graph: dict[str, set[str]] = defaultdict(set)
    for path in _ROOT.rglob("*.py"):
        module = ".".join(path.relative_to(_ROOT.parent).with_suffix("").parts)
        # A module that imports nothing still has to appear as a node, or
        # `test_text_is_a_leaf` would pass by not finding it.
        graph.setdefault(module, set())
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("rl_cli")
            ):
                graph[module].add(node.module)
            elif isinstance(node, ast.Import):
                graph[module].update(n.name for n in node.names if n.name.startswith("rl_cli"))
    return graph


_GRAPH = _imports()


def test_the_import_graph_was_read_at_all():
    """A sweep over an empty graph passes every assertion below it."""
    assert len(_GRAPH) > 50, "expected the whole package; found almost nothing to check"
    assert any(_layer(m) == "services" for m in _GRAPH)


@pytest.mark.parametrize("module", sorted(_GRAPH))
def test_a_module_only_imports_the_layers_its_own_is_allowed(module):
    """The layer table is the whole rule; anything absent from it is a break."""
    origin = _layer(module)
    permitted = _ALLOWED[origin] | _LEAVES
    for imported in sorted(_GRAPH[module]):
        target = _layer(imported)
        if target == origin:
            continue
        assert target in permitted, (
            f"{module} imports {imported}: {origin} may not depend on {target}"
        )


def test_the_layers_are_a_dag():
    """No two layers may depend on each other, however indirectly.

    Two packages can be mutually dependent with no single module in a
    cycle: ``models.payload`` once took ``sanitize`` from the package whose
    renderers read ``models.payload``, and no import ever failed. So the
    rule is checked over the edges between layers, not between modules.
    """
    edges: dict[str, set[str]] = defaultdict(set)
    for module, imported in _GRAPH.items():
        origin = _layer(module)
        edges[origin].update(_layer(i) for i in imported if _layer(i) != origin)

    # `.get`, not `edges[...]`: `edges` is a defaultdict, and indexing a
    # layer with no outgoing edges inserts it, which is a mutation in the
    # middle of iterating the same mapping below.
    def reaches(start: str) -> set[str]:
        seen: set[str] = set()
        stack = list(edges.get(start, ()))
        while stack:
            layer = stack.pop()
            if layer not in seen:
                seen.add(layer)
                stack.extend(edges.get(layer, ()))
        return seen

    downstream = {layer: reaches(layer) for layer in list(edges)}
    cycles = sorted(
        f"{a} <-> {b}"
        for a, from_a in downstream.items()
        for b in from_a
        if a < b and a in downstream.get(b, ())
    )
    assert not cycles, f"layers depend on each other: {cycles}"


def test_text_is_a_leaf():
    """It is at the top level *because* it imports nothing from the package.

    Put an in-package import here and it stops being safe for ``models`` and
    ``services`` to reach, which is a cycle between two packages.
    """
    assert _GRAPH["rl_cli.text"] == set()


def test_no_service_reaches_the_renderers():
    """What ``services/protocols.py`` says ``NullNotifier`` exists to ensure.

    Stated against the whole of ``render`` rather than against Rich by name,
    because the way this leaks is a module that imports the console, not the
    console itself: ``services.availability`` once pulled Rich in through a
    sanitizer that lived beside the renderers.
    """
    for module, imported in _GRAPH.items():
        if _layer(module) != "services":
            continue
        for target in imported:
            assert not target.startswith("rl_cli.render"), f"{module} imports {target}"
            assert not target.startswith("rl_cli.cli"), f"{module} imports {target}"


def test_models_and_config_name_no_console_and_no_appliance():
    """These two are read under ``-o json``, where no console is built.

    The SDK is named here beside Rich and Click because these layers state
    the rules both sides of a boundary agree on -- what a payload, a hash or
    a run's exit status is -- and a value that has to import the appliance
    client to be defined is a service's, not a rule.
    """
    for layer in ("models", "config"):
        for path in (_ROOT / layer).rglob("*.py"):
            source = path.read_text()
            for node in ast.walk(ast.parse(source)):
                names = []
                if isinstance(node, ast.Import):
                    names = [n.name for n in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    root = name.split(".")[0]
                    # Named by path from the package root, not by basename:
                    # the two layers could each hold a `shapes.py`, and
                    # "shapes.py imports rich" would not say which.
                    assert root not in {"rich", "click", "ReversingLabs"}, (
                        f"{path.relative_to(_ROOT)} imports {name}"
                    )
