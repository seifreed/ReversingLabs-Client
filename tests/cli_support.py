"""What every CLI command test needs: an invocation context, and stubs for the appliance.

``tests/test_cli_commands.py`` grew to hold the whole click layer's tests
and the scaffolding they share. The tests now sit in the file for the area
they are about — yara, downloads, config, ticloud, the command matrix —
and the scaffolding is here rather than copied into each of them.

Not ``conftest.py``: that file is the suite-wide hermeticity contract,
which every test in the repository inherits whether it touches the CLI or
not. These helpers are the CLI tests' own, so they are imported by name.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock

import click
from click.testing import CliRunner, Result

from rl_cli.cli.commands.a1000 import a1000
from rl_cli.cli.commands.config import config
from rl_cli.cli.commands.ticloud import ticloud
from rl_cli.cli.context import CliContext
from rl_cli.config import Settings
from rl_cli.render.output import OutputFormat, OutputFormatter, RichOutput
from rl_cli.services.a1000 import (
    A1000MetadataService,
    A1000NetworkService,
    A1000ReportService,
    A1000SampleService,
    A1000Service,
    A1000Session,
    A1000YaraService,
)
from rl_cli.services.base import BaseService

SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SHA1 = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

# Every service the a1000 commands build. The group's commands are spread
# across them, so a run has to stub the lot to cover any one command.
A1000_SERVICES = (
    A1000Service,
    A1000SampleService,
    A1000ReportService,
    A1000NetworkService,
    A1000YaraService,
    A1000MetadataService,
)


def make_context(tmp_path) -> CliContext:
    """What the root group puts in ``ctx.obj``, pointed at the test's own tmp_path.

    Each CLI test module wraps this in its own one-line ``ctx_obj``
    fixture; the object itself is built the same way for all of them.
    """
    settings = Settings(cache_dir=tmp_path / "cache", config_dir=tmp_path / "config")
    return CliContext(
        settings=settings,
        output=RichOutput(),
        formatter=OutputFormatter(OutputFormat.JSON),
        # What the root group puts there: one connection per invocation,
        # which every A1000 service the command builds is built over.
        session=A1000Session(settings),
        # These tests invoke the groups directly, so there is no root
        # callback and nothing was probed.
        warn_if_unavailable=None,
    )


def invoke(group, args, ctx_obj, **kwargs) -> Result:
    return CliRunner().invoke(group, args, obj=ctx_obj, **kwargs)


def stub_response(payload) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    return response


def stub_client(ctx_obj) -> MagicMock:
    """Give the invocation's session a stub SDK client — no appliance is reached."""
    client = MagicMock()
    ctx_obj.session.client = client
    return client


def stub_service(monkeypatch, service_cls, payload) -> None:
    """Answer every service call of ``service_cls`` with ``payload``.

    Each service inherits the connection and the shared client methods, so
    ``vars(service_cls)`` alone would leave those live and every command
    would render nothing. The walk stops at ``BaseService``, whose helpers
    are not service calls.
    """
    for klass in service_cls.__mro__:
        if klass is BaseService:
            break
        for name, attribute in vars(klass).items():
            if (
                name.startswith("_")
                or isinstance(attribute, staticmethod)
                or not callable(attribute)
            ):
                continue
            monkeypatch.setattr(service_cls, name, lambda self, *args, **kwargs: payload)


def flat(result) -> str:
    """The rendered output with rich's line wrapping folded back out."""
    return " ".join(result.output.split())


# ---------------------------------------------------------------------------
# What a command reaches, resolved rather than grepped for.
#
# Three files ask questions of the shape "which commands do X": that every
# command which asks before it acts offers ``--yes``, that the README names
# those commands, and that every command reaching an irreversible operation
# asks at all. They used to answer it three times over, and two of those
# three answers were the same substring search over ``inspect.getsource`` —
# so a marker that stopped matching stopped matching in both at once, and
# neither would have said so. Worse, the search saw only the command's own
# body: a command whose guard lived one helper call away read as a command
# that never asks, which is the cron-hang the ``--yes`` flag exists to
# prevent, shipping with the sweep green.
#
# So the resolution lives here once, over the parsed modules, and the three
# files make three different claims on top of it.
# ---------------------------------------------------------------------------

GROUPS = {"a1000": a1000, "ticloud": ticloud, "config": config}

_COMMANDS_PACKAGE = "rl_cli.cli.commands"


def _commands_root() -> Path:
    from rl_cli.cli import commands

    return Path(commands.__file__).parent


def command_modules() -> list[Path]:
    """Every module the command groups are written in."""
    modules = sorted(_commands_root().rglob("*.py"))
    assert len(modules) > 5, "the command modules moved; this sweep is reading nothing"
    return modules


@cache
def module_tree(module: Path) -> ast.Module:
    return ast.parse(module.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Which definition a name means: the one in scope where the call is
# written, not the first module — or the first closure — that happened to
# spell the name.
#
# The lookup used to be one ``setdefault`` over every command module at
# once, so a name two modules both define resolved to whichever module
# sorted first. Nine names collide today — ``confirm``, ``search``,
# ``download``, ``upload``, ``lookup``, ``ip_report``, ``domain_report``,
# ``decorate``, ``__call__`` — and the consequence was live: ``config
# init`` calls ``click.confirm``, which records the bare name ``confirm``,
# and the resolver handed it ``a1000/metadata.py``'s nested ``confirm``,
# whose body calls ``confirmed``. The command was credited with reaching
# the shared guard it has never called.
#
# Answering per module was half of it. The other half is that a module is
# not one scope: nearly every command here writes a local ``def fetch()``,
# and a lookup that answered with every same-named definition in the file
# handed one command its neighbour's closure. A read-only ``info`` next to
# a ``delete`` came out as a command that deletes and asks nothing, which
# is a contributor being told to put a prompt in front of a command that
# touches nothing.
#
# So a name resolves in the scopes enclosing the body that wrote it,
# innermost first, and crosses a file boundary only along an import that
# module actually writes.
# ---------------------------------------------------------------------------


class _Scopes(NamedTuple):
    """Which names each scope of one module binds, and what encloses each scope.

    A scope is the module or a ``def``. A ``class`` body is neither: its
    methods are reached through an instance rather than by the lexical
    name, so they are bound where the class is — which is what lets
    ``config init``'s ``click.confirm`` resolve to ``_ClickPrompter``'s
    one-line ``confirm`` and not to ``a1000/metadata.py``'s.
    """

    binds: dict[ast.AST, dict[str, tuple[ast.AST, ...]]]
    enclosing: dict[ast.AST, ast.AST]
    module: ast.Module

    def visible_from(self, node: ast.AST, name: str) -> tuple[ast.AST, ...]:
        """What ``name`` binds to for a body written at ``node``, innermost scope first."""
        scope: ast.AST | None = node
        while scope is not None:
            found = self.binds.get(scope, {}).get(name)
            if found:
                return found
            scope = self.enclosing.get(scope)
        return ()

    def at_module_level(self, name: str) -> tuple[ast.AST, ...]:
        """What ``name`` binds to for anything importing this module."""
        return self.binds.get(self.module, {}).get(name, ())


def _bound_by(node: ast.AST) -> list[tuple[str, ast.AST]]:
    """The names ``node`` binds, each with the body a resolution walks into.

    A ``def`` and a ``class`` bind themselves. An assignment binds its
    value, which is how a method parked in a module-level table or on a
    class attribute is reached at all: the command names the table, and
    the table is where the method is.
    """
    if isinstance(node, ast.FunctionDef | ast.ClassDef):
        return [(node.name, node)]
    if isinstance(node, ast.Assign):
        return [(target.id, node.value) for target in node.targets if isinstance(target, ast.Name)]
    if isinstance(node, ast.AnnAssign) and node.value is not None:
        return [(node.target.id, node.value)] if isinstance(node.target, ast.Name) else []
    return []


def _scopes_of(tree: ast.Module) -> _Scopes:
    """Read one module's scopes off its tree.

    Only a ``def`` opens one, so an ``if`` or a ``with`` binds into the
    scope it is written in, and a ``class`` body binds into the scope the
    class is written in.
    """
    binds: dict[ast.AST, dict[str, list[ast.AST]]] = {}
    enclosing: dict[ast.AST, ast.AST] = {}

    def collect(scope: ast.AST, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            for name, body in _bound_by(child):
                binds.setdefault(scope, {}).setdefault(name, []).append(body)
                enclosing[body] = scope
            collect(child if isinstance(child, ast.FunctionDef) else scope, child)

    collect(tree, tree)
    return _Scopes(
        {
            scope: {name: tuple(bodies) for name, bodies in named.items()}
            for scope, named in binds.items()
        },
        enclosing,
        tree,
    )


@cache
def _scopes(module: Path) -> _Scopes:
    return _scopes_of(module_tree(module))


def _package_of(module: Path) -> str:
    """The dotted package a relative import inside ``module`` is relative to."""
    return ".".join([_COMMANDS_PACKAGE, *module.relative_to(_commands_root()).parts[:-1]])


def _module_named(dotted: str) -> Path | None:
    """The command module ``dotted`` names, or ``None`` for anything outside."""
    if dotted != _COMMANDS_PACKAGE and not dotted.startswith(f"{_COMMANDS_PACKAGE}."):
        return None
    tail = dotted[len(_COMMANDS_PACKAGE) :].strip(".")
    base = _commands_root().joinpath(*tail.split(".")) if tail else _commands_root()
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _import_source(module: Path, node: ast.ImportFrom) -> Path | None:
    """The command module ``node`` imports from, relative form included."""
    dotted = node.module or ""
    if node.level:
        package = _package_of(module)
        for _ in range(node.level - 1):
            package = package.rpartition(".")[0]
        dotted = f"{package}.{dotted}" if dotted else package
    return _module_named(dotted)


@cache
def command_module_imports() -> dict[Path, dict[str, Path]]:
    """Per command module, the names it imports from another command module.

    The only edges a name may cross a file on. Published so that the tests
    can claim what this file merely implements: that a resolution landing
    in another module landed there along an import somebody wrote.
    """
    edges: dict[Path, dict[str, Path]] = {}
    for module in command_modules():
        names: dict[str, Path] = {}
        for node in ast.walk(module_tree(module)):
            if not isinstance(node, ast.ImportFrom):
                continue
            source = _import_source(module, node)
            if source is None or source == module:
                continue
            for alias in node.names:
                names[alias.asname or alias.name] = source
        edges[module] = names
    return edges


class _Body(NamedTuple):
    """Something a name binds to, and the module that wrote it.

    The module travels with the body because it is what the next name
    inside it resolves against, and the name because a body that is a
    value rather than a ``def`` has no name of its own to be walked under.
    """

    module: Path | None
    name: str
    node: ast.AST


type _Resolver = Callable[[_Body, str], tuple[_Body, ...]]


def _published_by(module: Path, name: str) -> tuple[_Body, ...]:
    """What ``module`` offers an importer under ``name``, its re-exports followed.

    ``a1000/__init__`` hands on what ``_shared`` defines, so the walk
    continues along the import; it stops on a module already visited, so a
    cycle of re-exports cannot spin.
    """
    seen: set[Path] = set()
    here: Path | None = module
    while here is not None and here not in seen:
        seen.add(here)
        found = _scopes(here).at_module_level(name)
        if found:
            return tuple(_Body(here, name, node) for node in found)
        here = command_module_imports().get(here, {}).get(name)
    return ()


def _resolve_in_commands(origin: _Body, name: str) -> tuple[_Body, ...]:
    """What ``name`` means where ``origin`` is written: a scope it sits in, else an import."""
    module = origin.module
    if module is None:
        return ()
    found = _scopes(module).visible_from(origin.node, name)
    if found:
        return tuple(_Body(module, name, node) for node in found)
    source = command_module_imports().get(module, {}).get(name)
    return _published_by(source, name) if source else ()


# ---------------------------------------------------------------------------
# What a body reaches, and how firmly.
#
# The resolution used to record a name only where it stood in a call's
# ``func`` position, which reads a bound method handed over as a value as a
# method nobody touched. Verified against the real sweep: of the five ways
# this codebase can spell a delete, only ``service.delete_sample(h)`` was
# seen. ``partial(service.delete_sample, h)`` was not — and that is the
# shape ``run_step`` invites, since it takes a callable, and the shape
# ``a1000/samples.py`` already writes for its renderer. A delete command
# written that way asked nothing and passed every sweep.
#
# So a name *referenced* counts, not only one *called*. That over-reports:
# a name is reported as reached where the body only mentions it — a method
# looked up and thrown away, or a local variable that happens to be spelled
# like a helper. The two questions asked of a reach want opposite
# treatments of that, so they are answered separately rather than
# averaged:
#
# * "does this command act irreversibly?" reads ``referenced``. Handing a
#   delete to something that takes a callable is a delete, and a false
#   positive costs a command being made to ask.
# * "does this command put its question?" reads ``invoked``. A question is
#   put by being asked; a mention of ``confirmed`` is not consent. A false
#   positive there would excuse a silent delete, so that direction stays
#   strict.
#
# Two readings, and therefore two walks. One walk folding both readings
# was strict in name only: it followed references to decide which bodies
# to open, then credited the command with everything those bodies call. A
# command that merely named a guarded helper — and called it nowhere —
# came out as a command that asks. The edges a reading follows have to be
# the edges it reports, so ``invoked`` is closed over invocations alone.
#
# Comments and docstrings are in neither: neither is an ``ast.Name``.
# ---------------------------------------------------------------------------


class Reach(NamedTuple):
    """What a body reaches, and the bodies the resolution walked to say so."""

    invoked: frozenset[str]
    referenced: frozenset[str]
    walked: frozenset[str]


def _names_of(expression: ast.expr) -> set[str]:
    """The name an expression is, dotted form included where it has one."""
    if isinstance(expression, ast.Name):
        return {expression.id}
    if isinstance(expression, ast.Attribute):
        names = {expression.attr}
        if isinstance(expression.value, ast.Name):
            names.add(f"{expression.value.id}.{expression.attr}")
        return names
    return set()


def _attribute_literal(expression: ast.expr) -> set[str]:
    """The attribute a ``getattr(x, "name")`` names, when the name is a literal.

    The one shape with no ``ast.Attribute`` to read: the method is picked
    by string. A computed string is out of reach of any parse and stays
    invisible — recorded here rather than papered over, since the answer
    for it is the same as for a dispatch table built at runtime.
    """
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "getattr"
        and len(expression.args) >= 2
        and isinstance(expression.args[1], ast.Constant)
        and isinstance(expression.args[1].value, str)
    ):
        return {expression.args[1].value}
    return set()


def _invoked_names(node: ast.AST) -> set[str]:
    """Every name ``node`` puts in call position.

    ``click.confirm(...)`` answers ``click.confirm`` and ``confirm``;
    ``service.delete_sample(...)`` answers ``delete_sample``. Nested
    functions and lambdas come with the walk, which is what sees the call
    a command makes from inside ``run_step(..., lambda: ...)``.
    """
    names: set[str] = set()
    for call in ast.walk(node):
        if isinstance(call, ast.Call):
            names |= _names_of(call.func) | _attribute_literal(call.func)
    return names


def _referenced_names(node: ast.AST) -> set[str]:
    """Every name ``node`` reads, whether or not it calls it.

    A load, so ``self.pages = ...`` is not a reference to ``pages``: what
    is being asked is which behaviour this body can hand on, and a write
    hands on nothing.
    """
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name | ast.Attribute) and isinstance(child.ctx, ast.Load):
            names |= _names_of(child)
        elif isinstance(child, ast.Call):
            names |= _attribute_literal(child)
    return names


class _Edges(NamedTuple):
    """One reading of a body, and the bindings a name read that way is followed to.

    ``into_values`` is what keeps the two readings apart. A reference may
    land on a value — a dispatch table, a class attribute — and reading
    that value is how a delete parked there is seen at all. An invocation
    may not: a name called is a ``def`` called, and walking into a class
    a body merely constructs would credit it with every question its
    methods put.
    """

    names: Callable[[ast.AST], set[str]]
    into_values: bool


_INVOCATIONS = _Edges(_invoked_names, into_values=False)
_REFERENCES = _Edges(_referenced_names, into_values=True)


def _closure(start: _Body, resolve: _Resolver, edges: _Edges) -> tuple[set[str], set[str]]:
    """Every name ``start`` reaches along ``edges``, and the bodies walked to say so.

    The transitive closure, not the body: a command that puts its question
    through a helper is a command that asks, and a command that deletes
    through one is a command that deletes.

    Keyed on the body rather than on its name — two modules may each write
    a ``confirm``, and one of them standing for the other is the bug this
    resolution exists to have stopped making.
    """
    names: set[str] = set()
    walked: set[str] = set()
    pending = [start]
    seen: set[ast.AST] = set()
    while pending:
        body = pending.pop()
        if body.node in seen:
            continue
        seen.add(body.node)
        walked.add(f"{body.module.name}:{body.name}" if body.module else body.name)
        for name in edges.names(body.node):
            names.add(name)
            pending.extend(
                found
                for found in resolve(body, name)
                if edges.into_values or isinstance(found.node, ast.FunctionDef)
            )
    return names, walked


def _reach(start: _Body, resolve: _Resolver) -> Reach:
    """Everything ``start`` reaches, through the helpers ``resolve`` finds.

    Twice over, because the two readings follow different edges. What was
    walked is the reference closure's, which is the wider of the two: a
    name a body calls is a name it reads.
    """
    invoked, _ = _closure(start, resolve, _INVOCATIONS)
    referenced, walked = _closure(start, resolve, _REFERENCES)
    return Reach(frozenset(invoked), frozenset(referenced), frozenset(walked))


def reached_in(tree: ast.Module, function_name: str) -> Reach:
    """The reach of a function in ``tree``, a module the resolver has never seen.

    What the tests that prove this resolution use, so the shape they claim
    is caught is written out rather than made by editing a real command.
    The rule is the real one: the scopes enclosing the call first, then
    the command modules the tree writes an import for — and nothing else,
    so a synthetic module cannot borrow a body it never asked for.
    """
    scopes = _scopes_of(tree)
    imported: dict[str, Path] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            source = _module_named(node.module or "")
            if source is not None:
                for alias in node.names:
                    imported[alias.asname or alias.name] = source

    def resolve(origin: _Body, name: str) -> tuple[_Body, ...]:
        if origin.module is not None:
            return _resolve_in_commands(origin, name)
        found = scopes.visible_from(origin.node, name)
        if found:
            return tuple(_Body(None, name, node) for node in found)
        source = imported.get(name)
        return _published_by(source, name) if source else ()

    start = scopes.at_module_level(function_name)[0]
    return _reach(_Body(None, function_name, start), resolve)


def commands() -> dict[str, click.Command]:
    """Every registered command, as ``group name``."""
    return {
        f"{group_name} {name}": command
        for group_name, group in GROUPS.items()
        for name, command in group.commands.items()
    }


def _callback_body(command: click.Command) -> _Body | None:
    """The ``def`` behind ``command``, and the module that wrote it."""
    callback = command.callback
    if callback is None:
        return None
    module = Path(sys.modules[callback.__module__].__file__ or "")
    for node in ast.walk(module_tree(module)):
        if isinstance(node, ast.FunctionDef) and node.name == callback.__name__:
            return _Body(module, node.name, node)
    return None


def reached_by(command: click.Command) -> Reach:
    """Everything ``command`` reaches, its helpers included."""
    start = _callback_body(command)
    empty: frozenset[str] = frozenset()
    return Reach(empty, empty, empty) if start is None else _reach(start, _resolve_in_commands)


# The two ways this CLI puts a question: the guard every command that acts
# irreversibly shares, and the bare prompt the config wizard still writes.
# Named as calls rather than as text, so a comment mentioning ``confirmed()``
# is not a command that asks and a docstring quoting ``click.confirm`` is
# not either — both of which the substring sweep counted.
ASKING_CALLS = frozenset({"confirmed", "click.confirm"})


def commands_that_ask(over: dict[str, click.Command] | None = None) -> dict[str, click.Command]:
    """Every command that puts a question before it acts, guard or prompt.

    Read off what the command *invokes*: a question is put by being asked.
    A body that only mentions ``confirmed`` — hands it to something, keeps
    it in a table, names it in a docstring — has not asked, and counting
    that as consent is the one over-report that would excuse a silent
    delete.

    ``over`` narrows the sweep to a caller's own commands, which is how
    the tests point the real sweep at a command written to fail it rather
    than at the ones it was written against.
    """
    return {
        where: command
        for where, command in (commands() if over is None else over).items()
        if reached_by(command).invoked & ASKING_CALLS
    }


def commands_offering_yes(over: dict[str, click.Command] | None = None) -> set[str]:
    """Every command with a ``--yes`` to get past its question, off the parser.

    The other half of the claim, and taken from click's parsed parameters
    rather than from the source, so it cannot go stale with the resolution
    above.
    """
    return {
        where
        for where, command in (commands() if over is None else over).items()
        if any("--yes" in param.opts for param in command.params)
    }
