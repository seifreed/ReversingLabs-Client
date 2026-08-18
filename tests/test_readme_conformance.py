"""The README's examples must be commands the CLI actually has.

Every one of these was wrong at some point: `config test`, `ticloud
file-reputation`, `ticloud ip-reputation`, `yara-create --file`,
`--no-color`, `--debug`, and `create-pdf`/`download-pdf` in the command
table. They are copy-paste-and-fail for a new user, and nothing caught
them because documentation is not executed.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import re
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeGuard

import pytest
from click.testing import CliRunner
from pydantic_settings import BaseSettings

from rl_cli import services
from rl_cli.cli.commands.a1000 import a1000
from rl_cli.cli.commands.config import config
from rl_cli.cli.commands.ticloud import ticloud
from rl_cli.cli.main import cli
from rl_cli.config import clear_settings_cache, get_settings
from rl_cli.models.output_format import OutputFormat
from rl_cli.services import A1000Service, TitaniumCloudNetworkService, TitaniumCloudService
from rl_cli.services.base import BaseService
from tests.cli_support import GROUPS, commands_offering_yes, commands_that_ask

README = Path(__file__).resolve().parent.parent / "README.md"
_TEXT = README.read_text()

# Placeholders the README uses in place of real arguments.
_PLACEHOLDER = re.compile(r"<[^>]+>|SHA256_HASH|/path/to/\S+|rules\.yar")
SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# Every landmark this file slices the README on. `str.index` raises
# ValueError on a heading that has been renamed or reordered, and two of
# these slices are taken at import time to build `parametrize` lists — so
# one edited heading took all ~40 checks in this file down as a collection
# error reading "substring not found", naming neither the heading nor the
# README. A renamed landmark is one broken lookup, so it is now one named
# failure, and the checks that do not depend on it still run.
_TICLOUD_TABLE = "### TitaniumCloud Commands"
_A1000_TABLE = "### A1000 Command Groups"
_GLOBAL_FLAGS = "### Global Flags"
_LIBRARY = "## Python Library"
_CONFIG_ROW = "| `rl-cli config`"
_LANDMARKS = (_TICLOUD_TABLE, _A1000_TABLE, _GLOBAL_FLAGS, _LIBRARY, _CONFIG_ROW)


def _block(start: str, end: str) -> str:
    """The README between two landmarks, or nothing if either is gone."""
    if start not in _TEXT or end not in _TEXT:
        return ""
    return _TEXT[_TEXT.index(start) : _TEXT.index(end)]


def _needs(*landmarks: str) -> None:
    """Leave a missing landmark to the one test whose subject it is.

    Without this, renaming a heading failed every check that slices on it
    as well, burying the one failure that says which heading moved.
    """
    missing = [landmark for landmark in landmarks if landmark not in _TEXT]
    if missing:
        pytest.skip(f"README.md no longer has {missing}")


def test_the_readme_landmarks_these_checks_slice_on_are_present() -> None:
    missing = [landmark for landmark in _LANDMARKS if landmark not in _TEXT]
    assert not missing, f"README.md no longer has these landmarks: {missing}"


def _readme_invocations() -> list[str]:
    return sorted({m.strip() for m in re.findall(r"^\s*(rl-cli [^\n|#]+)", _TEXT, re.M)})


def _command_table_names() -> list[str]:
    return sorted(set(re.findall(r"`([a-z0-9][a-z0-9-]*)`", _block(_A1000_TABLE, _LIBRARY))))


def _ticloud_table_names() -> list[str]:
    block = _block(_TICLOUD_TABLE, _A1000_TABLE)
    return sorted(set(re.findall(r"^\| `([a-z0-9-]+)`", block, re.M)))


@pytest.mark.parametrize("invocation", _readme_invocations())
def test_documented_invocation_is_valid(invocation: str) -> None:
    """Parse each example with --help so no request is made."""
    parts = shlex.split(_PLACEHOLDER.sub("X", invocation))
    assert parts[0] == "rl-cli"

    result = CliRunner().invoke(cli, [*parts[1:], "--help"])
    combined = result.output + (result.stderr or "")
    assert "No such command" not in combined, invocation
    assert "no such option" not in combined.lower(), invocation


@pytest.mark.parametrize("name", _command_table_names())
def test_command_table_lists_real_commands(name: str) -> None:
    assert name in a1000.commands, f"README lists a1000 '{name}', which does not exist"


@pytest.mark.parametrize("name", _ticloud_table_names())
def test_ticloud_table_lists_real_commands(name: str) -> None:
    assert name in ticloud.commands, f"README lists ticloud '{name}', which does not exist"


def test_every_a1000_command_is_documented() -> None:
    """The other direction, which is how eleven commands went missing.

    Checking only documented-implies-exists lets a new command ship with
    no mention anywhere, and `config-dump`, `test`, `yara-content`,
    `yara-delete`, the two retro-status commands, `yara-update-interval`
    and the four `yara-repo-*` commands all did exactly that while a
    66-test conformance suite stayed green.
    """
    _needs(_A1000_TABLE, _LIBRARY)
    undocumented = sorted(set(a1000.commands) - set(_command_table_names()))
    assert not undocumented, f"A1000 commands missing from the README table: {undocumented}"


def test_every_ticloud_command_is_documented() -> None:
    _needs(_TICLOUD_TABLE, _A1000_TABLE)
    undocumented = sorted(set(ticloud.commands) - set(_ticloud_table_names()))
    assert not undocumented, f"ticloud commands missing from the README table: {undocumented}"


def test_documented_config_subcommands_exist() -> None:
    _needs(_CONFIG_ROW)
    row = next(line for line in _TEXT.splitlines() if line.startswith(_CONFIG_ROW))
    for name in re.findall(r"`([a-z-]+)`", row.split("|")[2]):
        assert name in config.commands, f"README lists config '{name}', which does not exist"


def _python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _TEXT, re.S)


def _built_service(value: ast.expr) -> type | None:
    """The service class an assignment builds, if it builds one.

    Both spellings the README uses: ``TitaniumCloudService(settings)`` and
    ``session.service(A1000SampleService)``. The name is resolved against
    ``rl_cli.services``, so an example naming a class the package does not
    export fails here rather than in a reader's terminal.
    """
    if not isinstance(value, ast.Call):
        return None
    if isinstance(value.func, ast.Name):
        name = value.func.id
    elif (
        isinstance(value.func, ast.Attribute)
        and value.func.attr == "service"
        and value.args
        and isinstance(value.args[0], ast.Name)
    ):
        name = value.args[0].id
    else:
        return None
    built = getattr(services, name, None)
    if isinstance(built, type) and issubclass(built, BaseService):
        return built
    return None


def _service_objects() -> dict[str, type]:
    """The service class behind each name the README's examples bind.

    Read out of the examples rather than tabulated here: the documented
    shape is a session handing out focused services, so which names exist
    is the README's business, and every one of them still has to be a real
    service whose methods the tests below check.
    """
    objects: dict[str, type] = {}
    for source in _python_blocks():
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            built = _built_service(node.value)
            if isinstance(target, ast.Name) and built is not None:
                objects[target.id] = built
    return objects


_SERVICE_OBJECTS = _service_objects()


def _service_calls(tree: ast.AST) -> list[tuple[str, str, ast.Call]]:
    """Every ``samples.x(...)`` / ``ticloud.x(...)`` call in a block."""
    return [
        (node.func.value.id, node.func.attr, node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in _SERVICE_OBJECTS
    ]


def _returns_a_list(service: str, method: str) -> bool:
    """Whether the wrapper is annotated ``list[...] | None``."""
    func: Any = getattr(_SERVICE_OBJECTS[service], method, None)
    return str(getattr(func, "__annotations__", {}).get("return", "")).startswith("list[")


def test_the_examples_still_bind_recognisable_services() -> None:
    """Every check below is vacuous if no name is recognised as a service."""
    assert "ticloud" in _SERVICE_OBJECTS
    a1000_services = [cls for cls in _SERVICE_OBJECTS.values() if issubclass(cls, A1000Service)]
    assert len(a1000_services) >= 2, "the A1000 examples no longer build focused services"


@pytest.mark.parametrize("source", _python_blocks())
def test_documented_python_calls_exist(source: str) -> None:
    """The library examples must call methods the services actually have."""
    for service, method, _ in _service_calls(ast.parse(source)):
        assert hasattr(_SERVICE_OBJECTS[service], method), f"{service}.{method} does not exist"


def _settings_builders() -> set[str]:
    """How the examples get a ``Settings``: the constructor, or a loader."""
    builders: set[str] = set()
    for source in _python_blocks():
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in {"Settings", "get_settings"}:
                builders.add(node.func.id)
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "Settings"
            ):
                builders.add(f"Settings.{node.func.attr}")
    return builders


def test_the_library_examples_build_settings_through_a_loader() -> None:
    """A documented example must reach the config file the CLI reads.

    ``Settings()`` holds only the values it is handed and looks at no disk,
    so the documented ``settings = Settings()`` left ``host`` on its
    default: with the appliance in a config file and the token in the
    environment, a library consumer's token — and every hash they looked
    up — went to ReversingLabs' public cloud instead of their own box.
    """
    builders = _settings_builders()

    assert builders, "no example builds settings at all; this check has stopped checking"
    assert "Settings" not in builders, (
        "a README example calls Settings() directly, which reads no config file — "
        "use get_settings() or Settings.load()"
    )


def test_the_library_examples_run_on_a_single_connection(
    a1000_connections, tmp_path, monkeypatch
) -> None:
    """The documented script must work, and authenticate exactly once.

    Run, not just parsed: the shape it teaches is focused services sharing
    one session, and the whole reason to teach it is that a service
    opening a connection of its own costs a token POST each. Every
    connection is counted at ``A1000Session._open``, and none is opened.
    """
    sample = tmp_path / "file.exe"
    sample.write_bytes(b"MZ stub")
    # TitaniumCloud has no session to stub, so the example's calls to it are
    # answered here rather than on the wire. Both halves: the example builds
    # each of them, which is the shape the CLI itself now uses — a command
    # takes the service for the endpoints it calls, not one class over all
    # ten API families.
    for service_cls in (TitaniumCloudService, TitaniumCloudNetworkService):
        for name in dir(service_cls):
            if not name.startswith("_") and callable(getattr(service_cls, name)):
                monkeypatch.setattr(service_cls, name, lambda self, *a, **kw: {})

    run_block = builtins.exec
    namespace: dict[str, Any] = {}
    for source in _python_blocks():
        runnable = source.replace("/path/to/file.exe", str(sample)).replace("SHA256_HASH", SHA256)
        run_block(compile(runnable, str(README), "exec"), namespace)

    assert len(a1000_connections) == 1, "the documented script authenticated more than once"
    assert namespace["session"].client is None, "session.close() left a live client"


def _list_valued_names(tree: ast.AST) -> set[str]:
    """Names bound to the result of a wrapper annotated ``list[...]``."""
    calls = {id(call): (service, method) for service, method, call in _service_calls(tree)}
    names = set()
    for node in ast.walk(tree):
        target: ast.expr
        if isinstance(node, ast.NamedExpr):
            target = node.target
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        else:
            continue
        bound = calls.get(id(node.value))
        if isinstance(target, ast.Name) and bound and _returns_a_list(*bound):
            names.add(target.id)
    return names


@pytest.mark.parametrize("source", _python_blocks())
def test_documented_python_uses_list_results_as_lists(source: str) -> None:
    """`r["matches"]`/`r.get("matches")` on a list-returning wrapper raises."""
    tree = ast.parse(source)
    lists = _list_valued_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert not (node.value.id in lists and node.attr == "get"), (
                f"{node.value.id} is a list, not a mapping"
            )
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            assert node.value.id not in lists, f"{node.value.id} is a list; it has no string keys"


def test_documented_global_flags_exist() -> None:
    _needs(_GLOBAL_FLAGS, _A1000_TABLE)
    documented = set(re.findall(r"`(--[a-z-]+)", _block(_GLOBAL_FLAGS, _A1000_TABLE)))
    real = {opt for param in cli.params for opt in param.opts}
    assert documented <= real, f"README documents flags the CLI lacks: {documented - real}"


# The README calls these two files "the complete reference", which is a
# claim about them that only holds while every key they document reaches a
# setting and every setting they leave out does not exist. Both drifted:
# `output.pager` was documented and read nowhere, and TICLOUD_USER_AGENT /
# RL_CLI_CONFIG_FILE were read and documented nowhere.
ENV_EXAMPLE = README.parent / ".env.example"
CONFIG_EXAMPLE = README.parent / "config.example.yaml"


def _is_section(annotation: Any) -> TypeGuard[type[BaseSettings]]:
    """A field whose value is itself a settings model, not a variable."""
    return isinstance(annotation, type) and issubclass(annotation, BaseSettings)


def _settings_models() -> list[type[BaseSettings]]:
    from rl_cli.config.settings import Settings

    sections = [
        field.annotation
        for field in Settings.model_fields.values()
        if _is_section(field.annotation)
    ]
    return [Settings, *sections]


def _documented_variables() -> set[str]:
    return set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", ENV_EXAMPLE.read_text(), re.M))


def _real_variables() -> set[str]:
    """Every environment variable the settings models actually read."""
    names = set()
    for model in _settings_models():
        prefix = model.model_config.get("env_prefix", "")
        names |= {
            f"{prefix}{name}".upper()
            for name, field in model.model_fields.items()
            # `missing_profile` is the loader talking to itself, not input;
            # a section is a group of variables rather than one of them.
            if not field.exclude and not _is_section(field.annotation)
        }
    return names


def _uncommented_env_example() -> str:
    """Every variable .env.example documents, with the optional ones enabled."""
    return "".join(
        re.sub(r"^#\s*", "", line) + "\n"
        for line in ENV_EXAMPLE.read_text().splitlines()
        if re.match(r"^#?\s*[A-Z][A-Z0-9_]+=", line)
    )


def test_the_documented_env_file_is_one_the_cli_can_run_with(tmp_path, monkeypatch) -> None:
    """Writing what this file says, in full, must not break every command.

    Nothing loaded the file before — the checks around this one compare its
    variable *names* against the models — and three of the settings it
    tells users to write took a ``StrictBool``, which refuses the strings
    an environment is made of. Uncommenting the lines the project documents
    made every command exit 1 on ``verify_ssl: Input should be a valid
    boolean``, with nothing to say which line to delete.
    """
    written = _uncommented_env_example()
    # Named here so a file that stops documenting them cannot leave this
    # test passing on a .env that no longer states the three.
    for variable in ("A1000_VERIFY_SSL", "TICLOUD_VERIFY_SSL", "TICLOUD_SHARE_URL_LOOKUPS"):
        assert f"{variable}=" in written, f".env.example no longer documents {variable}"

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(written)
    clear_settings_cache()

    result = CliRunner().invoke(cli, ["config", "list-profiles"])

    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert "cannot be used" not in result.output
    # The file has to have been read, or the run above proves only that an
    # ignored .env is harmless.
    assert get_settings().a1000.host == "https://your-a1000-instance.local"


def test_env_example_documents_nothing_the_code_ignores() -> None:
    documented = _documented_variables()
    assert documented <= _real_variables(), (
        f".env.example documents variables nothing reads: {documented - _real_variables()}"
    )


def test_env_example_documents_every_variable_the_code_reads() -> None:
    missing = _real_variables() - _documented_variables()
    assert not missing, f".env.example is missing variables the code reads: {missing}"


def test_config_example_sets_nothing_the_code_ignores() -> None:
    import yaml

    from rl_cli.config.settings import Settings

    profile = yaml.safe_load(CONFIG_EXAMPLE.read_text())["default"]
    for section, values in profile.items():
        model = Settings.model_fields[section].annotation
        assert _is_section(model), f"config.example.yaml sets an unknown section: {section}"
        fields = model.model_fields
        unknown = set(values) - set(fields)
        assert not unknown, f"config.example.yaml sets unknown {section} keys: {unknown}"


# `OutputFormat` is the one definition of this vocabulary, and everything in
# the code derives from it: `-o`'s choices, `config set-output` and the
# settings validator. Nothing in the documentation does — the names are
# written out again by hand in four places, so a format added to the enum
# works everywhere and is documented nowhere.
_OUTPUT_ROW = "| `--output <fmt>`, `-o` |"
_OUTPUTS_BLOCK = "### Supported Outputs"
_FORMATS = {member.value for member in OutputFormat}


def _output_flag_row() -> str:
    return next((line for line in _TEXT.splitlines() if line.startswith(_OUTPUT_ROW)), "")


def _commented_formats(path: Path) -> set[str]:
    """The comma-separated names commented beside an example's format setting."""
    lines = [line for line in path.read_text().splitlines() if re.search(r"format:|FORMAT=", line)]
    if not lines:
        return set()
    return {word.strip() for word in lines[0].rpartition("#")[2].split(",") if word.strip()}


_FORMAT_LISTS: dict[str, Callable[[], set[str]]] = {
    "the README --output row": lambda: set(
        re.findall(r"`([a-z0-9]+)`", _output_flag_row().rpartition("Output format:")[2])
    ),
    "config.example.yaml": lambda: _commented_formats(CONFIG_EXAMPLE),
    ".env.example": lambda: _commented_formats(ENV_EXAMPLE),
}


@pytest.mark.parametrize("where", sorted(_FORMAT_LISTS))
def test_the_documented_format_list_is_the_enum(where: str) -> None:
    """Each hand-written list of format names spells out exactly ``OutputFormat``."""
    documented = _FORMAT_LISTS[where]()

    assert documented, f"{where} no longer lists the output formats"
    assert documented == _FORMATS, (
        f"{where} misses {sorted(_FORMATS - documented)} "
        f"and names {sorted(documented - _FORMATS)}, which is no format the CLI has"
    )


def test_the_supported_outputs_block_names_every_format() -> None:
    """The block groups the formats by what they are for, so it names them all.

    Forward only: it also names the report formats a dynamic-analysis
    report comes in, which are not ``-o`` values.
    """
    block = re.search(rf"{re.escape(_OUTPUTS_BLOCK)}\n+```text\n(.*?)```", _TEXT, re.S)

    assert block, f"README.md no longer has a {_OUTPUTS_BLOCK} block"
    listed = block.group(1).lower()
    # Both word boundaries: without the closing one, "Rich-formatted
    # tables" satisfied the check for the `table` format, so deleting
    # `table` from the block left this green.
    missing = sorted(fmt for fmt in _FORMATS if not re.search(rf"\b{fmt}\b", listed))
    assert not missing, f"the Supported Outputs block does not name {missing}"


def test_every_output_format_a_readme_example_passes_exists() -> None:
    """An example is copy-paste, so `--output <fmt>` has to name a real format."""
    passed = set(re.findall(r"--output ([a-z0-9]+)", _TEXT))

    assert passed, "no example passes --output; this check has stopped checking"
    assert passed <= _FORMATS, f"a README example passes {sorted(passed - _FORMATS)}"


def test_the_documented_a1000_command_count_matches_the_group() -> None:
    """The A1000 heading states a number, and the landmark check reads only its prefix."""
    heading = re.search(rf"{re.escape(_A1000_TABLE)} \((\d+) Commands\)", _TEXT)

    assert heading, f"the {_A1000_TABLE} heading no longer states a command count"
    assert int(heading.group(1)) == len(a1000.commands), (
        f"the README heading says {heading.group(1)} A1000 commands, "
        f"and there are {len(a1000.commands)}"
    )


# ``config init`` prompts and takes no ``--yes`` on purpose: it is the
# wizard, and every value it writes comes from a prompt, so a run that
# cannot answer has nothing to save. The paragraph below is about the gates
# a script has to get past, so it leaves that one out --
# tests/test_cli_commands.py exempts it by name for the same reason.
_NOT_SCRIPTABLE = {"config init"}
_ASKING = "commands ask before they act"
_COUNT_WORDS = (
    "no",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
)


def _commands_that_prompt() -> set[str]:
    """Every command that asks, resolved rather than grepped for.

    The resolution is ``tests/cli_support.py``'s and shared with
    tests/test_cli_commands.py — this file used to carry its own copy of
    the same substring search, which is not a second derivation but the
    same one twice: both missed a question put through a helper, and both
    counted a comment that mentions ``confirmed()``. What this file adds
    is the claim about the README, which nothing else makes.
    """
    return set(commands_that_ask()) - _NOT_SCRIPTABLE


def _commands_offering_yes() -> set[str]:
    """Every command with a ``--yes`` to get past its prompt, read off the parser.

    The other half of the claim, and taken from click's parsed parameters
    rather than from the source, so it cannot go stale with the resolution
    above: a destructive command is one that offers the escape hatch, and
    the set that offers it is the set that must ask.
    """
    return commands_offering_yes() - _NOT_SCRIPTABLE


def _prompt_paragraph() -> str:
    if _ASKING not in _TEXT:
        return ""
    start = _TEXT.rindex("\n", 0, _TEXT.index(_ASKING)) + 1
    return _TEXT[start:].split("\n\n")[0]


def _documented_prompting() -> set[str]:
    """The commands the paragraph names, as ``group name``.

    Bare names are the A1000's, because the paragraph sits in the A1000
    section; a name qualified with its group is read as written. Spans
    opening with an option are the flags the sentence quotes, not commands.
    """
    documented = set()
    for span in re.findall(r"`([^`]+)`", _prompt_paragraph()):
        words = span.split()
        if words[0].startswith("-"):
            continue
        documented.add(" ".join(words[:2]) if words[0] in GROUPS else f"a1000 {words[0]}")
    return documented


def test_the_documented_prompting_commands_are_the_ones_that_prompt() -> None:
    """The paragraph names a set and states its size; both are claims about the code.

    The ``--yes`` sweep asserts that every command which prompts has the
    flag, so an eighth one shipping never failed it — and ``config
    create-profile`` became the eighth with the README still saying seven.

    The set is pinned against the commands that offer ``--yes`` rather
    than against "at least one, so the grep still works": a destructive
    command that quietly loses its confirmation keeps its flag, and a
    floor would have let it through with the README still naming it.
    """
    paragraph = _prompt_paragraph()
    real = _commands_that_prompt()

    assert paragraph, "the README no longer says which commands ask before they act"
    assert real == _commands_offering_yes(), (
        "a command offers --yes and no longer asks, or asks with no --yes to get past it: "
        f"{sorted(real ^ _commands_offering_yes())}"
    )
    assert _documented_prompting() == real, (
        f"the README's prompt paragraph misses {sorted(real - _documented_prompting())} "
        f"and names {sorted(_documented_prompting() - real)}"
    )
    assert len(real) < len(_COUNT_WORDS), "spell the count out and add the word to _COUNT_WORDS"
    stated = paragraph.split()[0].lower()
    assert stated == _COUNT_WORDS[len(real)], (
        f"the README says {stated} {_ASKING}, and {len(real)} do"
    )


def test_the_sdk_coverage_claim_matches_the_code() -> None:
    """The badge and the feature table state a number; keep it true.

    It went stale the moment a redundant wrapper was deleted — the count
    dropped to 59 while two places in the README still said 60. A claim
    nothing checks is a claim that drifts.

    A wrapper is counted by naming the SDK method, not by calling it on
    the spot. This looked for ``.method(`` and so measured the spelling
    rather than the claim: when the six IP lookups moved to
    ``self._first_page(address, self.client.network_urls_from_ip, ...)``
    — the method handed over to be called, which is what keeps the guard
    ahead of the SDK — the count fell to 53 and this failed a README
    sentence that was still true.
    """
    import re as _re

    from ReversingLabs.SDK.a1000 import A1000

    methods = {
        name
        for name, _ in inspect.getmembers(A1000, inspect.isfunction)
        if not name.startswith("_")
    }
    source = "\n".join(
        path.read_text()
        for path in Path(__file__).resolve().parent.parent.joinpath("rl_cli").rglob("*.py")
    )
    wrapped = {m for m in methods if _re.search(rf"\.{_re.escape(m)}\b", source)}

    claimed = _re.search(r"Wraps (\d+) of the (\d+) ReversingLabs A1000 SDK methods", _TEXT)
    assert claimed, "the SDK coverage sentence is gone from the README"
    assert (int(claimed.group(1)), int(claimed.group(2))) == (len(wrapped), len(methods))
    assert f"{len(wrapped)}%20of%20{len(methods)}" in _TEXT, "the coverage badge disagrees"
