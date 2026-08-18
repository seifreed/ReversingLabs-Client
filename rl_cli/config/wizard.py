"""The question sequence `config init` puts, with nothing that reads a terminal.

The sections come off the model, never from a list here, so that a
settings section which ships is one `config init` offers without an edit —
the same rule :func:`~rl_cli.config.settings.section_models` keeps on the
loading side.

The caller supplies the asking. This package may not import click or rich
(``tests/test_architecture.py``), which is what keeps the sequence
answerable by a test, a script or a different front end — and what keeps
the object being filled in, the one ``config save`` writes to a 0600
credentials file, out of reach of a terminal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any, Protocol

from pydantic_settings import BaseSettings

from rl_cli.config.settings import (
    A1000Settings,
    OutputSettings,
    TitaniumCloudSettings,
    section_models,
)
from rl_cli.models.output_format import OutputFormat


class Prompter(Protocol):
    """Whatever puts the questions and comes back with answers.

    ``secret`` is the one thing a caller may not infer for itself: it is
    what the CLI turns into ``hide_input=True``, and a password echoed to
    a terminal is the failure mode this argument exists to prevent.
    """

    def confirm(self, question: str, /, *, default: bool = False) -> bool: ...

    def ask(
        self,
        question: str,
        /,
        *,
        default: str | None = None,
        secret: bool = False,
        choices: Sequence[str] | None = None,
    ) -> str: ...


# Words that make a settings field a credential. Read against the field
# name because that is all a section nobody wrote a sequence for offers:
# getting this wrong prints a token on a terminal, so it errs towards
# hiding.
_CREDENTIAL_WORDS = ("password", "token", "secret", "key")

# A section's questions, put to whichever concrete settings model the
# field holds. Typed against ``Any`` rather than a union of the three:
# each sequence below is annotated with the model it actually asks about,
# which is what mypy checks the field names against.
type _Questions = Callable[[Any, Prompter], None]


def configure(settings: BaseSettings, prompter: Prompter) -> None:
    """Fill in whichever sections the user agrees to configure."""
    for name in section_models(type(settings)):
        question, ask_about = _SECTIONS.get(name, (_offer(name), _every_field))
        if prompter.confirm(question):
            ask_about(getattr(settings, name), prompter)


def _offer(name: str) -> str:
    """How a section with no sequence of its own is announced."""
    return f"Configure {name.replace('_', ' ')} settings?"


def _titanium_cloud(section: TitaniumCloudSettings, prompter: Prompter) -> None:
    section.host = prompter.ask("TitaniumCloud host", default=section.host)
    section.username = prompter.ask("Username", default=section.username or "")
    section.password = prompter.ask("Password", secret=True)


def _a1000(section: A1000Settings, prompter: Prompter) -> None:
    section.host = prompter.ask("A1000 host", default=section.host)

    if prompter.confirm("Use token authentication?"):
        section.token = prompter.ask("API Token", secret=True)
    else:
        section.username = prompter.ask("Username", default=section.username or "")
        section.password = prompter.ask("Password", secret=True)


def _output(section: OutputSettings, prompter: Prompter) -> None:
    # The answer is the chosen word; the field holds the member, so the
    # conversion happens here rather than leaving a string in a settings
    # object the rest of the CLI reads as an ``OutputFormat``.
    section.format = OutputFormat(
        prompter.ask(
            "Default output format",
            default=section.format.value,
            choices=tuple(fmt.value for fmt in OutputFormat),
        )
    )
    section.color = prompter.confirm("Enable colored output?", default=section.color)


def _every_field(section: BaseSettings, prompter: Prompter) -> None:
    """Ask about each field of a section that states no sequence of its own.

    Deliberately every field: the sequences above ask about the few that
    identify an appliance because that is the wording an analyst already
    knows, and a section nobody has worded yet has no such shortlist.
    """
    for name in type(section).model_fields:
        label = name.replace("_", " ").capitalize()
        current = getattr(section, name)
        if isinstance(current, bool):
            setattr(section, name, prompter.confirm(f"{label}?", default=current))
        elif isinstance(current, Enum):
            setattr(
                section,
                name,
                type(current)(
                    prompter.ask(
                        label,
                        default=str(current.value),
                        choices=tuple(str(member.value) for member in type(current)),
                    )
                ),
            )
        else:
            setattr(
                section,
                name,
                prompter.ask(
                    label,
                    default="" if current is None else str(current),
                    secret=any(word in name for word in _CREDENTIAL_WORDS),
                ),
            )


# The wording and the order an analyst following the README is answering.
# A section absent from here is offered by :func:`_offer` and asked about
# field by field, which is what makes a new one need no edit here either.
_SECTIONS: dict[str, tuple[str, _Questions]] = {
    "titanium_cloud": ("Configure TitaniumCloud API?", _titanium_cloud),
    "a1000": ("Configure A1000 platform?", _a1000),
    "output": ("Configure output settings?", _output),
}
