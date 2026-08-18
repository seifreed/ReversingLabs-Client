"""The Yara Online Source repository: the fields written together, as one value."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

# What the token reads as once it has been shown to anyone. Masked rather
# than dropped so a reader can still tell a private repository from a
# public one, and so a `-o json` consumer keeps the key it indexes on.
_MASKED = "***"

# The field the appliance would be echoing our credential back in. It is a
# token for somebody else's system — a GitHub PAT, typically — and it is
# the one repository field the CLI never asks the appliance for.
_CREDENTIAL = "api_token"


@dataclass(frozen=True, slots=True, repr=False)
class YaraRepositorySpec:
    """What one Yara Online Source repository is, in the fields that define it.

    The create and update endpoints write these five as a unit — update is
    a full-resource PUT, so all five go out on every call. They travel as
    one value rather than five same-typed parameters threaded through two
    handlers and two service calls: nothing but the spelling of the
    parameter names holds that order together, so a name in the URL slot or
    a token in the branch slot type-checks, reaches the appliance, and
    replaces the repository with the fields transposed.

    ``source_branch`` is the one field that may be absent: naming no
    branch is how the endpoint is told to fall back to the repository's
    own ``main`` or ``master``, which is not the same as naming one.

    Frozen because it is the command line's settled answer — the spec is
    built once from the parsed options and read by the call that sends it.
    """

    repository_url: str
    name: str
    source_branch: str | None
    api_token: str
    import_update_preferences: str

    def __repr__(self) -> str:
        """The spec with its token masked, because a repr of it gets printed.

        ``services/decorators.py`` reports a failed service call as the
        SDK's message followed by ``method(<first argument>!r)``, and
        ``create_yara_repository(spec)`` takes this as that first argument.
        The generated repr would put a live GitHub token in the error line
        of any create that failed.
        """
        shown = []
        for field in fields(self):
            value = _MASKED if field.name == _CREDENTIAL else getattr(self, field.name)
            shown.append(f"{field.name}={value!r}")
        return f"{type(self).__name__}({', '.join(shown)})"


def redact_api_token(answer: Any) -> Any:
    """A repository record, or a list of them, with any token in it masked.

    Whatever these endpoints answer is handed straight to the formatter,
    so a field the appliance echoes back is printed verbatim — to a
    terminal, to a log, or into a ``-o json`` pipeline. For every other
    field that is the point; for the token it would publish a credential
    to a third-party repository that the analyst is shown nowhere else and
    that nothing here has any reason to read back.
    """
    if isinstance(answer, list):
        return [redact_api_token(entry) for entry in answer]
    if isinstance(answer, dict) and answer.get(_CREDENTIAL):
        return {**answer, _CREDENTIAL: _MASKED}
    return answer
