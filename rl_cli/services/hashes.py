"""The batch guard the bulk endpoints of both APIs share."""

from __future__ import annotations

from collections.abc import Callable

from rl_cli.models.validators import HashType, normalize_hash, validate_hash


def normalized_batch(
    hashes: list[str],
    allowed: tuple[HashType, ...],
    on_error: Callable[[str], None],
) -> list[str] | None:
    """The batch as the bulk endpoints want it, or ``None`` after saying why.

    Two jobs, both of which the SDK's own refusal leaves undone. It names
    the entry that sank the batch, where the SDK's "the given hash input
    string is not a valid hexadecimal value" says neither which entry nor
    where it sat — no help at all in a 500-line ``--hash-file``. And it
    answers the batch rather than a bool, because the judgement is made on
    the stripped, lowercased entry: a caller handed a yes/no would forward
    its own raw list, sending hashes the guard never looked at.

    ``allowed`` is the endpoint family's own rule: the A1000 bulk calls take
    every type the appliance recognises, TitaniumCloud's take three of them.
    ``on_error`` is how the caller says so, and is where a service adds
    whatever its own refusal has to spell out.
    """
    batch: list[str] = []
    for position, hash_value in enumerate(hashes, start=1):
        normalised = normalize_hash(hash_value)
        if normalised is None or validate_hash(normalised) not in allowed:
            on_error(f"Invalid hash format at entry {position} of {len(hashes)}: {hash_value}")
            return None
        batch.append(normalised)
    return batch
