"""Shared safety rules for commands that write downloaded samples."""

from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path

from rl_cli.cli.context import run_step
from rl_cli.render.output import RichOutput
from rl_cli.storage.archives import private_mkdir


def fetch_into(
    output: RichOutput,
    destination: Path,
    *,
    private_root: Path | None = None,
    warnings: Sequence[str],
    spinner: str,
    success: str,
    failure: str,
    fetch: Callable[[], bool],
) -> bool:
    """Make ``destination``, warn, fetch into it, and remove empty output."""
    for message in warnings:
        output.warning(message)

    created = [path for path in (destination, *destination.parents) if not path.exists()]
    root = private_root or destination
    root.mkdir(parents=True, exist_ok=True)
    private_mkdir(destination, root)

    written = False
    try:
        written = bool(run_step(output, spinner, fetch, success=success, failure=failure))
    finally:
        if not written:
            # Keep partial output, but remove directories this call left empty.
            for path in created:
                with suppress(OSError):
                    path.rmdir()
    return written


def download_to(
    output: RichOutput,
    hash_value: str,
    output_dir: Path,
    *,
    validate: Callable[[str], str | None],
    fetch: Callable[[str, Path], bool],
) -> None:
    """Validate a hash, warn, and fetch one live sample into ``output_dir``."""
    sample = validate(hash_value)
    if sample is None:
        return

    output_path = output_dir / f"{sample}.malware"
    warnings = [
        f"WARNING: about to write live malware to {output_path}. Handle with extreme caution!"
    ]
    if output_path.exists():
        warnings.append(f"{output_path} already exists and will be replaced")

    fetch_into(
        output,
        output_dir,
        warnings=warnings,
        spinner=f"Downloading sample {sample[:16]}...",
        success=f"Sample downloaded to {output_path}",
        failure="Failed to download sample",
        fetch=lambda: fetch(sample, output_path),
    )
