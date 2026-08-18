"""What every ReversingLabs service is, independent of what it talks to."""

from __future__ import annotations

from rl_cli.config import Settings
from rl_cli.services.errors import error_advice
from rl_cli.services.protocols import Notifier, NullNotifier


class BaseService:
    """What every ReversingLabs service has: its configuration and its reporting.

    Deliberately concrete, and deliberately without a connection lifecycle:
    only the A1000 side has one, and it belongs to
    :class:`~rl_cli.services.a1000.session.A1000Session`.
    """

    def __init__(self, settings: Settings, output: Notifier | None = None):
        self.settings = settings
        self.output: Notifier = output or NullNotifier()

    def handle_error(self, error: Exception, context: str = "") -> None:
        """Report a failed call in terms the user can do something about.

        Every service reports through here. The failures an analyst actually
        hits are named by :func:`~rl_cli.services.errors.error_advice`, which
        can name the profile, the config file and the appliance; the
        original message is kept in parentheses rather than replaced, and
        the internal locator (our class, method and a repr of the argument)
        is printed only when we have nothing better to say.
        """
        detail = str(error) or type(error).__name__
        advice = error_advice(error, self.settings)
        if advice:
            self.output.error(f"{advice} ({detail})")
        else:
            self.output.error(f"{detail} ({context})" if context else detail)
