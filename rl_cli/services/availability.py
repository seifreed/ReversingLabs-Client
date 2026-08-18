"""Service availability checker for ReversingLabs APIs.

What a probe run *is* — the statuses, and the shape of the document — is
in :mod:`rl_cli.models.availability`, because the renderers grade against
the same statuses and may not import this package. Those names are
re-exported here, where a probe run comes from.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import cache
from types import UnionType
from typing import Any, TypeGuard, Union, get_args, get_origin, get_type_hints, is_typeddict

from rl_cli.models.availability import Availability, ServiceProbe, ServiceStatus
from rl_cli.services.a1000 import A1000MetadataService, A1000Session
from rl_cli.services.credentials import is_real_credential
from rl_cli.services.protocols import Notifier, NullNotifier
from rl_cli.services.titanium_cloud import TitaniumCloudService
from rl_cli.storage.probe_cache import ProbeCache
from rl_cli.text import sanitize

__all__ = [
    "CACHE_DURATION",
    "APIAvailabilityChecker",
    "Availability",
    "ServiceProbe",
    "ServiceStatus",
]

# Empty-file SHA256 — guaranteed to be a well-formed hash for liveness probes.
_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_DUMMY_SHA1 = "0" * 40


def _probe(
    status: ServiceStatus,
    message: str,
    *,
    credentials: bool = False,
    accessible: bool = False,
) -> ServiceProbe:
    """One finished measurement of one service.

    Every exit path of a probe returns a whole ``ServiceProbe`` built here,
    so that no half-filled probe can reach the cache.
    """
    return {
        "status": status.value,
        "message": message,
        "credentials_configured": credentials,
        "api_accessible": accessible,
    }


@cache
def _required_fields(shape: Any) -> dict[str, Any]:
    """Each key ``shape`` requires, and the type its value must have.

    Read off the annotations rather than restated, so a field added to one
    of these ``TypedDict``\\ s is checked for without an edit here. Memoised
    because the answer is a fact about the class.
    """
    hints = get_type_hints(shape)
    return {name: hints[name] for name in shape.__required_keys__}


def _stated_as(value: object, hint: Any) -> bool:
    """Whether ``value`` is written the way the annotation says it is.

    ``bool`` is a subclass of ``int``, so a count stated as ``true`` would
    otherwise pass for one and be rendered as "Services Available: True/2".

    **Every** annotation a ``TypedDict`` may carry has to be met here, and
    without raising: the guard reads the shapes rather than restating them,
    so it meets whatever the next field is written as, and ``check_all``
    puts no ``try`` around it. Bare ``isinstance`` raises on a
    parameterised generic and on ``Any``, so a parameterised generic is
    checked as the container it names, a union as any of its arms, ``Any``
    as the presence of the key alone, and an annotation ``isinstance`` can
    still make nothing of is left unjudged rather than counted as damage.

    The element types inside a container are not checked: the annotation's
    origin is all this can state about a JSON value without walking it.
    """
    if hint is Any:
        return True
    if is_typeddict(hint):
        return _has_shape(value, hint)
    origin = get_origin(hint)
    if origin is Union or origin is UnionType:
        return any(_stated_as(value, arm) for arm in get_args(hint))
    if hint is int:
        return isinstance(value, int) and not isinstance(value, bool)
    try:
        return isinstance(value, origin or hint)
    except TypeError:
        return True


def _has_shape(value: object, shape: Any) -> bool:
    """Whether ``value`` is an object carrying every key ``shape`` requires, as its type.

    Recursive, so how deep the check goes is read off the annotations too:
    a ``TypedDict`` nested in another is checked to its own full depth.
    """
    if not isinstance(value, dict):
        return False
    return all(
        name in value and _stated_as(value[name], hint)
        for name, hint in _required_fields(shape).items()
    )


def _is_probe_run(document: object) -> TypeGuard[Availability]:
    """Whether a loaded document is a probe run this version can render.

    The cache is keyed on the profile and the credentials, not on this
    shape, so a document written before a release that adds a probed
    service or restates a field outlives that release by up to
    :data:`CACHE_DURATION`. Each required key therefore has to be present
    *and* stated as its annotation says: a value of the wrong type is a
    cache miss exactly like a missing key, and is re-probed rather than
    handed to a renderer mypy certified. Extra keys are welcome.
    """
    return _has_shape(document, Availability)


# How long a probe run stays good for. The policy is the prober's, not
# the cache file's: 24 h is how stale an "is the appliance reachable"
# answer may be, which is a fact about probing an appliance.
CACHE_DURATION = timedelta(hours=24)


class APIAvailabilityChecker:
    """Check availability of ReversingLabs APIs based on credentials.

    Probes go through the services rather than instantiating SDK clients
    directly, so connection logic stays in one place. It takes a session
    rather than the settings to make the A1000 probe unavoidably reuse the
    connection the command then uses: probing a different appliance from
    the one about to be used answers the wrong question, twice as slowly.

    Deliberately the one class here that names two concrete siblings rather
    than taking a protocol: it answers "can *these two* be reached".
    """

    def __init__(
        self, session: A1000Session, output: Notifier | None = None, *, verbose: bool = False
    ):
        self.session = session
        self.settings = session.settings
        self.output: Notifier = output or NullNotifier()
        self.cache = ProbeCache(
            self.settings.cache_dir / "api_availability.json",
            self._cache_identity,
            duration=CACHE_DURATION,
            # Nothing that goes wrong with the cache file is fatal, so a
            # default run stays quiet. ``verbose`` is passed in rather than
            # read off the settings, which must not be mutated by ``-v``:
            # ``config save`` would persist it.
            report=self.output.warning if verbose else None,
        )

    def _cache_identity(self) -> tuple[str | None, ...]:
        """What a cached answer belongs to: this profile and these appliances.

        Availability lives in a single file, so without this the result
        computed for one profile is served to another for the full 24
        hours. The credentials are in it so that editing one invalidates
        the answer; the cache hashes them with its salt and stores neither.
        """
        ticloud = self.settings.titanium_cloud
        a1000 = self.settings.a1000
        return (
            self.settings.profile,
            ticloud.host,
            ticloud.username,
            ticloud.password,
            a1000.host,
            a1000.username,
            a1000.password,
            a1000.token,
        )

    def check_all(self, force: bool = False) -> Availability:
        """Probe both services. Cached for 24 h unless ``force`` is set."""
        if not force:
            cached = self.cache.load()
            # The cache stores and returns a plain JSON document, so this is
            # the only layer that can tell a probe run this version renders
            # from one an older version left behind. A document that cannot
            # be rendered is a cache miss — re-probe and overwrite — rather
            # than an error the analyst has to clear by hand.
            if _is_probe_run(cached):
                return cached

        titanium_cloud = self._check_titanium_cloud()
        a1000 = self._check_a1000()
        probes = (titanium_cloud, a1000)
        availability: Availability = {
            "timestamp": datetime.now().isoformat(),
            "titanium_cloud": titanium_cloud,
            "a1000": a1000,
            "summary": {
                "services_available": sum(
                    probe["status"] == ServiceStatus.AVAILABLE.value for probe in probes
                ),
                "services_total": len(probes),
            },
        }

        self.cache.save(availability)
        return availability

    def clear_cache(self) -> None:
        self.cache.clear()

    # ---------- Per-service probes ----------

    def _check_titanium_cloud(self) -> ServiceProbe:
        tc_settings = self.settings.titanium_cloud
        if not tc_settings.username or not tc_settings.password:
            return _probe(ServiceStatus.UNAVAILABLE, "No TitaniumCloud credentials configured")
        # Whether a credential counts is ``is_real_credential``'s to say, on
        # both services; empty is split out above only to word the message.
        if not is_real_credential(tc_settings.username) or not is_real_credential(
            tc_settings.password
        ):
            return _probe(
                ServiceStatus.UNAVAILABLE,
                "TitaniumCloud credentials not configured (placeholder values)",
            )

        # The service reports transport errors through its notifier, and
        # this one keeps what it is told: the probe is the only place the
        # user hears about it, so a dropped reason cannot be recovered.
        probe = _ProbeOutput()
        service = TitaniumCloudService(self.settings, output=probe)
        if service.get_file_reputation(_EMPTY_SHA256):
            return _probe(
                ServiceStatus.AVAILABLE,
                "TitaniumCloud API is available and working",
                credentials=True,
                accessible=True,
            )
        return _probe(
            ServiceStatus.ERROR,
            f"TitaniumCloud probe failed: {sanitize(probe.reason)}",
            credentials=True,
        )

    def _check_a1000(self) -> ServiceProbe:
        a1000_settings = self.settings.a1000
        # The same rule the session must pick its authentication with: a
        # token that is still the example string is not a token to send.
        has_token = is_real_credential(a1000_settings.token)
        has_creds = is_real_credential(a1000_settings.username) and is_real_credential(
            a1000_settings.password
        )
        if not has_token and not has_creds:
            return _probe(ServiceStatus.UNAVAILABLE, "No A1000 credentials configured")

        auth_type = "token" if has_token else "username/password"

        # The metadata service is the narrowest one carrying both halves of
        # the probe: ``test_connection`` from the shared client, and
        # ``get_classification`` of its own. Both answer a bare bool and
        # report the reason to their notifier, so the reason is recovered
        # from this probe and from the session's memory.
        probe = _ProbeOutput()
        service = self.session.service(A1000MetadataService, probe)
        # A lightweight second ask, for a Check Status call that failed
        # without saying why. The session remembers an appliance that did
        # not answer, so this costs a function call rather than another
        # connect timeout.
        if service.test_connection() or service.get_classification(_DUMMY_SHA1) is not None:
            return _probe(
                ServiceStatus.AVAILABLE,
                f"A1000 API is available ({auth_type} authentication)",
                credentials=True,
                accessible=True,
            )
        return _probe(
            ServiceStatus.ERROR,
            f"A1000 probe failed: {sanitize(self._a1000_failure(probe))}",
            credentials=True,
        )

    def _a1000_failure(self, probe: _ProbeOutput) -> str:
        """Why the A1000 probe failed, in the words of whatever failed.

        Under username/password, constructing the client is a token POST, so
        the session's remembered exception is the whole story. Under token
        authentication nothing is sent until a call is made, so there the
        reason is what the probe's own calls reported.

        The callers must sanitise this before it becomes a message: it is
        written to the cache and replayed ahead of every command for the
        next 24 hours, so an escape sequence in it drives the terminal on
        every invocation rather than once.
        """
        failure = self.session.failure
        if failure is not None:
            return str(failure) or type(failure).__name__
        return probe.reason


class _ProbeOutput(NullNotifier):
    """The silent notifier, but keeping a probe's errors instead of dropping them.

    The availability check runs at every CLI invocation, so SDK error noise
    must not reach the user on every call — but the reason cannot be dropped
    either, since the failed probe is the only thing the user is shown.

    Override only :meth:`error` here: :class:`NullNotifier` is already
    silent down to the spinner that draws nothing and still answers
    ``task_ids[0]``.
    """

    def __init__(self) -> None:
        self.reason = "the probe call returned no data and reported no error"

    def error(self, message: str) -> None:
        self.reason = message
