"""One rule, one mechanism: a partial-answer verdict never outlives its call.

Several services hold a public flag saying the answer they just produced
is not the whole one — ``pages_left_unfetched`` on three of them,
``matches_are_partial`` on the fourth. A caller reads that cell after the
call, so a lookup that measured nothing must not hand back the verdict of
the lookup before it: that prints a partial-answer notice over a whole
answer.

These are structural tests rather than behavioural ones on purpose. The
behavioural tests beside them prove the rule holds for the lookups that
exist today; what is checked here is that it will hold for the lookup
somebody adds next year without knowing the mechanism is there, and that
there is one mechanism to find rather than one per subpackage.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import re
import textwrap
from typing import Any
from unittest.mock import MagicMock

import pytest

import rl_cli.services
from rl_cli.services.base import BaseService
from rl_cli.services.partial_answers import cleared_attribute, cleared_per_call

# The services that hold such a flag today. Pinned so that a detector
# which has quietly stopped finding anything fails here rather than
# passing every other test in this file for free.
_SERVICES_WITH_A_PER_CALL_VERDICT = {
    "A1000NetworkService": "pages_left_unfetched",
    "A1000SampleService": "pages_left_unfetched",
    "A1000YaraService": "matches_are_partial",
    "TitaniumCloudNetworkService": "pages_left_unfetched",
    "TitaniumCloudService": "pages_left_unfetched",
}

_SUBPACKAGES = ("rl_cli.services.a1000", "rl_cli.services.titanium_cloud")


def _service_classes() -> list[type]:
    """Every class defined under ``rl_cli.services``, subpackages included."""
    classes: list[type] = []
    for module_info in pkgutil.walk_packages(rl_cli.services.__path__, "rl_cli.services."):
        module = importlib.import_module(module_info.name)
        for _, member in inspect.getmembers(module, inspect.isclass):
            if member.__module__ == module_info.name:
                classes.append(member)
    return classes


def _holds_instance_state(service: type) -> bool:
    """Whether ``service`` is a shape that can carry a verdict between calls at all.

    Two shapes cannot, and both are ruled out by what they are rather than
    by name, so the next one of either is ruled out too. A ``Protocol`` is
    an interface: it has no ``__init__`` to run and no instance to hold a
    cell. A ``NamedTuple`` is a tuple: whatever it was built with it keeps,
    so there is no cell for one call to leave set for the next to read.
    """
    return not getattr(service, "_is_protocol", False) and not issubclass(service, tuple)


def _built(service: type) -> Any:
    """``service.__init__`` run on a blank instance, a stand-in per argument.

    Deliberately unguarded: a service class this cannot build fails the
    tests below by raising here, which is the point. A ``try`` around it
    would turn "we could not read this one" into "this one holds no
    verdict", which is the same green as compliance.
    """
    instance: Any = object.__new__(service)
    # Read off the class rather than through the instance: what the class
    # itself resolves ``__init__`` to is what will run on the object above.
    initialiser = inspect.getattr_static(service, "__init__")
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    required = [
        parameter
        for parameter in list(inspect.signature(initialiser).parameters.values())[1:]
        if parameter.default is inspect.Parameter.empty and parameter.kind not in variadic
    ]
    initialiser(
        instance,
        *(MagicMock() for p in required if p.kind is not inspect.Parameter.KEYWORD_ONLY),
        **{p.name: MagicMock() for p in required if p.kind is inspect.Parameter.KEYWORD_ONLY},
    )
    return instance


def _instance_state(instance: Any) -> dict[str, Any]:
    """Every attribute a built instance actually holds, ``__slots__`` included.

    ``vars()`` alone reads ``__dict__``, and a class that declares
    ``__slots__`` keeps its cells out of it: the verdict is in a slot
    descriptor, so a slotted service read through ``vars()`` looks like a
    service holding nothing at all.
    """
    state = dict(getattr(instance, "__dict__", {}))
    for klass in type(instance).__mro__:
        slots = getattr(klass, "__slots__", ())
        for slot in (slots,) if isinstance(slots, str) else slots:
            if hasattr(instance, slot):
                state[slot] = getattr(instance, slot)
    return state


# The vocabulary of a partial answer. One of the two readings below; it is
# what catches a verdict on a class whose lookups are elsewhere, and it is
# why an ordinary ``dry_run`` flag is not dragged in with them.
_PARTIALITY = re.compile(
    r"(^|_)(partial|incomplete|unfetched|unread|truncated|capped|clipped"
    r"|omitted|dropped|discarded|remaining|left|more)(_|$)"
)


def _attribute_targets(target: ast.expr) -> set[str]:
    """The attribute names an assignment target writes, unpacking included."""
    if isinstance(target, ast.Attribute):
        return {target.attr}
    if isinstance(target, ast.Starred):
        return _attribute_targets(target.value)
    if isinstance(target, ast.Tuple | ast.List):
        return {name for element in target.elts for name in _attribute_targets(element)}
    return set()


def _declared_boolean(service: type) -> set[str]:
    """The attributes ``service`` or a base annotates with a boolean type.

    What tells ``self.pages_left_unfetched: bool | None = None`` — a
    verdict nothing has measured yet — apart from ``self.client: Any =
    None``, which is a connection nobody has opened, and
    ``self.failure: Exception | None = None``, which is why it did not
    open. All three start at ``None``; only the annotation says which of
    them is a boolean fact about the call that just ran.

    Read out of the source, because the annotation on ``self.x`` inside
    ``__init__`` never reaches ``__annotations__`` — it is a statement,
    not a declaration. That an *unannotated* ``= None`` is therefore not
    read as a verdict is the price, and it is affordable here: mypy runs
    over this package with ``disallow_untyped_defs``, and an unannotated
    ``self.x = None`` is inferred ``None`` and refuses the ``self.x =
    True`` that would make it a verdict at all.
    """
    names: set[str] = set()
    for klass in service.__mro__:
        if klass is object:
            continue
        for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(klass)))):
            if isinstance(node, ast.AnnAssign) and "bool" in ast.unparse(node.annotation):
                names |= _attribute_targets(node.target) | (
                    {node.target.id} if isinstance(node.target, ast.Name) else set()
                )
    return names


def _written_outside_the_constructor(service: type) -> set[str]:
    """The attributes ``service``'s own body assigns anywhere but ``__init__``.

    This is what separates a verdict from a configuration flag without
    asking either of them to be spelled a particular way. A verdict is
    *produced* by the service: some lookup measures it and writes it, so
    the class assigns it somewhere other than where it was started. A
    public ``dry_run`` is handed in by the caller and the service only
    ever reads it — one assignment, in ``__init__``, and never again.

    Read off the class's own source, so ``matches, self.matches_are_partial
    = walked`` counts like ``self.pages_left_unfetched = False`` does; a
    ``setattr`` naming a literal counts too. A write from *outside* the
    class does not, which is the gap this reading leaves and the reason it
    is not the only reading here.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(service)))
    written: set[str] = set()
    for definition in ast.walk(tree):
        if not isinstance(definition, ast.FunctionDef) or definition.name == "__init__":
            continue
        for node in ast.walk(definition):
            if isinstance(node, ast.Assign):
                written |= {name for t in node.targets for name in _attribute_targets(t)}
            elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.For):
                written |= _attribute_targets(node.target)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "setattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                written.add(node.args[1].value)
    return written


def _per_call_verdicts(service: type) -> set[str]:
    """The public attributes of a fresh ``service`` that are partial-answer verdicts.

    Three questions, and an attribute is one when it answers the first and
    either of the others:

    1. Does it start at "nothing measured yet" — ``False``, or a
       ``None`` the class annotates as boolean? Read off a *constructed
       instance*, which answers for every spelling of the initialisation
       there is — ``self.x: bool = False``, ``self.a, self.b = False,
       False``, ``bool()``, a reset delegated to a helper — and for what a
       subclass inherits. Slots are read too, so a class with no
       ``__dict__`` is not a class with no verdict.
    2. Does the service *write* it outside its constructor? A verdict is
       produced by a lookup; a configuration flag is handed in and only
       read. This is what stops ``self.dry_run = False`` — an ordinary
       public flag, and not a verdict — being reported as an uncleared
       one, which is what "any public attribute that is ``False``" did.
    3. Does its name say partiality? The reading that catches a verdict
       the class itself never writes, and the one that would catch a
       ``pages_left_unfetched`` on a class whose lookups live in a mixin.

    Two readings for the second half rather than one, because each covers
    the other's blind spot, and because a single reading that passes its
    own non-vacuity test on one spelling is exactly what produced the
    finding this replaces.

    What stays open is stated at the bottom of this file and pinned there:
    a verdict that is neither named like one nor written by its own class,
    and a verdict that starts at something other than ``False`` or
    ``None``.

    Building an instance takes no credentials — these constructors record
    what they were handed and open nothing — so every argument is a
    stand-in.
    """
    if not _holds_instance_state(service):
        return set()
    produced = _written_outside_the_constructor(service)
    boolean = _declared_boolean(service)
    return {
        name
        for name, value in _instance_state(_built(service)).items()
        if not name.startswith("_")
        and (value is False or (value is None and name in boolean))
        and (name in produced or _PARTIALITY.search(name))
    }


def _uncovered(services: list[type]) -> dict[str, set[str]]:
    """The verdicts each service holds that the shared mechanism does not clear."""
    gaps: dict[str, set[str]] = {}
    for service in services:
        missing = _per_call_verdicts(service) - {cleared_attribute(service)}
        if missing:
            gaps[service.__name__] = missing
    return gaps


def _misnamed(services: list[type]) -> dict[str, str]:
    """Services whose decorator names something that is not a verdict they hold.

    The other half of making the coverage explicit. ``cleared_per_call``
    writes ``False`` to whatever name it is handed, so a typo makes a
    fresh attribute on every call, clears nothing, and leaves the real
    verdict standing — while every other test in this file passes,
    because the class is decorated and the decorated name is cleared.
    """
    return {
        service.__name__: named
        for service in services
        if (named := cleared_attribute(service)) is not None
        and named not in _per_call_verdicts(service)
    }


class _Annotated(BaseService):
    """``self.x: bool = False`` — an ``ast.AnnAssign``, and mypy-clean."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.pages_left_unfetched: bool = False


class _Tupled(BaseService):
    """Two cells in one statement, which is one ``Assign`` of a ``Tuple``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.pages_left_unfetched, self.answer_is_whole = False, True


def _starts_false() -> bool:
    return False


class _Computed(BaseService):
    """A call rather than the literal — ``bool()`` is the same shape.

    The value is an ``ast.Call``, so a detector matching ``ast.Constant``
    sees an ``__init__`` that initialises nothing.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.pages_left_unfetched = _starts_false()


class _ResetInAHelper(BaseService):
    """``__init__`` delegates, so its own source assigns nothing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._reset()

    def _reset(self) -> None:
        self.pages_left_unfetched = False


# The four ordinary ways to start a flag at ``False``. Every one of them
# was invisible to the source-reading detector this file used to carry.
_SPELLINGS = (_Annotated, _Tupled, _Computed, _ResetInAHelper)


class _AnOrdinaryConfigFlag(BaseService):
    """A public flag the caller sets and the service only reads.

    Not a verdict about an answer, and the reading that selected any
    public attribute which ``is False`` reported it as one: a contributor
    got a failure about uncleared partial answers over a boolean that has
    nothing to do with partial answers.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.dry_run = False

    def delete_everything(self) -> str:
        return "pretended" if self.dry_run else "deleted"


class _NotMeasuredYet(BaseService):
    """``bool | None``, so "not measured" is not the same as "nothing left".

    A genuine verdict, and invisible to a detector that knows one spelling
    of the starting value.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.pages_left_unfetched: bool | None = None


class _Slotted(BaseService):
    """No ``__dict__`` for the verdict to live in, so ``vars()`` sees nothing."""

    __slots__ = ("pages_left_unfetched",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.pages_left_unfetched = False


class _NamedNothingLikeAVerdict(BaseService):
    """A verdict in words no vocabulary would guess, written by its own lookup.

    Here so the two readings are shown to be two: this one is outside the
    partiality vocabulary entirely and is caught by what writes it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.records_the_appliance_still_holds = False

    def newest_lookup(self) -> list[str]:
        self.records_the_appliance_still_holds = True
        return []


class TestThereIsOneMechanismAndItIsShared:
    """Two mechanisms for one rule teach a convention that is false next door."""

    def test_the_mechanism_belongs_to_neither_subpackage(self) -> None:
        """A1000 and TitaniumCloud must reach the same module for it.

        A mechanism living inside one of the two is a rule a reader of the
        other cannot find, and an import from a peer subpackage's internals
        for whoever does find it.
        """
        home = cleared_per_call.__module__

        assert not home.startswith(_SUBPACKAGES), (
            f"the shared mechanism lives in {home}, which is one subpackage's own"
        )

    def test_both_subpackages_are_covered_by_that_one_mechanism(self) -> None:
        covered = {
            service.__name__: cleared_attribute(service)
            for service in _service_classes()
            if cleared_attribute(service) is not None
        }

        assert covered == _SERVICES_WITH_A_PER_CALL_VERDICT


class TestEveryPerCallVerdictIsCleared:
    """The rule has to hold for the service nobody has written yet."""

    def test_the_services_that_hold_a_verdict_are_the_ones_we_think(self) -> None:
        """Keeps the detector below honest: it must still find all four."""
        found = {
            service.__name__: sorted(_per_call_verdicts(service))
            for service in _service_classes()
            if _per_call_verdicts(service)
        }

        assert found == {
            name: [attribute] for name, attribute in _SERVICES_WITH_A_PER_CALL_VERDICT.items()
        }

    def test_no_service_holds_a_verdict_nothing_clears(self) -> None:
        assert _uncovered(_service_classes()) == {}

    def test_a_service_added_without_the_mechanism_is_caught(self) -> None:
        """The check has teeth: a fifth service that forgets it fails here."""

        class A1000FutureService(BaseService):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.answer_is_partial = False

        assert _uncovered([A1000FutureService]) == {"A1000FutureService": {"answer_is_partial"}}

    @pytest.mark.parametrize(
        "service",
        _SPELLINGS,
        ids=[service.__name__ for service in _SPELLINGS],
    )
    def test_a_service_is_caught_however_it_spells_the_initialisation(self, service: type) -> None:
        """The teeth are not in one spelling of ``= False``.

        The detector this replaced read ``ast.Assign`` of a literal
        ``False`` off the source of ``__init__``, and every class here
        got past it: it found nothing, so it reported nothing to cover.
        The annotated form is the one a typed codebase writes first.
        """
        assert _uncovered([service]) == {service.__name__: {"pages_left_unfetched"}}

    def test_a_subclass_of_a_covered_service_is_not_covered_by_its_parent(self) -> None:
        """It inherits the flag and none of the clearing, which is the gap.

        ``cleared_per_call`` wraps the methods a class defines, so a
        subclass's own lookups clear nothing — while the flag its parent
        initialised is still there for a caller to read. Reading the
        verdict off a built instance is what sees the inherited cell;
        reading ``__init__``'s source saw an empty subclass.
        """

        @cleared_per_call("pages_left_unfetched")
        class Covered(BaseService):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.pages_left_unfetched = False

        class Extended(Covered):
            def newer_lookup(self) -> int:
                return 1

        assert _uncovered([Extended]) == {"Extended": {"pages_left_unfetched"}}

    def test_every_public_entry_point_of_a_covered_service_clears_it(self) -> None:
        """Not only the lookups whose author remembered to route through it.

        The reset is at the entry point of every public method and every
        public property the class defines, so an entry point added later
        is covered by having been written at all.
        """
        unwrapped: dict[str, list[str]] = {}
        for service in _service_classes():
            attribute = cleared_attribute(service)
            if attribute is None:
                continue
            unwrapped[service.__name__] = sorted(
                name
                for name, member in vars(service).items()
                if not name.startswith("_")
                and not isinstance(member, staticmethod | classmethod)
                and cleared_attribute(member) != attribute
            )

        assert unwrapped == {name: [] for name in _SERVICES_WITH_A_PER_CALL_VERDICT}, (
            "a public member of a covered service is not an entry point that clears; "
            "if it is not an entry point at all, say which shape it is and why"
        )

    def test_a_public_property_is_an_entry_point_like_any_other(self) -> None:
        """It is read with an instance in hand, so it answers for this call.

        Wrapping only ``inspect.isfunction`` members left a property
        answering out of the previous call's state — verified: the flag
        stayed ``True`` across it.
        """

        @cleared_per_call("pages_left_unfetched")
        class WithAProperty(BaseService):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.pages_left_unfetched = False

            @property
            def newest_answer(self) -> str:
                return "a fresh answer"

        service = WithAProperty(MagicMock())
        service.pages_left_unfetched = True

        assert service.newest_answer == "a fresh answer"
        assert service.pages_left_unfetched is False
        assert cleared_attribute(vars(WithAProperty)["newest_answer"]) == "pages_left_unfetched"


class TestAVerdictIsToldApartFromAnyFalseBoolean:
    """Public and ``is False`` was one spelling of the concept, not the concept.

    It was wrong in both directions at once. Over-strict: an ordinary
    ``self.dry_run = False`` on a service is a configuration flag, and was
    reported as an uncleared partial-answer verdict — a contributor gets a
    failure about partial answers over a boolean that is not one. Blind:
    a verdict spelled ``bool | None = None`` for "not measured yet" was
    invisible, and so was any verdict on a class with ``__slots__``, which
    keeps its cells out of ``__dict__``.

    So what is asked now is what a verdict *is*: a boolean-shaped cell,
    starting unmeasured, that either the service writes itself — a
    lookup produces a verdict; a caller hands in a flag — or names as
    partiality. Each case below is read twice, once by the reading this
    replaced and once by the one above, so the contrast is run rather
    than asserted.
    """

    def _any_false_public_attribute(self, service: type) -> set[str]:
        """The reading this replaced, written out."""
        return {
            name
            for name, value in vars(_built(service)).items()
            if not name.startswith("_") and value is False
        }

    def test_an_ordinary_config_flag_is_not_reported_as_a_verdict(self) -> None:
        assert self._any_false_public_attribute(_AnOrdinaryConfigFlag) == {"dry_run"}
        assert _per_call_verdicts(_AnOrdinaryConfigFlag) == set()
        assert _uncovered([_AnOrdinaryConfigFlag]) == {}

    def test_a_verdict_that_starts_at_not_measured_is_reported(self) -> None:
        assert self._any_false_public_attribute(_NotMeasuredYet) == set()
        assert _uncovered([_NotMeasuredYet]) == {"_NotMeasuredYet": {"pages_left_unfetched"}}

    def test_a_verdict_kept_in_a_slot_is_reported(self) -> None:
        assert self._any_false_public_attribute(_Slotted) == set()
        assert _uncovered([_Slotted]) == {"_Slotted": {"pages_left_unfetched"}}

    def test_a_verdict_named_nothing_like_one_is_reported_by_what_writes_it(self) -> None:
        """The vocabulary is one of the two readings, and this name is outside it."""
        assert not _PARTIALITY.search("records_the_appliance_still_holds")
        assert _uncovered([_NamedNothingLikeAVerdict]) == {
            "_NamedNothingLikeAVerdict": {"records_the_appliance_still_holds"}
        }

    def test_a_verdict_its_class_never_writes_is_reported_by_its_name(self) -> None:
        """The other reading, alone: ``_NotMeasuredYet`` has no lookup at all."""
        assert _written_outside_the_constructor(_NotMeasuredYet) == set()
        assert _per_call_verdicts(_NotMeasuredYet) == {"pages_left_unfetched"}

    def test_the_named_attribute_is_one_the_service_holds(self) -> None:
        """What the decorator declares and what the detector finds are one thing."""
        assert _misnamed(_service_classes()) == {}

    def test_a_decorator_naming_an_attribute_that_is_not_there_is_caught(self) -> None:
        """The teeth for that: a typo clears a cell nobody reads, in silence."""

        @cleared_per_call("pages_lefft_unfetched")
        class Typo(BaseService):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.pages_left_unfetched = False

        assert _misnamed([Typo]) == {"Typo": "pages_lefft_unfetched"}
        assert _uncovered([Typo]) == {"Typo": {"pages_left_unfetched"}}


class TestTheGapsThatStayOpenAreNamed:
    """The shapes the mechanism and its detector do not reach, pinned, not assumed."""

    def test_a_verdict_only_something_outside_the_class_writes_is_not_seen(self) -> None:
        """Both readings miss it, and the miss is a decision rather than an oversight.

        The name reading is a vocabulary and this name is outside it; the
        write reading is over the class's own source and this write is in
        a module function. Nothing structural is left to read, so what
        closes it is the declaration: ``@cleared_per_call`` names the
        attribute, and :func:`_misnamed` then holds the declaration to
        the same standard.
        """

        class _WrittenFromOutside(BaseService):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.records_the_appliance_still_holds = False

        def measure(service: _WrittenFromOutside) -> None:
            service.records_the_appliance_still_holds = True

        service = _WrittenFromOutside(MagicMock())
        measure(service)

        assert service.records_the_appliance_still_holds is True, "it really is a verdict"
        assert _uncovered([_WrittenFromOutside]) == {}

    def test_a_verdict_that_starts_at_a_count_is_not_seen(self) -> None:
        """``= 0`` is the same fact spelled as a number, and out of scope here.

        Deliberately, rather than by omission: the mechanism clears to
        ``False``, so a counting verdict is not something it could cover
        even once found. Widening the detector to reach it would report a
        shape nothing can fix.
        """

        class _CountsWhatIsLeft(BaseService):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.pages_left_unfetched = 0

        assert _uncovered([_CountsWhatIsLeft]) == {}

    def test_the_unreachable_members_of_a_covered_service_are_the_ones_we_think(self) -> None:
        """A ``staticmethod`` gets no instance, so it has no verdict to clear.

        It also cannot set one, which is why it is left alone rather than
        refused. Pinned by name so that the shape is a decision each time
        rather than a category anything can be dropped into.
        """
        unreachable = {
            service.__name__: sorted(
                name
                for name, member in vars(service).items()
                if not name.startswith("_") and isinstance(member, staticmethod | classmethod)
            )
            for service in _service_classes()
            if cleared_attribute(service) is not None
        }

        assert unreachable == {
            "A1000NetworkService": [],
            "A1000SampleService": ["build_search_query"],
            "A1000YaraService": [],
            "TitaniumCloudNetworkService": [],
            "TitaniumCloudService": [],
        }

    def test_the_inherited_public_surface_of_a_covered_service_is_the_one_we_think(self) -> None:
        """Inherited methods are deliberately not wrapped; which ones is not.

        ``connect`` and ``test_connection`` say nothing about any answer,
        and clearing on one would let a liveness probe between a lookup
        and its caller's read erase a verdict that was true — that is why
        the mechanism only wraps what a class defines. The cost is that a
        *mixin* supplying a public lookup would be inherited too, and so
        left unwrapped, answering out of the previous call's state.
        Nothing can tell those two apart from the shape alone, so the
        surface is pinned here: a mixin appearing under a covered service
        fails this and has to be ruled on.
        """
        inherited = {
            service.__name__: sorted(
                name
                for name in dir(service)
                if not name.startswith("_") and name not in vars(service)
            )
            for service in _service_classes()
            if cleared_attribute(service) is not None
        }

        connection = ["client", "connect", "connection_summary", "disconnect", "handle_error"]
        assert inherited == {
            "A1000NetworkService": [*connection, "test_connection"],
            "A1000SampleService": [*connection, "test_connection"],
            "A1000YaraService": [*connection, "test_connection"],
            "TitaniumCloudNetworkService": ["handle_error"],
            "TitaniumCloudService": ["handle_error"],
        }
