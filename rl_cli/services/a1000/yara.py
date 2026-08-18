"""YARA rulesets, retro scans, online-source repositories and update jobs."""

from __future__ import annotations

from typing import Any

from rl_cli.models.validators import normalize_ruleset_name
from rl_cli.models.yara_repository import YaraRepositorySpec
from rl_cli.services.a1000.service import (
    A1000_SHORT_HASH_TYPES,
    A1000Service,
    list_from_envelope,
)
from rl_cli.services.a1000.session import A1000Session
from rl_cli.services.decorators import with_client
from rl_cli.services.http import json_on, succeeded
from rl_cli.services.partial_answers import cleared_per_call
from rl_cli.services.protocols import Notifier

# The Yara Online Source API takes the update mode as an integer code; the
# user-facing names live here so no caller has to know the wire values.
_IMPORT_UPDATE_PREFERENCES = {"manual": 0, "auto-update": 1, "auto-import": 2}

# How many pages of matches one walk may fetch, so that an appliance
# answering "there is a next page" forever cannot walk the CLI off the end
# of the world. 200 pages of the 500-record page every other A1000 listing
# serves is 100000 records, the ceiling ``_shared_inputs.MAX_LIMIT`` puts
# on the largest listing this CLI will fetch.
#
# It bounds requests rather than records, because this endpoint takes the
# page size the appliance chose for page one and not one this walk sets.
# What the number has to cover is a ruleset with matches past it: a walk
# that stops short of a filtered hash cannot say the ruleset has no match
# for it, and reports a failed lookup instead.
_MAX_MATCH_PAGES = 200

# The same bound for the repository listing, whose ``--all`` passes no
# cap at all. 200 pages of 100 is 20000 repositories -- far past any real
# appliance, and 200 requests rather than an unbounded number.
_MAX_REPOSITORY_PAGES = 200


def _match_entries(page: list[Any]) -> list[dict[str, Any]]:
    """The records of one page of YARA matches, dicts only."""
    return [entry for entry in page if isinstance(entry, dict)]


@cleared_per_call("matches_are_partial")
class A1000YaraService(A1000Service):
    """Service for everything YARA on the appliance."""

    def __init__(self, session: A1000Session, output: Notifier | None = None):
        super().__init__(session, output)
        # Whether the last match walk gave up before the appliance ran out
        # of pages, for a caller that has to know the list it is holding is
        # not the whole answer: a fact about the answer, with no advice in
        # it. It speaks for the call that just finished and for no earlier
        # one, which is why the class decorator clears it at every entry
        # point and :meth:`get_yara_matches` is the only thing that sets it.
        self.matches_are_partial = False

    def _supported_ruleset(self, name: str) -> str | None:
        """The ruleset name to send, or ``None`` after saying why.

        The value goes into a URL the SDK builds by string formatting, so
        an unchecked name is not a lookup that fails, it is a different
        request that succeeds. See
        :func:`~rl_cli.models.validators.normalize_ruleset_name` for what a
        name can reshape.

        Reported with ``!r`` rather than bare: a name that failed this
        check is exactly the kind that carries a newline, a tab or a bidi
        override, and echoing one unescaped makes the error line lie about
        what was typed.
        """
        normalised = normalize_ruleset_name(name)
        if normalised is None:
            self.output.error(
                f"Invalid YARA ruleset name: {name!r} (expected 3-48 characters, "
                "each a letter, a digit, an underscore or a hyphen)"
            )
        return normalised

    def _import_update_preference(self, mode: str) -> int | None:
        """The A1000's preference code for ``mode``, or ``None`` after saying why.

        Bad input is reported and refused, as every other validator on
        these services does it. Do not raise instead: ``with_client`` is
        the path for a failure we did not anticipate.
        """
        code = _IMPORT_UPDATE_PREFERENCES.get(mode)
        if code is None:
            self.output.error(
                f"Invalid update mode '{mode}'. Expected one of: "
                f"{', '.join(_IMPORT_UPDATE_PREFERENCES)}."
            )
        return code

    # ---------- YARA rulesets on the appliance ----------
    @with_client(default=None)
    def list_yara_rulesets(self) -> list[dict[str, Any]] | None:
        """Every ruleset on the appliance, not only the caller's own.

        The endpoint's ``owner_type`` defaults to ``my``, which shows a
        token whose user owns nothing an empty appliance, while
        ``yara-content`` and ``yara-matches`` still work on the rulesets it
        could not see. Hence ``all``. Older appliances answer a bare list
        rather than the paging envelope, so both shapes are accepted.
        """
        data = json_on(self.client.get_yara_rulesets_on_the_appliance_v2(owner_type="all"))
        if isinstance(data, list):
            return data
        return list_from_envelope(data, "results")

    @with_client(default=False)
    def create_yara_ruleset(
        self,
        name: str,
        content: str,
        publish: bool | None = None,
        ticloud: bool | None = None,
    ) -> bool:
        """Create or update a YARA ruleset.

        The SDK uses a single endpoint that creates the ruleset if it does
        not exist and creates a new revision otherwise. ``publish`` controls
        synchronization to other appliances in the same C1000 cluster;
        ``ticloud`` controls synchronization with TitaniumCloud.

        Guarded even though this one sends the name in a POST body, where
        it reshapes nothing: a name the appliance will not store is a name
        every sibling command then cannot ask about, and every method here
        guards so that none of them is the one that does not.
        """
        ruleset = self._supported_ruleset(name)
        if ruleset is None:
            return False
        # Refused here because the SDK drops it instead: its payload builder
        # assembles the body under `if content:`
        # (ReversingLabs/SDK/a1000.py, __create_post_payload), so empty
        # content is omitted from the POST entirely, the appliance is sent a
        # name with no rules, and the 2xx that comes back reads as a ruleset
        # this command stored.
        if not content.strip():
            self.output.error(f"YARA ruleset '{ruleset}' has no rules to store")
            return False
        response = self.client.create_or_update_yara_ruleset(
            ruleset, content, publish=publish, ticloud=ticloud
        )
        return succeeded(response)

    @with_client(default=False)
    def toggle_yara_ruleset(self, name: str, enable: bool) -> bool:
        ruleset = self._supported_ruleset(name)
        if ruleset is None:
            return False
        # SDK signature is (enabled, name) — not (name, enabled).
        return succeeded(self.client.enable_or_disable_yara_ruleset(enabled=enable, name=ruleset))

    def _unreadable_matches(self, ruleset: str, page: int | None = None) -> None:
        """Name the page whose envelope we could not read.

        Which page it was is the whole point: a walk that broke on page
        seven is otherwise indistinguishable from one that broke on page
        one, and both from a ruleset with no matches.
        """
        where = f"page {page} of matches" if page else "match response"
        self.output.error(f"YARA {where} for '{ruleset}' carried no results list")

    def _walk_matches(self, name: str) -> tuple[list[dict[str, Any]], bool] | None:
        """Every match page for ``name`` up to :data:`_MAX_MATCH_PAGES`, and whether more remain.

        ``None`` is a page whose envelope could not be read, which is
        reported here and is not a ruleset without matches.

        The cursor is read straight off the body, here and in
        :meth:`list_yara_repositories_aggregated`: a ``list_from_envelope``
        that answered is what proves the body is a dict, so the two walks
        must keep reading ``next`` only after it has.
        """
        result = json_on(self.client.get_yara_ruleset_matches_v2(name))
        records = list_from_envelope(result, "results")
        if records is None:
            self._unreadable_matches(name)
            return None

        matches = _match_entries(records)
        # The first page's size is the server's choice, so the following
        # pages must be asked for in that same size or the slices will not
        # line up — and it must be what the server sent, not what survived
        # ``_match_entries``, or one malformed record on page one shifts
        # every page boundary after it.
        page, page_size = 1, len(records)
        while page_size and result.get("next") and page < _MAX_MATCH_PAGES:
            page += 1
            result = json_on(
                self.client.get_yara_ruleset_matches_v2(name, page=page, page_size=page_size)
            )
            records = list_from_envelope(result, "results")
            if records is None:
                self._unreadable_matches(name, page)
                return None
            # An empty page is not the end of the walk — only a missing
            # ``next`` is — so this must not break out on one.
            matches.extend(_match_entries(records))

        if not result.get("next"):
            return matches, False
        self.output.warning(
            f"Stopped after {page} page(s) of matches for '{name}'; the appliance "
            "says there are more, so this list is partial."
        )
        return matches, True

    @with_client(default=None)
    def get_yara_matches(
        self, ruleset: str, hash_value: str | None = None
    ) -> list[dict[str, Any]] | None:
        """Matching samples for a ruleset, optionally one sample only.

        The appliance wraps the matches in a ``results`` envelope, which is
        unwrapped here so callers get the list they asked for.

        The endpoint has no server-side per-sample filter — its second
        positional parameter is ``page``, not a hash — so ``hash_value`` is
        applied here. Being ours, that filter needs its own guard, and it
        reads ``sha256``, ``sha1`` and ``md5`` off each record, so a SHA512
        could never match one and is refused rather than answered "no".

        The filter is only as truthful as the set it runs over: the
        endpoint documents an unpaged answer but pages anyway once a
        ruleset has a few hundred matches, so its envelope is followed to
        the end before answering.

        When the walk gives up at :data:`_MAX_MATCH_PAGES` with the
        appliance still promising more, an empty answer is not "no match",
        it is "we did not look at all of it", and is answered as the failed
        lookup it is. A truncated list with matches in it is handed back
        having warned that it is partial, and :attr:`matches_are_partial`
        records that for anything holding the service afterwards.
        """
        name = self._supported_ruleset(ruleset)
        if name is None:
            return None

        wanted = None
        if hash_value:
            wanted = self._supported_hash(hash_value, A1000_SHORT_HASH_TYPES)
            if wanted is None:
                return None

        walked = self._walk_matches(name)
        if walked is None:
            return None
        matches, self.matches_are_partial = walked

        if wanted is not None:
            matches = [
                entry
                for entry in matches
                if wanted
                in {str(entry.get(key) or "").lower() for key in ("sha256", "sha1", "md5")}
            ]

        if self.matches_are_partial and not matches:
            self.output.error(
                f"Cannot say whether ruleset '{name}' has no matches: the walk stopped "
                f"after {_MAX_MATCH_PAGES} page(s) with more still to read."
            )
            return None
        return matches

    @with_client(default=False)
    def delete_yara_ruleset(self, name: str) -> bool:
        ruleset = self._supported_ruleset(name)
        if ruleset is None:
            return False
        return succeeded(self.client.delete_yara_ruleset(ruleset))

    @with_client(default=None)
    def get_yara_content(self, name: str) -> str | None:
        # The SDK formats this straight into ``?name=...``, unquoted, so
        # ``prod#ignored`` would fetch ``prod`` while the CLI reported on
        # ``prod#ignored``.
        ruleset = self._supported_ruleset(name)
        if ruleset is None:
            return None
        response = self.client.get_yara_ruleset_contents(ruleset)
        succeeded(response)
        return response.text

    # ---------- YARA retro scan ----------
    # Both retro endpoints are action POSTs that document no response body
    # — a 200 with ``{}``, or a 204 — so the status is the answer and
    # neither may be read as JSON.
    @with_client(default=False)
    def yara_cloud_retro_scan(self, ruleset: str, operation: str) -> bool:
        """Start, stop, or clear a YARA Cloud Retro scan for the given ruleset.

        The name is an unquoted path segment of
        ``/api/yara/ruleset/{ruleset_name}/cloud-retro-hunt/``, so an
        unguarded ``../../../api/samples/v2/list/details`` would POST to
        the sample listing endpoint rather than to a retro hunt.
        """
        name = self._supported_ruleset(ruleset)
        if name is None:
            return False
        op = operation.upper()
        if op not in ("START", "STOP", "CLEAR"):
            self.output.error(f"Invalid operation '{operation}'. Expected START, STOP, or CLEAR.")
            return False
        return succeeded(self.client.start_or_stop_yara_cloud_retro_scan(op, name))

    @with_client(default=False)
    def yara_local_retro_scan(self, operation: str) -> bool:
        """Start or stop the YARA Local Retro scan on the appliance."""
        op = operation.upper()
        if op not in ("START", "STOP"):
            self.output.error(f"Invalid operation '{operation}'. Expected START or STOP.")
            return False
        return succeeded(self.client.start_or_stop_yara_local_retro_scan(op))

    @with_client(default=None)
    def get_yara_cloud_retro_scan_status(self, ruleset_name: str) -> dict[str, Any] | None:
        name = self._supported_ruleset(ruleset_name)
        if name is None:
            return None
        return json_on(self.client.get_yara_cloud_retro_scan_status(name))

    @with_client(default=None)
    def get_yara_local_retro_scan_status(self) -> dict[str, Any] | None:
        return json_on(self.client.get_yara_local_retro_scan_status())

    # ---------- YARA Online Source repositories ----------
    @with_client(default=None)
    def list_yara_repositories(
        self,
        active_filter: str | None = None,
        page_size: int | None = None,
        page_number: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """List one page of Yara Online Source repositories.

        ``active_filter`` accepts ``"all"``, ``"user"`` or ``"system"``.
        The paging envelope is unwrapped, as everywhere else here, so that
        no repositories reads as none rather than as a truthy dict.
        """
        if page_size is not None and page_size < 1:
            self.output.error("Invalid repository page size: expected a positive integer")
            return None
        if page_number is not None and page_number < 1:
            self.output.error("Invalid repository page: expected a positive integer")
            return None
        data = json_on(
            self.client.get_yara_repositories(
                active_filter=active_filter,
                page_size=page_size,
                page_number=page_number,
            )
        )
        return list_from_envelope(data, "results")

    @with_client(default=None)
    def list_yara_repositories_aggregated(
        self,
        active_filter: str | None = None,
        page_size: int = 100,
        max_results: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """Every Yara Online Source repository, within a budget.

        Three things stop this walk: the appliance's own answer carried no
        ``next``; the caller's ``max_results``; and
        :data:`_MAX_REPOSITORY_PAGES`, which is the walk's own. A page that
        carried nothing is not one of them — only a missing ``next`` says
        where the listing ends, and stopping on an empty page hands back
        the repositories fetched so far as though they were all of them.

        Walked here rather than handed to the SDK's
        ``get_yara_repositories_aggregated`` (SDK a1000.py:1811-1843), which
        exits only on a page saying it is the last and so cannot stop
        against an appliance that promises another forever.

        The page number is counted here and never read out of the envelope,
        so an appliance that promises another page without moving its
        cursor cannot hold the walk on one page.
        """
        if page_size < 1:
            self.output.error("Invalid repository page size: expected a positive integer")
            return None
        collected: list[dict[str, Any]] = []
        for page_number in range(1, _MAX_REPOSITORY_PAGES + 1):
            body = json_on(
                self.client.get_yara_repositories(
                    active_filter=active_filter,
                    page_size=page_size,
                    page_number=page_number,
                )
            )
            entries = list_from_envelope(body, "results")
            if entries is None:
                self.output.error("Repository response carried no results list")
                return None
            collected.extend(entries)
            if max_results is not None and len(collected) >= max_results:
                return collected[:max_results]
            if not body.get("next"):
                return collected
        self.output.warning(
            f"Stopped after {_MAX_REPOSITORY_PAGES} pages; "
            "the appliance holds more repositories than this."
        )
        return collected[:max_results] if max_results is not None else collected

    @with_client(default=None)
    def create_yara_repository(self, spec: YaraRepositorySpec) -> dict[str, Any] | None:
        """Create a Yara Online Source repository.

        The five fields arrive as one value rather than as five string
        parameters, because all five are same-typed: transposed — the name
        into the URL slot, the token into the branch — they type-check and
        reach the appliance.

        ``spec.import_update_preferences`` is one of ``manual``,
        ``auto-update`` or ``auto-import``; the A1000's integer codes are
        an implementation detail of this method.

        A ``source_branch`` of ``None`` is sent as ``""``, which is how the
        endpoint is told to fall back to the repository's own ``main`` or
        ``master``. The SDK rejects ``None`` outright, and naming ``main``
        here would defeat the fallback.
        """
        preference = self._import_update_preference(spec.import_update_preferences)
        if preference is None:
            return None
        return json_on(
            self.client.create_yara_repository(
                repository_url=spec.repository_url,
                name=spec.name,
                source_branch=spec.source_branch or "",
                api_token=spec.api_token,
                import_update_preferences=preference,
            ),
        )

    @with_client(default=None)
    def update_yara_repository(
        self, repository_id: int, spec: YaraRepositorySpec
    ) -> dict[str, Any] | None:
        """Replace a Yara Online Source repository.

        Takes the same spec as :meth:`create_yara_repository`, for the same
        reason and with the same field names. ``repository_id`` says which
        repository is being replaced rather than being one of the fields
        replaced, so it is not part of the spec.

        The endpoint is a full-resource PUT: every one of the five fields
        is written, so anything not carried by ``spec`` is not left alone,
        it is overwritten. There is no partial update, and the repository
        listing does not hand back the stored ``api_token``, so a value
        the caller does not supply cannot be filled in from the appliance.
        """
        preference = self._import_update_preference(spec.import_update_preferences)
        if preference is None:
            return None
        return json_on(
            self.client.update_yara_repository(
                repository_id=repository_id,
                repository_url=spec.repository_url,
                name=spec.name,
                # Same fallback as the create above: the SDK raises
                # WrongInputError on None, and "" is how the endpoint is
                # told to keep using the repository's own main or master.
                source_branch=spec.source_branch or "",
                api_token=spec.api_token,
                import_update_preferences=preference,
            )
        )

    @with_client(default=False)
    def delete_yara_repository(self, repository_id: int, remove_rulesets: bool = False) -> bool:
        """Delete a custom Yara Online Source repository."""
        response = self.client.delete_yara_repository(
            repository_id=repository_id, remove_rulesets=remove_rulesets
        )
        return succeeded(response)

    # ---------- YARA publish + update job ----------
    @with_client(default=False)
    def publish_yara_ruleset(self, ruleset_name: str) -> bool:
        # Another unquoted path segment:
        # ``/api/yara/publish/ruleset/{ruleset_name}/``.
        name = self._supported_ruleset(ruleset_name)
        if name is None:
            return False
        return succeeded(self.client.publish_single_yara_ruleset(name))

    @with_client(default=False)
    def publish_all_yara_rulesets(self) -> bool:
        return succeeded(self.client.publish_all_yara_rulesets())

    @with_client(default=False)
    def run_yara_update(self) -> bool:
        return succeeded(self.client.run_yara_update())

    @with_client(default=False)
    def set_yara_update_interval(self, seconds: int) -> bool:
        """Configure the YARA update job interval in seconds.

        Setting ``seconds`` to ``0`` disables the auto-update job (this
        introduces a short maintenance downtime per the SDK).
        """
        if not isinstance(seconds, int) or seconds < 0:
            self.output.error(
                f"Invalid interval '{seconds}'. Expected non-negative integer seconds."
            )
            return False
        return succeeded(self.client.set_yara_update_interval(seconds))

    @with_client(default=False)
    def reset_yara_update_interval(self) -> bool:
        return succeeded(self.client.reset_yara_update_interval())
