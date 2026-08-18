"""User tags, classifications and containers attached to a sample."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rl_cli.services.a1000.service import (
    A1000_SHORT_HASH_TYPES,
    A1000Service,
    list_from_envelope,
)
from rl_cli.services.decorators import with_client
from rl_cli.services.http import json_on, succeeded

# Where a custom classification is stored. The two are separate records,
# so every call that names one has to be told which.
_CLASSIFICATION_SYSTEMS = ("local", "ticloud")


class A1000MetadataService(A1000Service):
    """Service for the metadata an analyst attaches to a known sample."""

    def _valid_system(self, system: str) -> bool:
        """Guard the classification store, for setting and deleting alike.

        Both paths check, or the same typo is refused on one and passed to
        the SDK unexamined on its twin.
        """
        if system in _CLASSIFICATION_SYSTEMS:
            return True
        self.output.error(f"Invalid system '{system}'. Expected local or ticloud.")
        return False

    # ---------- User tags ----------
    @with_client(default=False)
    def add_user_tags(self, hash_value: str, tags: list[str]) -> bool:
        # The tag endpoints take no SHA512, and the guard judges a stripped,
        # lowercased hash — so it is the returned value that goes to the SDK.
        sample = self._supported_hash(hash_value, A1000_SHORT_HASH_TYPES)
        if sample is None:
            return False
        return succeeded(self.client.post_user_tags(sample, tags))

    @with_client(default=None)
    def get_user_tags(self, hash_value: str) -> list[str] | None:
        """The sample's user tags, or ``None`` when the answer was unreadable.

        The endpoint answers a bare list, and some appliances wrap it in
        ``{"tags": [...]}``. Anything else is a failed read, not an empty
        one: ``remove_all_user_tags`` resolves "every tag" against this, so
        an unreadable list reported as no tags removes nothing and says it
        succeeded.

        The key is ``required`` rather than read as an empty page precisely
        because a bare list is the documented answer: a body that is an
        object at all is already unexpected.
        """
        sample = self._supported_hash(hash_value, A1000_SHORT_HASH_TYPES)
        if sample is None:
            return None
        result = json_on(self.client.get_user_tags(sample))
        if result is None:
            return None
        if isinstance(result, list):
            return result
        tags = list_from_envelope(result, "tags", required=True)
        if tags is None:
            self.output.error("User-tag response carried no tag list")
        return tags

    @with_client(default=False)
    def remove_user_tags(self, hash_value: str, tags: list[str]) -> bool:
        """Remove ``tags`` from the sample.

        The endpoint has no "remove everything" mode and the SDK rejects a
        missing tag list outright, so removing every tag is the caller's
        job: read them with :meth:`get_user_tags` and pass those.
        """
        sample = self._supported_hash(hash_value, A1000_SHORT_HASH_TYPES)
        if sample is None:
            return False
        return succeeded(self.client.delete_user_tags(sample, tags))

    def remove_all_user_tags(
        self, hash_value: str, confirm: Callable[[list[str]], bool] | None = None
    ) -> list[str] | None:
        """Remove every user tag the sample carries; return the tags removed.

        "All" is not something the endpoint can be asked for, so it has to
        be resolved against the tags the sample actually has. That
        read-modify-write lives here rather than in a command, so every
        caller means the same thing by it.

        ``confirm`` is asked before anything goes and is handed exactly the
        tags that would go, so the caller can prompt without doing the
        lookup itself. Declining removes nothing.

        Answers the removed tags, ``[]`` when there was nothing to remove
        or the caller declined, and ``None`` when the sample's tags could
        not be read or the removal failed.
        """
        current = self.get_user_tags(hash_value)
        if current is None:
            return None
        if not current:
            self.output.warning("No user tags to remove")
            return []
        if confirm is not None and not confirm(current):
            return []
        return current if self.remove_user_tags(hash_value, current) else None

    # ---------- Classification ----------
    @with_client(default=False)
    def set_classification(
        self,
        hash_value: str,
        classification: str,
        threat_name: str | None = None,
        system: str = "local",
    ) -> bool:
        """Set a custom classification on ``hash_value``.

        ``system`` selects where the classification is stored: ``local``
        (this appliance) or ``ticloud``.
        """
        if not self._valid_system(system):
            return False
        sample = self._supported_hash(hash_value)
        if sample is None:
            return False
        response = self.client.set_classification(
            sample, classification, system, threat_name=threat_name
        )
        return succeeded(response)

    @with_client(default=None)
    def get_classification(self, hash_value: str) -> dict[str, Any] | None:
        sample = self._supported_hash(hash_value)
        if sample is None:
            return None
        return json_on(self.client.get_classification_v3(sample))

    @with_client(default=False)
    def delete_classification(self, hash_value: str, system: str = "local") -> bool:
        """Delete the custom classification held by ``local`` or ``ticloud``.

        The two are separate records — deleting the local one leaves a
        TitaniumCloud classification standing — so which one is meant has
        to be said, exactly as it is when setting one.
        """
        if not self._valid_system(system):
            return False
        sample = self._supported_hash(hash_value)
        if sample is None:
            return False
        return succeeded(self.client.delete_classification(sample, system=system))

    # ---------- Containers ----------
    @with_client(default=None)
    def list_containers(self, hash_value: str) -> list[dict[str, Any]] | None:
        """The containers the sample was extracted from, or ``None`` if unreadable.

        An empty body is an empty answer, not a failure: this is a bulk
        endpoint, and "if a requested hash doesn't have a container, it will
        not be included in the response".

        Alone among the A1000 listings this one's envelope has no
        corroboration anywhere in the vendored SDK, so ``results`` is
        ``required`` rather than read as an empty page: a body that spells
        it otherwise is far more likely to mean we are reading the wrong key
        than that the sample has no parents.

        Being a bulk endpoint that omits the hashes it has nothing for, it
        is at least as likely to key its answer by hash as to flatten it, so
        a ``results`` object is read as that map. We ask about one hash, so
        every entry in it is about that hash however the appliance spells
        the key.
        """
        sample = self._supported_hash(hash_value, A1000_SHORT_HASH_TYPES)
        if sample is None:
            return None
        data = json_on(self.client.list_containers_for_hashes([sample]))
        if data is None:
            return None
        per_hash = data.get("results") if isinstance(data, dict) else None
        if isinstance(per_hash, dict):
            return self._containers_per_hash(per_hash)
        results = list_from_envelope(data, "results", required=True)
        if results is None:
            self.output.error("Container response carried no results list")
        return results

    def _containers_per_hash(self, results: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Flatten a hash-keyed container map, or ``None`` after saying why."""
        containers: list[dict[str, Any]] = []
        for entry in results.values():
            if isinstance(entry, dict):
                containers.append(entry)
            elif isinstance(entry, list):
                containers.extend(record for record in entry if isinstance(record, dict))
            else:
                self.output.error("Container response carried no results list")
                return None
        return containers
