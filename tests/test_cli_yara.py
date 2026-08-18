"""The a1000 YARA commands: rulesets, retro hunts and Online Source repositories.

One file per seam the commands themselves are split along — see
rl_cli/cli/commands/a1000/yara.py, yara_retro.py and yara_repos.py.
"""

import inspect
import io
import json
from dataclasses import fields
from typing import Any, ClassVar
from unittest.mock import MagicMock

import click
import pytest

from rl_cli.cli.commands.a1000 import a1000
from rl_cli.models.yara_repository import YaraRepositorySpec, redact_api_token
from rl_cli.services.a1000 import A1000YaraService
from tests.cli_support import SHA256, flat, invoke, make_context, stub_client, stub_response


@pytest.fixture
def ctx_obj(tmp_path):
    return make_context(tmp_path)


class TestYaraRepoListCountsRepositories:
    """``--all`` emits repositories, not the pages they arrived in."""

    def _client(self, ctx_obj, *, pages=None, envelope=None) -> MagicMock:
        client = stub_client(ctx_obj)
        if pages is not None:
            client.get_yara_repositories.side_effect = [
                stub_response({"next": "/x" if rest else None, "results": page})
                for rest, page in ((pages[index + 1 :], page) for index, page in enumerate(pages))
            ]
        else:
            client.get_yara_repositories.return_value = stub_response(envelope or {})
        return client

    def test_all_emits_repositories_and_caps_them(self, ctx_obj):
        self._client(ctx_obj, pages=[[{"id": 1}, {"id": 2}], [{"id": 3}, {"id": 4}]])

        result = invoke(a1000, ["yara-repo-list", "--all", "--max-results", "3"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == [{"id": 1}, {"id": 2}, {"id": 3}]

    def test_an_appliance_with_no_repositories_says_so(self, ctx_obj):
        self._client(ctx_obj, envelope={"count": 0, "next": None, "results": []})

        result = invoke(a1000, ["yara-repo-list"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "No YARA repositories configured" in result.output
        assert not ctx_obj.output.status.failed


class TestYaraRepoUpdateReplacesTheWholeRepository:
    """The endpoint is a full-resource PUT, so a default is a silent reset.

    Every field the user did not retype went out with the CLI's default
    underneath a success message: a private repository lost its token, and
    an Auto-Import repository was downgraded to Manual.
    """

    def _record(self, monkeypatch) -> list[tuple[int, YaraRepositorySpec]]:
        sent: list[tuple[int, YaraRepositorySpec]] = []

        def update_yara_repository(service, repository_id, spec):
            sent.append((repository_id, spec))
            return {"id": repository_id}

        monkeypatch.setattr(A1000YaraService, "update_yara_repository", update_yara_repository)
        return sent

    def _replacement_fields(self) -> list[str]:
        """Every replaced field but the token, which the test below withholds."""
        return ["--url", "https://h/r", "--name", "r", "--branch", "dev", "--mode", "manual"]

    def test_a_partial_update_is_refused_rather_than_blanking_the_rest(self, ctx_obj, monkeypatch):
        """Everything but the token is given, so the token is what is missing.

        ``--mode`` is passed for that reason: the token has a second
        source now (``--api-token-stdin``), so click cannot demand the
        flag on its own and the demand is made in the command body — which
        runs only once click is satisfied with the rest of the line.
        """
        sent = self._record(monkeypatch)

        result = invoke(
            a1000,
            ["yara-repo-update", "3", *self._replacement_fields()],
            ctx_obj,
        )

        assert result.exit_code != 0
        assert sent == [], "a partial update reached the appliance and blanked the rest"
        assert "--api-token" in flat(result)

    def test_all_five_fields_go_out_as_given(self, ctx_obj, monkeypatch):
        sent = self._record(monkeypatch)

        result = invoke(
            a1000,
            [
                "yara-repo-update",
                "3",
                "--url",
                "https://h/r",
                "--name",
                "r",
                "--branch",
                "dev",
                "--api-token",
                "ghp_secret",
                "--mode",
                "auto-import",
            ],
            ctx_obj,
        )

        assert result.exit_code == 0, result.output
        assert sent == [
            (
                3,
                YaraRepositorySpec(
                    repository_url="https://h/r",
                    name="r",
                    source_branch="dev",
                    api_token="ghp_secret",
                    import_update_preferences="auto-import",
                ),
            )
        ]

    def test_the_help_says_it_replaces(self, ctx_obj):
        result = invoke(a1000, ["yara-repo-update", "--help"], ctx_obj)

        assert "replacement" in flat(result)


class TestYaraRepoCreateLeavesTheBranchFallbackAlone:
    """`main` for the caller broke every repository whose default is `master`."""

    def test_no_branch_is_passed_as_none(self, ctx_obj, monkeypatch):
        sent: list[YaraRepositorySpec] = []

        def create(self: A1000YaraService, spec: YaraRepositorySpec) -> dict[str, int]:
            sent.append(spec)
            return {"id": 1}

        monkeypatch.setattr(A1000YaraService, "create_yara_repository", create)

        result = invoke(a1000, ["yara-repo-create", "--url", "https://h/r", "--name", "r"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert sent[0].source_branch is None


class TestRepositoryFieldsCannotBeTransposedOnTheWayDown:
    """Five same-typed fields were named by hand in each of two handlers.

    Both commands write the same record, the update one as a full-resource
    PUT, and every field was retyped twice on the way from click to the
    SDK. Two of them transposed — the name into the URL slot, the token
    into the branch — is five string arguments in a different order, which
    no type checker and no endpoint refuses. The pairing is now made by
    the field list of one spec, so this test drives the command line off
    the declared option names too, rather than pairing flags with values
    of its own.

    The spec now reaches the service itself, so there is no longer any
    point on the way down where the five are five arguments: the service
    signature is asserted below, because a service that took them apart
    again would restore the swap while every value assertion here still
    passed.
    """

    REPOSITORY: ClassVar[dict[str, str]] = {
        "repository_url": "https://github.example/rules",
        "name": "analyst-rules",
        "source_branch": "release-2024",
        "api_token": "ghp_privatetoken",
        "import_update_preferences": "auto-update",
    }

    def _client(self, ctx_obj) -> MagicMock:
        client = stub_client(ctx_obj)
        client.create_yara_repository.return_value = stub_response({"id": 7})
        client.update_yara_repository.return_value = stub_response({"id": 7})
        return client

    def _command_line(self, command_name: str) -> list[str]:
        """Every spec field, under the flag the command declares for it."""
        flags = {param.name: param.opts[0] for param in a1000.commands[command_name].params}
        return [arg for field, value in self.REPOSITORY.items() for arg in (flags[field], value)]

    @pytest.mark.parametrize("command_name", ["yara-repo-create", "yara-repo-update"])
    def test_every_spec_field_is_an_option_of_both_commands(self, command_name):
        """The spec is filled from ``ctx.params`` by name, so a rename must fail here."""
        declared = {param.name for param in a1000.commands[command_name].params}

        assert {field.name for field in fields(YaraRepositorySpec)} <= declared

    @pytest.mark.parametrize(
        "wrapper,expected",
        [
            ("create_yara_repository", ["self", "spec"]),
            ("update_yara_repository", ["self", "repository_id", "spec"]),
        ],
    )
    def test_the_service_takes_the_five_fields_as_one_value(self, wrapper, expected):
        """No parameter of either wrapper is a repository field on its own.

        Five same-typed strings side by side in a signature is the swap
        itself; one spec has no order to get wrong. ``repository_id`` says
        which repository is replaced rather than being one of the fields
        replaced, so it stays a parameter of its own.
        """
        signature = inspect.signature(getattr(A1000YaraService, wrapper))

        assert list(signature.parameters) == expected

    def test_create_puts_every_field_in_its_own_slot(self, ctx_obj):
        client = self._client(ctx_obj)

        result = invoke(
            a1000, ["yara-repo-create", *self._command_line("yara-repo-create")], ctx_obj
        )

        assert result.exit_code == 0, result.output
        assert client.create_yara_repository.call_args.kwargs == {
            **self.REPOSITORY,
            # The wire code for auto-update; the names are the CLI's.
            "import_update_preferences": 1,
        }

    def test_update_puts_every_field_in_its_own_slot(self, ctx_obj):
        client = self._client(ctx_obj)

        result = invoke(
            a1000, ["yara-repo-update", "7", *self._command_line("yara-repo-update")], ctx_obj
        )

        assert result.exit_code == 0, result.output
        assert client.update_yara_repository.call_args.kwargs == {
            "repository_id": 7,
            **self.REPOSITORY,
            "import_update_preferences": 1,
        }


class TestARepositoryTokenIsNeverPrintedBack:
    """The token authenticates to a third party — a GitHub PAT, typically.

    Whatever these endpoints answer goes straight to the formatter, so a
    field the appliance echoes back is printed verbatim into a terminal, a
    log, or a ``-o json`` pipeline.
    """

    def test_a_listed_repository_shows_the_mask_and_not_the_token(self, ctx_obj):
        client = stub_client(ctx_obj)
        client.get_yara_repositories.return_value = stub_response(
            {"count": 1, "next": None, "results": [{"id": 1, "api_token": "ghp_secret"}]}
        )

        result = invoke(a1000, ["yara-repo-list"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == [{"id": 1, "api_token": "***"}]
        assert "ghp_secret" not in result.output

    def test_the_created_repository_document_shows_the_mask(self, ctx_obj):
        client = stub_client(ctx_obj)
        client.create_yara_repository.return_value = stub_response(
            {"id": 7, "api_token": "ghp_secret"}
        )

        result = invoke(a1000, ["yara-repo-create", "--url", "https://h/r", "--name", "r"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == {"id": 7, "api_token": "***"}
        assert "ghp_secret" not in result.output

    def test_an_answer_with_nothing_to_mask_is_handed_on_unchanged(self):
        """Records these endpoints answer are not always mappings, or tokened."""
        answer = [{"id": 1}, {"id": 2, "api_token": ""}, "not-a-record"]

        assert redact_api_token(answer) == answer

    def test_the_spec_does_not_print_the_token_it_carries(self):
        """A failed service call is reported with the repr of its first argument."""
        spec = YaraRepositorySpec(
            repository_url="https://h/r",
            name="r",
            source_branch=None,
            api_token="ghp_secret",
            import_update_preferences="manual",
        )

        assert "ghp_secret" not in repr(spec)
        assert "api_token='***'" in repr(spec)


class TestTheRepositoryTokenNeedNotBeAnArgument:
    """A live third-party PAT was only statable as ``--api-token <token>``.

    An argument is the one place this CLI cannot keep a credential: argv
    is readable in ``ps`` by every other user on the machine for the life
    of the process, and the line lands in the shell history file
    afterwards — while the same secret is written 0600 in the config
    file, redacted by `config show`, and masked in the spec's own repr.
    ``--api-token-stdin`` is the way that keeps it off the command line;
    the flag still works, unchanged, for scripts and for the empty token
    that states a public repository.
    """

    TOKEN: ClassVar[str] = "ghp_pipedsecret"
    CREATE: ClassVar[list[str]] = ["yara-repo-create", "--url", "https://h/r", "--name", "r"]
    UPDATE: ClassVar[list[str]] = [
        "yara-repo-update",
        "3",
        "--url",
        "https://h/r",
        "--name",
        "r",
        "--branch",
        "dev",
        "--mode",
        "manual",
    ]

    def _record(self, monkeypatch) -> list[YaraRepositorySpec]:
        sent: list[YaraRepositorySpec] = []

        def create_yara_repository(service, spec):
            sent.append(spec)
            return {"id": 1}

        def update_yara_repository(service, repository_id, spec):
            sent.append(spec)
            return {"id": repository_id}

        monkeypatch.setattr(A1000YaraService, "create_yara_repository", create_yara_repository)
        monkeypatch.setattr(A1000YaraService, "update_yara_repository", update_yara_repository)
        return sent

    @pytest.mark.parametrize("command", ["CREATE", "UPDATE"])
    def test_a_piped_token_reaches_the_appliance(self, ctx_obj, monkeypatch, command):
        sent = self._record(monkeypatch)

        result = invoke(
            a1000,
            [*getattr(self, command), "--api-token-stdin"],
            ctx_obj,
            input=f"{self.TOKEN}\n",
        )

        assert result.exit_code == 0, result.output
        assert [spec.api_token for spec in sent] == [self.TOKEN]

    @pytest.mark.parametrize(
        ("piped", "expected"),
        [
            ("ghp_x\n", "ghp_x"),
            # A token file written on Windows is a file with one line in it.
            ("ghp_x\r\n", "ghp_x"),
            ("ghp_x", "ghp_x"),
            # One ending, not every trailing blank: a `.strip()` that
            # reshaped a credential would be unfindable from its 401.
            (" ghp_x \n", " ghp_x "),
            ("ghp_x\n\n", "ghp_x\n"),
        ],
    )
    def test_only_the_last_line_ending_is_taken_off(self, ctx_obj, monkeypatch, piped, expected):
        sent = self._record(monkeypatch)

        result = invoke(a1000, [*self.CREATE, "--api-token-stdin"], ctx_obj, input=piped)

        assert result.exit_code == 0, result.output
        assert sent[0].api_token == expected

    def test_a_terminal_is_asked_without_echoing_and_off_stdout(self, ctx_obj, monkeypatch):
        """At a terminal the token is neither typed on the command line nor shown on it.

        The prompt goes to stderr because stdout is the command's
        document: `-o json ... > repo.json` must not gain a
        "Repository API token:" line at the top of it.
        """
        asked: list[dict] = []

        class _Terminal(io.StringIO):
            def isatty(self) -> bool:
                return True

        def prompt(text: str, **kwargs: Any) -> str:
            asked.append(kwargs)
            return self.TOKEN

        monkeypatch.setattr(click, "get_text_stream", lambda name: _Terminal())
        monkeypatch.setattr(click, "prompt", prompt)
        sent = self._record(monkeypatch)

        result = invoke(a1000, [*self.CREATE, "--api-token-stdin"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert sent[0].api_token == self.TOKEN
        assert asked == [{"hide_input": True, "err": True}]

    def test_the_two_sources_are_not_a_precedence_puzzle(self, ctx_obj, monkeypatch):
        """Both at once states two secrets; sending either silently is the bug."""
        sent = self._record(monkeypatch)

        result = invoke(
            a1000,
            [*self.CREATE, "--api-token", "ghp_flag", "--api-token-stdin"],
            ctx_obj,
            input=f"{self.TOKEN}\n",
        )

        assert result.exit_code == 2
        assert "not both" in flat(result)
        assert sent == []

    def test_the_flag_still_works_and_no_token_at_all_is_a_public_repository(
        self, ctx_obj, monkeypatch
    ):
        sent = self._record(monkeypatch)

        assert invoke(a1000, [*self.CREATE, "--api-token", "ghp_flag"], ctx_obj).exit_code == 0
        assert invoke(a1000, self.CREATE, ctx_obj).exit_code == 0
        assert [spec.api_token for spec in sent] == ["ghp_flag", ""]

    @pytest.mark.parametrize("command_name", ["yara-repo-create", "yara-repo-update"])
    def test_the_help_offers_the_safe_way_and_carries_no_token(self, ctx_obj, command_name):
        result = invoke(a1000, [command_name, "--help"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "--api-token-stdin" in flat(result)
        assert "ghp_" not in result.output

    @pytest.mark.parametrize("source", ["flag", "stdin"])
    def test_no_message_this_command_prints_carries_the_token(self, ctx_obj, source):
        """Success line, rendered document and failure line, from either source.

        The failure line is the one that had to be fixed for: a failed
        service call is reported as the SDK's message followed by the repr
        of its first argument, which is the spec holding the token.
        """
        client = stub_client(ctx_obj)
        client.create_yara_repository.return_value = stub_response(
            {"id": 7, "api_token": self.TOKEN}
        )
        given = ["--api-token", self.TOKEN] if source == "flag" else ["--api-token-stdin"]

        created = invoke(a1000, [*self.CREATE, *given], ctx_obj, input=f"{self.TOKEN}\n")

        assert created.exit_code == 0, created.output
        assert json.loads(created.stdout) == {"id": 7, "api_token": "***"}
        assert self.TOKEN not in created.output

        client.create_yara_repository.side_effect = RuntimeError("appliance said no")

        failed = invoke(a1000, [*self.CREATE, *given], ctx_obj, input=f"{self.TOKEN}\n")

        assert "Failed to create YARA repository" in flat(failed)
        assert self.TOKEN not in failed.output


class TestYaraRepoListPagingOptionsAreNotHalfIgnored:
    """Each option was read on one branch only, and ignored in silence on the other."""

    def _stub(self, monkeypatch, repositories):
        monkeypatch.setattr(
            A1000YaraService,
            "list_yara_repositories_aggregated",
            lambda self, **kwargs: (
                repositories[: kwargs["max_results"]]
                if kwargs.get("max_results") is not None
                else repositories
            ),
        )
        monkeypatch.setattr(
            A1000YaraService, "list_yara_repositories", lambda self, **kwargs: repositories
        )

    def test_page_with_all_is_refused(self, ctx_obj, monkeypatch):
        self._stub(monkeypatch, [{"id": 1}])

        result = invoke(a1000, ["yara-repo-list", "--all", "--page", "2"], ctx_obj)

        assert result.exit_code != 0
        assert "not both" in flat(result)

    def test_max_results_without_all_is_refused(self, ctx_obj, monkeypatch):
        self._stub(monkeypatch, [{"id": 1}, {"id": 2}])

        result = invoke(a1000, ["yara-repo-list", "--max-results", "1"], ctx_obj)

        assert result.exit_code != 0
        assert "--all" in flat(result)

    @pytest.mark.parametrize("budget", ["0", "-5"])
    def test_a_budget_of_nothing_is_refused_before_anything_is_fetched(
        self, ctx_obj, monkeypatch, budget
    ):
        """A cap that caps nothing is a wrong command line, not a walk to make.

        This flag was declared a bare ``int`` here and ``IntRange(min=1)``
        on the ticloud pivots, which is the same option answering the same
        mistake two ways: ``--max-results -5`` was accepted and bounded
        nothing, and the SDK pagers behind both read a falsy budget as
        "every page there is".
        """
        monkeypatch.setattr(
            A1000YaraService,
            "list_yara_repositories_aggregated",
            lambda self, **kwargs: pytest.fail("a budget of nothing reached the appliance"),
        )

        result = invoke(a1000, ["yara-repo-list", "--all", "--max-results", budget], ctx_obj)

        assert result.exit_code == 2
        assert "max-results" in flat(result)


class TestYaraRepoListSaysWhenTheCapCutTheWalkShort:
    """The walk stopped on ``--max-results`` and the listing said nothing.

    So the first N repositories read as all of them — the one truncation in
    this group that named no way to see the rest. It says so in the same
    wording ``list`` and the IP lookups use, naming its own option.
    """

    def _stub(self, monkeypatch, repositories):
        monkeypatch.setattr(
            A1000YaraService,
            "list_yara_repositories_aggregated",
            lambda self, **kwargs: repositories[: kwargs.get("max_results")],
        )

    def test_a_result_the_size_of_the_cap_is_announced_as_partial(self, ctx_obj, monkeypatch):
        self._stub(monkeypatch, [{"id": 1}, {"id": 2}, {"id": 3}])

        result = invoke(a1000, ["yara-repo-list", "--all", "--max-results", "2"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "More results may be waiting; raise --max-results to fetch them." in flat(result)

    def test_a_walk_that_overshot_the_cap_is_partial_too(self, ctx_obj, monkeypatch):
        """The trim to the cap is the SDK's; an SDK that pages past it overshoots."""
        monkeypatch.setattr(
            A1000YaraService,
            "list_yara_repositories_aggregated",
            lambda self, **kwargs: [{"id": 1}, {"id": 2}, {"id": 3}],
        )

        result = invoke(a1000, ["yara-repo-list", "--all", "--max-results", "2"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "More results may be waiting; raise --max-results to fetch them." in flat(result)

    def test_a_result_short_of_the_cap_is_the_whole_set(self, ctx_obj, monkeypatch):
        self._stub(monkeypatch, [{"id": 1}])

        result = invoke(a1000, ["yara-repo-list", "--all", "--max-results", "2"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "More results" not in flat(result)

    def test_an_uncapped_listing_says_nothing_about_a_cap(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(
            A1000YaraService, "list_yara_repositories", lambda self, **kwargs: [{"id": 1}]
        )

        result = invoke(a1000, ["yara-repo-list"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "More results" not in flat(result)


class TestYaraCreateReadsTheRuleFileItWasGiven:
    """Third-party rule files are routinely latin-1, and a path can be a directory."""

    def _record(self, monkeypatch) -> list[str]:
        sent: list[str] = []

        def create_yara_ruleset(service, name, content, **kwargs):
            sent.append(content)
            return True

        monkeypatch.setattr(A1000YaraService, "create_yara_ruleset", create_yara_ruleset)
        return sent

    def test_a_non_utf8_rule_file_still_reaches_the_appliance(self, ctx_obj, monkeypatch, tmp_path):
        rules = tmp_path / "rules.yar"
        rules.write_bytes('rule a { strings: $s = "caf\xe9" condition: $s }'.encode("cp1252"))
        sent = self._record(monkeypatch)

        result = invoke(a1000, ["yara-create", "myrules", str(rules)], ctx_obj)

        assert result.exit_code == 0, result.output
        assert sent and "rule a" in sent[0]

    def test_a_directory_is_rejected_by_the_parser(self, ctx_obj, monkeypatch, tmp_path):
        self._record(monkeypatch)

        result = invoke(a1000, ["yara-create", "myrules", str(tmp_path)], ctx_obj)

        assert result.exit_code == 2
        assert "directory" in flat(result).lower()


class TestYaraContentOfAnEmptyRulesetIsAnAnswer:
    """A ruleset built from an empty file reports 200 and no body."""

    def test_an_empty_ruleset_is_not_a_failure(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(A1000YaraService, "get_yara_content", lambda self, name: "")

        result = invoke(a1000, ["yara-content", "empty-ruleset"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert not ctx_obj.output.status.failed
        assert "has no content" in flat(result)


class TestYaraToggleSaysWhatItFailedToDo:
    def test_the_failure_line_uses_the_infinitive(self, ctx_obj, monkeypatch):
        monkeypatch.setattr(A1000YaraService, "toggle_yara_ruleset", lambda self, n, e: False)

        result = invoke(a1000, ["yara-toggle", "myrules"], ctx_obj)

        assert "Failed to enable YARA ruleset" in flat(result)


class TestRetroScanCommandsBelieveTheStatus:
    """A 2xx with no body is what these endpoints answer, and it means done."""

    def test_cloud_retro_start_with_an_empty_body_is_success(self, ctx_obj):
        client = stub_client(ctx_obj)
        client.start_or_stop_yara_cloud_retro_scan.return_value = stub_response({})

        result = invoke(a1000, ["yara-cloud-retro", "myrules", "-o", "start"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert not ctx_obj.output.status.failed
        assert "Cloud Retro START acknowledged" in flat(result)

    def test_local_retro_start_with_no_content_is_success(self, ctx_obj):
        client = stub_client(ctx_obj)
        response = MagicMock(status_code=204, text="")
        response.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
        client.start_or_stop_yara_local_retro_scan.return_value = response

        result = invoke(a1000, ["yara-local-retro", "-o", "start"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert not ctx_obj.output.status.failed
        assert "Expecting value" not in flat(result)


class TestYaraMatchesAnswerFromEveryPage:
    """`No YARA matches found` off page one was a confident wrong answer."""

    def test_a_hash_on_the_second_page_is_found(self, ctx_obj):
        client = stub_client(ctx_obj)
        client.get_yara_ruleset_matches_v2.side_effect = [
            stub_response({"count": 2, "next": "https://a/p2", "results": [{"sha256": "a" * 64}]}),
            stub_response({"count": 2, "next": None, "results": [{"sha256": SHA256}]}),
        ]

        result = invoke(a1000, ["yara-matches", "myrules", SHA256], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "No YARA matches found" not in flat(result)
        assert json.loads(result.stdout) == [{"sha256": SHA256}]


class TestATruncatedMatchWalkNeverAnswersNo:
    """The walk stops at 100 pages; the per-sample filter runs on this side.

    Together those made "did ruleset X hit sample Y?" answerable with a
    wrong ``no``: the service warned that the list was partial and handed
    back the empty list anyway, and ``empty=`` turned it into a warning
    and exit 0 — the same output a real miss produces.
    """

    def _endless(self, ctx_obj):
        """An appliance that promises another page forever."""
        client = stub_client(ctx_obj)
        client.get_yara_ruleset_matches_v2.return_value = stub_response(
            {"count": 9999, "next": "https://a/next", "results": [{"sha256": "a" * 64}]}
        )
        return client

    def test_a_miss_over_a_truncated_walk_is_not_reported_as_a_miss(self, ctx_obj):
        self._endless(ctx_obj)

        result = invoke(a1000, ["yara-matches", "myrules", SHA256], ctx_obj)

        assert "No YARA matches found" not in flat(result)
        assert "partial" in flat(result)
        assert "Cannot say whether" in flat(result)
        assert ctx_obj.output.status.failed

    def test_a_miss_over_a_complete_walk_still_reads_as_a_miss(self, ctx_obj):
        client = stub_client(ctx_obj)
        client.get_yara_ruleset_matches_v2.return_value = stub_response(
            {"count": 0, "next": None, "results": []}
        )

        result = invoke(a1000, ["yara-matches", "myrules", SHA256], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "No YARA matches found" in flat(result)
        assert not ctx_obj.output.status.failed

    def test_matches_found_before_the_cap_are_still_answered(self, ctx_obj):
        """Only the empty answer is unanswerable; a hit is a hit."""
        self._endless(ctx_obj)

        result = invoke(a1000, ["yara-matches", "myrules", "a" * 64], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "partial" in flat(result)
        assert json.loads(result.stdout)


class TestARulesetNameCannotReshapeTheRequest:
    """The SDK interpolates the name into paths and query strings unquoted.

    ``ReversingLabs/SDK/a1000.py`` formats it into
    ``/api/yara/ruleset/{ruleset_name}/cloud-retro-hunt/`` and builds
    ``name={ruleset_name}`` by concatenation, with no ``quote()`` anywhere
    in the chain — the same defect ``normalize_domain`` was written to
    close for the domain endpoints.
    """

    # The exit status of these runs comes from ``cli.result_callback``,
    # which a group-level invocation never reaches, so what is asserted
    # here is the failure the callback reads: ``output.status.failed``.
    HOSTILE = (
        "../../../api/samples/v2/list/details",
        "prod#ignored",
        "a&name=core",
        "prod?page=2",
        "prod%2fcore",
        "prod core",
        "my\nrules",
    )

    @pytest.mark.parametrize("command", ["yara-content", "yara-matches", "yara-publish"])
    @pytest.mark.parametrize("name", HOSTILE)
    def test_nothing_reaches_the_appliance(self, ctx_obj, command, name):
        client = stub_client(ctx_obj)

        result = invoke(a1000, [command, name], ctx_obj)

        assert ctx_obj.output.status.failed
        assert "Invalid YARA ruleset name" in flat(result)
        assert client.method_calls == [], "a hostile ruleset name reached the SDK"

    @pytest.mark.parametrize("command", ["yara-cloud-retro-status", "yara-cloud-retro"])
    @pytest.mark.parametrize("name", HOSTILE)
    def test_the_retro_commands_refuse_it_too(self, ctx_obj, command, name):
        client = stub_client(ctx_obj)
        operation = ["-o", "start"] if command == "yara-cloud-retro" else []

        result = invoke(a1000, [command, name, *operation], ctx_obj)

        assert ctx_obj.output.status.failed
        assert "Invalid YARA ruleset name" in flat(result)
        assert client.method_calls == []

    def test_the_error_does_not_echo_control_characters_at_the_terminal(self, ctx_obj):
        stub_client(ctx_obj)

        result = invoke(a1000, ["yara-content", "  ‮exe.dcoips\t\n  "], ctx_obj)

        assert "‮" not in result.output
        assert "\\u202e" in result.output

    @pytest.mark.parametrize("name", ["myrules", "MyRules", "core-2024_v3", "abc", "a" * 48])
    def test_a_name_the_appliance_documents_is_still_sent(self, ctx_obj, name):
        client = stub_client(ctx_obj)
        client.get_yara_ruleset_contents.return_value = stub_response("rule x {}")
        client.get_yara_ruleset_contents.return_value.text = "rule x {}"

        result = invoke(a1000, ["yara-content", name], ctx_obj)

        assert result.exit_code == 0, result.output
        client.get_yara_ruleset_contents.assert_called_once_with(name)

    def test_case_survives_because_ruleset_names_are_case_sensitive(self, ctx_obj):
        """Lowercasing one would ask about a ruleset the appliance lacks."""
        client = stub_client(ctx_obj)
        client.publish_single_yara_ruleset.return_value = stub_response({})

        invoke(a1000, ["yara-publish", "MixedCase"], ctx_obj)

        client.publish_single_yara_ruleset.assert_called_once_with("MixedCase")


class TestClearingRetroHistoryIsConfirmed:
    """``clear`` discards the ruleset's retro-hunt history on the appliance.

    It shared one ``--operation`` choice with the harmless ``start`` and
    ``stop`` and so inherited their silence, while ``yara-delete`` next
    door prompts before removing a ruleset that can simply be re-uploaded.
    """

    def test_clear_asks_first_and_a_no_stops_it(self, ctx_obj):
        client = stub_client(ctx_obj)

        result = invoke(a1000, ["yara-cloud-retro", "myrules", "-o", "clear"], ctx_obj, input="n\n")

        client.start_or_stop_yara_cloud_retro_scan.assert_not_called()
        assert result.exit_code == 0, result.output
        # The prompt names the ruleset whose history is about to go.
        assert "myrules" in flat(result)
        assert "Cancelled" in flat(result)

    def test_a_yes_goes_through(self, ctx_obj):
        client = stub_client(ctx_obj)
        client.start_or_stop_yara_cloud_retro_scan.return_value = stub_response({})

        result = invoke(a1000, ["yara-cloud-retro", "myrules", "-o", "clear"], ctx_obj, input="y\n")

        assert result.exit_code == 0, result.output
        client.start_or_stop_yara_cloud_retro_scan.assert_called_once_with("CLEAR", "myrules")

    @pytest.mark.parametrize("operation", ["start", "stop"])
    def test_the_harmless_operations_are_not_made_interactive(self, ctx_obj, operation):
        client = stub_client(ctx_obj)
        client.start_or_stop_yara_cloud_retro_scan.return_value = stub_response({})

        result = invoke(a1000, ["yara-cloud-retro", "myrules", "-o", operation], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "?" not in flat(result)
        client.start_or_stop_yara_cloud_retro_scan.assert_called_once_with(
            operation.upper(), "myrules"
        )


class TestDeletingARepositoryIsConfirmed:
    """Deleting a repository is irreversible, and it prompted for nothing.

    Every other destructive command in the group asks first, including
    ``yara-delete`` for a ruleset that can simply be re-uploaded. This one
    took a repository id and, with ``--remove-rulesets``, every ruleset
    that repository had imported -- which live on the appliance, not in
    the source it synced them from.
    """

    def test_a_no_deletes_nothing(self, ctx_obj):
        client = stub_client(ctx_obj)

        result = invoke(a1000, ["yara-repo-delete", "7"], ctx_obj, input="n\n")

        client.delete_yara_repository.assert_not_called()
        assert result.exit_code == 0, result.output
        assert "Cancelled" in flat(result)

    def test_the_prompt_names_the_repository(self, ctx_obj):
        stub_client(ctx_obj)

        result = invoke(a1000, ["yara-repo-delete", "7"], ctx_obj, input="n\n")

        assert "7" in flat(result)

    def test_the_prompt_says_when_rulesets_go_too(self, ctx_obj):
        """--remove-rulesets widens the blast radius, so it widens the prompt."""
        stub_client(ctx_obj)

        result = invoke(a1000, ["yara-repo-delete", "7", "--remove-rulesets"], ctx_obj, input="n\n")

        assert "ruleset" in flat(result)

    def test_a_yes_goes_through(self, ctx_obj):
        client = stub_client(ctx_obj)
        client.delete_yara_repository.return_value = stub_response({})

        result = invoke(a1000, ["yara-repo-delete", "7"], ctx_obj, input="y\n")

        assert result.exit_code == 0, result.output
        client.delete_yara_repository.assert_called_once()


class TestADestructiveYaraCommandCanStillBeScripted:
    """The prompt is the default, and ``--yes`` is how a script answers it.

    A caller with nothing on stdin cannot answer at all: ``click.confirm``
    reads that as an abort, so adding the prompt made a working
    ``rl-cli a1000 yara-repo-delete 7`` in a cron job exit 1 having deleted
    nothing, with no flag anywhere in the CLI to get past it.
    """

    def test_a_repository_delete_needs_no_terminal(self, ctx_obj):
        client = stub_client(ctx_obj)
        client.delete_yara_repository.return_value = stub_response({})

        result = invoke(a1000, ["yara-repo-delete", "7", "--yes"], ctx_obj, input="")

        assert result.exit_code == 0, result.output
        assert "?" not in flat(result), "the prompt was drawn anyway"
        client.delete_yara_repository.assert_called_once()

    def test_the_same_call_without_the_flag_aborts_and_deletes_nothing(self, ctx_obj):
        """What the finding was: a scripted delete with no answer to give."""
        client = stub_client(ctx_obj)

        result = invoke(a1000, ["yara-repo-delete", "7"], ctx_obj, input="")

        assert result.exit_code == 1
        client.delete_yara_repository.assert_not_called()

    def test_a_ruleset_delete_needs_no_terminal(self, ctx_obj):
        client = stub_client(ctx_obj)
        client.delete_yara_ruleset.return_value = stub_response({})

        result = invoke(a1000, ["yara-delete", "myrules", "--yes"], ctx_obj, input="")

        assert result.exit_code == 0, result.output
        client.delete_yara_ruleset.assert_called_once()

    def test_clearing_retro_history_needs_no_terminal(self, ctx_obj):
        client = stub_client(ctx_obj)
        client.start_or_stop_yara_cloud_retro_scan.return_value = stub_response({})

        result = invoke(a1000, ["yara-cloud-retro", "myrules", "-o", "clear", "--yes"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "?" not in flat(result)
        client.start_or_stop_yara_cloud_retro_scan.assert_called_once_with("CLEAR", "myrules")

    def test_an_interactive_no_still_stops_the_delete(self, ctx_obj):
        """The flag defaults to off: a bare invocation asks, as it did before."""
        client = stub_client(ctx_obj)

        result = invoke(a1000, ["yara-repo-delete", "7"], ctx_obj, input="n\n")

        assert result.exit_code == 0, result.output
        assert "Cancelled" in flat(result)
        client.delete_yara_repository.assert_not_called()


class TestYaraDeleteAsksBeforeItRemoves:
    """A ruleset is re-uploadable, but the delete still confirms first."""

    def test_a_no_stops_the_delete(self, ctx_obj, monkeypatch):
        called: list[str] = []
        monkeypatch.setattr(
            A1000YaraService,
            "delete_yara_ruleset",
            lambda self, name: called.append(name),
        )

        result = invoke(a1000, ["yara-delete", "myrules"], ctx_obj, input="n\n")

        assert result.exit_code == 0, result.output
        assert "Cancelled" in flat(result)
        assert called == []


class TestYaraCreateReportsAnUnreadableFile:
    """The path exists for click, and still cannot be read as text."""

    def test_an_os_error_is_reported_and_nothing_is_sent(self, ctx_obj, monkeypatch, tmp_path):
        rules = tmp_path / "rules.yar"
        rules.write_text("rule a { condition: true }")
        sent: list[str] = []

        def create(service, name, content):
            sent.append(content)
            return True

        monkeypatch.setattr(A1000YaraService, "create_yara_ruleset", create)

        def boom(_path):
            raise OSError("permission denied")

        monkeypatch.setattr("rl_cli.cli.commands.a1000.yara.read_text_lenient", boom)

        result = invoke(a1000, ["yara-create", "myrules", str(rules)], ctx_obj)

        assert result.exit_code == 0, result.output
        assert "Could not read" in flat(result)
        assert sent == []


class TestYaraPublishAllTakesTheOtherBranch:
    """No ruleset name and ``--all`` publishes every non-core ruleset."""

    def test_all_reaches_the_bulk_call(self, ctx_obj, monkeypatch):
        called: list[bool] = []

        def publish_all(service):
            called.append(True)
            return True

        monkeypatch.setattr(A1000YaraService, "publish_all_yara_rulesets", publish_all)

        result = invoke(a1000, ["yara-publish", "--all"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert called == [True]
        assert "all non-core" in flat(result)


class TestYaraUpdateIntervalReset:
    """``--reset`` restores the appliance default instead of setting seconds."""

    def test_reset_reaches_the_reset_call(self, ctx_obj, monkeypatch):
        called: list[bool] = []

        def reset(service):
            called.append(True)
            return True

        monkeypatch.setattr(A1000YaraService, "reset_yara_update_interval", reset)

        result = invoke(a1000, ["yara-update-interval", "--reset"], ctx_obj)

        assert result.exit_code == 0, result.output
        assert called == [True]
        assert "reset to default" in flat(result)
