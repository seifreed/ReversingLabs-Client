"""Tests for the TOON encoder against the reference spec examples."""

from rl_cli.render.toon import encode


class TestScalars:
    def test_primitives(self):
        assert encode(None) == "null"
        assert encode(True) == "true"
        assert encode(False) == "false"
        assert encode(42) == "42"
        assert encode(1.5) == "1.5"
        assert encode("hello") == "hello"

    def test_non_finite_floats_become_null(self):
        assert encode(float("nan")) == "null"
        assert encode(float("inf")) == "null"


class TestObjects:
    def test_flat_object(self):
        assert encode({"id": 1, "name": "Ada"}) == "id: 1\nname: Ada"

    def test_nested_object(self):
        assert encode({"user": {"id": 1}}) == "user:\n  id: 1"

    def test_empty_object_value(self):
        assert encode({"meta": {}}) == "meta: {}"


class TestArrays:
    def test_primitive_array_inline(self):
        assert encode({"tags": ["foo", "bar"]}) == "tags[2]: foo,bar"

    def test_empty_array(self):
        assert encode({"tags": []}) == "tags[0]:"

    def test_tabular_array_spec_example(self):
        data = {"items": [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 1}]}
        assert encode(data) == "items[2]{sku,qty}:\n  A1,2\n  B2,1"

    def test_non_uniform_objects_fall_back_to_list(self):
        data = {"items": [{"a": 1}, {"b": 2}]}
        assert encode(data) == "items[2]:\n  - a: 1\n  - b: 2"

    def test_mixed_array(self):
        assert encode({"xs": [1, {"a": 2}]}) == "xs[2]:\n  - 1\n  - a: 2"

    def test_root_array(self):
        assert encode([1, 2, 3]) == "[3]: 1,2,3"

    def test_an_empty_dict_item_is_its_own_marker(self):
        assert encode({"xs": [{}]}) == "xs[1]:\n  - {}"

    def test_a_nested_array_item_is_drawn_as_an_array(self):
        assert encode({"xs": [[1, 2]]}) == "xs[1]:\n  - [2]: 1,2"

    def test_a_multiline_object_item_indents_its_continuation_lines(self):
        """The continuation lines carry the item's depth, not a stripped one."""
        assert (
            encode({"xs": [{"a": 1}, {"a": {"deep": 2}}]})
            == "xs[2]:\n  - a: 1\n  - a:\n      deep: 2"
        )

    def test_a_multiline_array_item_indents_its_continuation_lines(self):
        assert (
            encode({"xs": [[{"a": 1}, {"b": 2}]]}) == "xs[1]:\n  - [2]:\n      - a: 1\n      - b: 2"
        )

    def test_nested_values_disqualify_tabular(self):
        data = {"items": [{"a": {"deep": 1}}, {"a": {"deep": 2}}]}
        out = encode(data)
        assert "{a}" not in out
        assert "deep: 1" in out


class TestQuoting:
    def test_strings_with_structural_chars_are_quoted(self):
        assert encode({"v": "a,b"}) == 'v: "a,b"'
        assert encode({"v": "a:b"}) == 'v: "a:b"'
        assert encode({"v": " padded "}) == 'v: " padded "'
        assert encode({"v": ""}) == 'v: ""'

    def test_lookalike_literals_are_quoted(self):
        assert encode({"v": "true"}) == 'v: "true"'
        assert encode({"v": "42"}) == 'v: "42"'
        assert encode({"v": "-1.5"}) == 'v: "-1.5"'

    def test_plain_words_with_spaces_stay_bare(self):
        assert encode({"v": "hello world"}) == "v: hello world"

    def test_newlines_escaped(self):
        assert encode({"v": "a\nb"}) == 'v: "a\\nb"'

    def test_non_identifier_keys_quoted(self):
        assert encode({"a key": 1}) == '"a key": 1'

    def test_leading_marker_characters_are_quoted(self):
        # A bare "-", "#" or '"' at the start would read as a list item,
        # a comment or an open quote, so the value is quoted.
        assert encode({"v": "-word"}) == 'v: "-word"'
        assert encode({"v": "#tag"}) == 'v: "#tag"'

    def test_a_lone_surrogate_becomes_a_writable_character(self):
        """``.json()`` decodes ``"\\ud800"`` to a lone surrogate.

        It is not encodable UTF-8, so an encoder answering a string that
        carries one hands back one that cannot be written. Both a value and
        a key are neutralised to the replacement character.
        """
        encode({"k\ud800": "v\ud800"}).encode("utf-8")
        assert encode({"v": "a\ud800b"}) == "v: a�b"
        assert encode({"k\ud800": 1}) == '"k�": 1'
