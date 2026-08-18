"""Tests for validation utilities."""

import pytest

from rl_cli.models.validators import (
    HashType,
    normalize_domain,
    normalize_hash,
    normalize_ip_address,
    normalize_ruleset_name,
    validate_hash,
    validate_url,
    validate_url_or_host,
)


class TestHashValidation:
    """Test hash validation functions."""

    def test_valid_md5(self):
        """Test valid MD5 hash."""
        assert validate_hash("d41d8cd98f00b204e9800998ecf8427e") == HashType.MD5

    def test_valid_sha1(self):
        """Test valid SHA1 hash."""
        assert validate_hash("da39a3ee5e6b4b0d3255bfef95601890afd80709") == HashType.SHA1

    def test_valid_sha256(self):
        """Test valid SHA256 hash."""
        assert (
            validate_hash("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
            == HashType.SHA256
        )

    def test_invalid_hash(self):
        """Test invalid hash."""
        assert validate_hash("invalid") is None
        assert validate_hash("12345") is None
        assert validate_hash("g" * 32) is None  # Non-hex characters

    def test_case_insensitive(self):
        """Test case insensitive hash validation."""
        assert validate_hash("D41D8CD98F00B204E9800998ECF8427E") == HashType.MD5

    def test_surrounding_whitespace_is_tolerated(self):
        """Only for the match — callers must send the normalised value, or
        the API sees the raw string this accepted."""
        assert validate_hash(" d41d8cd98f00b204e9800998ecf8427e\n") == HashType.MD5


class TestURLValidation:
    """Test URL validation functions."""

    def test_valid_urls(self):
        """Test valid URLs."""
        assert validate_url("https://example.com") is True
        assert validate_url("http://test.org/path") is True
        assert validate_url("https://sub.domain.com:8080/path?query=1") is True

    def test_invalid_urls(self):
        """Test invalid URLs."""
        assert validate_url("not a url") is False
        assert validate_url("example.com") is False  # Missing scheme
        assert validate_url("") is False
        assert validate_url("ftp://") is False  # Missing netloc


class TestIPValidation:
    """Test IP address validation."""

    def test_valid_ipv4(self):
        """Test valid IPv4 addresses."""
        assert normalize_ip_address("192.168.1.1") is not None
        assert normalize_ip_address("10.0.0.0") is not None
        assert normalize_ip_address("255.255.255.255") is not None
        assert normalize_ip_address("0.0.0.0") is not None

    def test_invalid_ipv4(self):
        """Test invalid IPv4 addresses."""
        assert normalize_ip_address("256.1.1.1") is None
        assert normalize_ip_address("192.168.1") is None
        assert normalize_ip_address("192.168.1.1.1") is None

    def test_valid_ipv6(self):
        """Test valid IPv6 addresses."""
        assert normalize_ip_address("::1") is not None
        assert normalize_ip_address("::") is not None
        assert normalize_ip_address("2001:0db8:85a3:0000:0000:8a2e:0370:7334") is not None

    def test_invalid_ip(self):
        """Test invalid IP addresses."""
        assert normalize_ip_address("not an ip") is None
        assert normalize_ip_address("") is None
        assert normalize_ip_address("example.com") is None


class TestIpv6Forms:
    """The old regex only matched fully expanded addresses, :: and ::1."""

    @pytest.mark.parametrize(
        "address",
        [
            "2001:db8::1",
            "::ffff:192.0.2.1",
            "2001:db8:0:0:1::1",
            "fe80::1",
            "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
            "::1",
            "::",
        ],
    )
    def test_accepts_real_ipv6(self, address):
        assert normalize_ip_address(address) is not None

    @pytest.mark.parametrize(
        "address",
        ["999.1.1.1", "1.2.3", "notanip", "", "2001:db8::1::2", "8.8.8.8.8"],
    )
    def test_still_rejects_nonsense(self, address):
        assert normalize_ip_address(address) is None

    def test_surrounding_whitespace_is_tolerated(self):
        assert normalize_ip_address(" 8.8.8.8 ") == "8.8.8.8"


class TestSha512:
    """13 A1000 endpoints accept SHA512; rejecting it here was stricter than the API."""

    SHA512 = (
        "b42ce5d90737fca3d7fe16f12a656edea4ef28ab2e19319a354d756d49f91019"
        "a42ef7d6a2d5087bcb2ba7decfca94fb80864d2e59debf4ba13c35c63b546d71"
    )

    def test_sha512_is_recognised(self):
        assert validate_hash(self.SHA512) == HashType.SHA512

    def test_sha512_is_case_insensitive(self):
        assert validate_hash(self.SHA512.upper()) == HashType.SHA512

    def test_lengths_between_the_known_ones_are_still_rejected(self):
        assert validate_hash("a" * 63) is None
        assert validate_hash("a" * 100) is None
        assert validate_hash("a" * 127) is None
        assert validate_hash("a" * 129) is None

    def test_non_hex_of_the_right_length_is_rejected(self):
        assert validate_hash("z" * 128) is None


class TestUrlVersusHost:
    """Fetching a URL needs a scheme; looking one up does not."""

    @pytest.mark.parametrize("value", ["https://example.com", "http://a.b/c?d=1"])
    def test_full_urls_pass_both(self, value):
        assert validate_url(value) is True
        assert validate_url_or_host(value) is True

    @pytest.mark.parametrize("value", ["example.com", "www.example.com", "sub.a.co.uk"])
    def test_bare_hosts_pass_only_the_lookup_check(self, value):
        assert validate_url(value) is False
        assert validate_url_or_host(value) is True

    # ``evil .com`` is the one entry here that carries both a dot and an
    # inner space. Without it every value in the list was refused for
    # want of a dot instead, so deleting the whitespace check outright
    # left the suite green — and a typed-in space reached the lookup as a
    # netloc the appliance answers about with an empty report.
    @pytest.mark.parametrize("value", ["not a url", "", "   ", "///bad", "nodot", "evil .com"])
    def test_nonsense_fails_both(self, value):
        assert validate_url(value) is False
        assert validate_url_or_host(value) is False

    @pytest.mark.parametrize("value", ["[", "[::1", "][", "http://[", "a.b[c"])
    def test_a_typo_with_a_bracket_is_rejected_not_raised(self, value):
        """urlparse raises "Invalid IPv6 URL"; the point of this check is to
        catch typos before the appliance sees them, not to crash on one."""
        assert validate_url(value) is False
        assert validate_url_or_host(value) is False


class TestNormalisedValuesAreWhatGetsSent:
    """The guards judged ``value.strip().lower()``; the callers sent ``value``.

    So a hash pasted with a newline passed a check that had never seen
    it and then failed inside the SDK, and an uppercase one reached the
    appliance uppercased.
    """

    MD5 = "d41d8cd98f00b204e9800998ecf8427e"

    def test_a_hash_comes_back_stripped_and_lowercased(self):
        assert normalize_hash(f"  {self.MD5.upper()}\n") == self.MD5

    @pytest.mark.parametrize("value", ["not-a-hash", "", "g" * 32, None, 123])
    def test_what_is_not_a_hash_normalises_to_nothing(self, value):
        assert normalize_hash(value) is None

    def test_an_address_comes_back_stripped(self):
        assert normalize_ip_address(" 8.8.8.8 ") == "8.8.8.8"

    @pytest.mark.parametrize("value", ["999.1.1.1", "example.com", "", None, 123])
    def test_what_is_not_an_address_normalises_to_nothing(self, value):
        assert normalize_ip_address(value) is None


class TestBareDomains:
    """``network_domain_report`` interpolates its argument into a path
    segment without quoting it, so anything with structure in it asks
    the appliance a different question — and gets answered, emptily."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("example.com", "example.com"),
            ("EVIL.COM", "evil.com"),
            (" sub.a.co.uk ", "sub.a.co.uk"),
            ("example.com.", "example.com"),
            ("xn--80ak6aa92e.com", "xn--80ak6aa92e.com"),
            ("my-host.example.com", "my-host.example.com"),
        ],
    )
    def test_a_bare_domain_passes_and_is_lowercased(self, value, expected):
        assert normalize_domain(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "https://evil.com/a/b?q=1",
            "http://evil.com",
            "evil.com:8080",
            "user@evil.com",
            "evil.com/path",
            "evil.com?q=1",
            "evil.com#frag",
            "evil com",
            "nodot",
            "",
            "   ",
            ".com",
            "a..b.com",
            "-evil.com",
            "evil-.com",
            "evil.com/../admin",
            "8.8.8.8",
            None,
            123,
        ],
    )
    def test_anything_with_structure_in_it_is_refused(self, value):
        assert normalize_domain(value) is None

    @pytest.mark.parametrize(
        "value",
        [
            "foo_bar.example.com",
            "_dmarc.example.com",
            "cdn_1.evil_c2.net",
            "host_.example.com",
        ],
    )
    def test_an_underscore_label_is_a_name_the_appliance_can_be_asked_about(self, value):
        """Illegal in the strict hostname grammar, legal in DNS, and used by C2.

        The guard exists to catch a URL typed where a domain belongs, not
        to re-derive RFC 1123: an underscore reshapes nothing about the
        path, so refusing it refused a lookup we could have made.
        """
        assert normalize_domain(value) == value

    @pytest.mark.parametrize("value", ["evil_.com/path", "foo_bar.com:8080", "a_b..c.com"])
    def test_an_underscore_does_not_smuggle_structure_past_the_guard(self, value):
        assert normalize_domain(value) is None

    @pytest.mark.parametrize(
        "value",
        ["a\n.bc", "good.com\n.evil.com", "foo\n.example.com", "evil.com\n"],
    )
    def test_a_newline_at_a_label_boundary_is_refused(self, value):
        """A ``$`` anchor let a per-label match keep a boundary newline.

        The value is interpolated unquoted into the request path, so a
        surviving ``\\n`` is a request-splitting primitive — exactly what
        this guard exists to stop. The per-label match meant the value's
        own ``strip`` never reached a newline sitting before a ``.``.
        """
        result = normalize_domain(value)
        assert result is None or "\n" not in result

    def test_the_lookup_check_is_deliberately_laxer(self):
        """``network_url_report`` quotes its argument and takes either form."""
        assert validate_url_or_host("https://evil.com/a/b?q=1") is True
        assert normalize_domain("https://evil.com/a/b?q=1") is None


class TestAnIdnIsEncodedRatherThanRefused:
    """``_DOMAIN_LABEL`` is ASCII-only, and used to judge the typed name.

    So the punycode spelling passed and the human one did not:
    ``münchen.de`` and ``évil.com`` — the way a domain appears in the
    phishing lure an analyst is pasting from — were answered with
    "Invalid domain: … (expected a bare domain such as example.com)" by
    nine TitaniumCloud lookups and six A1000 ones, while
    ``xn--mnchen-3ya.de`` went through. Only the user who already knew
    the answer could ask the question. Both APIs key on the punycode
    form, so encoding is the guard's job and not the analyst's.
    """

    @pytest.mark.parametrize(
        "typed,expected",
        [
            ("münchen.de", "xn--mnchen-3ya.de"),
            ("évil.com", "xn--vil-9la.com"),
            ("日本.jp", "xn--wgv71a.jp"),
            ("нэ.рф", "xn--m1a6a.xn--p1ai"),
            ("sub.münchen.de", "sub.xn--mnchen-3ya.de"),
            ("MÜNCHEN.de", "xn--mnchen-3ya.de"),
            ("  münchen.de  ", "xn--mnchen-3ya.de"),
            ("münchen.de.", "xn--mnchen-3ya.de"),
        ],
    )
    def test_a_human_spelled_domain_comes_back_as_the_key_the_apis_hold(self, typed, expected):
        assert normalize_domain(typed) == expected

    @pytest.mark.parametrize(
        "encoded", ["xn--mnchen-3ya.de", "xn--80ak6aa92e.com", "xn--e1afmkfd.xn--p1ai"]
    )
    def test_an_already_encoded_domain_is_left_alone(self, encoded):
        """The codec passes ASCII through, so encoding twice is not a change."""
        assert normalize_domain(encoded) == encoded

    @pytest.mark.parametrize("typed", ["münchen.de", "évil.com", "日本.jp", "ünïcödé.example.com"])
    def test_the_round_trip_names_the_domain_that_was_typed(self, typed):
        """The other direction: what is sent decodes back to what was read."""
        normalised = normalize_domain(typed)
        assert normalised is not None
        assert normalised.encode("ascii").decode("idna") == typed

    @pytest.mark.parametrize(
        "typed",
        [
            "münchen.de:8080",
            "https://münchen.de/a?q=1",
            "user@münchen.de",
            "münchen.de/path",
            "münchen..de",
            "ä" * 70 + ".com",
            "ünïcödé",
        ],
    )
    def test_encoding_does_not_smuggle_structure_past_the_guard(self, typed):
        """An empty or over-long label makes the codec raise ``UnicodeError``;
        anything else with structure in it survives encoding and is refused by
        the label check, exactly as its ASCII twin is."""
        assert normalize_domain(typed) is None

    def test_the_253_character_limit_is_measured_on_the_encoded_name(self):
        """Punycode is longer than what was typed, and it is what is sent.

        Six 35-character labels are 218 characters as typed and 254
        encoded, so a limit measured before encoding would have passed a
        name no DNS query can carry.
        """
        typed = ".".join(["ü" * 35] * 6) + ".de"
        assert len(typed) <= 253
        assert len(typed.encode("idna").decode("ascii")) > 253
        assert normalize_domain(typed) is None
        assert normalize_domain(("a" * 63 + ".") * 4 + "com") is None

    def test_the_limit_admits_a_name_of_exactly_253_and_no_more(self):
        """253 is the longest DNS name there is, so the guard is ``>``, not ``>=``."""
        at_limit = ".".join(["a" * 63, "a" * 63, "a" * 63, "a" * 61])
        assert len(at_limit) == 253
        assert normalize_domain(at_limit) == at_limit

        over_limit = ".".join(["a" * 63, "a" * 63, "a" * 63, "a" * 62])
        assert len(over_limit) == 254
        assert normalize_domain(over_limit) is None

    def test_the_ascii_verdicts_the_extraction_settled_are_unchanged(self):
        """The three subjects the two halves used to disagree about."""
        assert normalize_domain("1.2.3.4") is None
        assert normalize_domain("user@example.com") is None
        assert normalize_domain("a..b.com") is None

    def test_a_name_idna_2003_would_fold_keeps_its_own_registration(self):
        """``str.encode("idna")`` is IDNA 2003, and folds what IDNA 2008 keeps.

        ``faß.de`` and ``fass.de`` are two registrations. Under the
        stdlib codec the first was looked up as the second, so the CLI
        reported one domain's verdict under the other's name — the
        confident wrong answer this module exists to prevent, arriving
        through the encoder instead of through the guard.
        """
        assert normalize_domain("faß.de") == "xn--fa-hia.de"
        # 2003 folds GREEK SMALL LETTER FINAL SIGMA to GREEK SMALL LETTER
        # SIGMA and yields ``xn--wxaikc6b.gr``; 2008 keeps the final form,
        # which encodes to a different label entirely.
        assert normalize_domain("σόλος.gr") == "xn--wxaijb9b.gr"

    @pytest.mark.parametrize(
        "domain",
        ["_dmarc.example.com", "cdn_1.evil_c2.net", "foo_bar.example.com"],
        ids=["dmarc", "c2-shape", "underscore"],
    )
    def test_an_ascii_name_never_goes_near_the_encoder(self, domain):
        """``idna`` is strict about IDNA 2008 and refuses an underscore label.

        ``_dmarc.example.com`` is ordinary DNS and ``cdn_1.evil_c2.net``
        is a real C2 shape; ``_DOMAIN_LABEL`` accepts both on purpose.
        Putting every name through the encoder to normalise the handful
        that need it refused the ASCII ones this tool looks up most.
        """
        assert normalize_domain(domain) == domain


class TestAnAddressIsCanonicalisedBeforeItIsSent:
    """One address written two ways was two keys, and ``%`` was a path.

    The A1000 IP lookups interpolate this into a path segment unquoted
    (``/api/network-threat-intel/ip/{ip}/report/``), so an expanded or
    uppercased IPv6 address asked about a host the appliance files under
    its short form — the ``evil.com.`` / ``evil.com`` split that
    ``normalize_domain`` closes, left open for its sibling. And ``%`` is
    the percent-escape introducer, so a scope id reshapes that path the
    way a URL typed into the domain slot does.
    """

    @pytest.mark.parametrize(
        "typed,expected",
        [
            ("2001:DB8:0000:0000:0000:0000:0000:0001", "2001:db8::1"),
            ("2001:0db8:85a3:0000:0000:8a2e:0370:7334", "2001:db8:85a3::8a2e:370:7334"),
            ("2001:db8:0:0:1::1", "2001:db8::1:0:0:1"),
            ("::FFFF:192.0.2.1", "::ffff:192.0.2.1"),
            ("2001:db8::1", "2001:db8::1"),
            ("8.8.8.8", "8.8.8.8"),
        ],
    )
    def test_one_address_normalises_to_one_key(self, typed, expected):
        assert normalize_ip_address(typed) == expected

    @pytest.mark.parametrize(
        "typed,expected",
        [
            ("fe80::1%eth0", "fe80::1"),
            ("fe80::1%1", "fe80::1"),
            (" FE80::1%eth0 ", "fe80::1"),
        ],
    )
    def test_a_scope_id_is_dropped_rather_than_written_into_the_path(self, typed, expected):
        """A scope names an interface on the machine that typed it, which the
        appliance has no way to resolve — and its ``%`` is a path escape."""
        assert normalize_ip_address(typed) == expected

    def test_two_spellings_of_one_address_agree_after_normalising(self):
        expanded = normalize_ip_address("2001:0DB8:0000:0000:0000:0000:0000:0001")
        assert expanded == normalize_ip_address("2001:db8::1")

    @pytest.mark.parametrize("typed", ["999.1.1.1", "2001:db8::1::2", "example.com", "", None, 123])
    def test_what_is_not_an_address_still_normalises_to_nothing(self, typed):
        assert normalize_ip_address(typed) is None


class TestNonStringInput:
    """These are boundary checks: whatever arrives is a value to judge, not
    a string to trust. Raising AttributeError makes the caller handle it."""

    def test_a_non_string_is_not_a_hash(self):
        assert validate_hash(None) is None
        assert validate_hash(123) is None

    def test_a_non_string_is_not_an_address(self):
        assert normalize_ip_address(None) is None
        # An int is a packed address to ipaddress, and 123 is not the
        # address anybody meant by 123.
        assert normalize_ip_address(123) is None

    def test_a_non_string_is_not_a_ruleset_name(self):
        assert normalize_ruleset_name(None) is None
        assert normalize_ruleset_name(3) is None


class TestRulesetNameNormalisation:
    """The A1000's YARA endpoints put this in a URL without quoting it.

    ``/api/yara/ruleset/{ruleset_name}/cloud-retro-hunt/`` and
    ``/api/yara/publish/ruleset/{ruleset_name}/`` are path formats, and
    the content and matches lookups build ``name={ruleset_name}`` by
    concatenation — the same defect ``normalize_domain`` exists to close,
    left open for ruleset names.

    The accepted set is not a guess: the Spectra Intelligence YARA
    hunting API (TCA-0303) publishes ``^[a-z,A-Z,0-9,_-]*$`` at 3 to 48
    characters, and the Spectra Analyze YARA Hunting page repeats the
    length and says names should use only letters, digits and the
    underscore. See ``_RULESET_NAME`` in rl_cli/models/validators.py.
    """

    @pytest.mark.parametrize(
        "name",
        ["myrules", "MixedCase", "core-2024_v3", "abc", "a" * 48, "_leading", "999"],
    )
    def test_a_name_the_appliance_documents_is_accepted_unchanged(self, name):
        assert normalize_ruleset_name(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            # Each of these reshapes the request rather than failing it.
            "../../../api/samples/v2/list/details",
            "prod#ignored",
            "a&name=core",
            "prod?page=2",
            "prod%2fcore",
            "prod=core",
            "prod/core",
            "prod core",
            "my\nrules",
            "my\trules",
            "‮exe.dcoips",
            # And these are simply not names the appliance would store.
            "",
            "   ",
            "ab",
            "a" * 49,
        ],
    )
    def test_anything_else_is_refused(self, name):
        assert normalize_ruleset_name(name) is None

    def test_surrounding_whitespace_is_trimmed_like_a_pasted_hash(self):
        assert normalize_ruleset_name("  myrules\n") == "myrules"

    def test_case_is_left_alone_because_ruleset_names_are_case_sensitive(self):
        """Lowercasing would ask the appliance about a ruleset it lacks."""
        assert normalize_ruleset_name("MyRules") == "MyRules"
