"""The suite must not be able to reach the network, whatever a test stubs.

Two agents working on this repo made real requests to a live appliance
and to data.reversinglabs.com from inside pytest, because the isolation
was per-service: A1000 went through a session that the fixtures replaced,
TitaniumCloud built an SDK handle per call and went through nothing.
"""

from __future__ import annotations

import pytest
import requests


class TestNoTestCanReachTheNetwork:
    def test_a_direct_request_is_refused(self):
        with pytest.raises(AssertionError, match="reached the network"):
            requests.get("https://data.reversinglabs.com/api/databrowser/", timeout=1)

    def test_the_refusal_names_the_destination(self):
        with pytest.raises(AssertionError, match=r"appliance\.invalid"):
            requests.post("https://appliance.invalid/api/samples/", timeout=1)

    def test_a_session_built_by_an_sdk_client_is_refused_too(self):
        """The SDK uses its own requests.Session, not the module functions."""
        session = requests.Session()
        with pytest.raises(AssertionError, match="reached the network"):
            session.get("https://data.reversinglabs.com/", timeout=1)
