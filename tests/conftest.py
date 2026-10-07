"""Shared pytest configuration and fixtures."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _bypass_llm_model_probe():
    """Patch the optimizer's ``_probe_model`` to always succeed.

    ``ProjectConfig`` live-probes every search-space LLM missing from LiteLLM's
    static catalog, so tests built on placeholder model ids would otherwise hit
    the network. Mirrors the optimizer's own tests/conftest.py. Tests that need
    probe-failure behaviour re-patch ``agentic_autorag.config.models._probe_model``
    inside the test body.
    """
    with patch("agentic_autorag.config.models._probe_model", return_value=(True, None)):
        yield
