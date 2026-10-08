"""Test-wide safety net: the test suite can never make a live (paid) API call.

For every test:
  - Anthropic credentials are removed from the environment,
  - loading a local .env file is disabled (so a real key there is never picked up),
  - outbound network connections are blocked.

Tests that exercise the Claude client use an in-memory httpx MockTransport,
which needs no network, so they are unaffected.
"""

import socket

import pytest


class LiveNetworkBlocked(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def no_live_api(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("app.llm_client._load_env", lambda: None)

    def blocked(*args, **kwargs):
        raise LiveNetworkBlocked("Network access is disabled in tests (no live API calls).")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    yield
