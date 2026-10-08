"""The Claude client, exercised through the real Anthropic SDK with a mocked HTTP layer.

No network and no API key: an httpx MockTransport answers instead of the API.
This checks the exact request we send and how every response type is handled.
"""

import json

import pytest

anthropic = pytest.importorskip("anthropic")  # optional dependency: skip if not installed
httpx = pytest.importorskip("httpx")

from app.llm_client import FALLBACK_BETA, ClaudeJsonLLM, LLMError

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
          "required": ["ok"], "additionalProperties": False}


def message(text='{"ok": true}', stop_reason="end_turn", content=None):
    return {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content if content is not None else [{"type": "text", "text": text}],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 30},
    }


def make_llm(status=200, body=None, captured=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        return httpx.Response(status, json=body if body is not None else message())

    client = anthropic.Anthropic(api_key="test-key", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return ClaudeJsonLLM(model="claude-opus-5-5", client=client)


def call(llm):
    return llm.complete_json(system="sys", user="hello", schema=SCHEMA, effort="low", max_tokens=1000)


def test_request_shape():
    captured = []
    result = call(make_llm(captured=captured))
    assert result.data == {"ok": True}
    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.calls) == (120, 30, 1)

    request = captured[0]
    body = json.loads(request.content)
    assert request.url.path == "/v1/messages"
    assert FALLBACK_BETA in request.headers["anthropic-beta"]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"] == {"effort": "low",
                                     "format": {"type": "json_schema", "schema": SCHEMA}}
    assert body["system"] == "sys"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert "thinking" not in body and "temperature" not in body


def test_usage_cost_estimate():
    result = call(make_llm())
    assert result.usage.cost_usd("claude-opus-5-5") == pytest.approx((120 * 4 + 30 * 20) / 1e6)
    assert result.usage.cost_usd("unknown-model") is None


@pytest.mark.parametrize("body, message_part", [
    (message(stop_reason="refusal"), "declined"),
    (message(stop_reason="max_tokens"), "cut off"),
    (message(text="not json"), "not valid JSON"),
    (message(content=[]), "no text"),
])
def test_bad_responses_raise_non_fatal_errors(body, message_part):
    with pytest.raises(LLMError, match=message_part) as err:
        call(make_llm(body=body))
    assert not err.value.fatal


@pytest.mark.parametrize("status, fatal, message_part", [
    (401, True, "API key was rejected"),
    (403, True, "does not have access"),
    (404, True, "not found"),
    (429, True, "rate limit"),
    (400, False, "rejected the request"),
    (500, False, "returned an error"),
])
def test_http_errors_are_user_friendly(status, fatal, message_part):
    body = {"type": "error", "error": {"type": "error", "message": "details"}}
    with pytest.raises(LLMError, match=message_part) as err:
        call(make_llm(status=status, body=body))
    assert err.value.fatal is fatal


def test_missing_credentials_is_a_clear_fatal_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr("app.llm_client._load_env", lambda: None)
    with pytest.raises(LLMError, match="No API key") as err:
        ClaudeJsonLLM(model="claude-opus-5-5")
    assert err.value.fatal
