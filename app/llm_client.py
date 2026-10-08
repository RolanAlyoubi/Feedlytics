"""The only module that talks to the Claude API.

Everything else depends on the small ``JsonLLM`` interface below, so tests can
substitute a fake model and the rest of the app never handles API details.

Credentials come from the environment (``ANTHROPIC_API_KEY`` in ``.env``) and
are never written to disk or logs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app import config

try:  # .env support is optional at import time (e.g. in tests)
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

# Retry declined requests on Anthropic's recommended fallback model (server side).
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Values that mean "no key yet" (e.g. .env copied from .env.example unchanged).
PLACEHOLDER_KEYS = {"", "your-api-key-here", "sk-ant-...", "changeme"}


class LLMError(RuntimeError):
    """A user-facing error from the AI step. Callers fall back to the baseline.

    ``fatal`` errors (bad key, no access, network down) will repeat on every
    call, so the caller should stop calling the API for the rest of the run.
    """

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls

    def cost_usd(self, model: str) -> Optional[float]:
        prices = config.MODEL_PRICES.get(model)
        if not prices:
            return None
        return (self.input_tokens * prices[0] + self.output_tokens * prices[1]) / 1_000_000


@dataclass
class JsonResult:
    data: Dict[str, Any]
    usage: Usage = field(default_factory=Usage)
    model: str = ""


class JsonLLM:
    """Interface: send a prompt, get back JSON that matches ``schema``."""

    model: str = "unknown"

    def complete_json(self, system: str, user: str, schema: Dict[str, Any],
                      effort: str, max_tokens: int) -> JsonResult:  # pragma: no cover
        raise NotImplementedError


def configured_model() -> str:
    _load_env()
    return os.getenv("FEEDLYTICS_MODEL", config.DEFAULT_MODEL)


def has_credentials() -> bool:
    """True if a real API credential is configured (the key itself is never read out).

    Placeholder values from .env.example do not count, so an unedited .env
    behaves exactly like having no key at all.
    """
    _load_env()
    values = (os.getenv("ANTHROPIC_API_KEY"), os.getenv("ANTHROPIC_AUTH_TOKEN"))
    return any(v is not None and v.strip() not in PLACEHOLDER_KEYS for v in values)


def _load_env() -> None:
    if load_dotenv is not None:
        load_dotenv(override=False)


class ClaudeJsonLLM(JsonLLM):
    """Claude via the official SDK, with structured (JSON-schema) output."""

    def __init__(self, model: Optional[str] = None, client: Any = None):
        self.model = model or configured_model()
        if client is None:
            if not has_credentials():
                raise LLMError("No API key found. Add ANTHROPIC_API_KEY to your .env file.", fatal=True)
            try:
                import anthropic  # imported lazily: only the AI path needs the SDK
            except ImportError:
                raise LLMError("The 'anthropic' package is not installed (pip install anthropic). "
                               "It is only needed for the AI path.", fatal=True) from None
            client = anthropic.Anthropic(max_retries=3, timeout=120.0)
        self.client = client

    def complete_json(self, system: str, user: str, schema: Dict[str, Any],
                      effort: str, max_tokens: int) -> JsonResult:
        import anthropic

        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                betas=[FALLBACK_BETA],
                fallbacks="default",
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
        except anthropic.AuthenticationError:
            raise LLMError("The API key was rejected. Check ANTHROPIC_API_KEY in .env.", fatal=True) from None
        except anthropic.PermissionDeniedError:
            raise LLMError("This API key does not have access to the selected model.", fatal=True) from None
        except anthropic.NotFoundError:
            raise LLMError(f"Model '{self.model}' was not found. Check FEEDLYTICS_MODEL.", fatal=True) from None
        except anthropic.RateLimitError:
            raise LLMError("The API rate limit was reached. Wait a minute and try again.", fatal=True) from None
        except anthropic.BadRequestError as exc:
            raise LLMError(f"The API rejected the request: {exc.message}") from None
        except anthropic.APIStatusError as exc:
            raise LLMError(f"The AI service returned an error ({exc.status_code}). Try again later.") from None
        except anthropic.APIConnectionError:
            raise LLMError("Could not reach the AI service. Check your internet connection.", fatal=True) from None

        usage = Usage(
            input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
            calls=1,
        )
        if response.stop_reason == "refusal":
            raise LLMError("The AI model declined this request.")
        if response.stop_reason == "max_tokens":
            raise LLMError("The AI response was cut off (output limit reached).")

        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("The AI response contained no text.")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            raise LLMError("The AI response was not valid JSON.") from None
        return JsonResult(data=data, usage=usage, model=getattr(response, "model", self.model))
