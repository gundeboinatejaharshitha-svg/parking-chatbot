"""LLM abstraction and provider adapters.

The chatbot talks to the model only through ``LLMClient.complete``. Both
supported providers (OpenAI-compatible APIs and Ollama) speak the OpenAI chat
protocol, so a single adapter covers them; the provider setting only changes
the defaults. Tests inject a fake ``LLMClient`` and never hit the network.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import openai

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_OLLAMA_PLACEHOLDER_KEY = "ollama"  # Ollama ignores the key but the SDK requires one
_UNSET_KEY_VALUES = frozenset({"your-api-key", "changeme", "sk-..."})


class LLMError(Exception):
    """Raised when the language model cannot produce a reply."""


class LLMConfigurationError(LLMError):
    """Raised when the LLM settings are incomplete or inconsistent."""


class LLMClient(Protocol):
    """Minimal interface the chatbot needs from a language model."""

    def complete(self, system: str, user: str) -> str:
        """Return the assistant reply for one system + user message pair."""
        ...


class OpenAICompatibleClient:
    """Chat-completions adapter for OpenAI, Ollama and other compatible servers.

    ``client`` is injectable so tests can pass a stub with the same
    ``chat.completions.create`` shape.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: float = 30.0,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._client = client or openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=1
        )

    @property
    def model(self) -> str:
        """Model identifier sent with every request."""
        return self._model

    def complete(self, system: str, user: str) -> str:
        """Call the chat-completions endpoint and return the stripped reply text."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except openai.OpenAIError as exc:
            logger.error("LLM request failed: %s", type(exc).__name__)
            raise LLMError(f"LLM request failed: {type(exc).__name__}") from exc

        choices = getattr(response, "choices", None) or []
        content = choices[0].message.content if choices else None
        if not content or not content.strip():
            raise LLMError("LLM returned an empty reply.")
        return content.strip()


def build_llm_client(settings: Settings | None = None) -> LLMClient:
    """Create the configured LLM client from settings (fails fast on bad config)."""
    settings = settings or get_settings()
    provider = settings.llm_provider
    api_key = settings.llm_api_key.strip()
    base_url = settings.llm_base_url

    if provider == "ollama":
        base_url = base_url or OLLAMA_DEFAULT_BASE_URL
        api_key = api_key or _OLLAMA_PLACEHOLDER_KEY
    elif not api_key or api_key in _UNSET_KEY_VALUES:
        raise LLMConfigurationError(
            "LLM_API_KEY is not set (still the .env.example placeholder?). "
            "Set a real key for LLM_PROVIDER=openai or switch to LLM_PROVIDER=ollama."
        )

    logger.info("LLM provider=%s model=%s base_url=%s", provider, settings.llm_model, base_url)
    return OpenAICompatibleClient(
        model=settings.llm_model,
        api_key=api_key,
        base_url=base_url,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
    )


__all__ = [
    "OLLAMA_DEFAULT_BASE_URL",
    "LLMClient",
    "LLMConfigurationError",
    "LLMError",
    "OpenAICompatibleClient",
    "build_llm_client",
]
