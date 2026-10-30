"""Tests for app.chatbot.llm with a stubbed OpenAI client (no network)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import openai
import pytest

from app.chatbot.llm import (
    OLLAMA_DEFAULT_BASE_URL,
    LLMConfigurationError,
    LLMError,
    OpenAICompatibleClient,
    build_llm_client,
)
from app.config import Settings


@dataclass
class _Message:
    content: str | None


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]


@dataclass
class StubOpenAI:
    """Mimics ``openai.OpenAI().chat.completions.create``."""

    content: str | None = "  The answer.  "
    error: Exception | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        outer = self

        class _Completions:
            def create(self, **kwargs: Any) -> _Response:
                outer.requests.append(kwargs)
                if outer.error is not None:
                    raise outer.error
                return _Response([_Choice(_Message(outer.content))])

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"llm_provider": "openai", "llm_model": "gpt-4o-mini", "llm_api_key": ""}
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_complete_sends_system_and_user_and_strips_reply() -> None:
    stub = StubOpenAI()
    client = OpenAICompatibleClient(model="m", temperature=0.2, client=stub)
    assert client.complete("SYS", "USER") == "The answer."
    request = stub.requests[0]
    assert request["model"] == "m" and request["temperature"] == 0.2
    assert request["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]


def test_empty_reply_raises_llm_error() -> None:
    client = OpenAICompatibleClient(model="m", client=StubOpenAI(content="   "))
    with pytest.raises(LLMError, match="empty"):
        client.complete("s", "u")


def test_sdk_errors_are_wrapped() -> None:
    error = openai.APIConnectionError(request=openai._base_client.httpx.Request("POST", "http://x"))
    client = OpenAICompatibleClient(model="m", client=StubOpenAI(error=error))
    with pytest.raises(LLMError, match="APIConnectionError"):
        client.complete("s", "u")


def test_build_requires_real_api_key_for_openai() -> None:
    with pytest.raises(LLMConfigurationError, match="LLM_API_KEY"):
        build_llm_client(_settings(llm_api_key="   "))
    with pytest.raises(LLMConfigurationError, match="placeholder"):
        build_llm_client(_settings(llm_api_key="your-api-key"))


def test_build_openai_with_key_and_ollama_defaults() -> None:
    openai_client = build_llm_client(_settings(llm_api_key="sk-test", llm_model="gpt-x"))
    assert isinstance(openai_client, OpenAICompatibleClient)
    assert openai_client.model == "gpt-x"

    ollama_client = build_llm_client(_settings(llm_provider="ollama", llm_model="llama3.2"))
    assert isinstance(ollama_client, OpenAICompatibleClient)
    assert str(ollama_client._client.base_url).rstrip("/") == OLLAMA_DEFAULT_BASE_URL
