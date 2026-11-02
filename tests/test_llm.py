"""Tests for app.chatbot.llm with a stubbed OpenAI client (no network)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock, patch

import httpx
import openai
import pytest

from app.chatbot.llm import (
    OLLAMA_DEFAULT_BASE_URL,
    LazyLLMClient,
    LLMConfigurationError,
    LLMError,
    OpenAICompatibleClient,
    build_llm_client,
)
from app.config import Settings
from tests.conftest import FakeLLM


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


@pytest.mark.parametrize("content", [None, "", "   ", "\n\t"])
def test_empty_reply_raises_llm_error(content: str | None) -> None:
    client = OpenAICompatibleClient(model="m", client=StubOpenAI(content=content))
    with pytest.raises(LLMError, match="empty"):
        client.complete("s", "u")


def test_sdk_errors_are_wrapped() -> None:
    error = openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
    client = OpenAICompatibleClient(model="m", client=StubOpenAI(error=error))
    with pytest.raises(LLMError, match="APIConnectionError"):
        client.complete("s", "u")


@pytest.mark.parametrize("key", ["", "   ", "your-api-key", "changeme", "sk-..."])
def test_build_requires_real_api_key_for_openai(key: str) -> None:
    with pytest.raises(LLMConfigurationError, match="LLM_API_KEY"):
        build_llm_client(_settings(llm_api_key=key))


def test_build_openai_with_key_and_ollama_defaults() -> None:
    openai_client = build_llm_client(_settings(llm_api_key="sk-test", llm_model="gpt-x"))
    assert isinstance(openai_client, OpenAICompatibleClient)
    assert openai_client.model == "gpt-x"

    ollama_client = build_llm_client(_settings(llm_provider="ollama", llm_model="llama3.2"))
    assert isinstance(ollama_client, OpenAICompatibleClient)
    assert str(ollama_client._client.base_url).rstrip("/") == OLLAMA_DEFAULT_BASE_URL


def test_default_settings_use_local_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without environment overrides the app must not select a remote LLM."""
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "llama3.2:3b"
    assert settings.llm_api_key == ""
    with patch("app.chatbot.llm.openai.OpenAI") as constructor:
        build_llm_client(settings)
    assert constructor.call_args.kwargs["base_url"] == OLLAMA_DEFAULT_BASE_URL
    assert constructor.call_args.kwargs["api_key"] == "ollama"


def test_ollama_completion_uses_local_chat_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real SDK's request/response path against an offline transport."""
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "local-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "llama3.2:3b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "  LLM_OK  "},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        with openai.OpenAI(
            api_key="ollama", base_url=OLLAMA_DEFAULT_BASE_URL, http_client=http_client
        ) as sdk:
            monkeypatch.setattr("app.chatbot.llm.openai.OpenAI", lambda **kwargs: sdk)
            client = build_llm_client(
                _settings(llm_provider="ollama", llm_model="llama3.2:3b")
            )
            assert client.complete("system", "question") == "LLM_OK"

    assert len(requests) == 1
    assert str(requests[0].url) == "http://localhost:11434/v1/chat/completions"
    assert json.loads(requests[0].content) == {
        "model": "llama3.2:3b",
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
    }


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_build_forwards_custom_provider_settings(provider: str) -> None:
    """Keep the configured endpoint, credentials, timeout and generation settings."""
    settings = _settings(
        llm_provider=provider,
        llm_model="custom-model",
        llm_api_key="  test-key  ",
        llm_base_url="http://localhost:9000/v1",
        llm_temperature=0.3,
        llm_timeout_seconds=7.0,
    )
    with patch("app.chatbot.llm.openai.OpenAI") as constructor:
        client = build_llm_client(settings)
        assert isinstance(client, OpenAICompatibleClient)
        client.complete("system", "question")

    constructor.assert_called_once_with(
        api_key="test-key", base_url="http://localhost:9000/v1", timeout=7.0, max_retries=1
    )
    constructor.return_value.chat.completions.create.assert_called_once_with(
        model="custom-model",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
    )


def test_lazy_client_builds_only_on_first_completion_and_reuses_client() -> None:
    """SQL-only paths need no model; subsequent model calls reuse one client."""
    settings = _settings()
    fake = FakeLLM(reply="Model answer.")
    factory = Mock(return_value=fake)
    client = LazyLLMClient(settings, factory=factory)
    factory.assert_not_called()

    assert client.complete("system", "first question") == "Model answer."
    assert client.complete("system", "second question") == "Model answer."

    factory.assert_called_once_with(settings)
    assert fake.calls == [("system", "first question"), ("system", "second question")]


def test_lazy_client_retries_initialization_after_configuration_error() -> None:
    """A failed initialization must not cache an unusable client."""
    fake = FakeLLM(reply="Recovered.")
    factory = Mock(side_effect=[LLMConfigurationError("Missing key"), fake])
    client = LazyLLMClient(factory=factory)

    with pytest.raises(LLMConfigurationError, match="Missing key"):
        client.complete("system", "question")
    assert client.complete("system", "question") == "Recovered."
    assert factory.call_count == 2


def test_lazy_client_propagates_completion_error() -> None:
    """Do not hide provider failures behind a success-shaped response."""
    factory = Mock(return_value=FakeLLM(error=LLMError("Provider unavailable")))
    client = LazyLLMClient(factory=factory)

    with pytest.raises(LLMError, match="Provider unavailable"):
        client.complete("system", "question")
    factory.assert_called_once_with(None)


@pytest.mark.parametrize(
    ("status_code", "error_name"),
    [(401, "AuthenticationError"), (429, "RateLimitError"), (500, "InternalServerError")],
)
def test_sdk_http_errors_are_wrapped_without_exposing_provider_details(
    status_code: int, error_name: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Exercise real SDK status handling without contacting a provider."""
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"error": {"message": "sensitive-provider-detail", "type": "test_error"}},
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        with openai.OpenAI(
            api_key="test-key", http_client=http_client, max_retries=0
        ) as sdk:
            client = OpenAICompatibleClient(model="test-model", client=sdk)
            with pytest.raises(LLMError, match=error_name) as caught:
                client.complete("system", "question")

    assert isinstance(caught.value.__cause__, openai.APIStatusError)
    assert "sensitive-provider-detail" not in str(caught.value)
    assert "sensitive-provider-detail" not in caplog.text


@pytest.mark.parametrize("choices", [[], [{"message": {"role": "assistant", "content": None}}]])
def test_sdk_response_without_text_raises_llm_error(choices: list[dict[str, Any]]) -> None:
    """Reject missing choices and null content from OpenAI-compatible servers."""
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": choices,
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        with openai.OpenAI(api_key="test-key", http_client=http_client) as sdk:
            client = OpenAICompatibleClient(model="test-model", client=sdk)
            with pytest.raises(LLMError, match="empty"):
                client.complete("system", "question")


def test_sdk_timeout_is_wrapped() -> None:
    """Check that a timed-out provider request becomes the application's LLMError."""
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("test timeout", request=request)

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        with openai.OpenAI(
            api_key="test-key", http_client=http_client, max_retries=0
        ) as sdk:
            client = OpenAICompatibleClient(model="test-model", client=sdk)
            with pytest.raises(LLMError, match="APITimeoutError"):
                client.complete("system", "question")
