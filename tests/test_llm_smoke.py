"""Offline tests for the opt-in live LLM smoke check."""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from app.chatbot.llm import LLMConfigurationError, LLMError
from scripts import smoke_test_llm
from tests.conftest import FakeLLM


def test_smoke_check_passes_with_expected_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the check calls the model using only a synthetic prompt."""
    fake = FakeLLM(reply="LLM_OK")
    monkeypatch.setattr(smoke_test_llm, "configure_logging", lambda: None)
    monkeypatch.setattr(smoke_test_llm, "build_llm_client", lambda: fake)

    assert smoke_test_llm.main() == 0
    assert fake.calls == [
        (
            "This is a connectivity check. Reply with exactly LLM_OK and nothing else.",
            "Run the connectivity check.",
        )
    ]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (LLMConfigurationError("sensitive-detail"), "configure a valid LLM_API_KEY"),
        (LLMError("sensitive-detail"), "provider did not return a usable reply"),
    ],
)
def test_smoke_check_reports_failures(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: LLMError,
    message: str,
) -> None:
    """Fail explicitly without printing secrets from an exception."""
    monkeypatch.setattr(smoke_test_llm, "configure_logging", lambda: None)
    monkeypatch.setattr(smoke_test_llm, "build_llm_client", Mock(side_effect=error))

    with caplog.at_level(logging.ERROR):
        assert smoke_test_llm.main() == 1
    assert message in caplog.text
    assert "sensitive-detail" not in caplog.text


def test_smoke_check_reports_completion_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A constructed client alone is not proof that the model works."""
    fake = FakeLLM(error=LLMError("Provider unavailable"))
    monkeypatch.setattr(smoke_test_llm, "configure_logging", lambda: None)
    monkeypatch.setattr(smoke_test_llm, "build_llm_client", lambda: fake)

    assert smoke_test_llm.main() == 1
    assert len(fake.calls) == 1


def test_smoke_check_rejects_unexpected_reply(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A reachable model must still follow the simple check instruction."""
    fake = FakeLLM(reply="unexpected-model-output")
    monkeypatch.setattr(smoke_test_llm, "configure_logging", lambda: None)
    monkeypatch.setattr(smoke_test_llm, "build_llm_client", lambda: fake)

    assert smoke_test_llm.main() == 1
    assert "did not match the requested token" in caplog.text
    assert "unexpected-model-output" not in caplog.text
