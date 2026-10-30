"""Tests for the Streamlit HTTP client and rendered chat behavior."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from app.api.schemas import ChatResponse, SourceResponse
from app.ui.api_client import ChatApiClient, ChatApiError


def test_api_client_sends_session_and_validates_response() -> None:
    session_id = uuid.uuid4()
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        assert payload["message"] == "hello"
        assert payload["session_id"] == str(session_id)
        return httpx.Response(
            200,
            json={
                "session_id": str(session_id),
                "reply": "Hello.",
                "intent": "greeting",
                "sources": [],
                "flow": None,
                "flow_done": False,
            },
        )

    response = ChatApiClient(
        "http://testserver", transport=httpx.MockTransport(handle)
    ).send_chat("hello", session_id)

    assert response.reply == "Hello."
    assert requests[0].url.path == "/chat"


def test_api_client_hides_backend_error_details() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(503, json={"detail": "CP-100001 AB12CDE"})
    )
    client = ChatApiClient("http://testserver", transport=transport)

    with pytest.raises(ChatApiError, match="rejected"):
        client.send_chat("hello")


def test_api_client_wraps_invalid_user_input() -> None:
    client = ChatApiClient("http://testserver")

    with pytest.raises(ChatApiError, match="invalid response"):
        client.send_chat("   ")


def test_streamlit_ui_keeps_session_history_and_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    session_id = uuid.uuid4()
    source = SourceResponse(id="entrance", source="02_location_directions.md",
                            section="Entrance", score=0.9)
    response = ChatResponse(
        session_id=session_id,
        reply="Use the Station Entrance.",
        intent="static_question",
        sources=[source],
    )
    monkeypatch.setattr(ChatApiClient, "send_chat", lambda *_args, **_kwargs: response)
    script = Path(__file__).parents[1] / "app" / "ui" / "streamlit_app.py"

    app = AppTest.from_file(str(script)).run()
    app.chat_input[0].set_value("Where is the entrance?").run()

    messages = app.session_state["citypark_messages"]
    assert messages[0]["role"] == "user"
    assert messages[1]["content"] == "Use the Station Entrance."
    assert messages[1]["sources"][0]["source"] == "02_location_directions.md"
    assert app.session_state["citypark_session_id"] == str(session_id)


def test_streamlit_ui_shows_safe_error_when_api_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_client: ChatApiClient, _message: str, _session_id: object = None) -> ChatResponse:
        raise ChatApiError("backend details must not be shown")

    monkeypatch.setattr(ChatApiClient, "send_chat", fail)
    script = Path(__file__).parents[1] / "app" / "ui" / "streamlit_app.py"

    app = AppTest.from_file(str(script)).run()
    app.chat_input[0].set_value("hello").run()

    assert app.session_state["citypark_messages"][-1]["content"] == (
        "I couldn't reach the chat service. Please try again in a moment."
    )


def test_quick_action_button_sends_its_real_message(monkeypatch: pytest.MonkeyPatch) -> None:
    session_id = uuid.uuid4()
    sent: list[str] = []

    def record(_client: ChatApiClient, message: str, _session_id: object = None) -> ChatResponse:
        sent.append(message)
        return ChatResponse(
            session_id=session_id, reply="Here is availability.", intent="availability"
        )

    monkeypatch.setattr(ChatApiClient, "send_chat", record)
    script = Path(__file__).parents[1] / "app" / "ui" / "streamlit_app.py"

    app = AppTest.from_file(str(script)).run()
    app.button(key="quick_action_Check Availability").click().run()

    assert sent == ["What parking slots are available?"]
    assert app.session_state["citypark_messages"][0]["content"] == (
        "What parking slots are available?"
    )
    assert app.session_state["citypark_messages"][1]["content"] == "Here is availability."


def test_retry_button_resends_the_failed_message(monkeypatch: pytest.MonkeyPatch) -> None:
    session_id = uuid.uuid4()
    calls = {"count": 0}

    def flaky(_client: ChatApiClient, message: str, _session_id: object = None) -> ChatResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ChatApiError("down", "The chat service had a problem. Please try again.")
        return ChatResponse(session_id=session_id, reply="Welcome back.", intent="greeting")

    monkeypatch.setattr(ChatApiClient, "send_chat", flaky)
    script = Path(__file__).parents[1] / "app" / "ui" / "streamlit_app.py"

    app = AppTest.from_file(str(script)).run()
    app.chat_input[0].set_value("hello").run()
    assert app.session_state["citypark_last_failed_message"] == "hello"

    retry_buttons = [b for b in app.button if b.label == "Retry last message"]
    retry_buttons[0].click().run()

    assert calls["count"] == 2
    assert app.session_state["citypark_last_failed_message"] is None
    assert app.session_state["citypark_messages"][-1]["content"] == "Welcome back."


def test_waiting_indicator_shows_while_a_flow_is_in_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = uuid.uuid4()

    def mid_flow(_client: ChatApiClient, _message: str, _session_id: object = None) -> ChatResponse:
        return ChatResponse(
            session_id=session_id,
            reply="Which date would you like to park?",
            intent="book",
            flow="booking",
            flow_done=False,
        )

    monkeypatch.setattr(ChatApiClient, "send_chat", mid_flow)
    script = Path(__file__).parents[1] / "app" / "ui" / "streamlit_app.py"

    app = AppTest.from_file(str(script)).run()
    app.chat_input[0].set_value("I want to book parking").run()

    assert app.session_state["citypark_waiting_for_more"] is True
    captions = [c.value for c in app.caption]
    assert "Waiting for a few more details to continue..." in captions
