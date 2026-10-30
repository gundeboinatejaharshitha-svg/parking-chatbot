"""Small HTTP client for the Streamlit-to-FastAPI boundary."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import httpx
from pydantic import ValidationError

from app.api.schemas import ChatRequest, ChatResponse

CONNECTION_ERROR_MESSAGE = "I couldn't reach the chat service. Please try again in a moment."
INVALID_MESSAGE_MESSAGE = "Please enter a message between 1 and 4000 characters."
SERVER_ERROR_MESSAGE = (
    "The chat service had a problem answering that. Please try again in a moment."
)


class ChatApiError(RuntimeError):
    """Raised when the API cannot provide a valid chatbot response.

    ``user_message`` is always safe to show in the UI; ``str(exc)`` is for logs.
    """

    def __init__(self, message: str, user_message: str = CONNECTION_ERROR_MESSAGE) -> None:
        super().__init__(message)
        self.user_message = user_message


@dataclass
class ChatApiClient:
    """Call the backend without exposing transport or validation details to the UI."""

    base_url: str
    timeout_seconds: float = 30.0
    transport: httpx.BaseTransport | None = None

    def send_chat(self, message: str, session_id: UUID | str | None = None) -> ChatResponse:
        """Send one message and validate the safe public response shape."""
        try:
            request = ChatRequest(message=message, session_id=session_id)
        except ValidationError as exc:
            raise ChatApiError(
                "The chat service is unavailable or returned an invalid response.",
                INVALID_MESSAGE_MESSAGE,
            ) from exc
        try:
            with httpx.Client(
                base_url=self.base_url.rstrip("/"),
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.post("/chat", json=request.model_dump(mode="json"))
            if response.is_error:
                if response.status_code == 422:
                    user_message = INVALID_MESSAGE_MESSAGE
                elif response.status_code >= 500:
                    user_message = SERVER_ERROR_MESSAGE
                else:
                    user_message = CONNECTION_ERROR_MESSAGE
                raise ChatApiError(
                    f"The chat service rejected the request (HTTP {response.status_code}).",
                    user_message,
                )
            return ChatResponse.model_validate(response.json())
        except ChatApiError:
            raise
        except httpx.HTTPError as exc:
            raise ChatApiError(
                "The chat service is unavailable or returned an invalid response."
            ) from exc
        except (ValueError, ValidationError) as exc:
            raise ChatApiError(
                "The chat service is unavailable or returned an invalid response.",
                SERVER_ERROR_MESSAGE,
            ) from exc


__all__ = [
    "CONNECTION_ERROR_MESSAGE",
    "INVALID_MESSAGE_MESSAGE",
    "SERVER_ERROR_MESSAGE",
    "ChatApiClient",
    "ChatApiError",
]
