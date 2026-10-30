"""Small HTTP client for the Streamlit-to-FastAPI boundary."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import httpx
from pydantic import ValidationError

from app.api.schemas import ChatRequest, ChatResponse


class ChatApiError(RuntimeError):
    """Raised when the API cannot provide a valid chatbot response."""


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
            with httpx.Client(
                base_url=self.base_url.rstrip("/"),
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.post("/chat", json=request.model_dump(mode="json"))
            if response.is_error:
                raise ChatApiError("The chat service rejected the request.")
            return ChatResponse.model_validate(response.json())
        except ChatApiError:
            raise
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            raise ChatApiError(
                "The chat service is unavailable or returned an invalid response."
            ) from exc


__all__ = ["ChatApiClient", "ChatApiError"]
