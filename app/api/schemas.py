"""Validated API request and response models."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

SpaceType = Literal["standard", "compact", "ev", "disabled"]


class ChatRequest(BaseModel):
    """One user message and optional server-issued conversation identifier."""

    message: str = Field(min_length=1, max_length=4000)
    session_id: UUID | None = None

    @field_validator("message")
    @classmethod
    def require_non_whitespace(cls, value: str) -> str:
        """Reject empty messages without altering the caller's text."""
        if not value.strip():
            raise ValueError("Message cannot be blank.")
        return value


class SourceResponse(BaseModel):
    """Public citation metadata for a grounded answer."""

    id: str
    source: str
    section: str
    score: float


class ChatResponse(BaseModel):
    """Safe chatbot reply and conversation continuity fields."""

    session_id: UUID
    reply: str
    intent: str
    sources: list[SourceResponse] = Field(default_factory=list)
    flow: str | None = None
    flow_done: bool = False


class AvailabilityItem(BaseModel):
    """Availability for one SQL-backed parking-space type."""

    space_type: SpaceType
    total: int
    available: int


class AvailabilityResponse(BaseModel):
    """Current SQL-backed availability results."""

    items: list[AvailabilityItem]


class HealthResponse(BaseModel):
    """Minimal process health status."""

    status: Literal["ok"] = "ok"
