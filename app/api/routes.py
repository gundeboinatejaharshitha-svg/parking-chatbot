"""FastAPI routes that adapt requests to existing chatbot and SQL services."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.schemas import (
    AvailabilityItem,
    AvailabilityResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SourceResponse,
    SpaceType,
)
from app.chatbot.orchestrator import ConversationSession, Orchestrator, build_orchestrator
from app.config import get_settings
from app.data.sql_store import SqlStore

logger = logging.getLogger(__name__)


@dataclass
class _SessionEntry:
    """Conversation state serialized per bearer-like session identifier."""

    conversation: ConversationSession = field(default_factory=ConversationSession)
    lock: threading.Lock = field(default_factory=threading.Lock)


def install_routes(
    app: FastAPI,
    *,
    orchestrator: Orchestrator | None = None,
    sql_store: SqlStore | None = None,
) -> None:
    """Install routes and injectable service state on an application."""
    app.state.orchestrator = orchestrator
    app.state.sql_store = sql_store
    app.state.sessions: dict[UUID, _SessionEntry] = {}
    app.state.sessions_lock = threading.Lock()
    app.state.service_lock = threading.Lock()
    router = APIRouter()

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        logger.info("Rejected invalid API request")
        return JSONResponse(status_code=422, content={"detail": "Request validation failed."})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("API request failed: %s", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"detail": "The request could not be completed."},
        )

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @router.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        service = _get_orchestrator(request)
        session_id, entry = _resolve_session(request, body.session_id)
        with entry.lock:
            result = service.handle(body.message, entry.conversation)
        return ChatResponse(
            session_id=session_id,
            reply=result.text,
            intent=result.intent.value,
            sources=[
                SourceResponse(
                    id=source.id,
                    source=source.source,
                    section=source.section,
                    score=source.score,
                )
                for source in result.sources
            ],
            flow=result.flow,
            flow_done=result.flow_done,
        )

    @router.get("/availability", response_model=AvailabilityResponse)
    def availability(
        request: Request,
        space_type: Annotated[SpaceType | None, Query()] = None,
    ) -> AvailabilityResponse:
        store = _get_sql_store(request)
        rows = store.get_availability(space_type)
        return AvailabilityResponse(
            items=[
                AvailabilityItem(
                    space_type=row.space_type,
                    total=row.total,
                    available=row.available,
                )
                for row in rows
            ]
        )

    app.include_router(router)


def _resolve_session(request: Request, requested_id: UUID | None) -> tuple[UUID, _SessionEntry]:
    """Return an existing session or create a fresh opaque session identifier."""
    with request.app.state.sessions_lock:
        if requested_id is not None:
            existing = request.app.state.sessions.get(requested_id)
            if existing is not None:
                return requested_id, existing
        session_id = uuid4()
        entry = _SessionEntry()
        request.app.state.sessions[session_id] = entry
        return session_id, entry


def _get_orchestrator(request: Request) -> Orchestrator:
    """Build the chatbot service lazily so health and SQL endpoints need no LLM key."""
    service = request.app.state.orchestrator
    if service is None:
        with request.app.state.service_lock:
            service = request.app.state.orchestrator
            if service is None:
                service = build_orchestrator()
                request.app.state.orchestrator = service
    return service


def _get_sql_store(request: Request) -> SqlStore:
    """Return the SQL service without initializing vector or LLM components."""
    store = request.app.state.sql_store
    if store is None:
        with request.app.state.service_lock:
            store = request.app.state.sql_store
            if store is None:
                store = SqlStore.from_url(get_settings().sqlite_url)
                request.app.state.sql_store = store
    return store


__all__ = ["install_routes"]
