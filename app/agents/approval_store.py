"""Data access for human-in-the-loop approval requests (new table, same database)."""

from __future__ import annotations

import datetime as dt
import random
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.agents.approval_models import (
    APPROVED,
    EXPIRED,
    PENDING_APPROVAL,
    REFUSED,
    ApprovalBase,
    ApprovalRequest,
)

_MAX_REF_ATTEMPTS = 5


class ApprovalStoreError(Exception):
    """Base class for approval-store errors."""


class ApprovalRequestNotFoundError(ApprovalStoreError):
    """No request exists for the given ID."""


class AlreadyDecidedError(ApprovalStoreError):
    """The request was already approved or refused; decisions are not reprocessed."""


class RequestExpiredError(ApprovalStoreError):
    """The request's approval window has passed."""


class DuplicateRequestError(ApprovalStoreError):
    """An identical pending request already exists for this session."""


@dataclass(frozen=True)
class ApprovalDraft:
    """The subset of a booking draft needed to request approval."""

    session_id: str
    customer_name: str
    car_plate: str
    date: dt.date
    start_time: dt.time
    end_time: dt.time
    space_type: str


def _generate_request_id() -> str:
    return f"AR-{random.randint(0, 999999):06d}"  # noqa: S311 - not security-sensitive


class ApprovalStore:
    """CRUD and state transitions for ``ApprovalRequest`` rows."""

    def __init__(
        self,
        engine: Engine,
        now: Callable[[], dt.datetime] | None = None,
        ttl_hours: float = 24.0,
    ) -> None:
        self._engine = engine
        self._now = now or dt.datetime.now
        self._ttl_hours = ttl_hours
        self._session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        ApprovalBase.metadata.create_all(engine)

    def create_request(self, draft: ApprovalDraft) -> ApprovalRequest:
        """Create a PENDING_APPROVAL request, rejecting an identical pending duplicate."""
        now = self._now()
        with self._session_factory.begin() as session:
            existing = session.scalar(
                select(ApprovalRequest).where(
                    ApprovalRequest.session_id == draft.session_id,
                    ApprovalRequest.status == PENDING_APPROVAL,
                    ApprovalRequest.car_plate == draft.car_plate,
                    ApprovalRequest.date == draft.date,
                    ApprovalRequest.start_time == draft.start_time,
                    ApprovalRequest.space_type == draft.space_type,
                )
            )
            if existing is not None:
                raise DuplicateRequestError(existing.id)

            for _ in range(_MAX_REF_ATTEMPTS):
                request_id = _generate_request_id()
                if session.get(ApprovalRequest, request_id) is None:
                    break
            else:  # pragma: no cover - astronomically unlikely
                raise ApprovalStoreError("Could not generate a unique request ID.")

            row = ApprovalRequest(
                id=request_id,
                session_id=draft.session_id,
                customer_name=draft.customer_name,
                car_plate=draft.car_plate,
                date=draft.date,
                start_time=draft.start_time,
                end_time=draft.end_time,
                space_type=draft.space_type,
                status=PENDING_APPROVAL,
                created_at=now,
                expires_at=now + dt.timedelta(hours=self._ttl_hours),
            )
            session.add(row)
            session.flush()
            session.refresh(row)
            return row

    def get(self, request_id: str) -> ApprovalRequest:
        with self._session_factory() as session:
            row = session.get(ApprovalRequest, request_id)
        if row is None:
            raise ApprovalRequestNotFoundError(request_id)
        return row

    def list_pending(self) -> list[ApprovalRequest]:
        self._expire_overdue()
        with self._session_factory() as session:
            return list(
                session.scalars(
                    select(ApprovalRequest)
                    .where(ApprovalRequest.status == PENDING_APPROVAL)
                    .order_by(ApprovalRequest.created_at)
                )
            )

    def list_all(self) -> list[ApprovalRequest]:
        """Every request regardless of status, newest first (dashboard counts/history)."""
        self._expire_overdue()
        with self._session_factory() as session:
            return list(
                session.scalars(
                    select(ApprovalRequest).order_by(ApprovalRequest.created_at.desc())
                )
            )

    def _expire_overdue(self) -> None:
        now = self._now()
        with self._session_factory.begin() as session:
            overdue = session.scalars(
                select(ApprovalRequest).where(
                    ApprovalRequest.status == PENDING_APPROVAL,
                    ApprovalRequest.expires_at <= now,
                )
            )
            for row in overdue:
                row.status = EXPIRED

    def decide(
        self, request_id: str, approve: bool, actor: str, reason: str | None = None
    ) -> ApprovalRequest:
        """Record an approve/refuse decision; never reprocesses an already-decided request."""
        now = self._now()
        with self._session_factory.begin() as session:
            row = session.get(ApprovalRequest, request_id)
            if row is None:
                raise ApprovalRequestNotFoundError(request_id)
            if row.status == PENDING_APPROVAL and row.expires_at <= now:
                row.status = EXPIRED
            if row.status != PENDING_APPROVAL:
                if row.status == EXPIRED:
                    raise RequestExpiredError(request_id)
                raise AlreadyDecidedError(request_id)
            row.status = APPROVED if approve else REFUSED
            row.decided_by = actor
            row.decided_at = now
            row.refusal_reason = reason
            session.flush()
            session.refresh(row)
            return row

    def mark_booking_ref(self, request_id: str, booking_ref: str) -> None:
        """Link the SQL reservation created after approval back to its request."""
        with self._session_factory.begin() as session:
            row = session.get(ApprovalRequest, request_id)
            if row is not None:
                row.booking_ref = booking_ref

    def mark_refused(self, request_id: str, reason: str) -> None:
        """Flip an approved-but-unfulfillable request to refused (e.g. sold out)."""
        with self._session_factory.begin() as session:
            row = session.get(ApprovalRequest, request_id)
            if row is not None:
                row.status = REFUSED
                row.refusal_reason = reason

__all__ = [
    "AlreadyDecidedError",
    "ApprovalDraft",
    "ApprovalRequestNotFoundError",
    "ApprovalStore",
    "ApprovalStoreError",
    "DuplicateRequestError",
    "RequestExpiredError",
]
