"""ORM model for human-in-the-loop reservation approval requests.

A separate table and ``DeclarativeBase`` from ``app.data.sql_store`` (same database file,
same engine) so Stage 1's ``reservations`` table - which has a DB-level CHECK constraint
restricting ``status`` to ``confirmed``/``cancelled`` - never needs a schema change.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

PENDING_APPROVAL = "pending_approval"
APPROVED = "approved"
REFUSED = "refused"
EXPIRED = "expired"


class ApprovalBase(DeclarativeBase):
    pass


class ApprovalRequest(ApprovalBase):
    """One booking draft awaiting an administrator's decision."""

    __tablename__ = "approval_requests"

    id: Mapped[str] = mapped_column(primary_key=True)  # e.g. AR-123456
    session_id: Mapped[str] = mapped_column(index=True)
    customer_name: Mapped[str]
    car_plate: Mapped[str] = mapped_column(index=True)
    date: Mapped[dt.date]
    start_time: Mapped[dt.time]
    end_time: Mapped[dt.time]
    space_type: Mapped[str]
    status: Mapped[str] = mapped_column(default=PENDING_APPROVAL, index=True)
    booking_ref: Mapped[str | None] = mapped_column(default=None)
    refusal_reason: Mapped[str | None] = mapped_column(default=None)
    decided_by: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[dt.datetime]
    decided_at: Mapped[dt.datetime | None] = mapped_column(default=None)
    expires_at: Mapped[dt.datetime] = mapped_column()


__all__ = ["APPROVED", "EXPIRED", "PENDING_APPROVAL", "REFUSED", "ApprovalBase", "ApprovalRequest"]
