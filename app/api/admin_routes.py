"""Administrator endpoints for human-in-the-loop reservation approval (Stage 2)."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from app.agents.approval_agent import ApprovalAgent
from app.agents.approval_store import (
    AlreadyDecidedError,
    ApprovalRequestNotFoundError,
    RequestExpiredError,
)
from app.data.sql_store import mask_plate

logger = logging.getLogger(__name__)


class PendingRequestResponse(BaseModel):
    request_id: str
    customer_name: str
    car_plate: str
    date: str
    start_time: str
    end_time: str
    space_type: str
    status: str
    created_at: str
    decided_at: str | None = None
    decided_by: str | None = None
    booking_ref: str | None = None
    refusal_reason: str | None = None


class DecisionRequest(BaseModel):
    decision: Literal["approve", "refuse"]
    reason: str | None = None


class DecisionResponse(BaseModel):
    request_id: str
    status: str
    booking_ref: str | None = None
    refusal_reason: str | None = None


def _require_admin(request: Request, x_admin_key: str | None) -> None:
    expected = request.app.state.admin_api_key
    if not expected or x_admin_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized.")


def _get_agent(request: Request) -> ApprovalAgent:
    agent = request.app.state.approval_agent
    if agent is None:
        raise HTTPException(status_code=503, detail="Approval workflow is not configured.")
    return agent


def install_admin_routes(
    app: FastAPI, *, approval_agent: ApprovalAgent | None, admin_api_key: str
) -> None:
    """Install admin routes and the admin service state on an application."""
    app.state.approval_agent = approval_agent
    app.state.admin_api_key = admin_api_key
    router = APIRouter(prefix="/admin")

    @router.get("/requests", response_model=list[PendingRequestResponse])
    def list_requests(
        request: Request,
        x_admin_key: str | None = Header(default=None),
        status: Literal["pending_approval", "approved", "refused", "expired", "all"] = (
            "pending_approval"
        ),
    ) -> list[PendingRequestResponse]:
        """List requests; defaults to pending only, or ``status=all`` for the dashboard
        (counts + decision history)."""
        _require_admin(request, x_admin_key)
        agent = _get_agent(request)
        rows = agent.list_pending() if status == "pending_approval" else agent.list_all()
        if status not in ("pending_approval", "all"):
            rows = [r for r in rows if r.status == status]
        return [
            PendingRequestResponse(
                request_id=r.id,
                customer_name=r.customer_name,
                car_plate=mask_plate(r.car_plate),
                date=r.date.isoformat(),
                start_time=r.start_time.strftime("%H:%M"),
                end_time=r.end_time.strftime("%H:%M"),
                space_type=r.space_type,
                status=r.status,
                created_at=r.created_at.isoformat(timespec="minutes"),
                decided_at=r.decided_at.isoformat(timespec="minutes") if r.decided_at else None,
                decided_by=r.decided_by,
                booking_ref=r.booking_ref,
                refusal_reason=r.refusal_reason,
            )
            for r in rows
        ]

    @router.post("/requests/{request_id}/decision", response_model=DecisionResponse)
    def decide(
        request_id: str,
        body: DecisionRequest,
        request: Request,
        x_admin_key: str | None = Header(default=None),
    ) -> DecisionResponse:
        _require_admin(request, x_admin_key)
        agent = _get_agent(request)
        try:
            row = agent.decide(
                request_id, body.decision == "approve", actor="admin", reason=body.reason
            )
        except ApprovalRequestNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Request not found.") from exc
        except AlreadyDecidedError as exc:
            raise HTTPException(status_code=409, detail="Already decided.") from exc
        except RequestExpiredError as exc:
            raise HTTPException(status_code=410, detail="Request has expired.") from exc
        return DecisionResponse(
            request_id=row.id,
            status=row.status,
            booking_ref=row.booking_ref,
            refusal_reason=row.refusal_reason,
        )

    app.include_router(router)


__all__ = ["install_admin_routes"]
