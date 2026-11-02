"""Tests for the /admin approval endpoints (authorization, approve, refuse)."""

from __future__ import annotations

import datetime as dt

from fastapi.testclient import TestClient

from app.agents.approval_agent import ApprovalAgent
from app.agents.approval_store import ApprovalDraft, ApprovalStore
from app.data.sql_store import SqlStore
from app.main import create_app


class _Notifier:
    def notify(self, request_id: str, message: str) -> None:
        pass


def _client(sql_store: SqlStore, approval_store: ApprovalStore, admin_key: str) -> TestClient:
    agent = ApprovalAgent(sql_store, approval_store, _Notifier())
    app = create_app(sql_store=sql_store, approval_agent=agent)
    app.state.admin_api_key = admin_key  # override the env-driven default for the test
    return TestClient(app)


def _submit(approval_store: ApprovalStore, sql_store: SqlStore) -> str:
    agent = ApprovalAgent(sql_store, approval_store, _Notifier())
    result = agent.submit_request(
        ApprovalDraft(
            session_id="s1",
            customer_name="Jordan Reyes",
            car_plate="AB99XYZ",
            date=dt.date(2030, 1, 8),
            start_time=dt.time(10),
            end_time=dt.time(12),
            space_type="ev",
        )
    )
    return result.request.id


# ---- 8: unauthorized administrator action ------------------------------------------


def test_missing_admin_key_is_rejected(sql_store: SqlStore, approval_store: ApprovalStore) -> None:
    client = _client(sql_store, approval_store, "secret-key")
    response = client.get("/admin/requests")
    assert response.status_code == 401


def test_wrong_admin_key_is_rejected(sql_store: SqlStore, approval_store: ApprovalStore) -> None:
    request_id = _submit(approval_store, sql_store)
    client = _client(sql_store, approval_store, "secret-key")

    response = client.post(
        f"/admin/requests/{request_id}/decision",
        json={"decision": "approve"},
        headers={"X-Admin-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_correct_admin_key_can_list_and_decide(
    sql_store: SqlStore, approval_store: ApprovalStore
) -> None:
    request_id = _submit(approval_store, sql_store)
    client = _client(sql_store, approval_store, "secret-key")
    headers = {"X-Admin-Key": "secret-key"}

    pending = client.get("/admin/requests", headers=headers)
    assert pending.status_code == 200
    assert any(r["request_id"] == request_id for r in pending.json())

    decision = client.post(
        f"/admin/requests/{request_id}/decision",
        json={"decision": "approve"},
        headers=headers,
    )
    assert decision.status_code == 200
    assert decision.json()["status"] == "approved"
    assert decision.json()["booking_ref"]


def test_refusal_reason_is_returned(sql_store: SqlStore, approval_store: ApprovalStore) -> None:
    request_id = _submit(approval_store, sql_store)
    client = _client(sql_store, approval_store, "secret-key")

    response = client.post(
        f"/admin/requests/{request_id}/decision",
        json={"decision": "refuse", "reason": "Duplicate booking"},
        headers={"X-Admin-Key": "secret-key"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "request_id": request_id,
        "status": "refused",
        "booking_ref": None,
        "refusal_reason": "Duplicate booking",
    }


def test_duplicate_decision_returns_conflict(
    sql_store: SqlStore, approval_store: ApprovalStore
) -> None:
    request_id = _submit(approval_store, sql_store)
    client = _client(sql_store, approval_store, "secret-key")
    headers = {"X-Admin-Key": "secret-key"}
    client.post(f"/admin/requests/{request_id}/decision", json={"decision": "approve"},
                headers=headers)

    response = client.post(
        f"/admin/requests/{request_id}/decision", json={"decision": "refuse"}, headers=headers
    )
    assert response.status_code == 409


def test_unknown_request_id_returns_404(
    sql_store: SqlStore, approval_store: ApprovalStore
) -> None:
    client = _client(sql_store, approval_store, "secret-key")
    response = client.post(
        "/admin/requests/AR-999999/decision",
        json={"decision": "approve"},
        headers={"X-Admin-Key": "secret-key"},
    )
    assert response.status_code == 404


def test_admin_endpoints_disabled_when_no_agent_configured(sql_store: SqlStore) -> None:
    # require_admin_approval now defaults to true, so create_app auto-builds an
    # agent; explicitly clear it to test the admin-routes guard in isolation.
    app = create_app(sql_store=sql_store)
    app.state.admin_api_key = "secret-key"
    app.state.approval_agent = None
    client = TestClient(app)
    response = client.get("/admin/requests", headers={"X-Admin-Key": "secret-key"})
    assert response.status_code == 503
