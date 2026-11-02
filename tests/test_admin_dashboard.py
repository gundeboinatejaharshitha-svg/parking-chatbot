"""Tests for the Streamlit administrator dashboard page (mocked AdminApiClient)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.ui.admin_api_client import AdminApiClient, AdminApiError

PAGE = Path(__file__).parents[1] / "app" / "ui" / "pages" / "1_Administrator.py"

PENDING_ROW = {
    "request_id": "AR-100001",
    "customer_name": "Jordan Reyes",
    "car_plate": "A*****Z",
    "date": "2030-01-08",
    "start_time": "10:00",
    "end_time": "12:00",
    "space_type": "ev",
    "status": "pending_approval",
    "created_at": "2030-01-07T09:00",
    "decided_at": None,
    "decided_by": None,
    "booking_ref": None,
    "refusal_reason": None,
}
DECIDED_ROW = {
    **PENDING_ROW,
    "request_id": "AR-100002",
    "status": "refused",
    "decided_at": "2030-01-07T10:00",
    "decided_by": "admin",
    "refusal_reason": "Sold out",
}


def _login(app: AppTest, key: str = "correct-key") -> AppTest:
    app.text_input(key="admin_key_input").set_value(key)
    return app.button(key="FormSubmitter:admin_login-Log in").click().run()


def test_wrong_key_shows_error_and_does_not_log_in(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_list(self, admin_key, status="all"):
        raise AdminApiError("Unauthorized.", 401)

    monkeypatch.setattr(AdminApiClient, "list_requests", fail_list)
    app = _login(AppTest.from_file(str(PAGE)).run(), key="wrong-key")

    assert app.session_state["admin_api_key"] is None
    assert any("Incorrect administrator key" in e.value for e in app.error)


def test_correct_key_logs_in_and_shows_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    def list_requests(self, admin_key, status="all"):
        assert admin_key == "correct-key"
        return [PENDING_ROW] if status == "pending_approval" else [PENDING_ROW, DECIDED_ROW]

    monkeypatch.setattr(AdminApiClient, "list_requests", list_requests)
    app = _login(AppTest.from_file(str(PAGE)).run())

    assert app.session_state["admin_api_key"] == "correct-key"
    metrics = {m.label: m.value for m in app.get("metric")}
    assert metrics == {"Pending": "1", "Approved": "0", "Refused": "1"}
    assert "AR-100001" in app.markdown[1].value or any(
        "AR-100001" in b.value for b in app.markdown
    )


def test_approve_button_calls_decide_and_shows_booking_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def list_requests(self, admin_key, status="all"):
        return [PENDING_ROW] if status != "all" else [PENDING_ROW]

    def decide(self, admin_key, request_id, decision, reason=None):
        calls.append((admin_key, request_id, decision, reason))
        return {"request_id": request_id, "status": "approved", "booking_ref": "CP-900001"}

    monkeypatch.setattr(AdminApiClient, "list_requests", list_requests)
    monkeypatch.setattr(AdminApiClient, "decide", decide)
    app = _login(AppTest.from_file(str(PAGE)).run())

    app.button(key="approve_AR-100001").click().run()

    assert calls == [("correct-key", "AR-100001", "approve", None)]
    assert any("CP-900001" in s.value for s in app.success)


def test_refuse_button_sends_the_optional_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def list_requests(self, admin_key, status="all"):
        return [PENDING_ROW]

    def decide(self, admin_key, request_id, decision, reason=None):
        calls.append((request_id, decision, reason))
        return {"request_id": request_id, "status": "refused", "booking_ref": None}

    monkeypatch.setattr(AdminApiClient, "list_requests", list_requests)
    monkeypatch.setattr(AdminApiClient, "decide", decide)
    app = _login(AppTest.from_file(str(PAGE)).run())

    app.text_input(key="reason_AR-100001").set_value("Duplicate booking")
    app.button(key="refuse_AR-100001").click().run()

    assert calls == [("AR-100001", "refuse", "Duplicate booking")]


def test_duplicate_decision_shows_conflict_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def list_requests(self, admin_key, status="all"):
        return [PENDING_ROW]

    def decide(self, admin_key, request_id, decision, reason=None):
        raise AdminApiError("Already decided.", 409)

    monkeypatch.setattr(AdminApiClient, "list_requests", list_requests)
    monkeypatch.setattr(AdminApiClient, "decide", decide)
    app = _login(AppTest.from_file(str(PAGE)).run())

    app.button(key="approve_AR-100001").click().run()

    assert any("already decided" in e.value.lower() for e in app.error)


def test_decision_history_lists_decided_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    def list_requests(self, admin_key, status="all"):
        return [] if status == "pending_approval" else [DECIDED_ROW]

    monkeypatch.setattr(AdminApiClient, "list_requests", list_requests)
    app = _login(AppTest.from_file(str(PAGE)).run())

    assert len(app.dataframe) > 0, "expected a decision-history table to be rendered"
    history = app.dataframe[0].value
    assert history.iloc[0]["Request"] == "AR-100002"
    assert history.iloc[0]["Reason"] == "Sold out"


def test_logout_clears_session(monkeypatch: pytest.MonkeyPatch) -> None:
    def list_requests(self, admin_key, status="all"):
        return [PENDING_ROW]

    monkeypatch.setattr(AdminApiClient, "list_requests", list_requests)
    app = _login(AppTest.from_file(str(PAGE)).run())

    logout_buttons = [b for b in app.button if b.label == "Log out"]
    logout_buttons[0].click().run()

    assert app.session_state["admin_api_key"] is None
