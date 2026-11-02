"""Tests for Agent 2: the human-in-the-loop approval store and agent."""

from __future__ import annotations

import datetime as dt

import pytest

from app.agents.approval_agent import ApprovalAgent
from app.agents.approval_store import (
    AlreadyDecidedError,
    ApprovalDraft,
    ApprovalStore,
    DuplicateRequestError,
    RequestExpiredError,
)
from app.data.sql_store import NoAvailabilityError, SqlStore
from tests.conftest import FakeClock, FakeLLM


def _draft(sql_store: SqlStore, space_type: str = "ev") -> ApprovalDraft:
    return ApprovalDraft(
        session_id="session-1",
        customer_name="Jordan Reyes",
        car_plate="AB99XYZ",
        date=dt.date(2030, 1, 8),
        start_time=dt.time(10, 0),
        end_time=dt.time(12, 0),
        space_type=space_type,
    )


class _RecordingNotifier:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail = fail

    def notify(self, request_id: str, message: str) -> None:
        if self.fail:
            raise ConnectionError("admin notification service unreachable")
        self.calls.append((request_id, message))


def _agent(sql_store: SqlStore, approval_store: ApprovalStore, *, fail_notify: bool = False):
    notifier = _RecordingNotifier(fail=fail_notify)
    return ApprovalAgent(sql_store, approval_store, notifier), notifier


# ---- 1/2/6/7: submission, notification, pending status ---------------------------


def test_submit_request_creates_pending_request_and_notifies_admin(
    sql_store: SqlStore, approval_store: ApprovalStore
) -> None:
    agent, notifier = _agent(sql_store, approval_store)
    result = agent.submit_request(_draft(sql_store))

    assert result.request.status == "pending_approval"
    assert result.request.id.startswith("AR-")
    assert result.notified is True
    assert notifier.calls and notifier.calls[0][0] == result.request.id
    assert "AB99XYZ" in notifier.calls[0][1]


def test_customer_can_check_pending_status_with_id_and_plate(
    sql_store: SqlStore, approval_store: ApprovalStore
) -> None:
    agent, _ = _agent(sql_store, approval_store)
    result = agent.submit_request(_draft(sql_store))

    found = agent.get_status(result.request.id, "AB99XYZ")
    assert found is not None and found.status == "pending_approval"

    wrong_plate = agent.get_status(result.request.id, "ZZ00ZZZ")
    assert wrong_plate is None  # never confirms the ID is real for the wrong plate


# ---- 3/4/5: administrator approval, refusal, refusal reason ----------------------


def test_administrator_approval_creates_the_real_reservation(
    sql_store: SqlStore, approval_store: ApprovalStore
) -> None:
    before = sql_store.get_availability("ev")[0].available
    agent, _ = _agent(sql_store, approval_store)
    result = agent.submit_request(_draft(sql_store))
    assert sql_store.get_availability("ev")[0].available == before  # not yet decremented

    decided = agent.decide(result.request.id, approve=True, actor="admin")

    assert decided.status == "approved" and decided.booking_ref is not None
    assert sql_store.get_reservation_for_user(decided.booking_ref, "AB99XYZ").status == "confirmed"
    assert sql_store.get_availability("ev")[0].available == before - 1


def test_administrator_refusal_never_creates_a_reservation(
    sql_store: SqlStore, approval_store: ApprovalStore
) -> None:
    before = sql_store.get_availability("ev")[0].available
    agent, _ = _agent(sql_store, approval_store)
    result = agent.submit_request(_draft(sql_store))

    decided = agent.decide(result.request.id, approve=False, actor="admin", reason="Sold out")

    assert decided.status == "refused" and decided.booking_ref is None
    assert decided.refusal_reason == "Sold out"
    assert sql_store.get_availability("ev")[0].available == before


# ---- 9: duplicate decisions ------------------------------------------------------


def test_duplicate_decision_is_rejected(sql_store: SqlStore, approval_store: ApprovalStore) -> None:
    agent, _ = _agent(sql_store, approval_store)
    result = agent.submit_request(_draft(sql_store))
    agent.decide(result.request.id, approve=True, actor="admin")

    with pytest.raises(AlreadyDecidedError):
        agent.decide(result.request.id, approve=False, actor="admin")


# ---- 10: duplicate requests -------------------------------------------------------


def test_duplicate_pending_request_is_rejected(
    sql_store: SqlStore, approval_store: ApprovalStore
) -> None:
    agent, _ = _agent(sql_store, approval_store)
    agent.submit_request(_draft(sql_store))

    with pytest.raises(DuplicateRequestError):
        agent.submit_request(_draft(sql_store))


# ---- 11: expired requests ----------------------------------------------------------


def test_expired_request_cannot_be_decided(
    sql_store: SqlStore, approval_store: ApprovalStore, clock: FakeClock
) -> None:
    agent, _ = _agent(sql_store, approval_store)
    result = agent.submit_request(_draft(sql_store))

    clock.now += dt.timedelta(hours=25)  # past the 24h TTL

    with pytest.raises(RequestExpiredError):
        agent.decide(result.request.id, approve=True, actor="admin")


# ---- 12: availability changes before approval --------------------------------------


def test_sold_out_between_submission_and_approval_refuses_automatically(
    sql_store: SqlStore, approval_store: ApprovalStore
) -> None:
    agent, _ = _agent(sql_store, approval_store)
    result = agent.submit_request(_draft(sql_store, space_type="disabled"))

    # Someone else takes every remaining disabled space while this request is pending.
    while sql_store.get_availability("disabled")[0].available > 0:
        sql_store.create_reservation(
            customer_name="Filler", car_plate="FI11LER", date=dt.date(2030, 1, 8),
            start_time=dt.time(10), end_time=dt.time(11), space_type="disabled",
        )

    decided = agent.decide(result.request.id, approve=True, actor="admin")
    assert decided.status == "refused" and "sold out" in decided.refusal_reason.lower()


def test_submit_request_rejects_when_already_sold_out(
    sql_store: SqlStore, approval_store: ApprovalStore
) -> None:
    agent, _ = _agent(sql_store, approval_store)
    while sql_store.get_availability("disabled")[0].available > 0:
        sql_store.create_reservation(
            customer_name="Filler", car_plate="FI11LER", date=dt.date(2030, 1, 8),
            start_time=dt.time(10), end_time=dt.time(11), space_type="disabled",
        )

    with pytest.raises(NoAvailabilityError):
        agent.submit_request(_draft(sql_store, space_type="disabled"))


# ---- 14: communication failures ----------------------------------------------------


def test_notification_failure_still_saves_the_request(
    sql_store: SqlStore, approval_store: ApprovalStore
) -> None:
    agent, notifier = _agent(sql_store, approval_store, fail_notify=True)
    result = agent.submit_request(_draft(sql_store))

    assert result.notified is False
    assert agent.get_status(result.request.id, "AB99XYZ") is not None  # saved regardless


# ---- LLM-assisted message drafting (optional; deterministic fallback) --------------


def test_message_drafting_falls_back_to_template_when_llm_unavailable(
    sql_store: SqlStore, approval_store: ApprovalStore, fake_llm: FakeLLM
) -> None:
    from app.chatbot.llm import LLMError

    fake_llm.error = LLMError("down")
    notifier = _RecordingNotifier()
    agent = ApprovalAgent(sql_store, approval_store, notifier, llm=fake_llm)
    result = agent.submit_request(_draft(sql_store))

    assert result.notified is True
    assert result.request.id in notifier.calls[0][1]


def test_message_drafting_uses_llm_when_available(
    sql_store: SqlStore, approval_store: ApprovalStore, fake_llm: FakeLLM
) -> None:
    fake_llm.reply = "Please review this reservation request."
    notifier = _RecordingNotifier()
    agent = ApprovalAgent(sql_store, approval_store, notifier, llm=fake_llm)
    agent.submit_request(_draft(sql_store))

    assert notifier.calls[0][1] == "Please review this reservation request."
    assert len(fake_llm.calls) == 1
