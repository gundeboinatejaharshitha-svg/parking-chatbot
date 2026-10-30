"""Tests for the booking, cancel and lookup flows (in-memory SQL, fake clock)."""

from __future__ import annotations

import datetime as dt

from app.chatbot.reservation import (
    ABORT_MESSAGE,
    NOT_FOUND_REPLY,
    BookingFlow,
    BookingStep,
    CancelFlow,
    LookupFlow,
)
from app.data.sql_store import ReservationInfo, SqlStore
from tests.conftest import FIXED_NOW, FakeClock


def _available(store: SqlStore, space_type: str) -> int:
    return store.get_availability(space_type)[0].available


# ---- booking ------------------------------------------------------------------
def test_booking_happy_path_with_prefill(sql_store: SqlStore, clock: FakeClock) -> None:
    before = _available(sql_store, "ev")
    flow = BookingFlow(sql_store, now=clock)

    reply = flow.start("book an EV bay for tomorrow from 9 to 11")
    assert flow.draft.missing() is BookingStep.NAME
    assert "name" in reply.text.lower()

    assert "plate" in flow.handle("Alice Morgan").text.lower()
    reply = flow.handle("ab12 cde")
    assert "Shall I confirm" in reply.text and "AB12CDE" in reply.text and not reply.done
    assert _available(sql_store, "ev") == before  # nothing created before confirmation

    reply = flow.handle("yes")
    assert reply.done and isinstance(reply.outcome, ReservationInfo)
    info = reply.outcome
    assert info.booking_ref in reply.text and info.booking_ref.startswith("CP-")
    assert (info.date, info.start_time, info.end_time) == (
        FIXED_NOW.date() + dt.timedelta(days=1), dt.time(9), dt.time(11)
    )
    assert info.space_type == "ev" and info.car_plate == "AB12CDE"
    assert _available(sql_store, "ev") == before - 1
    # The booking is real: it can be looked up with reference + plate.
    assert sql_store.get_reservation_for_user(info.booking_ref, "AB12CDE").status == "confirmed"


def test_booking_step_by_step_with_validation(sql_store: SqlStore, clock: FakeClock) -> None:
    flow = BookingFlow(sql_store, now=clock)
    assert "date" in flow.start("I want to book").text.lower()

    assert "couldn't read that date" in flow.handle("banana").text
    assert "past" in flow.handle("2029-12-31").text
    assert "Sunday" in flow.handle("sunday").text  # 08:00-18:00
    assert "outside our opening hours" in flow.handle("7am").text
    assert "end time" in flow.handle("9").text.lower()
    assert "must be after 09:00" in flow.handle("8").text
    assert "outside our opening hours" in flow.handle("19").text
    assert "type of space" in flow.handle("11").text.lower()
    assert "choose one of" in flow.handle("helipad").text
    assert "name" in flow.handle("compact").text.lower()
    assert "2-100 letters" in flow.handle("x").text
    assert "plate" in flow.handle("Bob Jones").text.lower()
    assert "2-8 letters and digits" in flow.handle("hello").text
    assert "Shall I confirm" in flow.handle("TE57CAR").text
    assert "yes to confirm" in flow.handle("maybe").text


def test_booking_declined_or_aborted_creates_nothing(sql_store: SqlStore, clock: FakeClock) -> None:
    before = _available(sql_store, "standard")

    flow = BookingFlow(sql_store, now=clock)
    flow.start("book standard tomorrow 10 to 12")
    flow.handle("Alice Morgan")
    flow.handle("AB12CDE")
    reply = flow.handle("no")
    assert reply.done and reply.outcome is None and "haven't booked" in reply.text

    flow = BookingFlow(sql_store, now=clock)
    flow.start("book")
    reply = flow.handle("never mind")
    assert reply.done and reply.text == ABORT_MESSAGE
    assert _available(sql_store, "standard") == before


def test_booking_rejects_start_time_already_passed_today(
    sql_store: SqlStore, clock: FakeClock
) -> None:
    flow = BookingFlow(sql_store, now=clock)  # now is 09:00
    flow.start("book today")
    assert "already passed" in flow.handle("8am").text
    assert "end time" in flow.handle("10am").text.lower()


def test_booking_when_space_type_sold_out(sql_store: SqlStore, clock: FakeClock) -> None:
    while _available(sql_store, "disabled") > 0:
        sql_store.create_reservation(
            customer_name="Filler", car_plate="FI11LER",
            date=FIXED_NOW.date() + dt.timedelta(days=1),
            start_time=dt.time(10), end_time=dt.time(11), space_type="disabled",
        )
    flow = BookingFlow(sql_store, now=clock)
    flow.start("book tomorrow 10 to 11")
    reply = flow.handle("disabled")
    assert "no disabled" in reply.text and "Available now" in reply.text
    assert flow.draft.space_type is None


def test_booking_handles_race_at_confirmation(sql_store: SqlStore, clock: FakeClock) -> None:
    flow = BookingFlow(sql_store, now=clock)
    flow.start("book tomorrow 10 to 11 disabled")
    flow.handle("Alice Morgan")
    flow.handle("AB12CDE")
    while _available(sql_store, "disabled") > 0:  # someone else takes the last spaces
        sql_store.create_reservation(
            customer_name="Filler", car_plate="FI11LER",
            date=FIXED_NOW.date() + dt.timedelta(days=1),
            start_time=dt.time(10), end_time=dt.time(11), space_type="disabled",
        )
    reply = flow.handle("yes")
    assert reply.done and reply.outcome is None and "nothing was booked" in reply.text


# ---- cancel -------------------------------------------------------------------
def test_cancel_requires_ref_and_plate_and_confirms(sql_store: SqlStore) -> None:
    before = _available(sql_store, "standard")
    flow = CancelFlow(sql_store)
    assert "booking reference" in flow.start("cancel my booking").text
    assert "CP-123456" in flow.handle("1234").text
    assert "plate" in flow.handle("CP-100001").text
    reply = flow.handle("ab 12 cde")
    assert "CP-100001" in reply.text and "Do you want to cancel it" in reply.text
    assert "yes to cancel" in flow.handle("hmm").text
    reply = flow.handle("yes")
    assert reply.done and reply.outcome is not None and reply.outcome.refund_eligible is True
    assert "cancelled" in reply.text and "full refund" in reply.text
    assert _available(sql_store, "standard") == before + 1


def test_cancel_wrong_plate_and_unknown_ref_same_reply(sql_store: SqlStore) -> None:
    wrong_plate = CancelFlow(sql_store).start("cancel CP-100001 ZZ99ZZZ")
    unknown_ref = CancelFlow(sql_store).start("cancel CP-999999 AB12CDE")
    assert wrong_plate.done and unknown_ref.done
    assert wrong_plate.text == unknown_ref.text == NOT_FOUND_REPLY
    assert sql_store.get_reservation_for_user("CP-100001", "AB12CDE").status == "confirmed"


def test_cancel_it_means_yes_and_keep_means_no(sql_store: SqlStore) -> None:
    flow = CancelFlow(sql_store)
    flow.start("cancel CP-100002 CD34EFG")
    reply = flow.handle("no, keep it")
    assert reply.done and "unchanged" in reply.text

    flow = CancelFlow(sql_store)
    flow.start("cancel CP-100002 CD34EFG")
    reply = flow.handle("yes cancel it")
    assert reply.done and "has been cancelled" in reply.text

    again = CancelFlow(sql_store).start("cancel CP-100002 CD34EFG")
    assert again.done and "already cancelled" in again.text


def test_cancel_late_reports_no_refund(sql_store: SqlStore, clock: FakeClock) -> None:
    info = sql_store.get_reservation_for_user("CP-100003", "EF56GHJ")
    clock.now = dt.datetime.combine(info.date, info.start_time) - dt.timedelta(hours=5)
    flow = CancelFlow(sql_store)
    flow.start("cancel CP-100003 EF56GHJ")
    reply = flow.handle("yes")
    assert reply.done and reply.outcome is not None and reply.outcome.refund_eligible is False
    assert "no refund" in reply.text


# ---- lookup -------------------------------------------------------------------
def test_lookup_shows_details_only_with_matching_pair(sql_store: SqlStore) -> None:
    flow = LookupFlow(sql_store)
    flow.start("check my booking")
    flow.handle("cp100004")
    reply = flow.handle("GH78JKL")
    assert reply.done and "CP-100004" in reply.text and "disabled" in reply.text

    miss = LookupFlow(sql_store).start("status of CP-100004 AB12CDE")
    assert miss.done and miss.text == NOT_FOUND_REPLY


def test_lookup_can_be_aborted(sql_store: SqlStore) -> None:
    flow = LookupFlow(sql_store)
    flow.start("find my reservation")
    reply = flow.handle("stop")
    assert reply.done and reply.text == ABORT_MESSAGE
