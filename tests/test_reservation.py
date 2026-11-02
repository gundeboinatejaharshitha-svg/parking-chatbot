"""Tests for the booking, cancel and lookup flows (in-memory SQL, fake clock)."""

from __future__ import annotations

import datetime as dt

import pytest

from app.chatbot.reservation import (
    ABORT_MESSAGE,
    EXIT_STOPPED_MESSAGE,
    MAX_MISMATCH_ATTEMPTS,
    NO_PLATE_ALTERNATIVE_REPLY,
    NO_REFERENCE_ALTERNATIVE_REPLY,
    NOT_FOUND_REPLY,
    RETRY_LIMIT_MESSAGE,
    BookingFlow,
    BookingStep,
    CancelFlow,
    LookupFlow,
    RefPlateState,
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
    assert "2-8 letters and digits" in flow.handle("invalid!").text
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
    # A mismatch keeps the flow active so the user can correct either value
    # (see test_cancel_can_correct_plate_after_mismatch) rather than ending it.
    assert not wrong_plate.done and not unknown_ref.done
    assert wrong_plate.text == unknown_ref.text == NOT_FOUND_REPLY
    assert sql_store.get_reservation_for_user("CP-100001", "AB12CDE").status == "confirmed"


def test_cancel_requires_matching_reference_plate_and_explicit_yes(sql_store: SqlStore) -> None:
    before = _available(sql_store, "standard")

    missing_plate = CancelFlow(sql_store)
    assert "plate" in missing_plate.start("cancel CP-100001").text.lower()
    reply = missing_plate.handle("cancel")
    assert reply.done and reply.outcome is None
    assert _available(sql_store, "standard") == before
    assert sql_store.get_reservation_for_user("CP-100001", "AB12CDE").status == "confirmed"

    malformed_reference = CancelFlow(sql_store)
    prompt = malformed_reference.start("cancel CP-12345 AB12CDE")
    assert "booking reference" in prompt.text.lower()
    assert _available(sql_store, "standard") == before

    wrong_plate = CancelFlow(sql_store)
    reply = wrong_plate.start("cancel CP-100001 plate ABCD")
    assert not reply.done and reply.text == NOT_FOUND_REPLY
    assert _available(sql_store, "standard") == before

    flow = CancelFlow(sql_store)
    prompt = flow.start("cancel CP-100001 AB12CDE")
    assert not prompt.done and "Do you want to cancel it" in prompt.text
    reply = flow.handle("cancel it")
    assert reply.done and reply.outcome is None
    assert _available(sql_store, "standard") == before
    assert sql_store.get_reservation_for_user("CP-100001", "AB12CDE").status == "confirmed"

    confirmed = CancelFlow(sql_store)
    confirmed.start("cancel CP-100001 AB12CDE")
    reply = confirmed.handle("yes")
    assert reply.done and reply.outcome is not None
    assert _available(sql_store, "standard") == before + 1


def test_cancel_no_and_ambiguous_responses_never_cancel(sql_store: SqlStore) -> None:
    before = _available(sql_store, "standard")
    flow = CancelFlow(sql_store)
    flow.start("cancel CP-100001 AB12CDE")
    reply = flow.handle("no, keep it")
    assert reply.done and "unchanged" in reply.text
    assert _available(sql_store, "standard") == before

    flow = CancelFlow(sql_store)
    flow.start("cancel CP-100001 AB12CDE")
    reply = flow.handle("maybe cancel it")
    assert not reply.done and "Please answer yes" in reply.text
    assert _available(sql_store, "standard") == before


def test_cancel_late_reports_no_refund(sql_store: SqlStore, clock: FakeClock) -> None:
    info = sql_store.get_reservation_for_user("CP-100003", "EF56GHJ")
    clock.now = dt.datetime.combine(info.date, info.start_time) - dt.timedelta(hours=5)
    flow = CancelFlow(sql_store)
    flow.start("cancel CP-100003 EF56GHJ")
    reply = flow.handle("yes")
    assert reply.done and reply.outcome is not None and reply.outcome.refund_eligible is False
    assert "no refund" in reply.text


# ---- cancel: conversation-state regressions (mismatch retry, acknowledgements) ------


def test_cancel_starts_with_bare_word(sql_store: SqlStore) -> None:
    flow = CancelFlow(sql_store)
    reply = flow.start("cancel")
    assert flow.state is RefPlateState.WAITING_FOR_REFERENCE
    assert "booking reference" in reply.text
    # The bare trigger word itself must never be read as the vehicle plate.
    assert flow.car_plate is None


def test_repeating_cancel_intent_mid_flow_does_not_reset_it(sql_store: SqlStore) -> None:
    flow = CancelFlow(sql_store)
    flow.start("cancel")
    reply = flow.handle("i want to cancel")
    assert flow.state is RefPlateState.WAITING_FOR_REFERENCE
    assert flow.booking_ref is None and not reply.done
    assert "booking reference" in reply.text


def test_invalid_reference_format_keeps_waiting_for_reference(sql_store: SqlStore) -> None:
    flow = CancelFlow(sql_store)
    flow.start("cancel")
    reply = flow.handle("CP-123")
    assert flow.state is RefPlateState.WAITING_FOR_REFERENCE
    assert not reply.done and "CP-123456" in reply.text


def test_valid_reference_format_advances_to_plate(sql_store: SqlStore) -> None:
    flow = CancelFlow(sql_store)
    flow.start("cancel")
    reply = flow.handle("CP-100001")
    assert flow.state is RefPlateState.WAITING_FOR_PLATE
    assert flow.booking_ref == "CP-100001"
    assert "plate" in reply.text.lower()


def test_invalid_plate_format_keeps_waiting_for_plate(sql_store: SqlStore) -> None:
    flow = CancelFlow(sql_store)
    flow.start("cancel CP-100001")
    reply = flow.handle("###")
    assert flow.state is RefPlateState.WAITING_FOR_PLATE
    assert not reply.done and "vehicle plate" in reply.text.lower()


def test_cancel_can_correct_plate_after_mismatch(sql_store: SqlStore) -> None:
    before = _available(sql_store, "standard")
    flow = CancelFlow(sql_store)
    mismatch = flow.start("cancel CP-100001 ZZ99ZZZ")
    assert not mismatch.done and mismatch.text == NOT_FOUND_REPLY
    assert flow.state is RefPlateState.RETRY_AFTER_MISMATCH

    corrected = flow.handle("AB12CDE")
    assert not corrected.done and "Do you want to cancel it" in corrected.text
    reply = flow.handle("yes")
    assert reply.done and reply.outcome is not None
    assert _available(sql_store, "standard") == before + 1


def test_cancel_can_correct_reference_after_mismatch(sql_store: SqlStore) -> None:
    flow = CancelFlow(sql_store)
    mismatch = flow.start("cancel CP-999999 AB12CDE")
    assert not mismatch.done
    assert flow.state is RefPlateState.RETRY_AFTER_MISMATCH

    corrected = flow.handle("CP-100001")
    assert not corrected.done and "Do you want to cancel it" in corrected.text
    assert flow.booking_ref == "CP-100001" and flow.car_plate == "AB12CDE"


def test_cancel_state_is_preserved_across_a_database_mismatch(sql_store: SqlStore) -> None:
    before = _available(sql_store, "standard")
    flow = CancelFlow(sql_store)
    flow.start("cancel CP-100001 ZZ99ZZZ")
    # Still waiting on a correction, not ended: another mismatch keeps it alive too.
    reply = flow.handle("AB11AAA")
    assert not reply.done and reply.text == NOT_FOUND_REPLY
    assert flow.state is RefPlateState.RETRY_AFTER_MISMATCH
    assert _available(sql_store, "standard") == before


def test_i_dont_have_my_reference_explains_requirement(sql_store: SqlStore) -> None:
    flow = CancelFlow(sql_store)
    flow.start("cancel")
    reply = flow.handle("I don't have my booking reference.")
    assert not reply.done and reply.text == NO_REFERENCE_ALTERNATIVE_REPLY
    assert flow.state is RefPlateState.WAITING_FOR_REFERENCE


def test_i_lost_my_reference_explains_requirement(sql_store: SqlStore) -> None:
    flow = CancelFlow(sql_store)
    flow.start("cancel")
    reply = flow.handle("I lost my booking reference.")
    assert not reply.done and reply.text == NO_REFERENCE_ALTERNATIVE_REPLY


def test_i_dont_know_my_plate_explains_requirement(sql_store: SqlStore) -> None:
    flow = CancelFlow(sql_store)
    flow.start("cancel CP-100001")
    reply = flow.handle("I don't know my vehicle registration.")
    assert not reply.done and reply.text == NO_PLATE_ALTERNATIVE_REPLY


def test_ok_while_waiting_for_reference_is_not_read_as_a_reference(sql_store: SqlStore) -> None:
    flow = CancelFlow(sql_store)
    flow.start("cancel")
    reply = flow.handle("OK")
    assert flow.state is RefPlateState.WAITING_FOR_REFERENCE
    assert not reply.done and "booking reference" in reply.text


def test_ok_while_waiting_for_plate_is_not_read_as_a_plate(sql_store: SqlStore) -> None:
    flow = CancelFlow(sql_store)
    flow.start("cancel CP-100001")
    reply = flow.handle("OK")
    assert flow.state is RefPlateState.WAITING_FOR_PLATE
    assert flow.car_plate is None
    assert not reply.done and "vehicle registration plate" in reply.text


def test_ok_at_confirmation_is_governed_by_existing_yes_no_rules(sql_store: SqlStore) -> None:
    # "OK" only ever reaches this point once the full summary has already been
    # shown and explicit confirmation is actually being asked for; the existing
    # yes/no parser (unchanged here) already treats it as an affirmative there.
    flow = CancelFlow(sql_store)
    flow.start("cancel CP-100001 AB12CDE")
    assert flow.state is RefPlateState.WAITING_FOR_CONFIRMATION
    reply = flow.handle("OK")
    assert reply.done and reply.outcome is not None


def test_explicit_no_keeps_the_booking(sql_store: SqlStore) -> None:
    before = _available(sql_store, "standard")

    kept = CancelFlow(sql_store)
    kept.start("cancel CP-100001 AB12CDE")
    reply = kept.handle("no")
    assert reply.done and reply.outcome is None and "unchanged" in reply.text
    assert _available(sql_store, "standard") == before
    assert sql_store.get_reservation_for_user("CP-100001", "AB12CDE").status == "confirmed"


def test_explicit_yes_cancels_the_booking(sql_store: SqlStore) -> None:
    before = _available(sql_store, "standard")

    flow = CancelFlow(sql_store)
    flow.start("cancel CP-100001 AB12CDE")
    reply = flow.handle("yes")
    assert reply.done and reply.outcome is not None
    assert _available(sql_store, "standard") == before + 1


def test_never_mind_aborts_the_cancellation(sql_store: SqlStore) -> None:
    before = _available(sql_store, "standard")
    flow = CancelFlow(sql_store)
    flow.start("cancel")
    reply = flow.handle("never mind")
    assert reply.done and reply.text == ABORT_MESSAGE
    assert _available(sql_store, "standard") == before


@pytest.mark.parametrize("phrase", ["It is not available.", "I don't remember it."])
def test_additional_dont_know_reference_phrasings(sql_store: SqlStore, phrase: str) -> None:
    flow = CancelFlow(sql_store)
    flow.start("cancel")
    reply = flow.handle(phrase)
    assert not reply.done and reply.text == NO_REFERENCE_ALTERNATIVE_REPLY


def test_no_exits_while_waiting_for_reference(sql_store: SqlStore) -> None:
    before = _available(sql_store, "standard")
    flow = CancelFlow(sql_store)
    flow.start("cancel")
    flow.handle("I don't have it.")
    reply = flow.handle("no")
    assert reply.done and reply.text == EXIT_STOPPED_MESSAGE
    assert _available(sql_store, "standard") == before


@pytest.mark.parametrize(
    "phrase", ["no", "no thanks", "I don't want to cancel", "leave it", "go back"]
)
def test_exit_phrases_stop_cancellation_during_retry(sql_store: SqlStore, phrase: str) -> None:
    flow = CancelFlow(sql_store)
    flow.start("cancel CP-999999 ZZ99ZZZ")  # mismatch -> retry state
    reply = flow.handle(phrase)
    assert reply.done and reply.text == EXIT_STOPPED_MESSAGE


def test_retry_limit_stops_after_max_mismatch_attempts(sql_store: SqlStore) -> None:
    flow = CancelFlow(sql_store)
    flow.start("cancel CP-999999 ZZ99ZZZ")
    reply = flow.handle("CP-888888 YY88YYY")
    assert not reply.done  # one retry left
    reply = flow.handle("CP-777777 XX77XXX")
    assert reply.done and reply.text == RETRY_LIMIT_MESSAGE
    assert flow.mismatch_count == MAX_MISMATCH_ATTEMPTS


# ---- lookup -------------------------------------------------------------------
def test_lookup_shows_details_only_with_matching_pair(sql_store: SqlStore) -> None:
    flow = LookupFlow(sql_store)
    flow.start("check my booking")
    flow.handle("cp100004")
    reply = flow.handle("GH78JKL")
    assert reply.done and "CP-100004" in reply.text and "disabled" in reply.text

    miss = LookupFlow(sql_store).start("status of CP-100004 AB12CDE")
    assert not miss.done and miss.text == NOT_FOUND_REPLY


def test_lookup_can_be_aborted(sql_store: SqlStore) -> None:
    flow = LookupFlow(sql_store)
    flow.start("find my reservation")
    reply = flow.handle("stop")
    assert reply.done and reply.text == ABORT_MESSAGE
