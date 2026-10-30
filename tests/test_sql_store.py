"""Tests for app.data.sql_store on an in-memory SQLite engine with a fake clock.

The store itself is protected and already verified by scripts/verify_sql_store.py;
these tests pin its contract so later stages cannot regress it unnoticed.
"""

from __future__ import annotations

import datetime as dt

import pytest

from app.data.seed import seed
from app.data.sql_store import (
    CancellationNotAllowedError,
    InvalidReservationError,
    NoAvailabilityError,
    ReservationNotFoundError,
    SqlStore,
    UnknownSpaceTypeError,
    create_db_engine,
    mask_name,
    mask_plate,
    mask_ref,
    normalize_plate,
)

NOW = dt.datetime(2030, 1, 7, 9, 0)  # a Monday
TODAY = NOW.date()


class Clock:
    """Mutable fake clock injected into SqlStore."""

    def __init__(self, now: dt.datetime) -> None:
        self.now = now

    def __call__(self) -> dt.datetime:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock(NOW)


@pytest.fixture
def store(clock: Clock) -> SqlStore:
    store = SqlStore(create_db_engine("sqlite://"), now=clock)
    seed(store, today=TODAY)
    return store


def _book(store: SqlStore, days_ahead: int = 3, space_type: str = "standard", **overrides):
    params = dict(
        customer_name="Test Driver",
        car_plate="TE57CAR",
        date=TODAY + dt.timedelta(days=days_ahead),
        start_time=dt.time(10),
        end_time=dt.time(12),
        space_type=space_type,
    )
    params.update(overrides)
    return store.create_reservation(**params)


def _available(store: SqlStore, space_type: str = "standard") -> int:
    return store.get_availability(space_type)[0].available


# ---- read-only data --------------------------------------------------------
def test_seed_is_idempotent_and_exposes_reference_data(store: SqlStore) -> None:
    assert seed(store, today=TODAY) == {
        "spaces": 0, "pricing": 0, "working_hours": 0, "reservations": 0
    }
    assert [a.space_type for a in store.get_availability()] == [
        "standard", "compact", "ev", "disabled"
    ]
    assert store.get_prices("ev")[0].price_per_hour == pytest.approx(4.5)
    sunday = store.get_working_hours("sunday")[0]
    assert (sunday.open_time, sunday.close_time) == (dt.time(8), dt.time(18))
    with pytest.raises(UnknownSpaceTypeError):
        store.get_prices("helipad")


def test_is_open_respects_working_hours(store: SqlStore) -> None:
    sunday = TODAY + dt.timedelta(days=6)
    assert store.is_open(TODAY, dt.time(6), dt.time(22))
    assert not store.is_open(TODAY, dt.time(3), dt.time(4))
    assert store.is_open(sunday, dt.time(8), dt.time(18))
    assert not store.is_open(sunday, dt.time(7), dt.time(9))
    assert not store.is_open(TODAY, dt.time(12), dt.time(10))


# ---- create / cancel --------------------------------------------------------
def test_create_decrements_and_cancel_restores_availability(store: SqlStore) -> None:
    before = _available(store)
    info = _book(store)
    assert info.booking_ref.startswith("CP-") and len(info.booking_ref) == 9
    assert info.status == "confirmed" and info.car_plate == "TE57CAR"
    assert _available(store) == before - 1

    result = store.cancel_reservation(info.booking_ref, "TE57CAR")
    assert result.booking_ref == info.booking_ref
    assert _available(store) == before


def test_wrong_plate_and_unknown_ref_raise_identical_error(store: SqlStore) -> None:
    info = _book(store)
    errors = []
    for ref, plate in ((info.booking_ref, "ZZ99ZZZ"), ("CP-999999", "TE57CAR")):
        with pytest.raises(ReservationNotFoundError) as exc_info:
            store.cancel_reservation(ref, plate)
        errors.append(str(exc_info.value))
        with pytest.raises(ReservationNotFoundError):
            store.get_reservation_for_user(ref, plate)
    assert errors[0] == errors[1]
    assert _available(store) == 86 - 2 - 1  # seed: 2 standard samples, plus ours, untouched


def test_refund_eligibility_depends_on_24h_window(store: SqlStore, clock: Clock) -> None:
    early = _book(store, days_ahead=3)
    assert store.cancel_reservation(early.booking_ref, "TE57CAR").refund_eligible is True

    late = _book(store, days_ahead=4)
    start = dt.datetime.combine(late.date, late.start_time)
    clock.now = start - dt.timedelta(hours=5)
    result = store.cancel_reservation(late.booking_ref, "TE57CAR")
    assert result.refund_eligible is False
    assert result.hours_before_start == pytest.approx(5.0)

    exactly_24h = _book(store, days_ahead=6)
    clock.now = dt.datetime.combine(exactly_24h.date, exactly_24h.start_time) - dt.timedelta(
        hours=24
    )
    assert store.cancel_reservation(exactly_24h.booking_ref, "TE57CAR").refund_eligible is True


def test_cancel_after_start_or_twice_is_rejected(store: SqlStore, clock: Clock) -> None:
    info = _book(store)
    store.cancel_reservation(info.booking_ref, "TE57CAR")
    with pytest.raises(CancellationNotAllowedError):
        store.cancel_reservation(info.booking_ref, "TE57CAR")

    other = _book(store, days_ahead=4)
    clock.now = dt.datetime.combine(other.date, other.start_time) + dt.timedelta(minutes=1)
    with pytest.raises(CancellationNotAllowedError):
        store.cancel_reservation(other.booking_ref, "TE57CAR")


# ---- validation ---------------------------------------------------------------
def test_rejects_past_date_and_bad_time_range(store: SqlStore) -> None:
    with pytest.raises(InvalidReservationError, match="future"):
        _book(store, days_ahead=-1)
    with pytest.raises(InvalidReservationError, match="End time"):
        _book(store, start_time=dt.time(12), end_time=dt.time(12))
    with pytest.raises(InvalidReservationError, match="End time"):
        _book(store, start_time=dt.time(12), end_time=dt.time(10))
    with pytest.raises(InvalidReservationError, match="working hours"):
        _book(store, start_time=dt.time(4), end_time=dt.time(5))
    with pytest.raises(InvalidReservationError):
        _book(store, customer_name="X")
    with pytest.raises(InvalidReservationError):
        _book(store, car_plate="!!")
    with pytest.raises(UnknownSpaceTypeError):
        _book(store, space_type="helipad")


def test_plate_matching_is_case_and_space_insensitive(store: SqlStore) -> None:
    info = _book(store, car_plate="te 57-car")
    assert info.car_plate == "TE57CAR"
    assert store.get_reservation_for_user(info.booking_ref.lower(), " te57 car ").booking_ref == (
        info.booking_ref
    )
    assert store.cancel_reservation(info.booking_ref, "Te 57 CaR").booking_ref == info.booking_ref


def test_no_availability_raises_and_leaves_counter_at_zero(store: SqlStore) -> None:
    for _ in range(_available(store, "disabled")):
        _book(store, space_type="disabled")
    assert _available(store, "disabled") == 0
    with pytest.raises(NoAvailabilityError):
        _book(store, space_type="disabled")
    assert _available(store, "disabled") == 0


def test_masking_helpers_never_expose_full_values() -> None:
    assert normalize_plate(" ab-12 cde ") == "AB12CDE"
    assert mask_plate("AB12CDE") == "A*****E"
    assert mask_name("Alice Morgan") == "A***"
    assert mask_ref("CP-123456") == "CP-****56"
