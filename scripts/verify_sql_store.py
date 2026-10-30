"""Manual check of SqlStore behaviour. Run after seeding."""

from __future__ import annotations

import datetime as dt

from app.config import get_settings
from app.data.sql_store import (
    InvalidReservationError,
    ReservationNotFoundError,
    SqlStore,
)


def main() -> None:
    url = get_settings().sqlite_url
    store = SqlStore.from_url(url)

    print("Availability:", store.get_availability())
    print("EV price:", store.get_prices("ev"))
    print("Sunday hours:", store.get_working_hours("sunday"))

    day = dt.date.today() + dt.timedelta(days=3)
    print("is_open 10-12:", store.is_open(day, dt.time(10), dt.time(12)))
    print("is_open 03-04:", store.is_open(day, dt.time(3), dt.time(4)))

    print("Lookup sample:", store.get_reservation_for_user("cp-100001", "ab 12 cde").booking_ref)

    errors = []
    for ref, plate in (("CP-100001", "ZZ99ZZZ"), ("CP-999999", "AB12CDE")):
        try:
            store.cancel_reservation(ref, plate)
        except ReservationNotFoundError as exc:
            errors.append((type(exc).__name__, str(exc)))
    print("Identical errors:", len(errors) == 2 and errors[0] == errors[1], errors[0])

    try:
        store.create_reservation(
            customer_name="Test Driver", car_plate="TE57CAR",
            date=dt.date.today() - dt.timedelta(days=1),
            start_time=dt.time(10), end_time=dt.time(12), space_type="standard",
        )
    except InvalidReservationError as exc:
        print("Past date rejected:", exc)

    before = store.get_availability("standard")[0].available

    first = store.create_reservation(
        customer_name="Test Driver", car_plate="te 57 car", date=day,
        start_time=dt.time(10), end_time=dt.time(12), space_type="standard",
    )
    print("Created:", first.booking_ref, "| available:", store.get_availability("standard")[0].available)
    print("Cancel (3 days ahead):", store.cancel_reservation(first.booking_ref, "TE57CAR"))

    day2 = dt.date.today() + dt.timedelta(days=4)
    second = store.create_reservation(
        customer_name="Test Driver", car_plate="TE57CAR", date=day2,
        start_time=dt.time(10), end_time=dt.time(12), space_type="standard",
    )
    fake_now = dt.datetime.combine(day2, dt.time(10)) - dt.timedelta(hours=5)
    late_store = SqlStore.from_url(url, now=lambda: fake_now)
    print("Cancel (5h before):", late_store.cancel_reservation(second.booking_ref, "TE57CAR"))

    after = store.get_availability("standard")[0].available
    print("Availability restored:", before == after)


if __name__ == "__main__":
    main()