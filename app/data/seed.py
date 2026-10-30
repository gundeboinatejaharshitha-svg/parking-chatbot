"""Idempotent seed script. Usage: python -m app.data.seed [--reset]."""

from __future__ import annotations

import argparse
import datetime as dt
import logging

from sqlalchemy import select, update

from app.config import configure_logging, get_settings
from app.data.sql_store import (
    STATUS_CONFIRMED,
    ParkingSpace,
    Pricing,
    Reservation,
    SqlStore,
    WorkingHour,
)

logger = logging.getLogger(__name__)

# space_type -> (total, currently available before sample reservations)
SPACES: dict[str, tuple[int, int]] = {
    "standard": (120, 86),
    "compact": (40, 25),
    "ev": (20, 9),
    "disabled": (12, 7),
}

# space_type -> (price_per_hour, price_per_day) in GBP
PRICES: dict[str, tuple[float, float]] = {
    "standard": (3.50, 24.00),
    "compact": (3.00, 20.00),
    "ev": (4.50, 32.00),
    "disabled": (2.50, 18.00),
}

# day_of_week (0=Mon) -> (open, close). Sunday is the shorter day.
WORKING_HOURS: dict[int, tuple[dt.time, dt.time]] = {
    **{day: (dt.time(6, 0), dt.time(22, 0)) for day in range(6)},
    6: (dt.time(8, 0), dt.time(18, 0)),
}

# (booking_ref, name, plate, days_from_today, start, end, space_type)
SAMPLE_RESERVATIONS: list[tuple[str, str, str, int, dt.time, dt.time, str]] = [
    ("CP-100001", "Alice Morgan", "AB12CDE", 2, dt.time(9, 0), dt.time(12, 0), "standard"),
    ("CP-100002", "Ben Carter", "CD34EFG", 3, dt.time(10, 0), dt.time(14, 0), "compact"),
    ("CP-100003", "Chloe Evans", "EF56GHJ", 3, dt.time(13, 0), dt.time(17, 0), "ev"),
    ("CP-100004", "David Khan", "GH78JKL", 4, dt.time(8, 30), dt.time(11, 30), "disabled"),
    ("CP-100005", "Emma Wilson", "JK90LMN", 5, dt.time(12, 0), dt.time(16, 0), "standard"),
]


def seed(store: SqlStore, today: dt.date | None = None) -> dict[str, int]:
    """Populate the database; safe to run repeatedly. Returns rows added per table."""
    store.init_db()
    today = today or dt.date.today()
    added = {"spaces": 0, "pricing": 0, "working_hours": 0, "reservations": 0}

    with store.transaction() as session:
        # Live availability is never overwritten once it exists.
        for stype, (total, available) in SPACES.items():
            if session.scalar(select(ParkingSpace).where(ParkingSpace.space_type == stype)) is None:
                session.add(ParkingSpace(space_type=stype, total=total, available=available))
                added["spaces"] += 1

        # Reference data is upserted so edits here are applied on re-run.
        for stype, (per_hour, per_day) in PRICES.items():
            price = session.scalar(select(Pricing).where(Pricing.space_type == stype))
            if price is None:
                session.add(Pricing(space_type=stype, price_per_hour=per_hour, price_per_day=per_day))
                added["pricing"] += 1
            else:
                price.price_per_hour, price.price_per_day = per_hour, per_day

        for day, (open_t, close_t) in WORKING_HOURS.items():
            hours = session.scalar(select(WorkingHour).where(WorkingHour.day_of_week == day))
            if hours is None:
                session.add(
                    WorkingHour(day_of_week=day, open_time=open_t, close_time=close_t, is_closed=False)
                )
                added["working_hours"] += 1
            else:
                hours.open_time, hours.close_time, hours.is_closed = open_t, close_t, False

        session.flush()

        # Sample reservations use fixed references, so re-running adds nothing.
        for ref, name, plate, offset, start, end, stype in SAMPLE_RESERVATIONS:
            if session.scalar(select(Reservation.id).where(Reservation.booking_ref == ref)) is not None:
                continue
            session.add(
                Reservation(
                    booking_ref=ref,
                    customer_name=name,
                    car_plate=plate,
                    date=today + dt.timedelta(days=offset),
                    start_time=start,
                    end_time=end,
                    space_type=stype,
                    status=STATUS_CONFIRMED,
                    created_at=dt.datetime.now(),
                )
            )
            session.execute(
                update(ParkingSpace)
                .where(ParkingSpace.space_type == stype, ParkingSpace.available > 0)
                .values(available=ParkingSpace.available - 1)
            )
            added["reservations"] += 1

    return added


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Seed the CityPark Central database.")
    parser.add_argument("--reset", action="store_true", help="Drop all tables first.")
    args = parser.parse_args(argv)

    configure_logging()
    store = SqlStore.from_url(get_settings().sqlite_url)
    if args.reset:
        logger.warning("Resetting database before seeding")
        store.reset_db()
    summary = seed(store)
    logger.info("Seed complete, rows added: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())