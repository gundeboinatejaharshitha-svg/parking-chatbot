"""SQLAlchemy models and dynamic data access for CityPark Central.

Dynamic data (availability, prices, working hours, reservations) lives here.
The LLM never queries this module directly: the orchestrator calls it and
passes only the results to the language model, never reservation rows.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import secrets
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass

from sqlalchemy import (
    CheckConstraint,
    Engine,
    Numeric,
    String,
    case,
    create_engine,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Business constants (single source of truth, mirrored in the static docs)
# --------------------------------------------------------------------------
SPACE_TYPES: tuple[str, ...] = ("standard", "compact", "ev", "disabled")
STATUS_CONFIRMED = "confirmed"
STATUS_CANCELLED = "cancelled"
FREE_CANCELLATION_WINDOW = dt.timedelta(hours=24)
LATE_EXIT_GRACE_PERIOD = dt.timedelta(minutes=15)
BOOKING_REF_PREFIX = "CP-"
NOT_FOUND_MESSAGE = "No reservation matches the booking reference and car plate provided."
DAY_NAMES: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

_MAX_REF_ATTEMPTS = 10
_PLATE_PATTERN = re.compile(r"^[A-Z0-9]{2,8}$")


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------
class SqlStoreError(Exception):
    """Base class for all data-layer errors."""


class NoAvailabilityError(SqlStoreError):
    """Raised when no space of the requested type is available."""


class ReservationNotFoundError(SqlStoreError):
    """Raised for an unknown reference OR a wrong plate (identical message)."""

    def __init__(self) -> None:
        super().__init__(NOT_FOUND_MESSAGE)


class InvalidReservationError(SqlStoreError):
    """Raised when reservation input fails validation."""


class UnknownSpaceTypeError(InvalidReservationError):
    """Raised when a space type is not one of SPACE_TYPES."""


class CancellationNotAllowedError(SqlStoreError):
    """Raised after identity is verified but the booking cannot be cancelled."""


# --------------------------------------------------------------------------
# Helpers (pure functions, easy to unit test)
# --------------------------------------------------------------------------
def normalize_plate(plate: str) -> str:
    """Uppercase a plate and strip spaces and hyphens."""
    return re.sub(r"[\s-]", "", plate).upper()


def mask_plate(plate: str) -> str:
    """Mask a plate for logs, e.g. AB12CDE -> A*****E."""
    cleaned = normalize_plate(plate)
    if len(cleaned) < 4:
        return "***"
    return f"{cleaned[0]}{'*' * (len(cleaned) - 2)}{cleaned[-1]}"


def mask_name(name: str) -> str:
    """Mask a customer name for logs, e.g. Alice Morgan -> A***."""
    cleaned = name.strip()
    return f"{cleaned[0]}***" if cleaned else "***"


def mask_ref(ref: str) -> str:
    """Mask a booking reference for logs, e.g. CP-123456 -> CP-****56."""
    cleaned = ref.strip().upper()
    if len(cleaned) < 8:
        return "***"
    return f"{cleaned[:3]}****{cleaned[-2:]}"


def is_overstay(booked_end: dt.datetime, at: dt.datetime) -> bool:
    """Return True if `at` is later than the booked end plus the grace period."""
    return at > booked_end + LATE_EXIT_GRACE_PERIOD


# --------------------------------------------------------------------------
# ORM models
# --------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Declarative base for all tables."""


class ParkingSpace(Base):
    """Live availability per space type."""

    __tablename__ = "parking_spaces"
    __table_args__ = (
        CheckConstraint(
            "available >= 0 AND available <= total", name="ck_spaces_available_range"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    space_type: Mapped[str] = mapped_column(String(20), unique=True)
    total: Mapped[int]
    available: Mapped[int]


class Pricing(Base):
    """Price list per space type."""

    __tablename__ = "pricing"

    id: Mapped[int] = mapped_column(primary_key=True)
    space_type: Mapped[str] = mapped_column(String(20), unique=True)
    price_per_hour: Mapped[float] = mapped_column(Numeric(8, 2, asdecimal=False))
    price_per_day: Mapped[float] = mapped_column(Numeric(8, 2, asdecimal=False))


class WorkingHour(Base):
    """Opening hours per weekday (0 = Monday ... 6 = Sunday)."""

    __tablename__ = "working_hours"

    id: Mapped[int] = mapped_column(primary_key=True)
    day_of_week: Mapped[int] = mapped_column(unique=True)
    open_time: Mapped[dt.time | None]
    close_time: Mapped[dt.time | None]
    is_closed: Mapped[bool] = mapped_column(default=False)


class Reservation(Base):
    """A single-day parking reservation."""

    __tablename__ = "reservations"
    __table_args__ = (
        CheckConstraint(
            f"status IN ('{STATUS_CONFIRMED}', '{STATUS_CANCELLED}')",
            name="ck_reservations_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    booking_ref: Mapped[str] = mapped_column(String(12), unique=True, index=True)
    customer_name: Mapped[str] = mapped_column(String(100))
    car_plate: Mapped[str] = mapped_column(String(10), index=True)
    date: Mapped[dt.date]
    start_time: Mapped[dt.time]
    end_time: Mapped[dt.time]
    space_type: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(12), default=STATUS_CONFIRMED)
    created_at: Mapped[dt.datetime]


# --------------------------------------------------------------------------
# Plain return objects (no ORM objects leave this module)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SpaceAvailability:
    """Availability of one space type."""

    space_type: str
    total: int
    available: int


@dataclass(frozen=True)
class PriceInfo:
    """Prices of one space type."""

    space_type: str
    price_per_hour: float
    price_per_day: float


@dataclass(frozen=True)
class WorkingHoursInfo:
    """Opening hours of one weekday."""

    day_of_week: int
    day_name: str
    open_time: dt.time | None
    close_time: dt.time | None
    is_closed: bool


@dataclass(frozen=True)
class ReservationInfo:
    """A reservation as seen by its verified owner."""

    booking_ref: str
    customer_name: str
    car_plate: str
    date: dt.date
    start_time: dt.time
    end_time: dt.time
    space_type: str
    status: str
    created_at: dt.datetime


@dataclass(frozen=True)
class CancellationResult:
    """Outcome of a successful cancellation."""

    booking_ref: str
    refund_eligible: bool
    hours_before_start: float


# --------------------------------------------------------------------------
# Engine factory
# --------------------------------------------------------------------------
def create_db_engine(url: str) -> Engine:
    """Create an engine; handles SQLite threading and in-memory test databases."""
    if not url.startswith("sqlite"):
        return create_engine(url)
    if url in ("sqlite://", "sqlite:///:memory:"):
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    return create_engine(url, connect_args={"check_same_thread": False})


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------
class SqlStore:
    """Data access for dynamic parking data. Engine and clock are injected."""

    def __init__(
        self,
        engine: Engine,
        now: Callable[[], dt.datetime] | None = None,
    ) -> None:
        self._engine = engine
        self._now = now or dt.datetime.now
        self._session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    @classmethod
    def from_url(
        cls, url: str, now: Callable[[], dt.datetime] | None = None
    ) -> SqlStore:
        """Build a store from a SQLAlchemy URL."""
        return cls(create_db_engine(url), now=now)

    # ---- schema / admin ---------------------------------------------------
    def init_db(self) -> None:
        """Create all tables if they do not exist."""
        Base.metadata.create_all(self._engine)

    def reset_db(self) -> None:
        """Drop and recreate all tables (destroys data; used by seed --reset)."""
        Base.metadata.drop_all(self._engine)
        Base.metadata.create_all(self._engine)

    def transaction(self) -> AbstractContextManager[Session]:
        """Open a transaction for seeding/admin tasks (never used by the chatbot)."""
        return self._session_factory.begin()

    # ---- validation helpers ----------------------------------------------
    @staticmethod
    def _validate_space_type(raw: str) -> str:
        normalized = raw.strip().lower()
        if normalized not in SPACE_TYPES:
            raise UnknownSpaceTypeError(
                f"Unknown space type. Valid types: {', '.join(SPACE_TYPES)}."
            )
        return normalized

    @staticmethod
    def _parse_day(day: int | str) -> int:
        if isinstance(day, int):
            if 0 <= day <= 6:
                return day
        else:
            key = day.strip().lower()
            for index, name in enumerate(DAY_NAMES):
                if key in (name.lower(), name.lower()[:3]):
                    return index
        raise ValueError("Day must be 0 (Monday) to 6 (Sunday) or a weekday name.")

    @staticmethod
    def _to_info(row: Reservation) -> ReservationInfo:
        return ReservationInfo(
            booking_ref=row.booking_ref,
            customer_name=row.customer_name,
            car_plate=row.car_plate,
            date=row.date,
            start_time=row.start_time,
            end_time=row.end_time,
            space_type=row.space_type,
            status=row.status,
            created_at=row.created_at,
        )

    @staticmethod
    def _generate_booking_ref() -> str:
        return f"{BOOKING_REF_PREFIX}{secrets.randbelow(1_000_000):06d}"

    # ---- read-only dynamic data -------------------------------------------
    def get_availability(self, space_type: str | None = None) -> list[SpaceAvailability]:
        """Return live availability, optionally for one space type."""
        stmt = select(ParkingSpace).order_by(ParkingSpace.id)
        if space_type is not None:
            stmt = stmt.where(
                ParkingSpace.space_type == self._validate_space_type(space_type)
            )
        with self._session_factory() as session:
            rows = session.scalars(stmt).all()
        return [SpaceAvailability(r.space_type, r.total, r.available) for r in rows]

    def get_prices(self, space_type: str | None = None) -> list[PriceInfo]:
        """Return prices, optionally for one space type."""
        stmt = select(Pricing).order_by(Pricing.id)
        if space_type is not None:
            stmt = stmt.where(
                Pricing.space_type == self._validate_space_type(space_type)
            )
        with self._session_factory() as session:
            rows = session.scalars(stmt).all()
        return [PriceInfo(r.space_type, r.price_per_hour, r.price_per_day) for r in rows]

    def get_working_hours(self, day: int | str | None = None) -> list[WorkingHoursInfo]:
        """Return working hours for all days or one day (0-6 or weekday name)."""
        stmt = select(WorkingHour).order_by(WorkingHour.day_of_week)
        if day is not None:
            stmt = stmt.where(WorkingHour.day_of_week == self._parse_day(day))
        with self._session_factory() as session:
            rows = session.scalars(stmt).all()
        return [
            WorkingHoursInfo(
                day_of_week=r.day_of_week,
                day_name=DAY_NAMES[r.day_of_week],
                open_time=r.open_time,
                close_time=r.close_time,
                is_closed=r.is_closed,
            )
            for r in rows
        ]

    def is_open(self, date: dt.date, start: dt.time, end: dt.time) -> bool:
        """Return True if [start, end] on `date` falls inside working hours."""
        if end <= start:
            return False
        with self._session_factory() as session:
            row = session.scalar(
                select(WorkingHour).where(WorkingHour.day_of_week == date.weekday())
            )
        if row is None or row.is_closed or row.open_time is None or row.close_time is None:
            return False
        return row.open_time <= start and end <= row.close_time

    # ---- reservations -----------------------------------------------------
    def create_reservation(
        self,
        *,
        customer_name: str,
        car_plate: str,
        date: dt.date,
        start_time: dt.time,
        end_time: dt.time,
        space_type: str,
    ) -> ReservationInfo:
        """Validate and atomically create a reservation.

        The availability check and decrement are one conditional UPDATE inside
        the same transaction as the INSERT, so concurrent requests cannot
        oversell a space type.
        """
        name = customer_name.strip()
        if not 2 <= len(name) <= 100:
            raise InvalidReservationError("Name must be between 2 and 100 characters.")
        plate = normalize_plate(car_plate)
        if not _PLATE_PATTERN.match(plate):
            raise InvalidReservationError("Car plate must be 2-8 letters or digits.")
        stype = self._validate_space_type(space_type)
        if end_time <= start_time:
            raise InvalidReservationError(
                "End time must be after start time (single-day reservations only)."
            )
        now = self._now()
        if dt.datetime.combine(date, start_time) <= now:
            raise InvalidReservationError("The reservation must start in the future.")
        if not self.is_open(date, start_time, end_time):
            raise InvalidReservationError("The requested time is outside working hours.")

        for _ in range(_MAX_REF_ATTEMPTS):
            ref = self._generate_booking_ref()
            try:
                with self._session_factory.begin() as session:
                    claimed = session.execute(
                        update(ParkingSpace)
                        .where(
                            ParkingSpace.space_type == stype,
                            ParkingSpace.available > 0,
                        )
                        .values(available=ParkingSpace.available - 1)
                    )
                    if claimed.rowcount != 1:
                        raise NoAvailabilityError(
                            f"No {stype} spaces are currently available."
                        )
                    row = Reservation(
                        booking_ref=ref,
                        customer_name=name,
                        car_plate=plate,
                        date=date,
                        start_time=start_time,
                        end_time=end_time,
                        space_type=stype,
                        status=STATUS_CONFIRMED,
                        created_at=now,
                    )
                    session.add(row)
                    session.flush()  # surfaces a duplicate booking_ref here
                    info = self._to_info(row)
            except IntegrityError:
                logger.warning("Booking reference collision, retrying: %s", mask_ref(ref))
                continue
            logger.info(
                "Reservation created: ref=%s plate=%s name=%s type=%s date=%s",
                mask_ref(ref),
                mask_plate(plate),
                mask_name(name),
                stype,
                date,
            )
            return info
        raise SqlStoreError("Could not generate a unique booking reference.")

    def get_reservation_for_user(self, booking_ref: str, car_plate: str) -> ReservationInfo:
        """Return a reservation only if reference AND plate both match.

        The query filters on both values, so another user's row is never loaded.
        """
        ref = booking_ref.strip().upper()
        plate = normalize_plate(car_plate)
        with self._session_factory() as session:
            row = session.scalar(
                select(Reservation).where(
                    Reservation.booking_ref == ref,
                    Reservation.car_plate == plate,
                )
            )
            info = self._to_info(row) if row is not None else None
        if info is None:
            logger.info("Lookup failed: ref=%s plate=%s", mask_ref(ref), mask_plate(plate))
            raise ReservationNotFoundError()
        return info

    def cancel_reservation(self, booking_ref: str, car_plate: str) -> CancellationResult:
        """Cancel a reservation before it starts and restore availability.

        Wrong plate and unknown reference raise the same ReservationNotFoundError.
        Refund eligibility: at least 24 hours before the start time.
        """
        ref = booking_ref.strip().upper()
        plate = normalize_plate(car_plate)
        now = self._now()

        with self._session_factory.begin() as session:
            row = session.scalar(select(Reservation).where(Reservation.booking_ref == ref))
            if row is None or row.car_plate != plate:
                logger.info(
                    "Cancellation rejected: ref=%s plate=%s", mask_ref(ref), mask_plate(plate)
                )
                raise ReservationNotFoundError()
            if row.status != STATUS_CONFIRMED:
                raise CancellationNotAllowedError("This reservation is already cancelled.")
            start_at = dt.datetime.combine(row.date, row.start_time)
            if now >= start_at:
                raise CancellationNotAllowedError(
                    "A reservation cannot be cancelled once it has started."
                )
            cancelled = session.execute(
                update(Reservation)
                .where(Reservation.id == row.id, Reservation.status == STATUS_CONFIRMED)
                .values(status=STATUS_CANCELLED)
            )
            if cancelled.rowcount != 1:  # lost a race with another cancel
                raise CancellationNotAllowedError("This reservation is already cancelled.")
            session.execute(
                update(ParkingSpace)
                .where(ParkingSpace.space_type == row.space_type)
                .values(
                    available=case(
                        (ParkingSpace.available < ParkingSpace.total, ParkingSpace.available + 1),
                        else_=ParkingSpace.available,
                    )
                )
            )
            lead = start_at - now

        logger.info("Reservation cancelled: ref=%s", mask_ref(ref))
        return CancellationResult(
            booking_ref=ref,
            refund_eligible=lead >= FREE_CANCELLATION_WINDOW,
            hours_before_start=round(lead.total_seconds() / 3600, 2),
        )


__all__ = [
    "BOOKING_REF_PREFIX",
    "FREE_CANCELLATION_WINDOW",
    "LATE_EXIT_GRACE_PERIOD",
    "SPACE_TYPES",
    "Base",
    "CancellationNotAllowedError",
    "CancellationResult",
    "InvalidReservationError",
    "NoAvailabilityError",
    "ParkingSpace",
    "PriceInfo",
    "Pricing",
    "Reservation",
    "ReservationInfo",
    "ReservationNotFoundError",
    "SpaceAvailability",
    "SqlStore",
    "SqlStoreError",
    "UnknownSpaceTypeError",
    "WorkingHour",
    "WorkingHoursInfo",
    "create_db_engine",
    "is_overstay",
    "mask_name",
    "mask_plate",
    "mask_ref",
    "normalize_plate",
]