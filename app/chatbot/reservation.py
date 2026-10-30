"""Reservation workflow and validation logic.

Three small state machines drive the conversational side of reservations:
``BookingFlow`` (collect details -> confirm -> create), ``CancelFlow`` and
``LookupFlow`` (reference + plate -> verify -> act). Every value is parsed and
validated in code; the LLM is never involved and nothing here is "confirmed"
unless the matching ``SqlStore`` call succeeded.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from app.chatbot.parsing import (
    parse_booking_ref,
    parse_date,
    parse_name,
    parse_plate,
    parse_space_type,
    parse_time,
    parse_time_range,
    parse_yes_no,
)
from app.data.sql_store import (
    SPACE_TYPES,
    CancellationNotAllowedError,
    CancellationResult,
    InvalidReservationError,
    NoAvailabilityError,
    ReservationInfo,
    ReservationNotFoundError,
    SqlStore,
    SqlStoreError,
    mask_plate,
    mask_ref,
)

logger = logging.getLogger(__name__)

ABORT_WORDS = frozenset({"cancel", "stop", "quit", "exit", "abort", "never mind", "nevermind"})
ABORT_MESSAGE = "No problem, I've stopped. Ask me anything else whenever you're ready."
SPACE_TYPE_LABELS = {
    "standard": "standard",
    "compact": "compact",
    "ev": "EV charging",
    "disabled": "disabled (blue badge)",
}


@dataclass(frozen=True)
class FlowReply:
    """What a flow says back, whether it has finished, and any SQL outcome."""

    text: str
    done: bool = False
    outcome: ReservationInfo | CancellationResult | None = None


class Flow(Protocol):
    """A multi-turn conversation handled entirely in code."""

    @property
    def name(self) -> str: ...

    def start(self, text: str) -> FlowReply: ...

    def handle(self, text: str) -> FlowReply: ...


def _fmt_time(value: dt.time) -> str:
    return value.strftime("%H:%M")


def _fmt_date(value: dt.date) -> str:
    return value.strftime("%A %d %B %Y")


def _is_abort(text: str) -> bool:
    lowered = text.strip().lower().rstrip("!.")
    return lowered in ABORT_WORDS or lowered.startswith(("cancel this", "stop this", "forget it"))


# --------------------------------------------------------------------------
# Booking
# --------------------------------------------------------------------------
class BookingStep(str, Enum):
    """Order in which booking details are collected."""

    DATE = "date"
    START = "start"
    END = "end"
    SPACE_TYPE = "space_type"
    NAME = "name"
    PLATE = "plate"
    CONFIRM = "confirm"


@dataclass
class BookingDraft:
    """Details collected so far; only complete drafts reach ``create_reservation``."""

    date: dt.date | None = None
    start_time: dt.time | None = None
    end_time: dt.time | None = None
    space_type: str | None = None
    customer_name: str | None = None
    car_plate: str | None = None

    def missing(self) -> BookingStep:
        """The first step whose value is still missing (CONFIRM when complete)."""
        for step, value in (
            (BookingStep.DATE, self.date),
            (BookingStep.START, self.start_time),
            (BookingStep.END, self.end_time),
            (BookingStep.SPACE_TYPE, self.space_type),
            (BookingStep.NAME, self.customer_name),
            (BookingStep.PLATE, self.car_plate),
        ):
            if value is None:
                return step
        return BookingStep.CONFIRM


@dataclass
class BookingFlow:
    """Collects booking details step by step, validating each against the SQL layer."""

    store: SqlStore
    now: Callable[[], dt.datetime] = dt.datetime.now
    draft: BookingDraft = field(default_factory=BookingDraft)

    name = "booking"

    # ---- public --------------------------------------------------------------
    def start(self, text: str) -> FlowReply:
        """Pre-fill anything mentioned in the opening message, then ask for the rest."""
        today = self.now().date()
        date = parse_date(text, today)
        if date is not None and self._set_date(date) is None:
            time_range = parse_time_range(text)
            if time_range is not None:
                if self._set_start(time_range[0]) is None:
                    self._set_end(time_range[1])
        space_type = parse_space_type(text)
        if space_type is not None:
            self._set_space_type(space_type)
        return FlowReply(self._prompt())

    def handle(self, text: str) -> FlowReply:
        """Consume one user message for the current step."""
        if _is_abort(text):
            logger.info("Booking flow aborted at step %s", self.draft.missing().value)
            return FlowReply(ABORT_MESSAGE, done=True)

        step = self.draft.missing()
        if step is BookingStep.CONFIRM:
            return self._confirm(text)

        error = self._apply(step, text)
        if error:
            return FlowReply(f"{error} {self._prompt()}")
        return FlowReply(self._prompt())

    # ---- step handling -------------------------------------------------------
    def _apply(self, step: BookingStep, text: str) -> str | None:
        """Parse and validate ``text`` for ``step``; return an error message or None."""
        if step is BookingStep.DATE:
            date = parse_date(text, self.now().date())
            if date is None:
                return "I couldn't read that date (try 'tomorrow', 'Friday' or 2030-01-15)."
            return self._set_date(date)
        if step is BookingStep.START:
            time_range = parse_time_range(text)
            if time_range is not None:
                error = self._set_start(time_range[0])
                return error or self._set_end(time_range[1])
            start = parse_time(text)
            if start is None:
                return "I couldn't read that time (try '9am' or '14:30')."
            return self._set_start(start)
        if step is BookingStep.END:
            end = parse_time(text)
            if end is None:
                return "I couldn't read that time (try '11am' or '17:00')."
            return self._set_end(end)
        if step is BookingStep.SPACE_TYPE:
            space_type = parse_space_type(text)
            if space_type is None:
                return "Please choose one of: standard, compact, EV or disabled."
            return self._set_space_type(space_type)
        if step is BookingStep.NAME:
            name = parse_name(text)
            if name is None:
                return "Please give the name for the booking (2-100 letters)."
            self.draft.customer_name = name
            return None
        if step is BookingStep.PLATE:
            plate = parse_plate(text)
            if plate is None:
                return "Please give the vehicle plate (2-8 letters and digits, e.g. AB12CDE)."
            self.draft.car_plate = plate
            return None
        return None

    def _set_date(self, date: dt.date) -> str | None:
        today = self.now().date()
        if date < today:
            return "That date is in the past."
        hours = self.store.get_working_hours(date.weekday())
        if not hours or hours[0].is_closed:
            return f"We're closed on {date.strftime('%A')}s. Please pick another date."
        self.draft.date = date
        self.draft.start_time = self.draft.end_time = None
        return None

    def _set_start(self, start: dt.time) -> str | None:
        assert self.draft.date is not None
        if dt.datetime.combine(self.draft.date, start) <= self.now():
            return "That start time has already passed."
        hours = self.store.get_working_hours(self.draft.date.weekday())[0]
        if hours.open_time and hours.close_time:
            if not hours.open_time <= start < hours.close_time:
                return self._hours_hint(self.draft.date)
        self.draft.start_time = start
        self.draft.end_time = None
        return None

    def _set_end(self, end: dt.time) -> str | None:
        assert self.draft.date is not None and self.draft.start_time is not None
        if end <= self.draft.start_time:
            return f"The end time must be after {_fmt_time(self.draft.start_time)}."
        if not self.store.is_open(self.draft.date, self.draft.start_time, end):
            return self._hours_hint(self.draft.date)
        self.draft.end_time = end
        return None

    def _set_space_type(self, space_type: str) -> str | None:
        availability = self.store.get_availability(space_type)[0]
        if availability.available <= 0:
            alternatives = [
                SPACE_TYPE_LABELS[a.space_type]
                for a in self.store.get_availability()
                if a.available > 0 and a.space_type != space_type
            ]
            hint = f" Available now: {', '.join(alternatives)}." if alternatives else ""
            label = SPACE_TYPE_LABELS[space_type]
            return f"Sorry, no {label} spaces are available right now.{hint}"
        self.draft.space_type = space_type
        return None

    def _hours_hint(self, date: dt.date) -> str:
        hours = self.store.get_working_hours(date.weekday())[0]
        if hours.is_closed or hours.open_time is None or hours.close_time is None:
            return f"We're closed on {date.strftime('%A')}s."
        return (
            f"That's outside our opening hours on {date.strftime('%A')} "
            f"({_fmt_time(hours.open_time)}-{_fmt_time(hours.close_time)})."
        )

    # ---- prompts and confirmation --------------------------------------------
    def _prompt(self) -> str:
        step = self.draft.missing()
        draft = self.draft
        if step is BookingStep.DATE:
            return "Which date would you like to park? (e.g. 'tomorrow', 'Friday', 2030-01-15)"
        if step is BookingStep.START:
            assert draft.date is not None
            hours = self.store.get_working_hours(draft.date.weekday())[0]
            span = (
                f" We're open {_fmt_time(hours.open_time)}-{_fmt_time(hours.close_time)} that day."
                if hours.open_time and hours.close_time
                else ""
            )
            return f"{_fmt_date(draft.date)}.{span} What start time? (or 'from 9 to 11')"
        if step is BookingStep.END:
            assert draft.start_time is not None
            return f"Starting at {_fmt_time(draft.start_time)}. What end time?"
        if step is BookingStep.SPACE_TYPE:
            options = ", ".join(SPACE_TYPE_LABELS[s] for s in SPACE_TYPES)
            return f"Which type of space do you need? ({options})"
        if step is BookingStep.NAME:
            return "What name should the booking be under?"
        if step is BookingStep.PLATE:
            return "What is the vehicle registration plate?"
        return self._summary() + " Shall I confirm this booking? (yes/no)"

    def _summary(self) -> str:
        d = self.draft
        assert d.date and d.start_time and d.end_time and d.space_type and d.customer_name
        assert d.car_plate
        return (
            f"Here's your booking: {SPACE_TYPE_LABELS[d.space_type]} space on {_fmt_date(d.date)} "
            f"from {_fmt_time(d.start_time)} to {_fmt_time(d.end_time)}, "
            f"for {d.customer_name}, plate {d.car_plate}."
        )

    def _confirm(self, text: str) -> FlowReply:
        decision = parse_yes_no(text)
        if decision is None:
            return FlowReply("Please answer yes to confirm or no to cancel. " + self._prompt())
        if not decision:
            logger.info("Booking declined at confirmation")
            return FlowReply("Okay, I haven't booked anything.", done=True)

        d = self.draft
        assert d.date and d.start_time and d.end_time and d.space_type and d.customer_name
        assert d.car_plate
        try:
            info = self.store.create_reservation(
                customer_name=d.customer_name,
                car_plate=d.car_plate,
                date=d.date,
                start_time=d.start_time,
                end_time=d.end_time,
                space_type=d.space_type,
            )
        except NoAvailabilityError:
            return FlowReply(
                f"Sorry, the last {SPACE_TYPE_LABELS[d.space_type]} space was just taken, "
                "so nothing was booked. Would you like to try another space type?",
                done=True,
            )
        except InvalidReservationError as exc:
            return FlowReply(
                f"I couldn't complete the booking: {exc} Nothing was booked.", done=True
            )
        except SqlStoreError:
            logger.exception("Reservation creation failed")
            return FlowReply(
                "Something went wrong while saving the booking, so nothing was booked. "
                "Please try again in a moment.",
                done=True,
            )

        logger.info(
            "Booking confirmed ref=%s plate=%s",
            mask_ref(info.booking_ref),
            mask_plate(info.car_plate),
        )
        label = SPACE_TYPE_LABELS[info.space_type]
        return FlowReply(
            f"Your booking is confirmed. Reference: {info.booking_ref}. "
            f"{label[0].upper()}{label[1:]} space on {_fmt_date(info.date)} "
            f"from {_fmt_time(info.start_time)} to {_fmt_time(info.end_time)}, "
            f"plate {info.car_plate}. "
            "Keep the reference and plate together; you'll need both to look up or cancel.",
            done=True,
            outcome=info,
        )


# --------------------------------------------------------------------------
# Lookup and cancel (reference + plate always required together)
# --------------------------------------------------------------------------
NOT_FOUND_REPLY = (
    "I couldn't find a reservation matching that booking reference and plate. "
    "Please check both and try again."
)
_CANCEL_WORD = re.compile(r"\bcancel", re.IGNORECASE)
_KEEP_WORD = re.compile(r"\b(don'?t|do not|not|keep|no)\b", re.IGNORECASE)


@dataclass
class _RefPlateFlow:
    """Shared collection of booking reference and plate."""

    store: SqlStore
    booking_ref: str | None = None
    car_plate: str | None = None

    def start(self, text: str) -> FlowReply:
        """Pre-fill reference/plate from the opening message, then continue."""
        self.booking_ref = parse_booking_ref(text)
        self.car_plate = parse_plate(text)
        return self._advance()

    def handle(self, text: str) -> FlowReply:
        """Consume one user message."""
        if _is_abort(text):
            return FlowReply(ABORT_MESSAGE, done=True)
        if self.booking_ref is None:
            ref = parse_booking_ref(text)
            if ref is None:
                return FlowReply("Please give the booking reference in the form CP-123456.")
            self.booking_ref = ref
            return self._advance()
        if self.car_plate is None:
            plate = parse_plate(text)
            if plate is None:
                return FlowReply("Please give the vehicle plate (e.g. AB12CDE).")
            self.car_plate = plate
            return self._advance()
        return self._after_identified(text)

    def _advance(self) -> FlowReply:
        if self.booking_ref is None:
            return FlowReply("What is the booking reference? (format CP-123456)")
        if self.car_plate is None:
            return FlowReply("And the vehicle registration plate?")
        return self._identified()

    def _fetch(self) -> ReservationInfo | None:
        assert self.booking_ref is not None and self.car_plate is not None
        try:
            return self.store.get_reservation_for_user(self.booking_ref, self.car_plate)
        except ReservationNotFoundError:
            return None

    def _identified(self) -> FlowReply:  # pragma: no cover - overridden
        raise NotImplementedError

    def _after_identified(self, text: str) -> FlowReply:  # pragma: no cover - overridden
        raise NotImplementedError


def _describe(info: ReservationInfo) -> str:
    return (
        f"{info.booking_ref}: {SPACE_TYPE_LABELS[info.space_type]} space on {_fmt_date(info.date)} "
        f"from {_fmt_time(info.start_time)} to {_fmt_time(info.end_time)}, plate {info.car_plate}, "
        f"status {info.status}."
    )


class LookupFlow(_RefPlateFlow):
    """Shows a reservation once reference and plate both match."""

    name = "lookup"

    def _identified(self) -> FlowReply:
        info = self._fetch()
        if info is None:
            return FlowReply(NOT_FOUND_REPLY, done=True)
        return FlowReply(f"Found it. {_describe(info)}", done=True, outcome=info)

    def _after_identified(self, text: str) -> FlowReply:  # pragma: no cover - lookup ends early
        return self._identified()


class CancelFlow(_RefPlateFlow):
    """Cancels a reservation after showing it and getting explicit confirmation."""

    name = "cancel"

    def handle(self, text: str) -> FlowReply:
        """Like the base flow, but "cancel it" at the confirmation step means yes."""
        identified = self.booking_ref is not None and self.car_plate is not None
        if identified and _CANCEL_WORD.search(text) and not _KEEP_WORD.search(text):
            return self._do_cancel()
        return super().handle(text)

    def _identified(self) -> FlowReply:
        info = self._fetch()
        if info is None:
            return FlowReply(NOT_FOUND_REPLY, done=True)
        if info.status != "confirmed":
            return FlowReply(f"{_describe(info)} It is already cancelled.", done=True)
        return FlowReply(
            f"{_describe(info)} Free cancellation applies 24 hours or more before the start; "
            "later cancellations are not refunded. Do you want to cancel it? (yes/no)"
        )

    def _after_identified(self, text: str) -> FlowReply:
        decision = parse_yes_no(text)
        if decision is None:
            return FlowReply("Please answer yes to cancel the booking or no to keep it.")
        if not decision:
            return FlowReply("Okay, your booking is unchanged.", done=True)
        return self._do_cancel()

    def _do_cancel(self) -> FlowReply:
        assert self.booking_ref is not None and self.car_plate is not None
        try:
            result = self.store.cancel_reservation(self.booking_ref, self.car_plate)
        except ReservationNotFoundError:
            return FlowReply(NOT_FOUND_REPLY, done=True)
        except CancellationNotAllowedError as exc:
            return FlowReply(f"I couldn't cancel this booking: {exc}", done=True)
        refund = (
            "You're eligible for a full refund."
            if result.refund_eligible
            else "As this is less than 24 hours before the start, no refund is due."
        )
        return FlowReply(
            f"Booking {result.booking_ref} has been cancelled. {refund}", done=True, outcome=result
        )


__all__ = [
    "ABORT_MESSAGE",
    "NOT_FOUND_REPLY",
    "SPACE_TYPE_LABELS",
    "BookingDraft",
    "BookingFlow",
    "BookingStep",
    "CancelFlow",
    "Flow",
    "FlowReply",
    "LookupFlow",
]
