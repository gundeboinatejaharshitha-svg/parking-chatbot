"""Deterministic parsers for user input (dates, times, plates, references...).

Everything here is pure and rule-based so the reservation flow never depends on
the LLM to interpret booking details. Each parser returns ``None`` when it
cannot confidently extract a value; callers then re-prompt the user.
"""

from __future__ import annotations

import datetime as dt
import re

from app.data.sql_store import SPACE_TYPES, normalize_plate

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
# Full name, 3-letter abbreviation, or common 4/5-letter forms (tues, weds, thurs).
_WEEKDAY_PATTERNS = tuple(
    re.compile(rf"\b{name[:3]}(?:{name[3:]}|s|{name[3:4]}s)?\b") for name in WEEKDAYS
)
MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)  # fmt: skip

_SPACE_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "standard": ("standard", "regular", "normal", "ordinary"),
    "compact": ("compact", "small car", "small"),
    "ev": ("ev", "electric", "charging", "charger", "e-vehicle"),
    "disabled": ("disabled", "accessible", "blue badge", "wheelchair", "accessibility"),
}
_YES = {"y", "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "correct", "go ahead"}
_NO = {"n", "no", "nope", "nah", "cancel", "stop", "abort", "quit", "exit", "never mind"}

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DMY_DATE = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})(?:[/.-](\d{2,4}))?\b")
_DAY_MONTH = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]+)(?:\s+(\d{4}))?\b"
)
_MONTH_DAY = re.compile(r"\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b")
_TIME = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\b", re.IGNORECASE)
_TIME_RANGE = re.compile(
    r"(?:from\s+|between\s+)?"
    r"(\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm)?)"
    r"\s*(?:-|–|to|until|till|and)\s*"
    r"(\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm)?)",
    re.IGNORECASE,
)
_BOOKING_REF = re.compile(r"\bCP\s*-?\s*(\d{6})\b", re.IGNORECASE)
_PLATE_FULL = re.compile(r"^[A-Z0-9]{2,8}$")
_PLATE_STOPWORDS = frozenset(
    "A AN THE MY IS IT ITS ON IN AT TO OF AND FOR NO OK OKAY CAR REG PLATE NUMBER VEHICLE "
    "REGISTRATION LICENSE LICENCE".split()
)
_NAME_PREFIX = re.compile(r"^(?:my name is|i am|i'm|it's|its|this is|name:?)\s+", re.IGNORECASE)
_NAME_ALLOWED = re.compile(r"^[A-Za-z][A-Za-z' .-]{0,98}[A-Za-z.]$")


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------
def _safe_date(year: int, month: int, day: int) -> dt.date | None:
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def _month_from_name(name: str) -> int | None:
    name = name.lower()
    for index, month in enumerate(MONTHS, start=1):
        if name == month or (len(name) >= 3 and month.startswith(name)):
            return index
    return None


def _next_weekday(today: dt.date, weekday: int, *, skip_today: bool) -> dt.date:
    delta = (weekday - today.weekday()) % 7
    if delta == 0 and skip_today:
        delta = 7
    return today + dt.timedelta(days=delta)


def parse_date(text: str, today: dt.date) -> dt.date | None:
    """Extract a calendar date from free text.

    Supports ISO (2030-01-09), day-first numeric (9/1/2030, 9-1), "today",
    "tomorrow", weekday names ("next friday" = the following week's Friday) and
    "9 January" / "January 9" (year defaults to the next occurrence).
    """
    lowered = text.lower().strip()
    if not lowered:
        return None

    if "day after tomorrow" in lowered:
        return today + dt.timedelta(days=2)
    if "tomorrow" in lowered:
        return today + dt.timedelta(days=1)
    if "today" in lowered or lowered == "tonight":
        return today

    match = _ISO_DATE.search(lowered)
    if match:
        year, month, day = (int(group) for group in match.groups())
        return _safe_date(year, month, day)

    match = _DMY_DATE.search(lowered)
    if match:
        day, month = int(match.group(1)), int(match.group(2))
        year_text = match.group(3)
        if year_text:
            year = int(year_text)
            year = year + 2000 if year < 100 else year
            return _safe_date(year, month, day)
        candidate = _safe_date(today.year, month, day)
        if candidate is None:
            return None
        return candidate if candidate >= today else _safe_date(today.year + 1, month, day)

    for weekday_index, pattern in enumerate(_WEEKDAY_PATTERNS):
        if pattern.search(lowered):
            name = WEEKDAYS[weekday_index]
            is_next = re.search(rf"\bnext\s+{name[:3]}", lowered) is not None
            date = _next_weekday(today, weekday_index, skip_today=True)
            if is_next and date - today < dt.timedelta(7):
                date += dt.timedelta(days=7)
            return date

    for pattern, day_group, month_group in ((_DAY_MONTH, 1, 2), (_MONTH_DAY, 2, 1)):
        for match in pattern.finditer(lowered):
            month = _month_from_name(match.group(month_group))
            if month is None:
                continue
            day = int(match.group(day_group))
            year_text = match.group(3)
            if year_text:
                return _safe_date(int(year_text), month, day)
            candidate = _safe_date(today.year, month, day)
            if candidate is None:
                return None
            return candidate if candidate >= today else _safe_date(today.year + 1, month, day)
    return None


# --------------------------------------------------------------------------
# Times
# --------------------------------------------------------------------------
def _to_time(hour_text: str, minute_text: str | None, meridiem: str | None) -> dt.time | None:
    hour = int(hour_text)
    minute = int(minute_text) if minute_text else 0
    if meridiem:
        meridiem = meridiem.lower().replace(".", "")
        if not 1 <= hour <= 12:
            return None
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return dt.time(hour, minute)


def parse_time(text: str) -> dt.time | None:
    """Extract a single clock time ("9", "9:30", "9.30pm", "14:00", "noon")."""
    lowered = text.lower().strip()
    if lowered in {"noon", "midday", "12 noon"}:
        return dt.time(12, 0)
    if lowered == "midnight":
        return dt.time(0, 0)
    match = _TIME.search(lowered)
    if not match:
        return None
    return _to_time(match.group(1), match.group(2), match.group(3))


def parse_time_range(text: str) -> tuple[dt.time, dt.time] | None:
    """Extract "from 9 to 11", "9am-5pm", "between 10 and 12" as (start, end)."""
    match = _TIME_RANGE.search(text)
    if not match:
        return None
    start_text, end_text = match.group(1), match.group(2)
    start, end = parse_time(start_text), parse_time(end_text)
    if start is None or end is None:
        return None
    # "9 to 5pm": try inheriting the end's am/pm for the start when it helps.
    end_meridiem = re.search(r"(am|pm)", end_text, re.IGNORECASE)
    if end_meridiem and not re.search(r"(am|pm)", start_text, re.IGNORECASE):
        inherited = parse_time(f"{start_text}{end_meridiem.group(1)}")
        if inherited is not None and inherited < end:
            start = inherited
    if end <= start and end_meridiem is None and end.hour < 12:
        # "9 to 5" with no am/pm: read the end as afternoon.
        end = dt.time(end.hour + 12, end.minute)
    return (start, end) if end > start else None


# --------------------------------------------------------------------------
# Booking details
# --------------------------------------------------------------------------
def parse_space_type(text: str) -> str | None:
    """Map wording like "electric", "accessible bay" to a canonical space type."""
    lowered = f" {text.lower()} "
    for space_type in SPACE_TYPES:
        for alias in _SPACE_TYPE_ALIASES[space_type]:
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                return space_type
    return None


def parse_booking_ref(text: str) -> str | None:
    """Extract a booking reference and normalise it to ``CP-XXXXXX``."""
    match = _BOOKING_REF.search(text)
    return f"CP-{match.group(1)}" if match else None


def parse_plate(text: str) -> str | None:
    """Extract a vehicle plate (2-8 letters/digits after removing spaces and hyphens).

    A message that is only a plate is accepted as-is; otherwise the first token
    group that looks like a plate is used. Plates must contain at least one
    digit so ordinary words are never mistaken for one.
    """
    whole = normalize_plate(text)
    if _is_plate(whole):
        return whole
    tokens = [t for t in re.split(r"[^A-Z0-9-]+", text.upper()) if t.strip("-")]
    for size in (2, 1, 3):
        for start in range(len(tokens) - size + 1):
            window = tokens[start : start + size]
            if window[0] in _PLATE_STOPWORDS or window[-1] in _PLATE_STOPWORDS:
                continue
            if size > 1 and not any(c.isdigit() for c in window[0] + window[-1]):
                continue
            candidate = normalize_plate("".join(window))
            if _is_plate(candidate) and not candidate.startswith("CP"):
                return candidate
    return None


def _is_plate(candidate: str) -> bool:
    return bool(
        _PLATE_FULL.match(candidate)
        and any(c.isdigit() for c in candidate)
        and any(c.isalpha() for c in candidate)
    )


def parse_name(text: str) -> str | None:
    """Extract a customer name ("my name is Alice Morgan" -> "Alice Morgan")."""
    cleaned = _NAME_PREFIX.sub("", text.strip()).strip().rstrip(".")
    if _NAME_ALLOWED.match(cleaned) and 2 <= len(cleaned) <= 100:
        return cleaned
    return None


def parse_yes_no(text: str) -> bool | None:
    """Return True for an affirmative, False for a negative, None if unclear."""
    lowered = re.sub(r"[^a-z' ]", " ", text.lower()).strip()
    if not lowered:
        return None
    if lowered in _YES or lowered.split()[0] in _YES:
        return True
    if lowered in _NO or lowered.split()[0] in _NO or "never mind" in lowered:
        return False
    return None


def parse_weekday(text: str, today: dt.date) -> int | None:
    """Return 0-6 for a weekday mentioned in the text, including today/tomorrow."""
    lowered = text.lower()
    if "tomorrow" in lowered:
        return (today.weekday() + 1) % 7
    if "today" in lowered or "tonight" in lowered:
        return today.weekday()
    for index, pattern in enumerate(_WEEKDAY_PATTERNS):
        if pattern.search(lowered):
            return index
    return None


__all__ = [
    "MONTHS",
    "WEEKDAYS",
    "parse_booking_ref",
    "parse_date",
    "parse_name",
    "parse_plate",
    "parse_space_type",
    "parse_time",
    "parse_time_range",
    "parse_weekday",
    "parse_yes_no",
]
