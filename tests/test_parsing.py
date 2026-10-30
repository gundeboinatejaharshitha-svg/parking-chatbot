"""Tests for the deterministic parsers in app.chatbot.parsing."""

from __future__ import annotations

import datetime as dt

import pytest

from app.chatbot.parsing import (
    parse_booking_ref,
    parse_date,
    parse_name,
    parse_plate,
    parse_space_type,
    parse_time,
    parse_time_range,
    parse_weekday,
    parse_yes_no,
)

TODAY = dt.date(2030, 1, 7)  # Monday


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tomorrow", dt.date(2030, 1, 8)),
        ("the day after tomorrow", dt.date(2030, 1, 9)),
        ("today please", TODAY),
        ("2030-01-15", dt.date(2030, 1, 15)),
        ("15/01/2030", dt.date(2030, 1, 15)),
        ("15/1", dt.date(2030, 1, 15)),
        ("3/1", dt.date(2031, 1, 3)),  # already passed this year -> next year
        ("friday", dt.date(2030, 1, 11)),
        ("next friday", dt.date(2030, 1, 18)),
        ("monday", dt.date(2030, 1, 14)),  # today is Monday -> next week
        ("tues", dt.date(2030, 1, 8)),
        ("12 March", dt.date(2030, 3, 12)),
        ("March 12th 2031", dt.date(2031, 3, 12)),
        ("book for 9 jan at 10", dt.date(2030, 1, 9)),
        ("I have no money", None),
        ("31/2", None),
        ("banana", None),
    ],
)
def test_parse_date(text: str, expected: dt.date | None) -> None:
    assert parse_date(text, TODAY) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("9", dt.time(9)),
        ("9am", dt.time(9)),
        ("9:30", dt.time(9, 30)),
        ("9.30pm", dt.time(21, 30)),
        ("14:00", dt.time(14)),
        ("12pm", dt.time(12)),
        ("12am", dt.time(0)),
        ("noon", dt.time(12)),
        ("13pm", None),
        ("25:00", None),
        ("soon", None),
    ],
)
def test_parse_time(text: str, expected: dt.time | None) -> None:
    assert parse_time(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("from 9 to 11", (dt.time(9), dt.time(11))),
        ("9am-5pm", (dt.time(9), dt.time(17))),
        ("9 to 5pm", (dt.time(9), dt.time(17))),
        ("between 10 and 12", (dt.time(10), dt.time(12))),
        ("9 to 5", (dt.time(9), dt.time(17))),
        ("10:00 until 12:30", (dt.time(10), dt.time(12, 30))),
        ("at 10", None),
    ],
)
def test_parse_time_range(text: str, expected: tuple[dt.time, dt.time] | None) -> None:
    assert parse_time_range(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("AB12CDE", "AB12CDE"),
        ("ab 12 cde", "AB12CDE"),
        ("my plate is AB12 CDE", "AB12CDE"),
        ("the car is te57-car", "TE57CAR"),
        ("CP-100001 and AB12CDE", "AB12CDE"),
        ("cancel CP-100001", None),
        ("hello", None),
        ("1234", None),
        ("ABCDEFGHIJ12", None),
    ],
)
def test_parse_plate(text: str, expected: str | None) -> None:
    assert parse_plate(text) == expected


def test_parse_booking_ref_normalises() -> None:
    assert parse_booking_ref("ref cp 100001 please") == "CP-100001"
    assert parse_booking_ref("CP-100002") == "CP-100002"
    assert parse_booking_ref("CP-12345") is None
    assert parse_booking_ref("no ref") is None


def test_parse_name() -> None:
    assert parse_name("Alice Morgan") == "Alice Morgan"
    assert parse_name("my name is Alice Morgan") == "Alice Morgan"
    assert parse_name("I'm Jean-Luc O'Neil.") == "Jean-Luc O'Neil"
    assert parse_name("x") is None
    assert parse_name("1234") is None
    assert parse_name("a" * 101) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("yes", True),
        ("Yes please!", True),
        ("confirm", True),
        ("no thanks", False),
        ("never mind", False),
        ("cancel", False),
        ("maybe", None),
        ("", None),
    ],
)
def test_parse_yes_no(text: str, expected: bool | None) -> None:
    assert parse_yes_no(text) == expected


def test_parse_space_type_and_weekday() -> None:
    assert parse_space_type("an electric bay") == "ev"
    assert parse_space_type("blue badge please") == "disabled"
    assert parse_space_type("Compact") == "compact"
    assert parse_space_type("standard space") == "standard"
    assert parse_space_type("a space") is None

    assert parse_weekday("open on sunday?", TODAY) == 6
    assert parse_weekday("tomorrow", TODAY) == 1
    assert parse_weekday("today", TODAY) == 0
    assert parse_weekday("opening hours", TODAY) is None
