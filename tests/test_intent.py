"""Tests for the rule-based intent router."""

from __future__ import annotations

import pytest

from app.chatbot.intent import Intent, classify


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hi", Intent.GREETING),
        ("Hello there!", Intent.GREETING),
        ("thanks", Intent.GREETING),
        ("how many spaces are available", Intent.AVAILABILITY),
        ("is there any space right now", Intent.AVAILABILITY),
        ("any EV bays free?", Intent.AVAILABILITY),
        ("are you full", Intent.AVAILABILITY),
        ("how much does parking cost", Intent.PRICES),
        ("what is the price per hour for ev", Intent.PRICES),
        ("prices?", Intent.PRICES),
        ("what are your opening hours", Intent.HOURS),
        ("what time do you close on sunday", Intent.HOURS),
        ("are you open on sunday", Intent.HOURS),
        ("I want to book a space", Intent.BOOK),
        ("book a space for tomorrow 9 to 11", Intent.BOOK),
        ("can I make a reservation", Intent.BOOK),
        ("reserve an EV bay", Intent.BOOK),
        ("cancel my booking", Intent.CANCEL),
        ("I want to cancel CP-100001", Intent.CANCEL),
        ("can I cancel my booking", Intent.CANCEL),
        ("check my booking CP-100001", Intent.LOOKUP),
        ("show my reservation", Intent.LOOKUP),
        ("CP-100001 AB12CDE", Intent.LOOKUP),
        ("what's the status of my booking", Intent.LOOKUP),
    ],
)
def test_action_and_dynamic_intents(text: str, expected: Intent) -> None:
    assert classify(text) is expected


@pytest.mark.parametrize(
    "text",
    [
        "how late can I cancel for free",
        "what happens if I cancel late",
        "is there a fee for cancelling",
        "what is the refund policy",
        "how do I make a reservation",
        "what details do I need to book",
        "where is the entrance",
        "what is the height limit",
        "do you have EV chargers",
        "can I stay overnight",
        "are pets allowed",
        "is there a grace period",
        "what happens if I overstay",
        "I lost my booking reference",
        "where are the disabled bays",
        "which payment methods are accepted",
        "what is my booking reference",
        "",
    ],
)
def test_policy_and_general_questions_go_to_rag(text: str) -> None:
    assert classify(text) is Intent.STATIC_QUESTION


def test_booking_reference_forces_action_even_with_policy_words() -> None:
    assert classify("what is the refund if I cancel CP-100001") is Intent.CANCEL
