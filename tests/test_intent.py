"""Tests for the rule-based intent router."""

from __future__ import annotations

import pytest

from app.chatbot.intent import ConversationalCue, Intent, classify, classify_cue


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


@pytest.mark.parametrize(
    ("last_topic", "text", "expected"),
    [
        (Intent.AVAILABILITY, "what about standard parking?", Intent.AVAILABILITY),
        (Intent.AVAILABILITY, "and compact?", Intent.AVAILABILITY),
        (Intent.PRICES, "what about EV?", Intent.PRICES),
        (Intent.HOURS, "what about friday?", Intent.HOURS),
        (Intent.HOURS, "and tomorrow?", Intent.HOURS),
    ],
)
def test_ambiguous_followup_reuses_last_sql_topic(
    last_topic: Intent, text: str, expected: Intent
) -> None:
    assert classify(text, last_topic) is expected


def test_followup_topic_is_ignored_without_a_resolvable_subtopic() -> None:
    assert classify("what about that", Intent.AVAILABILITY) is Intent.STATIC_QUESTION


def test_followup_topic_does_not_override_a_policy_question() -> None:
    assert classify("what is the refund policy for standard", Intent.AVAILABILITY) is (
        Intent.STATIC_QUESTION
    )


# ---- natural-language variations (regression for the "what are the ... available ..."
# false-policy-signal bug: _POLICY_SIGNAL's bare "what are the" / "what is the" phrases used
# to block availability and hours regardless of what followed them) ---------------------


@pytest.mark.parametrize(
    "text",
    [
        "What parking slots are available?",
        "What are the available slots for parking?",
        "Do you have any available parking spaces?",
        "Is parking available?",
        "Can I get a parking slot?",
        "Show me the parking availability.",
        "How many parking spaces are vacant?",
        "Are there any empty parking spots?",
        "I need to know the available parking spaces.",
        "What is the current parking capacity?",
        "Can you check available slots?",
    ],
)
def test_availability_phrasing_variations(text: str) -> None:
    assert classify(text) is Intent.AVAILABILITY


def test_original_bug_report_pair_classify_identically() -> None:
    # The exact pair from the bug report: both must resolve to the same intent.
    assert classify("What parking slots are available?") is Intent.AVAILABILITY
    assert classify("What are the available slots for parking?") is Intent.AVAILABILITY


@pytest.mark.parametrize(
    "text",
    [
        "I want to reserve a parking space.",
        "Can I book a parking slot?",
        "Help me park my car.",
        "I need parking for tomorrow.",
        "Make a reservation.",
    ],
)
def test_booking_phrasing_variations(text: str) -> None:
    assert classify(text) is Intent.BOOK


@pytest.mark.parametrize(
    "text",
    [
        "Find my reservation.",
        "Can you check my booking?",
        "I want to see my reservation details.",
        "Where is my booking confirmation?",
    ],
)
def test_lookup_phrasing_variations(text: str) -> None:
    assert classify(text) is Intent.LOOKUP


@pytest.mark.parametrize(
    "text",
    [
        "Cancel my parking reservation.",
        "I don't need my parking slot anymore.",
        "I want to cancel my booking.",
    ],
)
def test_cancel_phrasing_variations(text: str) -> None:
    assert classify(text) is Intent.CANCEL


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What are your parking charges?", Intent.PRICES),
        ("How much does parking cost?", Intent.PRICES),
        ("What are the parking rates?", Intent.PRICES),
        ("What time do you open?", Intent.HOURS),
        ("What are your operating hours?", Intent.HOURS),
    ],
)
def test_price_and_hours_phrasing_variations(text: str, expected: Intent) -> None:
    assert classify(text) is expected


@pytest.mark.parametrize(
    "text",
    [
        "Do you provide EV charging?",
        "Where are you located?",
    ],
)
def test_informational_questions_still_go_to_rag(text: str) -> None:
    # These describe a service/feature rather than asking for live SQL data or an
    # action, so they must keep going to RAG, not be swept up by the new keywords.
    assert classify(text) is Intent.STATIC_QUESTION


# ---- conversational cues (ack/gratitude/yes/no/more-info) -----------------------------


@pytest.mark.parametrize(
    "text",
    ["OK", "OK.", "Okay", "Alright", "Got it", "Understood", "Sure", "Fine"],
)
def test_acknowledgement_cue(text: str) -> None:
    assert classify_cue(text) is ConversationalCue.ACKNOWLEDGEMENT


@pytest.mark.parametrize("text", ["Thanks", "Thank you", "Thanks a lot", "thanks very much"])
def test_gratitude_cue(text: str) -> None:
    assert classify_cue(text) is ConversationalCue.GRATITUDE


@pytest.mark.parametrize("text", ["Yes", "Yeah", "Correct", "Absolutely"])
def test_affirmative_cue(text: str) -> None:
    assert classify_cue(text) is ConversationalCue.AFFIRMATIVE


@pytest.mark.parametrize("text", ["No", "Nope", "Not really", "I don't want to"])
def test_negative_cue(text: str) -> None:
    assert classify_cue(text) is ConversationalCue.NEGATIVE


@pytest.mark.parametrize(
    "text",
    [
        "Tell me more",
        "Explain that",
        "Can you elaborate?",
        "What else?",
        "Give me more details",
        "Can you explain that in simple terms?",
    ],
)
def test_more_info_cue(text: str) -> None:
    assert classify_cue(text) is ConversationalCue.MORE_INFO


@pytest.mark.parametrize(
    "text",
    [
        "What parking slots are available?",
        "Can I cancel my booking, yes or no?",
        "asdkjhasdkjh random nonsense",
        "",
        "   ",
    ],
)
def test_real_questions_and_nonsense_are_not_cues(text: str) -> None:
    assert classify_cue(text) is None
