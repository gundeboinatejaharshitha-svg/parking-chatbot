"""Intent detection and routing logic.

Routing is rule-based on purpose: it is deterministic, testable offline, and it
keeps the decision of *which data source answers* (SQL vs. vector store vs. a
code-validated reservation flow) out of the language model's hands. Anything
that is not recognised falls through to ``STATIC_QUESTION`` where the RAG
layer either answers from the knowledge base or says it does not know.
"""

from __future__ import annotations

import re
from enum import Enum

from app.chatbot.parsing import parse_booking_ref


class Intent(str, Enum):
    """What the user wants; decides which backend handles the message."""

    GREETING = "greeting"
    AVAILABILITY = "availability"
    PRICES = "prices"
    HOURS = "hours"
    BOOK = "book"
    CANCEL = "cancel"
    LOOKUP = "lookup"
    STATIC_QUESTION = "static_question"


def _any(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(rf"(?:{p})" for p in patterns), re.IGNORECASE)


_GREETING = re.compile(
    r"^\W*(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|cheers|bye|goodbye)"
    r"\b[\w\s!.,']{0,25}$",
    re.IGNORECASE,
)

# Phrases that signal a *policy* question rather than an action request, e.g.
# "how late can I cancel" or "what details do I need to book".
_POLICY_SIGNAL = _any(
    r"\bhow (late|long|early|far|do i|do you|does|can i|to)\b",
    r"\bwhat (happens|if|is the|are the|details|do i need|information)\b",
    r"\b(policy|policies|refund|fee|fees|penalt\w*|charge[sd]?|free of charge|for free)\b",
    r"\b(allowed|possible|rule|rules|process|procedure|steps?)\b",
    r"\bno[- ]show\b",
)

_CANCEL_ACTION = _any(r"\bcancel(l?ing|l?ed|s)?\b", r"\bcall off\b")
_LOOKUP_ACTION = _any(
    r"\b(look ?up|find|check|show|view|see|get|retrieve|status of|details of|where is)\b"
    r".{0,30}\b(my )?(booking|reservation)\b",
    r"\b(my )?(booking|reservation) (status|details)\b",
    r"\bdo i have (a )?(booking|reservation)\b",
)
_BOOK_ACTION = _any(
    r"\b(book|reserve)\b",
    r"\b(make|create|place|need|want|like|get|start)\b.{0,20}\b(booking|reservation)\b",
    r"\b(new|a) (booking|reservation)\b",
    r"\bbooking\b.{0,10}\bfor (tomorrow|today|\d|mon|tue|wed|thu|fri|sat|sun)",
)

_HOURS = _any(
    r"\b(opening|working|business) (hours|times?)\b",
    r"\bwhat time\b.{0,20}\b(open|close|shut)",
    r"\b(open|opens|opened|closing|close|closes|shut|shuts)\b",
    r"\b(hours|24/7|24 hours)\b",
)
_PRICES = _any(
    r"\b(price|prices|pricing|cost|costs|rate|rates|tariff|tariffs)\b",
    r"\bhow much\b",
    r"\bper (hour|day)\b",
    r"\b(hourly|daily) (rate|price|charge)\b",
    r"\bcheap\w*\b|\bexpensive\b",
)
_AVAILABILITY = _any(
    r"\bavailab\w*\b",
    r"\bvacan\w*\b",
    r"\b(spaces?|spots?|bays?|places?)\b.{0,15}\b(left|free|available|open)\b",
    r"\b(free|any|empty)\b.{0,10}\b(spaces?|spots?|bays?|places?)\b",
    r"\bhow many\b.{0,20}\b(spaces?|spots?|bays?)\b",
    r"\b(is|are) there (any )?(room|space|spaces|spots?)\b",
    r"\bfull\b",
)
_POLICY_PRICE_CONTEXT = _any(
    r"\b(cancel\w*|refund\w*|penalt\w*|overstay\w*|fine|late|no[- ]show)\b"
)
_STATIC_HOURS_CONTEXT = _any(r"\b(overnight|24 hours (before|or more)|grace)\b")


def classify(text: str) -> Intent:
    """Classify a single user message into an ``Intent``."""
    message = text.strip()
    if not message:
        return Intent.STATIC_QUESTION
    if _GREETING.match(message):
        return Intent.GREETING

    policy = _POLICY_SIGNAL.search(message) is not None
    has_ref = parse_booking_ref(message) is not None
    wants_cancel = _CANCEL_ACTION.search(message) is not None
    wants_lookup = _LOOKUP_ACTION.search(message) is not None
    wants_book = _BOOK_ACTION.search(message) is not None

    # A booking reference in the message is a strong signal of an action request.
    if wants_cancel and (has_ref or not policy):
        return Intent.CANCEL
    if (wants_lookup or (has_ref and not wants_book)) and (has_ref or not policy):
        return Intent.LOOKUP
    if wants_book and not policy:
        return Intent.BOOK

    if _HOURS.search(message) and not _STATIC_HOURS_CONTEXT.search(message) and not policy:
        return Intent.HOURS
    if _PRICES.search(message) and not _POLICY_PRICE_CONTEXT.search(message):
        return Intent.PRICES
    if _AVAILABILITY.search(message) and not policy:
        return Intent.AVAILABILITY
    return Intent.STATIC_QUESTION


__all__ = ["Intent", "classify"]
