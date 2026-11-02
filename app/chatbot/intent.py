"""Intent detection and routing logic.

Routing is rule-based on purpose: it is deterministic, testable offline, and it
keeps the decision of *which data source answers* (SQL vs. vector store vs. a
code-validated reservation flow) out of the language model's hands. Anything
that is not recognised falls through to ``STATIC_QUESTION`` where the RAG
layer either answers from the knowledge base or says it does not know.
"""

from __future__ import annotations

import datetime as dt
import re
from enum import Enum

from app.chatbot.parsing import (
    parse_approval_ref,
    parse_booking_ref,
    parse_space_type,
    parse_weekday,
)


class Intent(str, Enum):
    """What the user wants; decides which backend handles the message."""

    GREETING = "greeting"
    AVAILABILITY = "availability"
    PRICES = "prices"
    HOURS = "hours"
    BOOK = "book"
    CANCEL = "cancel"
    LOOKUP = "lookup"
    APPROVAL_STATUS = "approval_status"
    STATIC_QUESTION = "static_question"


class ConversationalCue(str, Enum):
    """A short conversational act rather than an information request.

    Detection here is deliberately context-free (just the words used); what a
    cue *means* depends on the conversation state (is a flow waiting for this
    answer? did we just answer a question?) and is decided by the caller.
    """

    ACKNOWLEDGEMENT = "acknowledgement"
    GRATITUDE = "gratitude"
    AFFIRMATIVE = "affirmative"
    NEGATIVE = "negative"
    MORE_INFO = "more_info"


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

_CANCEL_ACTION = _any(
    r"\bcancel(l?ing|l?ed|s)?\b",
    r"\bcall off\b",
    r"\b(don'?t|do not|no longer) need\b.{0,25}\b(parking|space|slot|spot|booking|reservation)\b",
)
_LOOKUP_ACTION = _any(
    r"\b(look ?up|find|check|show|view|see|get|retrieve|status of|details of|where is)\b"
    r".{0,30}\b(my )?(booking|reservation)\b",
    r"\b(my )?(booking|reservation) (status|details|confirmation)\b",
    r"\bdo i have (a )?(booking|reservation)\b",
)
_BOOK_ACTION = _any(
    r"\b(book|reserve)\b",
    r"\b(make|create|place|need|want|like|get|start)\b.{0,20}\b(booking|reservation)\b",
    r"\b(new|a) (booking|reservation)\b",
    r"\bbooking\b.{0,10}\bfor (tomorrow|today|\d|mon|tue|wed|thu|fri|sat|sun)",
    r"\bhelp\b.{0,10}\bpark\b.{0,10}\b(my|the|a) car\b",
    r"\b(need|want)\b.{0,10}\bparking\b\s+for\s+(tomorrow|today|next|this|\d|mon|tue|wed|thu|fri|sat|sun)",
)

_HOURS = _any(
    r"\b(opening|working|business) (hours|times?)\b",
    r"\bwhat time\b.{0,20}\b(open|close|shut)",
    r"\b(open|opens|opened|closing|close|closes|shut|shuts)\b",
    r"\b(hours|24/7|24 hours)\b",
)
_PRICES = _any(
    r"\b(price|prices|pricing|cost|costs|rate|rates|tariff|tariffs|charges?)\b",
    r"\bhow much\b",
    r"\bper (hour|day)\b",
    r"\b(hourly|daily) (rate|price|charge)\b",
    r"\bcheap\w*\b|\bexpensive\b",
)
_AVAILABILITY = _any(
    r"\bavailab\w*\b",
    r"\bvacan\w*\b",
    r"\bcapacit\w*\b",
    r"\b(spaces?|spots?|bays?|places?|slots?)\b.{0,15}\b(left|free|available|open)\b",
    r"\b(free|any|empty)\b.{0,10}\b(spaces?|spots?|bays?|places?|slots?)\b",
    r"\bhow many\b.{0,20}\b(spaces?|spots?|bays?|slots?)\b",
    r"\b(is|are) there (any )?(room|space|spaces|spots?|slots?)\b",
    r"\bcan (i|you|we) (get|check|find)\b.{0,20}\b(space|spot|slot|bay)s?\b",
    r"\bfull\b",
)
_POLICY_PRICE_CONTEXT = _any(
    r"\b(cancel\w*|refund\w*|penalt\w*|overstay\w*|fine|late|no[- ]show)\b"
)
# Genuine policy questions about availability (not a live-data request), e.g. "is there a
# waiting list". The bare "what are the" / "what is the" phrasing used to block *every*
# availability or hours question that happened to use it (see _POLICY_SIGNAL); those two
# intents now use this narrower context instead, matching how _POLICY_PRICE_CONTEXT already
# scopes the prices intent.
_POLICY_AVAILABILITY_CONTEXT = _any(
    r"\bwait(ing)?[- ]?list\b",
    r"\bwhat happens\b.{0,25}\b(full|no (space|spots?|availability))\b",
)
_STATIC_HOURS_CONTEXT = _any(r"\b(overnight|24 hours (before|or more)|grace)\b")

# Bare conversational acts. Matched against the *whole* trimmed message (not just a
# substring search) so that a real sentence that happens to contain "ok" or "yes" still
# falls through to ``classify`` instead of being swallowed as small talk.
_ACK_WORDS = frozenset({"ok", "okay", "alright", "got it", "understood", "sure", "fine"})
_AFFIRMATIVE_WORDS = frozenset({"yes", "yeah", "yep", "yup", "correct", "absolutely", "right"})
_NEGATIVE_WORDS = frozenset({"no", "nope", "nah", "not really", "no thanks", "no thank you"})
_NEGATIVE_WANT_PREFIX = re.compile(r"^i (don'?t|do not) want to\b")
_GRATITUDE_PHRASE = re.compile(
    r"^(thanks( a lot| very much)?|thank you( very much)?|many thanks|thx)$"
)
# "What is my request ID?" / "repeat my request ID" / "what was the reference number" -
# asking the bot to recall a Stage 2 approval request ID it already gave out, rather than
# giving a fresh AR-XXXXXX/plate to look one up. Anchored to the end of the (stripped)
# message so a genuine policy question like "what is the request process" is not caught.
_REQUEST_ID_RECALL = _any(
    r"\bwhat('s| is| was)?\s+(my|the)\s+(approval )?(request|reference)\s*(id|number)?[?!.]*$",
    r"\b(repeat|remind me( of)?|tell me)\b.{0,20}\b(request|reference)\s*(id|number)?[?!.]*$",
)

_MORE_INFO = _any(
    r"^tell me more\b",
    r"^(can you )?explain (that|this|it)\b",
    r"\bcan you elaborate\b",
    r"^what else\b",
    r"^give me more details?\b",
    r"\bmore (info|information|details?)\b",
    r"\b(simpler|simple) terms\b",
)


def classify_cue(text: str) -> ConversationalCue | None:
    """Detect a short conversational act (ack/gratitude/yes/no/more-info).

    Returns ``None`` for anything else, including real questions that happen to
    contain one of these words ("Can I cancel my booking, yes or no?").
    """
    normalized = text.strip().lower().rstrip("!.,;: ")
    if not normalized:
        return None
    if _GRATITUDE_PHRASE.match(normalized):
        return ConversationalCue.GRATITUDE
    if _MORE_INFO.search(normalized):
        return ConversationalCue.MORE_INFO
    if normalized in _ACK_WORDS:
        return ConversationalCue.ACKNOWLEDGEMENT
    if normalized in _AFFIRMATIVE_WORDS:
        return ConversationalCue.AFFIRMATIVE
    if normalized in _NEGATIVE_WORDS or _NEGATIVE_WANT_PREFIX.match(normalized):
        return ConversationalCue.NEGATIVE
    return None


def classify(text: str, last_topic: Intent | None = None) -> Intent:
    """Classify a single user message into an ``Intent``.

    ``last_topic`` is the previous turn's SQL-backed topic (availability,
    prices or hours). A message with no topic keyword of its own but naming a
    space type or a day (e.g. "what about standard?" after an availability
    question) stays on that topic instead of falling through to RAG.
    """
    message = text.strip()
    if not message:
        return Intent.STATIC_QUESTION
    if _GREETING.match(message):
        return Intent.GREETING
    if parse_approval_ref(message) is not None:
        return Intent.APPROVAL_STATUS
    if _REQUEST_ID_RECALL.search(message) is not None:
        return Intent.APPROVAL_STATUS

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

    if _HOURS.search(message) and not _STATIC_HOURS_CONTEXT.search(message):
        return Intent.HOURS
    if _PRICES.search(message) and not _POLICY_PRICE_CONTEXT.search(message):
        return Intent.PRICES
    if _AVAILABILITY.search(message) and not _POLICY_AVAILABILITY_CONTEXT.search(message):
        return Intent.AVAILABILITY

    if not policy and last_topic in (Intent.AVAILABILITY, Intent.PRICES):
        if parse_space_type(message) is not None:
            return last_topic
    if not policy and last_topic is Intent.HOURS:
        if parse_weekday(message, dt.date.today()) is not None:
            return Intent.HOURS
    return Intent.STATIC_QUESTION


def classify_explicit_action(text: str) -> Intent | None:
    """Like ``classify``, but only ever returns BOOK/CANCEL/LOOKUP on an explicit verb.

    ``classify`` also returns LOOKUP for a bare booking reference with no other
    signal (a reasonable default for a fresh message), but that fallback means a
    bare reference typed as the literal answer to an *active* flow's pending
    question would look like a request to switch flows. Callers deciding whether
    to interrupt an active flow should use this instead.
    """
    message = text.strip()
    if not message:
        return None
    policy = _POLICY_SIGNAL.search(message) is not None
    if _CANCEL_ACTION.search(message) and not policy:
        return Intent.CANCEL
    if _LOOKUP_ACTION.search(message) and not policy:
        return Intent.LOOKUP
    if _BOOK_ACTION.search(message) and not policy:
        return Intent.BOOK
    return None


__all__ = [
    "ConversationalCue",
    "Intent",
    "classify",
    "classify_cue",
    "classify_explicit_action",
]
