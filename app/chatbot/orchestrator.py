"""Chatbot orchestration logic.

``Orchestrator.handle`` is the single entry point for a user message. It
routes by intent:

- availability / prices / hours  -> templated answers straight from ``SqlStore``
- booking / cancel / lookup      -> code-validated flows in ``reservation.py``
- everything else                -> ``RagAnswerer`` over the static knowledge base

The LLM only ever sees the static context and the user's question; it never
receives SQL rows and it never decides anything about reservations.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from app.chatbot.intent import (
    ConversationalCue,
    Intent,
    classify,
    classify_cue,
    classify_explicit_action,
)
from app.chatbot.llm import LazyLLMClient, LLMError
from app.chatbot.parsing import parse_space_type, parse_weekday
from app.chatbot.rag import RagAnswerer, SourceRef
from app.chatbot.reservation import (
    SPACE_TYPE_LABELS,
    BookingFlow,
    CancelFlow,
    Flow,
    LookupFlow,
)
from app.config import Settings, get_settings
from app.data.sql_store import SqlStore
from app.data.vector_store import VectorStore
from app.guardrails.input_filter import (
    PROMPT_INJECTION_MESSAGE,
    InputFilter,
    PromptInjectionError,
)

logger = logging.getLogger(__name__)

WELCOME_MESSAGE = (
    "Hello! I'm the CityPark Central assistant. I can tell you about parking rules, "
    "location, policies, live availability, prices and opening hours, and I can book, "
    "look up or cancel a reservation. How can I help?"
)
LLM_UNAVAILABLE_MESSAGE = (
    "Sorry, I can't answer that right now because the language model is unavailable. "
    "I can still check availability, prices and opening hours, or manage a reservation."
)
GRATITUDE_REPLY = "You're welcome! Let me know if you need help with parking."
NEGATIVE_REPLY = (
    "No problem! Let me know if you'd like to check availability, prices, "
    "opening hours, or make a reservation."
)
ACK_WITH_TOPIC_REPLY = "Great! Would you like to check another parking type or make a reservation?"
ACK_WITHOUT_TOPIC_REPLY = (
    "Sure! Let me know if you'd like to check availability, prices, opening hours, "
    "or make a reservation."
)
UNCLEAR_MORE_INFO_REPLY = "Sure - what would you like more detail about?"
FLOW_SWITCH_PREFIX = "No problem, let's do that instead. "

# Intents that mean "start this reservation flow"; used to detect a mid-flow topic change.
_FLOW_INTENTS = {Intent.BOOK: "booking", Intent.CANCEL: "cancel", Intent.LOOKUP: "lookup"}
# Intents handled directly in handle() without entering a flow; last_intent/last_topic
# are only meaningful for these, so a cue right after one of them can refer back to it.
_TOPIC_INTENTS = (Intent.AVAILABILITY, Intent.PRICES, Intent.HOURS)


@dataclass
class ConversationSession:
    """Per-conversation state: active flow plus enough context to interpret follow-ups."""

    flow: Flow | None = None
    turns: int = 0
    last_topic: Intent | None = None
    last_intent: Intent | None = None
    last_static_question: str | None = None


@dataclass(frozen=True)
class BotReply:
    """Reply text plus routing metadata for the UI/API layer."""

    text: str
    intent: Intent
    sources: tuple[SourceRef, ...] = ()
    flow: str | None = None
    flow_done: bool = False


@dataclass
class Orchestrator:
    """Routes messages to SQL, reservation flows or RAG."""

    sql_store: SqlStore
    rag: RagAnswerer
    now: Callable[[], dt.datetime] = field(default=dt.datetime.now)
    input_filter: InputFilter = field(default_factory=InputFilter)

    def handle(self, message: str, session: ConversationSession) -> BotReply:
        """Process one user message within ``session`` and return the reply."""
        session.turns += 1
        text = message.strip()
        if not text:
            return BotReply("Please type a question or a request.", Intent.STATIC_QUESTION)
        try:
            self.input_filter.check_injection(text)
        except PromptInjectionError:
            return BotReply(PROMPT_INJECTION_MESSAGE, Intent.STATIC_QUESTION)

        if session.flow is not None:
            switch_intent = self._detect_flow_switch(text, session.flow.name)
            if switch_intent is not None:
                return self._switch_flow(switch_intent, text, session)
            return self._continue_flow(text, session)

        cue = classify_cue(text)
        if cue is not None:
            return self._handle_cue(cue, text, session)

        intent = classify(text, session.last_topic)
        logger.info("intent=%s turn=%d", intent.value, session.turns)

        if intent in _TOPIC_INTENTS:
            session.last_topic = intent

        if intent is Intent.GREETING:
            session.last_intent = intent
            return BotReply(WELCOME_MESSAGE, intent)
        if intent is Intent.AVAILABILITY:
            session.last_intent = intent
            return BotReply(self._availability(text), intent)
        if intent is Intent.PRICES:
            session.last_intent = intent
            return BotReply(self._prices(text), intent)
        if intent is Intent.HOURS:
            session.last_intent = intent
            return BotReply(self._hours(text), intent)
        if intent in (Intent.BOOK, Intent.CANCEL, Intent.LOOKUP):
            return self._start_flow(intent, text, session)
        session.last_intent = intent
        session.last_static_question = text
        return self._static(text, intent)

    # ---- flows -----------------------------------------------------------------
    def _start_flow(self, intent: Intent, text: str, session: ConversationSession) -> BotReply:
        flow: Flow
        if intent is Intent.BOOK:
            flow = BookingFlow(self.sql_store, now=self.now)
        elif intent is Intent.CANCEL:
            flow = CancelFlow(self.sql_store)
        else:
            flow = LookupFlow(self.sql_store)
        reply = flow.start(text)
        session.flow = None if reply.done else flow
        return BotReply(reply.text, intent, flow=flow.name, flow_done=reply.done)

    def _continue_flow(self, text: str, session: ConversationSession) -> BotReply:
        flow = session.flow
        assert flow is not None
        reply = flow.handle(text)
        if reply.done:
            session.flow = None
        intent = {"booking": Intent.BOOK, "cancel": Intent.CANCEL}.get(flow.name, Intent.LOOKUP)
        return BotReply(reply.text, intent, flow=flow.name, flow_done=reply.done)

    def _detect_flow_switch(self, text: str, current_flow_name: str) -> Intent | None:
        """Detect a clear request to switch to a *different* reservation flow mid-conversation.

        Deliberately narrow: only an explicit book/cancel/lookup verb can interrupt
        an active flow (not the bare-reference-implies-lookup fallback that
        ``classify`` otherwise uses), so a reference typed as the literal answer
        to the active flow's pending question is never misread as a topic change.
        """
        candidate = classify_explicit_action(text)
        if candidate in _FLOW_INTENTS and _FLOW_INTENTS[candidate] != current_flow_name:
            return candidate
        return None

    def _switch_flow(self, intent: Intent, text: str, session: ConversationSession) -> BotReply:
        session.flow = None
        reply = self._start_flow(intent, text, session)
        return BotReply(
            FLOW_SWITCH_PREFIX + reply.text,
            reply.intent,
            flow=reply.flow,
            flow_done=reply.flow_done,
        )

    # ---- conversational acts (ack/gratitude/yes/no/more-info) ------------------
    def _handle_cue(
        self, cue: ConversationalCue, text: str, session: ConversationSession
    ) -> BotReply:
        if cue is ConversationalCue.GRATITUDE:
            return BotReply(GRATITUDE_REPLY, Intent.STATIC_QUESTION)
        if cue is ConversationalCue.NEGATIVE:
            return BotReply(NEGATIVE_REPLY, Intent.STATIC_QUESTION)
        if cue is ConversationalCue.MORE_INFO:
            return self._more_info(text, session)
        # ACKNOWLEDGEMENT or AFFIRMATIVE with no active flow: nothing is pending
        # confirmation (that only happens inside a flow), so treat both the same way.
        return BotReply(self._acknowledgement_reply(session), Intent.STATIC_QUESTION)

    def _acknowledgement_reply(self, session: ConversationSession) -> str:
        if session.last_topic is not None or session.last_intent is Intent.STATIC_QUESTION:
            return ACK_WITH_TOPIC_REPLY
        return ACK_WITHOUT_TOPIC_REPLY

    def _more_info(self, text: str, session: ConversationSession) -> BotReply:
        if session.last_static_question is None:
            return BotReply(UNCLEAR_MORE_INFO_REPLY, Intent.STATIC_QUESTION)
        combined = f"{session.last_static_question} {text.strip()}"
        return self._static(combined, Intent.STATIC_QUESTION)

    # ---- dynamic data (SQL only, no LLM) ----------------------------------------
    def _availability(self, text: str) -> str:
        space_type = parse_space_type(text)
        rows = self.sql_store.get_availability(space_type)
        if space_type is not None:
            row = rows[0]
            label = SPACE_TYPE_LABELS[row.space_type]
            if row.available == 0:
                return f"Sorry, there are no {label} spaces available right now (0 of {row.total})."
            return f"Right now {row.available} of {row.total} {label} spaces are available."
        parts = [
            f"{SPACE_TYPE_LABELS[r.space_type]}: {r.available} of {r.total}" for r in rows
        ]
        return "Current availability - " + "; ".join(parts) + "."

    def _prices(self, text: str) -> str:
        space_type = parse_space_type(text)
        rows = self.sql_store.get_prices(space_type)
        if space_type is not None:
            row = rows[0]
            label = SPACE_TYPE_LABELS[row.space_type]
            return (
                f"{label[0].upper()}{label[1:]} spaces cost £{row.price_per_hour:.2f} per hour "
                f"or £{row.price_per_day:.2f} per day."
            )
        parts = [
            f"{SPACE_TYPE_LABELS[r.space_type]}: £{r.price_per_hour:.2f}/hour or "
            f"£{r.price_per_day:.2f}/day"
            for r in rows
        ]
        return "Our prices are - " + "; ".join(parts) + "."

    def _hours(self, text: str) -> str:
        day = parse_weekday(text, self.now().date())
        rows = self.sql_store.get_working_hours(day)
        lines = []
        for row in rows:
            if row.is_closed or row.open_time is None or row.close_time is None:
                lines.append(f"{row.day_name}: closed")
            else:
                lines.append(
                    f"{row.day_name}: {row.open_time.strftime('%H:%M')}-"
                    f"{row.close_time.strftime('%H:%M')}"
                )
        if day is not None:
            return f"Opening hours on {lines[0]}."
        return "Opening hours - " + "; ".join(lines) + "."

    # ---- static knowledge (RAG) --------------------------------------------------
    def _static(self, text: str, intent: Intent) -> BotReply:
        try:
            answer = self.rag.answer(text)
        except LLMError:
            logger.warning("LLM unavailable; returning fallback message")
            return BotReply(LLM_UNAVAILABLE_MESSAGE, intent)
        return BotReply(answer.text, intent, sources=answer.sources)


def build_orchestrator(settings: Settings | None = None) -> Orchestrator:
    """Wire the orchestrator from configuration (real SQL, Chroma and LLM)."""
    settings = settings or get_settings()
    sql_store = SqlStore.from_url(settings.sqlite_url)
    vector_store = VectorStore()
    # Lazy so SQL, greeting and reservation paths work without a configured LLM.
    rag = RagAnswerer(vector_store, LazyLLMClient(settings))
    return Orchestrator(sql_store=sql_store, rag=rag)


__all__ = [
    "ACK_WITH_TOPIC_REPLY",
    "ACK_WITHOUT_TOPIC_REPLY",
    "FLOW_SWITCH_PREFIX",
    "GRATITUDE_REPLY",
    "LLM_UNAVAILABLE_MESSAGE",
    "NEGATIVE_REPLY",
    "UNCLEAR_MORE_INFO_REPLY",
    "WELCOME_MESSAGE",
    "BotReply",
    "ConversationSession",
    "Orchestrator",
    "build_orchestrator",
]
