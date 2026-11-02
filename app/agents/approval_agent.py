"""Agent 2: human-in-the-loop reservation approval.

Built with LangChain (``PromptTemplate``) for structuring the administrator notification
message; the LLM call itself reuses the existing ``LLMClient``/``LazyLLMClient`` abstraction
(no new provider integration) and always has a deterministic template fallback, so the
approval workflow never depends on a configured LLM. Every status change and SQL operation
is deterministic: the LLM is never asked to approve, refuse, or decide anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.prompts import PromptTemplate

from app.agents.approval_models import ApprovalRequest
from app.agents.approval_store import ApprovalDraft, ApprovalRequestNotFoundError, ApprovalStore
from app.agents.notifications import (
    ConsoleNotificationAdapter,
    EmailNotificationAdapter,
    NotificationAdapter,
)
from app.chatbot.llm import LazyLLMClient, LLMClient, LLMError
from app.config import Settings
from app.data.sql_store import NoAvailabilityError, SqlStore, create_db_engine, normalize_plate

logger = logging.getLogger(__name__)

_MESSAGE_PROMPT = PromptTemplate.from_template(
    "Write a short, professional notification (3-4 sentences) to a parking facility "
    "administrator, asking them to approve or refuse this reservation request. State only "
    "the facts given below; do not invent or add any details.\n\n"
    "Request ID: {request_id}\nSpace type: {space_type}\nDate: {date}\n"
    "Time: {start_time} to {end_time}\nCustomer: {customer_name}\nPlate: {car_plate}"
)
_SYSTEM_PROMPT = (
    "You write concise administrator notifications for a parking facility. "
    "Use only the facts given; never invent details."
)


def _deterministic_message(**f: str) -> str:
    return (
        f"New reservation approval request {f['request_id']}.\n"
        f"Customer: {f['customer_name']}, plate {f['car_plate']}.\n"
        f"{f['space_type']} space on {f['date']} from {f['start_time']} to {f['end_time']}.\n"
        "Reply approve or refuse (an optional reason may be given)."
    )


@dataclass(frozen=True)
class SubmitResult:
    """Outcome of submitting a request: the record plus whether the admin was notified."""

    request: ApprovalRequest
    notified: bool


class ApprovalAgent:
    """Prepares, sends, and processes human-in-the-loop reservation approvals."""

    def __init__(
        self,
        sql_store: SqlStore,
        approval_store: ApprovalStore,
        notifier: NotificationAdapter,
        llm: LLMClient | None = None,
    ) -> None:
        self._sql_store = sql_store
        self._approval_store = approval_store
        self._notifier = notifier
        self._llm = llm

    # ---- Agent 1 -> Agent 2 ----------------------------------------------------
    def submit_request(self, draft: ApprovalDraft) -> SubmitResult:
        """Validate current availability, persist a pending request, notify the admin."""
        availability = self._sql_store.get_availability(draft.space_type)[0]
        if availability.available <= 0:
            raise NoAvailabilityError(f"No {draft.space_type} spaces are currently available.")

        request = self._approval_store.create_request(draft)
        message = self._draft_message(request)
        notified = True
        try:
            self._notifier.notify(request.id, message)
        except Exception:
            logger.exception("Admin notification failed for %s", request.id)
            notified = False
        return SubmitResult(request=request, notified=notified)

    def _draft_message(self, request: ApprovalRequest) -> str:
        fields = {
            "request_id": request.id,
            "space_type": request.space_type,
            "date": request.date.isoformat(),
            "start_time": request.start_time.strftime("%H:%M"),
            "end_time": request.end_time.strftime("%H:%M"),
            "customer_name": request.customer_name,
            "car_plate": request.car_plate,
        }
        if self._llm is None:
            return _deterministic_message(**fields)
        try:
            return self._llm.complete(_SYSTEM_PROMPT, _MESSAGE_PROMPT.format(**fields))
        except LLMError:
            logger.warning("LLM unavailable for admin message drafting; using template")
            return _deterministic_message(**fields)

    # ---- administrator decision -> Agent 2 -------------------------------------
    def decide(
        self, request_id: str, approve: bool, actor: str, reason: str | None = None
    ) -> ApprovalRequest:
        """Apply an authenticated administrator's decision; never reprocesses one."""
        row = self._approval_store.decide(request_id, approve, actor, reason)
        if not approve:
            return row

        try:
            info = self._sql_store.create_reservation(
                customer_name=row.customer_name,
                car_plate=row.car_plate,
                date=row.date,
                start_time=row.start_time,
                end_time=row.end_time,
                space_type=row.space_type,
            )
        except NoAvailabilityError as exc:
            self._approval_store.mark_refused(request_id, f"Approved but sold out: {exc}")
            return self._approval_store.get(request_id)
        self._approval_store.mark_booking_ref(request_id, info.booking_ref)
        return self._approval_store.get(request_id)

    def list_pending(self) -> list[ApprovalRequest]:
        """All requests currently awaiting an administrator decision."""
        return self._approval_store.list_pending()

    def list_all(self) -> list[ApprovalRequest]:
        """Every request regardless of status (dashboard counts/history)."""
        return self._approval_store.list_all()

    # ---- Agent 1: customer status checks ---------------------------------------
    def get_status(self, request_id: str, car_plate: str) -> ApprovalRequest | None:
        """Return the request only if the plate matches - never confirms a guessed ID."""
        try:
            row = self._approval_store.get(request_id)
        except ApprovalRequestNotFoundError:
            return None
        if row.car_plate != normalize_plate(car_plate):
            return None
        return row


def build_approval_agent(settings: Settings, sql_store: SqlStore) -> ApprovalAgent:
    """Wire Agent 2 from configuration: console adapter unless email is configured."""
    engine = create_db_engine(settings.sqlite_url)
    approval_store = ApprovalStore(engine, ttl_hours=settings.approval_request_ttl_hours)
    if settings.admin_notify_method == "email" and settings.admin_email and settings.smtp_host:
        notifier: NotificationAdapter = EmailNotificationAdapter(
            smtp_host=settings.smtp_host,
            smtp_port=settings.smtp_port,
            smtp_username=settings.smtp_username,
            smtp_password=settings.smtp_password,
            from_address=settings.smtp_from or settings.smtp_username,
            admin_email=settings.admin_email,
        )
    else:
        notifier = ConsoleNotificationAdapter()
    return ApprovalAgent(sql_store, approval_store, notifier, llm=LazyLLMClient(settings))


__all__ = ["ApprovalAgent", "SubmitResult", "build_approval_agent"]
