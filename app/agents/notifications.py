"""Administrator notification adapters.

``ConsoleNotificationAdapter`` is the local development/mock adapter used whenever real
credentials are not configured - it logs the request so the full workflow is testable end
to end with no external service. ``EmailNotificationAdapter`` uses only the stdlib
``smtplib`` (no new dependency) and is used when SMTP settings are configured.
"""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Protocol

logger = logging.getLogger(__name__)


class NotificationAdapter(Protocol):
    """Delivers an approval-request message to the administrator."""

    def notify(self, request_id: str, message: str) -> None: ...


@dataclass
class ConsoleNotificationAdapter:
    """Mock/dev adapter: logs the notification instead of sending it externally."""

    def notify(self, request_id: str, message: str) -> None:
        logger.info("ADMIN NOTIFICATION [%s]:\n%s", request_id, message)


@dataclass
class EmailNotificationAdapter:
    """Sends the notification by email via SMTP (stdlib only)."""

    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    from_address: str
    admin_email: str
    use_tls: bool = True

    def notify(self, request_id: str, message: str) -> None:
        email = EmailMessage()
        email["Subject"] = f"CityPark Central: reservation approval needed ({request_id})"
        email["From"] = self.from_address
        email["To"] = self.admin_email
        email.set_content(message)
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as client:
                if self.use_tls:
                    client.starttls()
                if self.smtp_username:
                    client.login(self.smtp_username, self.smtp_password)
                client.send_message(email)
        except OSError:
            # Never let a notification failure block the approval workflow; the request
            # is already persisted and visible via the admin API regardless of delivery.
            logger.exception("Failed to email admin notification for %s", request_id)


__all__ = ["ConsoleNotificationAdapter", "EmailNotificationAdapter", "NotificationAdapter"]
