"""Small HTTP client for the Streamlit administrator dashboard -> FastAPI admin API.

Mirrors ``app.ui.api_client.ChatApiClient``'s pattern. The admin key is supplied by the
caller on every call (from Streamlit session state) and is never stored, logged, or
hardcoded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class AdminApiError(RuntimeError):
    """Raised for any non-2xx response; ``status_code`` lets the UI react appropriately."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class AdminApiClient:
    """Calls the /admin endpoints without exposing transport details to the UI."""

    base_url: str
    timeout_seconds: float = 15.0
    transport: httpx.BaseTransport | None = None

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url.rstrip("/"),
            timeout=self.timeout_seconds,
            transport=self.transport,
        )

    def _raise_for_error(self, response: httpx.Response) -> None:
        if not response.is_error:
            return
        try:
            detail = response.json().get("detail", "Request failed.")
        except ValueError:
            detail = "Request failed."
        raise AdminApiError(detail, response.status_code)

    def list_requests(self, admin_key: str, status: str = "all") -> list[dict[str, Any]]:
        """Fetch requests for the given status filter ("all" for the full dashboard)."""
        try:
            with self._client() as client:
                response = client.get(
                    "/admin/requests",
                    params={"status": status},
                    headers={"X-Admin-Key": admin_key},
                )
        except httpx.HTTPError as exc:
            raise AdminApiError("Could not reach the admin API.", 0) from exc
        self._raise_for_error(response)
        return response.json()

    def decide(
        self, admin_key: str, request_id: str, decision: str, reason: str | None = None
    ) -> dict[str, Any]:
        """Approve or refuse a request; raises ``AdminApiError`` with the real status code."""
        try:
            with self._client() as client:
                response = client.post(
                    f"/admin/requests/{request_id}/decision",
                    json={"decision": decision, "reason": reason},
                    headers={"X-Admin-Key": admin_key},
                )
        except httpx.HTTPError as exc:
            raise AdminApiError("Could not reach the admin API.", 0) from exc
        self._raise_for_error(response)
        return response.json()


__all__ = ["AdminApiClient", "AdminApiError"]
