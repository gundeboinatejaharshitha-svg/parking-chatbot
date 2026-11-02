"""Administrator dashboard: review and decide pending reservation approval requests.

A Streamlit multipage "page" alongside the existing customer chat entry point
(``app/ui/streamlit_app.py``) - same app, same `streamlit run` command, no separate
frontend. All data comes from the real /admin API (SQL-backed); nothing is hardcoded.
"""

from __future__ import annotations

import streamlit as st

from app.config import get_settings
from app.ui.admin_api_client import AdminApiClient, AdminApiError

st.set_page_config(page_title="CityPark Central - Administrator", page_icon="🛂")
st.title("Administrator Dashboard")

if "admin_api_key" not in st.session_state:
    st.session_state.admin_api_key = None


def _client() -> AdminApiClient:
    return AdminApiClient(get_settings().api_base_url)


def _login() -> None:
    st.caption("Sign in with the administrator API key to review reservation requests.")
    with st.form("admin_login"):
        key = st.text_input("Administrator API key", type="password", key="admin_key_input")
        submitted = st.form_submit_button("Log in")
    if not submitted:
        return
    if not key:
        st.warning("Please enter the administrator API key.")
        return
    try:
        _client().list_requests(key, status="pending_approval")
    except AdminApiError as exc:
        if exc.status_code == 401:
            st.error("Incorrect administrator key.")
        elif exc.status_code == 503:
            st.error("The approval workflow is not enabled on the backend.")
        else:
            st.error(f"Could not verify the key: {exc}")
        return
    # The key itself is held only in this browser session's memory, never logged.
    st.session_state.admin_api_key = key
    st.rerun()


def _logout() -> None:
    st.session_state.admin_api_key = None
    st.rerun()


def _space_label(space_type: str) -> str:
    return {"ev": "EV charging", "disabled": "disabled (blue badge)"}.get(
        space_type, space_type
    )


def _render_pending(client: AdminApiClient, admin_key: str, pending: list[dict]) -> None:
    st.subheader(f"Pending requests ({len(pending)})")
    if not pending:
        st.caption("No requests are currently waiting for review.")
        return
    for row in pending:
        with st.container(border=True):
            st.markdown(f"**{row['request_id']}** - requested {row['created_at']}")
            st.write(
                f"Customer: {row['customer_name']}  |  Plate: {row['car_plate']}  |  "
                f"{_space_label(row['space_type'])} space"
            )
            st.write(f"Date: {row['date']}  |  Time: {row['start_time']}-{row['end_time']}")
            reason = st.text_input(
                "Refusal reason (optional)", key=f"reason_{row['request_id']}"
            )
            col_approve, col_refuse = st.columns(2)
            if col_approve.button("Approve", key=f"approve_{row['request_id']}"):
                _decide(client, admin_key, row["request_id"], "approve", None)
            if col_refuse.button("Refuse", key=f"refuse_{row['request_id']}"):
                _decide(client, admin_key, row["request_id"], "refuse", reason or None)


def _decide(
    client: AdminApiClient, admin_key: str, request_id: str, decision: str, reason: str | None
) -> None:
    try:
        result = client.decide(admin_key, request_id, decision, reason)
    except AdminApiError as exc:
        if exc.status_code == 409:
            st.error("That request was already decided (refresh to see the current status).")
        elif exc.status_code == 410:
            st.error("That request has expired and can no longer be decided.")
        elif exc.status_code == 404:
            st.error("That request could not be found.")
        elif exc.status_code == 401:
            st.error("Your session is no longer authorized. Please log in again.")
            st.session_state.admin_api_key = None
        else:
            st.error(f"Could not record the decision: {exc}")
        return
    if result["status"] == "approved":
        st.success(f"Approved. Booking reference: {result['booking_ref']}.")
    else:
        st.success("Refused. The customer will see this the next time they check.")
    st.rerun()


def _render_history(decided: list[dict]) -> None:
    st.subheader("Decision history")
    if not decided:
        st.caption("No requests have been decided yet.")
        return
    st.dataframe(
        [
            {
                "Request": r["request_id"],
                "Status": r["status"],
                "Decided": r["decided_at"],
                "By": r["decided_by"],
                "Booking ref": r["booking_ref"] or "-",
                "Reason": r["refusal_reason"] or "-",
            }
            for r in decided
        ],
        hide_index=True,
        use_container_width=True,
    )


def main() -> None:
    if not st.session_state.admin_api_key:
        _login()
        return

    st.button("Log out", on_click=_logout)
    client = _client()
    admin_key = st.session_state.admin_api_key

    try:
        all_requests = client.list_requests(admin_key, status="all")
    except AdminApiError as exc:
        if exc.status_code == 401:
            st.error("Your session is no longer authorized. Please log in again.")
            st.session_state.admin_api_key = None
            st.rerun()
        else:
            st.error(f"Could not load requests: {exc}")
        return

    pending = [r for r in all_requests if r["status"] == "pending_approval"]
    approved = [r for r in all_requests if r["status"] == "approved"]
    refused = [r for r in all_requests if r["status"] == "refused"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Pending", len(pending))
    col2.metric("Approved", len(approved))
    col3.metric("Refused", len(refused))

    st.button("Refresh")  # any click reruns the script and re-fetches from the backend

    _render_pending(client, admin_key, pending)
    _render_history(approved + refused)


main()
