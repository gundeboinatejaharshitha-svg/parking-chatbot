"""Streamlit user interface backed exclusively by the FastAPI chat endpoint."""

from __future__ import annotations

import streamlit as st

from app.config import get_settings
from app.ui.api_client import ChatApiClient, ChatApiError

QUICK_ACTIONS = (
    ("Check Availability", "What parking slots are available?"),
    ("Book Parking", "I want to book parking"),
    ("Parking Information", "Tell me about your parking facilities and policies"),
    ("Find My Booking", "I want to check my booking"),
    ("Cancel Booking", "I want to cancel my booking"),
)


def _init_state() -> None:
    st.session_state.setdefault("citypark_messages", [])
    st.session_state.setdefault("citypark_session_id", None)
    st.session_state.setdefault("citypark_last_failed_message", None)
    st.session_state.setdefault("citypark_waiting_for_more", False)


def _send(message: str) -> None:
    """Submit one message through the real chat API and record both turns.

    Goes through the same orchestrator, guardrails and flows as any other
    request; quick actions are just canned text for this same path.
    """
    st.session_state.citypark_messages.append({"role": "user", "content": message})
    client = ChatApiClient(get_settings().api_base_url)
    try:
        with st.spinner("CityPark Assistant is thinking..."):
            response = client.send_chat(message, st.session_state.citypark_session_id)
    except ChatApiError as exc:
        st.session_state.citypark_last_failed_message = message
        st.session_state.citypark_messages.append(
            {"role": "assistant", "content": exc.user_message, "sources": []}
        )
        return

    st.session_state.citypark_last_failed_message = None
    st.session_state.citypark_session_id = str(response.session_id)
    st.session_state.citypark_waiting_for_more = bool(response.flow) and not response.flow_done
    st.session_state.citypark_messages.append(
        {
            "role": "assistant",
            "content": response.reply,
            "sources": [source.model_dump() for source in response.sources],
        }
    )


def _render_history() -> None:
    for item in st.session_state.citypark_messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("sources"):
                with st.expander("Sources"):
                    for source in item["sources"]:
                        st.caption(
                            f"{source['section']} — {source['source']} "
                            f"(score {source['score']:.3f})"
                        )


def _render_quick_actions() -> bool:
    """Render the quick-action row; return True if one was clicked this run."""
    st.caption("Quick actions")
    columns = st.columns(len(QUICK_ACTIONS))
    clicked = False
    for column, (label, prompt) in zip(columns, QUICK_ACTIONS, strict=True):
        if column.button(label, use_container_width=True, key=f"quick_action_{label}"):
            _send(prompt)
            clicked = True
    return clicked


def main() -> None:
    """Render the chat history, quick actions and the chat input."""
    st.set_page_config(page_title="CityPark Central Assistant", page_icon="🅿️")
    st.title("CityPark Central Assistant")
    st.caption("Ask about parking, availability, prices, opening hours, or reservations.")
    _init_state()

    if _render_quick_actions():
        st.rerun()

    _render_history()

    if st.session_state.citypark_last_failed_message:
        if st.button("Retry last message"):
            _send(st.session_state.citypark_last_failed_message)
            st.rerun()

    if st.session_state.citypark_waiting_for_more:
        st.caption("Waiting for a few more details to continue...")

    message = st.chat_input("How can I help?")
    if not message:
        return

    _send(message)
    st.rerun()


if __name__ == "__main__":
    main()
