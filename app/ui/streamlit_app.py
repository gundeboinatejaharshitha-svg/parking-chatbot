"""Streamlit user interface backed exclusively by the FastAPI chat endpoint."""

from __future__ import annotations

import streamlit as st

from app.config import get_settings
from app.ui.api_client import ChatApiClient, ChatApiError


def main() -> None:
    """Render the chat history and submit messages to the API."""
    st.set_page_config(page_title="CityPark Central Assistant", page_icon="🅿️")
    st.title("CityPark Central Assistant")
    st.caption("Ask about parking, availability, prices, opening hours, or reservations.")

    if "citypark_messages" not in st.session_state:
        st.session_state.citypark_messages = []
    if "citypark_session_id" not in st.session_state:
        st.session_state.citypark_session_id = None

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

    message = st.chat_input("How can I help?")
    if not message:
        return

    st.session_state.citypark_messages.append({"role": "user", "content": message})
    client = ChatApiClient(get_settings().api_base_url)
    try:
        response = client.send_chat(message, st.session_state.citypark_session_id)
    except ChatApiError as exc:
        reply = exc.user_message
        sources: list[dict[str, str | float]] = []
    else:
        st.session_state.citypark_session_id = str(response.session_id)
        reply = response.reply
        sources = [source.model_dump() for source in response.sources]

    st.session_state.citypark_messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
    st.rerun()


if __name__ == "__main__":
    main()
