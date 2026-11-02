"""HTTP API tests using in-memory services and a stub LLM."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.chatbot.orchestrator import Orchestrator
from app.chatbot.rag import RagAnswerer
from app.data.ingest import Chunk
from app.data.sql_store import SqlStore
from app.data.vector_store import VectorStore
from app.main import create_app
from tests.conftest import FakeClock, FakeLLM
from tests.guardrail_fakes import build_test_filters


def _client(
    sql_store: SqlStore,
    vector_store: VectorStore,
    fake_llm: FakeLLM,
    clock: FakeClock,
) -> TestClient:
    input_filter, output_filter = build_test_filters()
    rag = RagAnswerer(
        vector_store,
        fake_llm,
        min_score=0.0,
        input_filter=input_filter,
        output_filter=output_filter,
    )
    orchestrator = Orchestrator(sql_store=sql_store, rag=rag, now=clock)
    return TestClient(create_app(orchestrator=orchestrator, sql_store=sql_store))


def test_health_does_not_initialize_the_llm() -> None:
    client = TestClient(create_app())

    assert client.get("/health").json() == {"status": "ok"}


def test_health_and_sql_availability_endpoints(
    sql_store: SqlStore, vector_store: VectorStore, fake_llm: FakeLLM, clock: FakeClock
) -> None:
    client = _client(sql_store, vector_store, fake_llm, clock)

    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/availability", params={"space_type": "ev"})

    assert response.status_code == 200
    assert response.json() == {"items": [{"space_type": "ev", "total": 20, "available": 8}]}
    assert fake_llm.calls == []


def test_chat_returns_sources_and_continues_isolated_sessions(
    sql_store: SqlStore, vector_store: VectorStore, fake_llm: FakeLLM, clock: FakeClock
) -> None:
    vector_store.add_chunks(
        [
            Chunk(
                "entrance::01-test::0",
                "Which entrance is closest to the station?\nUse the Station Entrance.",
                "02_location_directions.md",
                "Which entrance is closest to the station?",
                0,
            )
        ]
    )
    fake_llm.reply = "Use the Station Entrance."
    client = _client(sql_store, vector_store, fake_llm, clock)

    answer = client.post("/chat", json={"message": "Which entrance is closest to the station?"})
    session_id = answer.json()["session_id"]
    assert answer.status_code == 200
    assert answer.json()["sources"][0]["source"] == "02_location_directions.md"

    first = client.post(
        "/chat",
        json={"message": "book standard tomorrow from 10 to 11", "session_id": session_id},
    )
    assert first.json()["flow"] == "booking"
    first_session_id = first.json()["session_id"]

    second = client.post("/chat", json={"message": "hello"})
    assert second.json()["intent"] == "greeting"
    assert second.json()["session_id"] != first_session_id

    continuation = client.post(
        "/chat",
        json={"message": "Alice Morgan", "session_id": first_session_id},
    )
    assert continuation.json()["flow"] == "booking"
    assert "plate" in continuation.json()["reply"].lower()


def test_chat_validation_does_not_echo_invalid_personal_input(
    sql_store: SqlStore, vector_store: VectorStore, fake_llm: FakeLLM, clock: FakeClock
) -> None:
    client = _client(sql_store, vector_store, fake_llm, clock)
    secret = "Alice Morgan CP-100001"

    response = client.post("/chat", json={"message": f"  {secret}  "})
    blank = client.post("/chat", json={"message": "  "})

    assert response.status_code == 200
    assert secret not in response.text
    assert blank.status_code == 422
    assert blank.json() == {"detail": "Request validation failed."}


def test_api_rejects_oversized_messages_and_invalid_space_types(
    sql_store: SqlStore, vector_store: VectorStore, fake_llm: FakeLLM, clock: FakeClock
) -> None:
    client = _client(sql_store, vector_store, fake_llm, clock)

    oversized = client.post("/chat", json={"message": "A" * 4001})
    invalid_space = client.get("/availability", params={"space_type": "spaceship"})

    assert oversized.status_code == 422
    assert invalid_space.status_code == 422
    assert oversized.json() == {"detail": "Request validation failed."}
    assert invalid_space.json() == {"detail": "Request validation failed."}
    assert fake_llm.calls == []


def test_chat_full_conversation_through_the_real_api(
    sql_store: SqlStore, vector_store: VectorStore, fake_llm: FakeLLM, clock: FakeClock
) -> None:
    """A realistic multi-turn conversation, driven entirely through POST /chat."""
    client = _client(sql_store, vector_store, fake_llm, clock)

    availability = client.post("/chat", json={"message": "What parking slots are available?"})
    session_id = availability.json()["session_id"]
    assert availability.json()["intent"] == "availability"

    ack = client.post("/chat", json={"message": "OK.", "session_id": session_id})
    assert "another parking type" in ack.json()["reply"]

    thanks = client.post("/chat", json={"message": "Thanks!", "session_id": session_id})
    assert "You're welcome" in thanks.json()["reply"]

    followup = client.post(
        "/chat", json={"message": "What about EV?", "session_id": session_id}
    )
    assert followup.json()["intent"] == "availability"
    assert "8 of 20" in followup.json()["reply"] or "EV" in followup.json()["reply"]

    switch_to_booking = client.post(
        "/chat", json={"message": "I want to book a parking space.", "session_id": session_id}
    )
    assert switch_to_booking.json()["flow"] == "booking"

    ok_as_date = client.post(
        "/chat", json={"message": "OK", "session_id": session_id}
    )
    assert ok_as_date.json()["flow"] == "booking"
    assert "couldn't read that date" in ok_as_date.json()["reply"]

    assert fake_llm.calls == []  # every turn above was answered without the LLM


def test_two_sessions_do_not_leak_conversational_context(
    sql_store: SqlStore, vector_store: VectorStore, fake_llm: FakeLLM, clock: FakeClock
) -> None:
    client = _client(sql_store, vector_store, fake_llm, clock)

    first = client.post("/chat", json={"message": "What parking slots are available?"})
    first_session = first.json()["session_id"]

    # A brand-new session has no prior topic, so "OK" gets the generic nudge, not the
    # "another parking type" reply that belongs to the first session's context.
    second_ok = client.post("/chat", json={"message": "OK."})
    second_session = second_ok.json()["session_id"]
    assert second_session != first_session
    assert "another parking type" not in second_ok.json()["reply"]

    first_ok = client.post("/chat", json={"message": "OK.", "session_id": first_session})
    assert "another parking type" in first_ok.json()["reply"]


def test_cancellation_conversation_survives_corrections_through_the_real_api(
    sql_store: SqlStore, vector_store: VectorStore, fake_llm: FakeLLM, clock: FakeClock
) -> None:
    """The exact reported bug conversation, replayed through POST /chat end to end:
    the cancellation flow must never be lost or silently switched to a lookup."""
    client = _client(sql_store, vector_store, fake_llm, clock)

    start = client.post("/chat", json={"message": "cancel"})
    session_id = start.json()["session_id"]
    assert start.json()["flow"] == "cancel"

    for message in ["i want to cancel", "CP-123", "CP-123345", "TN 0234 FD 90088"]:
        response = client.post(
            "/chat", json={"message": message, "session_id": session_id}
        )
        assert response.json()["intent"] == "cancel"
        assert response.json()["flow"] == "cancel"
        assert not response.json()["flow_done"]

    # Correct the reference (still mismatched: wrong plate from the previous attempt).
    still_wrong = client.post(
        "/chat", json={"message": "CP-100001", "session_id": session_id}
    )
    assert still_wrong.json()["flow"] == "cancel" and not still_wrong.json()["flow_done"]

    # Now correct the plate too: reference and plate finally match a real booking.
    corrected = client.post(
        "/chat", json={"message": "AB12CDE", "session_id": session_id}
    )
    assert "Do you want to cancel it" in corrected.json()["reply"]

    confirmed = client.post("/chat", json={"message": "yes", "session_id": session_id})
    assert confirmed.json()["flow_done"] is True
    assert "cancelled" in confirmed.json()["reply"]
    assert fake_llm.calls == []


def test_chat_service_errors_are_returned_without_internal_details(
    sql_store: SqlStore,
) -> None:
    class BrokenOrchestrator:
        def handle(self, _message: str, _session: object) -> None:
            raise RuntimeError("CP-100001 AB12CDE Alice Morgan")

    client = TestClient(
        create_app(orchestrator=BrokenOrchestrator(), sql_store=sql_store),
        raise_server_exceptions=False,
    )

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 500
    assert response.json() == {"detail": "The request could not be completed."}
    assert "CP-100001" not in response.text
