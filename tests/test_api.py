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
