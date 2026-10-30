"""Tests for app.chatbot.rag with a fake embedder and a fake LLM."""

from __future__ import annotations

import pytest

from app.chatbot.llm import LLMError
from app.chatbot.rag import UNAVAILABLE_MESSAGE, RagAnswerer, format_context
from app.data.ingest import Chunk
from app.data.vector_store import RetrievedChunk, VectorStore
from tests.conftest import FakeLLM

CHUNKS = [
    Chunk(
        "05_cancellation_policy::01-how-late-can-i-cancel::0",
        "How late can I cancel my booking for free?\nFree when cancelled 24 hours before.",
        "05_cancellation_policy.md",
        "How late can I cancel my booking for free at CityPark Central?",
        0,
    ),
    Chunk(
        "08_ev_charging::01-where-are-ev-bays::0",
        "Where are EV bays?\nEV bays are on level 1 with chargers.",
        "08_ev_charging.md",
        "Where are EV bays at CityPark Central and who can use them?",
        0,
    ),
]


@pytest.fixture
def loaded_store(vector_store: VectorStore) -> VectorStore:
    vector_store.add_chunks(CHUNKS)
    return vector_store


def test_grounded_answer_passes_context_and_returns_sources(
    loaded_store: VectorStore, fake_llm: FakeLLM
) -> None:
    fake_llm.reply = "You can cancel free of charge 24 hours before the start."
    rag = RagAnswerer(loaded_store, fake_llm, min_score=0.0, top_k=2)

    answer = rag.answer("cancel my booking free 24 hours before")

    assert answer.grounded is True
    assert answer.text == fake_llm.reply
    assert len(fake_llm.calls) == 1
    system, user = fake_llm.calls[0]
    assert user == "cancel my booking free 24 hours before"
    assert "Free when cancelled 24 hours before." in system
    assert "[1]" in system and UNAVAILABLE_MESSAGE in system
    assert answer.sources[0].source == "05_cancellation_policy.md"
    assert answer.sources[0].section.startswith("How late can I cancel")
    assert 0.0 <= answer.sources[0].score <= 1.0


def test_below_min_score_returns_unavailable_without_calling_llm(
    loaded_store: VectorStore, fake_llm: FakeLLM
) -> None:
    rag = RagAnswerer(loaded_store, fake_llm, min_score=0.99, top_k=2)
    answer = rag.answer("completely unrelated quantum physics question")
    assert answer.text == UNAVAILABLE_MESSAGE
    assert answer.grounded is False and answer.sources == ()
    assert fake_llm.calls == []


def test_empty_collection_returns_unavailable_without_calling_llm(
    vector_store: VectorStore, fake_llm: FakeLLM
) -> None:
    rag = RagAnswerer(vector_store, fake_llm, min_score=0.0)
    answer = rag.answer("anything")
    assert answer.text == UNAVAILABLE_MESSAGE
    assert fake_llm.calls == []


def test_llm_saying_unavailable_drops_sources(loaded_store: VectorStore, fake_llm: FakeLLM) -> None:
    fake_llm.reply = UNAVAILABLE_MESSAGE
    rag = RagAnswerer(loaded_store, fake_llm, min_score=0.0)
    answer = rag.answer("ev bays level chargers")
    assert answer.grounded is False and answer.sources == ()


def test_llm_errors_propagate(loaded_store: VectorStore, fake_llm: FakeLLM) -> None:
    fake_llm.error = LLMError("boom")
    rag = RagAnswerer(loaded_store, fake_llm, min_score=0.0)
    with pytest.raises(LLMError):
        rag.answer("ev bays level chargers")


def test_format_context_numbers_passages() -> None:
    text = format_context(
        [
            RetrievedChunk("a", "Body A", "a.md", "Section A", 0.9),
            RetrievedChunk("b", "Body B", "b.md", "Section B", 0.8),
        ]
    )
    assert text == "[1] (Section A - a.md)\nBody A\n\n[2] (Section B - b.md)\nBody B"
