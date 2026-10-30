"""Tests for app.data.vector_store using a fake embedder and a temporary Chroma path."""

from __future__ import annotations

from pathlib import Path

import chromadb
import pytest

from app.data.ingest import Chunk, run_ingestion
from app.data.vector_store import (
    EmptyCollectionError,
    RetrievedChunk,
    VectorStore,
    VectorStoreError,
)
from tests.conftest import fake_embed

CHUNKS = [
    Chunk("a::01-cancel::0", "Can I cancel?\nCancel free 24 hours before.", "a.md", "Cancel", 0),
    Chunk("b::01-ev::0", "EV bays?\nEV bays are on level 1 with chargers.", "b.md", "EV", 0),
    Chunk("c::01-height::0", "Height limit?\nMaximum vehicle height is 2 m.", "c.md", "Height", 0),
]


def test_query_on_empty_collection_raises_clear_error(vector_store: VectorStore) -> None:
    assert vector_store.count() == 0
    with pytest.raises(EmptyCollectionError, match="app.data.ingest"):
        vector_store.query("anything")


def test_add_count_and_query_returns_relevant_chunk_first(vector_store: VectorStore) -> None:
    assert vector_store.add_chunks(CHUNKS) == 3
    assert vector_store.count() == 3

    results = vector_store.query("cancel 24 hours before start", k=2)
    assert len(results) == 2
    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert results[0].id == "a::01-cancel::0"
    assert results[0].source == "a.md" and results[0].section == "Cancel"
    assert results[0].score >= results[1].score
    assert -1.0 <= results[1].score <= results[0].score <= 1.0 + 1e-6


def test_default_k_comes_from_settings_and_is_clamped(vector_store: VectorStore) -> None:
    vector_store.add_chunks(CHUNKS[:2])
    assert len(vector_store.query("ev chargers")) == 2  # top_k=3 clamped to count=2
    assert len(vector_store.query("ev chargers", k=10)) == 2


def test_upsert_is_idempotent_and_updates_text(vector_store: VectorStore) -> None:
    vector_store.add_chunks(CHUNKS)
    vector_store.add_chunks(CHUNKS)
    assert vector_store.count() == 3

    updated = Chunk("c::01-height::0", "Height limit?\nNow 2.1 m.", "c.md", "Height", 0)
    vector_store.add_chunks([updated])
    assert vector_store.count() == 3
    hit = vector_store.query("height limit 2.1", k=1)[0]
    assert hit.id == "c::01-height::0" and "2.1" in hit.text


def test_reset_empties_collection(vector_store: VectorStore) -> None:
    vector_store.add_chunks(CHUNKS)
    vector_store.reset()
    assert vector_store.count() == 0
    with pytest.raises(EmptyCollectionError):
        vector_store.query("cancel")
    vector_store.add_chunks(CHUNKS[:1])  # still usable after reset
    assert vector_store.count() == 1


def test_invalid_inputs(vector_store: VectorStore) -> None:
    assert vector_store.add_chunks([]) == 0
    with pytest.raises(VectorStoreError):
        vector_store.add_chunks([CHUNKS[0], CHUNKS[0]])
    vector_store.add_chunks(CHUNKS)
    with pytest.raises(ValueError):
        vector_store.query("   ")
    with pytest.raises(ValueError):
        vector_store.query("ev", k=0)


def test_embedding_model_is_lazy_and_injectable(tmp_path: Path) -> None:
    calls: list[int] = []

    def counting_embed(texts):
        calls.append(len(texts))
        return fake_embed(texts)

    client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    store = VectorStore(client=client, embed_fn=counting_embed, top_k=1)
    assert calls == []  # constructing the store embeds nothing
    store.add_chunks(CHUNKS)
    store.query("ev")
    assert calls == [3, 1]


def test_persistence_across_instances(tmp_path: Path) -> None:
    path = str(tmp_path / "chroma")
    first = VectorStore(path=path, embed_fn=fake_embed, collection_name="persist")
    first.add_chunks(CHUNKS)
    second = VectorStore(path=path, embed_fn=fake_embed, collection_name="persist")
    assert second.count() == 3
    assert second.query("height", k=1)[0].id == "c::01-height::0"


def test_run_ingestion_end_to_end_with_custom_corpus(
    tmp_path: Path, vector_store: VectorStore
) -> None:
    docs = tmp_path / "static"
    docs.mkdir()
    (docs / "01_a.md").write_text("# A\n\n## First at CityPark Central?\nBody one.\n", "utf-8")
    (docs / "02_b.md").write_text(
        "# B\n\n## Second?\nBody two.\n\n## Third?\nBody three.\n", "utf-8"
    )

    written = run_ingestion(
        vector_store, static_dir=docs, expected_chunks=3, expected_files=2
    )
    assert written == 3 and vector_store.count() == 3
    # Re-running (with and without reset) keeps exactly 3 chunks.
    run_ingestion(vector_store, static_dir=docs, expected_chunks=3, expected_files=2)
    run_ingestion(vector_store, static_dir=docs, reset=True, expected_chunks=3, expected_files=2)
    assert vector_store.count() == 3
    assert {r.source for r in vector_store.query("body", k=3)} == {"01_a.md", "02_b.md"}


def test_run_ingestion_fails_loudly_on_count_mismatch(
    tmp_path: Path, vector_store: VectorStore
) -> None:
    from app.data.ingest import IngestionError

    docs = tmp_path / "static"
    docs.mkdir()
    (docs / "01_a.md").write_text("## Only?\nBody.\n", "utf-8")
    with pytest.raises(IngestionError, match="Expected 38 chunks"):
        run_ingestion(vector_store, static_dir=docs, expected_files=None)
    assert vector_store.count() == 0  # nothing written on failure
