"""Shared fixtures. No test here may download a model or touch parking.db / chroma_db."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import chromadb
import pytest

from app.data.vector_store import VectorStore

EMBED_DIM = 16


def fake_embed(texts: Sequence[str]) -> list[list[float]]:
    """Deterministic bag-of-words hashing embedder; similar texts -> similar vectors."""
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * EMBED_DIM
        for word in text.lower().split():
            digest = hashlib.md5(word.encode("utf-8")).digest()  # noqa: S324 - not security
            vector[digest[0] % EMBED_DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


@pytest.fixture
def vector_store(tmp_path) -> VectorStore:
    """A VectorStore on a temporary Chroma path with the fake embedder injected."""
    client = chromadb.PersistentClient(
        path=str(tmp_path / "chroma"), settings=chromadb.Settings(anonymized_telemetry=False)
    )
    return VectorStore(client=client, embed_fn=fake_embed, top_k=3, collection_name="test")
