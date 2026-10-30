"""Shared fixtures. No test here may download a model, call an LLM, or touch
parking.db / chroma_db."""

from __future__ import annotations

import datetime as dt
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import chromadb
import pytest

from app.data.seed import seed
from app.data.sql_store import SqlStore, create_db_engine
from app.data.vector_store import VectorStore

EMBED_DIM = 16
FIXED_NOW = dt.datetime(2030, 1, 7, 9, 0)  # a Monday


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


class FakeClock:
    """Mutable clock injected into stores and flows."""

    def __init__(self, now: dt.datetime = FIXED_NOW) -> None:
        self.now = now

    def __call__(self) -> dt.datetime:
        return self.now


@dataclass
class FakeLLM:
    """Records prompts and returns canned replies; raises if ``error`` is set."""

    reply: str = "Canned answer."
    error: Exception | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.error is not None:
            raise self.error
        return self.reply


@pytest.fixture
def vector_store(tmp_path) -> VectorStore:
    """A VectorStore on a temporary Chroma path with the fake embedder injected."""
    client = chromadb.PersistentClient(
        path=str(tmp_path / "chroma"), settings=chromadb.Settings(anonymized_telemetry=False)
    )
    return VectorStore(client=client, embed_fn=fake_embed, top_k=3, collection_name="test")


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sql_store(clock: FakeClock) -> SqlStore:
    """Seeded in-memory SqlStore driven by the fake clock."""
    store = SqlStore(create_db_engine("sqlite://"), now=clock)
    seed(store, today=FIXED_NOW.date())
    return store


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
