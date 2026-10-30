"""Persistent vector store for the static CityPark Central knowledge base.

Wraps a ChromaDB ``PersistentClient`` and a sentence-transformers embedding
model. Only static policy text lives here; dynamic data (availability,
prices, hours, reservations) stays in ``app.data.sql_store``.

Scores
------
The collection uses cosine distance (``hnsw:space = cosine``). Chroma returns
``distance = 1 - cosine_similarity``, so ``RetrievedChunk.score`` is reported as
``1 - distance``, i.e. the cosine similarity in ``[-1, 1]`` where higher is
more similar (in practice ``[0, 1]`` for normalized MiniLM embeddings).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "citypark_static"
COLLECTION_METADATA: dict[str, str] = {"hnsw:space": "cosine"}
_EMBED_BATCH_SIZE = 64

EmbedFn = Callable[[Sequence[str]], list[list[float]]]
"""Maps a batch of texts to one embedding vector per text."""


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------
class VectorStoreError(Exception):
    """Base class for vector-store errors."""


class EmptyCollectionError(VectorStoreError):
    """Raised when querying a collection that holds no chunks."""

    def __init__(self, collection: str) -> None:
        super().__init__(
            f"Vector collection '{collection}' is empty. "
            "Run `python -m app.data.ingest` to ingest the static knowledge base."
        )


# --------------------------------------------------------------------------
# Data objects
# --------------------------------------------------------------------------
class ChunkLike(Protocol):
    """Minimal shape ``add_chunks`` needs; ``app.data.ingest.Chunk`` satisfies it."""

    @property
    def id(self) -> str: ...

    @property
    def text(self) -> str: ...

    @property
    def source(self) -> str: ...

    @property
    def section(self) -> str: ...

    @property
    def chunk_index(self) -> int: ...


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by ``VectorStore.query`` with its similarity score."""

    id: str
    text: str
    source: str
    section: str
    score: float


# --------------------------------------------------------------------------
# Embedding helpers
# --------------------------------------------------------------------------
def _load_sentence_transformer(model_name: str) -> EmbedFn:
    """Load a sentence-transformers model lazily and wrap it as an ``EmbedFn``."""
    from sentence_transformers import SentenceTransformer  # heavy import, deferred

    logger.info("Loading embedding model %s", model_name)
    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        vectors = model.encode(
            list(texts),
            batch_size=_EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]

    return embed


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------
class VectorStore:
    """Chroma-backed store for static knowledge chunks.

    All collaborators are injectable for tests: pass ``client`` to avoid touching
    the configured ``chroma_path`` and ``embed_fn`` to avoid downloading a model.
    The embedding model is loaded on first use, never at import time.
    """

    def __init__(
        self,
        *,
        path: str | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str | None = None,
        top_k: int | None = None,
        client: ClientAPI | None = None,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        settings = get_settings()
        self._path = path or settings.chroma_path
        self._collection_name = collection_name
        self._model_name = model_name or settings.embedding_model
        self._top_k = top_k or settings.top_k
        self._client = client or chromadb.PersistentClient(
            path=self._path, settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self._embed_fn = embed_fn
        self._collection = self._open_collection()

    # ---- internals -------------------------------------------------------
    def _open_collection(self) -> Collection:
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata=COLLECTION_METADATA,
            embedding_function=None,  # embeddings are always supplied explicitly
        )

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._embed_fn is None:
            self._embed_fn = _load_sentence_transformer(self._model_name)
        return self._embed_fn(texts)

    # ---- public API ------------------------------------------------------
    @property
    def collection_name(self) -> str:
        """Name of the underlying Chroma collection."""
        return self._collection_name

    def count(self) -> int:
        """Number of chunks currently stored."""
        return self._collection.count()

    def add_chunks(self, chunks: Iterable[ChunkLike]) -> int:
        """Embed and upsert chunks; re-adding the same IDs overwrites, never duplicates.

        Returns the number of chunks written.
        """
        items = list(chunks)
        if not items:
            logger.warning("add_chunks called with no chunks; nothing written")
            return 0

        ids = [chunk.id for chunk in items]
        if len(set(ids)) != len(ids):
            raise VectorStoreError("Chunk IDs must be unique within one add_chunks call.")

        documents = [chunk.text for chunk in items]
        metadatas: list[dict[str, Any]] = [
            {"source": chunk.source, "section": chunk.section, "chunk_index": chunk.chunk_index}
            for chunk in items
        ]
        embeddings = self._embed(documents)
        self._collection.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )
        logger.info(
            "Upserted %d chunks into '%s' (total now %d)",
            len(items),
            self._collection_name,
            self.count(),
        )
        return len(items)

    def query(self, text: str, k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-``k`` most similar chunks (default ``settings.top_k``).

        Results are ordered by descending score. ``k`` is clamped to the number of
        stored chunks. Raises ``EmptyCollectionError`` if nothing has been ingested
        and ``ValueError`` for a blank query or non-positive ``k``.
        """
        if not text or not text.strip():
            raise ValueError("Query text must not be empty.")
        top_k = self._top_k if k is None else k
        if top_k < 1:
            raise ValueError("k must be at least 1.")

        total = self.count()
        if total == 0:
            raise EmptyCollectionError(self._collection_name)
        n_results = min(top_k, total)

        [embedding] = self._embed([text])
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        ids = result["ids"][0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks = [
            RetrievedChunk(
                id=chunk_id,
                text=document or "",
                source=str((metadata or {}).get("source", "")),
                section=str((metadata or {}).get("section", "")),
                score=1.0 - float(distance),
            )
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]
        logger.debug("query k=%d returned %d chunks", n_results, len(chunks))
        return chunks

    def reset(self) -> None:
        """Delete every chunk by dropping and recreating the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._open_collection()
        logger.info("Collection '%s' reset", self._collection_name)


__all__ = [
    "COLLECTION_METADATA",
    "DEFAULT_COLLECTION",
    "ChunkLike",
    "EmbedFn",
    "EmptyCollectionError",
    "RetrievedChunk",
    "VectorStore",
    "VectorStoreError",
]
