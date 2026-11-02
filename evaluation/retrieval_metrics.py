"""Recall@K and Precision@K for the retrieval evaluation dataset.

Both are pure functions over plain lists of chunk IDs so they can be unit
tested without touching ChromaDB, the embedder, or the network.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Fraction of the top ``k`` retrieved chunks that are relevant.

    Precision@K = |relevant ∩ top-K retrieved| / K

    A query with no relevant chunks at all (``relevant_ids`` empty) is defined
    as precision 1.0 when nothing irrelevant was retrieved either (the model
    correctly found nothing to retrieve), else 0.0 - there is no "divide by
    zero relevant documents" case the way there is for recall.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = retrieved_ids[:k]
    if not relevant_ids:
        return 1.0 if not top_k else 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant_set)
    return hits / len(top_k) if top_k else 0.0


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Fraction of all relevant chunks that appear in the top ``k`` retrieved.

    Recall@K = |relevant ∩ top-K retrieved| / |relevant|

    A query with no relevant chunks is defined as recall 1.0 (there is
    nothing to miss), matching the "no relevant documents" case callers must
    handle explicitly rather than dividing by zero.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(relevant_set & top_k)
    return hits / len(relevant_set)


@dataclass(frozen=True)
class QueryResult:
    """Per-query retrieval outcome plus the metrics computed from it."""

    query_id: str
    query: str
    category: str
    relevant_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    precision_at_3: float
    recall_at_3: float
    precision_at_5: float
    recall_at_5: float


def evaluate_query(
    query_id: str,
    query: str,
    category: str,
    relevant_ids: Sequence[str],
    retrieved_ids: Sequence[str],
) -> QueryResult:
    """Compute Precision/Recall@3 and @5 for one query's retrieval result."""
    return QueryResult(
        query_id=query_id,
        query=query,
        category=category,
        relevant_ids=tuple(relevant_ids),
        retrieved_ids=tuple(retrieved_ids),
        precision_at_3=precision_at_k(retrieved_ids, relevant_ids, 3),
        recall_at_3=recall_at_k(retrieved_ids, relevant_ids, 3),
        precision_at_5=precision_at_k(retrieved_ids, relevant_ids, 5),
        recall_at_5=recall_at_k(retrieved_ids, relevant_ids, 5),
    )


@dataclass(frozen=True)
class AggregateMetrics:
    """Mean of each per-query metric across an evaluation run."""

    query_count: int
    mean_precision_at_3: float
    mean_recall_at_3: float
    mean_precision_at_5: float
    mean_recall_at_5: float


def aggregate(results: Sequence[QueryResult]) -> AggregateMetrics:
    """Average each metric across all query results; raises on an empty run."""
    if not results:
        raise ValueError("cannot aggregate an empty list of query results")
    n = len(results)
    return AggregateMetrics(
        query_count=n,
        mean_precision_at_3=sum(r.precision_at_3 for r in results) / n,
        mean_recall_at_3=sum(r.recall_at_3 for r in results) / n,
        mean_precision_at_5=sum(r.precision_at_5 for r in results) / n,
        mean_recall_at_5=sum(r.recall_at_5 for r in results) / n,
    )


__all__ = [
    "AggregateMetrics",
    "QueryResult",
    "aggregate",
    "evaluate_query",
    "precision_at_k",
    "recall_at_k",
]
