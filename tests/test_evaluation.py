"""Tests for the Stage 4 retrieval evaluation metrics (pure functions, no I/O)."""

from __future__ import annotations

import pytest

from evaluation.retrieval_dataset import RETRIEVAL_DATASET
from evaluation.retrieval_metrics import (
    aggregate,
    evaluate_query,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k_counts_relevant_hits_in_top_k() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = ["c", "z"]
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(0.5)  # only "c" in top 3
    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(0.5)  # "z" never retrieved


def test_recall_at_k_perfect_match() -> None:
    assert recall_at_k(["a", "b"], ["a", "b"], k=5) == pytest.approx(1.0)


def test_recall_at_k_with_no_relevant_documents_is_defined_as_one() -> None:
    # A query with no relevant documents has nothing to miss.
    assert recall_at_k(["a", "b", "c"], [], k=3) == 1.0
    assert recall_at_k([], [], k=3) == 1.0


def test_precision_at_k_counts_relevant_fraction_of_top_k() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = ["a", "c"]
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)
    assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 5)


def test_precision_at_k_with_no_relevant_documents() -> None:
    # Nothing relevant exists: precision is 1.0 only if nothing was retrieved either.
    assert precision_at_k([], [], k=3) == 1.0
    assert precision_at_k(["a", "b"], [], k=3) == 0.0


def test_precision_and_recall_at_k_reject_non_positive_k() -> None:
    with pytest.raises(ValueError):
        precision_at_k(["a"], ["a"], k=0)
    with pytest.raises(ValueError):
        recall_at_k(["a"], ["a"], k=-1)


def test_precision_at_k_handles_fewer_retrieved_than_k() -> None:
    # Boundary: only 2 results retrieved when k=5; precision is over what was
    # actually retrieved, not artificially diluted by the requested k.
    assert precision_at_k(["a", "b"], ["a"], k=5) == pytest.approx(0.5)


def test_evaluate_query_computes_all_four_metrics_consistently() -> None:
    result = evaluate_query(
        "q1", "a question", "category", ["x"], ["x", "y", "z", "w", "v"]
    )
    assert result.query_id == "q1"
    assert result.precision_at_3 == pytest.approx(1 / 3)
    assert result.recall_at_3 == 1.0
    assert result.precision_at_5 == pytest.approx(1 / 5)
    assert result.recall_at_5 == 1.0


def test_aggregate_averages_each_metric_across_queries() -> None:
    results = [
        evaluate_query("q1", "q1", "cat", ["a"], ["a", "b", "c"]),
        evaluate_query("q2", "q2", "cat", ["z"], ["x", "y", "w"]),
    ]
    agg = aggregate(results)
    assert agg.query_count == 2
    assert agg.mean_recall_at_3 == pytest.approx(0.5)  # one hit, one miss
    assert agg.mean_precision_at_3 == pytest.approx((1 / 3 + 0) / 2)


def test_aggregate_rejects_empty_results() -> None:
    with pytest.raises(ValueError):
        aggregate([])


def test_retrieval_dataset_entries_are_well_formed() -> None:
    # Boundary/sanity check on the dataset itself: unique IDs, non-empty
    # queries, and every entry exercises a known category.
    assert len(RETRIEVAL_DATASET) > 0
    ids = [item.query_id for item in RETRIEVAL_DATASET]
    assert len(ids) == len(set(ids))
    for item in RETRIEVAL_DATASET:
        assert item.query.strip()
        assert item.category


def test_retrieval_dataset_includes_a_no_relevant_document_case() -> None:
    # Outcome 3.A explicitly requires covering "no relevant documents".
    no_match = [item for item in RETRIEVAL_DATASET if not item.relevant_ids]
    assert no_match, "dataset must include at least one query with no relevant chunks"
