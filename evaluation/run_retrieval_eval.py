"""Run the retrieval evaluation dataset against the real vector store.

Usage::

    .venv\\Scripts\\python -m evaluation.run_retrieval_eval

Ingests the static knowledge base into the configured Chroma collection (real
MiniLM embeddings, no fakes), queries every question in
``evaluation.retrieval_dataset.RETRIEVAL_DATASET`` with the real
``VectorStore.query``, computes Precision/Recall@3 and @5 per question, and
writes both a per-question CSV and a full JSON report (per-question rows plus
the aggregate) under ``evaluation/results/``. This is a script, so printing
and writing files is intentional.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from app.config import configure_logging, get_settings
from app.data.ingest import run_ingestion
from app.data.vector_store import VectorStore
from evaluation.retrieval_dataset import RETRIEVAL_DATASET
from evaluation.retrieval_metrics import aggregate, evaluate_query

RESULTS_DIR = Path(__file__).parent / "results"
TOP_K = 5


def run(vector_store: VectorStore) -> tuple[list, object]:
    """Query every dataset question and compute metrics; return rows + aggregate."""
    results = []
    for item in RETRIEVAL_DATASET:
        retrieved = vector_store.query(item.query, k=TOP_K)
        retrieved_ids = [chunk.id for chunk in retrieved]
        result = evaluate_query(
            item.query_id, item.query, item.category, item.relevant_ids, retrieved_ids
        )
        results.append(result)
    return results, aggregate(results)


def _write_outputs(results: list, agg, settings) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / "retrieval_eval_latest.json"
    csv_path = RESULTS_DIR / "retrieval_eval_latest.csv"

    payload = {
        "generated_at": timestamp,
        "embedding_model": settings.embedding_model,
        "top_k": TOP_K,
        "query_count": len(results),
        "aggregate": asdict(agg),
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "query_id", "category", "query", "relevant_ids", "retrieved_ids",
                "precision_at_3", "recall_at_3", "precision_at_5", "recall_at_5",
            ]
        )  # fmt: skip
        for r in results:
            writer.writerow(
                [
                    r.query_id, r.category, r.query,
                    "|".join(r.relevant_ids), "|".join(r.retrieved_ids),
                    f"{r.precision_at_3:.3f}", f"{r.recall_at_3:.3f}",
                    f"{r.precision_at_5:.3f}", f"{r.recall_at_5:.3f}",
                ]
            )  # fmt: skip
    return json_path, csv_path


def main() -> int:
    configure_logging("WARNING")
    settings = get_settings()
    vector_store = VectorStore()
    chunk_count = run_ingestion(vector_store)
    print(f"Ingested {chunk_count} chunks ({vector_store.count()} stored).")

    results, agg = run(vector_store)

    print(f"\n{'query_id':<14}{'category':<22}{'P@3':>6}{'R@3':>6}{'P@5':>6}{'R@5':>6}")
    for r in results:
        print(
            f"{r.query_id:<14}{r.category:<22}"
            f"{r.precision_at_3:>6.2f}{r.recall_at_3:>6.2f}"
            f"{r.precision_at_5:>6.2f}{r.recall_at_5:>6.2f}"
        )

    print(f"\n{agg.query_count} queries evaluated.")
    print(f"Mean Precision@3: {agg.mean_precision_at_3:.3f}")
    print(f"Mean Recall@3:    {agg.mean_recall_at_3:.3f}")
    print(f"Mean Precision@5: {agg.mean_precision_at_5:.3f}")
    print(f"Mean Recall@5:    {agg.mean_recall_at_5:.3f}")

    json_path, csv_path = _write_outputs(results, agg, settings)
    print(f"\nSaved {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
