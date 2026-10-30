"""Stage 1 smoke test: seed SQL, ingest static docs, check retrieval quality.

Usage: ``python -m scripts.smoke_test_data``

Exits with code 1 if fewer than ``MIN_RANK1`` of the queries return the
expected chunk at rank 1. Prints ``RANK1: n/5`` and ``RESULT: PASS|FAIL``.
This is a script, so printing is intentional.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

from app.config import configure_logging, get_settings
from app.data.ingest import run_ingestion
from app.data.seed import seed
from app.data.sql_store import SqlStore
from app.data.vector_store import RetrievedChunk, VectorStore

TOP_N = 3
MIN_RANK1 = 4


@dataclass(frozen=True)
class Expectation:
    """A query and a predicate the rank-1 chunk must satisfy."""

    query: str
    description: str
    matches: Callable[[RetrievedChunk], bool]


def _source_starts(prefix: str) -> Callable[[RetrievedChunk], bool]:
    return lambda chunk: chunk.source.startswith(prefix)


def _id_contains(fragment: str) -> Callable[[RetrievedChunk], bool]:
    return lambda chunk: fragment in chunk.id


EXPECTATIONS: tuple[Expectation, ...] = (
    Expectation(
        "where is the entrance",
        "source starts with 02_location_directions",
        _source_starts("02_location_directions"),
    ),
    Expectation(
        "can I cancel my booking",
        "source starts with 05_cancellation_policy",
        _source_starts("05_cancellation_policy"),
    ),
    Expectation(
        "do you have EV chargers",
        "source starts with 08_ev_charging",
        _source_starts("08_ev_charging"),
    ),
    Expectation(
        "what is the height limit",
        "chunk id contains maximum-vehicle-height",
        _id_contains("maximum-vehicle-height"),
    ),
    Expectation(
        "I lost my booking reference",
        "source starts with 10_faq",
        _source_starts("10_faq"),
    ),
)


def print_sql_snapshot(store: SqlStore) -> None:
    """Print live availability and prices from the SQL layer."""
    print("== SQL availability ==")
    for row in store.get_availability():
        print(f"  {row.space_type:<9} {row.available:>3}/{row.total}")
    print("== SQL prices (GBP) ==")
    for row in store.get_prices():
        print(f"  {row.space_type:<9} {row.price_per_hour:>5.2f}/h  {row.price_per_day:>6.2f}/day")


def run_retrieval_checks(vector_store: VectorStore) -> int:
    """Run every expectation, print the top chunks, and return the rank-1 hit count."""
    hits = 0
    for expectation in EXPECTATIONS:
        results = vector_store.query(expectation.query, k=TOP_N)
        top = results[0] if results else None
        ok = top is not None and expectation.matches(top)
        hits += int(ok)
        print(f"\nQuery: {expectation.query!r}  (expect: {expectation.description})")
        for rank, chunk in enumerate(results, start=1):
            print(f"  {rank}. {chunk.score:.3f}  {chunk.id}")
        print(f"  rank-1 {'OK' if ok else 'MISS'}")
    return hits


def main() -> int:
    """Seed, ingest, print SQL data, run retrieval checks; return the exit code."""
    configure_logging("WARNING")
    settings = get_settings()

    sql_store = SqlStore.from_url(settings.sqlite_url)
    seed(sql_store)
    print_sql_snapshot(sql_store)

    vector_store = VectorStore()
    chunk_count = run_ingestion(vector_store)
    print(f"\n== Vector store: {chunk_count} chunks ingested, {vector_store.count()} stored ==")

    hits = run_retrieval_checks(vector_store)
    total = len(EXPECTATIONS)
    passed = hits >= MIN_RANK1
    print(f"\nRANK1: {hits}/{total}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
