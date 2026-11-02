"""Measure real response-time, retrieval-time and DB-time distributions, plus
behaviour under concurrent load against the live API.

Usage::

    .venv\\Scripts\\python -m evaluation.performance_eval

The in-process measurements (orchestrator/retrieval/SQL) always run. The API
and concurrency sections additionally require the FastAPI backend to already
be running at ``API_BASE_URL`` (default ``http://127.0.0.1:8000``, e.g. via
``start_app.ps1``); if it is not reachable, those sections are skipped and
reported as such rather than inventing numbers. This is a script, so
printing and writing files is intentional.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np

from app.chatbot.llm import LazyLLMClient
from app.chatbot.orchestrator import ConversationSession, Orchestrator
from app.chatbot.rag import RagAnswerer
from app.config import configure_logging, get_settings
from app.data.sql_store import SqlStore
from app.data.vector_store import VectorStore

RESULTS_DIR = Path(__file__).parent / "results"
SEQUENTIAL_REQUESTS = 20
CONCURRENT_WORKERS = 10
CONCURRENT_REQUESTS_PER_WORKER = 3

CHAT_MESSAGES = [
    "What parking slots are available?",
    "How much does a standard space cost?",
    "What are your opening hours?",
    "What are your parking rules?",
    "Where is the entrance?",
    "What is the maximum vehicle height?",
    "Can I cancel my booking for free?",
    "Do you have EV charging?",
]


@dataclass(frozen=True)
class TimingStats:
    """Count plus mean/median/p95 of a list of millisecond durations."""

    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def _stats(durations_s: list[float]) -> TimingStats:
    ms = np.array(durations_s) * 1000.0
    return TimingStats(
        count=len(ms),
        mean_ms=round(float(np.mean(ms)), 2),
        median_ms=round(float(np.median(ms)), 2),
        p95_ms=round(float(np.percentile(ms, 95)), 2),
        min_ms=round(float(np.min(ms)), 2),
        max_ms=round(float(np.max(ms)), 2),
    )


def _time_calls(fn, args_list) -> list[float]:
    durations = []
    for args in args_list:
        start = time.perf_counter()
        fn(*args)
        durations.append(time.perf_counter() - start)
    return durations


def measure_orchestrator(settings) -> tuple[float, TimingStats]:
    """Cold-start time (first call, loads the embedding model) plus steady-state
    end-to-end chatbot response time for a representative message mix."""
    sql_store = SqlStore.from_url(settings.sqlite_url)
    vector_store = VectorStore()
    rag = RagAnswerer(vector_store, LazyLLMClient(settings))
    orchestrator = Orchestrator(sql_store=sql_store, rag=rag)
    session = ConversationSession()

    # Use a RAG-path message for the warmup call: that is what actually triggers the
    # embedding model's first load, not an SQL-only query like availability.
    start = time.perf_counter()
    orchestrator.handle("What are your parking rules?", session)
    cold_start_s = time.perf_counter() - start

    messages = [CHAT_MESSAGES[i % len(CHAT_MESSAGES)] for i in range(1, SEQUENTIAL_REQUESTS + 1)]
    durations = _time_calls(lambda m: orchestrator.handle(m, session), [(m,) for m in messages])
    return cold_start_s, _stats(durations)


def measure_retrieval(settings) -> tuple[float, TimingStats]:
    """Cold-start time plus steady-state vector-store-only query time
    (embedding + ChromaDB similarity search)."""
    vector_store = VectorStore()
    questions = [m for m in CHAT_MESSAGES if "cost" not in m and "available" not in m]

    start = time.perf_counter()
    vector_store.query(questions[0], k=5)
    cold_start_s = time.perf_counter() - start

    questions = [questions[i % len(questions)] for i in range(1, SEQUENTIAL_REQUESTS + 1)]
    durations = _time_calls(lambda q: vector_store.query(q, k=5), [(q,) for q in questions])
    return cold_start_s, _stats(durations)


def measure_sql(settings) -> TimingStats:
    """SQL-only query time across the three dynamic-data read paths."""
    sql_store = SqlStore.from_url(settings.sqlite_url)
    calls = [
        (sql_store.get_availability,),
        (sql_store.get_prices,),
        (lambda: sql_store.get_working_hours(0),),
    ]
    calls = [calls[i % len(calls)] for i in range(SEQUENTIAL_REQUESTS)]
    durations = []
    for (fn,) in calls:
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return _stats(durations)


def _api_reachable(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url}/health", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


def measure_api_sequential(base_url: str) -> TimingStats:
    """Real POST /chat latency, one request at a time, against the live backend."""
    durations = []
    with httpx.Client(base_url=base_url, timeout=15.0) as client:
        for i in range(SEQUENTIAL_REQUESTS):
            message = CHAT_MESSAGES[i % len(CHAT_MESSAGES)]
            start = time.perf_counter()
            client.post("/chat", json={"message": message})
            durations.append(time.perf_counter() - start)
    return _stats(durations)


@dataclass(frozen=True)
class ConcurrencyResult:
    """Outcome of firing many /chat requests at once against the live backend."""

    workers: int
    requests_per_worker: int
    total_requests: int
    success_count: int
    error_count: int
    wall_clock_s: float
    throughput_rps: float
    latency: TimingStats


def measure_api_concurrent(base_url: str) -> ConcurrencyResult:
    """Fire requests from multiple threads at once and record latency/errors."""

    def _one_request(i: int) -> tuple[bool, float]:
        message = CHAT_MESSAGES[i % len(CHAT_MESSAGES)]
        start = time.perf_counter()
        try:
            with httpx.Client(base_url=base_url, timeout=15.0) as client:
                response = client.post("/chat", json={"message": message})
            ok = response.status_code == 200
        except httpx.HTTPError:
            ok = False
        return ok, time.perf_counter() - start

    total = CONCURRENT_WORKERS * CONCURRENT_REQUESTS_PER_WORKER
    start_wall = time.perf_counter()
    outcomes: list[tuple[bool, float]] = []
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as pool:
        futures = [pool.submit(_one_request, i) for i in range(total)]
        for future in as_completed(futures):
            outcomes.append(future.result())
    wall_clock = time.perf_counter() - start_wall

    successes = [d for ok, d in outcomes if ok]
    errors = [d for ok, d in outcomes if not ok]
    return ConcurrencyResult(
        workers=CONCURRENT_WORKERS,
        requests_per_worker=CONCURRENT_REQUESTS_PER_WORKER,
        total_requests=total,
        success_count=len(successes),
        error_count=len(errors),
        wall_clock_s=round(wall_clock, 3),
        throughput_rps=round(total / wall_clock, 2) if wall_clock > 0 else 0.0,
        latency=_stats(successes or [0.0]),
    )


def main() -> int:
    configure_logging("WARNING")
    settings = get_settings()
    api_base_url = settings.api_base_url

    print("== In-process timings (no network) ==")
    orchestrator_cold_s, orchestrator_stats = measure_orchestrator(settings)
    print(f"Chatbot cold start (1st call, loads embedder): {orchestrator_cold_s * 1000:.0f} ms")
    print(f"Chatbot steady-state (orchestrator.handle):    {orchestrator_stats}")
    retrieval_cold_s, retrieval_stats = measure_retrieval(settings)
    print(f"Retrieval cold start (1st call):               {retrieval_cold_s * 1000:.0f} ms")
    print(f"Retrieval steady-state (VectorStore.query):    {retrieval_stats}")
    sql_stats = measure_sql(settings)
    print(f"Database (SqlStore reads):                     {sql_stats}")

    api_stats = None
    concurrency = None
    api_up = _api_reachable(api_base_url)
    print(f"\n== Live API at {api_base_url}: {'reachable' if api_up else 'NOT reachable'} ==")
    if api_up:
        api_stats = measure_api_sequential(api_base_url)
        print(f"POST /chat sequential: {api_stats}")
        concurrency = measure_api_concurrent(api_base_url)
        print(
            f"POST /chat concurrent ({concurrency.workers} workers x "
            f"{concurrency.requests_per_worker}): {concurrency.success_count} ok, "
            f"{concurrency.error_count} errors, {concurrency.throughput_rps} req/s, "
            f"latency {concurrency.latency}"
        )
    else:
        print("Skipped API and concurrency measurements: start the backend first.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "sequential_requests": SEQUENTIAL_REQUESTS,
        "orchestrator_cold_start_ms": round(orchestrator_cold_s * 1000, 2),
        "orchestrator": asdict(orchestrator_stats),
        "retrieval_cold_start_ms": round(retrieval_cold_s * 1000, 2),
        "retrieval": asdict(retrieval_stats),
        "sql": asdict(sql_stats),
        "api_reachable": api_up,
        "api_sequential": asdict(api_stats) if api_stats else None,
        "api_concurrent": (
            {**asdict(concurrency), "latency": asdict(concurrency.latency)}
            if concurrency
            else None
        ),
    }
    out_path = RESULTS_DIR / "performance_eval_latest.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
