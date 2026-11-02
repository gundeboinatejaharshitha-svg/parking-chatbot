# CityPark Central Chatbot — Stage 4 Evaluation Report

All numbers in this report were produced by actually running
`evaluation/run_retrieval_eval.py` and `evaluation/performance_eval.py` against the real
ingested knowledge base, the real MiniLM embedding model, and (for the API/concurrency section)
the real running FastAPI backend — not estimated or hand-written. Raw machine-readable output is
in `evaluation/results/retrieval_eval_latest.{json,csv}` and
`evaluation/results/performance_eval_latest.json`; re-run either script at any time to regenerate
this report's numbers from scratch.

## 1. Project and environment

| | |
|---|---|
| Project | CityPark Central Chatbot (`parking-chatbot`) |
| Evaluation date | 2026-11-01 |
| Python | 3.11.9 |
| OS | Windows 10 (build 26200) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (v3.3.1) |
| Vector store | ChromaDB 0.6.3, persistent, local (`./chroma_db`) |
| Relational store | SQLite via SQLAlchemy 2.0, file-backed (`./parking.db`) |
| API framework | FastAPI 0.115.6 / Uvicorn |
| LLM | Not configured in this evaluation environment (placeholder `LLM_API_KEY`) — retrieval and performance numbers below are therefore for the **retrieval and application layers**, not for LLM generation quality; see [§9](#9-failed-test-cases-and-limitations). |

## 2. Evaluation dataset

`evaluation/retrieval_dataset.py` — 24 hand-labeled questions against the real 38-chunk
knowledge base (10 Markdown files under `app/data/static/`, one chunk per FAQ-style Q&A pair).
Ground truth was built by inspecting the actual chunk IDs produced by
`app.data.ingest.load_static_chunks()`, then writing **paraphrased** questions (not copies of a
chunk's own heading) so retrieval is genuinely exercised rather than trivially exact-matched.

| Category | Count | Example |
|---|---|---|
| `parking_rules` | 3 | "What are the parking rules I need to follow here?" |
| `parking_charges` | 3 | "What payment methods do you accept for parking charges?" |
| `location` | 3 | "Which entrance should I use if I'm arriving from the train station?" |
| `booking_instructions` | 3 | "How do I make a parking reservation and what information is needed?" |
| `cancellation_policy` | 3 | "If I cancel my reservation, will I get my money back?" |
| `general_information` | 7 | "Is there a maximum vehicle height for entering the car park?" |
| `out_of_scope` (no relevant chunk) | 2 | "What is the capital of France?" |

21 of 24 questions have exactly one relevant chunk; one (`cancel-01`) has two (the question
spans two FAQ entries); two (`no-match-*`) have zero, to exercise the "no relevant documents"
case explicitly.

## 3. Retrieval evaluation methodology

For each question: embed it with the real MiniLM model, retrieve the top-5 chunks from the real
ChromaDB collection (`VectorStore.query`), and compare the retrieved chunk IDs against the
hand-labeled relevant IDs using:

- **Recall@K** = `|relevant ∩ top-K| / |relevant|` (defined as `1.0` when a question has no
  relevant chunks — nothing to miss).
- **Precision@K** = `|relevant ∩ top-K| / K` (defined as `1.0` for a no-relevant-chunk question
  only if nothing was retrieved either, else `0.0`).

Implemented in `evaluation/retrieval_metrics.py` as pure functions, unit-tested in
`tests/test_evaluation.py` (positive cases, empty-relevant-set boundary, `k` validation, fewer
retrieved than `k`).

**Note on Precision@K with single-relevant-chunk questions:** 21 of 24 questions have exactly
one relevant chunk. For those, Precision@3 is mathematically capped at `1/3 ≈ 0.33` and
Precision@5 at `1/5 = 0.20` *even when retrieval is perfect* — the formula divides by `K`, not by
the number of relevant documents. Recall@K is the more informative metric for this dataset's
question types; Precision@K is included because it is explicitly required, and its ceiling is
reported alongside it rather than left to look like a retrieval failure.

## 4. Recall@3 and Recall@5

| Category | n | Recall@3 | Recall@5 |
|---|---:|---:|---:|
| parking_rules | 3 | 1.00 | 1.00 |
| parking_charges | 3 | 1.00 | 1.00 |
| location | 3 | 1.00 | 1.00 |
| booking_instructions | 3 | 0.67 | 1.00 |
| cancellation_policy | 3 | 1.00 | 1.00 |
| general_information | 7 | 1.00 | 1.00 |
| out_of_scope | 2 | 1.00 | 1.00 |
| **Mean (all 24 questions)** | **24** | **0.958** | **1.000** |

Only one question missed its relevant chunk within top-3 (recovered at top-5): `booking-02`,
"What do I need to show at the gate to enter the car park?", targeting
`04_booking_process::02-what-do-i-need-to-check-in-and-enter::0`. The correct chunk ranked 4th
(score 0.393); three semantically adjacent chunks (parking rules, parking levels, the booking
*process* chunk) scored marginally higher (0.408–0.427). This is a genuine, if narrow, retrieval
limitation — see [§9](#9-failed-test-cases-and-limitations).

## 5. Precision@3 and Precision@5

| Category | n | Precision@3 | Precision@5 |
|---|---:|---:|---:|
| parking_rules | 3 | 0.33 | 0.20 |
| parking_charges | 3 | 0.33 | 0.20 |
| location | 3 | 0.33 | 0.20 |
| booking_instructions | 3 | 0.22 | 0.20 |
| cancellation_policy | 3 | 0.44 | 0.27 |
| general_information | 7 | 0.33 | 0.20 |
| out_of_scope | 2 | 0.00 | 0.00 |
| **Mean (all 24 questions)** | **24** | **0.306** | **0.192** |

As explained in §3, 0.33/0.20 is the ceiling for a singleton-relevant question, so these
categories are effectively at their maximum possible precision. `cancellation_policy` is above
that ceiling (0.44/0.27) because `cancel-01` has two relevant chunks. `out_of_scope` is `0.0` by
definition — ChromaDB always returns its nearest neighbors even when none are truly relevant;
this is exactly why the production RAG pipeline (`RagAnswerer`) additionally applies
`RAG_MIN_SCORE` (0.3) before ever using a retrieved chunk, which is not reflected in this raw
retrieval metric — see §9 for a case where even that floor is not quite enough.

## 6. Response-time measurements

In-process, no network (`evaluation/performance_eval.py`, 20 sequential calls after a dedicated
warm-up call, same process):

| Measurement | Cold start (1st call) | Mean | Median | P95 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Chatbot (`Orchestrator.handle`, mixed SQL + RAG messages) | 11,630 ms | 47.5 ms | 44.0 ms | 101.4 ms | 0.5 ms | 103.9 ms |

Cold start loads the Sentence-Transformers model and Presidio/spaCy components for the first
time in the process; steady-state calls are two to three orders of magnitude faster. This
matches the project's existing lazy-initialization design (`LazyLLMClient`, lazy `VectorStore`
embedder, lazy Presidio analyzer) — the cost is paid once per process, not per request.

Live HTTP, against the already-running backend (`POST /chat`, 20 sequential requests):

| Measurement | Mean | Median | P95 | Min | Max |
|---|---:|---:|---:|---:|---:|
| `POST /chat` (sequential) | 59.1 ms | 50.6 ms | 161.8 ms | 1.8 ms | 181.4 ms |

## 7. Retrieval-time and database-time measurements

| Measurement | Cold start | Mean | Median | P95 |
|---|---:|---:|---:|---:|
| Retrieval only (`VectorStore.query`, embed + ChromaDB search) | 3,590 ms | 10.7 ms | 10.4 ms | 12.5 ms |
| Database only (`SqlStore` reads: availability/prices/hours) | n/a (no model to load) | 0.44 ms | 0.29 ms | 1.05 ms |

Database reads are sub-millisecond, as expected for local SQLite reads with no joins. Retrieval
is the dominant cost in a RAG-path chatbot turn (≈10 ms vs ≈0.4 ms for SQL), which is why the
orchestrator never routes dynamic facts (availability/prices/hours) through retrieval.

## 8. Concurrent request test results

10 worker threads × 3 requests each = 30 concurrent `POST /chat` requests against the live,
already-warm backend (`ThreadPoolExecutor`, `evaluation/performance_eval.py`):

| Metric | Value |
|---|---:|
| Total requests | 30 |
| Successful (HTTP 200) | 30 |
| Errors | 0 |
| Wall-clock time | 2.81 s |
| Throughput | 10.68 req/s |
| Latency — mean | 818.8 ms |
| Latency — median | 807.3 ms |
| Latency — P95 | 1,095.1 ms |
| Latency — max | 1,234.0 ms |

Zero errors under this load: correctness held up. Per-request latency rose roughly 15× versus
sequential (807 ms vs 51 ms median) because this is a single Uvicorn process with a single
`threading.Lock`-protected session store and SQLite backing (see `app/api/routes.py`,
`SqlStore`) — expected given the project's documented single-worker design
(`docs/PROJECT_CONTEXT.md`), not a defect. An optional `evaluation/locustfile.py` is provided for
sustained/ramping load testing beyond this one-shot burst test, if a future evaluation wants to
find the actual saturation point.

## 9. Failed test cases and limitations

**Retrieval:**
- `booking-02` missed its target chunk in the top 3 (recovered at top 5) — see §4. Root cause:
  the question's wording ("show at the gate to enter") shares vocabulary with other,
  non-target chunks (parking rules, parking levels) more strongly than with the target chunk's
  own phrasing ("check in and enter").
- An out-of-domain query sharing incidental vocabulary with the knowledge base can score *above*
  `RAG_MIN_SCORE`: "Can you recommend a good restaurant in the city center?" retrieved three
  chunks at 0.42/0.38/0.34 (location/general-info chunks), all above the 0.3 floor — verified
  directly against the live vector store during this evaluation. In this environment (no LLM
  configured) the chatbot still answered safely with the "LLM unavailable" fallback rather than
  attempting a generation from irrelevant context, but with a real LLM configured there is a
  real (if narrow) risk of an ungrounded-sounding answer for this class of question. A stricter
  threshold or a secondary relevance check would close this gap; not implemented in this pass to
  avoid regressing the documented RANK1 4/5 smoke-test baseline (`scripts/smoke_test_data.py`).
- A second out-of-domain query ("What is the capital of France?") correctly scored below 0.13
  for every retrieved chunk — well under the floor — confirming the threshold works for clearly
  unrelated questions; the restaurant example above is the harder, vocabulary-overlapping case.

**Guardrails:**
- Presidio's real NER (not the fake analyzer used by most unit tests) flags an ordinary day name
  (e.g. "Sunday") as a `DATE_TIME` entity and redacts it — a false positive discovered while
  writing `tests/test_security.py`, documented there and worked around in that specific test by
  choosing a date-word-free example. It does not affect the deterministic project-specific
  recognizers (booking references, plates, credentials), only generic-NER-detected spans.

**Performance:**
- Latency increases sharply under concurrent load (807 ms median vs 51 ms sequential) due to the
  single-process, single-lock, SQLite-backed design. Acceptable at this project's scale;
  documented as a known limitation (README "Known limitations", `PROJECT_CONTEXT.md`), not fixed
  in this pass since doing so (multi-worker sessions, a server-grade DB) is a larger
  architectural change outside Stage 4's scope.
- LLM generation quality could not be evaluated end-to-end in this environment: no real
  `LLM_API_KEY`/Ollama instance was configured, so the LLM call path (not the retrieval path) is
  untested here beyond the unit tests that use a fake LLM client.

## 10. Improvement recommendations

1. **Tighten the relevance floor or add a secondary check** for out-of-domain-but-vocabulary-
   overlapping questions (e.g. require the top chunk to clear a higher bar, or require a minimum
   score *gap* between the top-1 and a generic "no answer" baseline) — directly motivated by the
   restaurant-query finding in §9.
2. **Add a handful of booking-process synonyms** to the `04_booking_process.md` chunk's wording
   (or chunk text) to close the `booking-02` recall gap, since the fix is almost certainly
   phrasing, not model capacity.
3. **Re-run this evaluation with a real LLM configured** to additionally score generation
   quality (faithfulness to retrieved context, refusal rate on ungrounded questions) — this
   report only covers retrieval and application-layer performance.
4. **Run `evaluation/locustfile.py` for a sustained ramp test** (not just one 30-request burst)
   to find the actual throughput ceiling and whether latency degrades gracefully or falls off a
   cliff, before deciding whether multi-worker session storage is worth the added complexity.
5. **Expand the retrieval dataset** past 24 questions (e.g. multiple phrasings per chunk) for a
   statistically sturdier Recall/Precision estimate — the current dataset is representative but
   small enough that a single miss moves the aggregate by ~4 percentage points.
