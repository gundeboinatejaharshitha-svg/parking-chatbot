# Project Context

## Goal
CityPark Central is a fictional parking facility assistant that answers static policy questions with RAG and dynamic availability/price questions from a SQL-backed data layer. It also guides users through a step-by-step reservation flow with validation and cancellation support.

## Stack
- Python 3.11
- FastAPI + Streamlit
- ChromaDB for persistent vector storage
- SQLite + SQLAlchemy 2.0 for dynamic data
- sentence-transformers/all-MiniLM-L6-v2 embeddings
- Swappable LLM provider via environment-based abstraction (OpenAI-compatible API or Ollama)
- Presidio + spaCy for guardrails and prompt-injection protection
- pytest, ruff, Docker, GitHub Actions

## Architecture rules
- Keep business logic out of API routes.
- Use environment-driven configuration from a single config module.
- Add type hints and short docstrings throughout.
- Log via the logging module instead of print.
- Keep modules testable by dependency injection.
- Validate reservations in code; never rely on the LLM for reservation logic.
- Prevent any direct LLM access to the reservations table.

## Repository structure
- app/ with API, config, chatbot layers, data access, guardrails, and UI
- evaluation/ for metrics and benchmarking
- tests/ for unit and integration tests
- docs/ for project notes
- .github/workflows/ for CI later

## Decisions made

### Chunk ID format
`{file_stem}::{section_number:02d}-{slug[:40]}::{chunk_index}`
e.g. `03_parking_rules::02-where-is-stopping-or-parking-prohibited::0`
- One chunk per `## ` section; `section_number` is 1-based within the file.
- `slug` comes from the title after stripping a trailing " at CityPark Central" /
  " CityPark Central" (punctuation preserved), lowercased, non-alphanumerics -> `-`,
  truncated to 40 chars, trailing hyphens removed.
- Sections over 400 words are split into 400-word windows with 50 words overlap;
  pieces share the same section number and get `chunk_index` 0, 1, 2...
- Embedded/stored text = cleaned title + `\n` + body. Metadata: `section` (full
  original heading), `source` (file name, e.g. `05_cancellation_policy.md`), `chunk_index`.
- The 10 static files yield exactly 38 chunks; `app.data.ingest` fails loudly otherwise.

### SQL schema
SQLite via SQLAlchemy 2.0 (`app/data/sql_store.py`, protected):
- `parking_spaces(id, space_type UNIQUE, total, available)` with CHECK `0 <= available <= total`
- `pricing(id, space_type UNIQUE, price_per_hour, price_per_day)` (GBP, Numeric(8,2))
- `working_hours(id, day_of_week UNIQUE 0=Mon..6=Sun, open_time, close_time, is_closed)`
- `reservations(id, booking_ref UNIQUE idx, customer_name, car_plate idx, date, start_time,
  end_time, space_type, status CHECK in ('confirmed','cancelled'), created_at)`
- Space types: `standard`, `compact`, `ev`, `disabled`. Booking refs: `CP-` + 6 digits.
- Seed (`python -m app.data.seed [--reset]`) is idempotent: 4 spaces, 4 prices, 7 days,
  5 sample reservations `CP-100001..CP-100005`.

### SqlStore API
`SqlStore(engine, now=None)` / `SqlStore.from_url(url, now=None)`; `create_db_engine(url)`
handles `sqlite://` in-memory for tests. Public methods:
- `init_db()`, `reset_db()`, `transaction()` (seed/admin only)
- `get_availability(space_type=None) -> list[SpaceAvailability]`
- `get_prices(space_type=None) -> list[PriceInfo]`
- `get_working_hours(day=None) -> list[WorkingHoursInfo]` (day: 0-6 or weekday name)
- `is_open(date, start, end) -> bool`
- `create_reservation(*, customer_name, car_plate, date, start_time, end_time, space_type)
  -> ReservationInfo` (atomic availability decrement + insert; validates name 2-100 chars,
  plate 2-8 alphanumerics after normalisation, end > start, start in the future, within hours)
- `get_reservation_for_user(booking_ref, car_plate) -> ReservationInfo`
- `cancel_reservation(booking_ref, car_plate) -> CancellationResult(booking_ref,
  refund_eligible, hours_before_start)`; refund if >= 24h before start
- Wrong plate and unknown reference raise the SAME `ReservationNotFoundError`.
- Logs only masked values: `mask_plate`, `mask_name`, `mask_ref`.

### VectorStore API
`app/data/vector_store.py`, Chroma `PersistentClient` at `settings.chroma_path`, collection
`citypark_static` with `{"hnsw:space": "cosine"}`, telemetry disabled.
- `VectorStore(*, path=None, collection_name="citypark_static", model_name=None, top_k=None,
  client=None, embed_fn=None)` - all collaborators injectable; the sentence-transformers
  model loads lazily on first embed, never at import.
- `add_chunks(chunks) -> int` upserts (idempotent re-ingestion); IDs must be unique per call.
- `query(text, k=None) -> list[RetrievedChunk(id, text, source, section, score)]`,
  default `k = settings.top_k`, clamped to `count()`; raises `EmptyCollectionError` on an
  empty collection and `ValueError` on blank text / `k < 1`.
- `count() -> int`, `reset()` (drop + recreate collection).
- `score = 1 - cosine_distance` = cosine similarity; higher is better.
- `app.data.ingest.run_ingestion(store, reset=False)` / CLI `python -m app.data.ingest [--reset]`.

### Other
- **Tooling** (`pyproject.toml`): ruff line-length 100, rules `E,F,I,UP,B`, target py311;
  pytest `testpaths=["tests"]`, `pythonpath=["."]`.
- **Protected files** (never edit; E501 ignored for them in ruff): `app/data/sql_store.py`,
  `app/data/seed.py`, `scripts/verify_sql_store.py`, `app/data/static/*.md`.
- **Known simplification:** availability is one live counter per space type, not per time
  slot. A reservation decrements the counter at creation and a cancellation restores it,
  regardless of the booked date/time.
- `posthog` is pinned to 5.4.0 because chromadb 0.6.3's telemetry client crashes on >= 6.
- Tests never download models or touch `parking.db` / `chroma_db`: they inject a fake
  embedder, a temporary Chroma path, and an in-memory SQLite engine with a fake clock.
- Retrieval smoke test (`python -m scripts.smoke_test_data`) passes at RANK1 4/5; the known
  miss is "I lost my booking reference", where `04_booking_process` (what is my booking
  reference) narrowly outranks `10_faq` (lost reference). Both are relevant.

## How to resume in a new session
1. Read this file and `.github/copilot-instructions.md`.
2. Run `.\scripts\check.ps1` (ruff -> pytest -> seed --reset -> ingest --reset).
3. Optionally run `.\.venv\Scripts\python -m scripts.smoke_test_data` and
   `.\.venv\Scripts\python -m scripts.verify_sql_store`.
4. Continue with the first unticked stage below.

## Progress
- [x] Stage 1: Foundation + Data layer (SQL, static docs, vector store, ingestion)
- [ ] Stage 2: Chatbot core (LLM abstraction, RAG, intent routing, reservation flow)
- [ ] Stage 3: Guardrails + FastAPI + Streamlit UI
- [ ] Stage 4: Tests + Evaluation (Recall@K, Precision@K, MRR, latency, report)
- [ ] Stage 5: Docker, CI, README, slides, final review
