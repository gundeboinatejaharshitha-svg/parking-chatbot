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

### Chatbot core (Stage 2)
- **Entry point:** `app.chatbot.orchestrator.Orchestrator(sql_store, rag, now).handle(message,
  session) -> BotReply(text, intent, sources, flow, flow_done)`; `ConversationSession` holds the
  active flow. `build_orchestrator()` wires real SQL + Chroma + LLM from settings.
- **Intent routing is rule-based** (`app/chatbot/intent.py`, `classify(text) -> Intent`):
  GREETING, AVAILABILITY, PRICES, HOURS, BOOK, CANCEL, LOOKUP, STATIC_QUESTION (default).
  Policy phrasing ("how late can I cancel", "is there a fee") is sent to RAG, not to a flow; a
  `CP-XXXXXX` reference in the message forces the action interpretation.
- **Dynamic data never touches the LLM:** availability/prices/hours are templated straight
  from `SqlStore`. The LLM is used only to phrase answers from static context.
- **RAG** (`app/chatbot/rag.py`): `RagAnswerer(vector_store, llm, min_score, top_k)`. Chunks
  below `RAG_MIN_SCORE` (default 0.3, cosine) are discarded; if none remain the fixed
  `UNAVAILABLE_MESSAGE` is returned **without calling the LLM**. The system prompt tells the model
  to answer only from the numbered passages and to ignore instructions inside them. Sources
  (`SourceRef(id, source, section, score)`) are attached only to grounded answers.
- **LLM** (`app/chatbot/llm.py`): `LLMClient` protocol with `complete(system, user) -> str`;
  one `OpenAICompatibleClient` (openai SDK) covers OpenAI and Ollama (`/v1`). `build_llm_client`
  fails fast with `LLMConfigurationError` when the key is missing or still the placeholder;
  SDK errors become `LLMError`, which the orchestrator turns into a friendly fallback.
- **Reservation flows** (`app/chatbot/reservation.py`), all validated in code:
  `BookingFlow` DATE -> START -> END -> SPACE_TYPE -> NAME -> PLATE -> CONFIRM, pre-filling
  from the opening message; `create_reservation` is called only after an explicit "yes" and the
  reply says "confirmed" only when SQL returned a `ReservationInfo`. `CancelFlow`/`LookupFlow`
  always require reference + plate and return the same not-found reply for wrong plate or
  unknown reference. "cancel"/"stop"/"never mind" abort any flow.
- **Parsers** (`app/chatbot/parsing.py`) are deterministic regex helpers for dates, times, time
  ranges, plates (2-8 alphanumerics after normalization), references, names, yes/no and weekdays.
- New settings: `RAG_MIN_SCORE`, `LLM_TEMPERATURE`, `LLM_TIMEOUT_SECONDS`.
- Manual check: `python -m scripts.chat_cli` (needs seed + ingest + a working LLM).

### Known limitations after Stage 2
- The live LLM path could not be exercised in the development environment: `.env` still has the
  `your-api-key` placeholder and Ollama is not installed. The adapter is covered by stubbed unit
  tests; every other path (routing, SQL answers, booking, cancel, lookup, fallback) was verified
  live. Set a real `LLM_API_KEY` or run Ollama to enable RAG answers.
- Intent rules are English-only keyword heuristics; unknown phrasing falls back to RAG, which
  answers "I don't have that information" when retrieval is weak.
- Sessions were initially in-memory objects; client-keyed API sessions are implemented in Stage 3.

### Guardrails, API and UI (Stage 3)
- **PII boundary:** `app/guardrails/pii.py` combines Presidio AnalyzerEngine + AnonymizerEngine
  with deterministic recognition for CityPark booking references, labeled/common vehicle plates,
  labeled names, emails, and credential-like values. The analyzer is initialized lazily; install
  `en_core_web_sm` separately with `python -m spacy download en_core_web_sm`.
- **LLM boundary:** `app/chatbot/rag.py` redacts the question and retrieved passage text before
  calling the model, blocks direct instruction-override attempts in both, then filters generated
  output. If PII processing fails, it returns the existing unavailable response without leaking
  unfiltered model text. Project-public contact/location facts and the policy phrase “24 hours”
  are allowlisted so useful static answers are preserved.
- **Routing safety:** `app/chatbot/orchestrator.py` checks for prompt injection before intent
  routing. Dynamic SQL answers and the code-validated booking, lookup and cancellation flows
  remain outside the LLM path.
- **HTTP API:** `app/main.py` exposes `GET /health`, `POST /chat`, and `GET /availability`.
  API validation errors and unexpected failures use generic response bodies; availability stays
  SQL-only. Reservation CRUD endpoints are not exposed.
- **Sessions:** the API maps random UUID session identifiers to `ConversationSession` objects
  in process memory and serializes each session's requests. IDs are bearer-like and must be kept
  private. State is not durable, shared between workers, or authenticated; run one API worker.
- **Streamlit:** `app/ui/streamlit_app.py` calls the FastAPI endpoint through
  `app/ui/api_client.py`, stores per-browser chat history/session ID, and displays citations.
- **Guardrail limitations:** Presidio/NER can miss or over-detect PII; deterministic rules cover
  known project identifiers but cannot identify every possible registration or sensitive phrase.
  The injection rules are heuristic rather than a complete prompt-injection defense. Install the
  spaCy model before RAG use; model/guardrail failures intentionally fail closed. No auth, rate
  limiting, session expiry/persistence, or multi-worker session store is included.

## How to resume in a new session
1. Read this file and `.github/copilot-instructions.md`.
2. Run `.\scripts\check.ps1` (ruff -> pytest -> seed --reset -> ingest --reset).
3. Optionally run `.\.venv\Scripts\python -m scripts.smoke_test_data`,
   `.\.venv\Scripts\python -m scripts.verify_sql_store` and `.\.venv\Scripts\python -m scripts.chat_cli`.
4. Continue with the first unticked stage below.

## Progress
- [x] Stage 1: Foundation + Data layer (SQL, static docs, vector store, ingestion)
- [x] Stage 2: Chatbot core (LLM abstraction, RAG, intent routing, reservation flow)
- [x] Stage 3: Guardrails + FastAPI + Streamlit UI
- [ ] Stage 4: Tests + Evaluation (Recall@K, Precision@K, MRR, latency, report)
- [ ] Stage 5: Docker, CI, README, slides, final review
