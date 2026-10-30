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

### SQL schema

### SqlStore API

### VectorStore API

### Other

## Progress
- [ ] Stage 1: Foundation + Data layer (SQL, static docs, vector store, ingestion)
- [ ] Stage 2: Chatbot core (LLM abstraction, RAG, intent routing, reservation flow)
- [ ] Stage 3: Guardrails + FastAPI + Streamlit UI
- [ ] Stage 4: Tests + Evaluation (Recall@K, Precision@K, MRR, latency, report)
- [ ] Stage 5: Docker, CI, README, slides, final review
