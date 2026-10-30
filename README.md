# CityPark Central Chatbot

This project is a parking reservation assistant for a fictional parking facility, CityPark Central. It combines retrieval-augmented generation over static knowledge with dynamic availability checks from a SQL database and a guided reservation workflow.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env            # then edit if needed

.\.venv\Scripts\python -m app.data.seed --reset     # SQLite: spaces, prices, hours, samples
.\.venv\Scripts\python -m app.data.ingest --reset   # ChromaDB: 38 chunks from app/data/static
```

### PII model setup

The Python dependencies include Presidio and spaCy, but spaCy's English model is a separate
download. Install it once before using the guarded LLM/RAG path:

```powershell
.\.venv\Scripts\python -m spacy download en_core_web_sm
```

Tests inject fake analyzers and do not download or load this model. If the model or guardrail
processing is unavailable at runtime, RAG fails closed and does not send the request to the LLM.

## Verify

```powershell
.\scripts\check.ps1                                   # ruff -> pytest -> seed -> ingest
.\.venv\Scripts\python -m scripts.smoke_test_data     # retrieval quality, prints RANK1 n/5
.\.venv\Scripts\python -m scripts.verify_sql_store    # manual SqlStore walkthrough
```

## Chat from the terminal

```powershell
.\.venv\Scripts\python -m scripts.chat_cli            # type 'exit' to quit
```

The chatbot answers policy questions from the knowledge base (RAG, with sources), live
availability / prices / opening hours from SQLite, and walks you through booking, looking up
or cancelling a reservation (booking reference + plate are always required together).

## HTTP API and Streamlit UI

Start the API in one PowerShell window:

```powershell
.\.venv\Scripts\python -m uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` for the interactive API documentation. The implemented
routes are:

- `GET /health` — process health.
- `POST /chat` — message processing and conversation continuity; the response returns a
  `session_id` that the client sends on subsequent messages.
- `GET /availability` — current SQL-backed availability, optionally filtered with
  `?space_type=standard|compact|ev|disabled`.

In another PowerShell window, launch the UI:

```powershell
.\.venv\Scripts\python -m streamlit run app/ui/streamlit_app.py
```

Set `API_BASE_URL` in `.env` if the FastAPI server is not at `http://127.0.0.1:8000`.
Conversation sessions are held in process memory and are lost on restart; use one API worker
for consistent session handling. The random session ID is a bearer-like identifier and should
be kept private. The API does not provide user authentication or durable session storage.

### LLM configuration

| Variable | OpenAI-compatible API | Ollama (local) |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `ollama` |
| `LLM_MODEL` | e.g. `gpt-4o-mini` | e.g. `llama3.2` |
| `LLM_API_KEY` | required | ignored |
| `LLM_BASE_URL` | empty for api.openai.com | defaults to `http://localhost:11434/v1` |

`RAG_MIN_SCORE` (default `0.3`) is the cosine-similarity floor below which retrieved chunks
are ignored; when nothing clears it the bot says it doesn't have that information instead of
calling the model. Live availability, prices, hours and reservations never go through the LLM.

Project decisions and progress are tracked in [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md).
