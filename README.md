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

## Verify

```powershell
.\scripts\check.ps1                                   # ruff -> pytest -> seed -> ingest
.\.venv\Scripts\python -m scripts.smoke_test_data     # retrieval quality, prints RANK1 n/5
.\.venv\Scripts\python -m scripts.verify_sql_store    # manual SqlStore walkthrough
```

Project decisions and progress are tracked in [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md).
