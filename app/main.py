"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes import install_routes
from app.chatbot.orchestrator import Orchestrator
from app.data.sql_store import SqlStore


def create_app(
    *,
    orchestrator: Orchestrator | None = None,
    sql_store: SqlStore | None = None,
) -> FastAPI:
    """Create the API application, optionally injecting services for tests."""
    app = FastAPI(title="CityPark Central Assistant", version="1.0.0")
    install_routes(app, orchestrator=orchestrator, sql_store=sql_store)
    return app


app = create_app()

__all__ = ["app", "create_app"]
