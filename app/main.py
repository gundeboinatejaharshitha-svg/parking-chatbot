"""FastAPI application entry point."""

from fastapi import FastAPI

from app.agents.approval_agent import ApprovalAgent, build_approval_agent
from app.api.admin_routes import install_admin_routes
from app.api.routes import install_routes
from app.chatbot.orchestrator import Orchestrator
from app.config import get_settings
from app.data.sql_store import SqlStore


def create_app(
    *,
    orchestrator: Orchestrator | None = None,
    sql_store: SqlStore | None = None,
    approval_agent: ApprovalAgent | None = None,
) -> FastAPI:
    """Create the API application, optionally injecting services for tests."""
    app = FastAPI(title="CityPark Central Assistant", version="1.0.0")
    install_routes(app, orchestrator=orchestrator, sql_store=sql_store)

    settings = get_settings()
    if approval_agent is None and orchestrator is not None:
        approval_agent = getattr(orchestrator, "approval_agent", None)
    if approval_agent is None and orchestrator is None and settings.require_admin_approval:
        # Lazily built orchestrators wire their own agent on first /chat request; this
        # one serves /admin independently so it is not left unconfigured until then.
        approval_agent = build_approval_agent(settings, sql_store or SqlStore.from_url(
            settings.sqlite_url
        ))
    install_admin_routes(app, approval_agent=approval_agent, admin_api_key=settings.admin_api_key)
    return app


app = create_app()

__all__ = ["app", "create_app"]
