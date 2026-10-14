"""FastAPI app: serves the frontend at / and the JSON API under /api."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app import catalog
from app.config import FRONTEND_DIR
from app.jurisdiction import load_states
from app.schemas import DocumentOut, Health, StateOut
from app.services import Services, build_services


def create_app(services: Services | None = None) -> FastAPI:
    """Build the app. Pass `services` to inject fakes (tests); otherwise real clients are created at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or build_services()
        yield

    app = FastAPI(title="Compliance Copilot", version="0.1.0", lifespan=lifespan)

    def svc() -> Services:
        return app.state.services

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/api/health", response_model=Health)
    def health() -> Health:
        s = svc()
        llm = s.llm.health()
        issues = list(llm.get("issues", []))
        if s.store is None:
            issues.append(s.store_error or "Pinecone is not configured")
        return Health(
            status="ok" if not issues else "degraded",
            model=s.settings.llm_model,
            pinecone=s.store is not None,
            docs=catalog.indexed_count(),
            ollama=bool(llm.get("ok")),
            issues=issues,
        )

    @app.get("/api/states", response_model=list[StateOut])
    def states() -> list[StateOut]:
        return [StateOut(id=s.id, name=s.name, aliases=list(s.aliases)) for s in load_states()]

    @app.get("/api/documents", response_model=list[DocumentOut])
    def documents() -> list[DocumentOut]:
        return [DocumentOut(**d) for d in catalog.documents()]

    return app


app = create_app()
