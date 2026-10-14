"""FastAPI app: serves the frontend at / and the JSON API under /api."""

from __future__ import annotations

import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app import catalog, logs, pipeline
from app.config import DATA_DIR, FRONTEND_DIR
from app.jurisdiction import is_known_state, load_states
from app.schemas import (
    ChatRequest, ChatResponse, DocumentOut, Feedback, Health, IngestAccepted, IngestUrl, StateOut,
)
from app.services import Services, build_services

UPLOAD_DIR = DATA_DIR / "raw" / "uploads"
log = logging.getLogger("app")


def _doc_name(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"[_-]+", " ", stem).strip()[:120] or "Uploaded document"


def _valid_scope(scope: str) -> str:
    if scope != "central" and not is_known_state(scope):
        raise HTTPException(422, f"Unknown scope '{scope}'. Use 'central' or a state id from /api/states.")
    return scope


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

    @app.post("/api/chat", response_model=ChatResponse, response_model_exclude_none=True)
    def chat(req: ChatRequest) -> ChatResponse:
        resp, _ = pipeline.handle(svc(), req)
        return resp

    @app.post("/api/chat/stream")
    def chat_stream(req: ChatRequest) -> StreamingResponse:
        """Server-Sent Events: `meta`, then `token` events (RAG only), then `done` with the full answer."""

        def events():
            for name, data in pipeline.handle_stream(svc(), req):
                yield f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/feedback")
    def feedback(fb: Feedback) -> dict[str, bool]:
        a = fb.answer
        answer_text = a.get("answer", "") if isinstance(a, dict) else str(a or "")
        question = fb.question or (a.get("question", "") if isinstance(a, dict) else "")
        logs.append("feedback", {"question": question, "answer": answer_text, "value": fb.value,
                                 "type": a.get("type") if isinstance(a, dict) else None})
        return {"ok": True}

    def _index_in_background(path: Path, meta: dict[str, str]) -> None:
        from ingest.run import ingest_file  # imported here so the API starts fast

        try:
            out = ingest_file(path, meta, svc())
            failed = out.status.startswith("failed")
            catalog.set_job(meta["doc"], meta["scope"], out.status if failed else None)
        except Exception as e:  # report in the sidebar instead of crashing the worker
            log.exception("Indexing failed for %s", meta["doc"])
            catalog.set_job(meta["doc"], meta["scope"], f"failed: {e.__class__.__name__}")

    @app.post("/api/ingest/file", response_model=IngestAccepted)
    async def ingest_file_upload(background: BackgroundTasks, file: UploadFile = File(...),
                                 scope: str = Form("central")) -> IngestAccepted:
        scope = _valid_scope(scope)
        limit = svc().settings.max_upload_mb * 1024 * 1024
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"File is larger than {svc().settings.max_upload_mb} MB")
        if not data.startswith(b"%PDF"):
            raise HTTPException(415, "Only PDF files are accepted")
        name = _doc_name(file.filename or "upload.pdf")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        from ingest.download import slugify

        path = UPLOAD_DIR / f"{slugify(name)}.pdf"
        path.write_bytes(data)
        meta = {"doc": name, "scope": scope, "status": "current", "effective_date": "", "doc_type": "circular", "url": ""}
        catalog.set_job(name, scope, "indexing…")
        background.add_task(_index_in_background, path, meta)
        return IngestAccepted(name=name, scope=scope)

    @app.post("/api/ingest/url", response_model=IngestAccepted)
    def ingest_url(body: IngestUrl, background: BackgroundTasks) -> IngestAccepted:
        url = str(body.url)
        if body.url.scheme != "https":
            raise HTTPException(422, "Only https links are accepted")
        scope = _valid_scope(body.scope)
        name = _doc_name(urlparse(url).path.rsplit("/", 1)[-1] or urlparse(url).hostname or "Linked document")
        catalog.set_job(name, scope, "indexing…")

        def work() -> None:
            from ingest.download import download

            row = {"doc": name, "url": url, "scope": scope, "status": "current", "effective_date": "", "doc_type": "circular"}
            dl = download(row)
            if dl.path is None:
                catalog.set_job(name, scope, dl.status)
            elif dl.path.suffix != ".pdf" or not dl.path.read_bytes()[:4] == b"%PDF":
                catalog.set_job(name, scope, "failed: link is not a PDF")
            elif dl.path.stat().st_size > svc().settings.max_upload_mb * 1024 * 1024:
                catalog.set_job(name, scope, f"failed: larger than {svc().settings.max_upload_mb} MB")
            else:
                _index_in_background(dl.path, row)

        background.add_task(work)
        return IngestAccepted(name=name, scope=scope)

    return app


app = create_app()
