"""FastAPI app: serves the frontend at / and the JSON API under /api."""

from __future__ import annotations

import json
import logging
import re
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app import catalog, logs, pipeline, tools
from app.config import DATA_DIR, FRONTEND_DIR
from app.jurisdiction import is_known_state, load_states
from app.limits import RateLimiter, limit_message
from app.schemas import (
    MAX_MESSAGE_CHARS, ChatRequest, ChatResponse, DocumentOut, Feedback, Health, IngestAccepted, IngestUrl, StateOut,
)
from app.services import Services, build_services

UPLOAD_DIR = DATA_DIR / "raw" / "uploads"
FEEDBACK_ANSWER_CHARS = 4000  # what a feedback log line keeps of the answer
log = logging.getLogger("app")
PUBLIC_INGEST_OFF = ("Adding documents is turned off on the public demo. "
                     "Run Compliance Copilot on your own computer to index your own PDFs (see the README).")


def client_ip(request: Request, behind_proxy: bool) -> str:
    """The caller's IP. Behind Render's proxy the real one is the first X-Forwarded-For entry.

    A client can put anything in that header, so this limit stops casual overuse, not a determined abuser.
    """
    if behind_proxy:
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client else "unknown"


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

    @app.middleware("http")
    async def public_guard(request: Request, call_next):
        """Public mode: refuse ingestion before the upload is even read."""
        if request.url.path.startswith("/api/ingest/") and svc().settings.public_mode:
            return JSONResponse({"detail": PUBLIC_INGEST_OFF}, status_code=403)
        return await call_next(request)

    def rate_limited(request: Request) -> JSONResponse | None:
        """In public mode: None if this IP may ask now, else a 429 whose body is a normal answer."""
        s = svc().settings
        if not s.public_mode:
            return None
        verdict = svc().limiter.hit(client_ip(request, behind_proxy=True))
        if verdict.allowed:
            return None
        body = ChatResponse(type="not_found", answer=limit_message(verdict, s.rate_limit_per_min, s.rate_limit_per_day),
                            note="Rate limit for the public demo.")
        return JSONResponse(body.model_dump(exclude_none=True), status_code=429,
                            headers={"Retry-After": str(verdict.retry_after)})

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        # no-cache = "check with the server first" (a cheap 304 via the ETag), so a deploy reaches returning visitors
        return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, bool]:
        """Cheap liveness check for Render: no Groq or Pinecone calls."""
        return {"ok": True}

    @app.get("/api/health", response_model=Health)
    def health() -> Health:
        s = svc()
        llm = s.llm.health()
        issues = list(llm.get("issues", []))
        if s.store is None:
            issues.append(s.store_error or "Pinecone is not configured")
        cfg = s.settings
        return Health(
            status="ok" if not issues else "degraded",
            model=cfg.active_model or "(not set)",
            pinecone=s.store is not None,
            docs=catalog.indexed_count(),
            provider=cfg.llm_provider,
            embed_provider=cfg.embed_provider,
            llm_ok=bool(llm.get("ok")),
            public_mode=cfg.public_mode,
            issues=issues,
        )

    @app.get("/api/states", response_model=list[StateOut])
    def states() -> list[StateOut]:
        return [StateOut(id=s.id, name=s.name, aliases=list(s.aliases)) for s in load_states()]

    @app.get("/api/rates")
    def rates(on: date | None = None) -> dict:
        """Every statutory figure from data/rates.yaml as it applies on `on` (default today), with its source.
        Figures not configured yet are listed with value null."""
        return tools.resolved(tools.load_rates(), on)

    @app.get("/api/documents", response_model=list[DocumentOut])
    def documents() -> list[DocumentOut]:
        return [DocumentOut(**d) for d in catalog.documents()]

    @app.post("/api/chat", response_model=ChatResponse, response_model_exclude_none=True)
    def chat(req: ChatRequest, request: Request) -> ChatResponse | JSONResponse:
        if (refused := rate_limited(request)) is not None:
            return refused
        resp, _ = pipeline.handle(svc(), req)
        return resp

    @app.post("/api/chat/stream", response_model=None)
    def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse | JSONResponse:
        """Server-Sent Events: `meta`, then `token` events (RAG only), then `done` with the full answer."""
        if (refused := rate_limited(request)) is not None:
            return refused

        def events():
            for name, data in pipeline.handle_stream(svc(), req):
                yield f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    feedback_limiter: list[RateLimiter] = []  # its own budget, so rating answers doesn't use up chats

    @app.post("/api/feedback", response_model=None)
    def feedback(fb: Feedback, request: Request) -> dict[str, bool] | JSONResponse:
        """Log a thumbs up/down. In public mode it is rate limited and every logged field is capped,
        because each call appends to a file on the server's disk."""
        s = svc().settings
        if s.public_mode:
            if not feedback_limiter:
                feedback_limiter.append(RateLimiter(s.rate_limit_per_min, s.rate_limit_per_day))
            verdict = feedback_limiter[0].hit(client_ip(request, behind_proxy=True))
            if not verdict.allowed:
                return JSONResponse({"ok": False}, status_code=429, headers={"Retry-After": str(verdict.retry_after)})
        a = fb.answer
        answer_text = str(a.get("answer", "") if isinstance(a, dict) else a or "")[:FEEDBACK_ANSWER_CHARS]
        question = str(fb.question or (a.get("question", "") if isinstance(a, dict) else ""))[:MAX_MESSAGE_CHARS]
        logs.append("feedback", {"question": question, "answer": answer_text, "value": fb.value,
                                 "type": str(a.get("type"))[:20] if isinstance(a, dict) and a.get("type") else None})
        return {"ok": True}

    def _index_in_background(path: Path, meta: dict[str, str]) -> None:
        from ingest.run import ingest_file  # imported here so the API starts fast

        try:
            out = ingest_file(path, meta, svc().store, store_error=svc().store_error)
            failed = out.status.startswith("failed")
            catalog.set_job(meta["doc"], meta["scope"], out.status if failed else None)
            if not failed:
                svc().cache.clear()  # new passages can change earlier answers
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
            dl = download(row, max_bytes=svc().settings.max_upload_mb * 1024 * 1024)
            if dl.path is None:
                catalog.set_job(name, scope, dl.status)
            elif dl.path.suffix != ".pdf" or not dl.path.read_bytes()[:4] == b"%PDF":
                catalog.set_job(name, scope, "failed: link is not a PDF")
            else:
                _index_in_background(dl.path, row)

        background.add_task(work)
        return IngestAccepted(name=name, scope=scope)

    return app


app = create_app()
