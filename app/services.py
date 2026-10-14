"""The external services one request needs, bundled so tests can swap in fakes."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings
from app.llm import LLM, make_llm
from app.vectorstore import PineconeStore, VectorStore


@dataclass
class Services:
    settings: Settings
    llm: LLM
    store: VectorStore | None  # None when Pinecone isn't configured or reachable
    store_error: str = ""


def build_services(settings: Settings | None = None) -> Services:
    """Create real Ollama + Pinecone clients from settings."""
    s = settings or get_settings()
    llm = make_llm(s)
    if not s.pinecone_api_key:
        return Services(s, llm, None, "PINECONE_API_KEY is not set in .env")
    try:
        store: VectorStore | None = PineconeStore(s.pinecone_api_key, s.pinecone_index)
        return Services(s, llm, store)
    except Exception as e:  # network, auth or quota errors: keep the app up, report in /api/health
        return Services(s, llm, None, f"Pinecone unavailable: {e.__class__.__name__}: {str(e)[:160]}")
