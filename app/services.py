"""The external services one request needs, bundled so tests can swap in fakes."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings
from app.llm import LLM, make_llm
from app.vectorstore import VectorStore, make_store


@dataclass
class Services:
    settings: Settings
    llm: LLM
    store: VectorStore | None  # None when Pinecone isn't configured or reachable
    store_error: str = ""


def build_services(settings: Settings | None = None) -> Services:
    """Create the configured LLM (Groq or Ollama) and Pinecone store from settings."""
    s = settings or get_settings()
    store, error = make_store(s)
    return Services(s, make_llm(s), store, error)
