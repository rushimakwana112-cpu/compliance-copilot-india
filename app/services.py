"""The external services one request needs, bundled so tests can swap in fakes."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.limits import AnswerCache, RateLimiter
from app.llm import LLM, make_llm
from app.vectorstore import VectorStore, make_store


@dataclass
class Services:
    settings: Settings
    llm: LLM
    store: VectorStore | None  # None when Pinecone isn't configured or reachable
    store_error: str = ""
    cache: AnswerCache = field(default=None)  # type: ignore[assignment]  # built from settings below
    limiter: RateLimiter = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        s = self.settings
        if self.cache is None:  # not `or`: an empty cache is falsy
            self.cache = AnswerCache(s.cache_size)
        if self.limiter is None:
            self.limiter = RateLimiter(s.rate_limit_per_min, s.rate_limit_per_day)


def build_services(settings: Settings | None = None) -> Services:
    """Create the configured LLM (Groq or Ollama) and Pinecone store from settings."""
    s = settings or get_settings()
    store, error = make_store(s)
    return Services(s, make_llm(s), store, error)
