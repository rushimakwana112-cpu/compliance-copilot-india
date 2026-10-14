"""Settings loaded from environment variables and `.env`.

Everything configurable lives here so the rest of the code never reads
`os.environ` directly. Paths are resolved relative to the project root, so the
app works no matter which folder you start it from.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"
FRONTEND_DIR = ROOT / "frontend"


class Settings(BaseSettings):
    """Runtime configuration. Field names map to upper-case env vars."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # Which service writes answers: Groq in the cloud (free tier) or a local Ollama.
    llm_provider: Literal["groq", "ollama"] = "groq"
    groq_api_key: str = ""
    groq_model: str = ""  # pick one from `python -m app.list_models`; never hard-coded
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"  # the Ollama model
    llm_temperature: float = 0.1
    llm_max_tokens: int = 700

    # Who turns text into vectors: Pinecone's hosted model (no local GPU/Ollama) or Ollama bge-m3.
    embed_provider: Literal["pinecone", "ollama"] = "pinecone"
    embed_model: str = "bge-m3"  # Ollama embedding model (embed_provider=ollama)
    pinecone_embed_model: str = "llama-text-embed-v2"  # hosted model (embed_provider=pinecone)
    pinecone_api_key: str = ""
    pinecone_index: str = "compliance-copilot-cloud"
    pinecone_namespace: str = "compliance"

    score_threshold: float = 0.55
    router: Literal["rules", "llm"] = "rules"

    # Public demo: ingest endpoints off, per-IP limits on chat.
    public_mode: bool = False
    rate_limit_per_min: int = 10
    rate_limit_per_day: int = 60
    cache_size: int = 500

    @property
    def active_model(self) -> str:
        """The chat model in use for the selected provider."""
        return self.groq_model if self.llm_provider == "groq" else self.llm_model

    # Retrieval knobs (tuned with the eval, see README)
    top_k: int = 8
    keep_k: int = 4
    history_turns: int = 4

    # Ingestion limits for uploads
    max_upload_mb: int = 20

    db_path: Path = DATA_DIR / "sample_payroll.db"


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings object (tests clear the cache to override)."""
    return Settings()
