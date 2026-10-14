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

    pinecone_api_key: str = ""
    pinecone_index: str = "compliance-copilot"
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    embed_model: str = "bge-m3"
    score_threshold: float = 0.55
    router: Literal["rules", "llm"] = "rules"

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
