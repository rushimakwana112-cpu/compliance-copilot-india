"""Append-only JSON Lines logs (chats, feedback, ingest).

JSONL keeps one record per line, so files can be appended safely and read
back with a few lines of Python or pandas for the eval.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import LOGS_DIR

_lock = threading.Lock()


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append(name: str, record: dict[str, Any], logs_dir: Path = LOGS_DIR) -> None:
    """Append `record` (with a timestamp) to `logs/<name>.jsonl`."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"timestamp": now_iso(), **record}, ensure_ascii=False, default=str)
    with _lock, open(logs_dir / f"{name}.jsonl", "a", encoding="utf-8") as f:
        f.write(line + "\n")
