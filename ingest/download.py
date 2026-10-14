"""Step 1 of ingestion: download each source once into data/raw/.

Files that already exist are skipped, so re-running is cheap. Each file's
sha256 goes into data/manifest.json; indexing compares it later to spot
documents that changed upstream.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from app import catalog
from app.config import DATA_DIR

RAW_DIR = DATA_DIR / "raw"
TIMEOUT = 60
# Some government sites refuse requests without a browser-like User-Agent.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
log = logging.getLogger("ingest")


@dataclass
class Downloaded:
    doc: str
    path: Path | None
    status: str  # "downloaded" | "cached" | "failed: ..."


def slugify(name: str) -> str:
    """'Code on Wages, 2019' -> 'code-on-wages-2019' (used for file names and vector-id prefixes)."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:80] or "doc"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _extension(url: str, content_type: str) -> str:
    if url.lower().split("?")[0].endswith(".pdf") or "pdf" in content_type:
        return ".pdf"
    return ".html"


def existing_file(doc: str, raw_dir: Path = RAW_DIR) -> Path | None:
    for ext in (".pdf", ".html"):
        p = raw_dir / f"{slugify(doc)}{ext}"
        if p.exists():
            return p
    return None


def download(row: dict[str, str], raw_dir: Path = RAW_DIR, client: httpx.Client | None = None) -> Downloaded:
    """Download one sources.csv row unless its file is already in data/raw/."""
    doc, url = row["doc"], row["url"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    cached = existing_file(doc, raw_dir)
    if cached:
        catalog.update_manifest(doc, {"file": cached.name, "sha256": sha256_file(cached), "url": url})
        return Downloaded(doc, cached, "cached")
    if not url.lower().startswith("https://"):
        return Downloaded(doc, None, "failed: only https links are allowed")
    try:
        c = client or httpx.Client(headers=HEADERS, timeout=TIMEOUT, follow_redirects=True)
        r = c.get(url)
        r.raise_for_status()
    except httpx.HTTPError as e:
        log.warning("Download failed for %s (%s): %s", doc, url, e)
        return Downloaded(doc, None, f"failed: {e.__class__.__name__}")
    path = raw_dir / f"{slugify(doc)}{_extension(url, r.headers.get('content-type', ''))}"
    path.write_bytes(r.content)
    catalog.update_manifest(doc, {"file": path.name, "sha256": sha256_file(path), "url": url})
    return Downloaded(doc, path, "downloaded")
