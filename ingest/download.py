"""Step 1 of ingestion: download each source once into data/raw/.

Files that already exist are skipped, so re-running is cheap. Each file's
sha256 goes into data/manifest.json; indexing compares it later to spot
documents that changed upstream.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from app import catalog
from app.config import DATA_DIR

RAW_DIR = DATA_DIR / "raw"
TIMEOUT = 60
MAX_REDIRECTS = 5
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


def blocked_reason(url: str, resolve: Callable[[str], list[str]] | None = None) -> str | None:
    """Why this link may not be fetched, or None. Only https to public internet addresses is allowed:
    a link to localhost, a private network (10.x, 192.168.x…) or the cloud metadata address would let
    anyone who can add a link make this server read things only it can reach (SSRF)."""
    parts = urlparse(url)
    if parts.scheme != "https":
        return "only https links are allowed"
    if not parts.hostname:
        return "the link has no host"
    try:
        ips = (resolve or _resolve)(parts.hostname)
    except OSError:
        return f"{parts.hostname} was not found"
    for raw in ips:
        ip = ipaddress.ip_address(raw.split("%")[0])
        ip = getattr(ip, "ipv4_mapped", None) or ip
        if not ip.is_global:
            return f"{parts.hostname} is a private or local address"
    return None if ips else f"{parts.hostname} was not found"


def _resolve(host: str) -> list[str]:
    return [info[4][0] for info in socket.getaddrinfo(host, None)]


def _fetch(url: str, client: httpx.Client, max_bytes: int | None,
           resolve: Callable[[str], list[str]] | None) -> tuple[bytes, str]:
    """GET with every redirect hop checked by blocked_reason() and the body capped at max_bytes.
    Raises ValueError with a short reason when a hop is blocked or the file is too big."""
    for _ in range(MAX_REDIRECTS + 1):
        if reason := blocked_reason(url, resolve):
            raise ValueError(reason)
        with client.stream("GET", url) as r:
            if r.is_redirect and "location" in r.headers:
                url = urljoin(url, r.headers["location"])
                continue
            r.raise_for_status()
            body = bytearray()
            for block in r.iter_bytes():
                body += block
                if max_bytes is not None and len(body) > max_bytes:
                    raise ValueError(f"larger than {max_bytes // (1024 * 1024)} MB")
            return bytes(body), r.headers.get("content-type", "")
    raise ValueError("too many redirects")


def download(row: dict[str, str], raw_dir: Path = RAW_DIR, client: httpx.Client | None = None,
             max_bytes: int | None = None, resolve: Callable[[str], list[str]] | None = None) -> Downloaded:
    """Download one sources.csv row unless its file is already in data/raw/.
    Redirects are followed by hand so that every hop is checked (https, public address)."""
    doc, url = row["doc"], row["url"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    cached = existing_file(doc, raw_dir)
    if cached:
        catalog.update_manifest(doc, {"file": cached.name, "sha256": sha256_file(cached), "url": url})
        return Downloaded(doc, cached, "cached")
    try:
        c = client or httpx.Client(headers=HEADERS, timeout=TIMEOUT, follow_redirects=False)
        content, content_type = _fetch(url, c, max_bytes, resolve)
    except ValueError as e:
        log.warning("Download refused for %s (%s): %s", doc, url, e)
        return Downloaded(doc, None, f"failed: {e}")
    except httpx.HTTPError as e:
        log.warning("Download failed for %s (%s): %s", doc, url, e)
        return Downloaded(doc, None, f"failed: {e.__class__.__name__}")
    path = raw_dir / f"{slugify(doc)}{_extension(url, content_type)}"
    path.write_bytes(content)
    catalog.update_manifest(doc, {"file": path.name, "sha256": sha256_file(path), "url": url})
    return Downloaded(doc, path, "downloaded")
