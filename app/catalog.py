"""The document catalogue: what's planned, indexed, or indexing right now.

* data/sources.csv     → documents you plan to index (filled in by hand)
* data/manifest.json   → documents downloaded / indexed, with sha256 and chunk count
* in-memory jobs       → uploads and links being indexed in the background

GET /api/documents merges these three into one list for the sidebar.

The deployed app (Render) never runs ingestion: Pinecone already holds the documents
and data/manifest.json may not exist there. With no manifest (or an empty one), the
documents in sources.csv are taken as the indexed set.
"""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Any

from app.config import DATA_DIR

SOURCES_CSV = DATA_DIR / "sources.csv"
MANIFEST = DATA_DIR / "manifest.json"
SOURCE_FIELDS = ["url", "doc", "scope", "status", "effective_date", "doc_type"]

_lock = threading.Lock()
_jobs: dict[str, dict[str, str]] = {}  # doc name -> {"scope", "state": "indexing…" | "failed: ..."}


def read_sources(path: Path | None = None) -> list[dict[str, str]]:
    """Rows of sources.csv, skipping blank lines and lines starting with '#'."""
    path = path or SOURCES_CSV
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(lines)]


def read_manifest(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Manifest keyed by document name."""
    path = path or MANIFEST
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(data: dict[str, dict[str, Any]], path: Path | None = None) -> None:
    path = path or MANIFEST
    with _lock:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


def update_manifest(doc: str, fields: dict[str, Any], path: Path | None = None) -> None:
    """Merge `fields` into the manifest entry for `doc`."""
    path = path or MANIFEST
    with _lock:
        data = read_manifest(path)
        data[doc] = {**data.get(doc, {}), **fields}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


def set_job(doc: str, scope: str, state: str | None) -> None:
    """Track a background ingest job; `state=None` removes it."""
    with _lock:
        if state is None:
            _jobs.pop(doc, None)
        else:
            _jobs[doc] = {"scope": scope, "state": state}


def indexed_count(path: Path | None = None, sources: Path | None = None) -> int:
    """Documents with passages in the index; without a manifest, the documents in sources.csv."""
    indexed = sum(1 for e in read_manifest(path).values() if e.get("chunks"))
    return indexed or len(read_sources(sources))


def documents(sources: Path | None = None, manifest: Path | None = None) -> list[dict[str, str]]:
    """Merged list for the sidebar: jobs first, then indexed docs, then planned ones."""
    man = read_manifest(manifest)
    no_manifest = not any(e.get("chunks") for e in man.values())  # deployed app: trust sources.csv
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    with _lock:
        jobs = dict(_jobs)
    for name, job in jobs.items():
        out.append({"name": name, "scope": job["scope"], "status": "current", "meta": job["state"]})
        seen.add(name)
    for name, e in man.items():
        if name in seen or not e.get("chunks"):
            continue
        meta = f"In force {e['effective_date']}" if e.get("effective_date") else f"{e['chunks']} passages"
        out.append({"name": name, "scope": e.get("scope", "central"), "status": e.get("status", "current"), "meta": meta})
        seen.add(name)
    for row in read_sources(sources):
        if row["doc"] in seen:
            continue
        if no_manifest:
            meta = f"In force {row['effective_date']}" if row.get("effective_date") else "Indexed"
        else:
            meta = "not indexed yet"
        out.append({"name": row["doc"], "scope": row["scope"], "status": row["status"], "meta": meta})
    return out
