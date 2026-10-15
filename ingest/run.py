"""Run the whole ingestion pipeline: download → parse → chunk → embed + upsert.

PowerShell examples:
    python -m ingest.run                      # every row in data/sources.csv
    python -m ingest.run --limit 2            # first two documents only
    python -m ingest.run --only "Wages"       # rows whose doc name contains "Wages"
    python -m ingest.run --dry-run            # show what would change, touch nothing in Pinecone
    python -m ingest.run --force              # re-index even if the file hash is unchanged
    python -m ingest.run --reindex            # delete each doc's old vectors and index it again with the
                                              # current chunking (also covers PDFs uploaded in the UI)
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from app import catalog
from app.logs import now_iso
from app.config import DATA_DIR, get_settings
from app.vectorstore import VectorStore, make_store
from ingest.chunk import chunk_pages
from ingest.download import download, sha256_file, slugify
from ingest.index import index_document
from ingest.parse import parse

log = logging.getLogger("ingest")
UPLOAD_DIR = DATA_DIR / "raw" / "uploads"  # where the API saves PDFs uploaded from the sidebar


@dataclass
class Outcome:
    doc: str
    status: str
    pages: int = 0
    chunks: int = 0
    deleted: int = 0


def ingest_file(path: Path, meta: dict[str, str], store: VectorStore | None, *, store_error: str = "",
                force: bool = False, dry_run: bool = False, reindex: bool = False) -> Outcome:
    """Parse, chunk and index one local file described by `meta` (a sources.csv-style row).

    Needs only the Pinecone store: no LLM, and no Ollama when EMBED_PROVIDER=pinecone.
    `reindex=True` deletes every old vector of the document first, even if the file is unchanged,
    so a new chunking scheme never leaves old chunks behind.
    """
    force = force or reindex
    doc = meta["doc"]
    if store is None:
        return Outcome(doc, f"failed: {store_error or 'Pinecone not configured'}")
    sha = sha256_file(path)
    entry = catalog.read_manifest().get(doc, {})
    if entry.get("indexed_sha256") == sha and not force:
        return Outcome(doc, "unchanged", entry.get("pages", 0), entry.get("chunks", 0))
    pages = parse(path)
    if not pages:
        return Outcome(doc, "failed: no text extracted (scanned PDF without OCR?)")
    chunks = chunk_pages(pages, meta.get("doc_type", "act"), doc=doc)
    # Delete old vectors only if this doc was indexed before from a different file version.
    replace = reindex or (bool(entry.get("indexed_sha256")) and entry["indexed_sha256"] != sha)
    with tqdm(total=len(chunks), desc=f"  index {doc[:40]}", unit="chunk", leave=False) as bar:
        result = index_document(
            doc, chunks, {**meta, "source_url": meta.get("url", "")}, store,
            replace=replace, dry_run=dry_run, progress=bar.update,
        )
    if not dry_run:
        catalog.update_manifest(doc, {
            "url": meta.get("url", ""), "scope": meta.get("scope", "central"), "status": meta.get("status", "current"),
            "effective_date": meta.get("effective_date", ""), "doc_type": meta.get("doc_type", "act"),
            "sha256": sha, "indexed_sha256": sha, "pages": len(pages),
            "ocr_pages": sum(p.ocr for p in pages), "chunks": len(chunks), "indexed_at": now_iso(),
        })
    return Outcome(doc, "dry-run" if dry_run else ("re-indexed" if replace else "indexed"), len(pages), len(chunks), result.deleted)


def uploaded_rows(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    """Documents uploaded in the UI: in the manifest, not in sources.csv, with their PDF in data/raw/uploads."""
    planned = {r["doc"] for r in sources}
    rows = []
    for doc, e in catalog.read_manifest().items():
        path = UPLOAD_DIR / f"{slugify(doc)}.pdf"
        if doc not in planned and e.get("chunks") and path.exists():
            rows.append({"doc": doc, "url": e.get("url", ""), "scope": e.get("scope", "central"),
                         "status": e.get("status", "current"), "effective_date": e.get("effective_date", ""),
                         "doc_type": e.get("doc_type", "circular"), "_path": str(path)})
    return rows


def run(limit: int | None = None, only: str | None = None, force: bool = False, dry_run: bool = False,
        reindex: bool = False) -> list[Outcome]:
    """Ingest every (filtered) row of data/sources.csv and return one outcome per row."""
    rows = catalog.read_sources()
    if reindex:
        rows += uploaded_rows(rows)
    if only:
        rows = [r for r in rows if only.lower() in r["doc"].lower()]
    rows = rows[:limit] if limit else rows
    store, store_error = make_store(get_settings())
    if store is None:
        log.error("Pinecone store unavailable: %s", store_error)
    outcomes: list[Outcome] = []
    for row in tqdm(rows, desc="Documents", unit="doc"):
        if "_path" in row:  # an uploaded PDF: nothing to download
            path: Path | None = Path(row.pop("_path"))
        else:
            dl = download(row)
            path = dl.path
            if path is None:
                outcomes.append(Outcome(row["doc"], dl.status))
                continue
        try:
            outcomes.append(ingest_file(path, row, store, store_error=store_error, force=force, dry_run=dry_run,
                                        reindex=reindex))
        except Exception as e:  # one bad document must not stop the batch
            log.exception("Failed on %s", row["doc"])
            outcomes.append(Outcome(row["doc"], f"failed: {e.__class__.__name__}: {str(e)[:80]}"))
    return outcomes


def print_summary(outcomes: list[Outcome]) -> None:
    table = Table(title="Ingestion summary")
    for col in ("Document", "Status", "Pages", "Chunks", "Old vectors deleted"):
        table.add_column(col, justify="right" if col not in ("Document", "Status") else "left")
    for o in outcomes:
        style = "red" if o.status.startswith("failed") else None
        table.add_row(o.doc, o.status, str(o.pages), str(o.chunks), str(o.deleted), style=style)
    Console().print(table)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, help="only the first N rows")
    p.add_argument("--only", help="only rows whose doc name contains this text")
    p.add_argument("--force", action="store_true", help="re-index unchanged files")
    p.add_argument("--reindex", action="store_true",
                   help="delete each document's old vectors and index it again with the current chunking")
    p.add_argument("--dry-run", action="store_true", help="report changes without writing to Pinecone")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print_summary(run(a.limit, a.only, a.force, a.dry_run, a.reindex))


if __name__ == "__main__":
    main()
