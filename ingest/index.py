"""Step 4 of ingestion: embed chunks with bge-m3 and upsert them to Pinecone.

Vector ids are deterministic: `<doc-slug>#sha1(doc + page + chunk_no)`. Running
ingestion twice therefore overwrites the same vectors instead of adding
duplicates. The `<doc-slug>#` prefix lets us list every vector of one document,
which is how a changed document's old vectors are removed before re-indexing.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.llm import LLM
from app.vectorstore import VectorStore
from ingest.chunk import Chunk
from ingest.download import slugify

log = logging.getLogger("ingest")
EMBED_BATCH = 32
META_FIELDS = ("scope", "status", "effective_date", "doc_type", "source_url")


@dataclass
class IndexResult:
    doc: str
    upserted: int
    deleted: int


def vector_id(doc: str, page: int, chunk_no: int) -> str:
    digest = hashlib.sha1(f"{doc}|{page}|{chunk_no}".encode()).hexdigest()
    return f"{slugify(doc)}#{digest}"


def build_records(doc: str, chunks: list[Chunk], vectors: list[list[float]], meta: dict[str, str]) -> list[dict[str, Any]]:
    """Pinecone records with the metadata the retriever and the citations need."""
    return [
        {
            "id": vector_id(doc, c.page, c.chunk_no),
            "values": v,
            "metadata": {
                "doc": doc,
                "section": c.section,
                "page": c.page,
                "text": c.text,
                **{k: meta.get(k, "") for k in META_FIELDS},
            },
        }
        for c, v in zip(chunks, vectors)
    ]


def stale_ids(store: VectorStore, doc: str) -> list[str]:
    """Every vector id currently stored for `doc`."""
    return store.list_ids(f"{slugify(doc)}#")


def index_document(
    doc: str,
    chunks: list[Chunk],
    meta: dict[str, str],
    llm: LLM,
    store: VectorStore,
    *,
    replace: bool = False,
    dry_run: bool = False,
    progress: Callable[[int], None] | None = None,
) -> IndexResult:
    """Embed and upsert one document.

    `replace=True` (the file's hash changed) first deletes that document's old
    vectors, because a shorter new version would otherwise leave orphan chunks.
    `dry_run=True` only reports what would be deleted and upserted.
    """
    old = stale_ids(store, doc) if replace else []
    if dry_run:
        log.info("[dry-run] %s: would delete %d old vectors and upsert %d", doc, len(old), len(chunks))
        return IndexResult(doc, 0, 0)
    if old:
        log.info("%s changed: deleting %d old vectors", doc, len(old))
        store.delete(old)
    records: list[dict[str, Any]] = []
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        # bge-m3 works best on the passage text itself; we prefix the section so headings are searchable.
        vectors = llm.embed([f"{c.section}\n{c.text}" if c.section else c.text for c in batch])
        records.extend(build_records(doc, batch, vectors, meta))
        if progress:
            progress(len(batch))
    store.upsert(records)
    return IndexResult(doc, len(records), len(old))
