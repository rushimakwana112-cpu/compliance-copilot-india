"""Pinecone, in two flavours that look the same to the rest of the app.

    EMBED_PROVIDER=pinecone  IntegratedStore: Pinecone embeds the text itself with a hosted model
                             (llama-text-embed-v2). No GPU, no Ollama: just the Pinecone key.
    EMBED_PROVIDER=ollama    DenseStore: we embed with Ollama bge-m3 and send vectors.

Both take *text* in and give `Match`es out, so retrieval and ingestion don't
care where the embedding happens. The two index types are not compatible
(different models, different vector spaces), so each mode uses its own index.
Neither class ever deletes or reconfigures an index it didn't create.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

EMBED_DIM = 1024  # bge-m3 output size
EMBED_BATCH = 32
RECORD_BATCH = 96  # Pinecone's per-request limit for upsert_records with integrated embedding
FIELDS = ["doc", "section", "page", "text", "scope", "status", "effective_date", "doc_type", "source_url"]


class StoreError(RuntimeError):
    """Pinecone is misconfigured (wrong index type, missing key, ...)."""


@dataclass
class Match:
    id: str
    score: float
    metadata: dict[str, Any]


class VectorStore(Protocol):
    """Records are {"id", "text", "metadata", optional "embed_text"}."""

    def search(self, text: str, top_k: int, filter: dict[str, Any]) -> list[Match]: ...
    def upsert(self, records: list[dict[str, Any]]) -> None: ...
    def list_ids(self, prefix: str) -> list[str]: ...
    def delete(self, ids: list[str]) -> None: ...
    def count(self) -> int: ...


def _list_ids(index: Any, prefix: str, namespace: str) -> list[str]:
    ids: list[str] = []
    for page in index.list(prefix=prefix, namespace=namespace):  # one ListResponse per page
        ids.extend(item.id for item in page if item.id)
    return ids


class IntegratedStore:
    """Index with integrated embedding: upsert text, search by text."""

    def __init__(self, api_key: str, index_name: str, *, model: str = "llama-text-embed-v2",
                 namespace: str = "compliance", pc: Any = None) -> None:
        if pc is None:
            from pinecone import Pinecone  # imported lazily so tests don't need network

            pc = Pinecone(api_key=api_key)
        self.namespace = namespace
        if not pc.has_index(index_name):
            # A brand-new index; existing indexes are only ever read, never changed.
            pc.create_index_for_model(
                name=index_name, cloud="aws", region="us-east-1",
                embed={"model": model, "field_map": {"text": "text"}},
            )
        elif getattr(pc.describe_index(index_name), "embed", None) is None:
            raise StoreError(
                f"Pinecone index '{index_name}' exists but has no integrated embedding (it was made for "
                "EMBED_PROVIDER=ollama). Set PINECONE_INDEX to a new name, e.g. compliance-copilot-cloud; "
                "the existing index is left untouched."
            )
        self.index = pc.Index(name=index_name)

    def search(self, text: str, top_k: int, filter: dict[str, Any]) -> list[Match]:
        res = self.index.search(namespace=self.namespace, top_k=top_k, inputs={"text": text},
                                filter=filter, fields=FIELDS)
        return [Match(h.id, float(h.score), dict(h.fields or {})) for h in res.result.hits]

    def upsert(self, records: list[dict[str, Any]]) -> None:
        rows = [{"_id": r["id"], **r["metadata"], "text": r["text"]} for r in records]
        for i in range(0, len(rows), RECORD_BATCH):
            self.index.upsert_records(namespace=self.namespace, records=rows[i : i + RECORD_BATCH])

    def list_ids(self, prefix: str) -> list[str]:
        return _list_ids(self.index, prefix, self.namespace)

    def delete(self, ids: list[str]) -> None:
        for i in range(0, len(ids), 1000):
            self.index.delete(ids=ids[i : i + 1000], namespace=self.namespace)

    def count(self) -> int:
        ns = self.index.describe_index_stats().namespaces.get(self.namespace)
        return int(getattr(ns, "vector_count", 0) or 0)


class DenseStore:
    """Classic dense index: we embed (Ollama bge-m3) and send vectors."""

    def __init__(self, api_key: str, index_name: str, embed: Callable[[list[str]], list[list[float]]], *,
                 namespace: str = "", pc: Any = None) -> None:
        if pc is None:
            from pinecone import Pinecone, ServerlessSpec

            pc = Pinecone(api_key=api_key)
            spec = ServerlessSpec(cloud="aws", region="us-east-1")
        else:
            spec = {"serverless": {"cloud": "aws", "region": "us-east-1"}}
        if not pc.has_index(index_name):
            pc.create_index(name=index_name, dimension=EMBED_DIM, metric="cosine", spec=spec)
        elif getattr(pc.describe_index(index_name), "embed", None) is not None:
            raise StoreError(
                f"Pinecone index '{index_name}' uses integrated embedding (EMBED_PROVIDER=pinecone). "
                "Use a different PINECONE_INDEX for Ollama embeddings; the existing index is left untouched."
            )
        self.index = pc.Index(name=index_name)
        self.embed, self.namespace = embed, namespace
        self._last: tuple[str, list[float]] | None = None  # a comparison searches the same text per state

    def _vector(self, text: str) -> list[float]:
        if not self._last or self._last[0] != text:
            self._last = (text, self.embed([text])[0])
        return self._last[1]

    def search(self, text: str, top_k: int, filter: dict[str, Any]) -> list[Match]:
        res = self.index.query(vector=self._vector(text), top_k=top_k, filter=filter,
                               include_metadata=True, namespace=self.namespace)
        return [Match(m.id, float(m.score), dict(m.metadata or {})) for m in res.matches]

    def upsert(self, records: list[dict[str, Any]]) -> None:
        for i in range(0, len(records), EMBED_BATCH):
            batch = records[i : i + EMBED_BATCH]
            vectors = self.embed([r.get("embed_text") or r["text"] for r in batch])
            self.index.upsert(vectors=[{"id": r["id"], "values": v, "metadata": {**r["metadata"], "text": r["text"]}}
                                       for r, v in zip(batch, vectors)], namespace=self.namespace, show_progress=False)

    def list_ids(self, prefix: str) -> list[str]:
        return _list_ids(self.index, prefix, self.namespace)

    def delete(self, ids: list[str]) -> None:
        for i in range(0, len(ids), 1000):
            self.index.delete(ids=ids[i : i + 1000], namespace=self.namespace)

    def count(self) -> int:
        return int(self.index.describe_index_stats().total_vector_count)


def make_store(settings: Any) -> tuple[VectorStore | None, str]:
    """Build the store for `settings.embed_provider`. Returns (store, "") or (None, reason)."""
    if not settings.pinecone_api_key:
        return None, "PINECONE_API_KEY is not set"
    try:
        if settings.embed_provider == "pinecone":
            return IntegratedStore(settings.pinecone_api_key, settings.pinecone_index,
                                   model=settings.pinecone_embed_model, namespace=settings.pinecone_namespace), ""
        from app.llm import OllamaClient

        embedder = OllamaClient(base_url=settings.ollama_url, embed_model=settings.embed_model)
        return DenseStore(settings.pinecone_api_key, settings.pinecone_index, embedder.embed), ""
    except StoreError as e:
        return None, str(e)
    except Exception as e:  # network, auth or quota errors: keep the app up, report in /api/health
        return None, f"Pinecone unavailable: {e.__class__.__name__}: {str(e)[:160]}"


def scope_filter(states: list[str], statuses: tuple[str, ...] = ("current",)) -> dict[str, Any]:
    """Pinecone metadata filter: central law plus the given states, current docs only by default."""
    status: Any = statuses[0] if len(statuses) == 1 else {"$in": list(statuses)}
    return {"status": status, "scope": {"$in": ["central", *states]}}
