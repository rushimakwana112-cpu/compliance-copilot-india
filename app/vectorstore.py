"""Pinecone wrapper.

The app only needs five operations, so we hide the SDK behind a tiny class.
That keeps Pinecone-specific code in one file and lets tests swap in an
in-memory fake with the same methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

EMBED_DIM = 1024  # bge-m3 output size


@dataclass
class Match:
    id: str
    score: float
    metadata: dict[str, Any]


class VectorStore(Protocol):
    def query(self, vector: list[float], top_k: int, filter: dict[str, Any]) -> list[Match]: ...
    def upsert(self, records: list[dict[str, Any]]) -> None: ...
    def list_ids(self, prefix: str) -> list[str]: ...
    def delete(self, ids: list[str]) -> None: ...
    def count(self) -> int: ...


class PineconeStore:
    """Serverless Pinecone index. Creates the index on first use if it doesn't exist."""

    def __init__(self, api_key: str, index_name: str, *, create: bool = True) -> None:
        from pinecone import Pinecone, ServerlessSpec  # imported lazily so tests don't need network

        self.pc = Pinecone(api_key=api_key)
        if create and not self.pc.has_index(index_name):
            # Creating is safe: it never touches existing data. Free tier runs on aws/us-east-1.
            self.pc.create_index(
                name=index_name,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self.index = self.pc.Index(index_name)

    def query(self, vector: list[float], top_k: int, filter: dict[str, Any]) -> list[Match]:
        res = self.index.query(vector=vector, top_k=top_k, filter=filter, include_metadata=True)
        return [Match(m.id, float(m.score), dict(m.metadata or {})) for m in res.matches]

    def upsert(self, records: list[dict[str, Any]]) -> None:
        for i in range(0, len(records), 100):
            self.index.upsert(vectors=records[i : i + 100])

    def list_ids(self, prefix: str) -> list[str]:
        ids: list[str] = []
        for page in self.index.list(prefix=prefix):
            ids.extend(page)
        return ids

    def delete(self, ids: list[str]) -> None:
        for i in range(0, len(ids), 1000):
            self.index.delete(ids=ids[i : i + 1000])

    def count(self) -> int:
        return int(self.index.describe_index_stats().total_vector_count)


def scope_filter(states: list[str], statuses: tuple[str, ...] = ("current",)) -> dict[str, Any]:
    """Pinecone metadata filter: central law plus the given states, current docs only by default."""
    status: Any = statuses[0] if len(statuses) == 1 else {"$in": list(statuses)}
    return {"status": status, "scope": {"$in": ["central", *states]}}
