"""Pinecone stores with the SDK mocked: integrated embedding (cloud) and dense bge-m3 (local)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import vectorstore
from app.config import Settings
from app.vectorstore import DenseStore, IntegratedStore, StoreError, make_store, scope_filter


class FakeIndex:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def search(self, **kw):
        self.calls.append(("search", kw))
        hit = SimpleNamespace(id="ss#1", score=0.81, fields={"doc": "SS Code", "text": "gratuity passage", "page": 4})
        return SimpleNamespace(result=SimpleNamespace(hits=[hit]))

    def upsert_records(self, **kw):
        self.calls.append(("upsert_records", kw))

    def query(self, **kw):
        self.calls.append(("query", kw))
        return SimpleNamespace(matches=[SimpleNamespace(id="a", score=0.7, metadata={"doc": "D"})])

    def upsert(self, **kw):
        self.calls.append(("upsert", kw))

    def list(self, **kw):
        # Pinecone yields pages; iterating a page yields items with an .id
        yield [SimpleNamespace(id="doc#1"), SimpleNamespace(id="doc#2")]
        yield [SimpleNamespace(id="doc#3")]

    def delete(self, **kw):
        self.calls.append(("delete", kw))


class FakePC:
    def __init__(self, existing: dict[str, object] | None = None):
        self.existing = existing or {}
        self.created: list[tuple[str, dict]] = []
        self.index = FakeIndex()

    def has_index(self, name):
        return name in self.existing

    def describe_index(self, name):
        return SimpleNamespace(embed=self.existing[name])

    def create_index_for_model(self, **kw):
        self.created.append(("integrated", kw))

    def create_index(self, **kw):
        self.created.append(("dense", kw))

    def Index(self, name):
        return self.index


def test_integrated_creates_new_index_with_hosted_model():
    pc = FakePC()
    IntegratedStore("k", "compliance-copilot-cloud", pc=pc)
    kind, kw = pc.created[0]
    assert kind == "integrated" and kw["name"] == "compliance-copilot-cloud"
    assert kw["embed"] == {"model": "llama-text-embed-v2", "field_map": {"text": "text"}}


def test_integrated_never_touches_an_existing_dense_index():
    pc = FakePC(existing={"compliance-copilot": None})  # exists, no integrated embedding
    with pytest.raises(StoreError, match="left untouched"):
        IntegratedStore("k", "compliance-copilot", pc=pc)
    assert pc.created == [] and pc.index.calls == []


def test_integrated_reuses_its_own_index():
    pc = FakePC(existing={"cloud": SimpleNamespace(model="llama-text-embed-v2")})
    IntegratedStore("k", "cloud", pc=pc)
    assert pc.created == []


def test_integrated_search_by_text_with_jurisdiction_filter():
    pc = FakePC()
    store = IntegratedStore("k", "cloud", pc=pc, namespace="ns")
    flt = scope_filter(["gujarat"])
    matches = store.search("gratuity for fixed-term", 8, flt)
    _, kw = pc.index.calls[0]
    assert kw["inputs"] == {"text": "gratuity for fixed-term"} and kw["filter"] == flt
    assert kw["namespace"] == "ns" and kw["top_k"] == 8 and "text" in kw["fields"]
    assert (matches[0].id, matches[0].score, matches[0].metadata["doc"]) == ("ss#1", 0.81, "SS Code")


def test_integrated_upserts_text_plus_flat_metadata_in_batches():
    pc = FakePC()
    store = IntegratedStore("k", "cloud", pc=pc)
    store.upsert([{"id": f"d#{i}", "text": f"chunk {i}", "embed_text": "ignored", "metadata": {"doc": "D", "page": 1}}
                  for i in range(100)])
    batches = [kw["records"] for name, kw in pc.index.calls if name == "upsert_records"]
    assert [len(b) for b in batches] == [96, 4]
    assert batches[0][0] == {"_id": "d#0", "doc": "D", "page": 1, "text": "chunk 0"}


def test_list_ids_reads_every_page():
    assert IntegratedStore("k", "cloud", pc=FakePC()).list_ids("doc#") == ["doc#1", "doc#2", "doc#3"]


def test_dense_store_embeds_once_per_question_and_refuses_integrated_index():
    calls = []

    def embed(texts):
        calls.append(texts)
        return [[0.1] * 3 for _ in texts]

    store = DenseStore("k", "local", embed, pc=FakePC())
    store.search("q", 8, {}); store.search("q", 8, {})  # a 2-state comparison searches the same text twice
    assert calls == [["q"]]
    store.upsert([{"id": "a", "text": "body", "embed_text": "Section 1\nbody", "metadata": {"doc": "D"}}])
    assert calls[-1] == ["Section 1\nbody"]
    with pytest.raises(StoreError):
        DenseStore("k", "cloud", embed, pc=FakePC(existing={"cloud": SimpleNamespace(model="m")}))


def test_make_store_switches_on_embed_provider(monkeypatch):
    built = {}
    monkeypatch.setattr(vectorstore, "IntegratedStore", lambda *a, **k: built.setdefault("pinecone", (a, k)))
    monkeypatch.setattr(vectorstore, "DenseStore", lambda *a, **k: built.setdefault("ollama", (a, k)))
    store, err = make_store(Settings(pinecone_api_key="k", embed_provider="pinecone", _env_file=None))
    assert err == "" and "pinecone" in built
    make_store(Settings(pinecone_api_key="k", embed_provider="ollama", _env_file=None))
    assert "ollama" in built
    assert make_store(Settings(pinecone_api_key="", _env_file=None)) == (None, "PINECONE_API_KEY is not set")
