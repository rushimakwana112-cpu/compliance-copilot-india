"""Shared test fixtures: fake Ollama and fake Pinecone, so tests run offline.

The fakes implement the same methods as the real clients (see app/llm.py and
app/vectorstore.py). The fake embedder hashes words into a small vector, so
texts that share words get a higher cosine similarity, which is enough to test
retrieval logic without a real model.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from app.config import Settings
from app.services import Services
from app.vectorstore import Match

DIM = 256


def fake_embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeLLM:
    """Scriptable LLM. `responder(messages, json_mode)` returns the reply text."""

    model = "fake-qwen"

    def __init__(self, responder: Callable[[list[dict[str, str]], bool], str] | None = None) -> None:
        self.responder = responder or (lambda messages, json_mode: "NOT_FOUND")
        self.calls: list[list[dict[str, str]]] = []
        self.healthy = True

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        self.calls.append(messages)
        return self.responder(messages, json_mode)

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        text = self.chat(messages)
        for i in range(0, len(text), 7):
            yield text[i : i + 7]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [fake_embed(t) for t in texts]

    def health(self) -> dict[str, Any]:
        return {"ok": self.healthy, "issues": [] if self.healthy else ["Ollama is not running"]}


def _matches(meta: dict[str, Any], flt: dict[str, Any]) -> bool:
    for key, cond in flt.items():
        value = meta.get(key)
        if isinstance(cond, dict):
            if "$in" in cond and value not in cond["$in"]:
                return False
            if "$eq" in cond and value != cond["$eq"]:
                return False
        elif value != cond:
            return False
    return True


class FakeStore:
    """In-memory vector index with Pinecone-style metadata filters."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.searches: list[str] = []

    def search(self, text: str, top_k: int, filter: dict[str, Any]) -> list[Match]:
        self.searches.append(text)
        vector = fake_embed(text)
        scored = [
            Match(rid, sum(a * b for a, b in zip(vector, r["values"])), r["metadata"])
            for rid, r in self.records.items()
            if _matches(r["metadata"], filter)
        ]
        return sorted(scored, key=lambda m: m.score, reverse=True)[:top_k]

    def upsert(self, records: list[dict[str, Any]]) -> None:
        """Same record shape the real stores take: {"id", "text", "metadata"}."""
        for r in records:
            self.records[r["id"]] = {"id": r["id"], "values": fake_embed(r["text"]),
                                     "metadata": {**r["metadata"], "text": r["text"]}}

    def list_ids(self, prefix: str) -> list[str]:
        return [i for i in self.records if i.startswith(prefix)]

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self.records.pop(i, None)

    def count(self) -> int:
        return len(self.records)

    def add(self, text: str, **meta: Any) -> None:
        """Test helper: index one passage."""
        rid = f"{meta.get('doc', 'doc')}#{len(self.records)}"
        self.records[rid] = {"id": rid, "values": fake_embed(text), "metadata": {"text": text, **meta}}


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(pinecone_api_key="test", score_threshold=0.2, _env_file=None)


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def services(settings, llm, store) -> Services:
    return Services(settings, llm, store)


@pytest.fixture(autouse=True)
def isolated_files(tmp_path, monkeypatch):
    """Keep tests from writing to the real data/manifest.json and logs/."""
    from app import catalog, config

    monkeypatch.setattr(catalog, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    catalog._jobs.clear()
    yield


ACT_PAGES = [
    """CHAPTER V
GRATUITY
53. Payment of gratuity.—(1) Gratuity shall be payable to an employee on the termination of his
employment after he has rendered continuous service for not less than five years.
Provided that the completion of continuous service of five years shall not be necessary where
the termination of the employment of any employee is due to death or disablement or in case of
fixed term employee.""",
    """54. Determination of gratuity.—(1) For every completed year of service the employer shall pay
gratuity to an employee at the rate of fifteen days wages based on the rate of wages last drawn.
55. Nomination.—Each employee shall make nomination for the purpose of gratuity.""",
]


@pytest.fixture
def act_pdf(tmp_path):
    """A two-page PDF that looks like an Act, built with reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    path = tmp_path / "ss-code.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    for page in ACT_PAGES:
        y = 800
        for line in page.splitlines():
            c.drawString(40, y, line)
            y -= 16
        c.showPage()
    c.save()
    return path
