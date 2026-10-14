"""API contract tests: response shapes the frontend depends on."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services import Services


@pytest.fixture
def client(services):
    with TestClient(create_app(services)) as c:
        yield c


def test_index_serves_frontend(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Compliance Copilot" in r.text


def test_health_shape(client):
    body = client.get("/api/health").json()
    assert set(body) >= {"status", "model", "pinecone", "docs"}
    assert body["status"] == "ok"
    assert body["pinecone"] is True
    assert isinstance(body["docs"], int)


def test_health_reports_ollama_down(services, llm):
    llm.healthy = False
    with TestClient(create_app(services)) as c:
        body = c.get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["ollama"] is False
    assert any("Ollama" in i for i in body["issues"])


def test_health_reports_missing_pinecone(settings, llm):
    with TestClient(create_app(Services(settings, llm, None, "PINECONE_API_KEY is not set in .env"))) as c:
        body = c.get("/api/health").json()
    assert body["pinecone"] is False
    assert any("PINECONE_API_KEY" in i for i in body["issues"])


def test_states_shape(client):
    states = client.get("/api/states").json()
    assert {"id", "name", "aliases"} <= set(states[0])
    assert any(s["id"] == "gujarat" and "ahmedabad" in s["aliases"] for s in states)


def test_documents_shape(client):
    docs = client.get("/api/documents").json()
    assert docs, "sources.csv ships with example rows"
    assert all({"name", "scope", "status", "meta"} <= set(d) for d in docs)
