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
    assert body["llm_ok"] is False
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


def _rag_llm(messages, json_mode):
    return "Yes. Fixed-term employees qualify for gratuity [1]."


def test_chat_rag_contract(services, store, llm):
    llm.responder = _rag_llm
    store.add("gratuity fixed term employee continuous service", doc="Code on Social Security, 2020", section="Section 53",
              page=40, scope="central", status="current", doc_type="act")
    with TestClient(create_app(services)) as c:
        r = c.post("/api/chat", json={"message": "Gratuity rule for fixed term employee in Surat?", "history": [],
                                      "jurisdiction": {"central": True, "state": "maharashtra", "mode": "auto", "detected": False}})
    body = r.json()
    assert r.status_code == 200
    assert body["type"] == "rag"
    assert body["scope"] == {"state": "gujarat", "how": "detected"}  # backend re-detects "Surat"
    assert {"doc", "section", "page", "score", "snippet", "url"} <= set(body["citations"][0])
    assert isinstance(body["latency_ms"], int)


def test_chat_not_found_contract(client):
    body = client.post("/api/chat", json={"message": "Gratuity rule in Dubai?"}).json()
    assert body["type"] == "not_found" and body["citations"] == []


def test_chat_stream_events(services, store, llm):
    llm.responder = _rag_llm
    store.add("gratuity fixed term employee", doc="SS Code", page=1, scope="central", status="current", doc_type="act")
    with TestClient(create_app(services)) as c:
        text = c.post("/api/chat/stream", json={"message": "gratuity fixed term employee"}).text
    names = [ln.split(": ", 1)[1] for ln in text.splitlines() if ln.startswith("event: ")]
    assert names[0] == "meta" and names[-1] == "done" and "token" in names


def test_feedback_logged(client, tmp_path):
    r = client.post("/api/feedback", json={"value": "down", "answer": {"type": "rag", "answer": "x"}, "question": "q?"})
    assert r.json() == {"ok": True}
    assert '"value": "down"' in (tmp_path / "logs" / "feedback.jsonl").read_text()


def test_ingest_rejects_non_pdf_and_http(client):
    r = client.post("/api/ingest/file", files={"file": ("a.pdf", b"hello", "application/pdf")}, data={"scope": "central"})
    assert r.status_code == 415
    assert client.post("/api/ingest/url", json={"url": "http://example.gov.in/a.pdf"}).status_code == 422
    assert client.post("/api/ingest/file", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
                       data={"scope": "atlantis"}).status_code == 422


def test_ingest_file_shows_in_documents(client, act_pdf, monkeypatch, tmp_path):
    from app import main

    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    r = client.post("/api/ingest/file", files={"file": ("EPFO_circular.pdf", act_pdf.read_bytes(), "application/pdf")},
                    data={"scope": "gujarat"})
    assert r.status_code == 200 and r.json()["name"] == "EPFO circular"
    docs = {d["name"]: d for d in client.get("/api/documents").json()}
    assert docs["EPFO circular"]["scope"] == "gujarat"  # indexed in the background by the time we ask
