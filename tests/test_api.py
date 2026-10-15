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


def test_chat_not_found_contract(client, store):
    store.add("Monthly professional tax slab table", doc="Gujarat PT schedule", page=1, scope="gujarat",
              status="current", doc_type="notification")
    body = client.post("/api/chat", json={"message": "Gratuity rule for astronauts?"}).json()
    assert body["type"] == "not_found" and body["citations"] == []


def test_chat_search_failure_is_an_error_not_a_silent_not_found(client, store, caplog):
    # Empty index: 0 matches means a wrong index or filter, so it must not look like "not in my documents".
    body = client.post("/api/chat", json={"message": "Gratuity rule for astronauts?"}).json()
    assert body["type"] == "error" and body["answer"].startswith("Search failed: ")
    assert "0 matches" in body["answer"]
    assert any(r.levelname == "WARNING" and "Search failed" in r.getMessage() for r in caplog.records)

    def boom(*a, **k):
        raise ConnectionError("pinecone unreachable")

    store.search = boom
    body = client.post("/api/chat/stream", json={"message": "Gratuity rule in Kerala for astronauts?"}).text
    assert '"type": "error"' in body and "Search failed: ConnectionError: pinecone unreachable" in body


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


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
def test_foreign_place_is_refused_without_retrieval_or_llm(services, store, llm, endpoint):
    store.add("gratuity rule for employees", doc="SS Code", page=1, scope="central", status="current", doc_type="act")
    with TestClient(create_app(services)) as c:
        r = c.post(endpoint, json={"message": "Gratuity rule in Dubai?"})
    assert "I only cover Indian law" in r.text and "Dubai" in r.text and '"not_found"' in r.text
    assert store.searches == [] and llm.calls == []


def test_rates_endpoint_lists_figures_with_sources(services):
    with TestClient(create_app(services)) as c:
        body = c.get("/api/rates", params={"on": "2026-10-15"}).json()
        bad = c.get("/api/rates", params={"on": "not-a-date"})
    assert body["date"] == "2026-10-15"
    ceiling = body["rates"]["pf.wage_ceiling"]
    assert ceiling["value"] == 25000 and ceiling["source_doc"].startswith("MoLE notification S.O. 5109(E)")
    assert body["rates"]["gratuity.max_amount"]["value"] is None  # still to be notified: listed, not hidden
    assert body["due_dates"]["tds_salary_deposit"]["section"] == "rule 218(2)"
    assert "gujarat" in body["professional_tax"] and bad.status_code == 422
