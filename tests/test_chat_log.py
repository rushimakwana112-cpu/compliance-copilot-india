"""logs/chats.jsonl carries retrieval details for every route and outcome."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import create_app


def _records() -> list[dict]:
    return [json.loads(ln) for ln in (config.LOGS_DIR / "chats.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def seeded(store):
    for i in range(7):
        store.add(f"gratuity fixed term employee continuous service rule {i}", doc=f"SS Code part {i}",
                  section=f"Section {50 + i}", page=i + 1, scope="central", status="current", doc_type="act")
    return store


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
def test_llm_not_found_still_logs_scores(services, seeded, llm, endpoint):
    llm.responder = lambda messages, json_mode: "NOT_FOUND"  # passes the threshold, then the model refuses
    with TestClient(create_app(services)) as c:
        c.post(endpoint, json={"message": "gratuity fixed term employee"})
    rec = _records()[-1]
    assert rec["type"] == "not_found" and rec["route"] == "rag"
    assert len(rec["top_scores"]) == 5 and rec["top_scores"] == sorted(rec["top_scores"], reverse=True)
    assert len(rec["retrieved"]) == 7 and {"doc", "section", "score"} == set(rec["retrieved"][0])
    assert rec["retrieved"][0]["section"].startswith("Section ")
    assert rec["index"] == services.settings.pinecone_index
    assert rec["filters"] == [{"status": {"$in": ["current", "commentary"]}, "scope": {"$in": ["central", "gujarat"]}}]
    assert rec["retrieval"] == "searched"


def test_tool_route_logs_the_same_fields(services):
    with TestClient(create_app(services)) as c:
        c.post("/api/chat", json={"message": "Calculate PF for basic ₹18,000"})
    rec = _records()[-1]
    assert rec["route"] == "tool"
    assert rec["retrieval"] == "not run" and rec["top_scores"] == [] and rec["retrieved"] == []
    assert rec["index"] == services.settings.pinecone_index and "filters" in rec


def test_search_error_is_logged(services):
    with TestClient(create_app(services)) as c:  # empty fake index: 0 matches
        c.post("/api/chat", json={"message": "gratuity fixed term employee"})
    rec = _records()[-1]
    assert rec["type"] == "error" and rec["retrieval"] == "searched" and rec["filters"]
