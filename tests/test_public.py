"""Public demo mode: ingest blocked, per-IP rate limit, answer cache, request size limits."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.limits import AnswerCache, RateLimiter, normalize
from app.main import create_app
from app.schemas import ChatRequest, ChatResponse

GRATUITY = "Gratuity is payable to a fixed term employee [1]."


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def public(settings):
    settings.public_mode = True
    return settings


@pytest.fixture
def indexed(store, llm):
    store.add("gratuity fixed term employee continuous service", doc="SS Code", page=1,
              scope="central", status="current", doc_type="act")
    llm.responder = lambda messages, json_mode: GRATUITY
    return store


# ----------------------------------------------------------------- ingest blocked
def test_public_mode_blocks_ingest(public, services):
    with TestClient(create_app(services)) as c:
        up = c.post("/api/ingest/file", files={"file": ("a.pdf", b"%PDF-1.4 x", "application/pdf")})
        link = c.post("/api/ingest/url", json={"url": "https://labour.gov.in/a.pdf"})
        health = c.get("/api/health").json()
    assert up.status_code == link.status_code == 403
    assert "turned off on the public demo" in up.json()["detail"]
    assert health["public_mode"] is True


def test_ingest_allowed_when_not_public(services):
    with TestClient(create_app(services)) as c:
        r = c.post("/api/ingest/url", json={"url": "http://labour.gov.in/a.pdf"})
    assert r.status_code == 422  # reaches the endpoint (rejects http), not the public-mode 403


# ----------------------------------------------------------------- rate limiter
def test_rate_limiter_minute_window():
    clock = Clock()
    rl = RateLimiter(per_min=10, per_day=60, clock=clock)
    assert all(rl.hit("1.1.1.1").allowed for _ in range(10))
    refused = rl.hit("1.1.1.1")
    assert not refused.allowed and refused.reason == "minute" and 1 <= refused.retry_after <= 61
    assert rl.hit("2.2.2.2").allowed  # other IPs are unaffected
    clock.t += 60
    assert rl.hit("1.1.1.1").allowed  # the window slid


def test_rate_limiter_daily_cap_resets_next_day():
    clock = Clock(86400 * 20_000)
    rl = RateLimiter(per_min=10, per_day=60, clock=clock)
    for _ in range(60):
        assert rl.hit("ip").allowed
        clock.t += 7  # under 10/min, so only the daily cap can trip
    refused = rl.hit("ip")
    assert not refused.allowed and refused.reason == "day"
    clock.t = 86400 * 20_001 + 1
    assert rl.hit("ip").allowed


def test_api_returns_429_as_an_answer(public, services, indexed):
    public.rate_limit_per_min = 2
    services.limiter = RateLimiter(per_min=2, per_day=60)
    with TestClient(create_app(services)) as c:
        ok = [c.post("/api/chat", json={"message": "gratuity fixed term employee"}) for _ in range(2)]
        limited = c.post("/api/chat/stream", json={"message": "gratuity fixed term employee"})
        other_ip = c.post("/api/chat", json={"message": "gratuity fixed term employee"},
                          headers={"X-Forwarded-For": "9.9.9.9, 10.0.0.1"})
    assert [r.status_code for r in ok] == [200, 200]
    assert limited.status_code == 429 and int(limited.headers["Retry-After"]) > 0
    body = limited.json()
    assert body["type"] == "not_found" and "questions in a minute" in body["answer"]
    assert other_ip.status_code == 200


def test_no_rate_limit_when_not_public(services, indexed):
    services.limiter = RateLimiter(per_min=1, per_day=1)
    with TestClient(create_app(services)) as c:
        codes = {c.post("/api/chat", json={"message": "hello there"}).status_code for _ in range(3)}
    assert codes == {200}


# ----------------------------------------------------------------- answer cache
def test_normalize():
    assert normalize("  What is the PF  rate?? ") == normalize("what is the pf rate")


def test_cache_is_lru_and_skips_refusals():
    cache = AnswerCache(size=2)
    rag = ChatResponse(type="rag", answer="a")
    cache.put("q1", ["gujarat"], rag)
    cache.put("q2", ["gujarat"], rag)
    cache.get("q1", ["gujarat"])  # q1 is now most recent
    cache.put("q3", ["gujarat"], rag)
    assert cache.get("q2", ["gujarat"]) is None and cache.get("q1", ["gujarat"]) is not None
    assert cache.get("q1", ["kerala"]) is None  # scope is part of the key
    cache.put("q4", ["gujarat"], ChatResponse(type="not_found", answer="no"))
    assert cache.get("q4", ["gujarat"]) is None
    hit = cache.get("q1", ["gujarat"])
    assert hit.cached is True
    hit.answer = "changed"
    assert cache.get("q1", ["gujarat"]).answer == "a"  # callers get copies


def test_repeat_question_is_served_from_cache(services, indexed, llm):
    with TestClient(create_app(services)) as c:
        first = c.post("/api/chat", json={"message": "Gratuity fixed term employee?"}).json()
        calls = len(llm.calls)
        second = c.post("/api/chat", json={"message": "gratuity  fixed term employee"}).json()
        streamed = c.post("/api/chat/stream", json={"message": "gratuity fixed term employee"}).text
    assert first["type"] == "rag" and "cached" not in first
    assert second["cached"] is True and second["answer"] == first["answer"]
    assert '"cached": true' in streamed and "event: token" not in streamed
    assert len(llm.calls) == calls  # no model call for the repeats


def test_ingestion_clears_the_cache(services, monkeypatch):
    from ingest import run

    services.cache.put("q", ["gujarat"], ChatResponse(type="rag", answer="a"))
    monkeypatch.setattr(run, "ingest_file", lambda *a, **k: type("Out", (), {"status": "indexed"})())
    with TestClient(create_app(services)) as c:
        r = c.post("/api/ingest/file", files={"file": ("a.pdf", b"%PDF-1.4 x", "application/pdf")})
    assert r.status_code == 200 and len(services.cache) == 0


# ----------------------------------------------------------------- request limits
def test_message_and_history_limits(services):
    with TestClient(create_app(services)) as c:
        assert c.post("/api/chat", json={"message": "x" * 1001}).status_code == 422
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(20)]
    req = ChatRequest(message="and for Kerala?", history=history)
    assert len(req.history) == 8 and req.history[-1].content == "m19"


def test_feedback_is_capped_and_rate_limited_in_public_mode(public, services, tmp_path):
    """One feedback call used to append 9 MB to logs/feedback.jsonl, with no limit on how often."""
    public.rate_limit_per_min = 2
    huge = {"value": "up", "question": "q" * 2_000_000, "answer": {"type": "rag" * 1000, "answer": "a" * 2_000_000}}
    with TestClient(create_app(services)) as c:
        codes = [c.post("/api/feedback", json=huge).status_code for _ in range(3)]
        chat = c.post("/api/chat", json={"message": "hello there"})
    assert codes == [200, 200, 429]
    assert chat.status_code == 200  # feedback has its own budget: rating answers doesn't use up questions
    lines = (tmp_path / "logs" / "feedback.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and all(len(line) < 6000 for line in lines)
