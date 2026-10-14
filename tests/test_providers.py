"""LLM provider switching and Groq error handling, with the Groq SDK mocked."""

from __future__ import annotations

from types import SimpleNamespace

import groq
import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import QUOTA_MESSAGE, GroqClient, LLMError, OllamaClient, QuotaExceeded, make_llm
from app.main import create_app
from app.services import Services

REQ = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def rate_limited() -> groq.RateLimitError:
    return groq.RateLimitError("rate limit", response=httpx.Response(429, request=REQ), body=None)


def reply(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeGroqSDK:
    """Stands in for groq.Groq: records calls and plays back scripted results or exceptions."""

    def __init__(self, *results, models=("qwen-test", "llama-test")):
        self.results, self.calls = list(results), []
        self.models_calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=self._list)
        self._models = models

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def _list(self):
        self.models_calls += 1
        return SimpleNamespace(data=[SimpleNamespace(id=m) for m in self._models])


def groq_client(sdk, model="qwen-test"):
    return GroqClient("key", model, client=sdk, backoff=0)


def test_make_llm_switches_provider():
    assert isinstance(make_llm(Settings(llm_provider="groq", _env_file=None)), GroqClient)
    ollama = make_llm(Settings(llm_provider="ollama", _env_file=None))
    assert isinstance(ollama, OllamaClient)
    assert ollama.options["temperature"] == 0.1 and ollama.options["num_predict"] == 700


def test_groq_same_generation_settings_and_json_mode():
    sdk = FakeGroqSDK(reply('{"route": "rag"}'))
    out = groq_client(sdk).chat([{"role": "user", "content": "Reply as JSON"}], json_mode=True)
    call = sdk.calls[0]
    assert out == '{"route": "rag"}'
    assert (call["model"], call["temperature"], call["max_tokens"]) == ("qwen-test", 0.1, 700)
    assert call["response_format"] == {"type": "json_object"}


def test_groq_429_becomes_quota_exceeded_without_retry():
    sdk = FakeGroqSDK(rate_limited(), reply("never reached"))
    with pytest.raises(QuotaExceeded):
        groq_client(sdk).chat([{"role": "user", "content": "hi"}])
    assert len(sdk.calls) == 1


def test_groq_timeout_retried_once():
    sdk = FakeGroqSDK(groq.APITimeoutError(request=REQ), reply("ok"))
    assert groq_client(sdk).chat([{"role": "user", "content": "hi"}]) == "ok"
    sdk = FakeGroqSDK(groq.APITimeoutError(request=REQ), groq.APITimeoutError(request=REQ))
    with pytest.raises(LLMError, match="timed out twice"):
        groq_client(sdk).chat([{"role": "user", "content": "hi"}])


def test_groq_missing_key_or_model_is_a_clear_error():
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        GroqClient("", "m").chat([{"role": "user", "content": "hi"}])
    with pytest.raises(LLMError, match="GROQ_MODEL"):
        groq_client(FakeGroqSDK(), model="").chat([{"role": "user", "content": "hi"}])


def test_groq_has_no_embeddings():
    with pytest.raises(LLMError):
        groq_client(FakeGroqSDK()).embed(["x"])


def test_groq_health_uses_model_list_and_caches():
    sdk = FakeGroqSDK()
    assert groq_client(sdk).health()["ok"]
    c = groq_client(sdk, model="gone-model")
    h = c.health()
    assert not h["ok"] and "list_models" in h["issues"][0]
    c.health()
    assert sdk.models_calls == 2 and sdk.calls == []  # no tokens spent, second check cached


def test_groq_stream():
    chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t))]) for t in ("Yes", " [1].")]
    assert "".join(groq_client(FakeGroqSDK(iter(chunks))).stream([{"role": "user", "content": "q"}])) == "Yes [1]."


def test_quota_message_through_the_api(settings, store):
    settings.llm_provider = "groq"
    store.add("gratuity fixed term employee", doc="SS Code", page=1, scope="central", status="current", doc_type="act")
    llm = groq_client(FakeGroqSDK(rate_limited(), rate_limited()))
    with TestClient(create_app(Services(settings, llm, store))) as c:
        body = c.post("/api/chat", json={"message": "gratuity fixed term employee"}).json()
        text = c.post("/api/chat/stream", json={"message": "gratuity fixed term employee"}).text
    assert body["type"] == "not_found" and body["answer"] == QUOTA_MESSAGE and body["quota_exceeded"] is True
    assert '"quota_exceeded": true' in text


def test_health_reports_provider(settings, store):
    settings.llm_provider, settings.groq_model = "groq", "qwen-test"
    with TestClient(create_app(Services(settings, groq_client(FakeGroqSDK()), store))) as c:
        h = c.get("/api/health").json()
    assert (h["provider"], h["model"], h["llm_ok"], h["status"]) == ("groq", "qwen-test", True, "ok")
    assert "key" not in str(h).lower()
