"""LLM provider switching and Groq error handling, with the Groq SDK mocked."""

from __future__ import annotations

from types import SimpleNamespace

import groq
import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import (BUSY_MESSAGE, QUOTA_MESSAGE, Busy, GroqClient, LLMError, OllamaClient, QuotaExceeded,
                     classify_rate_limit, estimate_tokens, make_llm)
from app.main import create_app
from app.services import Services

REQ = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def rate_limited() -> groq.RateLimitError:
    return groq.RateLimitError("rate limit", response=httpx.Response(429, request=REQ), body=None)


def groq_429(limit: str, wait: str, retry_after: str = "") -> groq.RateLimitError:
    """A 429 shaped like Groq's: the limit is named in error.message, retry-after in the headers."""
    words = {"TPM": "tokens per minute", "RPM": "requests per minute", "TPD": "tokens per day",
             "RPD": "requests per day"}[limit]
    msg = (f"Rate limit reached for model `qwen-test` in organization `org_x` service tier `on_demand` on {words} "
           f"({limit}): Limit 8000, Used 7400, Requested 2100. Please try again in {wait}. Need more tokens? Upgrade")
    body = {"message": msg, "type": "tokens", "code": "rate_limit_exceeded"}
    headers = {"retry-after": retry_after} if retry_after else {}
    return groq.RateLimitError(f"Error code: 429 - {{'error': {body}}}",
                               response=httpx.Response(429, request=REQ, headers=headers), body=body)


class Sleeps(list):
    def __call__(self, seconds: float) -> None:
        self.append(seconds)


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


@pytest.mark.parametrize("text, headers, expected", [
    ("on tokens per minute (TPM): Limit 8000. Please try again in 7.5s.", {}, ("minute", "TPM", 7.5)),
    ("on requests per minute (RPM): Limit 30. Please try again in 450ms.", {}, ("minute", "RPM", 0.45)),
    ("on tokens per day (TPD): Limit 500000. Please try again in 12m3.2s.", {}, ("day", "TPD", 723.2)),
    ("on requests per day (RPD): Limit 1000.", {"retry-after": "3600"}, ("day", "RPD", 3600.0)),
    ("rate limit", {"retry-after": "9"}, ("minute", "", 9.0)),  # unknown limit, short wait
    ("rate limit", {}, ("day", "", None)),  # unknown limit, no wait: don't hammer the API
])
def test_classify_rate_limit(text, headers, expected):
    assert classify_rate_limit(text, headers) == expected


def test_per_minute_limit_waits_and_retries_once(caplog):
    sleeps = Sleeps()
    sdk = FakeGroqSDK(groq_429("TPM", "7.5s"), reply("ok"))
    client = GroqClient("key", "qwen-test", client=sdk, sleep=sleeps)
    with caplog.at_level("WARNING", logger="app"):
        assert client.chat([{"role": "user", "content": "x" * 400}]) == "ok"
    assert sleeps == [7.5] and len(sdk.calls) == 2
    assert "window=minute limit=TPM" in caplog.text and "prompt_tokens=100" in caplog.text


def test_per_minute_limit_wait_is_capped_then_busy():
    sleeps = Sleeps()
    sdk = FakeGroqSDK(groq_429("TPM", "45s"), groq_429("TPM", "40s"), reply("never reached"))
    with pytest.raises(Busy) as e:
        GroqClient("key", "qwen-test", client=sdk, sleep=sleeps).chat([{"role": "user", "content": "hi"}])
    assert sleeps == [20.0] and len(sdk.calls) == 2  # waited at most 20 s, retried once
    assert str(e.value) == BUSY_MESSAGE and e.value.limit == "TPM" and e.value.window == "minute"


def test_daily_limit_is_not_retried():
    sleeps = Sleeps()
    sdk = FakeGroqSDK(groq_429("TPD", "12m3s", retry_after="723"), reply("never reached"))
    with pytest.raises(QuotaExceeded) as e:
        GroqClient("key", "qwen-test", client=sdk, sleep=sleeps).chat([{"role": "user", "content": "hi"}])
    assert sleeps == [] and len(sdk.calls) == 1 and str(e.value) == QUOTA_MESSAGE and e.value.limit == "TPD"


def test_busy_reaches_the_api_and_the_log(settings, store, tmp_path):
    import json

    from app import config

    store.add("gratuity fixed term employee", doc="SS Code", page=1, scope="central", status="current", doc_type="act")
    llm = GroqClient("key", "qwen-test", client=FakeGroqSDK(groq_429("TPM", "2s"), groq_429("TPM", "2s")),
                     sleep=Sleeps())
    with TestClient(create_app(Services(settings, llm, store))) as c:
        body = c.post("/api/chat", json={"message": "gratuity fixed term employee"}).json()
    assert body["answer"] == BUSY_MESSAGE and body["rate_limited"] == "minute" and "quota_exceeded" not in body
    rec = json.loads((config.LOGS_DIR / "chats.jsonl").read_text().splitlines()[-1])
    assert rec["rate_limit"]["window"] == "minute" and rec["rate_limit"]["limit"] == "TPM"
    assert rec["rate_limit"]["prompt_tokens"] > 0


def test_estimate_tokens():
    assert estimate_tokens([{"role": "user", "content": "a" * 400}, {"role": "system", "content": "b" * 4}]) == 101


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
