"""Language-model providers behind one small interface.

    LLM_PROVIDER=groq    Groq's hosted API (free tier, fast). Needs GROQ_API_KEY and GROQ_MODEL.
    LLM_PROVIDER=ollama  A local Ollama server. Needs `ollama serve` and the model pulled.

Both get the same prompts, temperature (0.1) and max_tokens (700), so the
NOT_FOUND contract and the router's JSON output behave the same. The rest of
the app only sees `chat`, `stream`, `embed` and `health`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import httpx

T = TypeVar("T")
QUOTA_MESSAGE = "Daily demo limit reached. Please try again later."


class LLMError(RuntimeError):
    """Raised when the model provider is unreachable, misconfigured or returns an error."""


class QuotaExceeded(LLMError):
    """The provider answered 429: the free-tier limit is used up."""


class LLM(Protocol):
    """What the rest of the app needs from a language model (real or fake)."""

    model: str

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str: ...
    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def health(self) -> dict[str, Any]: ...


# ----------------------------------------------------------------- Ollama (local)
@dataclass
class OllamaClient:
    """Calls Ollama's /api/chat and /api/embed endpoints."""

    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    embed_model: str = "bge-m3"
    timeout: float = 120.0
    temperature: float = 0.1
    max_tokens: int = 700
    provider: str = "ollama"

    @property
    def options(self) -> dict[str, Any]:
        # Low temperature: faithful, repeatable answers. num_predict is Ollama's max_tokens.
        return {"temperature": self.temperature, "num_predict": self.max_tokens, "num_ctx": 8192}

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            r = httpx.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama is not reachable at {self.base_url} ({e.__class__.__name__}). Is `ollama serve` running?") from e
        if r.status_code == 404:
            raise LLMError(f"Model not found in Ollama. Run: ollama pull {payload.get('model')}")
        if r.status_code >= 400:
            raise LLMError(f"Ollama error {r.status_code}: {r.text[:200]}")
        return r.json()

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        """Return the full reply. With `json_mode`, Ollama constrains output to valid JSON."""
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False, "options": self.options}
        if json_mode:
            payload["format"] = "json"
        return self._post("/api/chat", payload)["message"]["content"]

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield reply tokens as Ollama produces them (newline-delimited JSON)."""
        payload = {"model": self.model, "messages": messages, "stream": True, "options": self.options}
        try:
            with httpx.stream("POST", f"{self.base_url}/api/chat", json=payload, timeout=self.timeout) as r:
                if r.status_code >= 400:
                    raise LLMError(f"Ollama error {r.status_code}. Is the model pulled? ollama pull {self.model}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama is not reachable at {self.base_url}.") from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts with the embedding model (bge-m3 → 1024 dims)."""
        if not texts:
            return []
        return self._post("/api/embed", {"model": self.embed_model, "input": texts})["embeddings"]

    def list_models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama is not running at {self.base_url}. Start it with: ollama serve") from e
        return sorted(m.get("name", "") for m in r.json().get("models", []))

    def health(self, *, check_embed: bool = True) -> dict[str, Any]:
        """Report whether Ollama answers and the needed models are pulled."""
        try:
            names = set(self.list_models())
        except LLMError as e:
            return {"ok": False, "issues": [str(e)]}

        def pulled(model: str) -> bool:
            return model in names or f"{model}:latest" in names  # "bge-m3" is listed as "bge-m3:latest"

        wanted = [self.model, self.embed_model] if check_embed else [self.model]
        issues = [f"Model not pulled. Run: ollama pull {m}" for m in wanted if not pulled(m)]
        return {"ok": not issues, "issues": issues}


# ----------------------------------------------------------------- Groq (cloud)
class GroqClient:
    """Groq's OpenAI-style chat API through the official `groq` SDK.

    The SDK's own retries are turned off (max_retries=0) so the behaviour is
    explicit: a timeout is retried once after a short backoff, a 429 becomes
    QuotaExceeded straight away (retrying would only burn more quota).
    """

    provider = "groq"
    embed_model = ""
    HEALTH_TTL = 60.0  # seconds; health is polled, model listing is not free of rate limits

    def __init__(self, api_key: str, model: str, *, temperature: float = 0.1, max_tokens: int = 700,
                 timeout: float = 30.0, backoff: float = 1.5, client: Any = None) -> None:
        self.api_key, self.model = api_key, model
        self.temperature, self.max_tokens, self.timeout, self.backoff = temperature, max_tokens, timeout, backoff
        self._client = client
        self._health: tuple[float, dict[str, Any]] | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise LLMError("GROQ_API_KEY is not set. Add it to .env (or the Render dashboard).")
            from groq import Groq  # imported lazily: Ollama-only installs don't need it

            self._client = Groq(api_key=self.api_key, timeout=self.timeout, max_retries=0)
        return self._client

    def _call(self, fn: Callable[[], T]) -> T:
        """Run one SDK call with the error mapping and the single timeout retry."""
        import groq

        for attempt in (1, 2):
            try:
                return fn()
            except groq.RateLimitError as e:
                raise QuotaExceeded(QUOTA_MESSAGE) from e
            except groq.APITimeoutError as e:
                if attempt == 1:
                    time.sleep(self.backoff)
                    continue
                raise LLMError("Groq timed out twice. Try again in a moment.") from e
            except groq.AuthenticationError as e:
                raise LLMError("GROQ_API_KEY was rejected by Groq.") from e
            except groq.NotFoundError as e:
                raise LLMError(f"Groq model '{self.model}' is not available. Run: python -m app.list_models") from e
            except groq.APIConnectionError as e:
                raise LLMError("Can't reach the Groq API (network error).") from e
            except groq.APIError as e:
                raise LLMError(f"Groq error: {str(e)[:200]}") from e
        raise LLMError("unreachable")  # pragma: no cover

    def _params(self, messages: list[dict[str, str]], **extra: Any) -> dict[str, Any]:
        if not self.model:
            raise LLMError("GROQ_MODEL is not set. Pick one with: python -m app.list_models")
        return {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens, **extra}

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        """Full reply. JSON mode uses Groq's `response_format` (the prompt must mention JSON, ours do)."""
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        params = self._params(messages, **extra)
        resp = self._call(lambda: self.client.chat.completions.create(**params))
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield tokens. Opening the stream gets the retry; an error mid-stream ends it as LLMError."""
        import groq

        params = self._params(messages, stream=True)
        chunks = self._call(lambda: self.client.chat.completions.create(**params))
        try:
            for chunk in chunks:
                token = chunk.choices[0].delta.content if chunk.choices else None
                if token:
                    yield token
        except groq.RateLimitError as e:
            raise QuotaExceeded(QUOTA_MESSAGE) from e
        except groq.APIError as e:
            raise LLMError(f"Groq stream interrupted: {str(e)[:120]}") from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise LLMError("Groq has no embedding API; embeddings come from EMBED_PROVIDER (pinecone or ollama).")

    def list_models(self) -> list[str]:
        """Model ids this key can use. Listing models costs no tokens."""
        return sorted(m.id for m in self._call(lambda: self.client.models.list()).data)

    def health(self) -> dict[str, Any]:
        """Check the key and model with a models-list call (cached for a minute)."""
        now = time.monotonic()
        if self._health and now - self._health[0] < self.HEALTH_TTL:
            return self._health[1]
        issues: list[str] = []
        try:
            models = self.list_models()
            if not self.model:
                issues.append("GROQ_MODEL is not set. Pick one with: python -m app.list_models")
            elif self.model not in models:
                issues.append(f"GROQ_MODEL '{self.model}' is not available to this key. Run: python -m app.list_models")
        except QuotaExceeded:
            pass  # the key works; it's just busy
        except LLMError as e:
            issues.append(str(e))
        result = {"ok": not issues, "issues": issues}
        self._health = (now, result)
        return result


def make_llm(settings: Any) -> LLM:
    """Build the chat model for `settings.llm_provider`."""
    if settings.llm_provider == "groq":
        return GroqClient(settings.groq_api_key, settings.groq_model,
                          temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens)
    return OllamaClient(base_url=settings.ollama_url, model=settings.llm_model, embed_model=settings.embed_model,
                        temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens)


def parse_json(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model output, tolerating ```json fences. None if invalid."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
