"""Language-model providers behind one small interface.

    LLM_PROVIDER=groq    Groq's hosted API (free tier, fast). Needs GROQ_API_KEY and GROQ_MODEL.
    LLM_PROVIDER=ollama  A local Ollama server. Needs `ollama serve` and the model pulled.

Both get the same prompts, temperature (0.1) and max_tokens (700), so the
NOT_FOUND contract and the router's JSON output behave the same. The rest of
the app only sees `chat`, `stream`, `embed` and `health`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import httpx

T = TypeVar("T")
log = logging.getLogger("app")
QUOTA_MESSAGE = "Daily demo limit reached. Please try again later."
BUSY_MESSAGE = "Busy right now, please retry in a minute."
MAX_RATE_WAIT = 20.0  # seconds: the longest we wait out a per-minute limit before retrying once


class LLMError(RuntimeError):
    """Raised when the model provider is unreachable, misconfigured or returns an error."""


class RateLimited(LLMError):
    """The provider answered 429. `window` is "minute" (TPM/RPM) or "day" (TPD/RPD)."""

    window = "minute"

    def __init__(self, message: str, *, limit: str = "", retry_after: float | None = None,
                 prompt_tokens: int | None = None) -> None:
        super().__init__(message)
        self.limit, self.retry_after, self.prompt_tokens = limit, retry_after, prompt_tokens


class Busy(RateLimited):
    """A per-minute limit (tokens or requests per minute) that a retry after waiting didn't clear."""


class QuotaExceeded(RateLimited):
    """A per-day limit: the free tier's daily tokens or requests are used up."""

    window = "day"


def estimate_tokens(messages: list[dict[str, str]] | str) -> int:
    """Rough prompt size: ~4 characters per token for English text. Good enough for budgets and logs."""
    text = messages if isinstance(messages, str) else "".join(m.get("content", "") for m in messages)
    return (len(text) + 3) // 4


LIMIT_NAME = re.compile(r"\((TPM|RPM|TPD|RPD|ASH|ASD)\)|\b(tokens|requests) per (minute|day)\b", re.IGNORECASE)
TRY_AGAIN = re.compile(r"try again in\s+(?:(\d+)m)?\s*(\d+(?:\.\d+)?)\s*(ms|s)?", re.IGNORECASE)


def classify_rate_limit(text: str, headers: Any = None) -> tuple[str, str, float | None]:
    """(window, limit, retry_after_seconds) from a 429's error text and headers.

    Groq's message names the limit: "Rate limit reached … on tokens per minute (TPM): Limit 8000 …
    Please try again in 7.5s". The retry-after header is used when the text has no wait time.
    Unknown limits count as per-minute when the wait is a minute or less, else per-day.
    """
    headers = headers or {}
    m = LIMIT_NAME.search(text)
    limit = ""
    if m and m.group(1):
        limit = m.group(1).upper()
    elif m:
        limit = ("T" if m.group(2).lower() == "tokens" else "R") + "P" + m.group(3)[0].upper()
    retry_after: float | None = None
    t = TRY_AGAIN.search(text)
    if t:
        secs = float(t.group(2)) / (1000 if (t.group(3) or "s").lower() == "ms" else 1)
        retry_after = int(t.group(1) or 0) * 60 + secs
    else:
        try:
            retry_after = float(headers.get("retry-after"))
        except (TypeError, ValueError):
            retry_after = None
    if limit.endswith("D"):
        window = "day"
    elif limit.endswith("M"):
        window = "minute"
    else:
        window = "minute" if retry_after is not None and retry_after <= 60 else "day"
    return window, limit, retry_after


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
    explicit: a timeout is retried once after a short backoff. A 429 is read to
    see which limit was hit: a per-minute limit (TPM/RPM) is waited out (at most
    20 s) and retried once, then becomes Busy; a per-day limit (TPD/RPD) becomes
    QuotaExceeded straight away, since retrying would only burn more quota.
    """

    provider = "groq"
    embed_model = ""
    HEALTH_TTL = 60.0  # seconds; health is polled, model listing is not free of rate limits
    HEALTH_TIMEOUT = 5.0  # seconds; a health check must answer fast or report the problem

    def __init__(self, api_key: str, model: str, *, temperature: float = 0.1, max_tokens: int = 700,
                 timeout: float = 30.0, backoff: float = 1.5, client: Any = None,
                 max_rate_wait: float = MAX_RATE_WAIT, sleep: Callable[[float], None] = time.sleep) -> None:
        self.api_key, self.model = api_key, model
        self.temperature, self.max_tokens, self.timeout, self.backoff = temperature, max_tokens, timeout, backoff
        self.max_rate_wait, self.sleep = max_rate_wait, sleep
        self.json_mode_ok = True  # turned off after the model fails JSON mode once
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

    def _rate_limit(self, e: Any, prompt_tokens: int | None) -> RateLimited:
        """Turn a groq.RateLimitError into Busy or QuotaExceeded, and log which limit was hit."""
        body = getattr(e, "body", None)
        text = f"{e} {json.dumps(body) if isinstance(body, dict | list) else body or ''}"
        response = getattr(e, "response", None)
        window, limit, retry_after = classify_rate_limit(text, getattr(response, "headers", None))
        log.warning("Groq rate limit: window=%s limit=%s retry_after=%s prompt_tokens=%s model=%s",
                    window, limit or "unknown", retry_after, prompt_tokens, self.model)
        kind = QuotaExceeded if window == "day" else Busy
        return kind(QUOTA_MESSAGE if window == "day" else BUSY_MESSAGE, limit=limit, retry_after=retry_after,
                    prompt_tokens=prompt_tokens)

    def _call(self, fn: Callable[[], T], prompt_tokens: int | None = None,
              passthrough: tuple[type[Exception], ...] = (), retry: bool = True) -> T:
        """Run one SDK call with the error mapping, the single timeout retry and the single rate-limit retry.
        Exceptions in `passthrough` reach the caller unchanged. `retry=False` skips both waits (health checks)."""
        import groq

        timeouts = waits = 0 if retry else 1
        while True:
            try:
                return fn()
            except passthrough:
                raise
            except groq.RateLimitError as e:
                err = self._rate_limit(e, prompt_tokens)
                if isinstance(err, Busy) and waits == 0:
                    waits += 1
                    self.sleep(min(err.retry_after if err.retry_after is not None else self.max_rate_wait,
                                   self.max_rate_wait))
                    continue
                raise err from e
            except groq.APITimeoutError as e:
                timeouts += 1
                if timeouts == 1:
                    self.sleep(self.backoff)
                    continue
                raise LLMError("Groq timed out twice. Try again in a moment." if retry else "Groq timed out.") from e
            except groq.AuthenticationError as e:
                raise LLMError("GROQ_API_KEY was rejected by Groq.") from e
            except groq.NotFoundError as e:
                raise LLMError(f"Groq model '{self.model}' is not available. Run: python -m app.list_models") from e
            except groq.APIConnectionError as e:
                raise LLMError("Can't reach the Groq API (network error).") from e
            except groq.APIError as e:
                raise LLMError(f"Groq error: {str(e)[:200]}") from e

    def _params(self, messages: list[dict[str, str]], **extra: Any) -> dict[str, Any]:
        if not self.model:
            raise LLMError("GROQ_MODEL is not set. Pick one with: python -m app.list_models")
        return {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens, **extra}

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        """Full reply. JSON mode uses Groq's `response_format` (the prompt must mention JSON, ours do).

        Some models answer JSON mode with 400 "Failed to generate JSON" (json_validate_failed). Then the
        call is repeated as plain text, the caller parses the text, and JSON mode stays off for this client.
        """
        import groq

        if json_mode and self.json_mode_ok:
            params = self._params(messages, response_format={"type": "json_object"})
            try:
                resp = self._call(lambda: self.client.chat.completions.create(**params), estimate_tokens(messages),
                                  passthrough=(groq.BadRequestError,))
                return resp.choices[0].message.content or ""
            except groq.BadRequestError as e:
                if "json" not in str(e).lower():
                    raise LLMError(f"Groq error: {str(e)[:200]}") from e
                log.warning("Groq model %s failed JSON mode (%s); using plain text from now on", self.model,
                            str(e)[:120])
                self.json_mode_ok = False
        params = self._params(messages)
        resp = self._call(lambda: self.client.chat.completions.create(**params), estimate_tokens(messages))
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield tokens. Opening the stream gets the retry; an error mid-stream ends it as LLMError."""
        import groq

        params = self._params(messages, stream=True)
        chunks = self._call(lambda: self.client.chat.completions.create(**params), estimate_tokens(messages))
        try:
            for chunk in chunks:
                token = chunk.choices[0].delta.content if chunk.choices else None
                if token:
                    yield token
        except groq.RateLimitError as e:
            raise self._rate_limit(e, estimate_tokens(messages)) from e
        except groq.APIError as e:
            raise LLMError(f"Groq stream interrupted: {str(e)[:120]}") from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise LLMError("Groq has no embedding API; embeddings come from EMBED_PROVIDER (pinecone or ollama).")

    def list_models(self, *, quick: bool = False) -> list[str]:
        """Model ids this key can use. Listing models costs no tokens.
        `quick`: a short timeout and no retries or rate-limit waits, for health checks."""
        client = self.client
        if quick and hasattr(client, "with_options"):
            client = client.with_options(timeout=self.HEALTH_TIMEOUT)
        return sorted(m.id for m in self._call(lambda: client.models.list(), retry=not quick).data)

    def health(self) -> dict[str, Any]:
        """Check the key and model with a quick models-list call (cached for a minute).
        Never waits or retries: Render's health check and the page's status pill are waiting on it."""
        now = time.monotonic()
        if self._health and now - self._health[0] < self.HEALTH_TTL:
            return self._health[1]
        issues: list[str] = []
        try:
            models = self.list_models(quick=True)
            if not self.model:
                issues.append("GROQ_MODEL is not set. Pick one with: python -m app.list_models")
            elif self.model not in models:
                issues.append(f"GROQ_MODEL '{self.model}' is not available to this key. Run: python -m app.list_models")
        except RateLimited:
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
