"""Thin client for a local Ollama server: chat, streaming chat, JSON mode, embeddings.

Ollama exposes a small REST API on http://localhost:11434. We call it with
httpx instead of a heavier SDK so every request is visible and easy to mock.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


class LLMError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


class LLM(Protocol):
    """What the rest of the app needs from a language model (real or fake)."""

    model: str

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str: ...
    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def health(self) -> dict[str, Any]: ...


@dataclass
class OllamaClient:
    """Calls Ollama's /api/chat and /api/embed endpoints."""

    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    embed_model: str = "bge-m3"
    timeout: float = 120.0
    # Low temperature: we want faithful, repeatable answers, not creativity.
    options: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.1, "num_ctx": 8192})

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

    def health(self) -> dict[str, Any]:
        """Report whether Ollama answers and both models are pulled."""
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
        except httpx.HTTPError:
            return {"ok": False, "issues": [f"Ollama is not running at {self.base_url}. Start it with: ollama serve"]}
        names = {m.get("name", "") for m in r.json().get("models", [])}

        def pulled(model: str) -> bool:
            # "bge-m3" is listed as "bge-m3:latest"
            return model in names or f"{model}:latest" in names

        issues = [f"Model not pulled. Run: ollama pull {m}" for m in (self.model, self.embed_model) if not pulled(m)]
        return {"ok": not issues, "issues": issues}


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
