"""Pydantic models for the HTTP API.

These mirror the contract the frontend already expects (see README, "API").
Optional fields are left out of the JSON when empty so responses stay small.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

Route = Literal["rag", "tool", "sql", "not_found", "error"]  # error: the search itself failed
MAX_MESSAGE_CHARS = 1000
MAX_HISTORY_MESSAGES = 8  # 4 turns of user + assistant


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = ""


class JurisdictionIn(BaseModel):
    central: bool = True
    state: str = "gujarat"
    mode: Literal["auto", "manual"] = "auto"
    detected: bool = False


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    history: list[Turn] = Field(default_factory=list)
    jurisdiction: JurisdictionIn = Field(default_factory=JurisdictionIn)

    @field_validator("history")
    @classmethod
    def last_turns(cls, history: list[Turn]) -> list[Turn]:
        """Only the last 4 turns are used, and each is capped, so a long chat can't inflate the prompt."""
        return [Turn(role=t.role, content=t.content[:4000]) for t in history[-MAX_HISTORY_MESSAGES:]]


class ScopeOut(BaseModel):
    state: str
    how: Literal["detected", "default", "manual"]
    # Extra, optional: every state searched (more than one for comparisons).
    states: list[str] | None = None


class Citation(BaseModel):
    doc: str
    section: str = ""
    page: int | None = None  # a whole page number; None when the source has no page (calculators)
    # None for calculator citations, which are not similarity matches.
    score: float | None = None
    snippet: str = ""
    url: str = ""
    status: str | None = None


class Table(BaseModel):
    columns: list[str]
    rows: list[list[str]]
    totalRow: bool = False


class ChatResponse(BaseModel):
    type: Route
    answer: str
    scope: ScopeOut | None = None
    citations: list[Citation] = Field(default_factory=list)
    table: Table | None = None
    tool: str | None = None
    sql: str | None = None
    rows_scanned: int | None = None
    note: str | None = None
    latency_ms: int = 0
    # Set only when true: the provider's free quota is used up / the answer came from the cache.
    quota_exceeded: bool | None = None
    cached: bool | None = None


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model: str
    pinecone: bool
    docs: int
    provider: str = "groq"
    embed_provider: str = "pinecone"
    llm_ok: bool = False
    public_mode: bool = False
    issues: list[str] = Field(default_factory=list)


class DocumentOut(BaseModel):
    name: str
    scope: str
    status: str
    meta: str


class StateOut(BaseModel):
    id: str
    name: str
    aliases: list[str]


class IngestUrl(BaseModel):
    url: HttpUrl
    scope: str = "central"


class IngestAccepted(BaseModel):
    accepted: bool = True
    name: str
    scope: str


class Feedback(BaseModel):
    value: Literal["up", "down"]
    question: str = ""
    # The frontend sends the whole answer object; plain text is accepted too.
    answer: Any = None
