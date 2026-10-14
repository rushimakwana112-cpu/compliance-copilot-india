"""Pydantic models for the HTTP API.

These mirror the contract the frontend already expects (see README, "API").
Optional fields are left out of the JSON when empty so responses stay small.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl

Route = Literal["rag", "tool", "sql", "not_found"]


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = ""


class JurisdictionIn(BaseModel):
    central: bool = True
    state: str = "gujarat"
    mode: Literal["auto", "manual"] = "auto"
    detected: bool = False


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list)
    jurisdiction: JurisdictionIn = Field(default_factory=JurisdictionIn)


class ScopeOut(BaseModel):
    state: str
    how: Literal["detected", "default", "manual"]
    # Extra, optional: every state searched (more than one for comparisons).
    states: list[str] | None = None


class Citation(BaseModel):
    doc: str
    section: str = ""
    page: str = ""
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


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model: str
    pinecone: bool
    docs: int
    ollama: bool = False
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
