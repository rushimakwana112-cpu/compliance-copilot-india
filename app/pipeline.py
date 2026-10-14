"""One chat turn, end to end. Shared by the API and the eval so both test the same code.

    message + history ─► follow-up rewrite ─► jurisdiction ─► router ─┬─► tools.py
                                                                      ├─► sql_agent.py
                                                                      └─► rag.py
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from app import logs, rag, sql_agent, tools
from app.jurisdiction import Scope, detect_states, resolve
from app.llm import LLMError
from app.router import route
from app.schemas import ChatRequest, ChatResponse, ScopeOut
from app.services import Services


@dataclass
class Turn:
    """Everything decided before answering, plus debug info for logs and eval."""

    message: str
    question: str  # standalone version of the message
    scope: Scope
    route: str
    top_scores: list[float] = field(default_factory=list)


def plan(services: Services, req: ChatRequest) -> Turn:
    """Rewrite follow-ups, resolve the jurisdiction and pick a route."""
    s = services.settings
    question = rag.rewrite_followup(services.llm, req.message, req.history, s.history_turns)
    j = req.jurisdiction
    scope = resolve(req.message, j.mode, j.state)
    if scope.how == "default" and question != req.message and detect_states(question):
        # "and the PT there?" → the rewritten question carries the state from earlier turns
        scope = resolve(question, j.mode, j.state)
    return Turn(req.message, question, scope, route(question, s.router, services.llm))


def _scope_out(scope: Scope) -> ScopeOut:
    return ScopeOut(state=scope.primary, how=scope.how, states=scope.states if len(scope.states) > 1 else None)


def _answer(services: Services, turn: Turn) -> ChatResponse:
    if turn.route == "tool":
        resp = tools.run(turn.question, turn.scope)
    elif turn.route == "sql":
        resp = sql_agent.run(services, turn.question)
    else:
        resp, passages = rag.answer(services, turn.question, turn.scope)
        turn.top_scores = [round(p.score, 3) for p in passages[:4]]
    resp.scope = resp.scope or _scope_out(turn.scope)
    return resp


def _llm_down(turn: Turn, err: Exception) -> ChatResponse:
    return ChatResponse(type="not_found", answer=f"The language model is not available: {err}",
                        scope=_scope_out(turn.scope), note="Check /api/health.")


def _log(turn: Turn, resp: ChatResponse) -> None:
    logs.append("chats", {
        "question": turn.message, "standalone": turn.question if turn.question != turn.message else None,
        "scope": turn.scope.states, "how": turn.scope.how, "route": turn.route, "type": resp.type,
        "top_scores": turn.top_scores, "latency_ms": resp.latency_ms,
    })


def handle(services: Services, req: ChatRequest, *, log: bool = True) -> tuple[ChatResponse, Turn]:
    """Answer one message (non-streaming)."""
    t0 = time.perf_counter()
    turn = plan(services, req)
    try:
        resp = _answer(services, turn)
    except LLMError as e:
        resp = _llm_down(turn, e)
    resp.latency_ms = int((time.perf_counter() - t0) * 1000)
    if log:
        _log(turn, resp)
    return resp, turn


def handle_stream(services: Services, req: ChatRequest) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield SSE events: `meta` (route + scope), `token`* (RAG only), then `done` with the full response."""
    t0 = time.perf_counter()
    try:
        turn = plan(services, req)
    except LLMError as e:
        yield "done", ChatResponse(type="not_found", answer=f"The language model is not available: {e}").model_dump(exclude_none=True)
        return
    yield "meta", {"route": turn.route, "scope": _scope_out(turn.scope).model_dump(exclude_none=True)}
    resp: ChatResponse
    try:
        if turn.route == "rag":
            resp = rag.not_found(turn.scope)
            for kind, value in rag.stream(services, turn.question, turn.scope):
                if kind == "token":
                    yield "token", {"t": value}
                else:
                    resp = value  # type: ignore[assignment]
            resp.scope = resp.scope or _scope_out(turn.scope)
            turn.top_scores = [c.score for c in resp.citations[:4] if c.score is not None]
        else:
            resp = _answer(services, turn)
    except LLMError as e:
        resp = _llm_down(turn, e)
    resp.latency_ms = int((time.perf_counter() - t0) * 1000)
    _log(turn, resp)
    yield "done", resp.model_dump(exclude_none=True)
