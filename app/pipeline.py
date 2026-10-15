"""One chat turn, end to end. Shared by the API and the eval so both test the same code.

    message + history ─► follow-up rewrite ─► jurisdiction ─► router ─┬─► tools.py
                                                                      ├─► sql_agent.py
                                                                      └─► rag.py
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from app import logs, rag, sql_agent, tools
from app.jurisdiction import Scope, detect_states, foreign_place, resolve
from app.llm import QUOTA_MESSAGE, LLMError, QuotaExceeded
from app.router import route
from app.schemas import ChatRequest, ChatResponse, ScopeOut
from app.services import Services

log = logging.getLogger("app")


@dataclass
class Turn:
    """Everything decided before answering, plus debug info for logs and eval."""

    message: str
    question: str  # standalone version of the message
    scope: Scope
    route: str
    trace: rag.Trace = field(default_factory=rag.Trace)  # retrieval details for the log


def plan(services: Services, req: ChatRequest) -> Turn:
    """Rewrite follow-ups, resolve the jurisdiction and pick a route."""
    s = services.settings
    question = rag.rewrite_followup(services.llm, req.message, req.history, s.history_turns)
    j = req.jurisdiction
    scope = resolve(req.message, j.mode, j.state)
    if scope.how == "default" and question != req.message and detect_states(question):
        # "and the PT there?" → the rewritten question carries the state from earlier turns
        scope = resolve(question, j.mode, j.state)
    return Turn(req.message, question, scope, route(question, s.router, services.llm), rag.Trace(index=s.pinecone_index))


def out_of_scope(services: Services, req: ChatRequest) -> tuple[Turn, ChatResponse] | None:
    """A question naming a foreign place is refused at once: no retrieval, no model call."""
    place = foreign_place(req.message)
    if place is None:
        return None
    j = req.jurisdiction
    scope = resolve(req.message, j.mode, j.state)
    turn = Turn(req.message, req.message, scope, "out_of_scope", rag.Trace(index=services.settings.pinecone_index))
    return turn, ChatResponse(
        type="not_found", scope=_scope_out(scope), note=f"Out of scope: the question names {place}.",
        answer=f"I only cover Indian law: central labour and payroll law plus Indian state rules. "
               f"I can't answer questions about {place}.",
    )


def _scope_out(scope: Scope) -> ScopeOut:
    return ScopeOut(state=scope.primary, how=scope.how, states=scope.states if len(scope.states) > 1 else None)


def _answer(services: Services, turn: Turn) -> ChatResponse:
    if turn.route == "tool":
        resp = tools.run(turn.question, turn.scope)
    elif turn.route == "sql":
        resp = sql_agent.run(services, turn.question)
    else:
        resp, _ = rag.answer(services, turn.question, turn.scope, turn.trace)
    resp.scope = resp.scope or _scope_out(turn.scope)
    return resp


def _llm_down(turn: Turn, err: Exception) -> ChatResponse:
    if isinstance(err, QuotaExceeded):
        return ChatResponse(type="not_found", answer=QUOTA_MESSAGE, scope=_scope_out(turn.scope), quota_exceeded=True)
    return ChatResponse(type="not_found", answer=f"The language model is not available: {err}",
                        scope=_scope_out(turn.scope), note="Check /api/health.")


def _search_failed(turn: Turn, err: rag.SearchError) -> ChatResponse:
    log.warning("Search failed for %r on index %s: %s", turn.question, turn.trace.index, err)
    return ChatResponse(type="error", answer=f"Search failed: {err}", scope=_scope_out(turn.scope))


def _log(turn: Turn, resp: ChatResponse) -> None:
    """One line per chat, for every route and outcome. Retrieval fields are empty when no search ran."""
    logs.append("chats", {
        "question": turn.message, "standalone": turn.question if turn.question != turn.message else None,
        "scope": turn.scope.states, "how": turn.scope.how, "route": turn.route, "type": resp.type,
        **turn.trace.log_fields(), "latency_ms": resp.latency_ms, "cached": resp.cached,
    })


def _cached(services: Services, turn: Turn, t0: float) -> ChatResponse | None:
    hit = services.cache.get(turn.question, turn.scope.states)
    if hit is not None:
        hit.latency_ms = int((time.perf_counter() - t0) * 1000)
    return hit


def handle(services: Services, req: ChatRequest, *, log: bool = True, use_cache: bool = True) -> tuple[ChatResponse, Turn]:
    """Answer one message (non-streaming). Repeated questions come from the answer cache."""
    t0 = time.perf_counter()
    if (guarded := out_of_scope(services, req)) is not None:
        turn, resp = guarded
        resp.latency_ms = int((time.perf_counter() - t0) * 1000)
        if log:
            _log(turn, resp)
        return resp, turn
    turn = plan(services, req)
    hit = _cached(services, turn, t0) if use_cache else None
    if hit is not None:
        if log:
            _log(turn, hit)
        return hit, turn
    try:
        resp = _answer(services, turn)
    except rag.SearchError as e:
        resp = _search_failed(turn, e)
    except LLMError as e:
        resp = _llm_down(turn, e)
    resp.latency_ms = int((time.perf_counter() - t0) * 1000)
    if use_cache:
        services.cache.put(turn.question, turn.scope.states, resp)
    if log:
        _log(turn, resp)
    return resp, turn


def handle_stream(services: Services, req: ChatRequest) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield SSE events: `meta` (route + scope), `token`* (RAG only), then `done` with the full response."""
    t0 = time.perf_counter()
    if (guarded := out_of_scope(services, req)) is not None:
        turn, resp = guarded
        resp.latency_ms = int((time.perf_counter() - t0) * 1000)
        _log(turn, resp)
        yield "meta", {"route": "not_found", "scope": resp.scope.model_dump(exclude_none=True) if resp.scope else {}}
        yield "done", resp.model_dump(exclude_none=True)
        return
    try:
        turn = plan(services, req)
    except LLMError as e:  # plan() already falls back on errors; this is a last resort
        quota = isinstance(e, QuotaExceeded)
        yield "done", ChatResponse(type="not_found", answer=QUOTA_MESSAGE if quota else f"The language model is not available: {e}",
                                   quota_exceeded=quota or None).model_dump(exclude_none=True)
        return
    yield "meta", {"route": turn.route, "scope": _scope_out(turn.scope).model_dump(exclude_none=True)}
    hit = _cached(services, turn, t0)
    if hit is not None:  # no tokens to stream: the whole answer arrives with `done`
        _log(turn, hit)
        yield "done", hit.model_dump(exclude_none=True)
        return
    resp: ChatResponse
    try:
        if turn.route == "rag":
            resp = rag.not_found(turn.scope)
            for kind, value in rag.stream(services, turn.question, turn.scope, turn.trace):
                if kind == "token":
                    yield "token", {"t": value}
                else:
                    resp = value  # type: ignore[assignment]
            resp.scope = resp.scope or _scope_out(turn.scope)
        else:
            resp = _answer(services, turn)
    except rag.SearchError as e:
        resp = _search_failed(turn, e)
    except LLMError as e:
        resp = _llm_down(turn, e)
    resp.latency_ms = int((time.perf_counter() - t0) * 1000)
    services.cache.put(turn.question, turn.scope.states, resp)
    _log(turn, resp)
    yield "done", resp.model_dump(exclude_none=True)
