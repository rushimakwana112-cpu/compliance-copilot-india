"""One chat turn, end to end. Shared by the API and the eval so both test the same code.

    message ─► language + short forms ─► guards ─► salary calculators ─► follow-up rewrite ─► router ─┬─► tools.py
                (lang.py)                (guards.py)  (salary.py)                                     ├─► sql_agent.py
                                                                                                      └─► rag.py

Guards and salary calculations answer without the router: they need no search and are never cached
(a CTC conversation depends on the earlier turns). Answers to Hindi, Gujarati or Hinglish messages
are written in that language.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from datetime import date

from app import guards, lang, logs, rag, salary, sql_agent, tools
from app.jurisdiction import Scope, detect_states, foreign_place, resolve
from app.llm import LLMError, RateLimited
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
    lang: str = "en"  # the language to answer in

    @property
    def cache_key(self) -> str:
        return self.question if self.lang == "en" else f"[{self.lang}] {self.question}"


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


def understand(services: Services, req: ChatRequest) -> tuple[ChatRequest, str]:
    """The message in plain English (short forms expanded, Hindi/Gujarati translated) and the language to answer in."""
    which = lang.detect(req.message)
    text = lang.to_english(services.llm, req.message, which)
    return (req.model_copy(update={"message": text}) if text != req.message else req), which


def early(services: Services, req: ChatRequest, which: str) -> tuple[Turn, ChatResponse] | None:
    """Guard answers (no employee data, injection, vague, invalid figures) and salary calculations."""
    j = req.jurisdiction
    scope = resolve(req.message, j.mode, j.state)
    trace = rag.Trace(index=services.settings.pinecone_index)
    if (resp := guards.check(req.message)) is not None:
        return Turn(req.message, req.message, scope, "guide", trace, which), resp
    if salary.wants(req.message, req.history):
        resp = salary.run(req.message, req.history, scope.primary, None, date.today())
        if resp is not None:
            resp.scope = _scope_out(scope)
            return Turn(req.message, req.message, scope, "tool", trace, which), resp
    return None


def finish(services: Services, turn: Turn, resp: ChatResponse) -> ChatResponse:
    """Add the advisor pointer and write fixed answers in the user's language (RAG answers already are)."""
    note = guards.advice_note(turn.message)
    if note and note not in resp.answer and resp.type in ("rag", "tool", "guide"):
        resp.answer = f"{resp.answer}\n\n{note}"
    if turn.lang != "en" and turn.route != "rag" and not resp.rate_limited:
        resp.answer = lang.translate(services.llm, resp.answer, turn.lang)  # type: ignore[arg-type]
    return resp


def _scope_out(scope: Scope) -> ScopeOut:
    return ScopeOut(state=scope.primary, how=scope.how, states=scope.states if len(scope.states) > 1 else None)


def _answer(services: Services, turn: Turn) -> ChatResponse:
    if turn.route == "tool":
        resp = tools.run(turn.question, turn.scope)
    elif turn.route == "sql":
        resp = sql_agent.run(services, turn.question) if services.settings.sample_data else guards.no_data()
    else:
        resp, _ = rag.answer(services, turn.question, turn.scope, turn.trace, lang=turn.lang)
    resp.scope = resp.scope or _scope_out(turn.scope)
    return resp


def _llm_down(turn: Turn | None, err: Exception) -> ChatResponse:
    scope = _scope_out(turn.scope) if turn else None
    if isinstance(err, RateLimited):
        if turn:
            turn.trace.rate_limit = {"window": err.window, "limit": err.limit or None, "retry_after": err.retry_after,
                                     "prompt_tokens": err.prompt_tokens}
        return ChatResponse(type="not_found", answer=str(err), scope=scope, rate_limited=err.window,  # type: ignore[arg-type]
                            quota_exceeded=True if err.window == "day" else None)
    return ChatResponse(type="not_found", answer=f"The language model is not available: {err}",
                        scope=scope, note="Check /api/health.")


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
    hit = services.cache.get(turn.cache_key, turn.scope.states)
    if hit is not None:
        hit.latency_ms = int((time.perf_counter() - t0) * 1000)
    return hit


def handle(services: Services, req: ChatRequest, *, log: bool = True, use_cache: bool = True) -> tuple[ChatResponse, Turn]:
    """Answer one message (non-streaming). Repeated questions come from the answer cache."""
    t0 = time.perf_counter()
    req, which = understand(services, req)
    if (guarded := out_of_scope(services, req) or early(services, req, which)) is not None:
        turn, resp = guarded
        turn.lang = which
        resp = finish(services, turn, resp)
        resp.latency_ms = int((time.perf_counter() - t0) * 1000)
        if log:
            _log(turn, resp)
        return resp, turn
    turn = plan(services, req)
    turn.lang = which
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
    resp = finish(services, turn, resp)
    resp.latency_ms = int((time.perf_counter() - t0) * 1000)
    if use_cache:
        services.cache.put(turn.cache_key, turn.scope.states, resp)
    if log:
        _log(turn, resp)
    return resp, turn


def handle_stream(services: Services, req: ChatRequest) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield SSE events: `meta` (route + scope), `token`* (RAG only), then `done` with the full response."""
    t0 = time.perf_counter()
    req, which = understand(services, req)
    if (guarded := out_of_scope(services, req) or early(services, req, which)) is not None:
        turn, resp = guarded
        turn.lang = which
        resp = finish(services, turn, resp)
        resp.latency_ms = int((time.perf_counter() - t0) * 1000)
        _log(turn, resp)
        yield "meta", {"route": "not_found" if turn.route == "out_of_scope" else turn.route, "scope": resp.scope.model_dump(exclude_none=True) if resp.scope else {}}
        yield "done", resp.model_dump(exclude_none=True)
        return
    try:
        turn = plan(services, req)
    except LLMError as e:  # plan() already falls back on errors; this is a last resort
        yield "done", _llm_down(None, e).model_dump(exclude_none=True)
        return
    turn.lang = which
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
            for kind, value in rag.stream(services, turn.question, turn.scope, turn.trace, lang=turn.lang):
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
    resp = finish(services, turn, resp)
    resp.latency_ms = int((time.perf_counter() - t0) * 1000)
    services.cache.put(turn.cache_key, turn.scope.states, resp)
    _log(turn, resp)
    yield "done", resp.model_dump(exclude_none=True)
