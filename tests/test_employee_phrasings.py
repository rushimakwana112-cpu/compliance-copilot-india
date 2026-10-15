"""How real employees and testers type questions: no "calculate", typos, lower-case names, the model's
refusal at the end of an answer. Each case here was answered wrongly before (see docs/build-log.md)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import rag
from app.jurisdiction import resolve
from app.main import create_app
from app.pipeline import handle
from app.router import route_rules
from app.schemas import ChatRequest


def ask(services, message: str):
    resp, turn = handle(services, ChatRequest(message=message), log=False, use_cache=False)
    return resp, turn


@pytest.mark.parametrize("message, starts", [
    ("what i my pf if my basic salary is 24000", "PF for a monthly basic wage"),  # the live-site screenshot
    ("what is my pf if my basic salary is 24000", "PF for a monthly basic wage"),  # used to get the "Basic" FAQ
    ("my basic is 24000, what will be my pf", "PF for a monthly basic wage"),
    ("pf for 24000 basic", "PF for a monthly basic wage"),
    ("PF deduction on basic 30000", "PF for a monthly basic wage"),
    ("what is employer pf on 24000 basic", "PF for a monthly basic wage"),
    ("eps on 24000 basic", "PF for a monthly basic wage"),
    ("whats my esic if gross is 18000", "ESIC"),
    ("my gross is 19000 esic deduction?", "ESIC"),
    ("pt on 15000 salary", "Professional Tax"),
    ("gratuity for 6 years with basic 40000", "Estimated gratuity"),
])
def test_amount_and_topic_go_to_the_calculator(services, store, llm, message, starts):
    resp, turn = ask(services, message)
    assert turn.route == "tool" and resp.type == "tool" and resp.answer.startswith(starts)
    assert store.searches == [] and llm.calls == []  # worked out in code, no search, no model


@pytest.mark.parametrize("message", [
    "Is ESIC applicable on ₹24,000 gross?",
    "Does PF apply if basic is above 30000?",
    "Who is eligible for ESIC if wages are 20000?",
    "Is PT mandatory for salary of 12000 in Gujarat?",
])
def test_law_questions_with_an_amount_still_search_the_law(message):
    assert route_rules(message) == "rag"


def test_is_this_correct_with_is_before_the_figure(services):
    resp, _ = ask(services, "is this correct pf on basic 18000 is 2160")
    assert resp.type == "tool" and ("matches my calculation" in resp.answer or "doesn't match" in resp.answer)


def test_salary_for_some_days(services):
    asked, _ = ask(services, "salary for 20 days if monthly salary 31000")
    assert asked.type == "guide" and "26 days" in asked.answer  # the divisor is company policy: ask, don't assume
    resp, _ = ask(services, "salary for 20 days if monthly salary 31000, 30 days basis")
    assert resp.type == "tool" and "Salary for 20 paid days" in resp.answer and "× 20" in resp.answer
    assert "₹31,000 ÷ 30 days" in resp.table.rows[0][1]


def test_lower_case_name_is_another_employee(services, store, llm):
    resp, _ = ask(services, "show salary of ramesh")
    assert resp.type == "guide" and "don't have any employee data" in resp.answer
    assert store.searches == [] and llm.calls == []


# ----------------------------------------------------------------- the model's refusal inside an answer
def _passages(services):
    services.store.add("provident fund schemes funds", doc="Code on Social Security, 2020", page=29,
                       scope="central", status="current", doc_type="act")
    scope = resolve("pf", "auto", "gujarat")
    passages, _ = rag.prepare(services, "provident fund schemes funds", scope)
    return passages, scope


def test_not_found_after_an_explanation_is_a_refusal(services):
    """Screenshot: "The passages do not answer the question.\\n\\nNOT_FOUND" was shown with 5 'cited' passages."""
    passages, scope = _passages(services)
    resp = rag.finish("The passages do not answer the question.\n\nNOT_FOUND", passages, scope, "q")
    assert resp.type == "not_found" and "NOT_FOUND" not in resp.answer and resp.citations == []


def test_cited_answer_drops_a_stray_not_found_line(services):
    passages, scope = _passages(services)
    resp = rag.finish("Funds are set up under the scheme [1].\n\nNOT_FOUND", passages, scope, "q")
    assert resp.type == "rag" and "NOT_FOUND" not in resp.answer and len(resp.citations) == 1


def test_answer_without_citation_markers_lists_no_cited_sources(services):
    passages, scope = _passages(services)
    resp = rag.finish("Funds are set up under the scheme.", passages, scope, "q")
    assert resp.citations == [] and len(resp.other_citations) == len(passages)
    assert "doesn't cite a passage" in resp.note


def test_stream_hides_a_late_not_found(services, store, llm):
    store.add("provident fund schemes funds officers central board", doc="Code on Social Security, 2020", page=29,
              scope="central", status="current", doc_type="act")
    llm.responder = lambda messages, json_mode: "The passages do not answer the question.\n\nNOT_FOUND"
    with TestClient(create_app(services)) as c:
        text = c.post("/api/chat/stream", json={"message": "provident fund schemes funds officers"}).text
    done = text.rsplit("event: done", 1)[1]
    assert '"type": "not_found"' in done and "NOT_FOUND" not in done
