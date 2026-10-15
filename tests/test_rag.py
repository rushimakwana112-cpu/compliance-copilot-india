"""RAG: jurisdiction filter, threshold, citation cleanup, follow-ups, comparison, streaming."""

from __future__ import annotations

import re

import pytest

from app import rag
from app.jurisdiction import Scope
from app.schemas import Turn

GRATUITY = "Gratuity is payable to a fixed term employee without five years of continuous service"
GUJ_PT = "Gujarat professional tax schedule monthly salary slabs and tax amounts"
MAH_PT = "Maharashtra professional tax schedule monthly salary slabs and tax amounts"


def seed(store):
    store.add(GRATUITY, doc="Code on Social Security, 2020", section="Section 53", page=40, scope="central",
              status="current", doc_type="act", effective_date="2025-11-21", source_url="https://x.gov.in/ss.pdf")
    store.add("Gratuity old act five years continuous service fixed term employee", doc="Payment of Gratuity Act, 1972",
              section="Section 4", page=3, scope="central", status="repealed", doc_type="act")
    store.add(GUJ_PT, doc="Gujarat PT schedule", page=1, scope="gujarat", status="current", doc_type="notification")
    store.add(MAH_PT, doc="Maharashtra PT schedule", page=2, scope="maharashtra", status="current", doc_type="notification")


def test_retrieval_respects_jurisdiction_and_status(services, store):
    seed(store)
    passages = rag.retrieve(services, "professional tax slabs", Scope(["gujarat"], "default"))
    docs = {p.doc for p in passages}
    assert "Gujarat PT schedule" in docs
    assert "Maharashtra PT schedule" not in docs
    assert "Payment of Gratuity Act, 1972" not in docs  # repealed excluded by default


def test_repealed_included_only_when_asked(services, store):
    seed(store)
    passages = rag.retrieve(services, "gratuity under the old Gratuity Act", Scope(["gujarat"], "default"))
    assert "Payment of Gratuity Act, 1972" in {p.doc for p in passages}


def test_below_threshold_skips_llm(services, store, llm):
    seed(store)
    services.settings.score_threshold = 0.99
    resp, _ = rag.answer(services, "gratuity for fixed term employees", Scope(["gujarat"], "default"))
    assert resp.type == "not_found" and llm.calls == []


def test_answer_cites_and_strips_bad_numbers(services, store, llm):
    seed(store)
    llm.responder = lambda m, j: "Yes. Fixed-term employees qualify without five years [1]. Also see [9]."
    resp, passages = rag.answer(services, "gratuity for fixed term employee", Scope(["gujarat"], "default"))
    assert resp.type == "rag"
    assert "[1]" in resp.answer and "[9]" not in resp.answer
    assert resp.citations[0].doc == passages[0].doc and resp.citations[0].url.startswith("https://")
    prompt = llm.calls[0][1]["content"]
    assert prompt.index("[1] ") < prompt.index("Question:")


def test_model_not_found_maps_to_not_found(services, store, llm):
    seed(store)
    llm.responder = lambda m, j: "NOT_FOUND"
    resp, _ = rag.answer(services, "gratuity for fixed term employee", Scope(["gujarat"], "default"))
    assert resp.type == "not_found"


def test_repealed_citation_adds_note(services, store, llm):
    seed(store)
    llm.responder = lambda m, j: "Under the old Act, five years were needed [1][2]."
    resp, _ = rag.answer(services, "gratuity five years under the old Gratuity Act", Scope(["gujarat"], "default"))
    assert resp.note and "Payment of Gratuity Act, 1972" in resp.note


def test_clean_citations():
    assert rag.clean_citations("A [1, 5] b [0] c [2].", 3) == "A [1] b c [2]."


def test_followup_rewrite(llm):
    llm.responder = lambda m, j: "What is the professional tax for 15000 salary in Gujarat?"
    history = [Turn(role="user", content="PT for 15000 salary in Pune"), Turn(role="assistant", content="...")]
    assert "Gujarat" in rag.rewrite_followup(llm, "and for Gujarat?", history)
    assert rag.rewrite_followup(llm, "and for Gujarat?", []) == "and for Gujarat?"  # no history, no rewrite


HISTORY = [Turn(role="user", content="Who is covered for gratuity under the Code on Social Security?"),
           Turn(role="assistant", content="Employees with five years of continuous service [1].")]


@pytest.mark.parametrize("message", [
    "and for Gujarat?", "What about Kerala?", "Same for fixed-term staff?", "Is it taxable?",
    "does that apply there?", "in Pune?",
])
def test_clear_followups_are_rewritten(llm, message):
    llm.responder = lambda m, j: "Is gratuity under the Code on Social Security taxable in Gujarat?"
    assert rag.needs_rewrite(message, HISTORY)
    assert rag.rewrite_followup(llm, message, HISTORY) != message and len(llm.calls) == 1


@pytest.mark.parametrize("message", [
    "Gratuity rule in Dubai?", "What is the PF rate?", "Is ESIC applicable on 24000 gross?",
    "Bonus for contract workers in Surat", "Is this gratuity rule valid for contract staff?",
])
def test_standalone_questions_are_used_as_is(llm, message):
    llm.responder = lambda m, j: "Who is covered for gratuity under the Code on Social Security in Dubai?"
    assert not rag.needs_rewrite(message, HISTORY)
    assert rag.rewrite_followup(llm, message, HISTORY) == message and llm.calls == []


def test_comparison_retrieves_each_state_and_asks_for_table(services, store, llm):
    seed(store)
    llm.responder = lambda m, j: "| State | PT |\n|---|---|\n| Gujarat | slab [1] |\n| Maharashtra | slab [2] |"
    scope = Scope(["gujarat", "maharashtra"], "detected")
    resp, passages = rag.answer(services, "compare professional tax slabs", scope)
    assert {"Gujarat PT schedule", "Maharashtra PT schedule"} <= {p.doc for p in passages}
    assert "markdown table" in llm.calls[0][0]["content"]
    assert resp.scope.states == ["gujarat", "maharashtra"]


def test_stream_holds_back_not_found(services, store, llm):
    seed(store)
    llm.responder = lambda m, j: "NOT_FOUND"
    events = list(rag.stream(services, "gratuity fixed term employee", Scope(["gujarat"], "default")))
    assert [k for k, _ in events] == ["done"] and events[0][1].type == "not_found"


def test_stream_tokens_join_to_answer(services, store, llm):
    seed(store)
    llm.responder = lambda m, j: "Yes. Fixed-term employees qualify [1]."
    events = list(rag.stream(services, "gratuity fixed term employee", Scope(["gujarat"], "default")))
    tokens = "".join(v for k, v in events if k == "token")
    assert tokens == "Yes. Fixed-term employees qualify [1]." and events[-1][1].type == "rag"


# ----------------------------------------------------------------- neighbours, boosts, pages
def seed_sections(store):
    texts = ["Nomination forms are kept by the employer for the record.",
             "The employer shall pay gratuity within thirty days of it becoming payable.",
             "Gratuity is payable to a fixed term employee on completion of the contract period.",
             "Interest is payable on delayed gratuity from the date it became payable.",
             "Disputes about gratuity go to the controlling authority."]
    for n, t in enumerate(texts):
        store.add(t, doc="Code on Social Security, 2020", section="Section 53 · Payment of gratuity", page=40.0 + n,
                  chunk_no=n, scope="central", status="current", doc_type="act",
                  chapter="Chapter V — Gratuity", heading="Payment of gratuity")
    store.add("Gratuity eligibility explained for fixed term employee", doc="Ministry FAQs", section="FAQ 3", page=2,
              chunk_no=0, scope="central", status="current", doc_type="faq")


def test_neighbour_expansion_merges_previous_and_next_chunk(services, store, llm):
    seed_sections(store)
    services.settings.keep_k = 1
    llm.responder = lambda m, j: "Yes, fixed term employees get gratuity [1]."
    resp, passages = rag.answer(services, "Gratuity is payable to a fixed term employee on completion of the contract period",
                                Scope(["gujarat"], "default"))
    main = passages[0]
    assert main.chunk_no == 2 and main.page == 42
    assert "within thirty days" in main.context and "Interest is payable on delayed" in main.context
    assert "Nomination forms" not in main.context and "controlling authority" not in main.context  # only ±1
    prompt = llm.calls[-1][1]["content"]
    assert "within thirty days" in prompt and "Interest is payable" in prompt
    cite = resp.citations[0]
    assert cite.page == 42 and "thirty days" not in cite.snippet and "fixed term employee" in cite.snippet


def test_neighbour_lookup_failure_keeps_the_answer(services, store, llm):
    seed_sections(store)
    real = store.search

    def flaky(text, top_k, filter):
        if "chunk_no" in filter:
            raise ConnectionError("down")
        return real(text, top_k, filter)

    store.search = flaky
    llm.responder = lambda m, j: "Yes [1]."
    resp, passages = rag.answer(services, "gratuity fixed term employee contract period", Scope(["gujarat"], "default"))
    assert resp.type == "rag" and passages[0].context == passages[0].text


def test_pages_are_ints_in_citations(services, store, llm):
    seed_sections(store)
    llm.responder = lambda m, j: "Yes [1]."
    resp, _ = rag.answer(services, "gratuity fixed term employee", Scope(["gujarat"], "default"))
    assert all(type(c.page) is int for c in resp.citations)
    prompt = llm.calls[-1][1]["content"]
    assert re.search(r"p\. 4\d ·", prompt) and not re.search(r"p\. \d+\.0", prompt)
