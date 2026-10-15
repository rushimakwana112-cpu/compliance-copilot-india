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


def test_named_document_wins_and_filters(services, store):
    text = "wages means all remuneration payable to an employee including basic pay and dearness allowance"
    store.add(text + " under the old law", doc="Minimum Wages Act, 1948", section="Section 2", page=1,
              scope="central", status="current", doc_type="act")
    store.add(text, doc="Code on Wages, 2019", section="Section 2 · Definitions", page=3,
              scope="central", status="current", doc_type="act")
    store.add("Code on Social Security definition of wages refers to the Code on Wages", doc="Code on Social Security, 2020",
              section="Section 2", page=5, scope="central", status="current", doc_type="act")
    passages = rag.retrieve(services, "what does wages include under the Code on Wages", Scope(["gujarat"], "default"))
    assert {p.doc for p in passages} == {"Code on Wages, 2019"}  # results from the named document exist → only those
    plain = rag.retrieve(services, "what does wages include", Scope(["gujarat"], "default"))
    assert len({p.doc for p in plain}) > 1


def test_named_document_missing_keeps_other_results(services, store):
    seed(store)
    passages = rag.retrieve(services, "gratuity for fixed term employee under the EPF Scheme", Scope(["gujarat"], "default"))
    assert passages  # no EPF Scheme chunks: nothing is filtered away


def test_definition_question_boosts_the_defined_term(services, store):
    store.add("employee means any person employed on wages to do skilled or unskilled work", doc="Code on Wages, 2019",
              section="Section 2 · Definitions", page=3, term="employee", scope="central", status="current", doc_type="act")
    store.add("wages means all remuneration; employee wages include basic pay", doc="Code on Wages, 2019",
              section="Section 2 · Definitions", page=4, term="wages", scope="central", status="current", doc_type="act")
    question = "What is the definition of wages?"
    passages = rag.retrieve(services, question, Scope(["gujarat"], "default"))
    top = passages[0]
    assert top.term == "wages" and top.rank_score - top.score >= rag.TERM_BOOST - 1e-9
    assert rag.Signals.of(question).term == "wage"


def test_section_title_boost(services, store):
    body = "The employer shall pay the amount within thirty days to the eligible employee"
    store.add(body + " as bonus", doc="Code on Wages, 2019", section="Section 26", page=10,
              chapter="Chapter IV — Payment Of Bonus", heading="Eligibility for bonus",
              scope="central", status="current", doc_type="act")
    store.add(body + " as wages", doc="Code on Wages, 2019", section="Section 17", page=6,
              chapter="Chapter III — Payment Of Wages", heading="Time limit for payment of wages",
              scope="central", status="current", doc_type="act")
    passages = rag.retrieve(services, "when must bonus be paid to the employee", Scope(["gujarat"], "default"))
    bonus = next(p for p in passages if p.section == "Section 26")
    wages = next(p for p in passages if p.section == "Section 17")
    assert bonus.rank_score - bonus.score == pytest.approx(rag.HEADING_BOOST) and wages.rank_score == wages.score
    assert passages[0].section == "Section 26"


def test_threshold_and_retrieval_defaults_per_embedding_mode():
    from app.config import Settings

    assert Settings(_env_file=None).score_threshold == 0.38  # pinecone llama-text-embed-v2
    assert Settings(_env_file=None, embed_provider="ollama").score_threshold == 0.55  # bge-m3
    assert Settings(_env_file=None, score_threshold=0.6).score_threshold == 0.6  # explicit value wins
    s = Settings(_env_file=None)
    assert (s.top_k, s.keep_k) == (12, 5)


def test_retrieval_fetches_top_k_and_keeps_keep_k(services, store):
    for i in range(15):
        store.add(f"gratuity rule number {i} for fixed term employee", doc=f"Doc {i}", section="S", page=i,
                  scope="central", status="current", doc_type="act")
    asked = []
    real = store.search
    store.search = lambda text, top_k, filter: asked.append(top_k) or real(text, top_k, filter)
    services.settings.top_k, services.settings.keep_k = 12, 5
    passages = rag.retrieve(services, "gratuity fixed term employee", Scope(["gujarat"], "default"))
    assert asked == [12] and len(passages) == 5


# ----------------------------------------------------------------- defined terms are pinned
@pytest.mark.parametrize("a, b, same", [
    ("fixed term employee", "fixed term employment", True),
    ("Fixed-Term Employees", "“fixed term employment”", True),
    ("principal employer", "principal employment", True),
    ("wages", "Wage", True),
    ("employee", "employment", False),  # only inside a compound
    ("worker", "wages", False),
])
def test_term_normalisation(a, b, same):
    assert rag.same_term(a, b) is same


def test_definition_chunk_is_sent_even_below_threshold(services, store, llm):
    store.add("Code on Social Security, 2020, Section 2 (Definitions) — fixed term employment: (34) engagement "
              "of an employee on the basis of a written contract for a fixed period", doc="Code on Social Security, 2020",
              section="Section 2 · Definitions", page=4, term="fixed term employment",
              scope="central", status="current", doc_type="act")
    store.add("Professional tax slab table", doc="Gujarat PT schedule", page=1, scope="gujarat",
              status="current", doc_type="notification")
    services.settings.score_threshold = 0.99  # nothing passes on score alone
    llm.responder = lambda m, j: "A fixed term employee is engaged on a written contract for a fixed period [1]."
    resp, passages = rag.answer(services, "What is a fixed term employee?", Scope(["gujarat"], "default"))
    assert resp.type == "rag" and passages[0].pinned and passages[0].term == "fixed term employment"
    assert "written contract for a fixed period" in llm.calls[-1][1]["content"]


def test_coverage_question_is_not_a_term(services):
    assert rag.Signals.of("Who is covered for gratuity under the SS Code?").term == ""


def test_coverage_question_also_gets_the_applicability_section(services, store, llm):
    for i in range(8):  # gratuity chunks that crowd the top results
        store.add(f"who is covered for gratuity under the SS code: an employee with continuous service, case {i}",
                  doc="Code on Social Security, 2020", section="Section 53 · Payment of gratuity", page=40 + i,
                  chapter="Chapter V — Gratuity", scope="central", status="current", doc_type="act")
    store.add("1. Short title, extent, commencement and application.—(4) The provisions of Chapter V on gratuity "
              "shall apply to every factory, mine, plantation, shop or establishment with the notified number of "
              "employees", doc="Code on Social Security, 2020", section="Section 1 · Short title, extent, commencement "
              "and application", page=1, chapter="Chapter I — Preliminary", scope="central", status="current", doc_type="act")
    services.settings.keep_k = 3
    llm.responder = lambda m, j: "Employees with continuous service [1]; it applies to factories and establishments [2]."
    question = "Who is covered for gratuity under the SS Code?"
    assert rag.coverage_topic(question) == "gratuity" and rag.coverage_topic("What is gratuity?") == ""
    resp, passages = rag.answer(services, question, Scope(["gujarat"], "default"))
    applicability = [p for p in passages if p.section.startswith("Section 1 ")]
    assert applicability and applicability[0].pinned
    system, user = llm.calls[-1][0]["content"], llm.calls[-1][1]["content"]
    assert "which establishments the provisions apply to" in system and "every factory, mine" in user


# ----------------------------------------------------------------- end matter and cited-only citations
SALIENT = GRATUITY + " as a salient feature of the Code"


def seed_commentary(store):
    store.add(GRATUITY, doc="Code on Social Security, 2020", section="Section 53", page=40, scope="central",
              status="current", doc_type="act", chunk_no=1)
    store.add(SALIENT, doc="Code on Social Security, 2020", section="Statement of Objects and Reasons", page=111,
              scope="central", status="commentary", doc_type="act", chunk_no=9)


def test_commentary_is_searched_but_ranked_below_operative_text(services, store):
    seed_commentary(store)
    question = "gratuity fixed term employee salient feature"
    passages = rag.retrieve(services, question, Scope(["gujarat"], "default"))
    by_section = {p.section: p for p in passages}
    salient, sec53 = by_section["Statement of Objects and Reasons"], by_section["Section 53"]
    assert salient.score > sec53.score  # the end matter is the closer match on similarity alone…
    assert salient.rank_score == pytest.approx(salient.score - rag.COMMENTARY_PENALTY)
    assert passages[0].section == "Section 53"  # …but the operative section ranks first


def test_commentary_is_labelled_not_repealed(services, store, llm):
    seed_commentary(store)
    llm.responder = lambda m, j: "Yes, fixed term employees get gratuity [1]."
    resp, _ = rag.answer(services, "gratuity fixed term employee salient feature", Scope(["gujarat"], "default"))
    prompt = llm.calls[0][1]["content"]
    assert "Statement of Objects and Reasons · p. 111 · Central · COMMENTARY (not operative law)" in prompt
    assert "REPEALED" not in prompt and resp.note is None


def test_renumber_keeps_only_cited_passages_in_order():
    assert rag.renumber("A [3]. B [1, 3]. C [3][1].", 4) == ("A [1]. B [1, 2]. C [1][2].", [3, 1])
    assert rag.renumber("No markers here.", 3) == ("No markers here.", [1, 2, 3])


def test_answer_lists_cited_passages_and_collapses_the_rest(services, store, llm):
    seed(store)
    for i in range(3):
        store.add(f"gratuity fixed term employee filler passage number {i}", doc=f"Filler {i}", page=i + 1,
                  scope="central", status="current", doc_type="act")
    llm.responder = lambda m, j: "Fixed term employees qualify [3], without five years [3][1]."
    resp, passages = rag.answer(services, "gratuity for fixed term employee", Scope(["gujarat"], "default"))
    assert resp.answer == "Fixed term employees qualify [1], without five years [1][2]."
    assert [c.doc for c in resp.citations] == [passages[2].doc, passages[0].doc]
    assert {c.doc for c in resp.other_citations} == {p.doc for p in passages} - {passages[0].doc, passages[2].doc}
    assert len(resp.citations) + len(resp.other_citations) == len(passages)


def test_streamed_answer_is_renumbered_in_done(services, store, llm):
    seed(store)
    store.add("gratuity fixed term employee another passage", doc="Other doc", page=2, scope="central",
              status="current", doc_type="act")
    llm.responder = lambda m, j: "Yes [2]."
    events = list(rag.stream(services, "gratuity for fixed term employee", Scope(["gujarat"], "default")))
    done = dict(events)["done"]
    assert done.answer == "Yes [1]." and len(done.citations) == 1 and done.other_citations
