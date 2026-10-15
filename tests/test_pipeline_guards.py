"""The chat pipeline answers guards and salary calculations before routing, in the user's language."""

from __future__ import annotations

from app import pipeline
from app.schemas import ChatRequest, Turn


def chat(services, message, history=()):
    return pipeline.handle(services, ChatRequest(message=message, history=list(history)), log=False)


def test_personal_record_needs_no_search_or_model(services, store, llm):
    resp, turn = chat(services, "What is my salary?")
    assert turn.route == "guide" and resp.type == "guide" and store.searches == [] and llm.calls == []
    assert resp.scope is not None  # the page shows "Central + Gujarat" like every other answer


def test_sample_data_questions_get_no_data_by_default(services, llm):
    resp, _ = chat(services, "How many of our staff are in Gujarat?")
    assert "don't have any employee data" in resp.answer and llm.calls == []


def test_ctc_conversation_runs_across_turns(services):
    first, turn = chat(services, "My CTC is 6 lakh, what is my in-hand?")
    assert turn.route == "tool" and first.type == "guide"
    resp, _ = chat(services, "1. yearly 2. 40% 3. yes, capped, included 4. no 5. Gujarat",
                   [Turn(role="user", content="My CTC is 6 lakh, what is my in-hand?"),
                    Turn(role="assistant", content=first.answer)])
    assert resp.type == "tool" and resp.tool == "ctc_in_hand" and "In-hand per month" in resp.table.rows[-1][0]


def test_hinglish_answer_is_translated_keeping_numbers(services, llm):
    llm.responder = lambda messages, json_mode: "Hinglish mein: " + messages[-1]["content"]
    resp, turn = chat(services, "mera PF balance kitna hai batao")
    assert turn.lang == "hinglish" and turn.route == "guide"
    assert resp.answer.startswith("Hinglish mein:") and "EPFO" in resp.answer


def test_tax_planning_with_figures_gets_advisor_note(services):
    resp, _ = chat(services, "Calculate PF for basic 18000, how can I save tax?")
    assert "tax advisor" in resp.answer


def test_check_my_calculation_matches_or_corrects(services):
    resp, turn = chat(services, "Is this calculation correct? 3 days unpaid leave on 26000 salary, 30 days method = 2600")
    assert turn.claim == 2600 and resp.answer.startswith("Your figure ₹2,600 matches")
    resp, _ = chat(services, "Is this correct? 3 days unpaid leave on 26000 salary, 30 days method, I got 3000")
    assert resp.answer.startswith("Your figure ₹3,000 doesn't match: I get ₹2,600")


def test_explanations_come_from_the_faq_with_rates_filled_in(services, llm):
    resp, turn = chat(services, "What is the difference between CTC and gross salary?")
    assert turn.route == "guide" and resp.tool == "faq:ctc_gross_net" and llm.calls == []
    resp, _ = chat(services, "What are the PF rates?")
    assert "{" not in resp.answer and "%" in resp.answer and resp.citations


def test_tax_and_labour_calculators_are_reached(services):
    assert chat(services, "tax on 12 lakh under new regime")[0].tool == "income_tax"
    assert chat(services, "overtime for 10 hours on salary 26000")[0].tool == "overtime"


def test_check_my_pf_calculation_goes_to_the_calculator(services):
    resp, turn = chat(services, "Is this correct? PF on basic 18000 = 2160")
    assert turn.route == "tool" and resp.answer.startswith("Your figure ₹2,160")
