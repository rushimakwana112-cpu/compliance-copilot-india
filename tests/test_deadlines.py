"""Due-date answers: routing, date arithmetic, and refusing when a date isn't configured."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from app import deadlines, tools
from app.jurisdiction import Scope
from app.router import route_rules

FIXTURE = Path(__file__).parent / "fixtures" / "rates.yaml"
TODAY = date(2026, 10, 15)


@pytest.fixture
def rates():
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


def scope(state="gujarat"):
    return Scope([state], "default")


@pytest.mark.parametrize("q", [
    "When is PF due?", "What is the due date for PF payment?", "Last date to deposit TDS on salary",
    "When should I file the 24Q return?", "Professional tax payment due date in Gujarat", "When is Form 16 due?",
    "ESIC deadline for September",
])
def test_due_date_questions_go_to_the_tools(q):
    assert deadlines.is_deadline_question(q) and route_rules(q) == "tool"


@pytest.mark.parametrize("q", ["What is PF?", "Calculate PF for basic ₹18,000", "Who is eligible for gratuity?"])
def test_other_questions_are_not_due_date_questions(q):
    assert not deadlines.is_deadline_question(q)


def test_next_monthly_due_rolls_over():
    assert deadlines.next_monthly_due(15, 1, date(2026, 10, 15)) == (date(2026, 10, 15), "September 2026")
    assert deadlines.next_monthly_due(15, 1, date(2026, 10, 16)) == (date(2026, 11, 15), "October 2026")
    assert deadlines.next_monthly_due(7, 1, date(2026, 12, 20)) == (date(2027, 1, 7), "December 2026")
    assert deadlines.next_monthly_due(31, 1, date(2026, 2, 1)) == (date(2026, 2, 28), "January 2026")


def test_pf_due_date_answer(rates):
    resp = tools.run("When is PF due?", scope(), rates, TODAY)
    assert resp.type == "tool" and resp.tool == "due_dates"
    row = resp.table.rows[0]
    assert row[1] == "by the 12th of the following month"
    assert row[2] == "12 November 2026 (for October 2026)"
    assert "Test PF scheme, para 9" in row[3] and resp.citations[0].doc == "Test PF scheme"
    assert "Test note." in resp.answer


def test_tds_deposit_and_quarterly_statement(rates):
    dep = tools.run("Due date to deposit TDS on salary", scope(), rates, TODAY)
    assert dep.table.rows[0][0] == "TDS on salary: deposit" and dep.table.rows[0][2] == "5 November 2026 (for October 2026)"
    q = tools.run("When is the 24Q TDS return due?", scope(), rates, TODAY)
    assert [r[0] for r in q.table.rows] == ["Quarterly TDS statement (salary)"]
    assert q.table.rows[0][2] == "20 October 2026 (Q2)"
    f16 = tools.run("Form 16 due date", scope(), rates, TODAY)
    assert f16.table.rows[0][2] == "10 June 2027" and "Form 130" in f16.table.rows[0][0]


def test_professional_tax_due_date_uses_the_state(rates):
    resp = tools.run("Professional tax payment due date", scope("gujarat"), rates, TODAY)
    assert resp.table.rows[0][0] == "Professional Tax (Gujarat)" and resp.table.rows[0][2] == "18 October 2026 (for September 2026)"
    missing = tools.run("Professional tax payment due date", scope("maharashtra"), rates, TODAY)
    assert missing.answer.startswith("Due date not configured: due_dates.professional_tax.maharashtra")


def test_null_due_date_is_not_guessed(rates):
    resp = tools.run("When is ESIC due?", scope(), rates, TODAY)
    assert resp.answer.startswith("Due date not configured: due_dates.esic") and "Test ESIC regulation" in resp.answer
    both = tools.run("When are PF and ESIC due?", scope(), rates, TODAY)
    assert len(both.table.rows) == 1 and "Not configured yet: due_dates.esic" in both.answer


def test_calculation_with_due_word_still_calculates(rates):
    resp = tools.run("Calculate PF due for basic ₹18,000", scope(), rates, TODAY)
    assert resp.tool.startswith("pf_calculator")


def test_api_rates_lists_due_dates(rates):
    out = tools.resolved(rates, TODAY)["due_dates"]
    assert out["pf"]["value"] == 12 and out["pf"]["source_doc"] == "Test PF scheme"
    assert out["esic"]["value"] is None and out["professional_tax.gujarat"]["value"] == 18
