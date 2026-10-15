"""Salary calculations from the employee's own figures. Uses the made-up rates in tests/fixtures/rates.yaml
(PF 10% with an ₹18,000 ceiling, ESIC 1% / 3% up to ₹20,000, Gujarat PT ₹100 above ₹10,000)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from app import salary
from app.schemas import Turn

TODAY = date(2026, 10, 15)


@pytest.fixture
def rates():
    return yaml.safe_load((Path(__file__).parent / "fixtures" / "rates.yaml").read_text(encoding="utf-8"))


def ask(text, rates, history=()):
    return salary.run(text, list(history), "gujarat", rates, TODAY)


def converse(rates, *messages):
    history, resp = [], None
    for m in messages:
        resp = ask(m, rates, history)
        history += [Turn(role="user", content=m), Turn(role="assistant", content=resp.answer)]
    return resp


def last_row(resp):
    return resp.table.rows[-1]


# ----------------------------------------------------------------- CTC → in-hand
def test_ctc_asks_all_missing_details_in_one_message(rates):
    r = ask("My CTC is 6 lakh, what is my in-hand?", rates)
    assert r.type == "guide" and r.table is None
    for q in ("monthly or yearly", "basic salary", "full basic or capped", "₹18,000", "ESIC", "state", "tax regime"):
        assert q in r.answer


def test_ctc_numbered_reply_is_calculated_step_by_step(rates):
    r = converse(rates, "My CTC is 6 lakh, what is my in-hand?", "1. yearly 2. 40% 3. yes, capped, included 4. no")
    steps = [row[0] for row in r.table.rows]
    assert steps == ["CTC per month", "Basic", "− Employer PF (in CTC)", "= Gross salary", "− Employee PF",
                     "− Professional Tax (Gujarat)", "= In-hand per month (before TDS)"]
    # 50,000 − employer PF 1,800 (10% of 18,000 cap) = 48,200 gross; − PF 1,800 − PT 100 = 46,300
    assert last_row(r)[2] == "₹46,300" and "₹46,300 a month" in r.answer
    assert "FY 2026-27" in r.note and "HR/payroll" in r.note and "rates can change" in r.note


def test_esic_not_asked_when_gross_is_above_the_ceiling(rates):
    r = ask("My CTC is ₹6 lakh per year, in-hand?", rates)
    assert "ESIC" not in r.answer and "basic" in r.answer


def test_unknown_pf_shows_two_labelled_results(rates):
    r = converse(rates, "My CTC is ₹6 lakh per year, in-hand?", "basic 50%, not sure about PF")
    assert r.type == "tool" and "With PF" in r.answer and "Without PF" in r.answer
    assert "both results" in r.answer and "ESIC does not apply" in r.answer


def test_unknown_esic_shows_two_results_when_it_could_apply(rates):
    r = converse(rates, "CTC 2.4 lakh per year, in hand?", "basic 50%, PF capped not included, not sure about ESIC")
    assert "With ESIC" in r.answer and "Without ESIC" in r.answer and "With PF" not in r.answer


def test_short_form_salary_asks_ctc_or_gross_then_calculates_from_gross(rates):
    r = ask("salary 25k in-hand?", rates)
    assert "CTC (cost to company) or your gross salary" in r.answer
    r = converse(rates, "My gross salary is 15000 per month, in-hand?", "basic 8000, PF yes on full basic, ESIC yes")
    # 15,000 − PF 800 (10% of 8,000) − ESIC 150 (1%) − PT 100 = 13,950; employer shares are on top of gross
    assert last_row(r)[2] == "₹13,950"
    assert "Employer" not in " ".join(row[0] for row in r.table.rows)


# ----------------------------------------------------------------- days
def test_per_day_asks_the_company_divisor(rates):
    r = ask("per day salary for 30000", rates)
    assert r.type == "guide" and "26 days" in r.answer and "30 days" in r.answer and "calendar" in r.answer


def test_unpaid_leave_deduction(rates):
    r = ask("3 days unpaid leave deduction on 26000 salary, 30 days method", rates)
    assert "₹2,600" in r.answer and "₹23,400" in r.answer


def test_joining_mid_month_uses_calendar_days(rates):
    r = ask("joined on 12th October, salary 31000, calendar days", rates)
    assert "for 20 paid days" in r.answer and last_row(r)[2] == "₹20,000"


def test_leaving_mid_month(rates):
    r = ask("resignation, last working day 20th, salary 30000, 30 days", rates)
    assert "₹20,000" in r.answer and "full and final" in r.answer


def test_26_day_basis_asks_for_paid_days(rates):
    assert "paid days" in ask("joined on 12th, salary 26000, 26 days", rates).answer


def test_impossible_date_is_pointed_out(rates):
    assert "only 30 days" in ask("joined on 31st September, salary 30000, calendar days", rates).answer


def test_law_question_about_absence_is_not_a_calculation():
    assert not salary.wants("Can my employer cut pay if I'm absent?", [])


# ----------------------------------------------------------------- increment
def test_increment_with_arrears(rates):
    r = ask("10% increment on 40000 monthly, arrears for 3 months", rates)
    assert "₹44,000 a month" in r.answer and "₹12,000" in r.answer


def test_increment_asks_for_missing_inputs(rates):
    r = ask("I got a 10% hike", rates)
    assert r.type == "guide" and "salary the increment applies to" in r.answer
