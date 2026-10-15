"""Income tax: slabs, rebate, cess and deductions from the shipped data/rates.yaml (tax year 2026-27)."""

from __future__ import annotations

from datetime import date

from app import tax

TODAY = date(2026, 10, 15)


def ask(q):
    return tax.run(q, "gujarat", None, TODAY)


def total(resp, regime=None):
    rows = resp.table.rows
    if regime:
        start = rows.index([f"**{regime.title()} regime**", "", ""])
        rows = rows[start:]
    return next(r[2] for r in rows if r[0] == "= Total tax for the year")


def test_rebate_makes_tax_nil_up_to_the_limit():
    r = ask("tax on 12 lakh under new regime")
    assert total(r) == "₹0" and any(row[0] == "− Rebate" for row in r.table.rows)


def test_old_vs_new_with_80c_and_80d():
    r = ask("income tax on 15 LPA old vs new with 80C 1.5 lakh and 80D 18000")
    # New: 14,25,000 taxable → 20,000 + 40,000 + 33,750 = 93,750 + 4% cess = 97,500
    assert total(r, "new") == "₹97,500"
    # Old: 15,00,000 − 50,000 − PT 2,400 − 1,50,000 − 18,000 = 12,79,600 → 12,500 + 1,00,000 + 83,880 = 1,96,380 + 4% cess
    assert total(r, "old") == "₹2,04,235"
    assert "new regime saves you" in r.answer and "ignored there" in r.answer
    assert "FY 2026-27" in r.note and r.citations


def test_hra_exemption_is_the_least_of_three():
    r = ask("hra exemption basic 30000 per month hra 15000 rent 12000 Ahmedabad")
    assert "₹1,08,000 a year" in r.answer  # rent 1,44,000 − 10% of basic 3,60,000


def test_missing_inputs_are_asked_for():
    assert "monthly or yearly" in ask("tax on 50000").answer
    r = ask("which regime is better for 10 lakh with HRA")
    assert "HRA received" in r.answer and "Which city" in r.answer


def test_slabs_question_lists_both_regimes():
    r = ask("what are the old vs new regime tax slabs")
    assert {row[0] for row in r.table.rows} == {"New", "Old (below 60)"} and "Standard deduction" in r.answer


def test_marginal_relief_just_above_the_rebate_limit():
    r = ask("tax on 1285000 per year new regime")  # taxable 12,10,000: tax 61,500 capped at 10,000
    assert any(row[0] == "− Marginal relief" for row in r.table.rows) and total(r) == "₹10,400"


def test_due_date_questions_are_not_tax_calculations():
    assert not tax.wants("When is TDS on salary due?")
