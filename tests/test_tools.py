"""Calculators. These tests use made-up rates from a fixture, never the real file,
so they check the arithmetic, not the law."""

from __future__ import annotations

import pytest

from app.jurisdiction import Scope
from app.router import route_rules
from app.tools import (
    RateMissing, esic_calculator, gratuity_estimate, inr, load_rates, parse_amounts, pf_calculator,
    professional_tax, run,
)


def r(v, doc="Test circular"):
    return {"value": v, "source_doc": doc, "effective_date": "2025-01-01"}


@pytest.fixture
def rates():
    return {
        "pf": {"employee_rate": r(0.10), "employer_rate": r(0.10), "eps_rate": r(0.05), "eps_wage_ceiling": r(10000)},
        "esic": {"wage_ceiling": r(20000), "employee_rate": r(0.01), "employer_rate": r(0.03)},
        "gratuity": {"days_per_year": r(10), "wage_days_divisor": r(20), "min_years_permanent": r(4),
                     "min_years_fixed_term": r(1), "round_up_after_months": r(6), "max_amount": r(100000)},
        "professional_tax": {"gujarat": {"source_doc": "Test PT schedule", "slabs": [
            {"from": 0, "to": 9999, "monthly_tax": 0}, {"from": 10000, "to": None, "monthly_tax": 100}]}},
    }


def test_inr_indian_grouping():
    assert inr(1234567) == "₹12,34,567" and inr(950) == "₹950" and inr(18000) == "₹18,000"


def test_pf_split(rates):
    out = pf_calculator(20000, rates)
    amounts = [row[-1] for row in out.table.rows]
    # employee 10% = 2000; EPS 5% of capped 10000 = 500; employer EPF 2000-500 = 1500; total 4000
    assert amounts == ["₹2,000", "₹500", "₹1,500", "₹4,000"]
    assert out.table.totalRow and out.citations[0].doc == "Test circular" and out.citations[0].score is None


def test_esic_within_and_above_ceiling(rates):
    within = esic_calculator(15000, rates)
    assert [row[-1] for row in within.table.rows] == ["₹150", "₹450", "₹600"]
    above = esic_calculator(25000, rates)
    assert "does not apply" in above.answer and above.table.rows[0][-1].startswith("Above")


def test_professional_tax_slab(rates):
    assert professional_tax("gujarat", 15000, rates).table.rows[0][-1] == "₹100"
    assert professional_tax("gujarat", 5000, rates).table.rows[0][-1] == "₹0"
    with pytest.raises(RateMissing):
        professional_tax("karnataka", 15000, rates)


def test_gratuity_rounding_ceiling_and_eligibility(rates):
    # 30000 × 10/20 = 15000 per year; 5 years 7 months rounds up to 6 → 90,000
    assert gratuity_estimate(30000, 5 + 7 / 12, "permanent", rates).table.rows[-1][-1] == "₹90,000"
    # 9 years would be 135,000 → capped at 100,000
    assert gratuity_estimate(30000, 9, "permanent", rates).table.rows[-1][-1] == "₹1,00,000"
    assert "Not yet eligible" in gratuity_estimate(30000, 2, "permanent", rates).answer
    assert gratuity_estimate(30000, 2, "fixed-term", rates).table is not None


def test_shipped_rates_are_unconfigured_and_say_so():
    # The repo ships rates.yaml with nulls on purpose: the calculator must refuse, not guess.
    out = run("Calculate PF for basic ₹18,000", Scope(["gujarat"], "default"), load_rates())
    assert out.type == "tool" and "Rate not configured" in out.answer and out.table is None


def test_parse_amounts():
    assert parse_amounts("PF for basic ₹18,000") == [18000]
    assert parse_amounts("gratuity on 40k for 7 years") == [40000]
    assert parse_amounts("ESIC on 1.5 lakh gross") == [150000]
    assert parse_amounts("section 53 for 2 states") == []


def test_run_picks_calculator(rates):
    s = Scope(["gujarat"], "detected")
    assert run("Calculate ESIC for gross 15,000", s, rates).tool.startswith("esic_calculator")
    assert run("compute PT on ₹15000 salary in Ahmedabad", s, rates).tool.startswith("professional_tax(state='gujarat'")
    assert run("calculate gratuity for basic+DA 30000 and 6 years fixed-term", s, rates).tool.endswith("'fixed-term')")
    assert run("how much PF on basic 18k", s, rates).tool == "pf_calculator(basic=18000)"


@pytest.mark.parametrize("q, expected", [
    ("Calculate PF for basic ₹18,000", "tool"),
    ("How much ESIC on 15000 gross?", "tool"),
    ("Is ESIC applicable on ₹24,000 gross?", "rag"),
    ("Is ESIC applicable to employees earning 24000?", "rag"),
    ("Which sample employees are above the ESIC limit?", "sql"),
    ("How many of our staff are in Gujarat?", "sql"),
    ("Gratuity rule for fixed-term employees", "rag"),
])
def test_router_rules(q, expected):
    assert route_rules(q) == expected
