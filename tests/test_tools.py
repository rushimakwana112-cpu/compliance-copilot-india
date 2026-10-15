"""Calculators. These tests use made-up rates from a fixture, never the real file,
so they check the arithmetic, not the law."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from app.jurisdiction import Scope
from app.router import route_rules
from app.tools import (
    RateMissing, esic_calculator, gratuity_estimate, inr, load_rates, parse_amounts, pf_calculator,
    professional_tax, question_date, rate, resolved, run,
)


FIXTURE = Path(__file__).parent / "fixtures" / "rates.yaml"
TODAY = date(2026, 10, 15)


@pytest.fixture
def rates():
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


def test_inr_indian_grouping():
    assert inr(1234567) == "₹12,34,567" and inr(950) == "₹950" and inr(18000) == "₹18,000"


def test_pf_split(rates):
    out = pf_calculator(20000, rates, TODAY)
    amounts = [row[-1] for row in out.table.rows]
    # PF wage = min(20000, ceiling 18000) = 18000; employee 10% = 1800; EPS 5% of capped 10000 = 500;
    # employer EPF 1800-500 = 1300; EDLI 1% of 18000 = 180; total 1800+1800+180 = 3780
    assert amounts == ["₹18,000", "₹1,800", "₹500", "₹1,300", "₹180", "₹3,780"]
    assert out.table.totalRow and out.citations[0].score is None
    # The output says which ceiling and notification it used
    assert "wage ceiling of ₹18,000 from New test notification" in out.answer
    assert "New test notification" in out.table.rows[0][1] and out.citations[0].doc == "New test notification"


def test_pf_uses_the_ceiling_valid_on_the_date(rates):
    old = pf_calculator(20000, rates, date(2026, 3, 1))
    assert old.table.rows[0][-1] == "₹12,000" and "Old test notification" in old.answer
    s = Scope(["gujarat"], "default")
    assert "₹12,000 from Old test notification" in run("PF for basic 20000 on 1 March 2026", s, rates, TODAY).answer
    assert "₹18,000 from New test notification" in run("PF for basic 20000", s, rates, TODAY).answer
    assert run("PF for basic 20000 in March 2026", s, rates, TODAY).tool == "pf_calculator(basic=20000)"


def test_current_version_wins_while_dates_are_unknown():
    node = [{"value": 1, "status": "historical", "source_doc": "old"}, {"value": 2, "status": "current", "source_doc": "new"}]
    out = rate({"pf": {"wage_ceiling": node}}, "pf.wage_ceiling", TODAY)
    assert (out.value, out.source_doc, out.label()) == (2, "new", "new, effective date not configured")


def test_null_value_says_rate_not_configured(rates):
    rates["pf"]["edli_rate"][0]["value"] = None
    out = run("Calculate PF for basic ₹18,000", Scope(["gujarat"], "default"), rates, TODAY)
    assert out.answer.startswith("Rate not configured: pf.edli_rate.") and out.table is None


@pytest.mark.parametrize("text, expected", [
    ("PF on basic 20000 on 1 Oct 2026", date(2026, 10, 1)),
    ("PF on basic 20000 as of 2026-07-01", date(2026, 7, 1)),
    ("PF on basic 20000 on 15/08/2026", date(2026, 8, 15)),
    ("PF on basic 20000 in March 2026", date(2026, 3, 1)),
    ("PF on basic 20000 from September 17, 2026", date(2026, 9, 17)),
    ("PF on basic 20000", None),
])
def test_question_date(text, expected):
    found, rest = question_date(text)
    assert found == expected
    assert parse_amounts(rest) == [20000]  # the year is not read as an amount


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


def test_shipped_rates_calculate_with_their_source():
    out = run("Calculate PF for basic ₹18,000", Scope(["gujarat"], "default"), load_rates(), TODAY)
    assert out.type == "tool" and out.table is not None and out.table.rows[-1][0] == "Total"
    assert any("S.O. 5109(E)" in c.doc for c in out.citations)


def test_shipped_rates_refuse_what_is_not_configured():
    # Maharashtra PT is still empty in rates.yaml: refuse, don't guess.
    out = run("Calculate professional tax on ₹15,000 salary in Pune", Scope(["maharashtra"], "default"), load_rates(), TODAY)
    assert out.answer.startswith("Rate not configured: professional_tax.maharashtra") and out.table is None
    # No gratuity ceiling is notified yet: the amount is shown uncapped and says so.
    g = run("Calculate gratuity for last basic + DA ₹40,000 and 7 years", Scope(["gujarat"], "default"), load_rates(), TODAY)
    assert any(row[0] == "Ceiling" and "not capped" in row[1] for row in g.table.rows)


def test_shipped_rates_are_dated_and_sourced():
    shipped = load_rates()
    on = resolved(shipped, TODAY)["rates"]["pf.wage_ceiling"]
    assert on["value"] == 25000 and on["effective_from"] == "2026-09-17" and on["status"] == "current"
    old = [v for v in shipped["pf"]["wage_ceiling"] if v["status"] == "historical"]
    assert [v["value"] for v in old] == [15000]
    for group in ("pf", "esic", "gratuity"):  # every figure is a list of dated versions with its source
        for name, node in shipped[group].items():
            assert isinstance(node, list) and all({"value", "effective_from", "effective_to", "source_doc", "status"}
                                                  <= set(v) for v in node), f"{group}.{name}"
            assert all(v["source_doc"] for v in node), f"{group}.{name}"


def test_resolved_lists_nulls(rates):
    rates["esic"]["employee_rate"][0]["value"] = None
    out = resolved(rates, TODAY)
    assert out["date"] == "2026-10-15" and out["rates"]["esic.employee_rate"]["value"] is None
    assert out["rates"]["pf.wage_ceiling"] == {"value": 18000, "effective_from": "2026-07-01", "effective_to": None,
                                               "source_doc": "New test notification", "section": "p. 2",
                                               "status": "current", "url": None}
    assert out["professional_tax"]["gujarat"]["slabs"][1]["monthly_tax"] == 100


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
    assert run("how much PF on basic 18k", s, rates, TODAY).tool == "pf_calculator(basic=18000)"


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
