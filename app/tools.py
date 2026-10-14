"""Payroll calculators: PF, ESIC, Professional Tax and gratuity.

Plain Python, no LLM. Language models are unreliable at arithmetic, and a
payroll figure that is off by one rupee is wrong, so the model never computes.
Every rate comes from data/rates.yaml together with the document it was taken
from, and that document is returned as the answer's citation. A rate that
hasn't been filled in yields "rate not configured", never a guess.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.config import DATA_DIR
from app.jurisdiction import Scope, load_states
from app.schemas import ChatResponse, Citation, Table

RATES_PATH = DATA_DIR / "rates.yaml"
NOTE = "Arithmetic is done in Python, not by the model. Rates come from data/rates.yaml and the cited document."


class RateMissing(Exception):
    """A value needed for the calculation is null in rates.yaml."""

    def __init__(self, key: str, source_doc: str = "") -> None:
        super().__init__(key)
        self.key, self.source_doc = key, source_doc


@dataclass
class Rate:
    value: float
    source_doc: str
    section: str = ""
    effective_date: str = ""
    url: str = ""


@lru_cache
def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_rates(path: Path | None = None) -> dict[str, Any]:
    """Parsed rates.yaml (cached per path). Call `_load.cache_clear()` after editing the file in a running app."""
    return _load(path or RATES_PATH)


def rate(rates: dict[str, Any], key: str) -> Rate:
    """Look up 'pf.employee_rate' → Rate, or raise RateMissing if the value is null."""
    node: Any = rates
    for part in key.split("."):
        node = (node or {}).get(part) if isinstance(node, dict) else None
    if not isinstance(node, dict) or node.get("value") is None:
        raise RateMissing(key, (node or {}).get("source_doc", "") if isinstance(node, dict) else "")
    return Rate(float(node["value"]), node.get("source_doc", ""), node.get("section", "") or "",
                str(node.get("effective_date") or ""), node.get("url", "") or "")


def inr(amount: float) -> str:
    """₹ with Indian digit grouping: 123456.4 -> '₹1,23,456'."""
    n = int(round(amount))
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = ",".join([head[max(0, i - 2) : i] for i in range(len(head), 0, -2)][::-1])
        s = f"{head},{tail}"
    return ("-₹" if n < 0 else "₹") + s


def pct(fraction: float) -> str:
    return f"{fraction * 100:g}%"


def _cite(*rates_used: Rate) -> list[Citation]:
    seen: dict[str, Citation] = {}
    for r in rates_used:
        if r.source_doc and r.source_doc not in seen:
            note = f"Effective {r.effective_date}" if r.effective_date else ""
            seen[r.source_doc] = Citation(doc=r.source_doc, section=r.section, page="", score=None, snippet=note, url=r.url)
    return list(seen.values())


# ----------------------------------------------------------------- calculators
def pf_calculator(basic: float, rates: dict[str, Any] | None = None) -> ChatResponse:
    """Employee and employer PF on a monthly basic wage, with the employer share split into EPS and EPF."""
    r = rates or load_rates()
    emp, er, eps, cap = (rate(r, f"pf.{k}") for k in ("employee_rate", "employer_rate", "eps_rate", "eps_wage_ceiling"))
    employee = basic * emp.value
    eps_base = min(basic, cap.value)
    eps_amt = eps_base * eps.value
    employer = basic * er.value
    rows = [
        ["Employee PF", f"{pct(emp.value)} of basic", inr(employee)],
        ["Employer → EPS", f"{pct(eps.value)} of {inr(eps_base)} (wage cap {inr(cap.value)})", inr(eps_amt)],
        ["Employer → EPF", f"{pct(er.value)} of basic minus EPS share", inr(employer - eps_amt)],
        ["Total to PF", "", inr(employee + employer)],
    ]
    return ChatResponse(
        type="tool", tool=f"pf_calculator(basic={basic:g})",
        answer=f"PF split for a monthly basic wage of {inr(basic)}:",
        table=Table(columns=["Component", "Basis", "Amount"], rows=rows, totalRow=True),
        citations=_cite(emp, er, eps, cap), note=NOTE,
    )


def esic_calculator(gross: float, rates: dict[str, Any] | None = None) -> ChatResponse:
    """ESIC contributions on monthly gross wages, or 'not covered' above the wage ceiling."""
    r = rates or load_rates()
    ceiling = rate(r, "esic.wage_ceiling")
    if gross > ceiling.value:
        return ChatResponse(
            type="tool", tool=f"esic_calculator(gross={gross:g})",
            answer=f"ESIC does not apply: gross wages of {inr(gross)} are above the wage ceiling of {inr(ceiling.value)}.",
            table=Table(columns=["Check", "Basis", "Result"], rows=[["Wage ceiling", inr(ceiling.value), "Above → not covered"]]),
            citations=_cite(ceiling),
            note=NOTE + " An employee already covered at the start of a contribution period may stay covered until it ends; check the cited notification.",
        )
    emp, er = rate(r, "esic.employee_rate"), rate(r, "esic.employer_rate")
    rows = [
        ["Employee ESIC", f"{pct(emp.value)} of gross", inr(gross * emp.value)],
        ["Employer ESIC", f"{pct(er.value)} of gross", inr(gross * er.value)],
        ["Total ESIC", "", inr(gross * (emp.value + er.value))],
    ]
    return ChatResponse(
        type="tool", tool=f"esic_calculator(gross={gross:g})",
        answer=f"ESIC applies: gross wages of {inr(gross)} are within the ceiling of {inr(ceiling.value)}.",
        table=Table(columns=["Component", "Basis", "Amount"], rows=rows, totalRow=True),
        citations=_cite(ceiling, emp, er), note=NOTE,
    )


def professional_tax(state: str, monthly_salary: float, rates: dict[str, Any] | None = None) -> ChatResponse:
    """Monthly Professional Tax from the state's slab table."""
    r = rates or load_rates()
    block = (r.get("professional_tax") or {}).get(state)
    if not block or not block.get("slabs"):
        raise RateMissing(f"professional_tax.{state}")
    src = Rate(0, block.get("source_doc", ""), block.get("section", ""), str(block.get("effective_date") or ""), block.get("url", ""))
    for slab in block["slabs"]:
        lo, hi, tax = slab.get("from"), slab.get("to"), slab.get("monthly_tax")
        if lo is None or tax is None:
            raise RateMissing(f"professional_tax.{state}.slabs", src.source_doc)
        if monthly_salary >= lo and (hi is None or monthly_salary <= hi):
            band = f"{inr(lo)} – {inr(hi)}" if hi is not None else f"{inr(lo)} and above"
            name = next((s.name for s in load_states() if s.id == state), state)
            return ChatResponse(
                type="tool", tool=f"professional_tax(state={state!r}, monthly_salary={monthly_salary:g})",
                answer=f"Professional Tax in {name} on a monthly salary of {inr(monthly_salary)}: {inr(tax)} per month.",
                table=Table(columns=["State", "Slab", "PT per month"], rows=[[name, band, inr(tax)]]),
                citations=_cite(src), note=NOTE,
            )
    raise RateMissing(f"professional_tax.{state}.slabs (no slab covers {inr(monthly_salary)})", src.source_doc)


def gratuity_estimate(last_basic_da: float, years: float, employment_type: str = "permanent",
                      rates: dict[str, Any] | None = None) -> ChatResponse:
    """Gratuity = last (basic + DA) × days_per_year / wage_days_divisor × counted years, capped at max_amount."""
    r = rates or load_rates()
    days, divisor, cap = (rate(r, f"gratuity.{k}") for k in ("days_per_year", "wage_days_divisor", "max_amount"))
    fixed = employment_type.lower().startswith("fixed")
    min_years = rate(r, "gratuity.min_years_fixed_term" if fixed else "gratuity.min_years_permanent")
    completed = math.floor(years)
    try:
        round_after = rate(r, "gratuity.round_up_after_months")
        if (years - completed) * 12 > round_after.value:
            completed += 1
    except RateMissing:
        round_after = None
    tool = f"gratuity_estimate(last_basic_da={last_basic_da:g}, years={years:g}, employment_type={employment_type!r})"
    used = [r_ for r_ in (days, divisor, cap, min_years, round_after) if r_]
    if years < min_years.value:
        return ChatResponse(
            type="tool", tool=tool, citations=_cite(*used), note=NOTE,
            answer=f"Not yet eligible: {employment_type} employees need {min_years.value:g} year(s) of continuous service; "
            f"{years:g} given. (Death or disablement are exceptions; see the cited section.)",
        )
    amount = last_basic_da * days.value / divisor.value * completed
    capped = min(amount, cap.value)
    rows = [
        ["Last basic + DA", "per month", inr(last_basic_da)],
        ["Counted years", f"{years:g} given" + (f", >{round_after.value:g} months rounds up" if round_after else ""), str(completed)],
        ["Formula", f"wage × {days.value:g} / {divisor.value:g} × years", inr(amount)],
    ]
    if capped < amount:
        rows.append(["Ceiling applied", f"max {inr(cap.value)}", inr(capped)])
    rows.append(["Gratuity payable", "", inr(capped)])
    return ChatResponse(
        type="tool", tool=tool, answer=f"Estimated gratuity for {completed} counted year(s) at {inr(last_basic_da)} last basic + DA:",
        table=Table(columns=["Item", "Basis", "Amount"], rows=rows, totalRow=True),
        citations=_cite(*used), note=NOTE,
    )


# ----------------------------------------------------------------- parsing the question
MONEY = re.compile(r"(?:₹|\brs\.?|\binr)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b|lakhs?\b|lacs?\b|l\b)?", re.IGNORECASE)
YEARS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", re.IGNORECASE)


def parse_amounts(text: str) -> list[float]:
    """Money amounts in order: '₹18,000', '18k', '1.5 lakh'. Numbers followed by 'years' are skipped."""
    out = []
    for m in MONEY.finditer(text):
        tail = text[m.end() : m.end() + 6].lower()
        if re.match(r"\s*(years?|yrs?|months?|days?)\b", tail):
            continue
        n = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        n *= 1_000 if unit == "k" else 100_000 if unit.startswith(("lakh", "lac", "l")) else 1
        if n >= 100:  # skip "section 53", "2 states"
            out.append(n)
    return out


def run(question: str, scope: Scope, rates: dict[str, Any] | None = None) -> ChatResponse:
    """Pick the calculator from the question, parse its inputs and run it."""
    q = question.lower()
    amounts = parse_amounts(question)
    if not amounts:
        return ChatResponse(type="tool", answer="Please include the monthly amount, e.g. \"Calculate PF for basic ₹18,000\".")
    amount = amounts[0]
    try:
        if "gratuity" in q:
            ym = YEARS.search(question)
            if not ym:
                return ChatResponse(type="tool", answer="How many years of service? e.g. \"gratuity for basic + DA ₹40,000 and 7 years\".")
            etype = "fixed-term" if re.search(r"fixed[- ]?term", q) else "permanent"
            return gratuity_estimate(amount, float(ym.group(1)), etype, rates)
        if re.search(r"\b(esic|esi)\b", q):
            return esic_calculator(amount, rates)
        if re.search(r"professional tax|\bpt\b", q):
            return professional_tax(scope.primary, amount, rates)
        return pf_calculator(amount, rates)
    except RateMissing as e:
        src = f" Fill it from {e.source_doc.removeprefix('TODO: ')}." if e.source_doc else ""
        return ChatResponse(
            type="tool", answer=f"Rate not configured: `{e.key}` is empty in data/rates.yaml, so I won't calculate.{src}",
            note="Add the value with its source_doc and effective_date, then ask again.",
        )
