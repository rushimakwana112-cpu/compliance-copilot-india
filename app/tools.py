"""Payroll calculators: PF, ESIC, Professional Tax and gratuity.

Plain Python, no LLM. Language models are unreliable at arithmetic, and a
payroll figure that is off by one rupee is wrong, so the model never computes.
Every rate comes from data/rates.yaml together with the document it was taken
from, and that document is returned as the answer's citation. Each figure is a
list of dated versions; the one valid on today's date (or on a date written in
the question) is used. A rate that hasn't been filled in yields
"Rate not configured: <field>", never a guess.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app import deadlines
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
    effective_from: str = ""
    url: str = ""
    effective_to: str = ""
    status: str = ""

    def label(self) -> str:
        """'Gazette notification 17 Sept 2026, effective 2026-10-01' (or 'effective date not configured')."""
        when = f"effective {self.effective_from}" if self.effective_from else "effective date not configured"
        return f"{self.source_doc or 'data/rates.yaml'}, {when}"


@lru_cache
def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_rates(path: Path | None = None) -> dict[str, Any]:
    """Parsed rates.yaml (cached per path). Call `_load.cache_clear()` after editing the file in a running app."""
    return _load(path or RATES_PATH)


def _date(value: Any) -> date | None:
    """YAML gives dates as date objects ('2026-10-01' unquoted) or strings; null stays None."""
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def versions(node: Any) -> list[dict[str, Any]]:
    """The dated versions of one figure. A single {value: …} mapping (the old format) is one version."""
    if isinstance(node, list):
        return [v for v in node if isinstance(v, dict)]
    return [node] if isinstance(node, dict) and ("value" in node or "slabs" in node) else []


def pick(node: Any, on: date | None = None) -> dict[str, Any] | None:
    """The version valid on `on` (default today). Among several valid ones (e.g. dates not filled in yet),
    a `current` one beats a `historical` one, then the latest effective_from wins."""
    on = on or date.today()
    valid = [v for v in versions(node)
             if (_date(v.get("effective_from")) or date.min) <= on <= (_date(v.get("effective_to")) or date.max)]
    if not valid:
        return None
    return max(valid, key=lambda v: (v.get("status", "current") != "historical",
                                     _date(v.get("effective_from")) or date.min))


def _node(rates: dict[str, Any], key: str) -> Any:
    node: Any = rates
    for part in key.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def _rate_of(v: dict[str, Any], value: float) -> Rate:
    return Rate(value, v.get("source_doc", "") or "", v.get("section", "") or "",
                str(v.get("effective_from") or v.get("effective_date") or ""), v.get("url", "") or "",
                str(v.get("effective_to") or ""), v.get("status", "") or "")


def rate(rates: dict[str, Any], key: str, on: date | None = None) -> Rate:
    """Look up 'pf.employee_rate' on a date → Rate, or raise RateMissing if no version applies or its value is null."""
    v = pick(_node(rates, key), on)
    if v is None or v.get("value") is None:
        raise RateMissing(key, (v or {}).get("source_doc", ""))
    return _rate_of(v, float(v["value"]))


def resolved(rates: dict[str, Any], on: date | None = None) -> dict[str, Any]:
    """Every figure as it applies on `on`, for GET /api/rates. Null values are listed, not hidden."""
    on = on or date.today()
    out: dict[str, Any] = {"date": on.isoformat(), "rates": {}, "professional_tax": {}, "due_dates": {}}
    for group in ("pf", "esic", "gratuity"):
        for name, node in (rates.get(group) or {}).items():
            v = pick(node, on)
            out["rates"][f"{group}.{name}"] = {
                "value": None if v is None else v.get("value"),
                **({k: (None if (x := (v or {}).get(k)) in (None, "") else str(x))
                    for k in ("effective_from", "effective_to", "source_doc", "section", "status", "url")}),
            }
    for state, node in (rates.get("professional_tax") or {}).items():
        v = pick(node, on) or {}
        out["professional_tax"][state] = {
            "slabs": v.get("slabs") or [],
            **{k: (None if (x := v.get(k)) in (None, "") else str(x))
               for k in ("effective_from", "effective_to", "source_doc", "section", "status", "url")},
        }
    due = rates.get("due_dates") or {}
    flat = {**{k: n for k, n in due.items() if k != "professional_tax"},
            **{f"professional_tax.{st}": n for st, n in (due.get("professional_tax") or {}).items()}}
    for name, node in flat.items():
        v = pick(node, on) or {}
        out["due_dates"][name] = {
            "value": v.get("value"), "month_offset": v.get("month_offset"), "note": v.get("note"),
            **{k: (None if (x := v.get(k)) in (None, "") else str(x))
               for k in ("effective_from", "effective_to", "source_doc", "section", "status", "url")},
        }
    return out


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
            note = f"Effective {r.effective_from}" if r.effective_from else "Effective date not configured"
            seen[r.source_doc] = Citation(doc=r.source_doc, section=r.section, page=None, score=None, snippet=note, url=r.url)
    return list(seen.values())


# ----------------------------------------------------------------- calculators
def pf_calculator(basic: float, rates: dict[str, Any] | None = None, on: date | None = None) -> ChatResponse:
    """Statutory PF on a monthly basic wage, on wages up to the PF wage ceiling valid on `on`:
    employee share, employer share split into EPS (up to the EPS wage cap) and EPF, plus EDLI."""
    r = rates or load_rates()
    ceiling = rate(r, "pf.wage_ceiling", on)
    emp, er, eps, cap, edli = (rate(r, f"pf.{k}", on)
                               for k in ("employee_rate", "employer_rate", "eps_rate", "eps_wage_cap", "edli_rate"))
    pf_wage = min(basic, ceiling.value)
    employee, employer = pf_wage * emp.value, pf_wage * er.value
    eps_base = min(basic, cap.value)
    eps_amt = eps_base * eps.value
    edli_amt = pf_wage * edli.value
    rows = [
        ["PF wage", f"basic {inr(basic)}, up to the wage ceiling of {inr(ceiling.value)} ({ceiling.label()})", inr(pf_wage)],
        ["Employee PF", f"{pct(emp.value)} of PF wage", inr(employee)],
        ["Employer → EPS", f"{pct(eps.value)} of {inr(eps_base)} (EPS wage cap {inr(cap.value)})", inr(eps_amt)],
        ["Employer → EPF", f"{pct(er.value)} of PF wage minus EPS share", inr(employer - eps_amt)],
        ["Employer → EDLI", f"{pct(edli.value)} of PF wage", inr(edli_amt)],
        ["Total", "", inr(employee + employer + edli_amt)],
    ]
    return ChatResponse(
        type="tool", tool=f"pf_calculator(basic={basic:g})",
        answer=f"PF for a monthly basic wage of {inr(basic)}, using the PF wage ceiling of {inr(ceiling.value)} "
               f"from {ceiling.label()}:",
        table=Table(columns=["Component", "Basis", "Amount"], rows=rows, totalRow=True),
        citations=_cite(ceiling, emp, er, eps, cap, edli), note=NOTE,
    )


def esic_calculator(gross: float, rates: dict[str, Any] | None = None, on: date | None = None) -> ChatResponse:
    """ESIC contributions on monthly gross wages, or 'not covered' above the wage ceiling."""
    r = rates or load_rates()
    ceiling = rate(r, "esic.wage_ceiling", on)
    if gross > ceiling.value:
        return ChatResponse(
            type="tool", tool=f"esic_calculator(gross={gross:g})",
            answer=f"ESIC does not apply: gross wages of {inr(gross)} are above the wage ceiling of {inr(ceiling.value)}.",
            table=Table(columns=["Check", "Basis", "Result"], rows=[["Wage ceiling", inr(ceiling.value), "Above → not covered"]]),
            citations=_cite(ceiling),
            note=NOTE + " An employee already covered at the start of a contribution period may stay covered until it ends; check the cited notification.",
        )
    emp, er = rate(r, "esic.employee_rate", on), rate(r, "esic.employer_rate", on)
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


def pt_monthly(state: str, monthly_salary: float, rates: dict[str, Any] | None = None,
               on: date | None = None) -> tuple[float, Rate, str]:
    """(monthly PT, its source, slab label) from the state's slab table valid on `on`. Raises RateMissing."""
    r = rates or load_rates()
    block = pick((r.get("professional_tax") or {}).get(state), on)
    if not block or not block.get("slabs"):
        raise RateMissing(f"professional_tax.{state}", (block or {}).get("source_doc", ""))
    src = _rate_of(block, 0)
    for slab in block["slabs"]:
        lo, hi, tax = slab.get("from"), slab.get("to"), slab.get("monthly_tax")
        if lo is None or tax is None:
            raise RateMissing(f"professional_tax.{state}.slabs", src.source_doc)
        if monthly_salary >= lo and (hi is None or monthly_salary <= hi):
            low = f"above {inr(math.floor(lo))}" if lo != int(lo) else f"{inr(lo)} and above"  # 12000.01 = above 12,000
            band = f"{inr(lo)} – {inr(hi)}" if hi is not None else low
            return float(tax), src, band
    raise RateMissing(f"professional_tax.{state}.slabs (no slab covers {inr(monthly_salary)})", src.source_doc)


def state_name(state: str) -> str:
    return next((s.name for s in load_states() if s.id == state), state)


def professional_tax(state: str, monthly_salary: float, rates: dict[str, Any] | None = None,
                     on: date | None = None) -> ChatResponse:
    """Monthly Professional Tax from the state's slab table valid on `on`."""
    tax, src, band = pt_monthly(state, monthly_salary, rates, on)
    name = state_name(state)
    return ChatResponse(
        type="tool", tool=f"professional_tax(state={state!r}, monthly_salary={monthly_salary:g})",
        answer=f"Professional Tax in {name} on a monthly salary of {inr(monthly_salary)}: {inr(tax)} per month.",
        table=Table(columns=["State", "Slab", "PT per month"], rows=[[name, band, inr(tax)]]),
        citations=_cite(src), note=NOTE,
    )


def gratuity_estimate(last_basic_da: float, years: float, employment_type: str = "permanent",
                      rates: dict[str, Any] | None = None, on: date | None = None) -> ChatResponse:
    """Gratuity = last (basic + DA) × days_per_year / wage_days_divisor × counted years, capped at max_amount."""
    r = rates or load_rates()
    days, divisor = (rate(r, f"gratuity.{k}", on) for k in ("days_per_year", "wage_days_divisor"))
    try:
        cap: Rate | None = rate(r, "gratuity.max_amount", on)
    except RateMissing:
        cap = None  # the ceiling is notified separately; without it the amount is shown uncapped, and says so
    fixed = employment_type.lower().startswith("fixed")
    min_years = rate(r, "gratuity.min_years_fixed_term" if fixed else "gratuity.min_years_permanent", on)
    completed = math.floor(years)
    try:
        round_after = rate(r, "gratuity.round_up_after_months", on)
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
    capped = min(amount, cap.value) if cap else amount
    rows = [
        ["Last basic + DA", "per month", inr(last_basic_da)],
        ["Counted years", f"{years:g} given" + (f", >{round_after.value:g} months rounds up" if round_after else ""), str(completed)],
        ["Formula", f"wage × {days.value:g} / {divisor.value:g} × years", inr(amount)],
    ]
    if cap and capped < amount:
        rows.append(["Ceiling applied", f"max {inr(cap.value)}", inr(capped)])
    elif not cap:
        rows.append(["Ceiling", "not configured in data/rates.yaml: amount not capped", "check"])
    rows.append(["Gratuity payable", "", inr(capped)])
    return ChatResponse(
        type="tool", tool=tool, answer=f"Estimated gratuity for {completed} counted year(s) at {inr(last_basic_da)} last basic + DA:",
        table=Table(columns=["Item", "Basis", "Amount"], rows=rows, totalRow=True),
        citations=_cite(*used), note=NOTE,
    )


# ----------------------------------------------------------------- parsing the question
CALC_WORDS = re.compile(r"\b(calculate|compute|how much|work out|estimate)\b", re.IGNORECASE)
MONEY = re.compile(r"(?:₹|\brs\.?|\binr)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b|lakhs?\b|lacs?\b|l\b)?", re.IGNORECASE)
YEARS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", re.IGNORECASE)
MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b"), lambda m: (int(m[3]), int(m[2]), int(m[1]))),
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + _MON + r",?\s+(\d{4})\b", re.IGNORECASE),
     lambda m: (int(m[3]), MONTHS[m[2].lower()], int(m[1]))),
    (re.compile(r"\b" + _MON + r"\s+(\d{1,2}),?\s+(\d{4})\b", re.IGNORECASE),
     lambda m: (int(m[3]), MONTHS[m[1].lower()], int(m[2]))),
    (re.compile(r"\b" + _MON + r",?\s+(\d{4})\b", re.IGNORECASE), lambda m: (int(m[2]), MONTHS[m[1].lower()], 1)),
]


def question_date(text: str) -> tuple[date | None, str]:
    """A date written in the question ('on 1 Oct 2026', 'in March 2026', '2026-10-01', '01/10/2026'),
    and the text with that date removed so its year isn't read as an amount."""
    for pattern, parts in DATE_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                return date(*parts(m)), text[: m.start()] + " " + text[m.end() :]
            except ValueError:
                continue
    return None, text


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


def run(question: str, scope: Scope, rates: dict[str, Any] | None = None, today: date | None = None) -> ChatResponse:
    """Pick the calculator from the question, parse its inputs and run it with the rates valid on
    the date in the question, or today."""
    q = question.lower()
    on, rest = question_date(question)
    on = on or today or date.today()
    if deadlines.is_deadline_question(question) and not CALC_WORDS.search(question):
        return deadlines.run(question, scope, rates or load_rates(), on)
    amounts = parse_amounts(rest)
    if not amounts:
        return ChatResponse(type="tool", answer="Please include the monthly amount, e.g. \"Calculate PF for basic ₹18,000\".")
    amount = amounts[0]
    try:
        if "gratuity" in q:
            ym = YEARS.search(question)
            if not ym:
                return ChatResponse(type="tool", answer="How many years of service? e.g. \"gratuity for basic + DA ₹40,000 and 7 years\".")
            etype = "fixed-term" if re.search(r"fixed[- ]?term", q) else "permanent"
            return gratuity_estimate(amount, float(ym.group(1)), etype, rates, on)
        if re.search(r"\b(esic|esi)\b", q):
            return esic_calculator(amount, rates, on)
        if re.search(r"professional tax|\bpt\b", q):
            return professional_tax(scope.primary, amount, rates, on)
        return pf_calculator(amount, rates, on)
    except RateMissing as e:
        src = f" Fill it from {e.source_doc.removeprefix('TODO: ')}." if e.source_doc else ""
        return ChatResponse(
            type="tool", answer=f"Rate not configured: {e.key}. It is empty in data/rates.yaml for {on.isoformat()}, "
                                f"so I won't calculate.{src}",
            note="Add the value with its source_doc and effective_from, then ask again.",
        )
