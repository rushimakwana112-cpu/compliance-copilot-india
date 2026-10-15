"""Due dates for payroll compliance: PF, ESIC, TDS on salary and Professional Tax.

Like the calculators, this is plain Python. Every date comes from the `due_dates`
section of data/rates.yaml with the document it was taken from; nothing is
hard-coded here. Two kinds of entry:

* monthly: `value` is the day of the month after the wage month ("by the 15th of
  the next month"), so the next due date can be worked out from today.
* schedule: `value` maps a label to "MM-DD" (quarterly statements, Form 16).

A null value answers "Due date not configured", never a guess.
"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import date
from typing import Any

from app.jurisdiction import Scope, load_states
from app.schemas import ChatResponse, Citation, Table

NOTE = ("Dates come from data/rates.yaml and the cited document. If a due date falls on a holiday, "
        "check the cited document or the official portal.")

DUE_INTENT = re.compile(
    r"\b(due ?dates?|due|deadlines?|last dates?|by when|when (?:should|must|do|does|to|is|are|will)|"
    r"(?:payment|deposit|submission|filing|return) dates?|date of (?:payment|deposit|submission|filing))\b",
    re.IGNORECASE,
)
TOPICS = [  # (key in due_dates, pattern, label)
    ("form16", re.compile(r"\bform[- ]?(?:16|130)\b", re.I), "Salary TDS certificate (Form 16 / Form 130)"),
    ("tds_quarterly_statement",
     re.compile(r"\b(24q|form[- ]?138|tds (?:return|statement)s?|quarterly (?:tds )?(?:return|statement))\b", re.I),
     "Quarterly TDS statement (salary)"),
    ("tds_salary_deposit", re.compile(r"\b(tds|tax deducted at source)\b", re.I), "TDS on salary: deposit"),
    ("esic", re.compile(r"\b(esic|esi)\b", re.I), "ESIC contributions"),
    ("pf", re.compile(r"\b(pf|epf|provident fund|ecr|epfo)\b", re.I), "PF contributions and ECR"),
    ("professional_tax", re.compile(r"\b(professional tax|pt)\b", re.I), "Professional Tax"),
]
MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December")


def is_deadline_question(text: str) -> bool:
    """'When is PF due?', 'TDS deposit due date', 'last date for PT payment in Gujarat'."""
    return bool(DUE_INTENT.search(text)) and any(p.search(text) for _, p, _ in TOPICS)


def topics(text: str) -> list[tuple[str, str]]:
    """(key, label) for each topic named; TDS deposit is dropped when the question is about 24Q or Form 16."""
    found = [(k, label) for k, p, label in TOPICS if p.search(text)]
    keys = {k for k, _ in found}
    if keys & {"form16", "tds_quarterly_statement"}:
        found = [t for t in found if t[0] != "tds_salary_deposit"]
    return found


def fmt(d: date) -> str:
    return f"{d.day} {MONTH_NAMES[d.month - 1]} {d.year}"


def _add_months(year: int, month: int, n: int) -> tuple[int, int]:
    m = year * 12 + month - 1 + n
    return m // 12, m % 12 + 1


def _due_in(year: int, month: int, day: int) -> date:
    return date(year, month, min(day, monthrange(year, month)[1]))


def next_monthly_due(day: int, offset: int, today: date) -> tuple[date, str]:
    """The next due date on or after `today`, and the wage month it is for."""
    for back in (offset, offset - 1):
        wy, wm = _add_months(today.year, today.month, -back)
        dy, dm = _add_months(wy, wm, offset)
        due = _due_in(dy, dm, day)
        if due >= today:
            return due, f"{MONTH_NAMES[wm - 1]} {wy}"
    raise AssertionError("unreachable")  # pragma: no cover


def next_schedule_due(schedule: dict[str, str], today: date) -> tuple[date, str]:
    """The next "MM-DD" date in the schedule on or after `today` (this year, else next year)."""
    upcoming = []
    for label, mmdd in schedule.items():
        m, d = (int(x) for x in str(mmdd).split("-"))
        due = date(today.year, m, d)
        upcoming.append((due if due >= today else date(today.year + 1, m, d), label))
    return min(upcoming)


def _source(v: dict[str, Any]) -> str:
    parts = [v.get("source_doc") or "data/rates.yaml"]
    if v.get("section"):
        parts.append(str(v["section"]))
    return ", ".join(parts)


def _cite(v: dict[str, Any]) -> Citation:
    return Citation(doc=v.get("source_doc") or "data/rates.yaml", section=str(v.get("section") or ""),
                    snippet=f"Effective {v['effective_from']}" if v.get("effective_from") else "", url=v.get("url") or "")


def run(question: str, scope: Scope, rates: dict[str, Any], today: date) -> ChatResponse:
    """Answer a due-date question from rates.yaml's `due_dates`."""
    from app.tools import pick  # the same dated-version lookup the calculators use

    table = (rates or {}).get("due_dates") or {}
    rows, cites, missing, notes = [], [], [], []
    for key, label in topics(question) or [("pf", "PF contributions and ECR")]:
        node = table.get(key)
        if key == "professional_tax":
            state = scope.primary
            node = (node or {}).get(state)
            name = next((s.name for s in load_states() if s.id == state), state)
            label = f"Professional Tax ({name})"
            key = f"professional_tax.{state}"
        v = pick(node, today)
        if v is None or v.get("value") in (None, "", {}):
            missing.append((f"due_dates.{key}", (v or {}).get("source_doc", "")))
            continue
        if isinstance(v["value"], dict):
            due, which = next_schedule_due(v["value"], today)
            rule = "; ".join(f"{k}: {int(str(mmdd).split('-')[1])} {MONTH_NAMES[int(str(mmdd).split('-')[0]) - 1]}"
                             for k, mmdd in v["value"].items())
            nxt = f"{fmt(due)} ({which})"
        elif isinstance(v["value"], str):
            m, d = (int(x) for x in v["value"].split("-"))
            due = date(today.year, m, d) if date(today.year, m, d) >= today else date(today.year + 1, m, d)
            rule, nxt = f"every year by {d} {MONTH_NAMES[m - 1]}", fmt(due)
        else:
            day, offset = int(v["value"]), int(v.get("month_offset") or 1)
            due, wage_month = next_monthly_due(day, offset, today)
            after = "the following month" if offset == 1 else f"{offset} months after the wage month"
            rule, nxt = f"by the {_ordinal(day)} of {after}", f"{fmt(due)} (for {wage_month})"
        rows.append([label, rule, nxt, _source(v)])
        cites.append(_cite(v))
        if v.get("note"):
            notes.append(f"{label}: {v['note']}")
    if missing and not rows:
        key, src = missing[0]
        from app.tools import missing_answer  # imported here: app.tools imports this module

        return missing_answer(key, src, what="due date", tool="due_dates")
    answer = "Due dates as of " + fmt(today) + ":"
    if notes:
        answer += "\n\n" + "\n".join(f"- {n}" for n in notes)
    if missing:
        answer += "\n\nNot configured yet: " + ", ".join(k for k, _ in missing) + "."
    return ChatResponse(type="tool", tool="due_dates", answer=answer,
                        table=Table(columns=["What", "Rule", "Next due", "Source"], rows=rows),
                        citations=cites, note=NOTE)


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
