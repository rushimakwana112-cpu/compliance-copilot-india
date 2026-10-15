"""Bonus, overtime and leave encashment from the employee's own figures.

Statutory figures come from data/rates.yaml (`bonus`, `overtime`, `leave`). Company-policy choices
(the per-day divisor for leave encashment) are asked for, never assumed.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from app import tools
from app.salary import DIVISOR_Q, FOOTER, _ask, _fy, amounts, days_method
from app.schemas import ChatResponse, Table
from app.tools import RateMissing, inr, pct, rate

BONUS = re.compile(r"\b(?:statutory )?bonus\b", re.I)
OVERTIME = re.compile(r"\bover ?time\b|\bot\b(?= (?:pay|hours?|wages?|amount|for|on|calculation))|\bextra hours?\b", re.I)
LEAVE_ENC = re.compile(r"\bleave encash\w*|\bencash\w* (?:of )?(?:my )?(?:\d+ )?(?:days? )?(?:of )?(?:earned |annual |privilege )?leaves?\b"
                       r"|\bel encash\w*", re.I)
HOURS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b", re.I)
DAYS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:days?)\b", re.I)
MONTHS_WORKED = re.compile(r"\b(\d{1,2})\s*months?\b", re.I)
PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _note(on: date, extra: str = "") -> str:
    return f"{extra}Rates from data/rates.yaml as they apply on {on.isoformat()} ({_fy(on)}); rates can change. {FOOTER}"


def wants(text: str) -> bool:
    """A bonus, overtime or leave-encashment calculation: the topic plus an amount."""
    return bool((BONUS.search(text) or OVERTIME.search(text) or LEAVE_ENC.search(text)) and amounts(text))


def run(text: str, rates: dict[str, Any] | None = None, today: date | None = None) -> ChatResponse:
    on = today or date.today()
    r = rates or tools.load_rates()
    try:
        if LEAVE_ENC.search(text):
            return leave_encashment(text, r, on)
        if OVERTIME.search(text):
            return overtime(text, r, on)
        return bonus(text, r, on)
    except RateMissing as e:
        src = f" Fill it from {e.source_doc}." if e.source_doc else ""
        extra = ""
        if e.key.startswith("bonus."):
            mn, mx = rate(r, "bonus.min_rate", on), rate(r, "bonus.max_rate", on)
            extra = (f" The rule: bonus is {pct(mn.value)} to {pct(mx.value)} of the wages (basic + DA) earned in the "
                     f"accounting year ({mn.source_doc}, {mn.section}, {mx.section}), but where wages exceed the notified "
                     "ceiling it is worked out as if wages were that ceiling, which I can't apply yet.")
        return ChatResponse(type="tool", answer=f"Rate not configured: {e.key}. It is empty in data/rates.yaml for "
                                                f"{on.isoformat()}, so I won't calculate.{src}{extra}",
                            note="Add the value with its source_doc and effective_from, then ask again.")


def bonus(text: str, r: dict[str, Any], on: date) -> ChatResponse:
    """Statutory bonus on monthly wages (basic + DA) for the months worked in the accounting year."""
    wage = amounts(text)[0]
    months_m = MONTHS_WORKED.search(text)
    months = int(months_m.group(1)) if months_m else None
    if months is None or not 1 <= months <= 12:
        return _ask(["For how many months of the accounting year did you work (12 for a full year)?"],
                    "To work out your bonus I need:")
    mn, mx, floor = rate(r, "bonus.min_rate", on), rate(r, "bonus.max_rate", on), rate(r, "bonus.min_amount", on)
    asked = PCT.search(text)
    if asked and not mn.value - 1e-9 <= float(asked.group(1)) / 100 <= mx.value + 1e-9:
        return ChatResponse(type="guide", answer=f"I can't use {asked.group(0)}: statutory bonus is between {pct(mn.value)} "
                                                 f"and {pct(mx.value)} of wages ({mn.source_doc}, {mn.section}).",
                            note="Invalid input.")
    ceiling = rate(r, "bonus.calculation_ceiling", on)
    counted = min(wage, ceiling.value)
    earned = counted * months
    rows = [["Wages per month (basic + DA)", "", inr(wage)]]
    if counted < wage:
        rows.append(["Wages counted for bonus", f"capped at {inr(ceiling.value)} ({ceiling.source_doc})", inr(counted)])
    rows.append(["Wages for the year", f"{inr(counted)} × {months} months", inr(earned)])
    low, high = max(earned * mn.value, floor.value), earned * mx.value
    rows += [["Minimum bonus", f"{pct(mn.value)} of wages, at least {inr(floor.value)}", inr(low)],
             ["Maximum bonus", f"{pct(mx.value)} of wages", inr(high)]]
    answer = f"Statutory bonus for the year: at least **{inr(low)}**, at most **{inr(high)}**."
    if asked:
        p = float(asked.group(1)) / 100
        rows.append([f"Bonus at {pct(p)}", f"{pct(p)} of {inr(earned)}", inr(earned * p)])
        answer = f"Bonus at {pct(p)}: **{inr(earned * p)}** for the year (the law allows {inr(low)} to {inr(high)})."
    days = rate(r, "bonus.min_days_worked", on)
    return ChatResponse(type="tool", tool="bonus", answer=answer + f" You must have worked at least {days.value:g} days in the year.",
                        table=Table(columns=["Item", "Basis", "Amount"], rows=rows),
                        citations=tools._cite(mn, mx, ceiling, days),
                        note=_note(on, "Between the minimum and maximum, the rate depends on the employer's allocable surplus. "))


def overtime(text: str, r: dict[str, Any], on: date) -> ChatResponse:
    money, hours = amounts(text), HOURS.search(text)
    asks = []
    if not hours:
        asks.append("How many overtime hours?")
    if asks:
        return _ask(asks, "To work out overtime pay I need:")
    mult, per_day, days = rate(r, "overtime.rate_multiplier", on), rate(r, "overtime.hours_per_day", on), \
        rate(r, "overtime.days_per_month", on)
    wage, h = money[0], float(hours.group(1))  # type: ignore[union-attr]
    hourly = wage / days.value / per_day.value
    pay = hourly * mult.value * h
    rows = [["Monthly wages (basic + DA)", "", inr(wage)],
            ["Daily rate", f"{inr(wage)} ÷ {days.value:g} days", inr(wage / days.value)],
            ["Hourly rate", f"daily ÷ {per_day.value:g} hours", f"₹{hourly:,.2f}"],
            ["Overtime rate", f"{mult.value:g} × hourly", f"₹{hourly * mult.value:,.2f}"],
            ["Overtime pay", f"{h:g} hours × ₹{hourly * mult.value:,.2f}", inr(pay)]]
    return ChatResponse(type="tool", tool="overtime", answer=f"Overtime pay for {h:g} hours: **{inr(pay)}** at {mult.value:g}× the normal rate.",
                        table=Table(columns=["Item", "Basis", "Amount"], rows=rows), citations=tools._cite(mult, days),
                        note=_note(on, "The hourly rate uses the wage rules' conversion (monthly = daily × 26, daily = 8 hours); "
                                       "your company may use a different basis, but overtime must be at least twice the normal rate. "))


def leave_encashment(text: str, r: dict[str, Any], on: date) -> ChatResponse:
    money, method = amounts(text), days_method(text)
    leave_days = next((float(m.group(1)) for m in DAYS.finditer(text)
                       if m.group(1) not in ("26", "30") or not re.match(r"\s*days?\s*(?:basis|method|divisor)", text[m.end(1):])), None)
    if method and leave_days is not None and str(int(leave_days)) == method and len(DAYS.findall(text)) == 1:
        leave_days = None  # "30 days" was the divisor, not the leave
    asks = []
    if leave_days is None:
        asks.append("How many days of leave are being encashed?")
    if method is None:
        asks.append(DIVISOR_Q.replace("per-day pay", "the per-day rate for leave encashment"))
    elif method == "calendar":
        asks.append("Is it ÷ 26 or ÷ 30 days? (Calendar days need the month; most companies use 26 or 30 for encashment.)")
    if asks:
        return _ask(asks, "To work out leave encashment I need:")
    wage, divisor = money[0], int(method or 30)
    per_day = wage / divisor
    pay = per_day * (leave_days or 0)
    cap = rate(r, "leave.carry_forward_max", on)
    rows = [["Monthly wages (basic + DA)", "", inr(wage)], ["Per-day rate", f"{inr(wage)} ÷ {divisor} days", inr(per_day)],
            ["Leave encashment", f"{inr(per_day)} × {leave_days:g} days", inr(pay)]]
    tax_note = ("Encashment while in service is taxable as salary. On retirement or leaving, part of it is exempt "
                "up to a limit notified under the Income-tax Act, 2025 (not yet configured in data/rates.yaml).")
    return ChatResponse(type="tool", tool="leave_encashment",
                        answer=f"Leave encashment for {leave_days:g} days: **{inr(pay)}**.\n\n- {tax_note}\n"
                               f"- Unused leave above {cap.value:g} days can't be carried forward and must be encashed "
                               f"({cap.source_doc}, {cap.section}).",
                        table=Table(columns=["Item", "Basis", "Amount"], rows=rows), citations=tools._cite(cap),
                        note=_note(on, "The divisor (26 or 30 days) is company policy. "))
