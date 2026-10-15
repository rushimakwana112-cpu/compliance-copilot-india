"""Salary calculations from figures the employee types: CTC to in-hand, per-day pay, loss of pay,
first and last month's salary, and increments with arrears.

The bot has no employee data, so everything here works on the numbers in the conversation.
Statutory figures (PF, ESIC, Professional Tax) come from data/rates.yaml through app.tools.
Company-policy choices (26 / 30 / calendar days, basic as a share of CTC, whether employer PF
is part of CTC) are never assumed: when one is missing, the answer is a short question asking
for it, and the calculation runs once the reply arrives. A reply can answer by number
("1. yearly 2. 40% 3. yes, capped, included").
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app import tools
from app.jurisdiction import detect_states
from app.schemas import ChatResponse, Table, Turn
from app.tools import RateMissing, inr, pct

FOOTER = "Check with HR/payroll for your actual records."


def _fy(on: date) -> str:
    start = on.year if on.month >= 4 else on.year - 1
    return f"FY {start}-{str(start + 1)[2:]}"


def _note(on: date) -> str:
    return (f"Statutory rates from data/rates.yaml as they apply on {on.isoformat()} ({_fy(on)}); rates can change. "
            f"Arithmetic is done in Python, not by the model. {FOOTER}")


def _ask(lines: list[str], intro: str) -> ChatResponse:
    body = intro + "\n\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(lines, 1)) if len(lines) > 1 else f"{intro} {lines[0]}"
    return ChatResponse(type="guide", answer=body, note="Waiting for the details needed to calculate accurately.")


# ----------------------------------------------------------------- parsing helpers
NUM = r"(\d[\d,]*(?:\.\d+)?)\s*(k|lakhs?|lacs?|l|lpa|cr)?\b"


def _amount(num: str, unit: str | None) -> float:
    n = float(num.replace(",", ""))
    u = (unit or "").lower()
    if u == "k":
        n *= 1_000
    elif u.startswith(("lakh", "lac")) or u in ("l", "lpa"):
        n *= 100_000
    elif u == "cr":
        n *= 10_000_000
    return n


def amounts(text: str) -> list[float]:
    out = []
    for m in re.finditer(r"(?:₹|\brs\.?|\binr)?\s*" + NUM, text, re.IGNORECASE):
        tail = text[m.end(): m.end() + 12].lower()
        if re.match(r"\s*(?:%|percent|days?|months?|years?|yrs?|hours?|hrs?|st|nd|rd|th)\b", tail) or m.group(0).strip().endswith("%"):
            continue
        n = _amount(m.group(1), m.group(2))
        if n >= 100:
            out.append(n)
    return out


MONTHLY = re.compile(r"\b(?:per month|a month|monthly|/\s*month|p\.?\s?m\.?|pm|every month|mahina|month)\b", re.IGNORECASE)
YEARLY = re.compile(r"\b(?:per annum|p\.?\s?a\.?|pa|yearly|annual(?:ly)?|a year|per year|/\s*year|lpa|annum|year)\b|\d\s*lpa\b",
                    re.IGNORECASE)


def period(text: str) -> str | None:
    """'monthly', 'yearly' or None when the text doesn't say (or says both)."""
    m, y = bool(MONTHLY.search(text)), bool(YEARLY.search(text))
    return "monthly" if m and not y else "yearly" if y and not m else None


DAYS_METHOD = re.compile(r"\b(26|30)\s*(?:-\s*)?days?\b(?!\s+(?:of|unpaid|leave|lop|absent|worked|working))"
                         r"|\b(calendar|actual)\s+days?\b|\bdays in (?:the|that) month\b", re.IGNORECASE)


def days_method(text: str) -> str | None:
    """'26', '30' or 'calendar' when the text names the company's divisor."""
    for m in DAYS_METHOD.finditer(text):
        if m.group(1):
            return m.group(1)
        return "calendar"
    return None


MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})


def month_in(text: str, today: date) -> tuple[int, int] | None:
    """(year, month) named in the text ('March', 'Mar 2026', 'this month', 'last month')."""
    t = text.lower()
    if re.search(r"\bthis month\b|\bcurrent month\b", t):
        return today.year, today.month
    if re.search(r"\blast month\b|\bprevious month\b", t) and not re.search(r"last month'?s? salary", t):
        y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
        return y, m
    m = re.search(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b\.?,?\s*(\d{4})?", t)
    if m:
        return int(m.group(2) or today.year), MONTHS[m.group(1)]
    return None


def _yes_no(text: str, subject: str) -> bool | None:
    """True/False when the text says the subject applies or not; None when it doesn't say."""
    t = text.lower()
    if re.search(rf"\b(?:no|not|without|don'?t have|doesn'?t apply|n/?a)\b[^.,;\n]{{0,15}}\b{subject}\b|\b{subject}\b[^.,;\n]{{0,15}}\b(?:no|not applicable|not deducted|n/?a|doesn'?t apply|nahi)\b", t):
        return False
    if re.search(rf"\b{subject}\b[^.,;\n]{{0,15}}\b(?:yes|applicable|applies|deducted|is there|haan)\b|\b(?:with|yes)\b[^.,;\n]{{0,6}}\b{subject}\b", t):
        return True
    return None


UNKNOWN = re.compile(r"\b(?:don'?t know|not sure|no idea|unsure|pata nahi|maloom nahi)\b", re.IGNORECASE)


# ----------------------------------------------------------------- CTC → in-hand
CTC_INTENT = re.compile(r"\bctc\b|\bcost to company\b|\bpackage\b|\blpa\b", re.IGNORECASE)
INHAND_INTENT = re.compile(r"\bin-hand\b|\bin hand\b|\btake[- ]?home\b|\bnet (?:salary|pay)\b|\bhow much will i get\b|\bmonthly salary\b"
                           r"|\bsalary (?:will be|would be)\b|\bwhat will (?:be )?my salary\b", re.IGNORECASE)
CTC_MARK = "To work out your in-hand salary"

QUESTIONS = {
    "kind": "Is that amount your CTC (cost to company) or your gross salary?",
    "period": "Is your CTC monthly or yearly?",
    "basic": "What is your basic salary (amount, or % of CTC)?",
    "pf": "Is PF applicable to you? If yes: is it calculated on your full basic or capped at the PF wage ceiling "
          "({ceiling}), and is the employer's PF contribution included in your CTC?",
    "esic": "Is ESIC applicable to you? (Only possible if your gross salary is {esic}/month or less.)",
    "state": "Which state do you work in? (For Professional Tax; I'll assume Gujarat if you don't say.)",
    "regime": "Optional: which tax regime are you on (old or new), if you want TDS included?",
}


@dataclass
class CTC:
    amount: float | None = None
    period: str | None = None
    basic: float | None = None          # amount, in the same period as the CTC
    basic_pct: float | None = None      # fraction of CTC
    pf: bool | None = None
    pf_known: bool = False              # the user answered the PF question (True/False or "don't know")
    pf_full: bool | None = None         # True: on full basic, False: capped at the ceiling
    er_in_ctc: bool | None = None
    esic: bool | None = None
    esic_known: bool = False
    state: str | None = None
    regime: str | None = None
    kind: str | None = None             # "ctc" or "gross" (the amount typed)
    asked: list[str] = field(default_factory=list)


def is_ctc_question(text: str) -> bool:
    """'My CTC is 6 LPA, in-hand?' or 'salary 25k in-hand?' (an amount plus in-hand)."""
    if CTC_INTENT.search(text):
        return bool(INHAND_INTENT.search(text) or re.search(r"\bsalary\b|\bmonthly\b", text, re.I))
    return bool(INHAND_INTENT.search(text) and amounts(text))


GROSS = re.compile(r"\bgross\b", re.IGNORECASE)


def _kind(text: str) -> str | None:
    g, c = bool(GROSS.search(text)), bool(CTC_INTENT.search(text) or re.search(r"cost to company", text, re.I))
    return "gross" if g and not c else "ctc" if c and not g else None


def ctc_conversation(text: str, history: list[Turn]) -> list[str] | None:
    """The user messages of a CTC → in-hand conversation that this message continues or starts, or None."""
    if is_ctc_question(text):
        return [text]
    last = next((t for t in reversed(history) if t.role == "assistant"), None)
    if last is None or CTC_MARK not in last.content:
        return None
    users: list[str] = []
    for t in reversed(history):  # back to the user message that started it
        if t.role == "user":
            users.insert(0, t.content)
            if is_ctc_question(t.content):
                break
    return [*users, text]


def _asked(history: list[Turn]) -> list[str]:
    """Question keys in the order the last CTC question message listed them."""
    last = next((t for t in reversed(history) if t.role == "assistant"), None)
    if last is None or CTC_MARK not in last.content:
        return []
    keys = []
    for line in last.content.splitlines():
        m = re.match(r"\s*\d+\.\s*(.*)", line)
        if m:
            keys += [k for k, q in QUESTIONS.items() if m.group(1).startswith(q.split("(")[0].split("?")[0][:25])]
    return keys


def _numbered(text: str) -> dict[int, str]:
    parts = re.split(r"(?:^|\s|,|;)(\d)\s*[.)\-:]\s+", " " + text)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def _parse_pf(part: str, c: CTC) -> None:
    t = part.lower()
    if UNKNOWN.search(t):  # "not sure about PF": both results are shown
        c.pf_known = True
        return
    yn = _yes_no(t, "pf")
    if yn is None and re.match(r"\s*(?:yes|haan|y)\b", t):
        yn = True
    if yn is None and re.match(r"\s*(?:no|nahi|n)\b", t):
        yn = False
    if re.search(r"\bfull\b|\bactual basic\b|\buncapped\b|\bwhole basic\b|\bentire basic\b", t):
        c.pf_full = True
    elif re.search(r"\bcap(?:ped)?\b|\bceiling\b|\blimit\b|\brestricted\b|15,?000|25,?000|\bstatutory\b", t):
        c.pf_full = False
    if re.search(r"\b(?:not included|excluded|not part of|over and above|on top|separate|outside)\b", t):
        c.er_in_ctc = False
    elif re.search(r"\bincluded\b|\bpart of (?:the )?ctc\b|\bin (?:my |the )?ctc\b|\binclusive\b", t):
        c.er_in_ctc = True
    if yn is None and (c.pf_full is not None or c.er_in_ctc is not None):
        yn = True  # "PF on full basic, included" means PF applies
    if yn is not None:
        c.pf, c.pf_known = yn, True


def _parse_esic(part: str, c: CTC) -> None:
    t = part.lower()
    if UNKNOWN.search(t):
        c.esic_known = True
        return
    yn = _yes_no(t, "esic")
    if yn is None and re.match(r"\s*(?:yes|haan|y)\b", t):
        yn = True
    if yn is None and re.match(r"\s*(?:no|nahi|n)\b", t):
        yn = False
    if yn is not None:
        c.esic, c.esic_known = yn, True


def _parse_basic(part: str, c: CTC, whole: bool) -> None:
    t = part.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", t) if not whole else \
        re.search(r"basic[^%\d]{0,25}(\d+(?:\.\d+)?)\s*%|(\d+(?:\.\d+)?)\s*%[^.\d]{0,20}basic", t)
    if m:
        c.basic_pct = float(next(g for g in m.groups() if g)) / 100
        return
    m = re.search(r"basic[^\d₹]{0,20}(?:₹|rs\.?|inr)?\s*" + NUM, t) if whole else re.search(r"(?:₹|rs\.?|inr)?\s*" + NUM, t)
    if m:
        c.basic = _amount(m.group(1), m.group(2))


CLAUSE = re.compile(r"[,;\n]|\.\s|\s(?:and|but|&)\s", re.IGNORECASE)


def parse_ctc(messages: list[str], history: list[Turn]) -> CTC:
    c = CTC()
    first = messages[0]
    m = re.search(r"\bctc[^\d₹]{0,25}(?:₹|rs\.?|inr)?\s*" + NUM, first, re.I) or \
        re.search(r"(?:₹|rs\.?|inr)?\s*" + NUM + r"[^.\d]{0,12}\b(?:ctc|package)\b", first, re.I) or \
        re.search(r"(?:₹|rs\.?|inr)?\s*" + NUM, first, re.I)
    if m:
        c.amount = _amount(m.group(1), m.group(2))
        if (m.group(2) or "").lower() == "lpa":
            c.period = "yearly"
    c.kind = _kind(first)
    asked = _asked(history)
    for i, msg in enumerate(messages):
        parts = _numbered(msg) if i == len(messages) - 1 and asked else {}
        if parts:
            for n, part in parts.items():
                if 1 <= n <= len(asked):
                    key = asked[n - 1]
                    if key == "kind":
                        c.kind = _kind(part) or c.kind
                    elif key == "period":
                        c.period = period(part) or c.period
                    elif key == "basic":
                        _parse_basic(part, c, whole=False)
                    elif key == "pf":
                        _parse_pf(part, c)
                    elif key == "esic":
                        _parse_esic(part, c)
                    elif key == "state":
                        st = detect_states(part)
                        c.state = st[0][0] if st else c.state
                    elif key == "regime":
                        r = re.search(r"\b(old|new)\b", part, re.I)
                        c.regime = r.group(1).lower() if r else c.regime
            continue
        rest = msg if i else re.sub(r"\bctc\b[^\d₹]{0,25}(?:₹|rs\.?|inr)?\s*" + NUM, " ", msg, count=1, flags=re.I)
        c.period = period(msg if i else first) or c.period
        if re.search(r"\bbasic\b", msg, re.I):
            _parse_basic(msg, c, whole=True)
        for clause in CLAUSE.split(msg):  # "PF capped, not sure about ESIC": each part to its subject
            if re.search(r"\bpf\b|\bprovident\b", clause, re.I) or (i and len(asked) == 1 and asked[0] == "pf"):
                _parse_pf(clause, c)
            if re.search(r"\besic?\b", clause, re.I) or (i and len(asked) == 1 and asked[0] == "esic"):
                _parse_esic(clause, c)
        if i and len(asked) == 1 and asked[0] == "basic" and c.basic is None and c.basic_pct is None:
            _parse_basic(rest, c, whole=False)
        if i and c.kind is None:
            c.kind = _kind(msg)
        if i and len(asked) == 1 and asked[0] == "period":
            c.period = period(msg) or c.period
        st = detect_states(msg)
        if st:
            c.state = st[0][0]
        r = re.search(r"\b(old|new)\s+(?:tax\s+)?regime\b", msg, re.I)
        if r:
            c.regime = r.group(1).lower()
    c.asked = asked
    return c


def ctc_in_hand(messages: list[str], history: list[Turn], default_state: str, rates: dict[str, Any] | None = None,
                today: date | None = None) -> ChatResponse:
    today = today or date.today()
    r = rates or tools.load_rates()
    c = parse_ctc(messages, history)
    try:
        ceiling = tools.rate(r, "pf.wage_ceiling", today)
        esic_ceiling = tools.rate(r, "esic.wage_ceiling", today)
    except RateMissing as e:
        return _missing(e, today)
    if c.amount is None:
        return _ask(["What is your CTC (amount), and is it monthly or yearly?"], "To work out your in-hand salary from your CTC, I need one detail:")
    # Ask only for what is missing. ESIC is asked only when gross could be within the ESIC ceiling.
    missing = []
    if c.kind is None:
        missing.append("kind")
    if c.kind == "gross":
        c.er_in_ctc = False  # gross salary never contains the employer's contributions
    if c.period is None:
        missing.append("period")
    if c.basic is None and c.basic_pct is None:
        missing.append("basic")
    if not c.pf_known or (c.pf and (c.pf_full is None or c.er_in_ctc is None)):
        missing.append("pf")
    monthly = c.amount / 12 if c.period == "yearly" else c.amount
    if not c.esic_known and (c.period is None or monthly * (1 - 0.13) <= esic_ceiling.value):
        missing.append("esic")
    first_round = not c.asked
    if first_round and c.state is None:
        missing.append("state")
    if first_round and c.regime is None and missing:
        missing.append("regime")
    if any(k in missing for k in ("kind", "period", "basic", "pf", "esic")):
        qs = [QUESTIONS[k].format(ceiling=inr(ceiling.value), esic=inr(esic_ceiling.value)) for k in missing]
        of = {"ctc": "your CTC", "gross": "your gross salary"}.get(c.kind or "", "")
        of = f"from {of} of {inr(c.amount)}" if of else f"from {inr(c.amount)}"
        intro = f"{CTC_MARK} {of}, I need a few details (answer in one message, by number if you like):"
        if len(qs) == 1:
            intro = f"{CTC_MARK} {of}, I need one more detail:"
            return ChatResponse(type="guide", answer=f"{intro}\n\n1. {qs[0]}", note="Waiting for the details needed to calculate accurately.")
        return _ask(qs, intro)
    state = c.state or default_state or "gujarat"
    try:
        return _compute(c, monthly, state, r, today)
    except RateMissing as e:
        return _missing(e, today)


def _scenario(c: CTC, monthly: float, state: str, r: dict[str, Any], on: date, pf: bool, esic: bool | None,
              pf_full: bool, er_in_ctc: bool) -> tuple[list[list[str]], float, list[Any], list[str]]:
    ceiling = tools.rate(r, "pf.wage_ceiling", on)
    emp_pf, er_pf = tools.rate(r, "pf.employee_rate", on), tools.rate(r, "pf.employer_rate", on)
    esic_ceiling = tools.rate(r, "esic.wage_ceiling", on)
    basic = c.basic / 12 if c.basic is not None and c.period == "yearly" else c.basic
    basic = basic if basic is not None else monthly * (c.basic_pct or 0)
    used: list[Any] = [ceiling, emp_pf, er_pf]
    notes: list[str] = []
    gross_given = c.kind == "gross"
    rows = [["Gross salary per month" if gross_given else "CTC per month", "", inr(monthly)]]
    pf_wage = (basic if pf_full else min(basic, ceiling.value)) if pf else 0.0
    er_amt = pf_wage * er_pf.value if pf and er_in_ctc else 0.0
    if pf:
        rows.append(["Basic", "", inr(basic)])
    if er_amt:
        rows.append(["− Employer PF (in CTC)", f"{pct(er_pf.value)} of {inr(pf_wage)}", inr(er_amt)])
    gross = monthly - er_amt
    er_esic_amt = 0.0
    if esic is not False:
        er_esic, emp_esic = tools.rate(r, "esic.employer_rate", on), tools.rate(r, "esic.employee_rate", on)
        candidate = gross if gross_given else gross / (1 + er_esic.value)
        if candidate <= esic_ceiling.value:
            gross, er_esic_amt = candidate, candidate * er_esic.value
            if not gross_given:
                rows.append(["− Employer ESIC (in CTC)", f"{pct(er_esic.value)} of gross", inr(er_esic_amt)])
            used += [esic_ceiling, er_esic, emp_esic]
        else:
            esic = False
            notes.append(f"ESIC does not apply: gross is above {inr(esic_ceiling.value)}/month.")
            used.append(esic_ceiling)
    if not gross_given:
        rows.append(["= Gross salary", "", inr(gross)])
    emp_amt = pf_wage * emp_pf.value if pf else 0.0
    if pf:
        basis = "full basic" if pf_full else f"basic up to {inr(ceiling.value)}"
        rows.append(["− Employee PF", f"{pct(emp_pf.value)} of {inr(pf_wage)} ({basis})", inr(emp_amt)])
    esic_emp_amt = 0.0
    if esic is not False and er_esic_amt:
        emp_esic = tools.rate(r, "esic.employee_rate", on)
        esic_emp_amt = gross * emp_esic.value
        rows.append(["− Employee ESIC", f"{pct(emp_esic.value)} of gross", inr(esic_emp_amt)])
    pt, pt_src, band = tools.pt_monthly(state, gross, r, on)
    used.append(pt_src)
    rows.append([f"− Professional Tax ({tools.state_name(state)})", band, inr(pt)])
    in_hand = round(gross - emp_amt - esic_emp_amt - pt)
    rows.append(["= In-hand per month (before TDS)", "", inr(in_hand)])
    if not er_in_ctc and pf and not gross_given:
        notes.append("Employer PF is paid on top of your CTC, so it isn't deducted from it.")
    if gross_given and (pf or er_esic_amt):
        notes.append("The employer's PF and ESIC are paid on top of gross salary, so they don't reduce your in-hand.")
    return rows, in_hand, used, notes


def _compute(c: CTC, monthly: float, state: str, r: dict[str, Any], on: date) -> ChatResponse:
    scenarios = []
    pf_options = [c.pf] if c.pf is not None else [True, False]
    # ESIC unknown to the user: show with and without. Not asked (gross above the ceiling): worked out from gross.
    esic_unsure = c.esic is None and c.esic_known
    esic_options = [c.esic] if c.esic is not None else [None, False] if esic_unsure else [None]
    for pf in pf_options:
        for esic in esic_options:
            pf_full = bool(c.pf_full) if c.pf_full is not None else False
            er_in = c.er_in_ctc if c.er_in_ctc is not None else True
            rows, in_hand, used, notes = _scenario(c, monthly, state, r, on, bool(pf), esic, pf_full, er_in)
            label = " · ".join(x for x in (
                ("With PF" if pf else "Without PF") if c.pf is None else "",
                ("With ESIC" if esic is not False else "Without ESIC") if esic_unsure else "") if x)
            scenarios.append((label, rows, in_hand, used, notes))
    # Two ESIC scenarios that came out the same (gross above the ceiling) are one.
    seen, unique = set(), []
    for s in scenarios:
        key = tuple(map(tuple, s[1]))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    first = unique[0]
    what = "gross salary" if c.kind == "gross" else "CTC"
    lines = [f"In-hand salary from a {what} of {inr(c.amount or 0)} {'a year' if c.period == 'yearly' else 'a month'}:"]
    rows: list[list[str]] = []
    for label, srows, in_hand, _, notes in unique:
        if len(unique) > 1:
            rows.append([f"**{label}**", "", ""])
            lines.append(f"- {label}: **{inr(in_hand)}** a month ({inr(in_hand * 12)} a year)")
        rows += srows
    if len(unique) == 1:
        lines[0] += f" **{inr(first[2])} a month** ({inr(first[2] * 12)} a year), before income-tax TDS."
    else:
        lines.append("You didn't know whether " + " and ".join(
            x for x in ("PF" if c.pf is None else "", "ESIC" if esic_unsure else "") if x) + " applies, so both results are shown.")
    extra = sorted({n for s in unique for n in s[4]})
    if c.pf is None and (c.pf_full is None or c.er_in_ctc is None):
        assumed = [f"on basic up to the PF wage ceiling of {inr(tools.rate(r, 'pf.wage_ceiling', on).value)}" if c.pf_full is None else "",
                   "with the employer's share inside the CTC" if c.er_in_ctc is None and c.kind != "gross" else ""]
        extra.append("The \"with PF\" result assumes PF " + " and ".join(a for a in assumed if a) + ".")
    extra.append("TDS (income tax) is not included." + (" Ask \"tax on ₹X under the new regime\" for it." if not c.regime else
                                                        f" For TDS under the {c.regime} regime, ask \"tax on {inr((first[2] + 0) * 12)} under the {c.regime} regime\"."))
    extra.append("If your CTC also includes gratuity, EDLI/admin charges, insurance or a variable bonus, your monthly gross "
                 "and in-hand will be lower than shown.")
    used = [u for s in unique for u in s[3]]
    return ChatResponse(type="tool", tool="ctc_in_hand", answer="\n".join(lines) + "\n\n" + "\n".join(f"- {n}" for n in extra),
                        table=Table(columns=["Step", "Basis", "Amount / month"], rows=rows),
                        citations=tools._cite(*used), note=_note(on))


def _missing(e: RateMissing, on: date) -> ChatResponse:
    src = f" Fill it from {e.source_doc}." if e.source_doc else ""
    return ChatResponse(type="tool", answer=f"Rate not configured: {e.key}. It is empty in data/rates.yaml for "
                                            f"{on.isoformat()}, so I won't calculate.{src}",
                        note="Add the value with its source_doc and effective_from, then ask again.")


# ----------------------------------------------------------------- per-day, loss of pay, first/last month
PER_DAY = re.compile(r"\bper[- ]?day\b|\bdaily (?:salary|wage|rate)\b|\bone day(?:'s)? (?:salary|pay)\b|\bday'?s salary\b", re.I)
LOP = re.compile(r"\bunpaid leave\b|\bleave without pay\b|\blwp\b|\bloss of pay\b|\babsent\b|\bdays? (?:deducted|cut)\b"
                 r"|\bdeduct(?:ed|ion)? for \d+ days?\b", re.I)
JOINING = re.compile(r"\bjoin(?:ed|ing)?\b[^.?]{0,25}\b(?:on|from)\b|\bfirst month(?:'s)? salary\b|\bjoined (?:mid|in the middle)\b", re.I)
LEAVING = re.compile(r"\blast working day\b|\brelieved on\b|\bleaving on\b|\bresignation\b[^.?]{0,40}\b(?:salary|month)\b"
                     r"|\blast month(?:'s)? salary\b|\bexit(?:ing)? on\b|\bmid-?month resignation\b", re.I)
DAY_OF_MONTH = re.compile(r"\b(?:on|from|till|until|upto|up to|last working day(?: is| was| on|:)?)\s+(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\b"
                          r"|\b(\d{1,2})(?:st|nd|rd|th)\s+(?:of\s+)?(?:the\s+)?(?:month|" + "|".join(calendar.month_name[1:]) + r")", re.I)
LEAVE_DAYS = re.compile(r"\b(\d+(?:\.5)?)\s*(?:days?)\b(?=[^.?]{0,25}\b(?:unpaid|leave|lwp|lop|absent|off|without pay|deducted|cut))"
                        r"|\b(?:unpaid|lwp|lop|absent)[^.?\d]{0,20}(\d+(?:\.5)?)\s*days?\b", re.I)
DIVISOR_Q = ("How does your company work out per-day pay: monthly salary ÷ 26 days, ÷ 30 days, or ÷ the calendar days "
             "of that month? It's company policy (see your appointment letter or ask HR).")


def is_day_question(text: str) -> bool:
    return bool(PER_DAY.search(text) or LOP.search(text) or JOINING.search(text) or LEAVING.search(text))


def day_calc(text: str, today: date | None = None) -> ChatResponse:
    today = today or date.today()
    kind = "lop" if LOP.search(text) else "join" if JOINING.search(text) else "leave" if LEAVING.search(text) else "per_day"
    money = amounts(text)
    method = days_method(text)
    when = month_in(text, today)
    asks = []
    if not money:
        asks.append("What is your monthly salary (the amount the deduction is based on, usually gross)?")
    if method is None:
        asks.append(DIVISOR_Q)
    day = None
    if kind in ("join", "leave"):
        m = DAY_OF_MONTH.search(text)
        day = int(next(g for g in m.groups() if g)) if m else None
        if day is None or not 1 <= day <= 31:
            asks.append("On which date of the month did you " + ("join?" if kind == "join" else "leave (last working day)?"))
    leave_days = None
    if kind == "lop":
        m = LEAVE_DAYS.search(text)
        leave_days = float(next(g for g in m.groups() if g)) if m else None
        if leave_days is None:
            asks.append("How many days of unpaid leave?")
    if method == "calendar" and kind != "per_day" and when is None:
        asks.append("Which month (so I know whether it has 28, 29, 30 or 31 days)?")
    if method == "calendar" and kind == "per_day" and when is None:
        asks.append("Which month (so I know whether it has 28, 29, 30 or 31 days)?")
    if asks:
        return _ask(asks, "To calculate this accurately I need:")
    salary = money[0]
    divisor = (calendar.monthrange(*when)[1] if method == "calendar" and when else int(method or 30))
    per_day = salary / divisor
    basis = f"{inr(salary)} ÷ {divisor} days" + (f" ({calendar.month_name[when[1]]} {when[0]})" if method == "calendar" and when else "")
    rows = [["Per-day salary", basis, inr(per_day)]]
    if kind == "per_day":
        answer = f"Per-day salary: **{inr(per_day)}** ({basis})."
    elif kind == "lop":
        assert leave_days is not None
        cut = per_day * leave_days
        rows += [["Deduction", f"{inr(per_day)} × {leave_days:g} days", inr(cut)], ["Salary after deduction", "", inr(salary - cut)]]
        answer = f"{leave_days:g} days of unpaid leave: **{inr(cut)}** deducted, so the month's salary is **{inr(salary - cut)}**."
    else:
        assert day is not None
        month_days = calendar.monthrange(*when)[1] if when else None
        if method == "26":
            return _ask(["How many paid days do you have in that month (working days plus paid weekly offs, as your "
                         "company counts them)? With a 26-day basis, pay = salary ÷ 26 × paid days."],
                        "With a 26-day basis the paid days depend on your company's calendar.")
        top = month_days if method == "calendar" else 30
        if month_days is not None and day > month_days:
            return _guide_invalid(f"{calendar.month_name[when[1]]} has only {month_days} days.")  # type: ignore[index]
        paid = (top - day + 1) if kind == "join" else min(day, top)
        pay = per_day * paid
        what = f"from the {_ord(day)} to the {_ord(top)}" if kind == "join" else f"from the 1st to the {_ord(min(day, top))}"
        rows += [["Paid days", what, str(paid)], ["Salary for the month", f"{inr(per_day)} × {paid}", inr(pay)]]
        answer = (f"{'First' if kind == 'join' else 'Last'} month's salary: **{inr(pay)}** for {paid} paid days. "
                  + ("This is salary only; full and final settlement adds leave encashment, bonus and gratuity if due, "
                     "and recovers any notice-pay shortfall." if kind == "leave" else ""))
    return ChatResponse(type="tool", tool=f"salary_days({kind})", answer=answer,
                        table=Table(columns=["Item", "Basis", "Amount"], rows=rows),
                        note="The per-day divisor (26, 30 or calendar days) is company policy; the law requires the deduction "
                             "for absence to be in proportion to the period absent. " + FOOTER)


def _ord(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _guide_invalid(msg: str) -> ChatResponse:
    return ChatResponse(type="guide", answer=f"I can't calculate that: {msg} Please check the date.", note="Invalid input.")


# ----------------------------------------------------------------- increment and arrears
INCREMENT = re.compile(r"\bincrement\b|\bhike\b|\braise\b|\bappraisal\b|\bincreased by\b", re.I)
PCT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)")
ARREARS = re.compile(r"\barrears?\b", re.I)
ARREAR_MONTHS = re.compile(r"\b(\d{1,2})\s*months?\b", re.I)


def is_increment_question(text: str) -> bool:
    return bool(INCREMENT.search(text) and (PCT.search(text) or ARREARS.search(text)))


def increment(text: str) -> ChatResponse:
    p = PCT.search(text)
    money = amounts(text)
    per = period(text)
    months = ARREAR_MONTHS.search(text)
    asks = []
    if not p:
        asks.append("What is the increment percentage?")
    if not money:
        asks.append("What is the salary the increment applies to (amount)?")
    if per is None:
        asks.append("Is that salary monthly or yearly?")
    if ARREARS.search(text) and not months:
        asks.append("For how many months are arrears due (from the effective date of the increment to the month it is paid)?")
    if asks:
        return _ask(asks, "To work out your new salary I need:")
    rate_ = float(p.group(1)) / 100  # type: ignore[union-attr]
    old = money[0]
    monthly_old = old / 12 if per == "yearly" else old
    new_m = monthly_old * (1 + rate_)
    diff = new_m - monthly_old
    rows = [["Old salary", "per month", inr(monthly_old)], ["Increment", f"{rate_ * 100:g}% of {inr(monthly_old)}", inr(diff)],
            ["New salary", "per month", inr(new_m)], ["New salary", "per year", inr(new_m * 12)]]
    answer = f"New salary: **{inr(new_m)} a month** ({inr(new_m * 12)} a year), up {inr(diff)} a month."
    if months:
        n = int(months.group(1))
        rows.append(["Arrears", f"{inr(diff)} × {n} months", inr(diff * n)])
        answer += f" Arrears for {n} months: **{inr(diff * n)}**."
    return ChatResponse(type="tool", tool="increment", answer=answer,
                        table=Table(columns=["Item", "Basis", "Amount"], rows=rows),
                        note="Arrears are the monthly increase × months it was due but not paid. PF and other deductions on "
                             "the arrears depend on which components were increased. " + FOOTER)


# ----------------------------------------------------------------- entry points
WORK_OUT = re.compile(r"\b(?:calculate|calculation|compute|how much|what (?:will|would) (?:be|i get)|work out|deduction (?:on|for))\b", re.I)


def wants(text: str, history: list[Turn]) -> bool:
    """True when this message is (or continues) a salary calculation. A day-based question needs an amount
    or a calculation word, so "Can my employer cut pay if I'm absent?" still goes to the law search."""
    if ctc_conversation(text, history) or is_increment_question(text):
        return True
    return is_day_question(text) and bool(amounts(text) or WORK_OUT.search(text) or PER_DAY.search(text))


def run(text: str, history: list[Turn], default_state: str, rates: dict[str, Any] | None = None,
        today: date | None = None) -> ChatResponse | None:
    convo = ctc_conversation(text, history)
    if convo:
        return ctc_in_hand(convo, history, default_state, rates, today)
    if is_increment_question(text):
        return increment(text)
    if wants(text, history):
        return day_calc(text, today)
    return None
