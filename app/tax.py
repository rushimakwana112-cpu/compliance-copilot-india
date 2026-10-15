"""Income tax on salary: old vs new regime, HRA exemption and monthly TDS.

Every slab, limit and rate comes from the `income_tax` section of data/rates.yaml (Income-tax Act,
2025 and Income-tax Rules, 2026). The arithmetic is plain Python. Inputs come only from the question:

* "Tax on ₹12 lakh under the new regime"
* "Old vs new regime for 15 LPA with 80C 1.5 lakh and 80D 18000"
* "Which regime is better: salary 1 lakh per month, basic 50000, HRA 20000, rent 24000 per month in Pune"

Missing figures that change the answer (monthly or yearly, the HRA details) are asked for, never assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app import deadlines, tools
from app.salary import FOOTER, NUM, _amount, _ask, _fy, period
from app.schemas import ChatResponse, Table
from app.tools import Rate, RateMissing, inr, pct, pick, rate

TAX_INTENT = re.compile(
    r"\b(?:income[- ]?tax|tax on|tax for|tax liability|tax payable|how much tax|tds on (?:my )?salary|monthly tds|"
    r"tds (?:per|every) month|which regime|old (?:vs|or|and|v/s) new|new (?:vs|or|and|v/s) old|tax regime|old regime|new regime|"
    r"calculate (?:my )?(?:income )?tax|tax calculation|tax calculator|hra exemption|exemption on hra|tds calculate)\b",
    re.IGNORECASE,
)
SLAB_Q = re.compile(r"\b(?:slabs?|tax rates?|standard deduction|rebate|87a)\b", re.IGNORECASE)

# (key, pattern for the label before the amount). The amount follows within a few words.
DEDUCTIONS = [
    ("s123", r"80\s*c(?!cd)|section 123|s\.?\s*123|ppf|elss|lic premium"),
    ("s126_parents", r"80\s*d[^.,;\d]{0,25}parents?|parents?'?\s*(?:health|medical|mediclaim)[^.,;\d]{0,15}"),
    ("s126_self", r"80\s*d|section 126|health insurance|mediclaim"),
    ("s124_3", r"80\s*ccd\s*\(?\s*1\s*b\s*\)?|section 124\(3\)|own nps|nps"),
    ("employer_nps", r"employer(?:'s)?\s+nps|80\s*ccd\s*\(?\s*2\s*\)?"),
    ("home_loan", r"home[- ]loan interest|interest on (?:my )?home loan|home loan|housing loan interest|24\s*\(?b\)?"),
    ("basic", r"basic(?: salary| pay)?(?: \+ ?da)?"),
    ("rent", r"rent(?: paid)?"),
    ("hra", r"hra(?! exemption)(?: received)?|house rent allowance"),
]
HRA_KEYS = ("hra", "rent", "basic")
MONTHLY_CLAUSE = re.compile(r"^\s*(?:per month|a month|monthly|/\s*month|pm|p\.m\.|every month)\b", re.IGNORECASE)
YEARLY_CLAUSE = re.compile(r"^\s*(?:per annum|a year|yearly|annually|annual|/\s*year|pa|p\.a\.)\b", re.IGNORECASE)
SENIOR = re.compile(r"\bsenior citizen\b|\b(?:age|aged)\s*(?:is\s*)?(6\d|7\d)\b|\b(6\d|7\d)\s*(?:years?|yrs?)\s*old\b", re.I)
SUPER_SENIOR = re.compile(r"\bsuper senior\b|\b(?:age|aged)\s*(?:is\s*)?([89]\d)\b|\b([89]\d)\s*(?:years?|yrs?)\s*old\b", re.I)
PARENTS_SENIOR = re.compile(r"\bparents?\b[^.]{0,30}\bsenior\b|\bsenior (?:citizen )?parents?\b", re.I)


def wants(text: str) -> bool:
    if deadlines.is_deadline_question(text) and not re.search(r"\b(?:calculate|how much|tax on)\b", text, re.I):
        return False
    return bool(TAX_INTENT.search(re.sub(r"\bprofessional tax\b|\bprofession tax\b", " ", text, flags=re.I)))


@dataclass
class TaxInput:
    income: float | None = None        # yearly gross salary
    income_period: str | None = None
    regime: str | None = None          # "old", "new", or None for both
    age: str = "below60"               # below60 | senior | super
    amounts: dict[str, float] = field(default_factory=dict)  # yearly
    hra_asked: bool = False
    parents_senior: bool = False
    hra_period_unknown: bool = False
    city: str | None = None
    metro: bool | None = None


def _yearly(value: float, after: str, default: str | None) -> float:
    if MONTHLY_CLAUSE.match(after):
        return value * 12
    if YEARLY_CLAUSE.match(after):
        return value
    return value * 12 if default == "monthly" else value


def parse(text: str, metro_cities: list[str]) -> TaxInput:
    t = TaxInput()
    rest = text
    found: list[tuple[str, float, str]] = []  # key, raw amount, text after it
    for key, label in DEDUCTIONS:
        pattern = re.compile(rf"\b(?:{label})\b[^\d₹.;]{{0,25}}(?:₹|rs\.?|inr)?\s*" + NUM, re.IGNORECASE)
        m = pattern.search(rest)
        if m and key not in t.amounts:
            found.append((key, _amount(m.group(1), m.group(2)), rest[m.end(): m.end() + 15]))
            rest = rest[: m.start()] + " ; " + rest[m.end():]
            t.amounts[key] = 0.0
    # The income: the first amount left, after the deduction figures are taken out.
    m = next((x for x in re.finditer(r"(?:₹|rs\.?|inr)?\s*" + NUM, rest, re.IGNORECASE)
              if _amount(x.group(1), x.group(2)) >= 1000  # skip "80", "60 years"
              and not re.match(r"\s*(?:%|years?|yrs?|days?|months?)", rest[x.end(): x.end() + 8], re.I)), None)
    if m:
        t.income = _amount(m.group(1), m.group(2))
        unit = (m.group(2) or "").lower()
        if unit in ("lpa",) or unit.startswith(("lakh", "lac", "cr", "l")) or t.income >= 500000:
            t.income_period = period(rest) or "yearly"
        else:
            t.income_period = period(rest)
        if t.income_period == "monthly":
            t.income *= 12
    # HRA figures without their own "per month" / "per year" follow the salary, or another HRA figure.
    hra_period = t.income_period or next(
        ("monthly" if MONTHLY_CLAUSE.match(a) else "yearly" for k, _, a in found
         if k in HRA_KEYS and (MONTHLY_CLAUSE.match(a) or YEARLY_CLAUSE.match(a))), None)
    for key, value, after in found:
        if key in HRA_KEYS and not (MONTHLY_CLAUSE.match(after) or YEARLY_CLAUSE.match(after)) and hra_period is None:
            t.hra_period_unknown = True
        t.amounts[key] = _yearly(value, after, hra_period if key in HRA_KEYS else "yearly")
    if re.search(r"\bnew regime\b", text, re.I) and not re.search(r"\bold\b", text, re.I):
        t.regime = "new"
    elif re.search(r"\bold regime\b", text, re.I) and not re.search(r"\bnew\b", text, re.I):
        t.regime = "old"
    if SUPER_SENIOR.search(text):
        t.age = "super"
    elif SENIOR.search(text):
        t.age = "senior"
    t.parents_senior = bool(PARENTS_SENIOR.search(text))
    t.hra_asked = bool(re.search(r"\bhra\b|\brent\b|house rent", text, re.I))
    for city in metro_cities:
        if re.search(rf"\b{city}\b", text, re.I) or (city == "Bengaluru" and re.search(r"\bbangalore\b", text, re.I)) or \
                (city == "Delhi" and re.search(r"\bnew delhi\b|\bncr\b", text, re.I)):
            t.city, t.metro = city, True
    if t.metro is None and re.search(r"\bnon[- ]?metro\b|\bother city\b", text, re.I):
        t.metro = False
    elif t.metro is None and re.search(r"\b(?:surat|vadodara|baroda|rajkot|nagpur|nashik|jaipur|lucknow|indore|kochi|noida|gurgaon|gurugram|"
                                       r"bhopal|chandigarh|coimbatore|thane|navi mumbai|gandhinagar|patna|bhubaneswar)\b", text, re.I):
        t.metro = False
    return t


# ----------------------------------------------------------------- arithmetic
def _slabs(r: dict[str, Any], key: str, on: date) -> tuple[list[dict[str, Any]], Rate]:
    v = pick((r.get("income_tax") or {}).get(key), on)
    if not v or not v.get("slabs") or any(s.get("rate") is None or s.get("from") is None for s in v["slabs"]):
        raise RateMissing(f"income_tax.{key}", (v or {}).get("source_doc", ""))
    return v["slabs"], tools._rate_of(v, 0)


def slab_tax(income: float, slabs: list[dict[str, Any]]) -> tuple[float, list[list[str]]]:
    total, rows = 0.0, []
    for s in slabs:
        lo, hi, r_ = float(s["from"]), s.get("to"), float(s["rate"])
        if income <= lo:
            break
        part = (min(income, float(hi)) if hi is not None else income) - lo
        amount = part * r_
        total += amount
        band = f"{inr(lo)} – {inr(hi)}" if hi is not None else f"above {inr(lo)}"
        rows.append([f"Tax at {pct(r_)}", f"{band}: {inr(part)} × {pct(r_)}", inr(amount)])
    return total, rows


def _surcharge(taxable: float, tax: float, slabs: list[dict[str, Any]], tax_slabs: list[dict[str, Any]]) -> float:
    """Surcharge with marginal relief: tax + surcharge just above a threshold may not exceed the tax at the
    threshold plus the income above it."""
    band = next((s for s in reversed(slabs) if taxable > float(s["from"])), None)
    if band is None:
        return 0.0
    threshold = float(band["from"])
    surcharge = tax * float(band["rate"])
    lower = next((s for s in reversed(slabs) if threshold > float(s["from"])), None)
    at_threshold = slab_tax(threshold, tax_slabs)[0] * (1 + (float(lower["rate"]) if lower else 0))
    return max(0.0, min(surcharge, at_threshold + (taxable - threshold) - tax))


def hra_exemption(t: TaxInput, r: dict[str, Any], on: date) -> tuple[float, list[list[str]], list[Rate]]:
    metro, other, excess = rate(r, "income_tax.hra_metro_rate", on), rate(r, "income_tax.hra_other_rate", on), \
        rate(r, "income_tax.hra_rent_excess_rate", on)
    basic, hra, rent = t.amounts["basic"], t.amounts["hra"], t.amounts["rent"]
    share = metro if t.metro else other
    options = [("HRA received", hra), (f"Rent − {pct(excess.value)} of basic + DA", max(0.0, rent - excess.value * basic)),
               (f"{pct(share.value)} of basic + DA ({t.city or ('metro city' if t.metro else 'non-metro city')})",
                share.value * basic)]
    exempt = min(v for _, v in options)
    rows = [[f"   HRA: {name}", "per year", inr(v)] for name, v in options]
    rows.append(["− HRA exemption (least of the three)", "rule 279", inr(exempt)])
    return exempt, rows, [metro, excess, share]


def compute(t: TaxInput, regime: str, r: dict[str, Any], on: date, state: str) -> tuple[float, list[list[str]], list[Rate], list[str]]:
    income = t.income or 0.0
    notes: list[str] = []
    used: list[Rate] = []
    sd = rate(r, f"income_tax.standard_deduction_{regime}", on)
    used.append(sd)
    rows = [["Gross salary", "per year", inr(income)], ["− Standard deduction", sd.section, inr(min(sd.value, income))]]
    taxable = income - min(sd.value, income)
    a = t.amounts
    if regime == "old":
        if t.hra_asked and {"hra", "rent", "basic"} <= a.keys():
            exempt, hrows, hused = hra_exemption(t, r, on)
            rows += hrows
            used += hused
            taxable -= exempt
        try:
            pt, pt_src, _ = tools.pt_monthly(state, income / 12, r, on)
            if pt:
                rows.append([f"− Professional Tax ({tools.state_name(state)})", "s.19(1) Sl. 1", inr(pt * 12)])
                taxable -= pt * 12
                used.append(pt_src)
        except RateMissing:
            notes.append("Professional Tax for your state isn't configured, so it isn't deducted in the old regime.")
        limits = {"s123": "deduction_s123", "s124_3": "deduction_s124_3", "home_loan": "deduction_home_loan_interest",
                  "s126_self": "deduction_s126_self_senior" if t.age != "below60" else "deduction_s126_self",
                  "s126_parents": "deduction_s126_parents_senior" if t.parents_senior else "deduction_s126_parents"}
        names = {"s123": "Section 123 (80C)", "s124_3": "Section 124(3) (80CCD(1B), NPS)", "home_loan": "Home-loan interest",
                 "s126_self": "Section 126 (80D), self/family", "s126_parents": "Section 126 (80D), parents"}
        for key, limit_key in limits.items():
            if a.get(key):
                lim = rate(r, f"income_tax.{limit_key}", on)
                used.append(lim)
                allowed = min(a[key], lim.value)
                rows.append([f"− {names[key]}", f"{inr(a[key])}, max {inr(lim.value)}", inr(allowed)])
                taxable -= allowed
    if a.get("employer_nps"):
        cap = rate(r, f"income_tax.employer_nps_rate_{regime}", on)
        used.append(cap)
        allowed = min(a["employer_nps"], cap.value * a["basic"]) if a.get("basic") else a["employer_nps"]
        rows.append(["− Employer's NPS contribution", f"max {pct(cap.value)} of basic + DA", inr(allowed)])
        taxable -= allowed
        if not a.get("basic"):
            notes.append(f"Employer NPS is allowed up to {pct(cap.value)} of basic + DA; give your basic to check the cap.")
    if regime == "new" and any(a.get(k) for k in ("s123", "s126_self", "s126_parents", "s124_3", "home_loan")):
        notes.append("The new regime doesn't allow 80C, 80D, 80CCD(1B), home-loan interest or HRA, so they are ignored there.")
    taxable = max(0.0, taxable)
    rows.append(["= Taxable income", "", inr(taxable)])
    slab_key = "new_regime_slabs" if regime == "new" else {"below60": "old_regime_slabs", "senior": "old_regime_slabs_senior",
                                                           "super": "old_regime_slabs_super_senior"}[t.age]
    slabs, src = _slabs(r, slab_key, on)
    used.append(src)
    tax, srows = slab_tax(taxable, slabs)
    rows += srows
    limit, most = rate(r, f"income_tax.rebate_{regime}_income_limit", on), rate(r, f"income_tax.rebate_{regime}_max", on)
    used.append(limit)
    rebate = 0.0
    if taxable <= limit.value:
        rebate = min(tax, most.value)
        if rebate:
            rows.append(["− Rebate", f"income up to {inr(limit.value)} ({limit.section})", inr(rebate)])
    elif regime == "new" and tax > taxable - limit.value:
        rebate = tax - (taxable - limit.value)
        rows.append(["− Marginal relief", f"tax capped at income above {inr(limit.value)} ({limit.section})", inr(rebate)])
    tax -= rebate
    sur_slabs, sur_src = _slabs(r, f"surcharge_{regime}", on)
    surcharge = _surcharge(taxable, tax, sur_slabs, slabs) if tax else 0.0
    if surcharge:
        rows.append(["+ Surcharge", "income above " + inr(next(float(s['from']) for s in sur_slabs)), inr(surcharge)])
        used.append(sur_src)
    cess = rate(r, "income_tax.cess_rate", on)
    used.append(cess)
    cess_amt = (tax + surcharge) * cess.value
    if cess_amt:
        rows.append(["+ Health & education cess", f"{pct(cess.value)} of tax", inr(cess_amt)])
    total = round(tax + surcharge + cess_amt)
    rows.append(["= Total tax for the year", "", inr(total)])
    rows.append(["Monthly TDS (spread over 12 months)", "", inr(total / 12)])
    return total, rows, used, notes


# ----------------------------------------------------------------- answers
def _metro_cities(r: dict[str, Any], on: date) -> list[str]:
    v = pick((r.get("income_tax") or {}).get("hra_metro_rate"), on) or {}
    return list(v.get("cities") or [])


def slabs_answer(r: dict[str, Any], on: date) -> ChatResponse:
    rows, used = [], []
    for regime, key in (("New", "new_regime_slabs"), ("Old (below 60)", "old_regime_slabs")):
        slabs, src = _slabs(r, key, on)
        used.append(src)
        for s in slabs:
            band = f"{inr(s['from'])} – {inr(s['to'])}" if s.get("to") is not None else f"above {inr(s['from'])}"
            rows.append([regime, band, pct(float(s["rate"]))])
    sd_n, sd_o = rate(r, "income_tax.standard_deduction_new", on), rate(r, "income_tax.standard_deduction_old", on)
    rn, rnm = rate(r, "income_tax.rebate_new_income_limit", on), rate(r, "income_tax.rebate_new_max", on)
    ro, rom = rate(r, "income_tax.rebate_old_income_limit", on), rate(r, "income_tax.rebate_old_max", on)
    cess = rate(r, "income_tax.cess_rate", on)
    used += [sd_n, rn, cess]
    answer = (f"Income-tax slabs for salaried individuals, tax year {_fy(on)[3:]} (the new regime is the default):\n\n"
              f"- Standard deduction: {inr(sd_n.value)} (new) or {inr(sd_o.value)} (old).\n"
              f"- Rebate: no tax if taxable income is up to {inr(rn.value)} in the new regime (rebate up to {inr(rnm.value)}, "
              f"with marginal relief just above it), or up to {inr(ro.value)} in the old regime (rebate up to {inr(rom.value)}).\n"
              f"- Add {pct(cess.value)} health and education cess on the tax.\n"
              "- The old regime allows deductions such as 80C, 80D, HRA and home-loan interest; the new regime doesn't "
              "(except the employer's NPS contribution).\n\n"
              "Ask \"tax on ₹12 lakh, old vs new\" to compare for your income.")
    return ChatResponse(type="tool", tool="income_tax_slabs", answer=answer,
                        table=Table(columns=["Regime", "Taxable income", "Rate"], rows=rows),
                        citations=tools._cite(*used), note=_note(on))


def _note(on: date) -> str:
    return (f"{_fy(on)} rates from data/rates.yaml (Income-tax Act, 2025); rates can change every Budget. "
            f"Assumes a resident individual with salary income only. {FOOTER}")


def run(text: str, state: str, rates: dict[str, Any] | None = None, today: date | None = None) -> ChatResponse:
    on = today or date.today()
    r = rates or tools.load_rates()
    try:
        return _run(text, state, r, on)
    except RateMissing as e:
        return tools.missing_answer(e.key, e.source_doc, on)


def _run(text: str, state: str, r: dict[str, Any], on: date) -> ChatResponse:
    t = parse(text, _metro_cities(r, on))
    hra_only = t.hra_asked and t.income is None and re.search(r"\bhra exemption\b|\bexemption on hra\b", text, re.I)
    if t.income is None and not hra_only:
        if SLAB_Q.search(text) or not re.search(r"\d", text):
            return slabs_answer(r, on)
        return _ask(["What is your gross salary, and is it monthly or yearly?"], "To work out your income tax I need:")
    asks = []
    if not hra_only and t.income_period is None:
        asks.append(f"Is {inr(t.income or 0)} your monthly or yearly salary?")
    if t.hra_asked and t.regime != "new":
        missing = [n for k, n in (("basic", "basic salary (+ DA)"), ("hra", "HRA received"), ("rent", "rent paid")) if not t.amounts.get(k)]
        if missing:
            asks.append("For the HRA exemption: your " + ", ".join(missing) + " (say per month or per year).")
        elif t.hra_period_unknown:
            asks.append("Are the basic, HRA and rent figures per month or per year?")
        if t.metro is None:
            asks.append("Which city do you live in? (Metro cities get 50% of basic, others 40%.)")
    if asks:
        return _ask(asks, "To calculate this accurately I need:")
    if hra_only:
        exempt, rows, used = hra_exemption(t, r, on)
        rows = [[n.strip(), b, v] for n, b, v in rows]
        return ChatResponse(type="tool", tool="hra_exemption",
                            answer=f"HRA exemption: **{inr(exempt)} a year** ({inr(exempt / 12)} a month). It applies only in the old regime.",
                            table=Table(columns=["Item", "Basis", "Amount"], rows=rows), citations=tools._cite(*used), note=_note(on))
    regimes = [t.regime] if t.regime else ["new", "old"]
    results = {g: compute(t, g, r, on, state) for g in regimes}
    rows: list[list[str]] = []
    lines: list[str] = []
    for g in regimes:
        total, grows, _, _ = results[g]
        if len(regimes) > 1:
            rows.append([f"**{g.title()} regime**", "", ""])
        rows += grows
        lines.append(f"- {g.title()} regime: **{inr(total)}** a year (about {inr(total / 12)} a month as TDS)")
    head = f"Income tax on a salary of {inr(t.income or 0)} a year, {_fy(on)}:"
    if len(regimes) > 1:
        new, old = results["new"][0], results["old"][0]
        better = "Both regimes give the same tax." if new == old else \
            f"The {'new' if new < old else 'old'} regime saves you {inr(abs(new - old))} a year."
        lines.append(better)
    notes = sorted({n for g in regimes for n in results[g][3]})
    if t.age == "below60" and "old" in regimes:
        notes.append("Old-regime slabs assume you are below 60; say if you are 60 or older.")
    notes.append("Your employer deducts TDS each month at the average rate on your estimated yearly salary, so the "
                 "monthly amount changes if your salary or declarations change during the year.")
    used = [u for g in regimes for u in results[g][2]]
    return ChatResponse(type="tool", tool="income_tax", answer=head + "\n" + "\n".join(lines) + "\n\n" + "\n".join(f"- {n}" for n in notes),
                        table=Table(columns=["Step", "Basis", "Amount"], rows=rows), citations=tools._cite(*used), note=_note(on))
