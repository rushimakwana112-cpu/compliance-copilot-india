"""Answers given before any routing: what the bot can't or won't do, and questions too vague to answer.

The bot has no employee data: no payslips, salary history, attendance, bank details or
personal records. It answers from compliance rules and from numbers the user types. These
checks run first, are plain regex (instant, no model call), and return a fixed answer:

* personal records ("What is my salary?", "Has my salary been credited?")
* another employee's data, or a list of employees
* prompt injection ("ignore your rules", "act as admin")
* off-topic chat (weather, jokes, cricket) and greetings
* vague one-word questions ("PF?", "tax?"): one short clarifying question
* invalid figures (negative or zero salary, 40 working days in a month)

`advice_note` adds a pointer to a tax advisor or HR when a question asks for personal tax
planning or is about a dispute.
"""

from __future__ import annotations

import re

from app.schemas import ChatResponse

CHECK_HR = "For your actual records, check with HR/payroll."
SCOPE_LINE = ("I answer payroll and labour-law questions for India: PF, ESIC, Professional Tax, TDS on salary, "
              "gratuity, bonus, wages and due dates, and I calculate them from figures you give me.")

AMOUNT = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:k|lakh|lac|l|cr)?\b|₹\s*\d", re.IGNORECASE)


def _has_amount(text: str) -> bool:
    return any(re.search(r"\d{3,}|\d\s*(?:k|lakh|lac|l)\b", m.group(0).replace(",", ""), re.I) for m in AMOUNT.finditer(text))


# ----------------------------------------------------------------- personal records
PERSONAL = re.compile(
    r"\b(?:what(?:'s| is| was)|show|tell|check|give|send|download|see|view|where is)\b[^?.]{0,25}\bmy\s+"
    r"(?:salary|pay ?slip|salary slip|ctc|pf balance|epf balance|pf amount|uan(?: number)?|leave balance|attendance|"
    r"bank|account|form ?16|form ?130|tds|tax deducted|bonus|increment|arrears|gratuity amount|net pay|take home)\b"
    r"|\b(?:has|is|was|did)\s+my\s+(?:salary|pay|bonus|arrears|reimbursement|pf)\b[^?.]{0,20}\b(?:credited|paid|deposited|come|transferred|received)\b"
    r"|\bsalary\s+(?:credited|not credited|came|not received)\b"
    r"|\bwhy\b[^?.]{0,25}\bmy\s+(?:salary|pay|in-hand|net pay|take home)\b[^?.]{0,25}\b(?:deducted|less|low|cut|reduced|different|short)\b"
    r"|\bwhy\b[^?.]{0,25}\b(?:deducted|cut)\b[^?.]{0,25}\bmy\s+(?:salary|pay)\b"
    r"|\bwhen will i get my (?:salary|bonus|arrears|pf|gratuity)\b"
    r"|\bmy\s+(?:pf balance|epf balance|uan|pay ?slip|salary slip|leave balance|attendance)\b"  # "mera PF balance batao"
    r"|\bmy\s+(?:salary|ctc|in-hand|pf)\s+how much\b",  # "mera salary kitna hai"
    re.IGNORECASE,
)
PORTALS = ("For your PF balance, use the EPFO member passbook (passbook.epfindia.gov.in) or the UMANG app with your UAN. "
           "For your payslip, salary credit or deductions, ask HR/payroll.")


def personal_record(text: str) -> ChatResponse | None:
    if not PERSONAL.search(text) or _has_amount(text):
        return None
    return _guide(
        "I can't see your salary, payslips, PF account or any personal records: I don't have access to employee data.\n\n"
        "I can calculate it if you share the figures, for example your basic salary, gross salary or days worked "
        "(\"My basic is ₹20,000, what is my PF?\").\n\n" + PORTALS,
        "No employee data: personal records are with HR/payroll and the official portals.")


# ----------------------------------------------------------------- other employees / lists
OTHER_EMPLOYEE = re.compile(
    r"\b(?:salary|ctc|pay|payslip|pf|details?|records?|address|bank|phone|tax)\s+of\s+(?!a\b|an\b|the\b|my\b|our\b|employees?\b|workers?\b|staff\b)(?-i:[A-Z][a-z]+)"
    r"|\b(?!(?:employee|employer|company|today|month|year|india|gujarat|maharashtra|state|government|worker|contractor|someone|everyone)'s)"
    r"(?-i:[A-Z][a-z]+)'s\s+(?:salary|ctc|pay|payslip|pf|bonus|increment|bank)"
    r"|\b(?:list|show|give|display|export|download)\b[^?.]{0,20}\b(?:all\s+)?(?:the\s+)?(?:employees|staff|workers|employee list|employee data|employee records)\b"
    r"|\ball employees\b|\bemployee (?:list|data|database|records|details)\b|\bsample (?:data|employees?|payroll)\b"
    r"|\bwho (?:earns|gets|has) the (?:most|highest|lowest|least)\b|\b(?:my|a) colleague'?s?\b[^?.]{0,20}\b(?:salary|pay|ctc)\b",
    re.IGNORECASE,
)


def other_employee(text: str) -> ChatResponse | None:
    if not OTHER_EMPLOYEE.search(text):
        return None
    return no_data()


def no_data() -> ChatResponse:
    """The answer to any question about employee records (also used when the sample-data route is off)."""
    return _guide("I don't have any employee data, so I can't show anyone's salary or list employees. "
                  "Salary details are confidential and are held by HR/payroll.\n\n" + SCOPE_LINE,
                  "No employee data.")


# ----------------------------------------------------------------- prompt injection
INJECTION = re.compile(
    r"\b(?:ignore|forget|disregard|override|bypass)\b[^.?]{0,30}\b(?:rules?|instructions?|prompt|guidelines?|restrictions?)\b"
    r"|\b(?:act|behave|pretend|roleplay)\s+(?:as|like)\s+(?:an?\s+)?(?:admin|administrator|root|developer|hr manager|"
    r"database|dan|unrestricted|system)\b"
    r"|\byou are now\b|\b(?:system|hidden|initial) prompt\b|\breveal\b[^.?]{0,20}\b(?:prompt|instructions|rules)\b"
    r"|\bdeveloper mode\b|\bjailbreak\b|\bsudo\b|\badmin (?:mode|access|password)\b",
    re.IGNORECASE,
)


def injection(text: str) -> ChatResponse | None:
    if not INJECTION.search(text):
        return None
    return _guide("I can't change my rules or act as an admin, and I don't have access to any employee data or systems. "
                  + SCOPE_LINE + " What would you like to know?", "Request outside the bot's rules.")


# ----------------------------------------------------------------- off-topic and greetings
OFF_TOPIC = re.compile(
    r"\b(?:weather|temperature outside|rain(?:ing)?|joke|jokes|funny|cricket|ipl|world cup|football|movie|movies|film|"
    r"song|songs|music|recipe|cook|poem|poetry|story|horoscope|astrology|bitcoin|crypto|share market|stock (?:tip|price)s?|"
    r"which stock|mutual fund|girlfriend|boyfriend|dating|capital of|president of|prime minister of|"
    r"write (?:me )?(?:an? )?(?:essay|code|program)|python code|homework)\b",
    re.IGNORECASE,
)
GREETING = re.compile(r"^\s*(?:hi+|hello+|hey+|namaste|namaskar|good (?:morning|afternoon|evening)|kem cho|"
                      r"thanks?|thank you|ok(?:ay)?|who are you\??|what can you do\??|help)[\s!.?]*$", re.IGNORECASE)


def off_topic(text: str) -> ChatResponse | None:
    if GREETING.match(text):
        return _guide("Hello! " + SCOPE_LINE + "\n\nTry: \"Calculate PF for basic ₹18,000\", "
                      "\"My CTC is ₹6 lakh a year, what is my in-hand?\" or \"When is TDS on salary due?\"", "Greeting.")
    if not OFF_TOPIC.search(text):
        return None
    return _guide("I only handle payroll and compliance questions. " + SCOPE_LINE, "Out of scope.", refused=True)


# ----------------------------------------------------------------- vague questions
VAGUE = [  # (pattern for the whole message, clarifying question)
    (r"pf|epf|provident fund|pf\s+kitna", "What would you like to know about PF: the contribution rates, your PF deduction "
     "for a given basic salary, the due date, or UAN, balance and withdrawal?"),
    (r"esi|esic", "What would you like to know about ESIC: who is covered, the contribution rates, or ESIC on a given gross salary?"),
    (r"tax|income tax|tds", "Which tax question: the old vs new regime slabs, your tax on a given annual income, "
     "how monthly TDS is worked out, or the TDS due dates?"),
    (r"pt|professional tax", "What would you like to know about Professional Tax: the slabs in your state, PT on a given salary, or the due date?"),
    (r"gratuity", "What would you like to know about gratuity: eligibility, the formula, or the amount for a given last basic + DA and years of service?"),
    (r"bonus", "What would you like to know about bonus: who is eligible, the minimum and maximum, or bonus on a given salary?"),
    (r"salary|ctc|in-hand|salary calculation", "What would you like to work out: in-hand salary from CTC, per-day salary, "
     "loss of pay, first or last month's salary, or an increment?"),
    (r"overtime|ot", "Do you want the overtime rule, or overtime pay for a given salary and number of hours?"),
    (r"leave|leave encashment", "Do you want the leave rules, or leave encashment for a given salary and number of days?"),
    (r"minimum wage|minimum wages", "Minimum wages depend on the state, zone and skill category. Which state, and which category "
     "(unskilled, semi-skilled or skilled)?"),
]


def vague(text: str) -> ChatResponse | None:
    words = re.sub(r"[?!.,]", " ", text.lower()).split()
    if not words or len(words) > 3 or _has_amount(text):
        return None
    msg = " ".join(words)
    for pattern, question in VAGUE:
        if re.fullmatch(rf"(?:{pattern})(?:\s+(?:please|pls|plz|kya|hai|question|query|info|details|help|calculate|calculation|calculator))?", msg):
            return _guide(question, "Clarifying question: the message names a topic but not what to work out.")
    return None


# ----------------------------------------------------------------- invalid figures
NEGATIVE = re.compile(r"(?:^|[\s(:=])(?:₹|rs\.?|inr)?\s*-\s*₹?\s*\d|\bminus\s+(?:₹|rs\.?)?\s*\d", re.IGNORECASE)
ZERO = re.compile(r"(?:₹|\brs\.?|\binr)\s*0+(?:\.0+)?(?![\d.])|\b(?:salary|basic|gross|ctc|wages?|pay)\s+(?:is\s+|of\s+|=\s*)?(?:₹\s*)?0+(?![\d.,])",
                  re.IGNORECASE)
MONTH_DAYS = re.compile(r"\b(\d+)\s+(?:working\s+days|days\s+(?:worked|working|present)|paid\s+days)\b"
                        r"|\b(\d+)\s+days\s+in\s+(?:a|the|one)\s+month\b|\bworked\s+(?:for\s+)?(\d+)\s+days\s+(?:in|this|last)\b",
                        re.IGNORECASE)


def invalid_figures(text: str) -> ChatResponse | None:
    problems = []
    if NEGATIVE.search(text):
        problems.append("an amount is negative")
    if ZERO.search(text):
        problems.append("the salary is ₹0")
    for m in MONTH_DAYS.finditer(text):
        days = int(next(g for g in m.groups() if g))
        if days > 31:
            problems.append(f"{days} days is more than a month has")
    if not problems:
        return None
    return _guide("I can't calculate that: " + "; ".join(problems) + ". Please check the figures and send them again "
                  "(amounts in ₹, positive, and at most 31 days in a month).", "Invalid input.")


# ----------------------------------------------------------------- advice pointer
TAX_PLANNING = re.compile(r"\b(?:how (?:can|do|should) i (?:save|reduce|avoid) tax|tax saving tips?|best (?:way|investment) to save tax|"
                          r"where should i invest|should i invest|which investment|tax planning)\b", re.IGNORECASE)
DISPUTE = re.compile(r"\b(?:not paying|didn't pay|did not pay|refus\w+ to pay|withheld|withholding|legal notice|court|sue|"
                     r"lawyer|complaint|harass\w*|terminated without|fired without|dispute|labour commissioner)\b", re.IGNORECASE)


def advice_note(text: str) -> str | None:
    if TAX_PLANNING.search(text):
        return "I can explain the rules and compare regimes for figures you give, but for personal tax planning please consult a tax advisor."
    if DISPUTE.search(text):
        return "For a dispute, speak to HR first; if it isn't resolved, the labour department or a lawyer can advise on your case."
    return None


TAX_PLANNING_ONLY = re.compile(r"^(?:(?!\d).)*$")  # no figures given


def tax_planning(text: str) -> ChatResponse | None:
    if TAX_PLANNING.search(text) and TAX_PLANNING_ONLY.match(text):
        return _guide("I can't advise on personal investments or tax planning. I can explain the old and new tax regimes, "
                      "the deductions each allows, and compare your tax under both if you give me your annual income and "
                      "deductions. For a personal plan, please consult a tax advisor.", "Tax planning: refer to an advisor.")
    return None


# ----------------------------------------------------------------- entry point
CHECKS = (injection, other_employee, personal_record, off_topic, tax_planning, invalid_figures, vague)


def check(text: str) -> ChatResponse | None:
    """The first guard answer that applies to the (normalized) message, or None to route it normally."""
    for fn in CHECKS:
        if (resp := fn(text)) is not None:
            return resp
    return None


def _guide(answer: str, note: str, refused: bool = False) -> ChatResponse:
    return ChatResponse(type="not_found" if refused else "guide", answer=answer, note=note)
