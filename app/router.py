"""Decide how to answer a message: calculator tool, SQL over sample data, or RAG.

The default is rules (keywords + regex): instant, free, and easy to debug.
Set ROUTER=llm to let Qwen classify instead; any parse error falls back to
the rules, so a confused model can never break routing.
"""

from __future__ import annotations

import re
from typing import Literal

from app.llm import LLM, LLMError, parse_json

Path_ = Literal["tool", "sql", "rag"]

AMOUNT = re.compile(r"(?:₹|\brs\.?|\binr)?\s*\d[\d,]*(?:\.\d+)?\s*(?:k\b|lakhs?\b|lacs?\b)?|\d{3,}", re.IGNORECASE)
CALC_INTENT = re.compile(
    r"\b(calculate|calculation|compute|how much|work out|estimate|break-?up|split|deduction for|contribution (?:for|on))\b",
    re.IGNORECASE,
)
CALC_TOPIC = re.compile(r"\b(pf|epf|provident fund|esic|esi|professional tax|pt|gratuity)\b", re.IGNORECASE)
# "employees" alone is not enough ("Is ESIC applicable to employees earning 24,000?" is a law question).
DATA_NOUN = re.compile(
    r"\b(sample (?:data|employees?|staff|payroll)|our (?:staff|employees?|team|workforce)|"
    r"employee (?:data|table|records)|payroll data|in the (?:database|db)|dataset|E\d{3})\b",
    re.IGNORECASE,
)
QUERY_VERB = re.compile(r"\b(list|which|how many|who|show|count|average|total|highest|lowest|top)\b", re.IGNORECASE)


def has_amount(text: str) -> bool:
    """True if the text contains a money-like number (3+ digits, or 18k / 1.5 lakh)."""
    return any(re.search(r"\d{3,}|\d\s*(k|lakh|lac)\b", m.group(0).replace(",", ""), re.I) for m in AMOUNT.finditer(text))


def route_rules(question: str) -> Path_:
    """Keyword/regex router."""
    if CALC_INTENT.search(question) and CALC_TOPIC.search(question) and has_amount(question):
        return "tool"
    if DATA_NOUN.search(question) and QUERY_VERB.search(question):
        return "sql"
    return "rag"


ROUTER_PROMPT = """Classify the user's question about Indian payroll law into exactly one route.
- "tool": asks to calculate PF, ESIC, professional tax or gratuity for a given amount.
- "sql": asks about the company's sample employee data (list, count, which employees).
- "rag": anything else, including questions about what the law says.
Reply as JSON: {"route": "tool" | "sql" | "rag"}"""


def route_llm(llm: LLM, question: str) -> Path_:
    """Ask the LLM to classify; fall back to rules on any error or unexpected value."""
    try:
        reply = llm.chat(
            [{"role": "system", "content": ROUTER_PROMPT}, {"role": "user", "content": question}], json_mode=True
        )
    except LLMError:
        return route_rules(question)
    value = (parse_json(reply) or {}).get("route")
    if value not in ("tool", "sql", "rag"):  # not JSON: accept a reply that names exactly one route
        found = set(re.findall(r"\b(tool|sql|rag)\b", reply.lower()))
        value = found.pop() if len(found) == 1 else None
    return value if value in ("tool", "sql", "rag") else route_rules(question)


def route(question: str, mode: str, llm: LLM) -> Path_:
    """Route with the configured strategy (`rules` or `llm`)."""
    return route_llm(llm, question) if mode == "llm" else route_rules(question)
