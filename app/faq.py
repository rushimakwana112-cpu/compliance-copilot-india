"""Fixed explanations from data/faq.yaml for common questions whose documents aren't in the search index
(CTC vs gross, UAN, Form 130, ESI benefits, deductions...). Figures are filled from data/rates.yaml with
the value valid today, so they change when the rates file does. No model call, no search.
"""

from __future__ import annotations

import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app import router, tools
from app.config import DATA_DIR
from app.salary import FOOTER, _fy
from app.schemas import ChatResponse, Citation
from app.tools import RateMissing, inr, pct

FAQ_PATH = DATA_DIR / "faq.yaml"
PLACEHOLDER = re.compile(r"\{(pct|inr|num):([a-z0-9_.]+)\}")
CALC = re.compile(r"\b(?:calculate|compute|how much|work out|deduction (?:on|for)|for (?:basic|gross|salary)|on (?:basic|gross|salary))\b", re.I)
AMOUNT = re.compile(r"\d[\d,]{2,}|\d\s*(?:k|lakh|lac)\b", re.I)


@lru_cache
def load(path: Path = FAQ_PATH) -> list[dict[str, Any]]:
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    for e in entries:
        e["_patterns"] = [re.compile(p, re.IGNORECASE) for p in e.get("patterns") or []]
    return entries


def match(text: str, entries: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """The first entry whose pattern matches, unless the message asks to calculate something with an amount."""
    if AMOUNT.search(text) and (CALC.search(text) or router.route_rules(text) == "tool"):
        return None
    return next((e for e in (entries or load()) if any(p.search(text) for p in e["_patterns"])), None)


def fill(answer: str, rates: dict[str, Any], on: date) -> tuple[str, list[tools.Rate]]:
    used: list[tools.Rate] = []

    def one(m: re.Match[str]) -> str:
        kind, key = m.group(1), m.group(2)
        try:
            r = tools.rate(rates, key, on)
        except RateMissing:
            return "(not configured in data/rates.yaml)"
        used.append(r)
        return pct(r.value) if kind == "pct" else inr(r.value) if kind == "inr" else f"{r.value:g}"

    return PLACEHOLDER.sub(one, answer), used


def answer(entry: dict[str, Any], rates: dict[str, Any] | None = None, today: date | None = None) -> ChatResponse:
    on = today or date.today()
    text, used = fill(entry["answer"].strip(), rates or tools.load_rates(), on)
    cites = [Citation(doc=s["doc"], section=s.get("section") or "", url=s.get("url") or "", snippet="")
             for s in entry.get("sources") or []]
    seen = {c.doc for c in cites}
    cites += [c for c in tools._cite(*used) if c.doc not in seen]
    note = f"Figures from data/rates.yaml as they apply on {on.isoformat()} ({_fy(on)}); rules and rates can change. {FOOTER}"
    return ChatResponse(type="guide", tool=f"faq:{entry['id']}", answer=text, citations=cites, note=note)
