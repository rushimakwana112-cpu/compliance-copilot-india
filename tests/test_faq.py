"""Fixed explanations: every placeholder resolves from rates.yaml, and Act questions are left to the search."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest

from app import faq

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry", faq.load(), ids=lambda e: e["id"])
def test_every_entry_fills_and_cites(entry):
    r = faq.answer(entry, None, date(2026, 10, 15))
    assert "{" not in r.answer and "not configured" not in r.answer
    assert r.citations and all(src["url"].startswith("https://") for src in entry["sources"])


@pytest.mark.parametrize("q,topic", [
    ("What is the difference between CTC and gross salary?", "ctc_gross_net"), ("What is HRA?", "salary_components"),
    ("How is the employer's 12% split between EPF and EPS?", "pf_rates"), ("How to transfer PF?", "uan_epfo"),
    ("Is PF withdrawal before 5 years taxable?", "pf_withdrawal_tax"), ("What are ESIC benefits and how to claim?", "esi_benefits"),
    ("Is professional tax the same in every state?", "pt_basics"), ("What is Form 16?", "form16"),
    ("Which deductions are allowed in the new regime?", "deductions"), ("Is gratuity taxable?", "gratuity_tax"),
    ("gratuity for 4 years 7 months?", "gratuity_service"), ("What is the 50% wage rule?", "wage_50_rule"),
    ("Contract vs permanent for PF/ESIC", "contract_staff"), ("What does full and final settlement include?", "full_and_final"),
])
def test_topics(q, topic):
    assert (faq.match(q) or {}).get("id") == topic


def test_calculations_and_act_questions_are_not_faq():
    assert faq.match("Calculate PF for basic ₹18,000") is None
    rows = csv.DictReader((ROOT / "eval" / "questions.csv").open(encoding="utf-8"))
    hijacked = [r["id"] for r in rows if r["type"] == "rag" and faq.match(r["question"])]
    assert hijacked == []
