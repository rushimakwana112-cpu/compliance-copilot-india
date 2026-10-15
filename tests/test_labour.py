"""Bonus, overtime and leave encashment from the shipped data/rates.yaml."""

from __future__ import annotations

import copy
from datetime import date

from app import labour, tools

TODAY = date(2026, 10, 15)


def test_bonus_waits_for_the_notified_ceiling():
    r = labour.run("bonus on 15000 salary for 12 months", None, TODAY)
    assert "(rate not configured)" in r.answer and "bonus.calculation_ceiling" in r.note and "8.33%" in r.answer
    assert "rates.yaml" not in r.answer


def test_bonus_with_a_configured_ceiling():
    rates = copy.deepcopy(tools.load_rates())
    rates["bonus"]["calculation_ceiling"] = [{"value": 7000, "status": "current", "source_doc": "Test notification"}]
    r = labour.run("bonus on 15000 salary for 12 months", rates, TODAY)
    assert r.table.rows[1][2] == "₹7,000" and "₹6,997" in r.answer and "₹16,800" in r.answer


def test_bonus_rate_outside_the_law_is_refused():
    assert "between 8.33% and 20%" in labour.run("bonus at 25% on 10000 for 12 months", None, TODAY).answer


def test_overtime_is_twice_the_hourly_rate():
    r = labour.run("overtime for 10 hours on salary 26000", None, TODAY)
    assert "₹2,500" in r.answer  # 26,000 ÷ 26 ÷ 8 = 125 an hour × 2 × 10


def test_leave_encashment_asks_for_the_divisor_then_calculates():
    assert "÷ 26 days" in labour.run("encash 10 days leave salary 26000", None, TODAY).answer
    r = labour.run("leave encashment 12 days basic 26000 26 days", None, TODAY)
    assert "₹12,000" in r.answer and "taxable" in r.answer
