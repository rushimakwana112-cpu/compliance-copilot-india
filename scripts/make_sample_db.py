"""Create data/sample_payroll.db with a synthetic `employees` table.

Every row is made up. Names are generated from small first/last-name lists and
salaries from a fixed random seed, so the file is identical on every run and
safe to commit. Never put real employee data in this database.

Usage (PowerShell):  python scripts/make_sample_db.py
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_payroll.db"
ROWS = 25
SEED = 42

FIRST = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Isha", "Vihaan", "Ananya", "Arjun", "Sara",
         "Dev", "Nisha", "Kunal", "Priya", "Yash", "Tara", "Neel", "Riya", "Om", "Zoya"]
LAST = ["Shah", "Patel", "Iyer", "Rao", "Mehta", "Desai", "Kulkarni", "Nair", "Joshi", "Reddy"]
STATES = ["gujarat"] * 9 + ["maharashtra"] * 6 + ["karnataka"] * 4 + ["delhi"] * 3 + ["tamil-nadu"] * 3
TYPES = ["permanent"] * 18 + ["fixed-term"] * 5 + ["contract"] * 2

SCHEMA = """
CREATE TABLE employees (
    emp_id          TEXT PRIMARY KEY,   -- e.g. E101
    name            TEXT NOT NULL,
    state           TEXT NOT NULL,      -- state id, same ids as data/states.yaml
    basic           INTEGER NOT NULL,   -- monthly basic wage, INR
    gross           INTEGER NOT NULL,   -- monthly gross wage, INR
    doj             TEXT NOT NULL,      -- date of joining, ISO yyyy-mm-dd
    employment_type TEXT NOT NULL CHECK (employment_type IN ('permanent', 'fixed-term', 'contract'))
);
"""


def make_rows(n: int = ROWS, seed: int = SEED) -> list[tuple]:
    """Return `n` deterministic synthetic employee rows."""
    rng = random.Random(seed)
    states = STATES[:]
    types = TYPES[:]
    rng.shuffle(states)
    rng.shuffle(types)
    rows = []
    for i in range(n):
        # A spread of salaries so ESIC / PF / PT questions have rows on both sides of any threshold.
        low_band = i % 3 == 0
        gross = rng.randrange(11_000, 24_000, 500) if low_band else rng.randrange(24_000, 95_000, 500)
        basic = int(round(gross * rng.uniform(0.40, 0.55), -2))
        doj = date(2014, 1, 1) + timedelta(days=rng.randrange(0, 365 * 11))
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        rows.append((f"E{101 + i}", name, states[i], basic, gross, doj.isoformat(), types[i]))
    return rows


def build(path: Path = DB_PATH) -> int:
    """(Re)create the database at `path` and return the number of rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rows = make_rows()
    with sqlite3.connect(path) as con:
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


if __name__ == "__main__":
    n = build()
    print(f"Wrote {n} synthetic employees to {DB_PATH}")
    with sqlite3.connect(DB_PATH) as con:
        for row in con.execute("SELECT * FROM employees ORDER BY emp_id LIMIT 5"):
            print("  ", row)
        print("   by state:", con.execute(
            "SELECT state, COUNT(*) FROM employees GROUP BY state ORDER BY 2 DESC").fetchall())
