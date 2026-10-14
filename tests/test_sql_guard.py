"""The SQL guard must block anything but a single SELECT on known tables/columns."""

from __future__ import annotations

import sqlite3

import pytest

from app import sql_agent
from app.sql_agent import SQLGuardError, bind_params, validate
from scripts.make_sample_db import build


@pytest.mark.parametrize("sql", [
    "DROP TABLE employees",
    "UPDATE employees SET gross = 0",
    "DELETE FROM employees",
    "INSERT INTO employees VALUES ('x')",
    "SELECT * FROM employees; DROP TABLE employees",
    "SELECT 1; SELECT 2",
    "SELECT * FROM salaries",
    "SELECT * FROM sqlite_master",
    "SELECT password FROM employees",
    "PRAGMA table_info(employees)",
    "ATTACH DATABASE 'x.db' AS x",
    "SELECT load_extension('evil')",
    "",
])
def test_guard_rejects(sql):
    with pytest.raises(SQLGuardError):
        validate(sql)


def test_guard_allows_select_and_adds_limit():
    assert validate("SELECT emp_id, gross FROM employees WHERE gross > :esic_ceiling").endswith("LIMIT 200")


def test_guard_caps_large_limit_and_keeps_small_one():
    assert validate("SELECT name FROM employees LIMIT 5000").endswith("LIMIT 200")
    assert validate("SELECT name FROM employees LIMIT 5").endswith("LIMIT 5")


def test_guard_allows_aliases_and_trailing_semicolon():
    validate("SELECT state, COUNT(*) AS n FROM employees GROUP BY state ORDER BY n DESC;")


def test_params_come_from_rates_never_from_sql():
    rates = {"esic": {"wage_ceiling": {"value": 20000, "source_doc": "x"}}}
    assert bind_params("SELECT name FROM employees WHERE gross > :esic_ceiling", rates) == {"esic_ceiling": 20000.0}
    with pytest.raises(SQLGuardError):
        bind_params("SELECT name FROM employees WHERE gross > :made_up", rates)


def test_database_is_opened_read_only(tmp_path):
    db = tmp_path / "s.db"
    build(db)
    with pytest.raises(sqlite3.OperationalError):
        sql_agent.execute("DELETE FROM employees", {}, db)  # even if the guard were bypassed


def test_run_retries_once_then_answers(services, llm, tmp_path):
    db = tmp_path / "s.db"
    build(db)
    services.settings.db_path = db
    replies = iter(['{"sql": "SELECT * FROM salaries"}',
                    '{"sql": "SELECT emp_id, name, gross FROM employees WHERE gross > :esic_ceiling"}'])
    llm.responder = lambda m, j: next(replies)
    rates = {"esic": {"wage_ceiling": {"value": 21000, "source_doc": "x"}}}
    out = sql_agent.run(services, "Which sample employees are above the ESIC limit?", rates)
    assert out.type == "sql" and out.rows_scanned == 25
    assert "unknown table 'salaries'" in llm.calls[-1][1]["content"]  # the retry saw the error
    assert out.table.columns == ["emp_id", "name", "gross"] and len(out.table.rows) > 0
    assert ":esic_ceiling" in out.sql


def test_run_fails_gracefully(services, llm, tmp_path):
    db = tmp_path / "s.db"
    build(db)
    services.settings.db_path = db
    llm.responder = lambda m, j: '{"sql": "DROP TABLE employees"}'
    out = sql_agent.run(services, "drop everything in our staff table", {})
    assert out.type == "sql" and "didn't run anything" in out.answer
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM employees").fetchone()[0] == 25
