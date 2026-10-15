"""Text-to-SQL over the synthetic sample database, with a strict guard.

The LLM writes the query; Python decides whether it may run. The guard parses
the SQL with sqlglot (a real SQL parser, not a regex) and only lets through a
single SELECT on known tables and columns, then adds LIMIT 200. The database
is also opened read-only, so even a query that slipped past the guard could
not change data. Two independent locks are better than one.
"""

from __future__ import annotations

import math
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

import sqlglot
import yaml
from sqlglot import exp

from app.config import DATA_DIR
from app.llm import LLMError, parse_json
from app.schemas import ChatResponse, Table
from app.services import Services
from app.tools import RateMissing, load_rates, rate

EXAMPLES_PATH = DATA_DIR / "sql_examples.yaml"
MAX_ROWS = 200
SCHEMA: dict[str, list[str]] = {
    "employees": ["emp_id", "name", "state", "basic", "gross", "doj", "employment_type"],
}
SCHEMA_TEXT = """Table employees (synthetic sample data, one row per employee):
  emp_id TEXT          e.g. 'E101'
  name TEXT
  state TEXT           lower-case id: 'gujarat', 'maharashtra', 'karnataka', 'delhi', 'tamil-nadu'
  basic INTEGER        monthly basic wage in INR
  gross INTEGER        monthly gross wage in INR
  doj TEXT             date of joining, 'YYYY-MM-DD'
  employment_type TEXT 'permanent' | 'fixed-term' | 'contract'"""
# Named parameters the model may use, and where their values come from.
PARAMS = {
    "esic_ceiling": "esic.wage_ceiling",
    "eps_wage_ceiling": "pf.eps_wage_ceiling",
    "gratuity_min_years": "gratuity.min_years_permanent",
}
FORBIDDEN = tuple(getattr(exp, n) for n in ("Insert", "Update", "Delete", "Drop", "Create", "Alter", "Command",
                                             "Pragma", "Attach", "Detach", "Transaction") if hasattr(exp, n))
FORBIDDEN_FUNCS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


class SQLGuardError(ValueError):
    """The generated SQL is not allowed to run."""


# ----------------------------------------------------------------- guard
def validate(sql: str, schema: dict[str, list[str]] = SCHEMA, max_rows: int = MAX_ROWS) -> str:
    """Return a safe version of `sql` (with LIMIT) or raise SQLGuardError."""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise SQLGuardError("empty query")
    if ";" in sql:
        raise SQLGuardError("multiple statements are not allowed")
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise SQLGuardError(f"could not parse SQL: {str(e).splitlines()[0]}") from e
    if len(statements) != 1:
        raise SQLGuardError("exactly one statement is allowed")
    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union)):
        raise SQLGuardError(f"only SELECT is allowed, got {stmt.key.upper()}")
    if any(isinstance(node, FORBIDDEN) for node in stmt.walk()):
        raise SQLGuardError("statement contains a write or admin command")
    for fn in stmt.find_all(exp.Anonymous, exp.Func):
        name = (fn.name if isinstance(fn, exp.Anonymous) else fn.sql_name()).lower()
        if name in FORBIDDEN_FUNCS:
            raise SQLGuardError(f"function {name}() is not allowed")

    ctes = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE)}
    for t in stmt.find_all(exp.Table):
        if t.name.lower() not in schema and t.name.lower() not in ctes:
            raise SQLGuardError(f"unknown table '{t.name}'")
    known = {c.lower() for cols in schema.values() for c in cols}
    aliases = {a.alias.lower() for a in stmt.find_all(exp.Alias)} | {a.lower() for a in ctes}
    for col in stmt.find_all(exp.Column):
        if col.name.lower() not in known and col.name.lower() not in aliases:
            raise SQLGuardError(f"unknown column '{col.name}'")

    limit = stmt.args.get("limit")
    if limit is None:
        stmt = stmt.limit(max_rows)
    else:
        try:
            n = int(limit.expression.name) if limit.expression else max_rows + 1
        except (AttributeError, ValueError):
            n = max_rows + 1
        if n > max_rows:
            stmt = stmt.limit(max_rows)
    return stmt.sql(dialect="sqlite")


def placeholders(sql: str) -> set[str]:
    return {p.name for p in sqlglot.parse_one(sql, read="sqlite").find_all(exp.Placeholder) if p.name}


def bind_params(sql: str, rates: dict[str, Any]) -> dict[str, float]:
    """Values for :named parameters, taken from rates.yaml. Unknown or unset parameters raise."""
    values = {}
    for name in placeholders(sql):
        if name not in PARAMS:
            raise SQLGuardError(f"unknown parameter :{name}; allowed: {', '.join(':' + p for p in PARAMS)}")
        values[name] = rate(rates, PARAMS[name]).value  # RateMissing if null
    return values


# ----------------------------------------------------------------- execution
def execute(sql: str, params: dict[str, float], db_path: Path) -> tuple[list[str], list[tuple]]:
    """Run on a read-only connection (mode=ro): the database file cannot be modified."""
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        cur = con.execute(sql, params)
        return [d[0] for d in cur.description or []], cur.fetchall()
    finally:
        con.close()


def table_rows(db_path: Path) -> int:
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        return int(con.execute("SELECT COUNT(*) FROM employees").fetchone()[0])
    finally:
        con.close()


# ----------------------------------------------------------------- generation
@lru_cache
def load_examples(path: Path = EXAMPLES_PATH) -> list[dict[str, str]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


_example_vectors: dict[str, list[list[float]]] = {}


def _cos(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def order_examples(services: Services, question: str, k_similar: int = 3, total: int = 8) -> list[dict[str, str]]:
    """Up to `total` examples, with the `k_similar` most similar (by embedding) placed last, nearest the question."""
    examples = load_examples()
    try:
        key = getattr(services.llm, "embed_model", "") or services.llm.model
        if key not in _example_vectors:
            _example_vectors[key] = services.llm.embed([e["question"] for e in examples])
        q = services.llm.embed([question])[0]
        scores = [_cos(q, v) for v in _example_vectors[key]]
    except LLMError:  # embedding unavailable: fall back to word overlap
        words = set(re.findall(r"\w+", question.lower()))
        scores = [len(words & set(re.findall(r"\w+", e["question"].lower()))) for e in examples]
    ranked = sorted(range(len(examples)), key=lambda i: scores[i], reverse=True)
    similar, rest = ranked[:k_similar], ranked[k_similar:]
    chosen = rest[: max(0, total - k_similar)][::-1] + similar[::-1]
    return [examples[i] for i in chosen]


SYSTEM = f"""You write one SQLite SELECT query for a question about a synthetic payroll table.
{SCHEMA_TEXT}

Rules:
- One SELECT statement only. Only the employees table and the columns above.
- Never write legal thresholds as numbers. Use these named parameters instead:
  :esic_ceiling (ESIC wage ceiling), :eps_wage_ceiling (EPS wage ceiling), :gratuity_min_years (gratuity minimum service).
- Numbers the user typed themselves may be written as literals.
- Reply with only the query inside a ```sql code block. No explanation."""

SQL_FENCE = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
BARE_SELECT = re.compile(r"^\s*((?:WITH|SELECT)\b.*)", re.IGNORECASE | re.DOTALL | re.MULTILINE)


def extract_sql(reply: str) -> str | None:
    """The query from a model reply: a ```sql block, else JSON {"sql": …}, else text starting at SELECT/WITH.

    Plain text, not JSON mode: some Groq models fail JSON mode with a 400 "Failed to generate JSON".
    """
    m = SQL_FENCE.search(reply)
    if m and m.group(1).strip():
        return m.group(1).strip()
    value = (parse_json(reply) or {}).get("sql")
    if isinstance(value, str) and value.strip():
        return value.strip()
    m = BARE_SELECT.search(reply)
    return m.group(1).strip().rstrip("`").strip() if m else None


def generate(services: Services, question: str, error: str | None = None, previous: str | None = None) -> str:
    """Ask the LLM for SQL. On a retry, show it the previous query and the error."""
    shots = "\n\n".join(f"Q: {e['question']}\nSQL: {e['sql']}" for e in order_examples(services, question))
    user = f"Examples:\n\n{shots}\n\nQ: {question}"
    if error:
        user += f"\n\nYour previous query was:\n{previous}\nIt failed with: {error}\nWrite a corrected query."
    reply = services.llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    sql = extract_sql(reply)
    if not sql:
        raise SQLGuardError("model did not return SQL")
    return sql


def _summary(columns: list[str], rows: list[tuple]) -> str:
    if not rows:
        return "No sample employees match."
    if len(rows) == 1 and len(columns) <= 3 and not {"emp_id", "name"} & set(columns):
        return "Result: " + ", ".join(f"{c.replace('_', ' ')} = {v}" for c, v in zip(columns, rows[0])) + "."
    noun = "row" if len(rows) == 1 else "rows"
    return f"{len(rows)} {noun} from the synthetic sample data" + (f" (first {MAX_ROWS} shown)." if len(rows) >= MAX_ROWS else ".")


def run(services: Services, question: str, rates: dict[str, Any] | None = None) -> ChatResponse:
    """Generate, validate and run SQL; retry once with the error; fail gracefully."""
    db = services.settings.db_path
    if not db.exists():
        return ChatResponse(type="sql", answer="The sample database is missing. Run: python scripts/make_sample_db.py")
    rates = rates if rates is not None else load_rates()
    error: str | None = None
    sql: str | None = None
    for _attempt in range(2):
        try:
            sql = generate(services, question, error, sql)
            safe = validate(sql)
            columns, rows = execute(safe, bind_params(safe, rates), db)
        except RateMissing as e:
            return ChatResponse(type="sql", sql=sql, answer=f"Rate not configured: `{e.key}` is empty in data/rates.yaml, "
                                "so I can't compare against the legal threshold yet.")
        except (SQLGuardError, sqlite3.Error) as e:
            error = str(e)
            continue
        return ChatResponse(
            type="sql", sql=safe, rows_scanned=table_rows(db), answer=_summary(columns, rows),
            table=Table(columns=columns, rows=[[("" if v is None else str(v)) for v in r] for r in rows]) if rows else None,
            note="Read-only query on synthetic data. Thresholds are passed as parameters from rates.yaml.",
        )
    return ChatResponse(type="sql", sql=sql, answer="I couldn't build a safe query for that question, so I didn't run anything.",
                        note=f"Last error: {error}")
