"""Evaluate the whole pipeline on eval/questions.csv and write eval/report.md.

PowerShell:
    python -m eval.run                         # full run (needs the LLM provider + Pinecone)
    python -m eval.run --sweep 0.34,0.38,0.42,0.47   # also test refusal thresholds (retrieval only, no LLM)

Metrics
    route accuracy     the router picked the expected path (not_found questions are expected to route to rag)
    retrieval hit@4    the expected document is among the first 4 citations (rag questions)
    answer contains    the answer contains the expected phrase (any of a|b|c, case-insensitive)
    not_found prec.    of the answers that refused, how many were supposed to refuse
    not_found recall   of the trick questions, how many were refused
    median latency     per question, end to end
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table as RichTable

from app import rag
from app.jurisdiction import resolve
from app.pipeline import handle
from app.schemas import ChatRequest, ChatResponse, JurisdictionIn, Turn
from app.services import Services, build_services
from ingest.chunk import MAX_CHARS, OVERLAP

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS = EVAL_DIR / "questions.csv"
REPORT = EVAL_DIR / "report.md"


@dataclass
class Row:
    id: str
    question: str
    type: str
    expected_answer_contains: str = ""
    expected_doc: str = ""
    state: str = "gujarat"
    follow_up_of: str = ""


@dataclass
class Result:
    row: Row
    route: str
    resp: ChatResponse
    checks: dict[str, bool | None] = field(default_factory=dict)


def load_questions(path: Path = QUESTIONS) -> list[Row]:
    with open(path, encoding="utf-8", newline="") as f:
        return [Row(**{k: (v or "").strip() for k, v in r.items()}) for r in csv.DictReader(f)]


def _any_in(alternatives: str, text: str) -> bool:
    return any(a.strip().lower() in text.lower() for a in alternatives.split("|") if a.strip())


def score(row: Row, route: str, resp: ChatResponse) -> dict[str, bool | None]:
    """Per-question checks. None = not applicable to this question."""
    expected_route = row.type if row.type in ("rag", "tool", "sql") else "rag"
    text = resp.answer + " " + " ".join(" ".join(r) for r in (resp.table.rows if resp.table else []))
    return {
        "route": route == expected_route,
        "hit@4": _any_in(row.expected_doc, " ".join(c.doc for c in resp.citations[:4]))
        if row.type == "rag" and row.expected_doc else None,
        "contains": _any_in(row.expected_answer_contains, text) if row.expected_answer_contains else None,
        "refused": resp.type == "not_found",
    }


def run(services: Services, rows: list[Row]) -> list[Result]:
    """Ask every question. Follow-ups get their parent question and answer as history."""
    done: dict[str, Result] = {}
    for row in rows:
        history: list[Turn] = []
        if row.follow_up_of and row.follow_up_of in done:
            parent = done[row.follow_up_of]
            history = [Turn(role="user", content=parent.row.question), Turn(role="assistant", content=parent.resp.answer)]
        req = ChatRequest(message=row.question, history=history,
                          jurisdiction=JurisdictionIn(state=row.state or "gujarat", mode="auto"))
        resp, turn = handle(services, req, log=False, use_cache=False)
        done[row.id] = Result(row, turn.route, resp, score(row, turn.route, resp))
    return list(done.values())


def _rate(values: list[bool | None]) -> tuple[float | None, int]:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals) if vals else None), len(vals)


def summarize(results: list[Result]) -> dict[str, tuple[float | None, int]]:
    refused = [r for r in results if r.checks["refused"]]
    tricks = [r for r in results if r.row.type == "not_found"]
    latencies = [r.resp.latency_ms for r in results]
    return {
        "route accuracy": _rate([r.checks["route"] for r in results]),
        "retrieval hit@4": _rate([r.checks["hit@4"] for r in results]),
        "answer contains": _rate([r.checks["contains"] for r in results]),
        "not_found precision": _rate([r.row.type == "not_found" for r in refused]),
        "not_found recall": _rate([r.checks["refused"] for r in tricks]),
        "median latency (s)": ((statistics.median(latencies) / 1000) if latencies else None, len(latencies)),
    }


def sweep(services: Services, rows: list[Row], thresholds: list[float]) -> list[tuple[float, float | None, float | None]]:
    """For each threshold: refusal precision and recall using retrieval scores only (fast, no LLM)."""
    best: list[tuple[bool, float]] = []
    for row in rows:
        if row.type not in ("rag", "not_found") or row.follow_up_of:
            continue
        scope = resolve(row.question, "auto", row.state or "gujarat")
        passages = rag.retrieve(services, row.question, scope)
        best.append((row.type == "not_found", max((p.score for p in passages), default=0.0)))
    out = []
    for t in thresholds:
        refused = [should for should, s in best if s < t]
        tricks = [s < t for should, s in best if should]
        out.append((t, _rate(refused)[0], _rate(tricks)[0]))
    return out


def _fmt(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.0f}%" if pct else f"{v:.2f}"


def write_report(results: list[Result], summary: dict, services: Services,
                 sweep_rows: list | None = None, path: Path = REPORT) -> str:
    s = services.settings
    lines = [
        "# Evaluation report", "",
        f"Run {datetime.now():%Y-%m-%d %H:%M} · model `{s.llm_provider}:{s.active_model}` · embeddings "
        f"`{s.pinecone_embed_model if s.embed_provider == 'pinecone' else s.embed_model}` · "
        f"threshold {s.score_threshold} · top_k {s.top_k} → keep {s.keep_k} · chunks {MAX_CHARS} chars / {OVERLAP} overlap · router `{s.router}`",
        "", "| Metric | Value | n |", "|---|---|---|",
    ]
    for name, (v, n) in summary.items():
        lines.append(f"| {name} | {_fmt(v, pct='latency' not in name)} | {n} |")
    if sweep_rows:
        lines += ["", "## Threshold sweep (retrieval only)", "", "| Threshold | Refusal precision | Trick questions refused |", "|---|---|---|"]
        lines += [f"| {t} | {_fmt(p)} | {_fmt(r)} |" for t, p, r in sweep_rows]
    lines += ["", "## Per question", "", "| id | type | route | answered as | hit@4 | contains | latency |", "|---|---|---|---|---|---|---|"]
    mark = {True: "✓", False: "✗", None: ""}
    for r in results:
        c = r.checks
        lines.append(f"| {r.row.id} | {r.row.type} | {r.route} {mark[c['route']]} | {r.resp.type} | {mark[c['hit@4']]} | "
                     f"{mark[c['contains']]} | {r.resp.latency_ms / 1000:.1f}s |")
    failed = [r for r in results if c_failed(r)]
    if failed:
        lines += ["", "## Failures to look at", ""]
        lines += [f"- **{r.row.id}** {r.row.question} → `{r.resp.type}`: {r.resp.answer[:160].replace(chr(10), ' ')}" for r in failed]
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def c_failed(r: Result) -> bool:
    c = r.checks
    wrong_refusal = c["refused"] != (r.row.type == "not_found")
    return wrong_refusal or c["route"] is False or c["hit@4"] is False or c["contains"] is False


def print_summary(summary: dict, sweep_rows: list | None) -> None:
    t = RichTable(title="Compliance Copilot eval")
    t.add_column("Metric"); t.add_column("Value", justify="right"); t.add_column("n", justify="right")
    for name, (v, n) in summary.items():
        t.add_row(name, _fmt(v, pct="latency" not in name), str(n))
    Console().print(t)
    if sweep_rows:
        st = RichTable(title="Threshold sweep")
        for col in ("Threshold", "Refusal precision", "Tricks refused"):
            st.add_column(col, justify="right")
        for th, p, r in sweep_rows:
            st.add_row(str(th), _fmt(p), _fmt(r))
        Console().print(st)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", help="comma-separated thresholds to test, e.g. 0.45,0.5,0.55,0.6")
    ap.add_argument("--only", help="comma-separated question ids")
    a = ap.parse_args()
    services = build_services()
    rows = load_questions()
    if a.only:
        keep = set(a.only.split(","))
        rows = [r for r in rows if r.id in keep]
    results = run(services, rows)
    summary = summarize(results)
    sweep_rows = sweep(services, rows, [float(x) for x in a.sweep.split(",")]) if a.sweep and services.store else None
    print_summary(summary, sweep_rows)
    write_report(results, summary, services, sweep_rows)
    print(f"Report written to {REPORT}")


if __name__ == "__main__":
    main()
