"""Evaluate the whole pipeline on eval/questions.csv and write eval/report.md.

PowerShell:
    python -m eval.run                         # full run (needs the LLM provider + Pinecone)
    python -m eval.run --sweep 0.34,0.38,0.42,0.47   # also test refusal thresholds (retrieval only, no LLM)
    python -m eval.run --delay 8               # seconds between questions (default 8, for Groq's free tier)

Skipped questions are listed in the report but left out of every metric:
    skipped (doc not ingested)   its `requires_doc` is not in data/manifest.json (e.g. the Gujarat PT schedule)
    skipped (rate limited)       the provider still refused after waiting out its retry-after once

Metrics
    route accuracy     the router picked the expected path (not_found questions: rag, or refused up front as out_of_scope)
    retrieval hit@4    the expected document is among the first 4 passages, cited ones first (rag questions)
    answer contains    the answer contains the expected phrases (a|b = any of, a&b = all of; case-insensitive)
    not_found prec.    of the answers that refused, how many were supposed to refuse
    not_found recall   of the trick questions, how many were refused
    median latency     per question, end to end
    avg prompt tokens  estimated size of the RAG answer prompts (chars / 4), to compare with the provider's TPM limit
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table as RichTable

from app import catalog, rag
from app.jurisdiction import foreign_place, resolve
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
    requires_doc: str = ""  # documents the answer needs: "a|b" any of, "a&b" all of (matched in document names)


@dataclass
class Result:
    row: Row
    route: str
    resp: ChatResponse
    checks: dict[str, bool | None] = field(default_factory=dict)
    skipped: str = ""  # "doc not ingested" | "rate limited"; skipped questions count in no metric
    prompt_tokens: int | None = None


def load_questions(path: Path = QUESTIONS) -> list[Row]:
    with open(path, encoding="utf-8", newline="") as f:
        return [Row(**{k: (v or "").strip() for k, v in r.items()}) for r in csv.DictReader(f)]


def _any_in(alternatives: str, text: str) -> bool:
    return any(a.strip().lower() in text.lower() for a in alternatives.split("|") if a.strip())


def _contains(expected: str, text: str) -> bool:
    """`a|b` = any of; `a&b|c` = a, and also b or c (checks that an answer is complete, not just on topic)."""
    return all(_any_in(group, text) for group in expected.split("&") if group.strip())


def score(row: Row, route: str, resp: ChatResponse) -> dict[str, bool | None]:
    """Per-question checks. None = not applicable to this question."""
    expected_route = row.type if row.type in ("rag", "tool", "sql") else "rag"
    text = resp.answer + " " + " ".join(" ".join(r) for r in (resp.table.rows if resp.table else []))
    return {
        "route": route == expected_route or (row.type == "not_found" and route == "out_of_scope"),
        "hit@4": _any_in(row.expected_doc, " ".join(c.doc for c in [*resp.citations, *resp.other_citations][:4]))
        if row.type == "rag" and row.expected_doc else None,
        "contains": _contains(row.expected_answer_contains, text) if row.expected_answer_contains else None,
        "refused": resp.type == "not_found",
    }


def indexed_docs() -> list[str] | None:
    """Names of the documents in the index (data/manifest.json), or None when there is no manifest."""
    if not catalog.MANIFEST.exists():
        return None
    return [name for name, e in catalog.read_manifest().items() if e.get("chunks")]


def has_docs(required: str, names: list[str]) -> bool:
    """'Gujarat Professional Tax|Gujarat PT&Maharashtra' against the indexed document names (case-insensitive)."""
    lowered = [n.lower() for n in names]
    return all(any(alt.strip().lower() in n for alt in group.split("|") if alt.strip() for n in lowered)
               for group in required.split("&") if group.strip())


def _skip(row: Row, reason: str) -> Result:
    return Result(row, "", ChatResponse(type="not_found", answer=f"skipped ({reason})"), skipped=reason)


def run(services: Services, rows: list[Row], *, delay: float = 0.0, max_wait: float = 90.0,
        docs: list[str] | None = None, sleep: Callable[[float], None] = time.sleep) -> list[Result]:
    """Ask every question. Follow-ups get their parent question and answer as history.

    `delay` seconds pass between questions (Groq's free tier counts tokens per minute). A question
    the provider refuses for a rate limit is asked once more after its retry-after (at most
    `max_wait` s); if it is still refused it is marked "rate limited". With `docs` (the indexed
    document names), a question whose `requires_doc` is missing is skipped without being asked.
    """
    done: dict[str, Result] = {}
    asked = 0
    for row in rows:
        if docs is not None and row.requires_doc and not has_docs(row.requires_doc, docs):
            done[row.id] = _skip(row, "doc not ingested")
            continue
        history: list[Turn] = []
        if row.follow_up_of and row.follow_up_of in done and not done[row.follow_up_of].skipped:
            parent = done[row.follow_up_of]
            history = [Turn(role="user", content=parent.row.question), Turn(role="assistant", content=parent.resp.answer)]
        req = ChatRequest(message=row.question, history=history,
                          jurisdiction=JurisdictionIn(state=row.state or "gujarat", mode="auto"))
        if asked and delay > 0:
            sleep(delay)
        asked += 1
        resp, turn = handle(services, req, log=False, use_cache=False)
        if resp.rate_limited:
            wait = (turn.trace.rate_limit or {}).get("retry_after")
            wait = 60.0 if wait is None else float(wait)
            if wait <= max_wait:
                sleep(wait)
                resp, turn = handle(services, req, log=False, use_cache=False)
        if resp.rate_limited:
            done[row.id] = _skip(row, "rate limited")
            continue
        done[row.id] = Result(row, turn.route, resp, score(row, turn.route, resp),
                              prompt_tokens=turn.trace.prompt_tokens)
    return list(done.values())


def _rate(values: list[bool | None]) -> tuple[float | None, int]:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals) if vals else None), len(vals)


def summarize(results: list[Result]) -> dict[str, tuple[float | None, int]]:
    """Metrics over the questions that were actually asked: skipped ones are left out."""
    results = [r for r in results if not r.skipped]
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
        "avg prompt tokens": avg_prompt_tokens(results),
    }


def avg_prompt_tokens(results: list[Result]) -> tuple[float | None, int]:
    tokens = [r.prompt_tokens for r in results if r.prompt_tokens and not r.skipped]
    return (statistics.mean(tokens) if tokens else None), len(tokens)


def sweep(services: Services, rows: list[Row], thresholds: list[float],
          docs: list[str] | None = None) -> list[tuple[float, float | None, float | None]]:
    """For each threshold: refusal precision and recall using retrieval scores only (fast, no LLM).
    Questions whose `requires_doc` is not indexed are left out, as in the main run."""
    best: list[tuple[bool, float]] = []
    for row in rows:
        if row.type not in ("rag", "not_found") or row.follow_up_of:
            continue
        if docs is not None and row.requires_doc and not has_docs(row.requires_doc, docs):
            continue
        if foreign_place(row.question):  # refused before retrieval, whatever the threshold
            best.append((row.type == "not_found", 0.0))
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


def _fmt_metric(name: str, v: float | None) -> str:
    if name == "avg prompt tokens":
        return "n/a" if v is None else f"{v:,.0f}"
    return _fmt(v, pct="latency" not in name)


def write_report(results: list[Result], summary: dict, services: Services,
                 sweep_rows: list | None = None, path: Path = REPORT) -> str:
    s = services.settings
    tokens, n_tokens = avg_prompt_tokens(results)
    skipped = [r for r in results if r.skipped]
    lines = [
        "# Evaluation report", "",
        f"Run {datetime.now():%Y-%m-%d %H:%M} · model `{s.llm_provider}:{s.active_model}` · embeddings "
        f"`{s.pinecone_embed_model if s.embed_provider == 'pinecone' else s.embed_model}` · "
        f"threshold {s.score_threshold} · top_k {s.top_k} → keep {s.keep_k} · chunks {MAX_CHARS} chars / {OVERLAP} overlap · router `{s.router}`",
        "",
        (f"GROQ_MODEL `{s.groq_model or 'not set'}` · avg prompt tokens {_fmt_metric('avg prompt tokens', tokens)} "
         f"(RAG answers, n={n_tokens}, budget {s.context_token_budget} for passages) · "
         f"{len(results) - len(skipped)} asked, {len(skipped)} skipped (not in the metrics)"),
        "", "| Metric | Value | n |", "|---|---|---|",
    ]
    for name, (v, n) in summary.items():
        lines.append(f"| {name} | {_fmt_metric(name, v)} | {n} |")
    if sweep_rows:
        lines += ["", "## Threshold sweep (retrieval only)", "", "| Threshold | Refusal precision | Trick questions refused |", "|---|---|---|"]
        lines += [f"| {t} | {_fmt(p)} | {_fmt(r)} |" for t, p, r in sweep_rows]
    lines += ["", "## Per question", "", "| id | type | route | answered as | hit@4 | contains | latency |", "|---|---|---|---|---|---|---|"]
    mark = {True: "✓", False: "✗", None: ""}
    for r in results:
        if r.skipped:
            lines.append(f"| {r.row.id} | {r.row.type} | | skipped ({r.skipped}) | | | |")
            continue
        c = r.checks
        lines.append(f"| {r.row.id} | {r.row.type} | {r.route} {mark[c['route']]} | {r.resp.type} | {mark[c['hit@4']]} | "
                     f"{mark[c['contains']]} | {r.resp.latency_ms / 1000:.1f}s |")
    failed = [r for r in results if not r.skipped and c_failed(r)]
    if failed:
        lines += ["", "## Failures to look at", ""]
        lines += [f"- **{r.row.id}** {r.row.question} → `{r.resp.type}`: {r.resp.answer[:160].replace(chr(10), ' ')}" for r in failed]
    if skipped:
        lines += ["", "## Skipped", ""]
        lines += [f"- **{r.row.id}** {r.row.question}: {r.skipped}"
                  + (f" (needs {r.row.requires_doc})" if r.skipped == "doc not ingested" else "") for r in skipped]
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def c_failed(r: Result) -> bool:
    c = r.checks
    wrong_refusal = c["refused"] != (r.row.type == "not_found")
    return wrong_refusal or c["route"] is False or c["hit@4"] is False or c["contains"] is False


def print_summary(summary: dict, sweep_rows: list | None, results: list[Result] | None = None) -> None:
    t = RichTable(title="Compliance Copilot eval")
    t.add_column("Metric"); t.add_column("Value", justify="right"); t.add_column("n", justify="right")
    for name, (v, n) in summary.items():
        t.add_row(name, _fmt_metric(name, v), str(n))
    skipped = [r for r in results or [] if r.skipped]
    if skipped:
        t.caption = "Skipped: " + ", ".join(f"{r.row.id} ({r.skipped})" for r in skipped)
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
    ap.add_argument("--delay", type=float, default=8.0, help="seconds between questions (default 8)")
    ap.add_argument("--max-wait", type=float, default=90.0, help="longest retry-after to wait out on a 429 (default 90)")
    a = ap.parse_args()
    services = build_services()
    docs = indexed_docs()
    if docs is None:
        print("data/manifest.json not found: questions with requires_doc are asked anyway.")
    rows = load_questions()
    if a.only:
        keep = set(a.only.split(","))
        rows = [r for r in rows if r.id in keep]
    results = run(services, rows, delay=a.delay, max_wait=a.max_wait, docs=docs)
    summary = summarize(results)
    sweep_rows = sweep(services, rows, [float(x) for x in a.sweep.split(",")], docs) if a.sweep and services.store else None
    print_summary(summary, sweep_rows, results)
    write_report(results, summary, services, sweep_rows)
    print(f"Report written to {REPORT}")


if __name__ == "__main__":
    main()
