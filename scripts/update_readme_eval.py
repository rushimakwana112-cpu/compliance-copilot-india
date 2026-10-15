"""Copy the metrics from eval/report.md into the README's results table.

    python -m scripts.update_readme_eval

Rewrites only the block between the `eval-results:start` and `eval-results:end` markers in README.md.
Exits with code 1 (and changes nothing) when the report has no results, e.g. the eval hasn't run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
REPORT = ROOT / "eval" / "report.md"
METRICS = ["route accuracy", "retrieval hit@4", "answer contains", "not_found precision", "not_found recall",
           "median latency (s)"]
BLOCK = re.compile(r"(<!-- eval-results:start[^\n]*-->\n).*?(<!-- eval-results:end -->)", re.DOTALL)


def parse_report(text: str) -> dict[str, object]:
    """Metric values plus asked/skipped counts from the report eval/run.py writes."""
    values = {}
    for name in METRICS:
        m = re.search(rf"^\| {re.escape(name)} \| ([^|]+) \|", text, re.MULTILINE)
        if m:
            values[name] = m.group(1).strip()
    counts = re.search(r"(\d+) asked, (\d+) skipped", text)
    reasons = re.findall(r"^- \*\*q\d+\*\* .*?: (doc not ingested|rate limited)", text, re.MULTILINE)
    return {"metrics": values, "asked": int(counts.group(1)) if counts else None,
            "skipped": int(counts.group(2)) if counts else 0,
            "missing_docs": reasons.count("doc not ingested"), "rate_limited": reasons.count("rate limited")}


def render(report: dict[str, object]) -> str:
    metrics: dict[str, str] = report["metrics"]  # type: ignore[assignment]
    rows = ["| Metric | Value |", "|---|---|"]
    for name in METRICS:
        value = metrics.get(name, "n/a")
        if name == "median latency (s)":
            name, value = "median latency", (f"{float(value):.1f} s" if value not in ("n/a", "") else "n/a")
        rows.append(f"| {name} | {value} |")
    why = []
    if report["missing_docs"]:
        why.append(f"{report['missing_docs']} because their documents aren't indexed yet")
    if report["rate_limited"]:
        why.append(f"{report['rate_limited']} because of Groq rate limits")
    if len(why) == 1 and int(why[0].split()[0]) == report["skipped"]:
        skipped = f", {report['skipped']} skipped " + why[0].split(" ", 1)[1]  # "7 skipped because ..."
    else:
        skipped = f", {report['skipped']} skipped" + (f" ({'; '.join(why)})" if why else "") if report["skipped"] else ""
    note = (f"Note: measured on {report['asked']} questions{skipped}; small sample, so treat as indicative. "
            "Updated by scripts/run_all.ps1.")
    return "\n".join(rows) + "\n\n" + note + "\n"


def update(readme: str, report_text: str) -> str:
    report = parse_report(report_text)
    if not report["metrics"] or report["asked"] is None:
        raise ValueError("eval/report.md has no results yet: run python -m eval.run first")
    if not BLOCK.search(readme):
        raise ValueError("README.md has no eval-results markers")
    return BLOCK.sub(lambda m: m.group(1) + render(report) + m.group(2), readme, count=1)


def main() -> int:
    try:
        new = update(README.read_text(encoding="utf-8"), REPORT.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"Could not update the README: {e}")
        return 1
    README.write_text(new, encoding="utf-8")
    print("README.md eval table updated from eval/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
