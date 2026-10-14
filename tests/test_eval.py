"""The eval harness itself, run end to end with fakes on a few questions."""

from __future__ import annotations

from eval.run import load_questions, run, summarize, sweep, write_report


def test_questions_file_shape():
    rows = load_questions()
    assert len(rows) == 30
    assert sum(r.type == "not_found" for r in rows) == 5
    assert sum(bool(r.follow_up_of) for r in rows) == 5
    ids = [r.id for r in rows]
    assert all(ids.index(r.follow_up_of) < ids.index(r.id) for r in rows if r.follow_up_of)


def test_eval_runs_with_fakes(services, store, llm, tmp_path):
    store.add("gratuity fixed term employee continuous service eligible", doc="Code on Social Security, 2020",
              page=40, scope="central", status="current", doc_type="act")
    llm.responder = lambda m, j: "Who is eligible for gratuity for fixed-term employees?" if "Rewrite" in m[0]["content"] \
        else "Yes, fixed-term employees qualify after continuous service [1]."
    rows = [r for r in load_questions() if r.id in {"q02", "q03", "q26"}]
    results = run(services, rows)
    summary = summarize(results)
    assert summary["route accuracy"][0] == 1.0
    assert summary["retrieval hit@4"] == (1.0, 2)
    assert summary["not_found recall"] == (1.0, 1)  # "Dubai" scores below the threshold
    assert results[1].resp.type == "rag"  # the follow-up was rewritten using q02 as history
    report = write_report(results, summary, services, sweep(services, rows, [0.1, 0.9]), tmp_path / "r.md")
    assert "| route accuracy | 100% | 3 |" in report and "Threshold sweep" in report
