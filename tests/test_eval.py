"""The eval harness itself, run end to end with fakes on a few questions."""

from __future__ import annotations

from app.llm import Busy, QuotaExceeded
from eval.run import (
    _contains,
    has_docs,
    indexed_docs,
    load_questions,
    run,
    summarize,
    sweep,
    write_report,
)


def test_questions_file_shape():
    rows = load_questions()
    assert len(rows) == 43
    assert all(r.type == "tool" for r in rows if r.id in {"q08", "q34", "q35"})  # due-date questions
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
    assert summary["not_found recall"] == (1.0, 1)  # "Dubai" is refused before retrieval
    assert results[1].resp.type == "rag"  # the follow-up was rewritten using q02 as history
    report = write_report(results, summary, services, sweep(services, rows, [0.1, 0.9]), tmp_path / "r.md")
    assert "| route accuracy | 100% | 3 |" in report and "Threshold sweep" in report


def test_contains_supports_all_of_groups():
    expected = "remuneration&basic pay&does not include|shall not include"
    assert _contains(expected, "Wages means all remuneration and includes basic pay, but does not include bonus")
    assert not _contains(expected, "Wages means all remuneration and includes basic pay")  # exclusions missing


def test_real_test_questions_are_in_the_eval():
    rows = {r.question: r for r in load_questions()}
    assert rows["Definition of wages under the Code on Wages"].type == "rag"
    assert rows["Who is covered for gratuity under the SS Code"].expected_answer_contains.count("&") == 1
    assert rows["What is a fixed term employee?"].expected_doc
    assert rows["Gratuity rule in Dubai?"].type == "not_found"


# ----------------------------------------------------------------- rate limits, missing documents, prompt tokens

GRATUITY = "Yes, fixed-term employees qualify after continuous service [1]."


class Sleeps(list):
    def __call__(self, seconds: float) -> None:
        self.append(seconds)


def _seed(store):
    store.add("gratuity fixed term employee continuous service eligible", doc="Code on Social Security, 2020",
              page=40, scope="central", status="current", doc_type="act")


def _rows(*ids):
    return [r for r in load_questions() if r.id in set(ids)]


def test_delay_between_questions(services, store, llm):
    _seed(store)
    llm.responder = lambda m, j: GRATUITY
    sleeps = Sleeps()
    results = run(services, _rows("q02", "q04", "q26"), delay=8, sleep=sleeps)
    assert sleeps == [8, 8] and not any(r.skipped for r in results)


def test_rate_limited_question_is_retried_after_retry_after_then_skipped(services, store, llm):
    _seed(store)
    calls = []

    def responder(m, j):
        calls.append(1)
        raise Busy("Busy right now, please retry in a minute.", limit="TPM", retry_after=12.0)

    llm.responder = responder
    sleeps = Sleeps()
    results = run(services, _rows("q02", "q04"), delay=0, sleep=sleeps)
    assert [r.skipped for r in results] == ["rate limited", "rate limited"]
    assert sleeps == [12.0, 12.0] and len(calls) == 4  # each asked twice
    summary = summarize(results)
    assert summary["route accuracy"] == (None, 0) and summary["answer contains"] == (None, 0)


def test_rate_limit_that_clears_counts_normally(services, store, llm):
    _seed(store)
    replies = iter([QuotaExceeded("Daily demo limit reached.", limit="RPD", retry_after=30.0), GRATUITY])

    def responder(m, j):
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return r

    llm.responder = responder
    sleeps = Sleeps()
    [result] = run(services, _rows("q02"), sleep=sleeps)
    assert sleeps == [30.0] and not result.skipped and result.checks["contains"] is True


def test_long_daily_wait_is_not_waited_out(services, store, llm):
    _seed(store)

    def responder(m, j):
        raise QuotaExceeded("Daily demo limit reached.", limit="TPD", retry_after=4000.0)

    llm.responder = responder
    sleeps = Sleeps()
    [result] = run(services, _rows("q02"), max_wait=90, sleep=sleeps)
    assert result.skipped == "rate limited" and sleeps == []


def test_missing_document_is_skipped_not_failed(services, store, llm, tmp_path):
    _seed(store)
    llm.responder = lambda m, j: GRATUITY
    docs = ["Code on Social Security, 2020", "Code on Wages, 2019"]
    results = run(services, _rows("q02", "q09", "q10", "q12"), docs=docs)
    assert [r.skipped for r in results] == ["", "doc not ingested", "doc not ingested", "doc not ingested"]
    assert len(llm.calls) >= 1 and summarize(results)["route accuracy"][1] == 1
    report = write_report(results, summarize(results), services, None, tmp_path / "r.md")
    assert "| q09 | rag | | skipped (doc not ingested) |" in report
    assert "q09** Professional tax for ₹15,000 salary in Ahmedabad: doc not ingested (needs Gujarat" in report
    assert "1 asked, 3 skipped" in report


def test_has_docs():
    names = ["Gujarat PT schedule 2024", "Code on Wages, 2019"]
    assert has_docs("Gujarat Professional Tax|Gujarat PT", names)
    assert not has_docs("Gujarat PT&Maharashtra PT", names)
    assert has_docs("code on wages", names)


def test_indexed_docs_reads_the_manifest():
    from app import catalog

    assert indexed_docs() is None  # no manifest yet: nothing is skipped
    catalog.update_manifest("Code on Wages, 2019", {"chunks": 40})
    catalog.update_manifest("Failed upload", {"chunks": 0})
    assert indexed_docs() == ["Code on Wages, 2019"]


def test_requires_doc_column():
    rows = {r.id: r for r in load_questions()}
    assert rows["q09"].requires_doc.startswith("Gujarat Professional Tax") and rows["q12"].requires_doc
    assert "Maharashtra Professional Tax" in rows["q11"].requires_doc
    assert not rows["q08"].requires_doc and not rows["q02"].requires_doc  # q08 is now a due-date tool question


def test_report_shows_model_and_average_prompt_tokens(services, store, llm, tmp_path):
    _seed(store)
    services.settings.groq_model = "qwen-test"
    llm.responder = lambda m, j: GRATUITY
    results = run(services, _rows("q02", "q04"))
    tokens = [r.prompt_tokens for r in results]
    assert all(t and t > 50 for t in tokens)
    summary = summarize(results)
    assert summary["avg prompt tokens"] == (sum(tokens) / 2, 2)
    report = write_report(results, summary, services, None, tmp_path / "r.md")
    assert f"GROQ_MODEL `qwen-test` · avg prompt tokens {sum(tokens) / 2:,.0f} (RAG answers, n=2" in report
