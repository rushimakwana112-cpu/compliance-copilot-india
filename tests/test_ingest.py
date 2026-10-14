"""Parsing, structure-aware chunking and idempotent indexing."""

from __future__ import annotations

from app import catalog
from ingest.chunk import chunk_pages, split_text
from ingest.index import vector_id
from ingest.parse import Page, parse_html, parse_pdf
from ingest.run import ingest_file

META = {"doc": "Code on Social Security, 2020", "scope": "central", "status": "current",
        "effective_date": "2025-11-21", "doc_type": "act", "url": "https://example.gov.in/ss.pdf"}


def test_parse_pdf_pages(act_pdf):
    pages = parse_pdf(act_pdf)
    assert [p.number for p in pages] == [1, 2]
    assert "continuous service" in pages[0].text


def test_parse_html_strips_chrome(tmp_path):
    f = tmp_path / "faq.html"
    f.write_text("<html><nav>Home | About</nav><header>Ministry</header><main><p>Q.1: Who is covered?</p>"
                 "<p>Every employee.</p></main><footer>Copyright</footer></html>")
    text = parse_html(f)[0].text
    assert "Who is covered" in text and "Home" not in text and "Copyright" not in text


def test_chunks_split_on_sections_and_keep_page(act_pdf):
    chunks = chunk_pages(parse_pdf(act_pdf))
    sections = [c.section for c in chunks]
    assert any(s.startswith("Section 53") for s in sections)
    assert any(s.startswith("Section 54") and c.page == 2 for s, c in zip(sections, chunks))
    # the chapter heading is merged into section 53, not left as a tiny chunk
    assert all(len(c.text) >= 40 for c in chunks)


def test_section_label_carries_over_page_break():
    pages = [Page(1, "12. Wages.—(1) " + "word " * 60), Page(2, "continued text of section twelve " * 10)]
    chunks = chunk_pages(pages)
    assert chunks[-1].page == 2 and chunks[-1].section.startswith("Section 12")


def test_split_text_caps_size_with_overlap():
    text = " ".join(f"sentence{i} is here." for i in range(400))
    parts = split_text(text, 1000, 150)
    assert all(len(p) <= 1000 for p in parts)
    assert parts[0][-60:].split()[-1] in parts[1]  # overlap repeats the end of the previous chunk


def test_vector_ids_are_deterministic():
    assert vector_id("Doc A", 3, 1) == vector_id("Doc A", 3, 1)
    assert vector_id("Doc A", 3, 1) != vector_id("Doc A", 3, 2)
    assert vector_id("Code on Wages, 2019", 1, 0).startswith("code-on-wages-2019#")


def test_ingest_is_idempotent_and_replaces_changed_files(act_pdf, services, store):
    first = ingest_file(act_pdf, META, services)
    n = store.count()
    assert first.status == "indexed" and n == first.chunks > 0
    assert ingest_file(act_pdf, META, services).status == "unchanged"
    # pretend the file changed upstream: old vectors are removed, no duplicates remain
    catalog.update_manifest(META["doc"], {"indexed_sha256": "old-hash"})
    again = ingest_file(act_pdf, META, services)
    assert again.status == "re-indexed" and again.deleted == n and store.count() == n
    meta = next(iter(store.records.values()))["metadata"]
    assert {"doc", "section", "page", "scope", "status", "effective_date", "doc_type", "source_url", "text"} <= set(meta)


def test_dry_run_touches_nothing(act_pdf, services, store):
    assert ingest_file(act_pdf, META, services, dry_run=True).status == "dry-run"
    assert store.count() == 0 and catalog.read_manifest() == {}
