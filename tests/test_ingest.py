"""Parsing, structure-aware chunking and idempotent indexing."""

from __future__ import annotations

import re

from app import catalog
from ingest.chunk import MAX_CHARS, OVERLAP, chunk_pages, split_text
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
    pages = [Page(1, "12. Wages.—(1) " + "word " * 60), Page(2, "continued text of section twelve. " * 80)]
    chunks = chunk_pages(pages)
    assert chunks[0].page == 1 and chunks[-1].page == 2 and chunks[-1].section.startswith("Section 12")


DEFINITIONS = """CHAPTER I
PRELIMINARY
2. Definitions.—In this Code, unless the context otherwise requires,—
(a) "appropriate Government" means,— (i) in relation to an establishment carried on by the Central
Government, the Central Government; and (ii) in relation to any other establishment, the State Government;
(b) "employee" means any person employed on wages by an establishment to do any skilled, semi-skilled or
unskilled work, but does not include a member of the Armed Forces of the Union;
(y) "wages" means all remuneration whether by way of salaries, allowances or otherwise, expressed in terms
of money or capable of being so expressed which would, if the terms of employment were fulfilled, be payable
to a person employed in respect of his employment, and includes,—
(i) basic pay;
(ii) dearness allowance; and
(iii) retaining allowance, if any,
but does not include—
(a) any bonus payable under any law for the time being in force, which does not form part of the
remuneration payable under the terms of employment;
(b) the value of any house-accommodation, or of the supply of light, water, medical attendance;
(c) any contribution paid by the employer to any pension or provident fund;
(d) any conveyance allowance or the value of any travelling concession;
(e) house rent allowance;
(f) overtime allowance;
(g) any commission payable to the employee:
Provided that, for calculating the wages under this clause, if payments made by the employer to the
employee under clauses (a) to (g) exceed the prescribed proportion of the remuneration, the amount which
exceeds it shall be deemed as remuneration and shall be accordingly added in wages under this clause.
Explanation.—For the removal of doubts, it is hereby declared that remuneration in kind is included.
(z) "worker" means any person employed in any industry to do any manual, unskilled, skilled, technical,
operational or clerical work for hire or reward;
3. Prohibition of discrimination on ground of gender.—(1) There shall be no discrimination in an
establishment among employees on the ground of gender in matters relating to wages.
"""


def test_definitions_get_one_chunk_per_term_with_the_full_definition():
    split = DEFINITIONS.index("(f) overtime")  # the wages definition runs across a page break
    pages = [Page(4, DEFINITIONS[:split]), Page(5, DEFINITIONS[split:])]
    chunks = chunk_pages(pages, doc="Code on Wages, 2019")
    defs = {c.term: c for c in chunks if c.term}
    assert set(defs) == {"appropriate government", "employee", "wages", "worker"}
    wages = defs["wages"].text
    assert wages.startswith("Code on Wages, 2019, Section 2 (Definitions) — wages:")
    for part in ("all remuneration", "(i) basic pay", "(iii) retaining allowance", "but does not include",
                 "(a) any bonus", "(g) any commission", "Provided that", "Explanation.—"):
        assert part in wages, part
    assert '"worker"' not in wages and '"employee"' not in wages
    assert defs["wages"].page == 4 and defs["worker"].page == 5
    assert defs["wages"].chapter == "Chapter I — Preliminary" and defs["wages"].section == "Section 2 · Definitions"
    # the next section is chunked normally
    assert any(c.section.startswith("Section 3") and not c.term for c in chunks)


def test_long_sections_cut_only_at_sub_clauses():
    clauses = "".join(f"({i}) " + f"rule text for sub-clause {i} " * 12 + "\n" for i in range(1, 9))
    chunks = chunk_pages([Page(1, "10. Payment of bonus.—" + clauses)])
    assert len(chunks) > 1 and all(len(c.text) <= MAX_CHARS + OVERLAP + 5 for c in chunks)
    for c in chunks[1:]:
        body = c.text.split("\n", 1)[1]  # after the "…" overlap line
        assert re.match(r"\(\d\) rule text for sub-clause", body), body[:40]
    assert chunks[1].text.startswith("… ")  # overlap carried from the previous chunk
    assert [c.chunk_no for c in chunks] == list(range(len(chunks)))


def test_chunk_pages_are_ints(act_pdf):
    assert all(type(c.page) is int for c in chunk_pages(parse_pdf(act_pdf)))


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
    first = ingest_file(act_pdf, META, store)
    n = store.count()
    assert first.status == "indexed" and n == first.chunks > 0
    assert ingest_file(act_pdf, META, store).status == "unchanged"
    # pretend the file changed upstream: old vectors are removed, no duplicates remain
    catalog.update_manifest(META["doc"], {"indexed_sha256": "old-hash"})
    again = ingest_file(act_pdf, META, store)
    assert again.status == "re-indexed" and again.deleted == n and store.count() == n
    meta = next(iter(store.records.values()))["metadata"]
    assert {"doc", "section", "page", "scope", "status", "effective_date", "doc_type", "source_url", "text",
            "chunk_no", "chapter", "heading", "term"} <= set(meta)
    assert type(meta["page"]) is int


def test_dry_run_touches_nothing(act_pdf, services, store):
    assert ingest_file(act_pdf, META, store, dry_run=True).status == "dry-run"
    assert store.count() == 0 and catalog.read_manifest() == {}


def test_reindex_replaces_old_chunks_even_when_file_is_unchanged(act_pdf, store):
    first = ingest_file(act_pdf, META, store)
    # a vector left over from the old chunking scheme, same document prefix
    store.add("old 1,000-char chunk", doc="code-on-social-security-2020")
    store.records["code-on-social-security-2020#old"] = store.records.popitem()[1]
    assert ingest_file(act_pdf, META, store).status == "unchanged"
    again = ingest_file(act_pdf, META, store, reindex=True)
    assert again.status == "re-indexed" and again.deleted == first.chunks + 1
    assert "code-on-social-security-2020#old" not in store.records and store.count() == again.chunks


def test_reindex_includes_uploaded_pdfs(act_pdf, store, tmp_path, monkeypatch):
    from ingest import run as run_mod

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "esic-circular.pdf").write_bytes(act_pdf.read_bytes())
    monkeypatch.setattr(run_mod, "UPLOAD_DIR", uploads)
    catalog.update_manifest("ESIC circular", {"scope": "gujarat", "status": "current", "chunks": 3})
    rows = run_mod.uploaded_rows([])
    assert [r["doc"] for r in rows] == ["ESIC circular"] and rows[0]["scope"] == "gujarat"
    assert run_mod.uploaded_rows([{"doc": "ESIC circular"}]) == []  # listed in sources.csv: handled there
