"""Step 3 of ingestion: structure-aware chunking.

Laws are written in sections, rules and FAQ questions, and a good answer
usually comes from one of them. So we first split on those headings (regex),
and only then cap each piece at ~1,000 characters with 150 characters of
overlap. Every chunk remembers its section label and the page it starts on,
which become the citation shown to the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ingest.parse import Page

MAX_CHARS = 1000
OVERLAP = 150
MIN_SEGMENT = 80  # heading-only segments (a lone "CHAPTER V  GRATUITY") are merged into the next one

# Each alternative captures a heading at the start of a line.
BOUNDARY = re.compile(
    r"""^\s*(?:
        (?P<chapter>CHAPTER\s+[IVXLC]+\b)                                   # CHAPTER VI
      | (?:Section|SECTION)\s+(?P<sec>\d{1,3}[A-Z]{0,2})\b                   # Section 53
      | (?:Rule|RULE)\s+(?P<rule>\d{1,3}[A-Z]{0,2})\b                        # Rule 12
      | (?P<num>\d{1,3}[A-Z]{0,2})\.\s+(?P<title>[A-Z][^\n.—–]{2,100}?)\s*\.?\s*[—–-]  # 53. Gratuity.—
      | (?:Q\.?|Question)\s*(?P<q>\d{1,3})\s*[.:)-]                          # Q.5: / Question 5)
    )""",
    re.MULTILINE | re.VERBOSE,
)


@dataclass
class Chunk:
    text: str
    page: int
    section: str
    chunk_no: int = 0


def _label(m: re.Match[str], doc_type: str) -> str:
    g = m.groupdict()
    if g["chapter"]:
        return "Chapter " + g["chapter"].split()[-1]
    if g["sec"]:
        return f"Section {g['sec']}"
    if g["rule"]:
        return f"Rule {g['rule']}"
    if g["num"]:
        word = "Rule" if doc_type == "rules" else "Para" if doc_type in ("circular", "notification") else "Section"
        return f"{word} {g['num']} · {g['title'].strip()}"
    return f"FAQ {g['q']}"


def segments(pages: list[Page], doc_type: str = "act") -> list[Chunk]:
    """Split pages at headings. A section that runs over a page break keeps its label on the next page."""
    out: list[Chunk] = []
    label = ""
    for page in pages:
        text = page.text
        cuts = [(m.start(), _label(m, doc_type)) for m in BOUNDARY.finditer(text)]
        starts = [0, *(c for c, _ in cuts)]
        labels = [label, *(lb for _, lb in cuts)]
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(text)
            piece = text[start:end].strip()
            if piece:
                out.append(Chunk(piece, page.number, labels[i]))
        if cuts:
            label = cuts[-1][1]
    # Merge tiny segments (e.g. a chapter title line) into the following one for context.
    merged: list[Chunk] = []
    carry: Chunk | None = None
    for seg in out:
        if carry:
            seg = Chunk(f"{carry.text}\n{seg.text}", carry.page, seg.section or carry.section)
            carry = None
        if len(seg.text) < MIN_SEGMENT:
            carry = seg
        else:
            merged.append(seg)
    if carry:
        merged.append(carry)
    return merged


def split_text(text: str, size: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Cap text at `size` chars, cutting at a sentence or word break, with `overlap` chars repeated."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start + size // 2 : end]
            brk = max(window.rfind(". "), window.rfind("\n"), window.rfind("; "))
            if brk == -1:
                brk = window.rfind(" ")
            if brk != -1:
                end = start + size // 2 + brk + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        nxt = max(end - overlap, start + 1)
        space = text.find(" ", nxt)  # don't start the next chunk mid-word
        start = space + 1 if space != -1 and space < end else nxt
    return [p for p in parts if p]


def chunk_pages(pages: list[Page], doc_type: str = "act", size: int = MAX_CHARS, overlap: int = OVERLAP) -> list[Chunk]:
    """Full chunking pipeline: headings first, then size cap. Chunks are numbered per page."""
    chunks: list[Chunk] = []
    per_page: dict[int, int] = {}
    for seg in segments(pages, doc_type):
        for piece in split_text(seg.text, size, overlap):
            n = per_page.get(seg.page, 0)
            per_page[seg.page] = n + 1
            chunks.append(Chunk(piece, seg.page, seg.section, n))
    return chunks
