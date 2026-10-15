"""Step 3 of ingestion: structure-aware chunking.

Laws are written in chapters, sections, rules and FAQ questions, and a good
answer usually comes from one of them. So we:

1. join the pages and split at those headings (regex), remembering the page
   each piece starts on, its chapter ("Chapter V — Gratuity") and its heading;
2. split a Definitions section into one chunk per defined term, prefixed with
   "<Doc>, Section 2 (Definitions) — <term>:" so the term is in the embedded text;
3. cap every other section at ~1,800 characters with 200 characters of overlap,
   cutting only where a sub-clause like (2) or (b) starts, never inside one.

Chunks are numbered across the whole document (`chunk_no`), so retrieval can
fetch a chunk's neighbours in the same section.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field

from ingest.parse import Page

MAX_CHARS = 1800
OVERLAP = 200
DEF_MAX_CHARS = 3000  # a definition up to this size stays in one chunk
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

# Where a numbered or lettered sub-clause starts: "(1) ", "(2A) ", "(a) ", "(zz) ", "(iv) ".
SUBCLAUSE = re.compile(r"(?:(?<=\s)|(?<=[—–:;.,-])|^)\((?:\d{1,3}[A-Z]?|[a-z]{1,2}|[ivxl]{1,6})\)\s", re.MULTILINE)
NUMBERED = re.compile(r"(?:(?<=\s)|(?<=[—–:;.,-])|^)\(\d{1,3}[A-Z]?\)\s", re.MULTILINE)

# A defined term: '(y) "wages" means', '(12) “employee” includes', or a line starting '"wages" means'.
DEFINITION = re.compile(
    r"""(?:\((?P<mark>\d{1,3}[A-Z]?|[a-z]{1,2})\)\s*|^\s*)
        [“"‘'](?P<term>[^”"’'\n]{1,80})[”"’']\s*,?\s*
        (?:means|includes|shall\s+(?:mean|include)|has\s+the\s+meaning|in\s+relation\s+to)\b""",
    re.MULTILINE | re.VERBOSE | re.IGNORECASE,
)
DEFINITIONS_HEADING = re.compile(r"\bdefinitions?\b|\binterpretation\b", re.IGNORECASE)


@dataclass
class Chunk:
    text: str
    page: int
    section: str
    chunk_no: int = 0  # position in the document, used to find neighbours
    chapter: str = ""  # "Chapter V — Gratuity"
    heading: str = ""  # the section's own title, e.g. "Payment of gratuity"
    term: str = ""  # the defined term, for chunks of a Definitions section


@dataclass
class Section:
    label: str
    chapter: str
    heading: str
    start: int  # offset in the joined document text
    text: str = ""
    pages: list[tuple[int, int]] = field(default_factory=list)  # (offset, page number) for page_at


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


def _chapter_title(text: str, end: int) -> str:
    """The line after 'CHAPTER V' (or the rest of its line) is the chapter's title."""
    for line in text[end : end + 200].splitlines()[:3]:
        line = line.strip(" .—–-")
        if line:
            return "" if re.match(r"^\d", line) or len(line) > 80 else line.title().replace("'S", "'s")
    return ""


def _join(pages: list[Page]) -> tuple[str, list[int], list[int]]:
    """All pages as one text, plus the offset where each page starts."""
    parts, offsets, numbers, pos = [], [], [], 0
    for page in pages:
        offsets.append(pos)
        numbers.append(page.number)
        parts.append(page.text)
        pos += len(page.text) + 1
    return "\n".join(parts), offsets, numbers


def segments(pages: list[Page], doc_type: str = "act") -> list[Section]:
    """Split the document at headings. Sections may run across page breaks."""
    text, offsets, numbers = _join(pages)

    def page_at(pos: int) -> int:
        return numbers[max(0, bisect.bisect_right(offsets, pos) - 1)] if numbers else 1

    out: list[Section] = []
    chapter = ""
    current = Section("", "", "", 0)
    for m in BOUNDARY.finditer(text):
        current.text = text[current.start : m.start()]
        out.append(current)
        label = _label(m, doc_type)
        heading = (m.group("title") or "").strip()
        if m.group("chapter"):
            title = _chapter_title(text, m.end())
            chapter = f"{label} — {title}" if title else label
        current = Section(label, chapter, heading, m.start())
    current.text = text[current.start :]
    out.append(current)

    # Merge tiny segments (a chapter title line) into the next one, and drop empty ones.
    merged: list[Section] = []
    carry: Section | None = None
    for sec in out:
        if not sec.text.strip():
            continue
        if carry:
            sec = Section(sec.label or carry.label, sec.chapter or carry.chapter, sec.heading or carry.heading,
                          carry.start, carry.text + sec.text)
            carry = None
        if len(sec.text.strip()) < MIN_SEGMENT:
            carry = sec
        else:
            merged.append(sec)
    if carry:
        merged.append(carry)
    for sec in merged:
        sec.pages = [(0, page_at(sec.start))] + [
            (o - sec.start, n) for o, n in zip(offsets, numbers) if sec.start < o < sec.start + len(sec.text)
        ]
    return merged


def _page(sec: Section, offset: int) -> int:
    """Page number where `offset` (inside the section's text) falls."""
    page = sec.pages[0][1]
    for start, number in sec.pages:
        if start <= offset:
            page = number
    return page


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def split_text(text: str, size: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Cap text at `size` chars, cutting at a sentence or word break, with `overlap` chars repeated."""
    text = _clean(text)
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


def _units(text: str, size: int) -> list[tuple[int, str]]:
    """(offset, text) pieces that are each at most `size`: whole numbered sub-clauses where possible,
    then lettered ones, then sentences for a single sub-clause that is still too long."""

    def cut(piece: str, base: int, pattern: re.Pattern[str] | None) -> list[tuple[int, str]]:
        if len(piece) <= size:
            return [(base, piece)]
        if pattern is None:
            out, pos = [], 0
            for part in split_text(piece, size, 0):
                found = piece.find(part[:40], pos)
                pos = found if found != -1 else pos
                out.append((base + pos, part))
            return out
        starts = sorted({0, *(m.start() for m in pattern.finditer(piece))})
        if len(starts) == 1:
            return cut(piece, base, SUBCLAUSE if pattern is NUMBERED else None)
        out: list[tuple[int, str]] = []
        for a, b in zip(starts, [*starts[1:], len(piece)]):
            out += cut(piece[a:b], base + a, SUBCLAUSE if pattern is NUMBERED else None)
        return out

    return cut(text, 0, NUMBERED)


def pack(text: str, size: int = MAX_CHARS, overlap: int = OVERLAP) -> list[tuple[int, str]]:
    """Group whole sub-clauses into chunks of at most `size` characters.

    Each chunk after the first starts with the last ~`overlap` characters of the previous
    one (marked with "…"), so a sentence cut at a boundary still has its lead-in.
    """
    chunks: list[tuple[int, str]] = []
    start, buf = 0, ""
    for offset, unit in _units(text, size):
        if buf and len(buf) + len(unit) > size:
            chunks.append((start, buf))
            buf = ""
        if not buf:
            start = offset
        buf += unit
    if buf.strip():
        chunks.append((start, buf))
    out: list[tuple[int, str]] = []
    for i, (offset, body) in enumerate(chunks):
        body = _clean(body)
        if i and overlap:
            tail = _clean(chunks[i - 1][1])[-overlap:]
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
            body = f"… {tail}\n{body}"
        if body:
            out.append((offset, body))
    return out


def is_definitions(sec: Section) -> bool:
    head = sec.heading or sec.text[:200]
    return bool(DEFINITIONS_HEADING.search(head)) and len(DEFINITION.findall(sec.text)) >= 2


def _section_no(label: str) -> str:
    m = re.search(r"(?:Section|Rule|Para)\s+(\d{1,3}[A-Z]{0,2})", label)
    return m.group(1) if m else "2"


def definition_chunks(sec: Section, doc: str) -> list[Chunk]:
    """One chunk per defined term, the whole definition (inclusions, exclusions, provisos) kept together."""
    matches = list(DEFINITION.finditer(sec.text))
    out: list[Chunk] = []
    intro = sec.text[: matches[0].start()]
    if len(_clean(intro)) >= MIN_SEGMENT:
        out.append(Chunk(_clean(intro), _page(sec, 0), sec.label, chapter=sec.chapter, heading=sec.heading))
    where = f"Section {_section_no(sec.label)} (Definitions)"
    for m, nxt in zip(matches, [*matches[1:], None]):
        end = nxt.start() if nxt else len(sec.text)
        body = sec.text[m.start() : end]
        term = _clean(m.group("term")).lower()
        prefix = f"{doc}, {where} — {term}:" if doc else f"{where} — {term}:"
        for offset, piece in pack(body, DEF_MAX_CHARS, OVERLAP):
            out.append(Chunk(f"{prefix} {piece}", _page(sec, m.start() + offset), sec.label,
                             chapter=sec.chapter, heading=sec.heading, term=term))
    return out


def chunk_pages(pages: list[Page], doc_type: str = "act", size: int = MAX_CHARS, overlap: int = OVERLAP,
                doc: str = "") -> list[Chunk]:
    """Full chunking pipeline: headings, then definitions per term, then the size cap."""
    chunks: list[Chunk] = []
    for sec in segments(pages, doc_type):
        if is_definitions(sec):
            chunks += definition_chunks(sec, doc)
            continue
        for offset, piece in pack(sec.text, size, overlap):
            chunks.append(Chunk(piece, _page(sec, offset), sec.label, chapter=sec.chapter, heading=sec.heading))
    for i, c in enumerate(chunks):
        c.chunk_no = i
    return chunks
