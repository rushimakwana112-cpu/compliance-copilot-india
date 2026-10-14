"""Step 2 of ingestion: turn a PDF or HTML file into text, page by page.

Scanned PDFs have pages with no text layer. For those we try OCR
(pytesseract + pdf2image) if it's installed, and otherwise warn and skip the
page, so a missing OCR setup never silently produces an empty document.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("ingest")
MIN_CHARS = 40  # fewer characters than this on a page means "probably scanned"


@dataclass
class Page:
    number: int  # 1-based, as printed in a PDF viewer
    text: str
    ocr: bool = False


def _ocr_page(path: Path, number: int) -> str | None:
    """OCR one page. Returns None when the OCR tools aren't installed."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        return None
    try:
        images = convert_from_path(str(path), first_page=number, last_page=number, dpi=300)
        return pytesseract.image_to_string(images[0]) if images else ""
    except Exception as e:  # Tesseract or Poppler binary missing on PATH
        log.warning("OCR failed on %s p.%d: %s", path.name, number, e)
        return None


def parse_pdf(path: Path) -> list[Page]:
    """Extract text from every page, using OCR for pages without a text layer."""
    from pypdf import PdfReader

    pages: list[Page] = []
    warned = False
    for i, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) >= MIN_CHARS:
            pages.append(Page(i, text))
            continue
        ocr_text = _ocr_page(path, i)
        if ocr_text is None:
            if not warned:
                log.warning("%s has scanned pages but OCR is not installed; they are skipped (see README, OCR).", path.name)
                warned = True
            continue
        log.warning("%s p.%d had no text layer; used OCR", path.name, i)
        if ocr_text.strip():
            pages.append(Page(i, ocr_text.strip(), ocr=True))
    return pages


def parse_html(path: Path) -> list[Page]:
    """Main text of an HTML page, without navigation, header, footer, scripts and styles."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["nav", "header", "footer", "script", "style", "noscript", "aside", "form"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    lines = [ln.strip() for ln in root.get_text("\n").splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return [Page(1, text)] if text else []


def parse(path: Path) -> list[Page]:
    """Dispatch on file extension."""
    return parse_pdf(path) if path.suffix.lower() == ".pdf" else parse_html(path)
