"""Repair words that PDF text extraction splits apart, like "gra tuity" or "G R A T U I T Y".

Government PDFs often use fonts whose spacing makes pypdf insert spaces inside
words. A split word no longer matches the question ("gratuity"), the heading
boosts, or the defined term, so we join fragments back together, but only when
the joined word is in a small legal vocabulary (so "pay ment" becomes "payment",
while ordinary pairs of words are left alone), and join spaced-out capital letters
inside headings.
"""

from __future__ import annotations

import re

LEGAL_WORDS = """
act acts allowance allowances amendment appropriate apprentice apprentices authority benefit benefits bonus
building central chapter chapters code commencement compensation construction continuous contract contractor
contractors contribution contributions controlling corporation dearness deduction deductions definition
definitions dependant dependants disablement disablement employee employees employer employers employment
establishment establishments exemption factory factories fixed government gratuity gratuities inspector
inspectors insurance interest interpretation labour maternity minimum nomination notification notifications
occupational overtime payment payments pension plantation preliminary prescribed proviso provident provisions
provided registration remuneration retirement retrenchment returns salary salaries schedule schedules scheme
section sections security service social superannuation termination wages worker workers workmen
payable applicable eligible eligibility entitled liability notified penalty penalties
""".split()
VOCAB = set(LEGAL_WORDS)
MAX_PARTS = 4

_TOKEN = re.compile(r"[A-Za-z]+")
_SPACED_CAPS = re.compile(r"\b(?:[A-Z] ){2,}[A-Z]\b")  # "G R A T U I T Y"
_HEADING = re.compile(r"^\s*(?:CHAPTER\b|\d{1,3}[A-Z]{0,2}\.\s+[A-Z]|[A-Z][A-Z '\-,]{6,}$)")


def _is_heading(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    upper = sum(c.isupper() for c in letters)
    return bool(_HEADING.match(line)) or (len(letters) >= 6 and upper / len(letters) >= 0.7)


def _join_vocab(line: str) -> str:
    """Join 2–4 neighbouring fragments separated by single spaces when the result is a legal word."""
    tokens = list(_TOKEN.finditer(line))
    out, pos, i = [], 0, 0
    while i < len(tokens):
        joined = None
        for n in range(min(MAX_PARTS, len(tokens) - i), 1, -1):
            group = tokens[i : i + n]
            if all(line[a.end() : b.start()] == " " for a, b in zip(group, group[1:])):
                word = "".join(t.group() for t in group)
                if word.lower() in VOCAB:
                    joined = (group[0].start(), group[-1].end(), word, n)
                    break
        if joined:
            start, end, word, n = joined
            out.append(line[pos:start] + word)
            pos, i = end, i + n
        else:
            i += 1
    return "".join(out) + line[pos:]


def fix_split_words(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        if _is_heading(line):
            line = _SPACED_CAPS.sub(lambda m: m.group().replace(" ", ""), line)
        lines.append(_join_vocab(line))
    return "\n".join(lines)
