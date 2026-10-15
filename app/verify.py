"""'Is this calculation correct? PF on basic 18,000 = 2,160': recalculate and compare with the user's figure.

The claimed figure is taken out of the message, the rest goes through the normal calculators, and the
answer says whether the user's figure matches the recalculated one, with the steps to check against.
"""

from __future__ import annotations

import re

from app.schemas import ChatResponse
from app.tools import inr

VERIFY = re.compile(
    r"\b(?:is (?:this|my|the|it) (?:calculation |calc |answer |amount |figure |math )?(?:correct|right|ok)|"
    r"(?:check|verify|recheck|validate) (?:my|this|the) (?:calculation|calc|math|figure|answer)|am i right|sahi hai)\b\??",
    re.IGNORECASE,
)
CLAIM = re.compile(r"(?:=|comes? to|is coming|i got|i get|getting|works? out to|equals|total(?: is)?|should be|\bis)\s*"
                   r"(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
MONEY = re.compile(r"[-]?₹[\d,]+")


def split(text: str) -> tuple[str, float] | None:
    """(the question without the check and the claimed figure, the claimed figure), or None."""
    if not VERIFY.search(text):
        return None
    claims = list(CLAIM.finditer(text))
    if not claims:
        return None
    last = claims[-1]
    if re.search(r"\d", text[last.end():]):  # "basic is 18000, pf 2160": the figure to check comes last
        return None
    claim = float(last.group(1).replace(",", ""))
    rest = VERIFY.sub(" ", text[: last.start()] + " " + text[last.end():])
    return re.sub(r"\s{2,}", " ", rest).strip(" .,?;:"), claim


def compare(resp: ChatResponse, claim: float) -> str | None:
    """A line saying whether `claim` matches a figure in the recalculated answer (to the rupee)."""
    figures = [cell for row in (resp.table.rows if resp.table else []) for cell in row[1:] if MONEY.fullmatch(str(cell))]
    figures += MONEY.findall(resp.answer)
    if not figures:
        return None
    mine = inr(claim)
    if mine in figures:
        return f"Your figure {mine} matches my calculation. The steps are below."
    if resp.table and len(resp.table.rows) > 1:
        # "PF on basic 18000 = 2500" means the employee's share, not the employee + employer total: name the
        # row closest to the user's figure, and the total only if that is a different row
        labelled = [(str(r[0]), str(r[-1])) for r in resp.table.rows if MONEY.fullmatch(str(r[-1]))]
        if labelled:
            label, value = min(labelled, key=lambda lv: abs(_rupees(lv[1]) - claim))
            line = f"Your figure {mine} doesn't match my calculation. The closest figure is {label}: {value}"
            last_label, last_value = labelled[-1]
            if (label, value) != (last_label, last_value):
                line += f" ({last_label}: {last_value})"
            return line + ". Compare your steps with the ones below."
    bold = re.search(r"\*\*(-?₹[\d,]+)", resp.answer)
    result = bold.group(1) if bold else (resp.table.rows[-1][-1] if resp.table else figures[-1])
    return f"Your figure {mine} doesn't match: I get {result}. Compare your steps with the ones below."


def _rupees(text: str) -> float:
    return float(text.replace("₹", "").replace(",", ""))
