"""Language and short forms: understand "pf kitna katega 18k" and answer in the asker's language.

* `detect` tells English from Hindi (Devanagari), Gujarati (Gujarati script) and Hinglish
  (Hindi written in Latin letters).
* `normalize` rewrites common short forms and Hinglish phrases into plain English for the
  router and calculators ("sal 25k in hand?" → "salary 25k in-hand?"). The original message is
  kept for the answer's language.
* `translate` turns a fixed English answer (calculator, guide) into the asker's language with the
  model, and keeps the English text if any number changed on the way.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from app.llm import LLM, LLMError

log = logging.getLogger("app")

Lang = Literal["en", "hi", "gu", "hinglish"]
NAMES = {"hi": "Hindi (Devanagari script)", "gu": "Gujarati (Gujarati script)",
         "hinglish": "Hinglish (Hindi in Latin letters, as the user wrote)"}

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
GUJARATI = re.compile(r"[઀-૿]")
# Hindi words written in Latin letters. Two or more of them mark a message as Hinglish.
HINGLISH_WORDS = re.compile(
    r"\b(kitna|kitni|kitne|katega|kategi|katta|kya|kyu|kyon|kaise|kaisa|kab|kaun|hai|hain|hoga|hogi|mera|meri|mere|"
    r"mujhe|muje|batao|bataiye|bata|nikalo|chahiye|milega|milegi|kar|karo|karna|wala|wali|ka|ki|ke|aur|se|pe|par|"
    r"tak|nahi|nahin|haan|ji|paisa|paise|tankhwah|vetan|salary\s+kitni)\b",
    re.IGNORECASE,
)

# Short forms and Hinglish phrases → plain English. Applied in order, whole words only.
SHORT_FORMS: list[tuple[str, str]] = [
    (r"\bkitna\s+katega\b|\bkitni\s+kategi\b|\bkitna\s+kat(?:ta|ega)\b", "how much is the deduction"),
    (r"\bkitna\s+milega\b|\bkitni\s+milegi\b|\bkitna\s+aayega\b", "how much will I get"),
    (r"\bkitna\b|\bkitni\b|\bkitne\b", "how much"),
    (r"\bkab\s+(?:tak\s+)?(?:bharna|jama|dena|bharni|deni)\b|\bkab\s+tak\b", "when is the due date"),
    (r"\bnikalo\b|\bbatao\b|\bbataiye\b|\bcalc\b|\bcalcu\b|\bcalci\b", "calculate"),
    (r"\bin\s*hand\b|\binhand\b|\bhaath\s+me\b|\bhath\s+me\b", "in-hand"),
    (r"\bsal\b|\bsalry\b|\bsallary\b|\bsalery\b|\btankhwah\b|\bvetan\b|\bpagaar\b|\bpagar\b", "salary"),
    (r"\bctc\b", "CTC"),
    (r"\bp\.?f\.?\b", "PF"),
    (r"\besi\b", "ESIC"),
    (r"\bp\.?t\.?\b(?=.*\b(?:tax|salary|slab|state|gujarat|maharashtra|how much|calculate|\d))", "professional tax"),
    (r"\bgrat(?:uity)?\b|\bgratuty\b|\bgratutiy\b", "gratuity"),
    (r"\bincr(?:ement)?\b|\bhike\b", "increment"),
    (r"\bresign(?:ed|ation)?\b", "resignation"),
    (r"\blop\b", "loss of pay"),
    (r"\bf\s*&\s*f\b|\bfnf\b|\bfull\s+n\s+final\b", "full and final settlement"),
    (r"\bmahina\b|\bmahine\b|\bmonthly\s+wise\b", "month"),
    (r"\bsaal\b|\bsalana\b", "year"),
    (r"\bmera\b|\bmeri\b|\bmere\b", "my"),
]


def detect(text: str) -> Lang:
    if GUJARATI.search(text):
        return "gu"
    if DEVANAGARI.search(text):
        return "hi"
    return "hinglish" if len(HINGLISH_WORDS.findall(text)) >= 2 else "en"


def normalize(text: str) -> str:
    """Plain-English version of the message for routing and parsing. Numbers are left untouched."""
    out = text
    for pattern, repl in SHORT_FORMS:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", out).strip()


NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[str]:
    return sorted(n.replace(",", "") for n in NUMBER.findall(text))


TRANSLATE_PROMPT = """Translate the user's text into {lang}. Rules:
- Keep every number, ₹ amount, percentage, date, form name, section and rule number exactly as written.
- Keep markdown (bold, bullets) as it is. Keep words like PF, ESIC, EPF, EPS, EDLI, TDS, CTC, HRA, UAN in English.
- Output only the translation."""


def translate(llm: LLM, text: str, lang: Lang) -> str:
    """The text in `lang`. Falls back to the English text if the model fails or changes any number."""
    if lang == "en" or not text.strip():
        return text
    try:
        out = llm.chat([{"role": "system", "content": TRANSLATE_PROMPT.format(lang=NAMES[lang])},
                        {"role": "user", "content": text}]).strip()
    except LLMError as e:
        log.info("Translation skipped: %s", e)
        return text
    if not out or _numbers(out) != _numbers(text):
        log.info("Translation dropped: numbers changed")
        return text
    return out


TO_ENGLISH_PROMPT = """Translate the user's message about Indian payroll into plain English for a search system.
Keep every number, ₹ amount, percentage, date and term like PF, ESIC, TDS, CTC exactly. Output only the translation."""


def to_english(llm: LLM, text: str, lang: Lang) -> str:
    """English version of a Hindi or Gujarati message, for routing and calculators.
    Hinglish and English only need `normalize`; if the model fails, the original text is used."""
    if lang not in ("hi", "gu"):
        return normalize(text)
    try:
        out = llm.chat([{"role": "system", "content": TO_ENGLISH_PROMPT}, {"role": "user", "content": text}]).strip()
    except LLMError as e:
        log.info("Translation to English skipped: %s", e)
        return text
    return normalize(out) if out else text
