"""Language detection, short forms and translation that never changes a number."""

from __future__ import annotations

from app import lang
from tests.conftest import FakeLLM


def test_detect():
    assert lang.detect("How much PF on 18000?") == "en"
    assert lang.detect("pf kitna katega 18000 basic") == "hinglish"
    assert lang.detect("मेरा पीएफ कितना कटेगा") == "hi"
    assert lang.detect("મારો પગાર કેટલો") == "gu"


def test_normalize_short_forms():
    assert lang.normalize("sal 25k in hand?") == "salary 25k in-hand?"
    assert lang.normalize("pf kitna katega 18000 basic") == "PF how much is the deduction 18000 basic"
    assert lang.normalize("tds calc") == "tds calculate"
    assert "gratuity" in lang.normalize("grat for 5 yrs")


def test_translate_keeps_numbers_or_falls_back():
    good = FakeLLM(lambda m, j: "पीएफ: ₹2,160 (12%)")
    assert lang.translate(good, "PF: ₹2,160 (12%)", "hi") == "पीएफ: ₹2,160 (12%)"
    bad = FakeLLM(lambda m, j: "पीएफ: ₹2,100 (12%)")
    assert lang.translate(bad, "PF: ₹2,160 (12%)", "hi") == "PF: ₹2,160 (12%)"
    assert lang.translate(good, "same", "en") == "same"


def test_to_english_only_calls_the_model_for_hindi_and_gujarati():
    llm = FakeLLM(lambda m, j: "how much pf for basic 18000")
    assert lang.to_english(llm, "sal 25k in hand?", "hinglish") == "salary 25k in-hand?" and llm.calls == []
    assert lang.to_english(llm, "बेसिक 18000 पर पीएफ कितना", "hi") == "how much PF for basic 18000"
