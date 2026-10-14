"""State detection and scope resolution."""

from __future__ import annotations

from app.jurisdiction import detect_states, resolve
from app.vectorstore import scope_filter


def test_detects_city_alias():
    assert detect_states("Professional tax for 15000 salary in Ahmedabad?") == [("gujarat", "ahmedabad")]


def test_detects_state_name_case_insensitive():
    assert detect_states("PT rules in MAHARASHTRA")[0][0] == "maharashtra"


def test_whole_words_only():
    # "pune" must not match inside another word, "surat" not inside "surati"
    assert detect_states("punekar and surati food") == []


def test_longest_alias_wins():
    assert detect_states("office in New Delhi") == [("delhi", "new delhi")]


def test_two_states_in_order_without_duplicates():
    found = detect_states("Compare Gujarat and Maharashtra PT, Surat vs Pune")
    assert [s for s, _ in found] == ["gujarat", "maharashtra"]


def test_auto_falls_back_to_default():
    s = resolve("Is ESIC applicable on 24000 gross?", "auto", "karnataka")
    assert (s.states, s.how) == (["karnataka"], "default")


def test_auto_detects_over_default():
    s = resolve("PT in Pune", "auto", "gujarat")
    assert (s.states, s.how, s.matches) == (["maharashtra"], "detected", ["pune"])


def test_manual_ignores_question():
    s = resolve("PT in Pune", "manual", "gujarat")
    assert (s.states, s.how) == (["gujarat"], "manual")


def test_unknown_default_state_is_replaced():
    assert resolve("hello", "auto", "atlantis").states == ["gujarat"]


def test_label():
    assert resolve("Gujarat vs Maharashtra", "auto", "gujarat").label() == "Central + Gujarat + Maharashtra"


def test_pinecone_filter_shape():
    assert scope_filter(["gujarat"]) == {"status": "current", "scope": {"$in": ["central", "gujarat"]}}
