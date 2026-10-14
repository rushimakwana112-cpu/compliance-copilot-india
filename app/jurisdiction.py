"""Which state's rules apply to a question.

Central law is always searched. On top of that we pick one or more states:

* manual mode   → the state chosen in the sidebar, nothing else.
* auto mode     → states named in the question (state or city aliases), in the
                  order they appear; if none, the user's default state.

The alias list lives in data/states.yaml and is also served to the frontend,
so both sides detect states the same way. The backend re-checks detection
rather than trusting the browser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml

from app.config import DATA_DIR

MAX_STATES = 3  # a comparison of more than three states would swamp the context window


@dataclass(frozen=True)
class State:
    id: str
    name: str
    aliases: tuple[str, ...]


@dataclass
class Scope:
    states: list[str]
    how: Literal["detected", "default", "manual"]
    matches: list[str] = field(default_factory=list)

    @property
    def primary(self) -> str:
        return self.states[0]

    def label(self, states: list[State] | None = None) -> str:
        """Human label, e.g. 'Central + Gujarat + Maharashtra'."""
        names = {s.id: s.name for s in (states or load_states())}
        return " + ".join(["Central", *(names.get(s, s) for s in self.states)])


@lru_cache
def load_states(path: Path = DATA_DIR / "states.yaml") -> list[State]:
    """Load the state list from YAML (cached)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [State(r["id"], r["name"], tuple(a.lower() for a in r.get("aliases", []))) for r in raw]


@lru_cache
def _alias_pattern(path: Path = DATA_DIR / "states.yaml") -> tuple[re.Pattern[str], dict[str, str]]:
    alias_to_id = {a: s.id for s in load_states(path) for a in s.aliases}
    # Longest alias first so "new delhi" wins over "delhi"; \b = whole words only ("pune" not in "punegaon").
    alts = sorted(alias_to_id, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(a) for a in alts) + r")\b", re.IGNORECASE)
    return pattern, alias_to_id


def detect_states(text: str) -> list[tuple[str, str]]:
    """Return [(state_id, matched_alias)] in order of first appearance, without duplicates."""
    pattern, alias_to_id = _alias_pattern()
    found: dict[str, str] = {}
    for m in pattern.finditer(text or ""):
        sid = alias_to_id[m.group(1).lower()]
        found.setdefault(sid, m.group(1).lower())
    return list(found.items())[:MAX_STATES]


def is_known_state(state_id: str) -> bool:
    return any(s.id == state_id for s in load_states())


def resolve(text: str, mode: str, default_state: str) -> Scope:
    """Decide which states to search for `text` (central is always added at query time)."""
    default = default_state if is_known_state(default_state) else load_states()[0].id
    if mode == "manual":
        return Scope([default], "manual")
    found = detect_states(text)
    if found:
        return Scope([sid for sid, _ in found], "detected", [alias for _, alias in found])
    return Scope([default], "default")
