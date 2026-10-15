"""Guards for the public demo: a per-IP rate limiter and an answer cache.

Both live in memory. That is enough for one free Render instance (one
process); with several instances each would keep its own counts, and a
restart resets them. Groq's own free-tier limit is the real backstop.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass

from app.schemas import ChatResponse

CACHEABLE = {"rag", "tool", "sql"}  # never cache refusals or errors: they may succeed next time


# ----------------------------------------------------------------- rate limit
@dataclass
class Verdict:
    allowed: bool
    retry_after: int = 0  # seconds until the next request would be allowed
    reason: str = ""  # "minute" or "day" when refused


class RateLimiter:
    """At most `per_min` requests in any 60 s window and `per_day` per calendar day (UTC), per key."""

    def __init__(self, per_min: int, per_day: int, clock: Callable[[], float] = time.time) -> None:
        self.per_min, self.per_day, self.clock = per_min, per_day, clock
        self._recent: dict[str, deque[float]] = {}
        self._daily: dict[str, int] = {}
        self._day = -1
        self._lock = threading.Lock()

    def hit(self, key: str) -> Verdict:
        """Count one request for `key`, or refuse it without counting."""
        now = self.clock()
        day = int(now // 86400)
        with self._lock:
            if day != self._day:  # a new day: forget yesterday's counts
                self._day, self._daily = day, {}
                self._recent = {k: q for k, q in self._recent.items() if q and now - q[-1] < 60}
            if self._daily.get(key, 0) >= self.per_day:
                return Verdict(False, int((day + 1) * 86400 - now) + 1, "day")
            window = self._recent.setdefault(key, deque())
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= self.per_min:
                return Verdict(False, int(60 - (now - window[0])) + 1, "minute")
            window.append(now)
            self._daily[key] = self._daily.get(key, 0) + 1
            return Verdict(True)


def limit_message(v: Verdict, per_min: int, per_day: int) -> str:
    if v.reason == "day":
        return f"You've asked {per_day} questions today, the daily limit for this free demo. Please come back tomorrow."
    return f"That's more than {per_min} questions in a minute. Please wait {v.retry_after} seconds and try again."


# ----------------------------------------------------------------- answer cache
def normalize(question: str) -> str:
    """'  What is the PF  rate?? ' and 'what is the pf rate' share one cache entry."""
    q = re.sub(r"\s+", " ", question.lower()).strip()
    return re.sub(r"[\s?.!]+$", "", q)


class AnswerCache:
    """LRU cache of finished answers, keyed by the standalone question and the states searched."""

    def __init__(self, size: int = 500) -> None:
        self.size = size
        self._data: OrderedDict[tuple[str, tuple[str, ...]], ChatResponse] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(question: str, states: list[str]) -> tuple[str, tuple[str, ...]]:
        return normalize(question), tuple(sorted(states))

    def get(self, question: str, states: list[str]) -> ChatResponse | None:
        k = self.key(question, states)
        with self._lock:
            hit = self._data.get(k)
            if hit is None:
                return None
            self._data.move_to_end(k)
        resp = hit.model_copy(deep=True)
        resp.cached = True
        return resp

    def put(self, question: str, states: list[str], resp: ChatResponse) -> None:
        if self.size <= 0 or resp.type not in CACHEABLE or resp.quota_exceeded or resp.rate_limited:
            return
        k = self.key(question, states)
        with self._lock:
            self._data[k] = resp.model_copy(deep=True)
            self._data.move_to_end(k)
            while len(self._data) > self.size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        """New documents can change answers, so ingestion empties the cache."""
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
