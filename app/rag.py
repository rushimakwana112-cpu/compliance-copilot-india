"""Retrieval-augmented answers with citations.

    follow-up rewrite → embed → Pinecone (jurisdiction filter) → re-rank → threshold
    → prompt with numbered passages → Qwen → strip invalid [n] markers

Below the score threshold we never call the LLM: a weak match is the moment a
model is most likely to make something up, so we refuse instead.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from app.jurisdiction import Scope
from app.llm import LLM, LLMError
from app.schemas import Citation, ChatResponse, ScopeOut, Turn
from app.services import Services
from app.vectorstore import scope_filter

DOC_TYPE_BOOST = 0.03
DOC_NAME_BOOST = 0.08  # the question names this document ("under the Code on Wages")
TERM_BOOST = 0.08  # "definition of wages" and this chunk defines "wages"
HEADING_BOOST = 0.05  # the chunk's chapter or heading is about the question's topic
COMMENTARY_PENALTY = 0.10  # end matter (Statement of Objects and Reasons…) ranks below the operative text
SEARCHED = ("current", "commentary")  # statuses searched by default; repealed laws only when asked
OLD = {"repealed", "historical"}
SNIPPET_CHARS = 320
NEIGHBOUR_CHARS = 900  # how much of the previous / next chunk is added around each passage
NOT_FOUND = "NOT_FOUND"
log = logging.getLogger("app")

# Words that hint which kind of document answers the question best.
DOC_TYPE_HINTS = {
    "circular": r"\b(rate|ceiling|limit|due date|deadline|interest|damages|circular)\b",
    "notification": r"\b(notif\w*|ceiling|wage limit|effective from|w\.e\.f)\b",
    "faq": r"\b(can|does|do|is|are|should)\b.*\?$",
    "rules": r"\b(rules?|form|register|return|filing|procedure)\b",
    "act": r"\b(section|act|code|definition|defined|means)\b",
}
REPEALED_HINT = re.compile(
    r"\b(repealed|old (?:act|law)|earlier (?:act|law)|before the (?:labour )?codes?|"
    r"(?:epf|esi|gratuity|bonus|minimum wages|payment of wages) act|19[4-9]\d)\b",
    re.IGNORECASE,
)
# A follow-up either opens like one ("and for Gujarat?", "what about Kerala?") or leans on a pronoun
# without naming its own topic ("is it taxable?"). Anything else is used exactly as typed.
FOLLOW_UP_START = re.compile(r"^\s*(and|what about|how about|same for|same in|same|also|what if|then)\b", re.I)
BARE_START = re.compile(r"^\s*(for|in)\b", re.I)  # "in Pune?" only counts when it names no topic
PRONOUN = re.compile(r"\b(it|its|that|this|these|those|they|them|there)\b", re.I)
TOPIC = re.compile(
    r"\b(pf|epf|eps|epfo|esic?|gratuity|bonus|wages?|salary|salaries|pay|payroll|pt|professional tax|tds|tax|"
    r"leave|overtime|maternity|retrenchment|notice|contract\w*|employees?|employers?|workers?|"
    r"code|act|rules?|section|ceiling|rate|contribution|deduction|returns?|registers?)\b",
    re.I,
)


class SearchError(RuntimeError):
    """Pinecone failed or returned nothing: report it instead of a silent not_found."""


@dataclass
class Trace:
    """What retrieval did, for logs/chats.jsonl: index, filters and every match."""

    index: str = ""
    filters: list[dict[str, Any]] = field(default_factory=list)
    matches: dict[str, tuple[float, str, str]] = field(default_factory=dict)  # id -> (score, doc, section)
    searched: bool = False
    prompt_tokens: int | None = None  # estimated size of the answer prompt (chars / 4)
    dropped: int = 0  # passages dropped to fit the token budget
    rate_limit: dict[str, Any] | None = None  # which provider limit refused the request

    def add(self, mid: str, score: float, meta: dict[str, Any]) -> None:
        old = self.matches.get(mid)
        if old is None or score > old[0]:
            self.matches[mid] = (score, str(meta.get("doc", "")), str(meta.get("section", "")))

    def ranked(self) -> list[tuple[float, str, str]]:
        return sorted(self.matches.values(), key=lambda m: m[0], reverse=True)

    def log_fields(self, n: int = 5) -> dict[str, Any]:
        ranked = self.ranked()
        return {
            "index": self.index,
            "filters": self.filters,
            "top_scores": [round(m[0], 3) for m in ranked[:n]],
            "retrieved": [{"doc": d, "section": sec, "score": round(sc, 3)} for sc, d, sec in ranked],
            "retrieval": "searched" if self.searched else "not run",
            "prompt_tokens": self.prompt_tokens,
            "dropped_passages": self.dropped,
            "rate_limit": self.rate_limit,
        }


def as_int(value: Any) -> int | None:
    """Pinecone returns numbers as floats (40.0); pages and chunk numbers are whole numbers."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


@dataclass
class Passage:
    doc: str
    section: str
    page: int | None
    score: float  # raw similarity from Pinecone
    rank_score: float  # score + boosts, used only for ordering
    text: str
    url: str
    status: str
    effective_date: str
    doc_type: str
    scope: str
    chunk_no: int | None = None
    chapter: str = ""
    heading: str = ""
    term: str = ""
    context: str = ""  # text sent to the model: the chunk plus the end of the previous and start of the next
    pinned: bool = False  # defines the question's term: sent to the model whatever its score

    def citation(self) -> Citation:
        """Citations always point at the main chunk, not at the neighbours added for context."""
        snippet = self.text if len(self.text) <= SNIPPET_CHARS else self.text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
        return Citation(doc=self.doc, section=self.section, page=self.page, score=round(self.score, 3),
                        snippet=snippet, url=self.url, status=self.status)


# ----------------------------------------------------------------- follow-ups
REWRITE_PROMPT = """Rewrite the user's LAST message as one standalone question about Indian payroll or labour law.
Keep the topic, state and amounts from earlier turns unless the last message changes them.
Return only the rewritten question, nothing else."""


def needs_rewrite(message: str, history: list[Turn]) -> bool:
    """True only for clear follow-ups. "Gratuity rule in Dubai?" names its own topic, so it stays as typed."""
    if not history:
        return False
    if FOLLOW_UP_START.search(message):
        return True
    has_topic = bool(TOPIC.search(message))
    return not has_topic and bool(PRONOUN.search(message) or BARE_START.search(message))


def rewrite_followup(llm: LLM, message: str, history: list[Turn], turns: int = 4) -> str:
    """Turn 'and for Gujarat?' into a full question using the last `turns` exchanges."""
    if not needs_rewrite(message, history):
        return message
    convo = "\n".join(f"{t.role}: {t.content[:600]}" for t in history[-turns * 2 :])
    try:
        out = llm.chat([
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": f"Conversation:\n{convo}\n\nLast message: {message}"},
        ]).strip().strip('"')
    except LLMError:
        return message
    return out if 5 <= len(out) <= 400 and out != NOT_FOUND else message


# ----------------------------------------------------------------- retrieval
def wanted_doc_types(question: str) -> set[str]:
    return {t for t, pat in DOC_TYPE_HINTS.items() if re.search(pat, question.strip(), re.IGNORECASE)}


# Ways people name a document → words that must all appear in the document's name.
DOC_NAMES: list[tuple[str, list[tuple[str, ...]]]] = [
    (r"\bcode on wages\b|\bwages? code\b", [("code", "wages")]),
    (r"\bcode on social security\b|\bsocial security code\b|\bss code\b", [("social", "security")]),
    (r"\bindustrial relations code\b|\bcode on industrial relations\b|\bir code\b", [("industrial", "relations")]),
    (r"\bosh code\b|\boccupational safety\b", [("occupational", "safety")]),
    (r"\bepf scheme\b|\bprovident funds? scheme\b", [("provident", "scheme"), ("epf", "scheme")]),
    (r"\beps\b|\bpension scheme\b", [("pension", "scheme")]),
    (r"\besi act\b|\bstate insurance act\b", [("state", "insurance", "act"), ("esi", "act")]),
    (r"\bgratuity act\b", [("gratuity", "act")]),
    (r"\bbonus act\b", [("bonus", "act")]),
    (r"\bminimum wages act\b", [("minimum", "wages", "act")]),
    (r"\bpayment of wages act\b", [("payment", "wages", "act")]),
    (r"\bmaternity benefit act\b", [("maternity", "act")]),
]
# Question topic → words to look for in a chunk's chapter, heading or section label.
HEADING_TOPICS: list[tuple[str, str]] = [
    (r"\bgratuity\b", r"gratuity"),
    (r"\bbonus\b", r"bonus"),
    (r"\bmaternity\b", r"maternity"),
    (r"\besic?\b|state insurance", r"state insurance|\besic?\b"),
    (r"\bp\.?f\b|\bepf\b|provident fund", r"provident fund"),
    (r"\bpension\b|\beps\b", r"pension"),
    (r"\bminimum wages?\b", r"minimum wage"),
    (r"\bovertime\b", r"overtime"),
    (r"\bretrench\w*|\blay-?off\b|\bclosure\b", r"retrench|lay-?off|closure"),
    (r"\bleave\b", r"leave"),
    (r"\bdefinitions?\b|\bdefine\b|\bmeaning\b", r"definition"),
]
TERM_QUESTION = re.compile(
    r"\b(?:(?:what|who) (?:is|are) (?:the )?(?:definition of |meaning of |meant by )?|definition of|define|meaning of)"
    r"\s*(?:the |a |an )?(?:term |word )?[\"'“‘]?"
    r"(?P<term>[a-z][a-z' -]{1,40}?)[\"'”’]?\s*(?:\b(?:under|in|as per|according to|per|for)\b.*)?[?.]*\s*$",
    re.IGNORECASE,
)


def _words(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


EMPLOY_WORDS = {"employee", "employer", "employment"}
# "who is covered ..." asks about coverage, not about a defined term.
NOT_TERMS = {"covered", "applicable", "eligible", "liable", "entitled", "responsible", "required", "exempt", "it", "this"}


def _word_singular(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    return w[:-1] if w.endswith("s") and not w.endswith(("ss", "us", "is")) else w


def _singular(term: str) -> str:
    """Lowercase, no quotes or hyphens, single spaces, each word singular: '“Fixed-Term Employees”' → 'fixed term employee'."""
    term = re.sub(r"[-‐–]", " ", term.lower())
    words = re.sub(r"[^a-z0-9' ]+", " ", term).replace("'", " ").split()
    return " ".join(_word_singular(w) for w in words)


def term_key(term: str) -> str:
    """Comparable form of a defined term. In a compound ("fixed term employee"), employee, employer and
    employment are treated as the same word, because questions and Acts word these differently."""
    words = _singular(term).split()
    if len(words) > 1 and words[-1] in EMPLOY_WORDS:
        words[-1] = "employ*"
    return " ".join(words)


def same_term(a: str, b: str) -> bool:
    return bool(a and b) and term_key(a) == term_key(b)


def term_variants(term: str) -> list[str]:
    """Spellings a chunk's `term` metadata might use for `term`, for an exact-match filter."""
    words = _singular(term).split()
    if not words:
        return []
    last = words[-1]
    lasts = sorted(EMPLOY_WORDS) if len(words) > 1 and last in EMPLOY_WORDS else [last]
    out: list[str] = []
    for end in lasts:
        for form in (end, end + "s" if not end.endswith("s") else end):
            head = words[:-1]
            out.append(" ".join([*head, form]))
            if len(head) >= 1:
                out.append("-".join(head) + " " + form if len(head) > 1 else head[0] + "-" + form)
    return sorted(set(out))


@dataclass
class Signals:
    """What the question says about where the answer lives, used to re-rank matches."""

    doc_types: set[str]
    doc_keys: list[tuple[str, ...]]  # each tuple: words a named document's title must contain
    term: str  # "wages" for "definition of wages"
    topics: list[str]  # heading patterns to boost

    @classmethod
    def of(cls, question: str) -> Signals:
        q = question.lower()
        keys = [k for pat, ks in DOC_NAMES if re.search(pat, q) for k in ks]
        m = TERM_QUESTION.search(question.strip())
        term = _singular(m.group("term")) if m else ""
        if term in NOT_TERMS:
            term = ""
        topics = [head for pat, head in HEADING_TOPICS if re.search(pat, q)]
        return cls(wanted_doc_types(question), keys, term, topics)

    def names_doc(self, doc: str) -> bool:
        words = _words(doc)
        return any(all(k in words for k in ks) for ks in self.doc_keys)

    def boost(self, p: Passage) -> float:
        b = DOC_TYPE_BOOST if p.doc_type in self.doc_types else 0.0
        if self.doc_keys and self.names_doc(p.doc):
            b += DOC_NAME_BOOST
        if same_term(self.term, p.term):
            b += TERM_BOOST
        where = f"{p.chapter} {p.heading} {p.section}".lower()
        if any(re.search(t, where) for t in self.topics):
            b += HEADING_BOOST
        if p.status == "commentary":
            b -= COMMENTARY_PENALTY
        return b


def _passage(meta: dict, score: float, signals: Signals) -> Passage:
    doc_type = str(meta.get("doc_type", ""))
    p = Passage(
        doc=str(meta.get("doc", "")), section=str(meta.get("section", "")), page=as_int(meta.get("page")),
        score=score, rank_score=score,
        text=str(meta.get("text", "")), url=str(meta.get("source_url", "")), status=str(meta.get("status", "current")),
        effective_date=str(meta.get("effective_date", "")), doc_type=doc_type, scope=str(meta.get("scope", "")),
        chunk_no=as_int(meta.get("chunk_no")), chapter=str(meta.get("chapter", "") or ""),
        heading=str(meta.get("heading", "") or ""), term=str(meta.get("term", "") or ""),
    )
    p.rank_score = score + signals.boost(p)
    return p


def _strip_overlap(text: str) -> str:
    """Drop the '… <end of previous chunk>' line that chunking puts at the start of a chunk."""
    return text.split("\n", 1)[1] if text.startswith("… ") and "\n" in text else text


def expand_neighbours(services: Services, question: str, passages: list[Passage]) -> None:
    """Fill `context` with the end of the previous and the start of the next chunk of the same doc and section.

    One filtered search per (doc, section) finds the neighbours by `chunk_no`. A failure here only
    loses context, so it is logged and the passages are used on their own.
    """
    wanted: dict[tuple[str, str], set[int]] = {}
    for p in passages:
        if p.chunk_no is not None:
            wanted.setdefault((p.doc, p.section), set()).update({p.chunk_no - 1, p.chunk_no + 1})
    found: dict[tuple[str, str, int], str] = {}
    for (doc, section), numbers in wanted.items():
        nums = sorted(n for n in numbers if n >= 0)
        if not nums or services.store is None:
            continue
        flt = {"doc": doc, "section": section, "chunk_no": {"$in": nums}}
        try:
            matches = services.store.search(question, len(nums), flt)
        except Exception as e:  # noqa: BLE001 — neighbours are a bonus, never a failure
            log.warning("Neighbour lookup failed for %s / %s: %s", doc, section, e)
            continue
        for m in matches:
            n = as_int(m.metadata.get("chunk_no"))
            if n is not None:
                found[(doc, section, n)] = str(m.metadata.get("text", ""))
    for p in passages:
        prev = found.get((p.doc, p.section, p.chunk_no - 1), "") if p.chunk_no is not None else ""
        nxt = found.get((p.doc, p.section, p.chunk_no + 1), "") if p.chunk_no is not None else ""
        parts = []
        if prev:
            tail = _strip_overlap(prev)[-NEIGHBOUR_CHARS:]
            parts.append("… " + (tail.split(" ", 1)[1] if " " in tail else tail))
        parts.append(_strip_overlap(p.text) if prev else p.text)
        if nxt:
            head = _strip_overlap(nxt)[:NEIGHBOUR_CHARS]
            parts.append((head.rsplit(" ", 1)[0] if " " in head else head) + " …")
        p.context = "\n".join(parts)


def retrieve(services: Services, question: str, scope: Scope, trace: Trace | None = None) -> list[Passage]:
    """Top passages for `question`. For a comparison, each state gets its own retrieval so neither is crowded out.

    Raises SearchError when Pinecone is missing, fails, or returns no matches at all: an empty
    result means a wrong index or filter, not "the documents don't cover this".
    """
    s = services.settings
    trace = trace if trace is not None else Trace()
    trace.index = trace.index or s.pinecone_index
    if services.store is None:
        raise SearchError(services.store_error or "Pinecone is not configured")
    statuses = (*SEARCHED, "repealed", "historical") if REPEALED_HINT.search(question) else SEARCHED
    signals = Signals.of(question)
    per_state_keep = s.keep_k if len(scope.states) == 1 else max(2, s.keep_k - 2)
    kept: dict[tuple, Passage] = {}
    trace.searched = True
    for state in scope.states:
        flt = scope_filter([state], statuses)
        trace.filters.append(flt)
        try:
            matches = services.store.search(question, s.top_k, flt)
        except Exception as e:  # network, auth, quota or a bad filter
            raise SearchError(f"{e.__class__.__name__}: {str(e)[:160]}") from e
        for m in matches:
            trace.add(m.id, m.score, m.metadata)
        candidates = [_passage(m.metadata, m.score, signals) for m in matches]
        if signals.doc_keys and len(scope.states) == 1:
            # "under the Code on Wages": when that document was found, answer from it alone.
            named = [p for p in candidates if signals.names_doc(p.doc)]
            candidates = named or candidates
        ranked = sorted(candidates, key=lambda p: p.rank_score, reverse=True)
        for p in ranked[:per_state_keep]:
            kept.setdefault((p.doc, p.page, p.text[:80]), p)
    pinned = _definition_passages(services, question, scope, statuses, signals, trace)
    pinned += [p for p in _applicability_passages(services, question, scope, statuses, signals, trace)
               if (p.doc, p.page, p.text[:80]) not in {(q.doc, q.page, q.text[:80]) for q in pinned}]
    for p in pinned:
        kept.pop((p.doc, p.page, p.text[:80]), None)
    if not trace.matches:
        raise SearchError(f"Pinecone index '{trace.index}' returned 0 matches for filter {trace.filters}")
    return pinned + sorted(kept.values(), key=lambda p: p.rank_score, reverse=True)


# "Who is covered for gratuity?" needs two things: which employees qualify (usually in the chapter itself)
# and which establishments the chapter applies to (often in Section 1 or a Schedule, far from the chapter).
COVERAGE_QUESTION = re.compile(
    r"\b(?:who (?:is|are) (?:covered|eligible|entitled)|covered (?:under|by)|coverage|applicab\w*|"
    r"appl(?:y|ies) to|eligib\w*|entitled to)\b", re.IGNORECASE)
COVERAGE_TOPICS: dict[str, tuple[str, str]] = {  # name → (question pattern, text pattern)
    "gratuity": (r"\bgratuity\b", r"gratuity"),
    "bonus": (r"\bbonus\b", r"bonus"),
    "employees' state insurance": (r"\besic?\b|state insurance", r"state insurance|\besic?\b"),
    "provident fund": (r"\bp\.?f\b|\bepf\b|provident fund", r"provident fund"),
    "maternity benefit": (r"\bmaternity\b", r"maternity"),
}
APPLIES = re.compile(r"\bappl(?:y|ies|ied|ication|icable|icability)\b", re.IGNORECASE)
ESTABLISHMENT = re.compile(r"\b(?:establishments?|factor(?:y|ies)|mines?|shops?|plantations?)\b", re.IGNORECASE)


def coverage_topic(question: str) -> str:
    """'gratuity' for 'Who is covered for gratuity under the SS Code?', else ''."""
    if not COVERAGE_QUESTION.search(question):
        return ""
    return next((name for name, (q, _) in COVERAGE_TOPICS.items() if re.search(q, question, re.IGNORECASE)), "")


def _applicability_passages(services: Services, question: str, scope: Scope, statuses: tuple[str, ...],
                            signals: Signals, trace: Trace) -> list[Passage]:
    """For a coverage question, the passages that say which establishments the topic's chapter applies to."""
    topic = coverage_topic(question)
    if not topic:
        return []
    text_pat = COVERAGE_TOPICS[topic][1]
    flt = scope_filter(scope.states, statuses)
    query = f"establishments to which the {topic} chapter applies; application of the provisions on {topic}"
    try:
        matches = services.store.search(query, services.settings.top_k, flt)  # type: ignore[union-attr]
    except Exception as e:  # noqa: BLE001 — the main search already worked
        log.warning("Applicability lookup failed for %s: %s", topic, e)
        return []
    out = []
    for m in matches:
        p = _passage(m.metadata, m.score, signals)
        where = f"{p.chapter} {p.heading} {p.section} {p.text}"
        if p.status == "commentary":
            continue  # "the Code applies to…" in the Statement of Objects and Reasons is not the law itself
        if re.search(text_pat, where, re.IGNORECASE) and APPLIES.search(where) and ESTABLISHMENT.search(p.text):
            trace.add(m.id, m.score, m.metadata)
            p.pinned = True
            out.append(p)
    return out[:2]


def _definition_passages(services: Services, question: str, scope: Scope, statuses: tuple[str, ...],
                         signals: Signals, trace: Trace) -> list[Passage]:
    """Chunks whose `term` is the question's subject ("What is a fixed term employee?"), found by an exact
    metadata filter. They are always sent to the model, because a definition can score low on similarity."""
    variants = term_variants(signals.term) if signals.term else []
    if not variants:
        return []
    flt = {**scope_filter(scope.states, statuses), "term": {"$in": variants}}
    trace.filters.append(flt)
    try:
        matches = services.store.search(question, 3, flt)  # type: ignore[union-attr]
    except Exception as e:  # noqa: BLE001 — the main search already worked; carry on without the pin
        log.warning("Definition lookup failed for %r: %s", signals.term, e)
        return []
    out = []
    for m in matches:
        trace.add(m.id, m.score, m.metadata)
        p = _passage(m.metadata, m.score, signals)
        if same_term(p.term, signals.term):
            p.pinned = True
            out.append(p)
    return out[:2]


# ----------------------------------------------------------------- generation
SYSTEM_PROMPT = """You are Compliance Copilot, an assistant for Indian payroll and labour law.
Answer ONLY from the numbered passages. Rules:
1. Start with a direct answer (Yes / No / the figure exactly as written in a passage), then explain briefly in 2-4 sentences.
2. Cite every claim with the passage number in square brackets, like [1] or [2]. Only use numbers of passages you were given.
3. Never state a rate, ceiling, slab or amount that is not written in the passages. Do not do arithmetic.
4. If passages conflict, prefer the one with the latest effective date and say that you did.
5. If you rely on a passage marked REPEALED, say clearly that the Act has been repealed. A passage marked
COMMENTARY explains why the law was made; answer from the operative sections when they cover the question.
6. If the passages do not answer the question, reply with exactly: NOT_FOUND
You may use **bold**, short bullet lists and markdown tables. No headings."""

COVERAGE_RULE = """The question asks who is covered. If the passages contain them, state both:
(a) which employees are eligible, and (b) which establishments the provisions apply to (for example a minimum
number of employees), each with its citation. If one of the two is not in the passages, say so."""

COMPARE_RULE = """The question compares states. Answer with a markdown table that has one row per state
({states}) and cite passages in the cells. If a state's passages don't cover it, write "not in documents" in its row.
Then add one summary sentence."""


def status_mark(status: str) -> str:
    if status in OLD:
        return "REPEALED"
    return "COMMENTARY (not operative law)" if status == "commentary" else ""


def build_messages(question: str, passages: list[Passage], scope: Scope) -> list[dict[str, str]]:
    """System prompt + numbered passages. Passage order = citation numbers."""
    blocks = []
    for i, p in enumerate(passages, 1):
        head = " · ".join(x for x in (p.doc, p.section, f"p. {p.page}" if p.page else "",
                                      f"effective {p.effective_date}" if p.effective_date else "",
                                      "Central" if p.scope == "central" else p.scope.title(),
                                      status_mark(p.status)) if x)
        blocks.append(f"[{i}] {head}\n{p.context or p.text}")
    system = SYSTEM_PROMPT + f"\nJurisdiction searched: {scope.label()}."
    if coverage_topic(question):
        system += "\n" + COVERAGE_RULE
    if len(scope.states) > 1:
        system += "\n" + COMPARE_RULE.format(states=", ".join(scope.label().split(" + ")[1:]))
    user = "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def clean_citations(answer: str, n_passages: int) -> str:
    """Drop citation numbers that don't match a passage: '[2, 9]' with 4 passages becomes '[2]'."""

    def fix(m: re.Match[str]) -> str:
        nums = [x.strip() for x in m.group(1).split(",") if 1 <= int(x) <= n_passages]
        return "[" + ", ".join(nums) + "]" if nums else ""

    text = re.sub(r"(?<=\S)[ \t]{2,}", " ", CITE.sub(fix, answer))  # spaces left by a removed marker
    return re.sub(r"[ \t]+([.,;:])", r"\1", text).strip()


def cited_numbers(answer: str) -> set[int]:
    return {int(x) for m in CITE.finditer(answer) for x in m.group(1).split(",")}


def is_not_found(text: str) -> bool:
    return text.strip().strip(".*`\"'").upper().startswith(NOT_FOUND)


def _scope_out(scope: Scope) -> ScopeOut:
    return ScopeOut(state=scope.primary, how=scope.how, states=scope.states if len(scope.states) > 1 else None)


def not_found(scope: Scope, best: float | None = None) -> ChatResponse:
    note = f"Best match scored {best:.2f}, below the threshold." if best is not None else None
    return ChatResponse(
        type="not_found",
        answer=f"I couldn't find this in my documents for {scope.label()}, so I won't guess.\n\n"
        "Try rephrasing, pick a different state, or add the official document from the sidebar.",
        scope=_scope_out(scope), note=note,
    )


def repealed_note(answer: str, passages: list[Passage]) -> str | None:
    used = cited_numbers(answer)
    old = sorted({f"{p.doc} ({p.status})" for i, p in enumerate(passages, 1) if i in used and p.status in OLD})
    if not old:
        return None
    return f"This answer cites {', '.join(old)}. The Labour Codes replaced it; check the current Code before relying on it."


def renumber(answer: str, n_passages: int) -> tuple[str, list[int]]:
    """Number cited passages 1, 2, 3… in the order the answer first cites them.

    Returns the rewritten answer and the old passage numbers in their new order:
    'A [3]. B [1, 3].' → ('A [1]. B [1, 2].', [3, 1]). With no markers, nothing changes.
    """
    order: list[int] = []
    for m in CITE.finditer(answer):
        for x in m.group(1).split(","):
            n = int(x)
            if 1 <= n <= n_passages and n not in order:
                order.append(n)
    if not order:
        return answer, list(range(1, n_passages + 1))
    new = {old: i for i, old in enumerate(order, 1)}

    def fix(m: re.Match[str]) -> str:
        nums = sorted({new[int(x)] for x in m.group(1).split(",") if int(x) in new})
        return "[" + ", ".join(map(str, nums)) + "]"

    return CITE.sub(fix, answer), order


def finish(raw: str, passages: list[Passage], scope: Scope) -> ChatResponse:
    """Turn the model's raw text into the API response.

    Only the passages the answer cites are listed as citations, renumbered in the order they are
    first cited; the rest go to `other_citations` (shown collapsed as "Other passages searched").
    """
    if is_not_found(raw):
        return not_found(scope)
    text = clean_citations(raw, len(passages))
    note = repealed_note(text, passages)
    text, order = renumber(text, len(passages))
    cited = [passages[i - 1].citation() for i in order]
    others = [p.citation() for i, p in enumerate(passages, 1) if i not in order]
    return ChatResponse(type="rag", answer=text, scope=_scope_out(scope), citations=cited,
                        other_citations=others, note=note)


def prepare(services: Services, question: str, scope: Scope,
            trace: Trace | None = None) -> tuple[list[Passage], ChatResponse | None]:
    """Retrieve and apply the threshold. Returns (passages, early_response_if_not_found). May raise SearchError."""
    passages = retrieve(services, question, scope, trace)
    best = max((p.score for p in passages), default=None)
    pinned = any(p.pinned for p in passages)  # a definition of the asked term always goes to the model
    if not pinned and (best is None or best < services.settings.score_threshold):
        return passages, not_found(scope, best)
    expand_neighbours(services, question, passages)
    return passages, None


def answer(services: Services, question: str, scope: Scope,
           trace: Trace | None = None) -> tuple[ChatResponse, list[Passage]]:
    """Non-streaming RAG answer."""
    passages, early = prepare(services, question, scope, trace)
    if early:
        return early, passages
    raw = services.llm.chat(build_messages(question, passages, scope))
    return finish(raw, passages, scope), passages


def stream(services: Services, question: str, scope: Scope, trace: Trace | None = None) -> Iterator[tuple[str, object]]:
    """Yield ("token", str) while generating, then ("done", ChatResponse).

    The first few characters are held back until we know the model didn't
    answer NOT_FOUND, so the user never sees that marker flash on screen.
    """
    passages, early = prepare(services, question, scope, trace)
    if early:
        yield "done", early
        return
    raw, released = "", False
    for token in services.llm.stream(build_messages(question, passages, scope)):
        raw += token
        if not released:
            if len(raw.strip()) < len(NOT_FOUND) + 1:
                continue
            if is_not_found(raw):
                break
            released = True
            yield "token", raw
            continue
        yield "token", token
    if not released and raw and not is_not_found(raw):
        yield "token", raw
    yield "done", finish(raw, passages, scope)
