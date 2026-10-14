"""Retrieval-augmented answers with citations.

    follow-up rewrite → embed → Pinecone (jurisdiction filter) → re-rank → threshold
    → prompt with numbered passages → Qwen → strip invalid [n] markers

Below the score threshold we never call the LLM: a weak match is the moment a
model is most likely to make something up, so we refuse instead.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from app.jurisdiction import Scope
from app.llm import LLM, LLMError
from app.schemas import Citation, ChatResponse, ScopeOut, Turn
from app.services import Services
from app.vectorstore import scope_filter

DOC_TYPE_BOOST = 0.03
SNIPPET_CHARS = 320
NOT_FOUND = "NOT_FOUND"

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
FOLLOW_UP = re.compile(r"^\s*(and|what about|how about|also|same|for|in|then|what if|ok|but)\b|\b(it|that|this|they|same)\b", re.I)


@dataclass
class Passage:
    doc: str
    section: str
    page: str
    score: float  # raw similarity from Pinecone
    rank_score: float  # score + boosts, used only for ordering
    text: str
    url: str
    status: str
    effective_date: str
    doc_type: str
    scope: str

    def citation(self) -> Citation:
        snippet = self.text if len(self.text) <= SNIPPET_CHARS else self.text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
        return Citation(doc=self.doc, section=self.section, page=str(self.page), score=round(self.score, 3),
                        snippet=snippet, url=self.url, status=self.status)


# ----------------------------------------------------------------- follow-ups
REWRITE_PROMPT = """Rewrite the user's LAST message as one standalone question about Indian payroll or labour law.
Keep the topic, state and amounts from earlier turns unless the last message changes them.
Return only the rewritten question, nothing else."""


def needs_rewrite(message: str, history: list[Turn]) -> bool:
    """Short or pronoun-heavy messages after earlier turns are probably follow-ups."""
    return bool(history) and (len(message.split()) <= 6 or bool(FOLLOW_UP.search(message)))


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


def _passage(meta: dict, score: float, boost_types: set[str]) -> Passage:
    doc_type = str(meta.get("doc_type", ""))
    return Passage(
        doc=str(meta.get("doc", "")), section=str(meta.get("section", "")), page=str(meta.get("page", "")),
        score=score, rank_score=score + (DOC_TYPE_BOOST if doc_type in boost_types else 0.0),
        text=str(meta.get("text", "")), url=str(meta.get("source_url", "")), status=str(meta.get("status", "current")),
        effective_date=str(meta.get("effective_date", "")), doc_type=doc_type, scope=str(meta.get("scope", "")),
    )


def retrieve(services: Services, question: str, scope: Scope) -> list[Passage]:
    """Top passages for `question`. For a comparison, each state gets its own retrieval so neither is crowded out."""
    s = services.settings
    assert services.store is not None
    statuses = ("current", "repealed", "historical") if REPEALED_HINT.search(question) else ("current",)
    boost = wanted_doc_types(question)
    per_state_keep = s.keep_k if len(scope.states) == 1 else max(2, s.keep_k - 1)
    kept: dict[tuple, Passage] = {}
    for state in scope.states:
        matches = services.store.search(question, s.top_k, scope_filter([state], statuses))
        ranked = sorted((_passage(m.metadata, m.score, boost) for m in matches), key=lambda p: p.rank_score, reverse=True)
        for p in ranked[:per_state_keep]:
            kept.setdefault((p.doc, p.page, p.text[:80]), p)
    return sorted(kept.values(), key=lambda p: p.rank_score, reverse=True)


# ----------------------------------------------------------------- generation
SYSTEM_PROMPT = """You are Compliance Copilot, an assistant for Indian payroll and labour law.
Answer ONLY from the numbered passages. Rules:
1. Start with a direct answer (Yes / No / the figure exactly as written in a passage), then explain briefly in 2-4 sentences.
2. Cite every claim with the passage number in square brackets, like [1] or [2]. Only use numbers of passages you were given.
3. Never state a rate, ceiling, slab or amount that is not written in the passages. Do not do arithmetic.
4. If passages conflict, prefer the one with the latest effective date and say that you did.
5. If you rely on a passage marked REPEALED, say clearly that the Act has been repealed.
6. If the passages do not answer the question, reply with exactly: NOT_FOUND
You may use **bold**, short bullet lists and markdown tables. No headings."""

COMPARE_RULE = """The question compares states. Answer with a markdown table that has one row per state
({states}) and cite passages in the cells. If a state's passages don't cover it, write "not in documents" in its row.
Then add one summary sentence."""


def build_messages(question: str, passages: list[Passage], scope: Scope) -> list[dict[str, str]]:
    """System prompt + numbered passages. Passage order = citation numbers."""
    blocks = []
    for i, p in enumerate(passages, 1):
        head = " · ".join(x for x in (p.doc, p.section, f"p. {p.page}" if p.page else "",
                                      f"effective {p.effective_date}" if p.effective_date else "",
                                      "Central" if p.scope == "central" else p.scope.title(),
                                      "REPEALED" if p.status != "current" else "") if x)
        blocks.append(f"[{i}] {head}\n{p.text}")
    system = SYSTEM_PROMPT + f"\nJurisdiction searched: {scope.label()}."
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
    old = sorted({f"{p.doc} ({p.status})" for i, p in enumerate(passages, 1) if i in used and p.status != "current"})
    if not old:
        return None
    return f"This answer cites {', '.join(old)}. The Labour Codes replaced it; check the current Code before relying on it."


def finish(raw: str, passages: list[Passage], scope: Scope) -> ChatResponse:
    """Turn the model's raw text into the API response."""
    if is_not_found(raw):
        return not_found(scope)
    text = clean_citations(raw, len(passages))
    return ChatResponse(type="rag", answer=text, scope=_scope_out(scope),
                        citations=[p.citation() for p in passages], note=repealed_note(text, passages))


def prepare(services: Services, question: str, scope: Scope) -> tuple[list[Passage], ChatResponse | None]:
    """Retrieve and apply the threshold. Returns (passages, early_response_if_not_found)."""
    if services.store is None:
        r = not_found(scope)
        r.note = services.store_error or "Pinecone is not configured."
        return [], r
    passages = retrieve(services, question, scope)
    best = max((p.score for p in passages), default=None)
    if best is None or best < services.settings.score_threshold:
        return passages, not_found(scope, best)
    return passages, None


def answer(services: Services, question: str, scope: Scope) -> tuple[ChatResponse, list[Passage]]:
    """Non-streaming RAG answer."""
    passages, early = prepare(services, question, scope)
    if early:
        return early, passages
    raw = services.llm.chat(build_messages(question, passages, scope))
    return finish(raw, passages, scope), passages


def stream(services: Services, question: str, scope: Scope) -> Iterator[tuple[str, object]]:
    """Yield ("token", str) while generating, then ("done", ChatResponse).

    The first few characters are held back until we know the model didn't
    answer NOT_FOUND, so the user never sees that marker flash on screen.
    """
    passages, early = prepare(services, question, scope)
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
