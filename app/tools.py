"""Calculator tools (step 5)."""

from __future__ import annotations

from app.jurisdiction import Scope
from app.schemas import ChatResponse


def run(question: str, scope: Scope) -> ChatResponse:
    return ChatResponse(type="tool", answer="Calculators arrive in step 5.")
