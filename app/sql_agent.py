"""Text-to-SQL over the sample database (step 6)."""

from __future__ import annotations

from app.schemas import ChatResponse
from app.services import Services


def run(services: Services, question: str) -> ChatResponse:
    return ChatResponse(type="sql", answer="SQL arrives in step 6.")
