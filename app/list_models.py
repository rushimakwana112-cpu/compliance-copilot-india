"""Print the chat models your configured provider can use.

    python -m app.list_models

For Groq, copy one id into GROQ_MODEL in .env. A Qwen or Llama instruct model
works best with these prompts; prefer one that supports JSON mode (most do).
"""

from __future__ import annotations

import re

from app.config import get_settings
from app.llm import LLMError, make_llm

RECOMMENDED = re.compile(r"qwen|llama", re.IGNORECASE)
NOT_CHAT = re.compile(r"whisper|tts|guard|embed|prompt-guard|playai|orpheus|distil", re.IGNORECASE)


def main() -> None:
    s = get_settings()
    llm = make_llm(s)
    try:
        models = llm.list_models()  # type: ignore[attr-defined]
    except LLMError as e:
        raise SystemExit(f"Could not list models: {e}")
    print(f"Provider: {s.llm_provider}   current model: {s.active_model or '(not set)'}\n")
    chat = [m for m in models if not NOT_CHAT.search(m)]
    for m in chat:
        mark = "  <- Qwen/Llama, good fit" if RECOMMENDED.search(m) else ""
        print(f"  {m}{mark}")
    skipped = len(models) - len(chat)
    if skipped:
        print(f"\n({skipped} speech, guard or embedding models hidden)")
    if s.llm_provider == "groq":
        print("\nSet one in .env, e.g.  GROQ_MODEL=<id from the list above>")


if __name__ == "__main__":
    main()
