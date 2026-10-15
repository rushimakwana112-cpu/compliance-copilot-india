# Compliance Copilot India: context for Claude

I'm Rushi, building a RAG chatbot for Indian payroll and labour law (portfolio project + Learnbay assignment, brand: Cyber Panther).

Stack: FastAPI backend + single-file HTML frontend; Groq API (qwen/qwen3.8-27b) for the LLM, Ollama as local fallback; Pinecone with integrated llama-text-embed-v2 embeddings (index compliance-copilot-cloud, namespace compliance); SQLite synthetic payroll data; deploy target Render free tier (PUBLIC_MODE=true there).
I run everything on Windows PowerShell with a .venv and test on my PC; the build environment can't reach Groq, Pinecone or .gov.in.

Legal context: the four Labour Codes are in force from 21 Nov 2025 (old Acts repealed). EPF/EPS/EDLI Schemes 2026 notified 1 July 2026. A Gazette notification of 17 Sept 2026 raised the EPFO wage ceiling to Rs 25,000. Legal figures must come only from cited official documents and data/rates.yaml, never be hard-coded.

How to work with me:
- Give exact PowerShell commands, one per line, and say what output to expect.
- Diagnose the cause of errors and eval failures before changing code.
- Run the tests after each step, commit per step, and push to feature/groq-cloud.
- Keep explanations simple; I'm learning. Add what I should learn to docs/build-log.md.

Full project context (stack, Git workflow, what is done, open items, plan): read docs/PROJECT_CONTEXT.md at the start of every session.
