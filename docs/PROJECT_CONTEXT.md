# Compliance Copilot India: full context for a new Claude session

Paste this whole file as the first message to any new Claude (Claude Code on your PC, claude.ai, or a cloud session), or say "read docs/PROJECT_CONTEXT.md first". It is written to be the only thing a new session needs. Last updated: 15 Oct 2026.

---

## 1. Who I am and how to work with me

- I'm **Rushi** (GitHub: `rushimakwana112-cpu`). This is my portfolio project and my Learnbay assignment, under my brand **Cyber Panther**.
- I'm learning AI while building it. Keep explanations simple and add what I should learn to `docs/build-log.md`.
- I work on **Windows PowerShell** with a `.venv`. Give exact PowerShell commands, one per line, and say what output I should expect.
- Diagnose the cause of an error or eval failure before changing code.
- After each step, run the tests, commit, and push.
- Ask me before heavy installs and before any destructive Pinecone action. Never touch existing Pinecone indexes.
- Never commit secrets. Never put API keys in the frontend.
- Use synthetic data only.

## 2. What the project is

**Compliance Copilot India** is a RAG chatbot for Indian payroll and labour law.

- **Live demo:** https://compliance-copilot-pykf.onrender.com (Render free tier, sleeps when idle, the first load takes about a minute)
- **Repo:** https://github.com/rushimakwana112-cpu/compliance-copilot-india

**Product rule (most important):** the chatbot has **no access to any employee data**: no payslips, salary history, attendance, bank details or personal records. It answers only from:
- official laws and notifications,
- `data/rates.yaml`,
- calculations on numbers the user types in the question.

For "my PF balance", "my payslip" and similar questions, it politely says it can't see records. It offers to calculate instead and points to HR/payroll or EPFO (passbook, UMANG).

## 3. Legal context (as of Oct 2026)

- The **four Labour Codes** are in force from **21 Nov 2025**: Wages, Industrial Relations, Social Security, and OSH. The old Acts are repealed.
- **EPF / EPS / EDLI Schemes 2026** were notified on 1 July 2026.
- A Gazette notification of **17 Sept 2026** (S.O. 5109(E)) raised the EPFO wage ceiling to **Rs 25,000**.
- Income tax uses the **Income-tax Act, 2025** (tax year 2026-27).
- **Rule:** legal figures come only from cited official documents and `data/rates.yaml`. They are never hard-coded. A test enforces this (section 7).

## 4. Tech stack

| Part | What we use |
|---|---|
| Backend | FastAPI (`app/main.py`), Python 3.12 |
| Frontend | One file: `frontend/index.html` (light "Heroic" theme) |
| LLM | Groq API, model `qwen/qwen3.8-27b` (`LLM_PROVIDER=groq`). Ollama `qwen2.5:7b` is the local fallback (`LLM_PROVIDER=ollama`) |
| Vector DB | Pinecone, index `compliance-copilot-cloud`, namespace `compliance`, with integrated `llama-text-embed-v2` embeddings (`EMBED_PROVIDER=pinecone`). Ollama `bge-m3` is the local alternative |
| Payroll data | SQLite `data/sample_payroll.db`, synthetic. The SQL route is off unless `SAMPLE_DATA=true` |
| Hosting | Render free tier, Blueprint in `render.yaml`, region singapore, `PUBLIC_MODE=true` |
| CI | GitHub Actions runs pytest on every push and PR to `main` |

Secrets live in `.env` (copied from `.env.example`, git-ignored) and in the Render dashboard. Only `GROQ_API_KEY` and `PINECONE_API_KEY` are secrets.

## 5. Git and deploy workflow

- `main` is the deployed branch. Render deploys from `main`.
- All work goes on **`feature/groq-cloud`**. Open a PR into `main`. Once the tests pass, merge. Rushi has given standing permission to merge.
- After each merge, reset `feature/groq-cloud` to `origin/main` and push it.
- Rushi sometimes pushes to `feature/groq-cloud` himself, so fetch and merge before pushing.
- Merge lesson: "keep my version" applies to conflicts only. A file changed on one side only (for example `eval/report.md` with a real run) keeps that side.
- **Render auto-deploy was off until 15 Oct 2026.** The live site stayed on PR #4 until a manual deploy. After every merge, check Render → Events to confirm the new commit is **Live**. Use Manual Deploy → "Deploy latest commit" if it isn't.
- Commit messages are plain English and describe the change. Do not put AI model names in commits or PRs.
- Cloud Claude sessions **cannot reach** Groq, Pinecone, .gov.in sites or onrender.com. Real runs (ingest, eval, live checks) happen on Rushi's PC.

Useful PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
git checkout feature/groq-cloud
git pull origin feature/groq-cloud
python -m pytest -q
uvicorn app.main:app --reload
.\scripts\run_all.ps1
```
- `python -m pytest -q` should end with `368 passed` (the count grows as tests are added).
- After `uvicorn ...`, open http://localhost:8000.
- `.\scripts\run_all.ps1` runs the whole pipeline: pull, install, test, re-index, eval, README numbers, commit, push. It prints an `OK:` line per step and `All done.` at the end.

## 6. How a question is answered (`app/pipeline.py`)

1. **understand():** detect the language (English, Hindi, Gujarati or Hinglish, `app/lang.py`) and translate to English. Expand short forms (sal, pf, ctc...). Split "is this calculation correct?" claims (`app/verify.py`).
2. **out_of_scope():** refuse foreign places and other jurisdictions (`app/jurisdiction.py`).
3. **early():** answers that need no search, in this order:
   - guards (`app/guards.py`): injection, other employees' data, personal records, off-topic, tax planning, invalid figures (₹0, negative amounts, 40 working days), vague one-word questions
   - CTC → in-hand conversation (`app/salary.py`): asks for all missing details in **one** message (monthly or yearly, basic, PF basis, ESIC, state with Gujarat as the default, tax regime), shows the steps, and gives two labelled results when PF or ESIC is unknown
   - income tax / TDS (`app/tax.py`)
   - bonus, overtime, leave encashment (`app/labour.py`)
   - salary helpers: per-day, loss of pay, joining/exit month, increments and arrears
   - fixed explanations (`app/faq.py` + `data/faq.yaml`), with figures filled from rates.yaml
4. **Router** (`app/router.py`) picks one route: `rag` (search the law), `tool` (PF/ESIC/PT/gratuity calculators and due dates, `app/tools.py`, `app/deadlines.py`), `sql` (only when SAMPLE_DATA is on), `guide`, or `not_found`.
5. **RAG** (`app/rag.py`, `app/vectorstore.py`): Pinecone search. Below `SCORE_THRESHOLD` (0.38 for Pinecone) it answers "Not in my documents" without calling the LLM. Only the passages actually cited are shown. The context is capped at `CONTEXT_TOKEN_BUDGET=3000` because the Groq free tier allows about 8k tokens per minute.
6. **finish():** add the verify line and the advice note, then translate back to the user's language.

Every answer:
- mentions the FY and that rates can change,
- uses ₹ with Indian number formatting,
- shows step-by-step workings,
- ends with "check with HR/payroll".

**Ingestion** (`ingest/`): `download → parse → chunk → index`, using official sources listed in `data/sources.csv` (9 sources, verified by Rushi). Changing chunking needs `python -m ingest.run --reindex` on Rushi's PC.

## 7. Rules enforced by tests (`tests/`, 368 tests, all fakes, no network)

- `test_no_hard_coded_legal_numbers` scans app/, ingest/, scripts/ and frontend/ for numbers like 15000, 21000, 25000, 0.12, 8.33 and similar. Even docstring examples must use other numbers.
- `test_no_secret_in_git_history` checks that no API key was ever committed. It reads git output as UTF-8 so it works on Windows.
- `eval/questions.csv` has 43 questions: rag 18, tool 12, guide 8, not_found 5. Run it with `python -m eval.run --delay 8`; it writes `eval/report.md`.

## 8. What's done (PRs #1–#8, all merged to main)

- **#1–#3:** Groq cloud mode, Pinecone integrated index, `run_all.ps1`.
- **#4:** publish readiness (`render.yaml`, `DEPLOY.md`, README).
- **#5:** live URL in the README.
- **#6:** PF, ESIC, PT and gratuity calculators; PF, TDS and PT due dates; rates.yaml filled from official documents.
- **#7:** the 11-category employee Q&A: salary/CTC, PF, ESI, PT, income tax/TDS, bonus, gratuity, minimum wages/overtime/leave, Labour Codes and due dates, personal-record questions, edge cases. Also: Hindi, Gujarati and Hinglish; the `guide` answer type; SQL off by default.
- **#8:** the Windows UTF-8 fix in the deploy tests.

Not merged yet, on `feature/groq-cloud`: guard answers also show the jurisdiction (commit d566ece). It merges once the live site is confirmed on the latest deploy.

## 9. Open items: what Rushi must do

These are also listed in `NEXT_STEPS_FOR_RUSHI.md`.

1. **Deploy the latest commit on Render.** Run Manual Deploy, then turn Auto-Deploy "On Commit". Then check that "My CTC is ₹6 lakh a year, what is my in-hand?" asks for the missing details.
2. **Confirm the rates marked `- check` in `data/rates.yaml`** against the official PDFs: EPF 12%, EPS 8.33%, EDLI 0.5%, PF due date, gratuity divisor 26 and its rounding. Then remove `- check` and add the url.
3. **Fill the empty rates when the notifications are found:**
   - `bonus.eligibility_wage_ceiling`, `bonus.calculation_ceiling`
   - `income_tax.gratuity_exemption_limit`, `income_tax.leave_encashment_exemption_limit`
   - Gujarat minimum wages
   - `gratuity.max_amount` (not notified under the Code yet)
   - Maharashtra PT slabs
4. Run `.\scripts\run_all.ps1` after changing rates or chunking, so the README eval numbers stay real.

## 10. Plan and ideas to improve next

Pick one per session. Each one needs tests, a build-log note and a PR.

- **Accuracy:** raise the eval scores. Look at the failing rows in `eval/report.md`, diagnose them, and fix retrieval or routing. Add eval questions for the new categories.
- **More states:** Maharashtra and Karnataka PT, and state minimum wages from official state sites.
- **Calculators:** TDS on bonus or arrears, full-and-final settlement with numbers, a full-year new vs old regime comparison table.
- **Retrieval:** a hybrid keyword + vector search, or a re-ranker, because plain similarity can't separate relevant from off-topic questions well.
- **Freshness:** a script that checks the official sources for new notifications and flags rates that changed.
- **Product:** streaming answers everywhere, a "copy calculation" button, a feedback 👍/👎 button logged to `logs/`, and a short demo video for the portfolio (plan in `docs/manual-steps.md`).
- **Ops:** keep Auto-Deploy on. Add an uptime ping so the free instance wakes faster for demos. Keep the README eval table updated.

## 11. Key files

| File | What it holds |
|---|---|
| `CLAUDE.md` | short rules for Claude (auto-loaded by Claude Code) |
| `docs/PROJECT_CONTEXT.md` | this file |
| `docs/build-log.md` | what was built and what to learn, step by step |
| `NEXT_STEPS_FOR_RUSHI.md` | manual steps and rates to confirm |
| `DEPLOY.md` | Render deploy steps |
| `data/rates.yaml` | every legal figure, dated, with its source |
| `data/faq.yaml` | fixed explanations with rate placeholders |
| `data/sources.csv` | the official documents that get indexed |
| `render.yaml` | Render Blueprint |
| `.env.example` | every setting, explained |
