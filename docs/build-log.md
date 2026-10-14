# Build log: what each step taught

Each step of the build, the command that proves it works, and what to learn from it.

## Step 1: synthetic sample database

`python scripts/make_sample_db.py` writes 25 made-up employees (5 states, three employment types, a third of them on low salaries so ESIC and PF thresholds have rows on both sides).

**What to learn:** a fixed random seed (`random.Random(42)`) makes "random" data identical on every run, so tests and eval results are repeatable and the DB file can be committed. Synthetic data also means you can publish the repo without leaking anyone's salary.

## Step 2: FastAPI skeleton and /api/health

`uvicorn app.main:app --reload` then open http://localhost:8000. With nothing running, health reports exactly what's missing:

```json
{"status":"degraded","model":"qwen2.5:7b","pinecone":false,"docs":0,"ollama":false,
 "issues":["Ollama is not running at http://localhost:11434. Start it with: ollama serve",
           "PINECONE_API_KEY is not set in .env"]}
```

**What to learn:** `create_app(services)` takes its Ollama and Pinecone clients as an argument (dependency injection). Production passes real clients, tests pass fakes from `tests/conftest.py`, so 17 tests run in under a second with no network. Keeping every external service behind a small interface (`LLM`, `VectorStore`) is what makes a RAG app testable.

## Step 3: ingestion

`python -m ingest.run --limit 2` downloads, parses, chunks and indexes the first two rows of `data/sources.csv`. On a small Act-like test PDF the chunker produces:

```
p.1 #0 [Section 53 · Payment of gratuity] 'CHAPTER V\nGRATUITY\n53. Payment of gratuity.—(1) Gratuity sha'
p.2 #0 [Section 54 · Determination of gratuity] '54. Determination of gratuity.—(1) For every completed year '
p.2 #1 [Section 55 · Nomination] '55. Nomination.—Each employee shall make nomination for the '
```

**What to learn:** chunking decides what the retriever can find. Cutting at every 1,000 characters would split "Section 54" in half and glue it to "Section 55", so the citation would be wrong and the passage would carry two topics. Splitting on legal headings first keeps one idea per chunk and gives every chunk a real citation. Deterministic vector ids (`doc-slug#sha1(doc|page|chunk)`) make re-runs overwrite rather than duplicate, and `--dry-run` lets you see a delete before it happens.

## Step 4: RAG answers with citations (plus streaming)

`POST /api/chat` with `"Gratuity rule for fixed term employee in Surat?"` and default state Maharashtra returns `type: "rag"`, `scope: {"state": "gujarat", "how": "detected"}` (the backend re-detected Surat), numbered citations, and `latency_ms`. `POST /api/chat/stream` sends `meta`, then `token` events, then `done` with the same JSON.

**What to learn:** the threshold check happens *before* the LLM. When the best passage is weak, a model asked to answer anyway invents something plausible, so refusing early is the cheapest hallucination guard you have. Numbering passages and cleaning `[n]` markers that don't match a passage turns "the model says it has a source" into "the source exists". Follow-ups like "and for Gujarat?" retrieve nothing on their own, so they're rewritten into a full question first.

## Step 5: calculator tools

`"Calculate PF for basic ₹18,000"` is routed to `pf_calculator(basic=18000)`. With the shipped `data/rates.yaml` (all values `null`) it answers *"Rate not configured: `pf.employee_rate` is empty in data/rates.yaml, so I won't calculate."* Once you fill the values from the EPFO circular, it returns a 4-row table (employee PF, EPS, EPF, total) citing that circular.

**What to learn:** this is "tool use" in its simplest form: the router spots a calculation and hands it to ordinary code. LLMs predict likely-looking digits rather than computing them, so payroll maths must never come from the model. Keeping every figure in one YAML file with its `source_doc` and `effective_date` means that when a rate changes you edit one line, and every answer still shows where the number came from.

## Step 6: text-to-SQL with a guard

`"Which sample employees are above the ESIC limit?"` → Qwen writes `SELECT emp_id, name, gross FROM employees WHERE gross > :esic_ceiling`, the guard adds `LIMIT 200`, `:esic_ceiling` is filled from `rates.yaml`, and the query runs on a read-only connection. The tests feed the guard `DROP`, `UPDATE`, `DELETE`, `INSERT`, `SELECT 1; SELECT 2`, unknown tables, unknown columns, `PRAGMA`, `ATTACH` and `load_extension()`, and all of them are rejected.

**What to learn:** never run model-written SQL as-is. Parsing it with a real parser (sqlglot) lets you check *what* it does, not just what it looks like, and opening SQLite with `mode=ro` is a second, independent lock. That pattern is called defence in depth. Passing legal thresholds as `:parameters` means the model never has to know (or invent) the ESIC ceiling. When the guard rejects a query, the model gets one retry with the error, which fixes most honest mistakes.

## Step 7: frontend (minimal, heroic, made for "Compliance Copilot")

The empty screen is now a hero: "Indian payroll law, answered with proof." with the question box directly under it, four starter questions, and three trust points (cites the document and page; calculations run in code; says "not found" instead of guessing). The light "Heroic" theme opens by default; dark "Cyber Panther" is one click away. The grid glow, neon shadows, frosted panels and card-in-card boxes are gone; answers sit on the page with a thin accent rule. New: live states and documents, safe markdown, clickable `[1]` markers, copy answer, export chat (.md), a "RAG · 1.8 s" footer, streaming with a Stop button, and a notice when /api/health reports a problem. Screenshots are in `docs/screenshots/`; checked at 1440 px and 390 px in both themes with no horizontal scroll.

One colour was added (none changed): `--accent-2-text` (#865C10) in the light theme. The gold accent is 3.8:1 on cream, below the 4.5:1 that small text needs, so gold *text* uses the darker shade and gold fills keep the original.

**What to learn:** never put model output into `innerHTML` directly. The markdown renderer escapes everything first and only then adds the few tags it creates itself, so a model (or a poisoned document) that writes `<script>` shows up as text. Streaming is mostly a user-experience feature: the answer takes the same time, but seeing it arrive makes the wait feel shorter, and the Stop button gives control back to the user.
