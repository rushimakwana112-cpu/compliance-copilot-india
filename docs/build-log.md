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

## Step 8: evaluation

`eval/questions.csv` has 30 questions: 16 RAG (5 of them follow-ups), 5 calculator, 4 SQL and 5 trick questions that must be refused (Dubai, California, stock tips, GST, IPL). `python -m eval.run --sweep 0.45,0.5,0.55,0.6,0.65` reports route accuracy, retrieval hit@4, answer-contains, not_found precision and recall, and median latency, and writes `eval/report.md`. The sweep re-scores only retrieval for each threshold, so it takes seconds and needs no LLM.

**What to learn:** without an eval, every change to the threshold, chunk size or prompt is a guess. The threshold is a trade-off you can now see: raise it and the trick questions get refused, but some real questions get refused too (precision vs recall). Change one knob, re-run, and keep the numbers in the README; that table is what makes this a portfolio project rather than a demo.

## Step 9: README

`README.md` covers the problem, a Mermaid architecture diagram, screenshots of both themes, six PowerShell setup commands, how jurisdiction works, how to add a document, the eval table (to fill after your first run), limitations and future work.

**What to learn:** a README is the interface of a portfolio project. A reviewer decides in a minute whether to read your code, and the architecture diagram plus an honest limitations section usually decide it.

# Phase 2: free in the cloud with Groq

Goal: the same copilot, deployed for free, while the local Ollama mode keeps working. Five steps, one commit each, tests after every step (113 pass at the end).

## Step 1: a provider layer for the language model

`LLM_PROVIDER=groq|ollama` picks a client in `app/llm.py`; both get the same prompts, temperature 0.1 and 700 max tokens, so the NOT_FOUND rule and the router's JSON behave the same. `python -m app.list_models` prints what your Groq key can use. A Groq 429 becomes the answer *"Daily demo limit reached. Please try again later."* with `quota_exceeded: true`; a timeout is retried once after a short pause.

**What to learn:** code against an interface, not a vendor. The pipeline only ever calls `chat`, `stream` and `health`, so adding Groq touched one file plus settings. Also notice which errors deserve a retry: a timeout is usually temporary, but a 429 means "you are over quota", and retrying it only burns more quota. Never hard-code a hosted model id; providers retire them, so list them at runtime and let configuration choose.

## Step 2: embeddings without Ollama

`EMBED_PROVIDER=pinecone` creates a new index with Pinecone's integrated embedding (`llama-text-embed-v2`, `field_map: text`). We upsert plain text plus metadata and search by text with the same jurisdiction filter, so ingestion needs only Python and the Pinecone key. `EMBED_PROVIDER=ollama` keeps the bge-m3 path. Neither mode touches an index it didn't create; pointing one mode at the other's index is refused with a clear message.

**What to learn:** an embedding model defines a vector space, and vectors from two models can't be compared, so each model needs its own index. Similarity scores also change scale between models, which is why `SCORE_THRESHOLD` must be re-tuned with the eval sweep after switching. Hosted ("integrated") embedding moves the heavy model off your machine: the trade-off is less control over the model in exchange for a tiny server.

## Step 3: public mode

`PUBLIC_MODE=true` blocks `/api/ingest/*` with a 403 (checked in middleware, before the upload is read) and the page hides the upload box. Each IP gets 10 chats a minute and 60 a day; a 500-entry LRU cache answers repeated questions without calling Groq and is emptied after any ingestion. Messages are capped at 1,000 characters and history at 4 turns.

**What to learn:** a public demo needs guards a local tool doesn't. Rate limits protect your free quota; a cache keyed by the *standalone* question plus the jurisdiction means "What is the PF rate?" and "what is the pf rate" share an answer, but Gujarat and Kerala don't. Only cache successes: a refusal or a quota error might succeed next time. And know the limits of your own guard: an IP read from `X-Forwarded-For` can be faked, so the limiter stops casual overuse, not attackers.

## Step 4: deploy on Render

`render.yaml` is a Render Blueprint: free Python service, `pip install -r requirements.txt`, `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, secrets entered in the dashboard (`sync: false`), and a `/healthz` check that makes no outside calls. Requirements are split: the web app (about 60 MB of RAM at start), ingestion (`pypdf`, `bs4`, `rich`, optional OCR) and tests. The page shows *"Waking up the server…"* when `/api/health` takes over 3 seconds.

**What to learn:** free hosting is a set of constraints you design for. 512 MB rules out torch, so the server must not do embedding or PDF parsing, and that pushes work to build time (ingestion on your PC) and to managed services (Groq, Pinecone). A sleeping free instance is a user-experience problem, solved in the UI. Keep secrets out of git entirely: config files list *which* variables exist, the platform stores the values.

## Step 5: tests and docs

New tests fake the Groq SDK and Pinecone to cover provider switching, the 429 path through both chat endpoints, the timeout retry, integrated-index creation and refusal, public mode blocking ingest, the rate limiter (minute window, daily reset, 429 body), the cache (LRU order, scope in the key, copies, cleared by ingestion) and deployment guards (no heavy packages in `requirements.txt`, no secret values in `render.yaml`). README now has "Two ways to run", "Deploy free on Render" and the environment variable table.

**What to learn:** fakes make cloud code testable offline and for free. A test that asserts `requirements.txt` has no torch, or that `render.yaml` holds no key, turns a deployment rule into something CI checks for you instead of something you have to remember.

## Fixes after review: honest logs and a stricter rewriter

`logs/chats.jsonl` showed `"top_scores": []` whenever the model answered NOT_FOUND after retrieval had passed the threshold, because the scores were taken from the final answer's citations. Retrieval now records a trace (index, filters, every match) that is logged for every route and outcome. A Pinecone failure or 0 matches returns `type: "error"` with *"Search failed: …"* and a WARNING. The follow-up rewriter used to fire on any message of six words or fewer, so *"Gratuity rule in Dubai?"* became a question about the Code on Social Security; it now rewrites only clear follow-ups.

**What to learn:** log what the system *saw*, not what it *said*. A refusal from the model and an empty search look the same in the answer but mean very different things, and only a log of the scores tells them apart. Never turn an infrastructure failure into a polite domain answer: "not in my documents" when the index is empty hides the bug. And keep LLM rewriting on a short leash: a rewrite that adds a topic the user never mentioned is a hallucination at the very first step of the pipeline.

## Retrieval fixes: chunks that match how laws are written

Definitions sections now become one chunk per defined term, so "wages" carries its whole definition (inclusions, exclusions, proviso and Explanation) in one piece, prefixed with "Code on Wages, 2019, Section 2 (Definitions) — wages:". Other sections are packed from whole sub-clauses up to ~1,800 characters with 200 overlap. Each chunk stores its chapter ("Chapter V — Gratuity"), heading, term, a document-wide `chunk_no` and an integer page. Retrieval fetches 12, re-ranks with document-name (+0.08), defined-term (+0.08) and section-title (+0.05) boosts, keeps 5, and adds the previous and next chunk of the same section as context. The threshold defaults to 0.47 for Pinecone embeddings. `python -m ingest.run --reindex` rebuilds the index with the new chunks.

**What to learn:** chunking is where most RAG quality is won or lost. A size-only splitter cut the "wages" definition in half, so the model saw the inclusions without the exclusions and answered confidently and wrongly. Splitting on the document's own structure (definitions, sub-clauses) keeps each idea whole. Re-ranking with cheap signals from the question ("under the Code on Wages", "definition of") fixes what pure vector similarity misses, and neighbour expansion gives the model the sentence before and after without making every chunk bigger. Boosts change only the order: the refusal threshold still uses the raw similarity, so a boost can never turn a weak match into an answer.

## After the first real test: the model is the real guard

Real results on llama-text-embed-v2 showed the similarity threshold can't separate good from bad questions: "What is a fixed term employee?" scored 0.43 and was refused, while "Gratuity rule in Dubai?" passed 0.47 (the model then refused it correctly). So the threshold is now 0.38 and only blocks nonsense, and the real guards moved to where they can be precise:
- **Foreign places** (Dubai, UAE, USA, UK, Singapore…) are refused before retrieval with "I only cover Indian law".
- **Defined terms**: the question's subject is normalised (singular, quotes, hyphens, employee/employment in compounds) and the chunk that defines it is fetched by an exact metadata filter and always sent to the model.
- **Coverage questions** ("who is covered for gratuity") also fetch the applicability passage (often Section 1 or a Schedule) and the prompt asks for both employee eligibility and establishment applicability.
- **Split words** from PDF extraction ("gra tuity", "G R A T U I T Y") are repaired before chunking.
- The eval gained these four questions, and `&` in expected answers checks that an answer is complete.

**What to learn:** a single number (similarity) is a weak judge of relevance; it mixes topic, wording and length. Use it only to block the obvious, and put precise rules where you know something for sure: a place name is out of scope, a definition answers "what is X", an applicability section answers "who is covered". Then let the model's strict NOT_FOUND rule handle the rest, and measure with an eval that checks completeness, not just whether the answer is on topic.

## Citation labels for Schedules and end matter, and cited-only sources

In the Code on Social Security the gratuity list on p. 97 was labelled "Section 7 · The Cine", and the Statement of Objects and Reasons (p. 111–112) became "Section 5" and "Section 8", because any "N. Heading.—" line started a section. Now Schedules ("First Schedule (see section 53)") and end matter ("Statement of Objects and Reasons", "Notes on Clauses", "Financial Memorandum") get their own labels and swallow their numbered lists. In the body, a numbered heading counts only when it follows the previous section number (53 → 53A → 54, with a gap of up to 3). End matter of a current Act is indexed with `status: commentary`, is still searched, ranks 0.10 below the operative text, and is marked "COMMENTARY (not operative law)" for the model. The answer now lists only the passages it cites, renumbered [1], [2]… in order, and the rest sit under a collapsed "Other passages searched".

**What to learn:** the structure of a document has states. The same line ("7. The Cine…") is a section in the body and a list item in a Schedule, so a parser that reads one line at a time without knowing where it is will mislabel things. Track where you are (body → schedule → end matter) and check that section numbers run in order. And show users what the answer actually used: listing every retrieved passage makes it look as if the answer rests on sources it never read.

## Free-tier limits: tell a busy minute from a used-up day

An eval run reported 13 RAG questions as "Daily demo limit reached" that had answered fine minutes earlier. Groq's free tier counts tokens per minute (~8,000 TPM for the model), and a RAG prompt with 5 passages plus neighbours was 5–8k tokens, so two questions in a row tripped the per-minute limit. Every 429 was treated as the daily quota. Now the 429's message and retry-after say which limit it was: a per-minute limit is waited out (up to 20 s) and retried once, then answered "Busy right now"; a daily limit says so at once. The passages are capped at ~3,000 tokens (neighbours trimmed first, then the lowest-ranked passages dropped), and `logs/chats.jsonl` records `prompt_tokens`. The SQL agent stopped using JSON mode, which this model failed with a 400 "Failed to generate JSON", and asks for a ```sql block instead. The eval waits between questions, honours retry-after, and skips (rather than fails) questions that hit a limit or need a document that isn't indexed.

**What to learn:** an error code is not a diagnosis. "429" covered two very different situations, and only the error body said which. Measure what you send (tokens per prompt) next to the limits you're sending into, and budget for them. And keep an eval's metrics honest: a question that couldn't be asked (no document, no quota) is not a wrong answer, so report it separately instead of letting it drag the score down.

## Query expansion: search in the law's words too

Eval q07 ("Is bonus paid to contract workers too?") was refused: the Code says "contract labour" and "contractor", so the passage scored below the threshold. `data/query_expansion.yaml` maps everyday words to the law's words (contract worker → contract labour, PF → provident fund, minimum wage → minimum rate of wages / floor wage …) and topic hints (bonus → the Code on Wages bonus chapter). Retrieval now searches the question plus up to 2 variants, keeps each passage's best score, de-duplicates and re-ranks as before. The variants are logged as `query_variants` in `logs/chats.jsonl`. The number of passages kept is unchanged, so the ~3,000-token budget still holds.

**What to learn:** embeddings match meaning, but not perfectly; a different word for the same thing can still cost enough similarity to fall below a threshold. A small hand-made synonym map is cheap, easy to read and easy to test, and it fixes the vocabulary gap between users and legal text without another model call.

## Answer style: "Yes," only for yes/no questions

q31 ("Definition of wages under the Code on Wages") came back starting with "Yes,", because the prompt said "start with a direct answer (Yes / No / …)". The prompt now says to start with Yes or No only for yes/no questions, and each request says which kind it is (questions starting with is/are/does/can/… are yes/no). As a safety net, a leading "Yes," or "No." before the answer to any other question is removed after generation.

**What to learn:** an instruction like "start with Yes/No" gets applied everywhere, even where it makes no sense. Tell the model what kind of question it has, and back up a rule that matters with a cheap check in code.

## Hard-coded legal numbers: audit before removing

Search (tracked files): `git grep -nE "15000|15,000|21000|21,000|8\.33|0\.0833|0\.75|3\.25"`. No hits for 0.75 or 3.25. Every hit, before any change:

| File:line | Text | What it is | Action |
|---|---|---|---|
| frontend/index.html:1053 | `amount * 0.12`, `Math.min(amount, 15000)`, `epsBase * 0.0833` | **Legal figures** in the offline PF demo (PF rate, EPS wage cap, EPS rate) | Removed: rates come from `/api/rates` (rates.yaml); offline the demo shows `[rate]` placeholders |
| frontend/index.html:1059 | `8.33% of …` | **Legal figure** (EPS rate) in the same demo row | Removed, as above (the 12% labels on lines 1058 and 1060 too) |
| frontend/index.html:567, 576 | "Professional tax for ₹15,000 salary in Ahmedabad" | A salary typed in an example question | Kept: user input, not a legal figure |
| frontend/index.html:1089 | "Monthly PT on ₹15,000" | The same example salary in a demo table header | Kept |
| eval/questions.csv:10, 20, 21 | "₹15,000 salary", "gross 15000" | Salaries in eval questions | Kept |
| tests/test_jurisdiction.py:10, tests/test_rag.py:79–80 | "15000 salary" | Salaries in test questions | Kept |
| tests/test_tools.py:45, 52, 55, 59, 82, 83, 90 | 15000 / 15,000 | Salaries passed to calculators that use the made-up fixture rates | Kept |
| tests/test_eval.py:138 | "₹15,000 salary" | Eval question text | Kept |
| tests/test_sql_guard.py:68 | `{"value": 21000}` | An inline rates fixture that happens to equal the real ESIC ceiling | Changed to a made-up 20000, like the other fixtures |

A new test (`tests/test_deploy.py`) fails if a legal-looking number (PF/EPS/ESIC rates or ceilings) appears in `app/`, `ingest/`, `scripts/` or `frontend/`.

**What to learn:** audit first, then change. A blind search-and-replace for "15000" would have broken the example questions, which are salaries a user typed. Only figures that state the law (rates, ceilings) belong in rates.yaml, with the document they came from.
