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
