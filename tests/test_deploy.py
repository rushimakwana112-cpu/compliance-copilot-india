"""Deployment guards: the cloud install stays small and no secret is committed."""

from __future__ import annotations

import subprocess
import sys

import yaml

from app.config import ROOT

HEAVY = ("torch", "sentence-transformers", "transformers", "pypdf", "pytesseract", "pdf2image", "reportlab")


def _packages(path: str) -> list[str]:
    lines = (ln.split("#")[0].strip() for ln in (ROOT / path).read_text().splitlines())
    return [ln for ln in lines if ln and not ln.startswith("-r")]


def test_cloud_requirements_are_light():
    pkgs = " ".join(_packages("requirements.txt")).lower()
    assert not [h for h in HEAVY if h in pkgs]
    assert "groq" in pkgs and "pinecone" in pkgs


def test_app_starts_without_ingest_libraries():
    code = ("import sys, app.main; "
            "print(sorted(m for m in ('pypdf', 'bs4', 'tqdm', 'ingest.run', 'ingest.parse') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_render_blueprint_has_no_secret_values():
    svc = yaml.safe_load((ROOT / "render.yaml").read_text())["services"][0]
    assert svc["plan"] == "free" and "uvicorn app.main:app" in svc["startCommand"] and "$PORT" in svc["startCommand"]
    env = {e["key"]: e for e in svc["envVars"]}
    for key in ("GROQ_API_KEY", "PINECONE_API_KEY"):
        assert env[key].get("sync") is False and "value" not in env[key]
    assert env["PUBLIC_MODE"]["value"] == "true"


def test_render_blueprint_matches_the_deploy_plan():
    svc = yaml.safe_load((ROOT / "render.yaml").read_text())["services"][0]
    assert svc["runtime"] == "python" and svc["plan"] == "free"
    assert svc["buildCommand"] == "pip install -r requirements.txt"
    assert svc["startCommand"] == "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
    assert svc["healthCheckPath"] == "/api/health"
    values = {e["key"]: e.get("value") for e in svc["envVars"]}
    assert values["LLM_PROVIDER"] == "groq" and values["EMBED_PROVIDER"] == "pinecone"
    assert values["PUBLIC_MODE"] == "true" and values["SCORE_THRESHOLD"] == "0.38"
    assert values["GROQ_MODEL"] == "qwen/qwen3.8-27b" and values["PINECONE_INDEX"] == "compliance-copilot-cloud"


def test_sample_db_is_shipped():
    assert (ROOT / "data" / "sample_payroll.db").stat().st_size > 0


def test_healthz_is_cheap(services):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(services)) as c:
        assert c.get("/healthz").json() == {"ok": True}


# Statutory rates and ceilings (PF 12%, EPS 8.33%, EPS cap, ESIC ceiling and rates). They may live only in
# data/rates.yaml with their source; salaries typed in example questions are fine and not matched here.
LEGAL_NUMBER = r"(?<![\d.,])(?:15000|21000|25000|0\.12|0\.0833|8\.33|0\.0075|0\.0325|0\.75|3\.25)(?![\d,])"


def test_no_hard_coded_legal_numbers():
    import re

    hits = []
    for folder in ("app", "ingest", "scripts", "frontend"):
        for path in (ROOT / folder).rglob("*"):
            if path.suffix not in (".py", ".html", ".js", ".ps1") or "__pycache__" in path.parts:
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(LEGAL_NUMBER, line) and "rgba(" not in line:
                    hits.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()[:80]}")
    assert hits == []


def test_publish_readiness():
    """The public demo: PUBLIC_MODE on, rate limit and cache on, .env never committed, wake-up notice shown."""
    from app.config import Settings

    svc = yaml.safe_load((ROOT / "render.yaml").read_text())["services"][0]
    env = {e["key"]: e for e in svc["envVars"]}
    assert env["PUBLIC_MODE"]["value"] == "true" and svc["healthCheckPath"] == "/api/health"
    s = Settings(_env_file=None)
    assert s.rate_limit_per_min > 0 and s.rate_limit_per_day > 0 and s.cache_size > 0
    ignored = [ln.strip() for ln in (ROOT / ".gitignore").read_text().splitlines()]
    assert ".env" in ignored
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    assert ".env" not in tracked and ".env.example" in tracked
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "Waking up the server" in html and 'fetch(API_BASE + "/api/health"' in html


def test_readme_eval_table_is_rewritten_from_the_report(services, store, llm, tmp_path):
    from eval.run import load_questions, run, summarize, write_report
    from scripts.update_readme_eval import update

    store.add("gratuity fixed term employee continuous service eligible", doc="Code on Social Security, 2020",
              page=40, scope="central", status="current", doc_type="act")
    llm.responder = lambda m, j: "Fixed-term employees qualify after continuous service [1]."
    rows = [r for r in load_questions() if r.id in {"q02", "q09", "q26"}]
    results = run(services, rows, docs=["Code on Social Security, 2020"])
    report = write_report(results, summarize(results), services, None, tmp_path / "r.md")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    new = update(readme, report)
    block = new.split("<!-- eval-results:start")[1].split("<!-- eval-results:end -->")[0]
    assert "| route accuracy | 100% |" in block and "| not_found recall | 100% |" in block
    assert "measured on 2 questions, 1 skipped because their documents aren't indexed yet" in block
    assert new.replace(block, "") == readme.replace(readme.split("<!-- eval-results:start")[1]
                                                    .split("<!-- eval-results:end -->")[0], "")  # nothing else changed


def test_readme_update_refuses_an_empty_report():
    import pytest

    from scripts.update_readme_eval import update

    with pytest.raises(ValueError, match="no results"):
        update((ROOT / "README.md").read_text(encoding="utf-8"), "# Evaluation report\n\nNot run yet.")


def test_run_all_script_covers_every_step():
    ps1 = (ROOT / "scripts" / "run_all.ps1").read_text(encoding="utf-8")
    assert ps1.isascii()  # Windows PowerShell 5.1 misreads UTF-8 without a BOM
    order = [".venv\\Scripts\\Activate.ps1", "HEAD branch:", "git checkout", "git pull", "pip install", "pytest", "ingest.run --reindex",
             "eval.run", "scripts.update_readme_eval", "git commit", "git push"]
    body = ps1.split('$ErrorActionPreference = "Stop"')[1]  # the code, not the help text at the top
    positions = [body.index(step) for step in order]
    assert positions == sorted(positions)
    assert ps1.count("$LASTEXITCODE -ne 0") >= 1 and "STOPPED at:" in ps1


def test_health_check_path_answers_200_even_when_degraded(settings, llm):
    """Render restarts a service whose health check fails; a missing key must show as degraded, not a crash loop."""
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.services import Services

    llm.healthy = False
    with TestClient(create_app(Services(settings, llm, None, "PINECONE_API_KEY is not set"))) as c:
        r = c.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "degraded"


def test_no_secret_in_git_history():
    log = subprocess.run(["git", "log", "--all", "-p"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    import re

    assert not re.search(r"gsk_[A-Za-z0-9]{20,}|pcsk_[A-Za-z0-9_]{20,}", log)
    files = subprocess.run(["git", "log", "--all", "--name-only", "--format="], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    assert ".env" not in files


def test_public_readme_and_deploy_guide():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    top = readme.split("## How it works")[0]
    assert "LIVE_URL_HERE" in top and "render.com/deploy?repo=" in top and top.count("\n1. ") + top.count("\n2. ") >= 2
    for section in ("## How it works", "```mermaid", "## Sources used", "## Evaluation", "<!-- eval-results:start",
                    "## Known free-tier limits", "## Not legal advice", "## Run it locally (Windows PowerShell)"):
        assert section in readme, section
    deploy = (ROOT / "DEPLOY.md").read_text(encoding="utf-8")
    assert "GROQ_API_KEY" in deploy and "PINECONE_API_KEY" in deploy and "/api/health" in deploy


def test_ci_runs_pytest_on_push_to_main():
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yml").read_text())
    on = wf.get("on", wf.get(True))  # YAML 1.1 reads the key `on` as True
    assert on["push"]["branches"] == ["main"]
    steps = " ".join(str(s.get("run", "")) for s in wf["jobs"]["pytest"]["steps"])
    assert "pytest" in steps and "requirements-dev.txt" in steps
