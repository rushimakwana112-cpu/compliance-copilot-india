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
    for key in ("GROQ_API_KEY", "PINECONE_API_KEY", "GROQ_MODEL"):
        assert env[key].get("sync") is False and "value" not in env[key]
    assert env["PUBLIC_MODE"]["value"] == "true"


def test_sample_db_is_shipped():
    assert (ROOT / "data" / "sample_payroll.db").stat().st_size > 0


def test_healthz_is_cheap(services):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(services)) as c:
        assert c.get("/healthz").json() == {"ok": True}
