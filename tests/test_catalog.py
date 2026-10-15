"""The documents list on the deployed app, where ingestion never runs and data/manifest.json may be missing."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app import catalog
from app.main import create_app

SOURCES = """# comment
url,doc,scope,status,effective_date,doc_type
https://a.example/1.pdf,"Code on Wages, 2019",central,current,2025-11-21,act
# https://a.example/2.pdf,"Commented out Code",central,current,2025-11-21,act
https://a.example/3.pdf,"PIB explainer",central,current,,faq
"""


def _sources(tmp_path):
    path = tmp_path / "sources.csv"
    path.write_text(SOURCES, encoding="utf-8")
    return path


def test_without_manifest_sources_are_listed_as_indexed(tmp_path):
    docs = catalog.documents(_sources(tmp_path), tmp_path / "missing.json")
    assert [d["name"] for d in docs] == ["Code on Wages, 2019", "PIB explainer"]
    assert docs[0]["meta"] == "In force 2025-11-21" and docs[1]["meta"] == "Indexed"
    assert catalog.indexed_count(tmp_path / "missing.json", _sources(tmp_path)) == 2


def test_empty_manifest_counts_as_missing(tmp_path):
    man = tmp_path / "manifest.json"
    man.write_text("{}", encoding="utf-8")
    assert all(d["meta"] != "not indexed yet" for d in catalog.documents(_sources(tmp_path), man))


def test_with_manifest_unindexed_sources_say_so(tmp_path):
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"Code on Wages, 2019": {"chunks": 40, "effective_date": "2025-11-21"}}), encoding="utf-8")
    docs = {d["name"]: d["meta"] for d in catalog.documents(_sources(tmp_path), man)}
    assert docs == {"Code on Wages, 2019": "In force 2025-11-21", "PIB explainer": "not indexed yet"}
    assert catalog.indexed_count(man, _sources(tmp_path)) == 1


def test_public_mode_documents_and_health_without_manifest(services, tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "SOURCES_CSV", _sources(tmp_path))
    services.settings.public_mode = True
    with TestClient(create_app(services)) as c:
        docs = c.get("/api/documents")
        health = c.get("/api/health").json()
    assert docs.status_code == 200 and len(docs.json()) == 2
    assert health["docs"] == 2 and health["public_mode"] is True
