"""The downloader used by POST /api/ingest/url and ingest.run: SSRF and size guards, with no network."""

from __future__ import annotations

import httpx
import pytest

from app import catalog
from ingest import download as dl

def fake_resolve(table: dict[str, list[str]]):
    def resolve(host: str) -> list[str]:
        if host not in table:
            raise OSError("not found")
        return table[host]
    return resolve


@pytest.fixture(autouse=True)
def no_manifest(monkeypatch):
    monkeypatch.setattr(catalog, "update_manifest", lambda *a, **k: None)


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


@pytest.mark.parametrize("url, ips, reason", [
    ("https://localhost/a.pdf", ["127.0.0.1"], "private or local"),
    ("https://127.0.0.1/a.pdf", ["127.0.0.1"], "private or local"),
    ("https://10.0.0.5/a.pdf", ["10.0.0.5"], "private or local"),
    ("https://192.168.1.1/a.pdf", ["192.168.1.1"], "private or local"),
    ("https://169.254.169.254/latest/meta-data", ["169.254.169.254"], "private or local"),  # cloud metadata
    ("https://[::1]/a.pdf", ["::1"], "private or local"),
    ("https://mapped.test/a.pdf", ["::ffff:127.0.0.1"], "private or local"),
    ("https://one-bad-ip.test/a.pdf", ["8.8.8.8", "127.0.0.1"], "private or local"),  # any private answer blocks
    ("http://8.8.8.8/a.pdf", ["8.8.8.8"], "only https"),
    ("https://nowhere.test/a.pdf", None, "not found"),
])
def test_private_and_local_addresses_are_refused(url, ips, reason):
    host = httpx.URL(url).host
    assert reason in dl.blocked_reason(url, fake_resolve({host: ips} if ips else {}))


def test_public_address_is_allowed():
    assert dl.blocked_reason("https://8.8.8.8/a.pdf", fake_resolve({"8.8.8.8": ["8.8.8.8"]})) is None


def test_private_link_is_never_requested(tmp_path):
    requested = []
    c = client(lambda req: requested.append(req.url) or httpx.Response(200, content=b"%PDF-1.4"))
    out = dl.download({"doc": "Secret", "url": "https://127.0.0.1/s.pdf"}, tmp_path, c,
                      resolve=fake_resolve({"127.0.0.1": ["127.0.0.1"]}))
    assert out.path is None and "private or local" in out.status and requested == []


def test_redirect_to_a_private_address_is_refused(tmp_path):
    """A public https link that redirects to localhost (or to http) must not be followed."""
    requested = []

    def handler(req):
        requested.append(str(req.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1:8000/admin.pdf"})
    out = dl.download({"doc": "Bounce", "url": "https://8.8.8.8/a.pdf"}, tmp_path, client(handler),
                      resolve=fake_resolve({"8.8.8.8": ["8.8.8.8"], "127.0.0.1": ["127.0.0.1"]}))
    assert out.path is None and out.status.startswith("failed:") and requested == ["https://8.8.8.8/a.pdf"]


def test_public_redirect_is_followed(tmp_path):
    def handler(req):
        if req.url.path == "/old.pdf":
            return httpx.Response(301, headers={"location": "/new.pdf"})
        return httpx.Response(200, content=b"%PDF-1.4 ok", headers={"content-type": "application/pdf"})
    out = dl.download({"doc": "Moved", "url": "https://8.8.8.8/old.pdf"}, tmp_path, client(handler),
                      resolve=fake_resolve({"8.8.8.8": ["8.8.8.8"]}))
    assert out.status == "downloaded" and out.path.read_bytes() == b"%PDF-1.4 ok"


def test_redirect_loop_stops(tmp_path):
    c = client(lambda req: httpx.Response(302, headers={"location": "/again"}))
    out = dl.download({"doc": "Loop", "url": "https://8.8.8.8/a.pdf"}, tmp_path, c,
                      resolve=fake_resolve({"8.8.8.8": ["8.8.8.8"]}))
    assert out.status == "failed: too many redirects"


def test_download_stops_at_the_size_limit(tmp_path):
    c = client(lambda req: httpx.Response(200, content=b"%PDF" + b"x" * (3 * 1024 * 1024)))
    out = dl.download({"doc": "Huge", "url": "https://8.8.8.8/a.pdf"}, tmp_path, c, max_bytes=1024 * 1024,
                      resolve=fake_resolve({"8.8.8.8": ["8.8.8.8"]}))
    assert out.path is None and out.status == "failed: larger than 1 MB"
    assert not list(tmp_path.iterdir())  # nothing written
