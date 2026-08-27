from pathlib import Path

import httpx
import pytest

from belge_gozu.corpus.manifest import load_manifest, load_manifest_from_text, probe

REPO_ROOT = Path(__file__).resolve().parents[2]  # veri yolları CWD'ye değil repo köküne göre

CSV = """doc_id,doc_name,doc_type,url
k6098,Türk Borçlar Kanunu,kanun,https://example.org/1.5.6098.pdf
rg1930,RG 1930 örneği,rg_tarihi,https://example.org/arsiv/1519.pdf
"""


def test_load(tmp_path: Path):
    p = tmp_path / "m.csv"
    p.write_text(CSV, encoding="utf-8")
    rows = load_manifest(p)
    assert [r.doc_id for r in rows] == ["k6098", "rg1930"]
    assert rows[0].doc_type == "kanun"


def test_bad_type_rejected(tmp_path: Path):
    p = tmp_path / "m.csv"
    p.write_text(CSV.replace("rg_tarihi", "bilinmeyen"), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(p)


def test_probe():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200 if "6098" in str(request.url) else 404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    rows = load_manifest_from_text(CSV)
    assert probe(rows, client) == [("k6098", 200), ("rg1930", 404)]


def test_shipped_manifest_parses_and_ids_unique():
    rows = load_manifest(REPO_ROOT / "data" / "manifest" / "v0_manifest.csv")
    ids = [r.doc_id for r in rows]
    assert len(ids) == len(set(ids))
    assert len(rows) >= 56


def test_http_client_keeps_tls_verification():
    import ssl

    from belge_gozu.corpus.manifest import build_ssl_context

    ctx = build_ssl_context()
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname


def test_probe_invalid_url_recorded_not_raised(monkeypatch):
    csv_with_bad_url = """doc_id,doc_name,doc_type,url
k6098,Türk Borçlar Kanunu,kanun,https://example.org/1.5.6098.pdf
rg1930,RG 1930 örneği,rg_tarihi,http://[bad
"""
    rows = load_manifest_from_text(csv_with_bad_url)

    def mock_head(url, **kwargs):
        if url == "http://[bad":
            raise httpx.InvalidURL("Invalid URL")
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    monkeypatch.setattr(client, "head", mock_head)
    results = probe(rows, client)
    # bad row recorded as failure, no exception raised
    assert ("rg1930", 0) in results
    # good row still probed
    assert ("k6098", 200) in results
