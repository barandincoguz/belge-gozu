from pathlib import Path

import httpx
import pytest

from belge_gozu.corpus.manifest import load_manifest, load_manifest_from_text, probe

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
