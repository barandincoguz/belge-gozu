import json
from pathlib import Path
from unittest.mock import patch

import httpx

from belge_gozu.corpus.download import download_all
from belge_gozu.corpus.manifest import load_manifest_from_text

CSV = """doc_id,doc_name,doc_type,url
a,Belge A,kanun,https://example.org/a.pdf
b,Belge B,kanun,https://example.org/b.pdf
"""


def make_client(fail_ids: set[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        name = str(request.url).rsplit("/", 1)[-1].removesuffix(".pdf")
        if name in fail_ids:
            return httpx.Response(404)
        return httpx.Response(200, content=b"%PDF-1.4 " + name.encode())

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_and_skip_on_rerun(tmp_path: Path):
    rows = load_manifest_from_text(CSV)
    sleeps: list[float] = []
    r1 = download_all(rows, tmp_path, make_client(set()), delay_s=1.0, sleep=sleeps.append)
    assert r1.ok == ["a", "b"] and r1.failed == []
    assert (tmp_path / "pdf" / "a.pdf").read_bytes().startswith(b"%PDF")
    assert sleeps == [1.0, 1.0]  # her istekten önce nazik bekleme
    r2 = download_all(rows, tmp_path, make_client(set()), delay_s=1.0, sleep=sleeps.append)
    assert r2.skipped == ["a", "b"] and len(sleeps) == 2  # idempotent: yeniden inmez


def test_failure_recorded(tmp_path: Path):
    rows = load_manifest_from_text(CSV)
    r = download_all(rows, tmp_path, make_client({"b"}), delay_s=0, sleep=lambda _: None)
    assert r.ok == ["a"] and r.failed == ["b"]
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["b"]["status"] == "failed"


def test_invalid_url_recorded_not_raised(tmp_path: Path):
    csv_with_bad = """doc_id,doc_name,doc_type,url
a,Belge A,kanun,http://[invalid-url
b,Belge B,kanun,https://example.org/b.pdf
"""
    rows = load_manifest_from_text(csv_with_bad)

    def handler_with_invalid(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "%5B" in url_str or "[" in url_str:
            raise httpx.InvalidURL(url_str)
        name = url_str.rsplit("/", 1)[-1].removesuffix(".pdf")
        return httpx.Response(200, content=b"%PDF-1.4 " + name.encode())

    client = httpx.Client(transport=httpx.MockTransport(handler_with_invalid))
    r = download_all(rows, tmp_path, client, delay_s=0, sleep=lambda _: None)
    assert r.ok == ["b"] and r.failed == ["a"]
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["a"]["status"] == "failed"


def test_oserror_recorded_not_raised(tmp_path: Path):
    rows = load_manifest_from_text(CSV)
    with patch("pathlib.Path.write_bytes", side_effect=OSError("disk full")):
        r = download_all(rows, tmp_path, make_client(set()), delay_s=0, sleep=lambda _: None)
        assert r.ok == [] and r.failed == ["a", "b"]
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["a"]["status"] == "failed"
        assert state["b"]["status"] == "failed"


def test_corrupt_state_self_heals(tmp_path: Path):
    rows = load_manifest_from_text(CSV)
    state_path = tmp_path / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{truncated")
    r = download_all(rows, tmp_path, make_client(set()), delay_s=0, sleep=lambda _: None)
    assert r.ok == ["a", "b"] and r.failed == []
    state = json.loads(state_path.read_text())
    assert state["a"]["status"] == "ok"
    assert state["b"]["status"] == "ok"
