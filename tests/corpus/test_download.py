import json
from pathlib import Path

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
