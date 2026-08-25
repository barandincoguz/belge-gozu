from pathlib import Path

import pymupdf as fitz
from typer.testing import CliRunner

from belge_gozu.cli import app

runner = CliRunner()

CSV = """doc_id,doc_name,doc_type,url
d1,Deneme Belgesi,kanun,https://example.org/d1.pdf
"""


def make_pdf(path: Path, pages: int) -> None:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=300, height=400)
        page.insert_text((50, 50), f"Sayfa {i + 1}")
    doc.save(path)


def test_render_and_fake_build(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BG_INDEX_DIR", str(tmp_path / "index"))
    (tmp_path / "manifest").mkdir(parents=True)
    (tmp_path / "manifest" / "v0_manifest.csv").write_text(CSV, encoding="utf-8")
    (tmp_path / "pdf").mkdir()
    make_pdf(tmp_path / "pdf" / "d1.pdf", pages=2)

    r1 = runner.invoke(app, ["corpus", "render", "--dpi", "72"])
    assert r1.exit_code == 0, r1.output
    r2 = runner.invoke(app, ["index", "build", "--fake"])
    assert r2.exit_code == 0, r2.output
    assert (tmp_path / "index" / "tokens.npy").exists()
    assert (tmp_path / "index" / "meta.parquet").exists()
