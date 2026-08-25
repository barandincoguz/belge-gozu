from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd

from belge_gozu.corpus.manifest import load_manifest_from_text
from belge_gozu.corpus.render import render_all

CSV = """doc_id,doc_name,doc_type,url
d1,Deneme Belgesi,kanun,https://example.org/d1.pdf
"""


def make_pdf(path: Path, pages: int) -> None:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=300, height=400)
        page.insert_text((50, 50), f"Sayfa {i + 1}")
    doc.save(path)


def test_render(tmp_path: Path):
    (tmp_path / "pdf").mkdir()
    make_pdf(tmp_path / "pdf" / "d1.pdf", pages=3)
    rows = load_manifest_from_text(CSV)
    df = render_all(rows, tmp_path, dpi=72)
    assert len(df) == 3
    assert df.page_id.tolist() == ["d1:1", "d1:2", "d1:3"]
    assert (tmp_path / "images" / "d1" / "0002.webp").exists()
    saved = pd.read_parquet(tmp_path / "meta.parquet")
    assert saved.page_id.tolist() == df.page_id.tolist()


def test_render_skips_missing_pdf(tmp_path: Path):
    (tmp_path / "pdf").mkdir()
    df = render_all(load_manifest_from_text(CSV), tmp_path, dpi=72)
    assert df.empty
