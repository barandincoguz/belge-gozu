from pathlib import Path

import pymupdf

from belge_gozu.corpus.text import extract_page_texts


def _pdf(path: Path, lines: list[str]) -> None:
    """Her satır için bir sayfa: sayfa N'nin metni lines[N-1] ile başlar."""
    doc = pymupdf.open()
    for text in lines:
        page = doc.new_page(width=300, height=400)
        page.insert_text((40, 60), text)
    doc.save(path)
    doc.close()


def test_extract_reads_1_based_pages(tmp_path: Path):
    _pdf(tmp_path / "k1.pdf", ["birinci sayfa", "ikinci sayfa"])
    df = extract_page_texts(tmp_path, ["k1:1", "k1:2"])
    assert df["page_id"].tolist() == ["k1:1", "k1:2"]
    assert "birinci" in df["text"][0] and "ikinci" in df["text"][1]


def test_row_order_follows_page_ids_not_pdf_order(tmp_path: Path):
    """Sıra sözleşmesi: dönen satırlar page_ids sırasında (BM25 hizası buna bağlı)."""
    _pdf(tmp_path / "k1.pdf", ["bir", "iki", "üç"])
    _pdf(tmp_path / "k2.pdf", ["dört"])
    ids = ["k1:3", "k2:1", "k1:1"]
    df = extract_page_texts(tmp_path, ids)
    assert df["page_id"].tolist() == ids
    assert "üç" in df["text"][0] and "dört" in df["text"][1] and "bir" in df["text"][2]


def test_missing_pdf_and_missing_page_give_empty_string(tmp_path: Path):
    """Eksik PDF / PDF'te olmayan sayfa: satır DÜŞMEZ, boş string olur.

    Satır düşürmek hizalamayı bozardı — indeksin n sayfası ile metnin n
    satırı birebir eşleşmek zorunda."""
    _pdf(tmp_path / "var.pdf", ["tek sayfa"])
    df = extract_page_texts(tmp_path, ["var:1", "var:2", "yok:1"])
    assert len(df) == 3
    assert df["text"][0].strip() != ""
    assert df["text"][1] == "" and df["text"][2] == ""


def test_empty_page_ids_returns_empty_frame_with_columns(tmp_path: Path):
    df = extract_page_texts(tmp_path, [])
    assert list(df.columns) == ["page_id", "text"] and len(df) == 0
