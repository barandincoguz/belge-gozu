from pathlib import Path

import pandas as pd
import pymupdf as fitz

from belge_gozu.corpus.manifest import ManifestRow

META_COLUMNS = ["page_id", "doc_id", "doc_name", "doc_type", "source_url", "page_no", "image_path"]


def render_all(rows: list[ManifestRow], corpus_dir: Path, dpi: int = 150) -> pd.DataFrame:
    records: list[dict] = []
    for row in rows:
        pdf_path = corpus_dir / "pdf" / f"{row.doc_id}.pdf"
        if not pdf_path.exists():
            continue
        img_dir = corpus_dir / "images" / row.doc_id
        img_dir.mkdir(parents=True, exist_ok=True)
        try:
            with fitz.open(pdf_path) as doc:
                for i, page in enumerate(doc, start=1):  # type: ignore[arg-type]
                    rel = f"images/{row.doc_id}/{i:04d}.webp"
                    out = corpus_dir / rel
                    if not out.exists():
                        pix = page.get_pixmap(dpi=dpi)
                        pix.pil_save(out, format="WEBP", quality=80)
                    records.append(
                        {
                            "page_id": f"{row.doc_id}:{i}",
                            "doc_id": row.doc_id,
                            "doc_name": row.doc_name,
                            "doc_type": row.doc_type,
                            "source_url": row.url,
                            "page_no": i,
                            "image_path": rel,
                        }
                    )
        except (fitz.FileDataError, RuntimeError):
            continue
    df = pd.DataFrame.from_records(records, columns=META_COLUMNS)
    df.to_parquet(corpus_dir / "meta.parquet", index=False)
    return df
