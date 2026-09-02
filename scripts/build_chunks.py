"""Chunk artefaktını üretir: `<index_dir>/chunks.parquet`.

Hem BM25 hem semantik kanal AYNI chunk listesini okumalı — ayrı ayrı üretirlerse
sıralamaları sessizce ayrışır ve birleşim yanlış chunk'ları eşleştirir. Metin
kanalının `page_texts.parquet`e bağlı olması gibi, D2 de bu artefakta bağlanır.

    uv run python scripts/build_chunks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from belge_gozu.config import Settings  # noqa: E402
from belge_gozu.corpus.chunking import chunk_document  # noqa: E402

ARTIFACT = "chunks.parquet"


def build(index_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(index_dir / "page_texts.parquet")
    df["doc"] = df.page_id.astype(str).str.split(":").str[0]
    df["pno"] = df.page_id.astype(str).str.split(":").str[1].astype(int)
    rows = []
    for doc, g in df.groupby("doc", sort=True):
        pages = [(int(r.pno), r.text) for _, r in g.sort_values("pno").iterrows()]
        for c in chunk_document(doc, pages):
            rows.append(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "kind": c.kind,
                    "heading": c.heading,
                    "text": c.text,
                    # parquet liste tutar; tuple -> list
                    "page_ids": list(c.page_ids),
                    # indekslenen metin: başlık ÖNEK olarak eklenir (B+ kolu,
                    # ölçüldü: sabit k'da +0.0328 R@5, tarihi-tarama'da +0.25)
                    "index_text": f"{c.heading}\n{c.text}" if c.heading else c.text,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    s = Settings()
    out = s.index_dir / ARTIFACT
    df = build(s.index_dir)
    df.to_parquet(out, index=False)
    n_pages = len({p for ps in df.page_ids for p in ps})
    print(f"{len(df)} chunk -> {out}")
    print(f"  madde {int((df.kind == 'article').sum())} · sayfa {int((df.kind == 'page').sum())}")
    print(f"  kapsanan sayfa: {n_pages}")
    print(f"  index_text medyan uzunluk: {int(df.index_text.str.len().median())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
