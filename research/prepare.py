"""Autoresearch hazırlık (BİR KEZ koşar; DONUK — bkz. research/program.md).

Üretir (data/research/ altına; hepsi yeniden-üretilebilir türev veri):
- page_texts.parquet : page_id + PDF katman metni (indeks sırasıyla hizalı;
  taranmış RG sayfalarında boş string)
- visual_scores.npz  : scores (n_q, n_pages) float32 — üretim int8 indeksinde
  ColSmol MaxSim skorları (görsel kanal DONUK; deneyler bunun üstüne kurulur)
- queries.json       : sorgu künyeleri (canary answerable=43 + 2 vaka analizi)

Koşum: uv run python research/prepare.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf

from belge_gozu.bench.dataset import load_bench
from belge_gozu.config import Settings
from belge_gozu.index.encode import ColSmolEncoder
from belge_gozu.index.loader import load_scorable_index
from belge_gozu.index.manifest import QUERY_FORMATS, read_manifest

OUT_DIR = Path("data/research")
CANARY = Path("data/bench/canary_v1.jsonl")
PDF_DIR = Path("data/pdf")

# Vitrin (chip) sorguları — birincil metriğe GİRMEZ (program.md: 2 soruya overfit yasak)
CASE_STUDIES = [
    {"qid": "chip1-uzun", "question": "Türk Medeni Kanunu'na göre yerleşim yeri nasıl tanımlanır?", "gold": ["k4721:4"]},
    {"qid": "chip2-izin", "question": "İş Kanunu'na göre yıllık ücretli izin süresi ne kadardır?", "gold": ["k4857:28", "k4857:27"]},
]


def extract_page_texts(page_ids: list[str]) -> pd.DataFrame:
    texts: dict[str, str] = {}
    for doc_id in sorted({pid.split(":", 1)[0] for pid in page_ids}):
        pdf = PDF_DIR / f"{doc_id}.pdf"
        if not pdf.exists():
            continue
        d = pymupdf.open(pdf)
        for i, page in enumerate(d, start=1):
            texts[f"{doc_id}:{i}"] = page.get_text()
        d.close()
    rows = [{"page_id": pid, "text": texts.get(pid, "")} for pid in page_ids]
    return pd.DataFrame(rows)


def main() -> None:
    s = Settings()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = load_scorable_index(s.index_dir)
    manifest = read_manifest(s.index_dir)
    page_ids = list(idx.page_ids)

    df = extract_page_texts(page_ids)
    empty = int((df.text.str.strip() == "").sum())
    df.to_parquet(OUT_DIR / "page_texts.parquet")
    print(f"metin: {len(df)} sayfa, {empty} boş (taranmış RG dahil)")

    bench = load_bench(CANARY, only_verified=False)
    answerable = [q for q in bench if q.answerable]
    queries = [
        {"qid": q.question_id, "question": q.question, "gold": list(q.gold_page_ids),
         "requires_visual": q.requires_visual, "role": "canary"}
        for q in answerable
    ] + [{**c, "requires_visual": False, "role": "case_study"} for c in CASE_STUDIES]

    enc = ColSmolEncoder(s.retriever_model, s.device, query_format=QUERY_FORMATS[s.query_format_id])
    mat = np.zeros((len(queries), len(page_ids)), dtype=np.float32)
    for j, q in enumerate(queries):
        mat[j] = idx.score_all(enc.encode_query(q["question"])).astype(np.float32)
        if (j + 1) % 10 == 0:
            print(f"  skor {j + 1}/{len(queries)}")

    np.savez_compressed(OUT_DIR / "visual_scores.npz", scores=mat)
    meta = {
        "index_dir": str(s.index_dir),
        "quantization": manifest.quantization if manifest else None,
        "query_format_id": s.query_format_id,
        "n_pages": len(page_ids),
        "queries": queries,
    }
    (OUT_DIR / "queries.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"görsel kanal: {mat.shape} — {s.index_dir} ({meta['quantization']})")
    print(f"sorgular: {len(answerable)} canary answerable + {len(CASE_STUDIES)} vaka analizi")
    print("-> data/research/{page_texts.parquet, visual_scores.npz, queries.json}")


if __name__ == "__main__":
    main()
