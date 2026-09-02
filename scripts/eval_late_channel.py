"""ÜRETİM modülleriyle geç-etkileşim kanalını ölçer — /tmp betikleriyle aynı sayılar mı?

Cerrahi operasyonun doğrulaması: deney betiklerinin ürettiği sayıları, üretime
alınan sınıflar (`retrieval.late.LateInteractionChannel`,
`retrieval.union.union_candidates`, `index.colbert_encode.ColBERTEncoder`)
birebir üretmeli. Üretmezse ölçülen konfigürasyon ile sevk edilen konfigürasyon
farklıdır ve bu, ölçüm ortamında görünmez.

BM25 SAYFA üzerinde kalır — donmuş reçete hiç değişmez (bkz. retrieval/late.py).

    uv run python scripts/eval_late_channel.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from belge_gozu.bench.metrics import recall_at_k  # noqa: E402
from belge_gozu.config import Settings  # noqa: E402
from belge_gozu.index.colbert_encode import ColBERTEncoder  # noqa: E402
from belge_gozu.retrieval.hybrid import load_text_channel  # noqa: E402
from belge_gozu.retrieval.late import LateInteractionChannel  # noqa: E402
from belge_gozu.retrieval.text import route_window  # noqa: E402
from belge_gozu.retrieval.text import routed_docs as routed  # noqa: E402
from belge_gozu.retrieval.union import union_candidates  # noqa: E402

ARMS = ("data/index-colbert-mogan-f16", "data/index-colbert-colmm-f16")
KS = (5, 20, 50)
# Ölçüm betiklerinin (yalnız-insan n=47) ürettiği sayılar — üretim yolu bunları
# yeniden üretmeli, yoksa sevk edilen konfigürasyon ölçülenden farklıdır.
EXPECTED = {"R@5": 0.7766, "R@20": 0.9149, "R@50": 0.9362, "paraphrase R@50": 0.8571}


def load_channel(path: Path, chunk_pages: dict[str, tuple[str, ...]]) -> LateInteractionChannel:
    side = json.loads((path / "colbert.json").read_text())
    return LateInteractionChannel(
        embeddings=np.load(path / "embs.npy"),
        offsets=np.load(path / "offsets.npy"),
        chunk_ids=json.loads((path / "chunk_ids.json").read_text()),
        chunk_pages=chunk_pages,
        encoder=ColBERTEncoder(
            side["model_repo"], side["revision"], document_length=side["document_length"]
        ),
    )


def main() -> int:
    s = Settings()
    chunks = pd.read_parquet(s.index_dir / "chunks.parquet")
    chunk_pages = {r.chunk_id: tuple(r.page_ids) for r in chunks.itertuples()}
    pages = pd.read_parquet(s.index_dir / "page_texts.parquet")
    page_ids = list(pages.page_id)
    bm25, doc_names = load_text_channel(s.index_dir, page_ids)  # DONMUŞ, sayfa üzerinde

    channels = [load_channel(REPO_ROOT / p, chunk_pages) for p in ARMS]
    bench = (REPO_ROOT / "data/bench/canary_v2.jsonl").read_text().splitlines()
    rows = [json.loads(x) for x in bench if x.strip()]
    human = [
        r for r in rows if r["answerable"] and r.get("verification_kind") == "human"
    ]

    acc: dict[str, float] = defaultdict(float)
    par = 0.0
    n_par = 0
    per_slice: dict[str, float] = defaultdict(float)
    n_slice: dict[str, int] = defaultdict(int)
    for r in human:
        q = r["question"]
        order = np.argsort(bm25.scores(q), kind="stable")[::-1]
        ranking = route_window([page_ids[i] for i in order], routed(q, doc_names))
        for ch in channels:
            ranking = union_candidates(ranking, ch.candidate_pages(q))
        gold = set(r["gold_page_ids"])
        n_slice[r["slice"]] += 1
        for k in KS:
            acc[f"R@{k}"] += recall_at_k(gold, ranking, k)
        per_slice[r["slice"]] += recall_at_k(gold, ranking, 50)
        if r["slice"] == "paraphrase":
            n_par += 1
            par += recall_at_k(gold, ranking, 50)

    n = len(human)
    got = {k: v / n for k, v in acc.items()}
    got["paraphrase R@50"] = par / n_par
    print(f"ÜRETİM YOLU · n={n} · paraphrase n={n_par}\n")
    ok = True
    for key, want in EXPECTED.items():
        have = got[key]
        match = abs(have - want) < 1e-3
        ok &= match
        print(f"  {key:18} ölçülen {want:.4f}   üretim {have:.4f}   {'✓' if match else '✗ SAPMA'}")
    print(f"\n{'dilim':28}{'n':>4}{'R@50':>10}")
    for sl in sorted(per_slice):
        print(f"{sl:28}{n_slice[sl]:>4}{per_slice[sl] / n_slice[sl]:>10.4f}")
    if not ok:
        raise SystemExit("üretim yolu ölçülen sayıları üretmedi — konfigürasyonlar ayrışmış")
    print("\nÜRETİM YOLU ÖLÇÜMLE BİREBİR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
