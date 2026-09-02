"""ColBERT geç-etkileşim indeksini üretir (D2 metin kanalı).

    uv run python scripts/build_colbert_index.py --out data/index-colbert-mogan-f16

Ölçülen maliyet (MPS, 10.531 chunk, Mogan @ document_length=512):
512 sn kodlama · 1.918.277 vektör · 491 MB fp16 · chunk başına 182,2 vektör.
Yeniden inşa dakikalar sürdüğü için bu indeks bir veritabanı değil ÖNBELLEKTİR.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from belge_gozu.config import Settings  # noqa: E402
from belge_gozu.index.colbert_encode import (  # noqa: E402
    MOGAN_REPO,
    MOGAN_REVISION,
    ColBERTEncoder,
)

# Belge sorgu gibi kodlanmışsa TAM 32 vektör verir. Bu eşik o sessiz hatayı
# inşa zamanında yakalar (ölçülen gerçek değer 182,2).
MIN_VECTORS_PER_CHUNK = 40


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data/index-colbert-mogan-f16")
    ap.add_argument("--repo", default=MOGAN_REPO)
    ap.add_argument("--revision", default=MOGAN_REVISION)
    ap.add_argument("--doc-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    s = Settings()
    chunks = pd.read_parquet(s.index_dir / "chunks.parquet")
    texts, ids = list(chunks.index_text), list(chunks.chunk_id)
    print(f"{len(texts)} chunk · model {args.repo}@{args.revision[:7]} · dlen {args.doc_len}")

    enc = ColBERTEncoder(args.repo, args.revision, document_length=args.doc_len)
    t0 = time.time()
    parts, lens = [], []
    for i in range(0, len(texts), args.batch):
        vecs = enc.encode_documents(texts[i : i + args.batch], batch_size=args.batch)
        parts.extend(vecs)
        lens.extend(len(v) for v in vecs)
        if (i // args.batch) % 25 == 0:
            print(f"  {min(i + args.batch, len(texts))}/{len(texts)}  {time.time() - t0:.0f}s")

    embs = np.concatenate(parts).astype(np.float16)
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lens)

    per_chunk = embs.shape[0] / len(ids)
    if per_chunk <= MIN_VECTORS_PER_CHUNK:
        raise SystemExit(
            f"chunk başına {per_chunk:.1f} vektör — belgeler SORGU gibi kodlanmış olabilir "
            f"(sorgu tam olarak {enc.cfg.query_length} verir). İnşa durduruldu."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "embs.npy", embs)
    np.save(args.out / "offsets.npy", offsets)
    (args.out / "chunk_ids.json").write_text(json.dumps(ids, ensure_ascii=False))
    # Model kimliği IndexManifest'e YAZILMAZ: `index/compat.py` manifestleri
    # görsel kodlayıcıya karşı karşılaştırıyor, yabancı bir model kimliği
    # dondurulmuş bir uyumluluk kontrolünü düşürebilir. Ayrı sidecar.
    (args.out / "colbert.json").write_text(
        json.dumps(
            {
                "model_repo": args.repo,
                "revision": args.revision,
                "document_length": enc.cfg.document_length,
                "query_length": enc.cfg.query_length,
                "dim": int(embs.shape[1]),
                "n_chunks": len(ids),
                "n_vectors": int(embs.shape[0]),
                "vectors_per_chunk": round(per_chunk, 1),
                "dtype": "float16",
                "encode_seconds": round(time.time() - t0),
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"{embs.shape[0]:,} vektör · {embs.nbytes / 1e6:.0f} MB fp16 · {per_chunk:.1f}/chunk")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
