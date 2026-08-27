"""T11/Step 2 — Sentence Transformers referansına karşı çapraz doğrulama.

Model kartı: "The Sentence Transformers configuration in this repository
reproduces the original training-time format." Yani model reposundaki ST
yapılandırması eğitim zamanı şablonunun BİRİNCİL kaynağıdır. Bu betik, bizim
`ColSmolEncoder`'ımızın (train-compat sorgu formatı + train-compat doküman
prompt'u) ST `MultiVectorEncoder` ile aynı embedding'leri ürettiğini sayısal
olarak gösterir — yani Step 1'de kilitlenen dizilerin doğru olduğunu.

Çalıştırma (model ağırlıkları HF önbelleğinden gelir, ~5 dk):

    uv run --with "sentence-transformers>=5.0" python scripts/ab_st_reference.py

CI'da koşmaz: model dokunan her şey runbook betiklerinde ya da `-m slow`'da.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from belge_gozu.index.manifest import (  # noqa: E402
    TRAIN_COMPAT_DOC_PROMPT,
    TRAIN_COMPAT_V1,
)

MODEL = "vidore/colSmol-500M"
QUERIES = [
    "Türk Medeni Kanunu'na göre yerleşim yeri nasıl tanımlanır?",
    "Yerleşim yeri nedir?",
]
AGREEMENT_TARGET = 0.99


def _compare(name: str, ours: np.ndarray, ref: np.ndarray) -> bool:
    """Token sayılarını basar; eşitse sign uyuşması + max|a-b| döner."""
    print(f"\n[{name}]")
    print(f"  bizim  : {ours.shape}  dtype={ours.dtype}")
    print(f"  ST ref : {ref.shape}  dtype={ref.dtype}")
    if ours.shape != ref.shape:
        print("  -> TOKEN SAYISI/ŞEKİL UYUŞMUYOR: şablon farklı, karşılaştırma anlamsız")
        return False
    a = ours.astype(np.float32)
    b = ref.astype(np.float32)
    agree = float(((a > 0) == (b > 0)).mean())
    max_abs = float(np.abs(a - b).max())
    cos = float((a * b).sum(axis=1).mean())  # satırlar L2-normalize: ortalama kosinüs
    print(f"  sign uyuşması : {agree:.6f}  (hedef >= {AGREEMENT_TARGET})")
    print(f"  max |a-b|     : {max_abs:.6f}")
    print(f"  ort. kosinüs  : {cos:.6f}")
    if agree < AGREEMENT_TARGET:
        print("  -> UYUŞMA DÜŞÜK")
    return agree >= AGREEMENT_TARGET


def _sample_image():
    from PIL import Image

    meta = Path("data/meta.parquet")
    if meta.exists():
        import pandas as pd

        row = pd.read_parquet(meta).iloc[0]
        p = Path("data") / row["image_path"]
        if p.exists():
            print(f"doküman örneği: {p}")
            with Image.open(p) as raw:
                return raw.convert("RGB")
    print("doküman örneği: sentetik (data/meta.parquet yok)")
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (512, 384, 3), dtype=np.uint8))


def main() -> int:
    device = os.environ.get("BG_DEVICE", "auto")

    print("=" * 72)
    print("Step 1'de kilitlenen diziler")
    print("=" * 72)
    print(f"query  : {TRAIN_COMPAT_V1.render('X')!r}")
    print(f"doc    : {TRAIN_COMPAT_DOC_PROMPT!r}")

    from belge_gozu.index.encode import ColSmolEncoder

    print("\nbizim encoder yükleniyor...")
    ours = ColSmolEncoder(
        MODEL,
        device,
        query_format=TRAIN_COMPAT_V1,
        visual_prompt_override=TRAIN_COMPAT_DOC_PROMPT,
    )
    print(f"  device={ours.device} doc_prompt={ours.doc_prompt!r}")

    from sentence_transformers import MultiVectorEncoder  # lazy: yalnız bu betikte

    print("ST referansı yükleniyor...")
    ref = MultiVectorEncoder(MODEL, device=ours.device)

    ok = True
    for i, q in enumerate(QUERIES, 1):
        our_q = ours.encode_query(q)
        ref_q = ref.encode_query([q], convert_to_numpy=True)[0]
        ok &= _compare(f"sorgu {i}: {q}", our_q, np.asarray(ref_q))

    img = _sample_image()
    our_d = ours.encode_pages([img])[0]
    ref_d = ref.encode_document([img], convert_to_numpy=True)[0]
    ok &= _compare("doküman (train-compat doc prompt)", our_d, np.asarray(ref_d))

    print("\n" + "=" * 72)
    print("SONUÇ:", "GEÇTİ" if ok else "GEÇMEDİ — şablon Step 1'e dönüp düzeltilmeli")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
