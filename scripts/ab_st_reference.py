"""T11/Step 2 — Sentence Transformers referansına karşı çapraz doğrulama.

Model kartı: "The Sentence Transformers configuration in this repository
reproduces the original training-time format." Yani model reposundaki ST
yapılandırması eğitim zamanı şablonunun BİRİNCİL kaynağıdır. Bu betik, bizim
`ColSmolEncoder`'ımızın (train-compat sorgu formatı + train-compat doküman
prompt'u) ST `MultiVectorEncoder` ile aynı girdiyi ve aynı embedding'leri
ürettiğini sayısal olarak gösterir — yani Step 1'de kilitlenen dizilerin doğru
olduğunu.

İki kademe var ve ikisi de PASS/FAIL basar (yeniden koşum kendi kendini doğrular):

  A) Token düzeyi (processor-only, model forward'ı YOK — bedelsiz)
     - her sorgu için bizim rendered string'imizin ve ST'nin ürettiği girdinin
       `input_ids`'leri birebir aynı mı;
     - negatif kontrol: iki doküman prompt'unun token sayıları farklı mı ve ST'nin
       doküman yolu hangisini kullanıyor (yani hangisi kilitli).
  B) Embedding düzeyi: token sayısı, sign uyuşması, max |a-b|.

Çalıştırma (model ağırlıkları HF önbelleğinden gelir, ~5 dk):

    uv run --with "sentence-transformers>=5.0" python scripts/ab_st_reference.py

`--device cpu` her iki tarafı da fp32'ye alır (bizim encoder mps/cuda'da fp16
koşar); artık farkın şablondan değil hassasiyetten geldiğini göstermek için:

    uv run --with "sentence-transformers>=5.0" python scripts/ab_st_reference.py --device cpu

CI'da koşmaz: model dokunan her şey runbook betiklerinde ya da `-m slow`'da.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from belge_gozu.index.manifest import (  # noqa: E402
    CPE_0_3_18_DOC_PROMPT,
    TRAIN_COMPAT_DOC_PROMPT,
    TRAIN_COMPAT_V1,
)

MODEL = "vidore/colSmol-500M"
QUERIES = [
    "Türk Medeni Kanunu'na göre yerleşim yeri nasıl tanımlanır?",
    "Yerleşim yeri nedir?",
]
AGREEMENT_TARGET = 0.99
# data/images/k6098/0001.webp için beklenen doküman token sayıları (referans sayfa
# yoksa sentetik görsele düşülür ve bu sayılar tutmaz — yalnız bilgi amaçlı basılır).
REFERENCE_PAGE = "images/k6098/0001.webp"
REFERENCE_DOC_TOKENS = {"train-compat": 871, "cpe-0.3.18": 875}


class Checks:
    """PASS/FAIL satırlarını basar ve genel sonucu tutar."""

    def __init__(self) -> None:
        self.ok = True

    def record(self, passed: bool, label: str) -> bool:
        self.ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        return passed


def _st_query_input_ids(ref, query: str):
    """ST'nin sorgu yolunun ürettiği input_ids (task='query' -> repo jinja'sı)."""
    fn = getattr(ref, "preprocess", None) or ref.tokenize
    return fn([query], task="query")["input_ids"][0]


def _token_level_checks(checks: Checks, ours_processor, ref, image) -> None:
    """Model forward'ı olmadan girdinin özdeşliğini kanıtlar."""
    import torch

    print("\n" + "=" * 72)
    print("A) Token düzeyi kontroller (processor-only, forward yok)")
    print("=" * 72)

    for i, q in enumerate(QUERIES, 1):
        rendered = TRAIN_COMPAT_V1.render(q)
        ours_ids = ours_processor.process_texts([rendered])["input_ids"][0]
        ref_ids = _st_query_input_ids(ref, q)
        identical = bool(
            ours_ids.shape == ref_ids.shape and torch.equal(ours_ids.cpu(), ref_ids.cpu())
        )
        print(f"\nsorgu {i}: {q}")
        print(f"  bizim rendered : {rendered!r}")
        print(f"  token sayısı   : bizim={len(ours_ids)} ST={len(ref_ids)}")
        print(f"  IDS IDENTICAL  : {identical}")
        checks.record(identical, f"sorgu {i}: input_ids ST ile birebir aynı")

    # Negatif kontrol: iki doküman prompt'u ayırt edilebilir mi, hangisi kilitli?
    print("\ndoküman prompt'u negatif kontrolü")
    counts: dict[str, int] = {}
    saved = ours_processor.visual_prompt_prefix
    try:
        for name, prompt in (
            ("train-compat", TRAIN_COMPAT_DOC_PROMPT),
            ("cpe-0.3.18", CPE_0_3_18_DOC_PROMPT),
        ):
            ours_processor.visual_prompt_prefix = prompt
            counts[name] = int(ours_processor.process_images([image])["input_ids"].shape[1])
            exp = REFERENCE_DOC_TOKENS[name]
            print(f"  {name:<12} -> {counts[name]} token  (referans sayfada beklenen {exp})")
            print(f"    {prompt!r}")
    finally:
        ours_processor.visual_prompt_prefix = saved

    checks.record(
        counts["train-compat"] != counts["cpe-0.3.18"],
        "iki doküman prompt'u ayırt edilebilir (token sayıları farklı)",
    )

    st_doc_tokens = int(ref.preprocess([image])["input_ids"].shape[1])
    print(f"  ST doküman yolu -> {st_doc_tokens} token")
    checks.record(
        st_doc_tokens == counts["train-compat"],
        f"ST doküman yolu train-compat prompt'unu kullanıyor ({st_doc_tokens}"
        f" == {counts['train-compat']}, cpe-0.3.18 = {counts['cpe-0.3.18']})",
    )


def _compare(checks: Checks, name: str, ours: np.ndarray, ref: np.ndarray) -> None:
    """Token sayılarını basar; eşitse sign uyuşması + max|a-b| ölçer."""
    print(f"\n[{name}]")
    print(f"  bizim  : {ours.shape}  dtype={ours.dtype}")
    print(f"  ST ref : {ref.shape}  dtype={ref.dtype}")
    if ours.shape != ref.shape:
        checks.record(False, f"{name}: şekil eşleşmiyor -> şablon farklı")
        return
    a = ours.astype(np.float32)
    b = ref.astype(np.float32)
    agree = float(((a > 0) == (b > 0)).mean())
    max_abs = float(np.abs(a - b).max())
    cos = float((a * b).sum(axis=1).mean())  # satırlar L2-normalize: ortalama kosinüs
    print(f"  sign uyuşması : {agree:.6f}  (hedef >= {AGREEMENT_TARGET})")
    print(f"  max |a-b|     : {max_abs:.6f}")
    print(f"  ort. kosinüs  : {cos:.6f}")
    checks.record(agree >= AGREEMENT_TARGET, f"{name}: sign uyuşması >= {AGREEMENT_TARGET}")


def _sample_image():
    """Referans sayfa (repo köküne göre çözülür); yoksa sentetik görsel."""
    from PIL import Image

    meta = REPO_ROOT / "data" / "meta.parquet"
    page = REPO_ROOT / "data" / REFERENCE_PAGE
    if page.exists():
        print(f"doküman örneği: {page}")
        with Image.open(page) as raw:
            return raw.convert("RGB")
    if meta.exists():
        import pandas as pd

        p = REPO_ROOT / "data" / pd.read_parquet(meta).iloc[0]["image_path"]
        if p.exists():
            print(f"doküman örneği: {p} (referans sayfa yok; token sayıları farklı olabilir)")
            with Image.open(p) as raw:
                return raw.convert("RGB")
    print(f"doküman örneği: sentetik ({page} yok; token sayıları farklı olacak)")
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (512, 384, 3), dtype=np.uint8))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--device",
        default=os.environ.get("BG_DEVICE", "auto"),
        help="ColSmolEncoder cihazı; 'cpu' her iki tarafı fp32'ye alır (varsayılan: auto)",
    )
    args = ap.parse_args()

    print("=" * 72)
    print("Step 1'de kilitlenen diziler")
    print("=" * 72)
    print(f"query  : {TRAIN_COMPAT_V1.render('X')!r}")
    print(f"doc    : {TRAIN_COMPAT_DOC_PROMPT!r}")

    from belge_gozu.index.encode import ColSmolEncoder

    print("\nbizim encoder yükleniyor...")
    ours = ColSmolEncoder(
        MODEL,
        args.device,
        query_format=TRAIN_COMPAT_V1,
        visual_prompt_override=TRAIN_COMPAT_DOC_PROMPT,
    )
    print(f"  device={ours.device} doc_prompt={ours.doc_prompt!r}")

    from sentence_transformers import MultiVectorEncoder  # lazy: yalnız bu betikte

    print("ST referansı yükleniyor...")
    ref = MultiVectorEncoder(MODEL, device=ours.device)

    checks = Checks()
    img = _sample_image()
    _token_level_checks(checks, ours.processor, ref, img)

    print("\n" + "=" * 72)
    print("B) Embedding düzeyi karşılaştırma")
    print("=" * 72)
    for i, q in enumerate(QUERIES, 1):
        _compare(
            checks,
            f"sorgu {i}: {q}",
            ours.encode_query(q),
            np.asarray(ref.encode_query([q], convert_to_numpy=True)[0]),
        )
    _compare(
        checks,
        "doküman (train-compat doc prompt)",
        ours.encode_pages([img])[0],
        np.asarray(ref.encode_document([img], convert_to_numpy=True)[0]),
    )

    print("\n" + "=" * 72)
    print("SONUÇ:", "GEÇTİ" if checks.ok else "GEÇMEDİ — şablon Step 1'e dönüp düzeltilmeli")
    print("=" * 72)
    return 0 if checks.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
