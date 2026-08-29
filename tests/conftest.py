from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from belge_gozu.index.encode import FakeEncoder
from belge_gozu.index.manifest import corpus_checksum, write_manifest
from belge_gozu.index.store import PackedIndex
from tests.index.test_manifest import make_manifest

# Metin kanalı (P1) fikstür metinleri — page_id -> sayfa metni.
# 1. sayfa başlık satırları KASITLI olarak gerçek biçimde: `extract_doc_name_tokens`
# bu satırlardan doküman adı türetir (d0 -> {"meden"}, d1 -> {"iş"}, d2 -> {"ceza"}),
# yani doküman-adı yönlendirmesi de bu fikstürle sınanabilir.
TINY_TEXTS = {
    "d0:1": (
        "TÜRK MEDENİ KANUNU\nYerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir.\n"
    ),
    "d1:1": ("İŞ KANUNU\nYıllık ücretli izin süresi hizmet süresine göre belirlenir.\n"),
    "d2:1": "TÜRK CEZA KANUNU\nKimseye suçu olmadan ceza verilemez.\n",
}


@pytest.fixture
def tiny_corpus(tmp_path: Path):
    """3 sayfalık sahte korpus: görüntüler + meta.parquet + FakeEncoder indeksi.

    Manifest `make_manifest()` varsayılanlarıyla yazılır; bunlar artık ÜRETİM
    değerleridir (train-compat-v1 + train-compat doküman prompt'u sha'sı).
    `FakeEncoder`ın `query_format`/`doc_prompt_sha256` niteliği yok, bu yüzden
    `create_app` uyumluluk kontrolü config'ten çözülen üretim değerlerine düşer
    (final review IMPORTANT-2) — yani bu fikstürle kurulan her app testi artık
    kontrolü gerçekten çalıştırır.

    P1: indeks dizinine ayrıca `page_texts.parquet` yazılır (hibrit pipeline'ın
    metin kanalı artefaktı) — varsayılan pipeline hibrit olduğu için bu dosya
    olmadan hiçbir app testi ayağa kalkamazdı. `corpus_checksum` bu dosyayı
    OKUMAZ, dolayısıyla manifest'i geçersizleştirmez.
    """
    enc = FakeEncoder()
    images, ids, records = [], [], []
    for i in range(3):
        rel = f"images/d{i}/0001.webp"
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (32, 32), (i * 40, 10, 10))
        img.save(p, format="WEBP")
        images.append(img)
        ids.append(f"d{i}:1")
        records.append(
            {
                "page_id": f"d{i}:1",
                "doc_id": f"d{i}",
                "doc_name": f"Belge {i}",
                "doc_type": "kanun",
                "source_url": "https://example.org",
                "page_no": 1,
                "image_path": rel,
            }
        )
    meta = pd.DataFrame.from_records(records)
    meta.to_parquet(tmp_path / "meta.parquet", index=False)
    idx = PackedIndex.build(ids, enc.encode_pages(images))
    idx_dir = tmp_path / "index"
    idx.save(idx_dir)
    meta.to_parquet(idx_dir / "meta.parquet", index=False)
    pd.DataFrame({"page_id": ids, "text": [TINY_TEXTS[i] for i in ids]}).to_parquet(
        idx_dir / "page_texts.parquet", index=False
    )
    write_manifest(
        idx_dir, make_manifest(corpus_checksum=corpus_checksum(idx_dir), n_pages=3, n_tokens=24)
    )
    return tmp_path, enc, np.array([])  # (data_dir, encoder, _)
