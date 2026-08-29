"""PDF metin katmanı çıkarımı — hibrit getirimin metin kanalı artefaktı (P1).

`research/prepare.py`'nin araştırma döngüsünde kullandığı çıkarımın üretim
sürümü: aynı pymupdf çağrısı, aynı 1-tabanlı sayfa numaralandırması. Tek fark
sözleşmenin sıkılaştırılması — pdf dizini parametre, dönen satır sırası verilen
`page_ids` ile BİREBİR hizalı. Hizalama kritik: BM25 skor vektörü indeksin
`page_ids` sırasına göre yorumlanıyor, bir satır kayması sessizce yanlış sayfayı
döndürür (bu yüzden `app/main.py` serve'de listeyi ayrıca karşılaştırır).

Ölçüm: 4222 sayfanın 4221'inde metin katmanı var (RG taramaları dahil, kaynak
OCR'lı) — findings 2026-08-29-autoresearch-text-channel.md §4.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pymupdf


def extract_page_texts(pdf_dir: Path, page_ids: list[str]) -> pd.DataFrame:
    """`page_id` ("doc:N", N 1-tabanlı) -> PDF metin katmanı; sıra page_ids sırası.

    Eksik PDF ya da PDF'te olmayan sayfa numarası BOŞ STRING üretir (hata
    değil): korpus 50 kanun + 6 RG taraması karışımı ve metin katmanı olmayan
    sayfa meşru bir durum — BM25 böyle bir sayfayı doğal olarak hiç
    döndürmez. Sessizce satır DÜŞÜRMEK ise hizalamayı bozardı, bu yüzden her
    page_id için tam olarak bir satır yazılır.
    """
    texts: dict[str, str] = {}
    for doc_id in sorted({pid.partition(":")[0] for pid in page_ids}):
        pdf = pdf_dir / f"{doc_id}.pdf"
        if not pdf.exists():
            continue
        with pymupdf.open(pdf) as doc:
            # `enumerate(doc)` çalışırdı ama pymupdf stub'ı Document'i Iterable
            # olarak bildirmiyor; load_page indeksle aynı sırayı verir.
            for i in range(doc.page_count):
                # get_text() dönüş tipi `option`a bağlı (str/list/dict);
                # varsayılan "text" seçeneği her zaman str döner.
                texts[f"{doc_id}:{i + 1}"] = cast(str, doc.load_page(i).get_text())
    return pd.DataFrame(
        {"page_id": list(page_ids), "text": [texts.get(pid, "") for pid in page_ids]}
    )
