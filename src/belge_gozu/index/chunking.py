"""Sayfa-hizalı chunk sınırları — indeks/getirim katmanının ORTAK sabiti.

Bu modül üç kopyayı birleştirir (final review IMPORTANT-5): `bench/oracle.py`
(`_chunk_bounds` + `CHUNK_TOKENS`), `index/quantize.py` (oracle'dan import
ediyordu — üretim paketini bench paketine bağımlı kılıyordu) ve
`retrieval/core.py` (kendi üçüncü kopyası + kendi sabiti). Artık üçü de
buradan okur; bench -> index yönü serbest, index/retrieval -> bench yönü YOK.
"""

import numpy as np

CHUNK_TOKENS = 500_000


def chunk_bounds(offsets: np.ndarray, chunk_tokens: int | None = None) -> list[int]:
    """Sayfa sınırlarına hizalı chunk başlangıç indeksleri (offsets üzerinden).

    `chunk_tokens=None` -> bu modülün global `CHUNK_TOKENS`'ı ÇAĞRI ANINDA
    okunur (import zamanında bağlanmaz), böylece testler modül global'ini
    monkeypatch'leyebilir. `None` kontrolü bilinçli olarak `or` DEĞİL: açıkça
    verilen `0` sessizce varsayılana dönüşmemeli.
    """
    if chunk_tokens is None:
        chunk_tokens = CHUNK_TOKENS
    bounds = [0]
    for i in range(1, len(offsets)):
        last = bounds[-1]
        if offsets[i] - offsets[last] >= chunk_tokens:
            bounds.append(i)
    if bounds[-1] != len(offsets) - 1:
        bounds.append(len(offsets) - 1)
    return bounds
