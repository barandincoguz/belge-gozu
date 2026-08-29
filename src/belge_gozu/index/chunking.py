"""İndeks katmanının paylaşılan sabitleri + sayfa-hizalı chunk sınırları.

Bu modül üç kopyayı birleştirir (final review IMPORTANT-5): `bench/oracle.py`
(`_chunk_bounds` + `CHUNK_TOKENS`), `index/quantize.py` (oracle'dan import
ediyordu — üretim paketini bench paketine bağımlı kılıyordu) ve
`retrieval/core.py` (kendi üçüncü kopyası + kendi sabiti). Artık üçü de
buradan okur; bench -> index yönü serbest, index/retrieval -> bench yönü YOK.

T14'te aynı gerekçeyle iki sayısal sabit daha buraya alındı (`EMBED_DIM`,
`INT8_MAX`): dört ayrı dosyada bağımsız literal olarak duruyorlardı ve
binary skorun normalizasyon böleni de tam olarak `EMBED_DIM`'dir — bağlantı
literal kalırsa biri değişip diğerleri kalabilir.
"""

import numpy as np

CHUNK_TOKENS = 500_000

# ColSmol-500M token embedding boyutu. Aynı zamanda binary MaxSim'in
# normalizasyon böleni: jeton başına ham skor `EMBED_DIM - 2*hamming`
# [-EMBED_DIM, EMBED_DIM] bandındadır, EMBED_DIM'e bölününce dot-product
# skorlarıyla aynı normalize [-1,1] bandına oturur (bkz. store.score_all).
EMBED_DIM = 128
# Simetrik int8 kuantizasyonun doyum noktası: scale = max|x| / INT8_MAX,
# kodlar [-INT8_MAX, INT8_MAX] aralığına kırpılır (bkz. quantize.Int8Index).
INT8_MAX = 127


def chunk_bounds(offsets: np.ndarray, chunk_tokens: int | None = None) -> list[int]:
    """Sayfa sınırlarına hizalı chunk başlangıç indeksleri (offsets üzerinden).

    `chunk_tokens=None` -> bu modülün global `CHUNK_TOKENS`'ı ÇAĞRI ANINDA
    okunur (import zamanında bağlanmaz). `None` kontrolü bilinçli olarak `or`
    DEĞİL: açıkça verilen `0` sessizce varsayılana dönüşmemeli.

    DİKKAT (review M8): bu "çağrı anında oku" davranışı yalnız buraya `None`
    ULAŞTIĞINDA geçerlidir. `PackedIndex`/`Int8Index` `CHUNK_TOKENS`'ı
    import anında bağlanan bir `ClassVar`dan çözer ve buraya somut bir sayı
    geçirir; yani modül global'ini monkeypatch'lemek onları ETKİLEMEZ —
    `FloatIndex` (None'ı olduğu gibi ileten) ve doğrudan çağrılar etkilenir.
    Sınıf tarafında test override'ı instance üstünden yapılır
    (`idx.CHUNK_TOKENS = ...`), ki testler zaten bunu kullanıyor.
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
