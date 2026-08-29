"""f16 oracle yardımcıları — `FloatIndex`'in ESKİ evi.

T14 (ruling D4): `FloatIndex` `belge_gozu.index.float_store`'a taşındı,
çünkü üretim kodu (`index/quantize.py`) onu buradan import ediyordu —
index -> bench yönünde bir katman inversiyonu. Bu modül artık yalnız
geriye dönük re-export + oracle'a özgü küçük yardımcılardır; mevcut
`from belge_gozu.bench.oracle import FloatIndex, native_float_scores`
importları AYNEN çalışmaya devam eder.
"""

import numpy as np

from belge_gozu.index.chunking import CHUNK_TOKENS, chunk_bounds
from belge_gozu.index.float_store import FloatIndex

__all__ = [
    "CHUNK_TOKENS",  # geriye dönük: eskiden bu modülün global'iydi (bkz. index/chunking.py)
    "FloatIndex",  # geriye dönük: eskiden burada tanımlıydı (bkz. index/float_store.py)
    "chunk_bounds",
    "native_float_scores",
    "rank_of",
]


def native_float_scores(findex: FloatIndex, q_emb: np.ndarray) -> np.ndarray:
    """(n_pages,) — `FloatIndex.score_all`'a ince delegasyon (aynı matematik).

    Skorlar per-query-token ortalamadır (~[-1,1]): PackedIndex/Int8Index ile
    AYNI ölçek. Fonksiyon biçimi oracle koşumlarındaki çağrı yerleri için
    korunuyor."""
    return findex.score_all(q_emb)


def rank_of(scores: np.ndarray, page_ids: list[str], target: str) -> int:
    """1-tabanlı sıra: eşitlikte iyimser değil (stable argsort pozisyonu)."""
    order = np.argsort(-scores, kind="stable")
    idx = page_ids.index(target)
    return int(np.flatnonzero(order == idx)[0]) + 1
