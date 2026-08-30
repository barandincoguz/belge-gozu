from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from belge_gozu.index.chunking import CHUNK_TOKENS, EMBED_DIM
from belge_gozu.index.encode import ENCODE_LIMIT, Encoder
from belge_gozu.index.store import PackedIndex, as_u64, binarize_pack
from belge_gozu.retrieval.types import PageHit
from belge_gozu.telemetry.collect import stage

if TYPE_CHECKING:
    # Yalnız anotasyon için (review M9): runtime'da import edilirse
    # `retrieval` -> `index.loader` -> `index.quantize` -> `index.float_store`
    # zinciri her import'ta çekilir. Sözleşme iddiası tip denetiminde aynen
    # geçerli, çalışma zamanı kenarı yok.
    from belge_gozu.index.loader import ScorableIndex


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n,16) uint8, b: (m,16) uint8 -> (n,m) int32 Hamming mesafeleri."""
    xa, xb = as_u64(a), as_u64(b)
    return np.bitwise_count(xa[:, None, :] ^ xb[None, :, :]).sum(axis=2).astype(np.int32)


def binary_maxsim(q_packed: np.ndarray, d_packed: np.ndarray) -> float:
    """HAM MaxSim toplamı: jeton başına [-EMBED_DIM, EMBED_DIM], normalize DEĞİL.

    Normalize [-1,1] skor için `n_q * EMBED_DIM`'e bölünür (bkz.
    `TwoStageRetriever.search`, `PackedIndex.score_all`)."""
    sim = EMBED_DIM - 2 * hamming_matrix(q_packed, d_packed)  # (n_q, n_d)
    return float(sim.max(axis=1).sum())


class TwoStageRetriever:
    def __init__(self, index: PackedIndex, meta: pd.DataFrame, encoder: Encoder | None):
        self.index = index
        self.encoder = encoder
        self.meta = meta.set_index("page_id", drop=False)

    def search_embedding(
        self, q_emb: np.ndarray, k: int, candidates: int
    ) -> list[tuple[int, float]]:
        """RAW MaxSim toplamları döner (normalize edilmemiş); normalize search()'te yapılır."""
        q_packed = binarize_pack(q_emb)
        q_vec = binarize_pack(q_emb.mean(axis=0, keepdims=True))
        # Aşama 1: sayfa vektörüyle Hamming eleme
        with stage("stage1_hamming"):
            dists = hamming_matrix(q_vec, self.index.page_vecs)[0]
            n_cand = min(candidates, len(dists))
            cand_ids = np.argpartition(dists, n_cand - 1)[:n_cand]
        # Aşama 2: adaylarda kesin binary MaxSim
        with stage("stage2_maxsim"):
            scored = [
                (int(i), binary_maxsim(q_packed, self.index.page_tokens(int(i)))) for i in cand_ids
            ]
            scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def search(self, query: str, k: int = 5, candidates: int = 200) -> list[PageHit]:
        if self.encoder is None:
            raise RuntimeError("encoder yapılandırılmamış")
        # savunmacı sınır, ölçüm: 40@c=8 sağlıklı (bkz. index/encode.py)
        with stage("query_encode"), ENCODE_LIMIT:
            q_emb = self.encoder.encode_query(query)
        hits = self.search_embedding(q_emb, k, candidates)
        n_q = max(1, q_emb.shape[0])
        out: list[PageHit] = []
        for i, score in hits:
            row = self.meta.loc[self.index.page_ids[i]]
            out.append(
                PageHit(
                    page_id=row["page_id"],
                    # T14: ham MaxSim toplamı `n_q * EMBED_DIM`'e bölünür —
                    # üretim (exhaustive) yolunun ürettiği AYNI normalize
                    # [-1,1] ölçek, böylece iki kol karşılaştırılabilir olur.
                    # DİKKAT: ortak ÖLÇEK, ortak EŞİK demek değildir — bu kol
                    # 1-bit dağılımında skorlar (bkz. config.py eşik yorumu).
                    score=score / (n_q * EMBED_DIM),
                    doc_name=row["doc_name"],
                    page_no=int(row["page_no"]),
                    image_path=row["image_path"],
                    source_url=row["source_url"],
                )
            )
        return out


class ExhaustiveRetriever:
    """Tüm korpus üstünde kesin MaxSim — indeks temsilinden BAĞIMSIZ.

    Skorlar normalize [-1,1]: sorgu jetonu başına ortalama MaxSim. Hangi
    indeks yüklüyse (packed/int8/float) kendi `score_all`'unu uygular ve
    ÜÇÜ DE aynı bandı döner (binary kol T14'te 128'e bölünerek bu banda
    taşındı).

    Ortak bant = KARŞILAŞTIRILABİLİRLİK, kalibrasyon taşınabilirliği DEĞİL:
    temsiller aynı ölçeğe girer ama aynı DAĞILIMA girmez (ölçüm: canary
    top-1 medyanı int8 0.6250 vs 1-bit 0.4953). Bu yüzden tek bir
    `min_score_threshold` yalnız üzerinde taşındığı temsilde geçerlidir —
    ayrıntı config.py'deki eşik yorumunda.

    Mean-sign Stage-1 kaldırıldı: ölçülen top-200 kesişimi %11.5-19 ve rank-2
    sonucu 1768'e atma karşı-örneği (spec §1.1). TwoStageRetriever yalnız
    ablasyon için durur (config: retrieval_pipeline="two-stage") ve YALNIZ
    PackedIndex ile çalışır (bkz. app/main.py ve cli.py korkulukları).

    T14: eski ad `ExhaustiveBinaryRetriever` alias olarak korunuyor (harness,
    testler ve betikler o adı import ediyor)."""

    # Ortak sabit (belge_gozu.index.chunking) — eskiden bu sınıfın kendi üçüncü
    # kopyasıydı; test override'ı için instance üstünde değiştirilebilir
    # (bkz. index/quantize.py'deki aynı desen). İndekse her çağrıda geçilir.
    CHUNK_TOKENS: ClassVar[int] = CHUNK_TOKENS

    def __init__(self, index: ScorableIndex, meta: pd.DataFrame, encoder: Encoder | None):
        self.index = index
        self.encoder = encoder
        self.meta = meta.set_index("page_id", drop=False)

    def score_all(self, q_emb: np.ndarray) -> np.ndarray:
        """(n_pages,) — normalize [-1,1] skorlar; çekirdek indeksin kendisinde.

        T14'te binary çekirdek `PackedIndex.score_all`'a taşındı: getirim
        katmanı artık temsile değil yalnız sözleşmeye (`ScorableIndex`) bakar."""
        return self.index.score_all(q_emb, chunk_tokens=self.CHUNK_TOKENS)

    def search_embedding(self, q_emb: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Per-query-token NORMALIZE edilmiş skorlar döner (score_all zaten böler)."""
        scores = self.score_all(q_emb)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order]

    def search(self, query: str, k: int = 5) -> list[PageHit]:
        if self.encoder is None:
            raise RuntimeError("encoder yapılandırılmamış")
        # savunmacı sınır, ölçüm: 40@c=8 sağlıklı (bkz. index/encode.py)
        with stage("query_encode"), ENCODE_LIMIT:
            q_emb = self.encoder.encode_query(query)
        with stage("exhaustive_maxsim"):
            hits = self.search_embedding(q_emb, k)
        out: list[PageHit] = []
        for i, score in hits:
            row = self.meta.loc[self.index.page_ids[i]]
            out.append(
                PageHit(
                    page_id=row["page_id"],
                    score=score,
                    doc_name=row["doc_name"],
                    page_no=int(row["page_no"]),
                    image_path=row["image_path"],
                    source_url=row["source_url"],
                )
            )
        return out


# Geriye dönük ad: sınıf T14'te temsil-bağımsız hale gelince "Binary" adı
# yanlış oldu, ama harness/testler/betikler bu adı import ediyor.
ExhaustiveBinaryRetriever = ExhaustiveRetriever
