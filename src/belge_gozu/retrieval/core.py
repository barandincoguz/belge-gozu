from typing import ClassVar

import numpy as np
import pandas as pd

from belge_gozu.index.chunking import CHUNK_TOKENS, chunk_bounds
from belge_gozu.index.encode import Encoder
from belge_gozu.index.store import PackedIndex, binarize_pack
from belge_gozu.retrieval.types import PageHit
from belge_gozu.telemetry.collect import stage


def _as_u64(packed: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(packed).view(np.uint64)  # (n,16) uint8 -> (n,2) uint64


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n,16) uint8, b: (m,16) uint8 -> (n,m) int32 Hamming mesafeleri."""
    xa, xb = _as_u64(a), _as_u64(b)
    return np.bitwise_count(xa[:, None, :] ^ xb[None, :, :]).sum(axis=2).astype(np.int32)


def binary_maxsim(q_packed: np.ndarray, d_packed: np.ndarray) -> float:
    sim = 128 - 2 * hamming_matrix(q_packed, d_packed)  # (n_q, n_d)
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
        with stage("query_encode"):
            q_emb = self.encoder.encode_query(query)
        hits = self.search_embedding(q_emb, k, candidates)
        n_q = max(1, q_emb.shape[0])
        out: list[PageHit] = []
        for i, score in hits:
            row = self.meta.loc[self.index.page_ids[i]]
            out.append(
                PageHit(
                    page_id=row["page_id"],
                    score=score / n_q,
                    doc_name=row["doc_name"],
                    page_no=int(row["page_no"]),
                    image_path=row["image_path"],
                    source_url=row["source_url"],
                )
            )
        return out


class ExhaustiveBinaryRetriever:
    """Tüm korpus üstünde kesin binary MaxSim. 4222 sayfada ~1.2 s (M4 Pro).

    Mean-sign Stage-1 kaldırıldı: ölçülen top-200 kesişimi %11.5-19 ve rank-2
    sonucu 1768'e atma karşı-örneği (spec §1.1). TwoStageRetriever yalnız
    ablasyon için durur (config: retrieval_pipeline="two-stage")."""

    # Ortak sabit (belge_gozu.index.chunking) — eskiden bu sınıfın kendi üçüncü
    # kopyasıydı; test override'ı için instance üstünde değiştirilebilir
    # (bkz. index/quantize.py'deki aynı desen).
    CHUNK_TOKENS: ClassVar[int] = CHUNK_TOKENS

    def __init__(self, index: PackedIndex, meta: pd.DataFrame, encoder: Encoder | None):
        self.index = index
        self.encoder = encoder
        self.meta = meta.set_index("page_id", drop=False)
        self.tokens = np.ascontiguousarray(np.asarray(index.tokens))
        self.offsets = np.asarray(index.offsets)

    def score_all(self, q_emb: np.ndarray) -> np.ndarray:
        q_packed = binarize_pack(q_emb)
        qa = _as_u64(q_packed)
        ta = _as_u64(self.tokens)
        n_pages = len(self.index.page_ids)
        out = np.empty(n_pages, dtype=np.float64)
        bounds = chunk_bounds(self.offsets, self.CHUNK_TOKENS)
        for b0, b1 in zip(bounds[:-1], bounds[1:], strict=True):
            t0, t1 = int(self.offsets[b0]), int(self.offsets[b1])
            ham = np.bitwise_count(qa[:, None, :] ^ ta[None, t0:t1, :]).sum(axis=2, dtype=np.int32)
            sim = 128 - 2 * ham
            starts = (self.offsets[b0:b1] - t0).astype(np.int64)
            # offsets kesin artan (PackedIndex.build sıfır-token sayfayı reddeder) ->
            # reduceat boş segment göremez.
            out[b0:b1] = np.maximum.reduceat(sim, starts, axis=1).sum(axis=0)
        return out / max(1, q_emb.shape[0])

    def search_embedding(self, q_emb: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Per-query-token NORMALIZE edilmiş skorlar döner (score_all zaten böler)."""
        scores = self.score_all(q_emb)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order]

    def search(self, query: str, k: int = 5) -> list[PageHit]:
        if self.encoder is None:
            raise RuntimeError("encoder yapılandırılmamış")
        with stage("query_encode"):
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
