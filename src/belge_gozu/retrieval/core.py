import numpy as np
import pandas as pd

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
