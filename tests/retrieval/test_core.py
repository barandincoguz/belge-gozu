import numpy as np
import pandas as pd

from belge_gozu.index.store import PackedIndex, binarize_pack
from belge_gozu.retrieval.core import TwoStageRetriever, binary_maxsim


def naive_maxsim(q_bits: np.ndarray, d_bits: np.ndarray) -> float:
    # saf referans: bit dizileri (n,128) 0/1 uint8
    total = 0
    for qrow in q_bits:
        best = -129
        for drow in d_bits:
            ham = int(np.sum(qrow != drow))
            best = max(best, 128 - 2 * ham)
        total += best
    return float(total)


def unpack(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(packed, axis=1)


def test_binary_maxsim_matches_naive():
    rng = np.random.default_rng(3)
    q = binarize_pack(rng.standard_normal((5, 128)).astype(np.float32))
    d = binarize_pack(rng.standard_normal((9, 128)).astype(np.float32))
    assert binary_maxsim(q, d) == naive_maxsim(unpack(q), unpack(d))


def build_fixture(n_pages: int = 30, seed: int = 11):
    rng = np.random.default_rng(seed)
    embs = [rng.standard_normal((8, 128)).astype(np.float32) for _ in range(n_pages)]
    ids = [f"d{i}:1" for i in range(n_pages)]
    idx = PackedIndex.build(ids, embs)
    meta = pd.DataFrame(
        {
            "page_id": ids,
            "doc_id": [f"d{i}" for i in range(n_pages)],
            "doc_name": [f"Belge {i}" for i in range(n_pages)],
            "doc_type": ["kanun"] * n_pages,
            "source_url": ["https://example.org"] * n_pages,
            "page_no": [1] * n_pages,
            "image_path": [f"images/d{i}/0001.webp" for i in range(n_pages)],
        }
    )
    return idx, meta, embs


def test_planted_needle_found():
    idx, meta, embs = build_fixture()
    retriever = TwoStageRetriever(idx, meta, encoder=None)
    hits = retriever.search_embedding(embs[17], k=5, candidates=10)
    assert hits[0][0] == 17  # sorgu = sayfa 17'nin embedding'i → ilk sırada 17
    assert hits[0][1] >= hits[1][1]


def test_stage1_prefilter_respected():
    idx, meta, embs = build_fixture()
    retriever = TwoStageRetriever(idx, meta, encoder=None)
    all_hits = retriever.search_embedding(embs[17], k=30, candidates=30)
    assert len(all_hits) == 30
    few = retriever.search_embedding(embs[17], k=5, candidates=3)
    assert len(few) == 3  # aday sayısı k'den küçükse sonuç aday sayısıyla sınırlı


def test_search_scores_normalized_per_query_token():
    idx, meta, embs = build_fixture()

    class OneTokenEncoder:
        def encode_pages(self, images):
            raise NotImplementedError

        def encode_query(self, text):
            return embs[17][:1]  # tek token → normalize skor = ham skor

    r = TwoStageRetriever(idx, meta, OneTokenEncoder())
    hits = r.search("x", k=1, candidates=30)
    raw = r.search_embedding(embs[17][:1], k=1, candidates=30)
    # T14: search() ham toplamı n_q * 128'e böler — exhaustive üretim
    # yolunun ürettiği AYNI normalize [-1,1] ölçek.
    assert hits[0].score == raw[0][1] / 1 / 128
    assert -1.0 <= hits[0].score <= 1.0


def test_search_normalizes_by_multi_token_query():
    idx, meta, embs = build_fixture()

    class MultiTokenEncoder:
        def encode_pages(self, images):
            raise NotImplementedError

        def encode_query(self, text):
            return embs[17]  # 8 token → normalize skor = ham skor / 8

    r = TwoStageRetriever(idx, meta, MultiTokenEncoder())
    hits = r.search("x", k=1, candidates=30)
    raw = r.search_embedding(embs[17], k=1, candidates=30)
    assert hits[0].score == raw[0][1] / embs[17].shape[0] / 128
    assert hits[0].score != raw[0][1]  # n_q>1: normalizasyon gerçekten iz bırakıyor
    assert -1.0 <= hits[0].score <= 1.0
