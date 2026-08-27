from pathlib import Path

import numpy as np

from belge_gozu.bench.oracle import FloatIndex, native_float_scores, rank_of


def make_findex(n_pages=5, tokens=4, seed=0):
    rng = np.random.default_rng(seed)
    embs = [rng.standard_normal((tokens, 128)).astype(np.float32) for _ in range(n_pages)]
    return FloatIndex.build([f"d{i}:1" for i in range(n_pages)], embs), embs


def test_roundtrip(tmp_path: Path):
    fi, _ = make_findex()
    fi.save(tmp_path)
    fi2 = FloatIndex.load(tmp_path, mmap=False)
    assert fi2.page_ids == fi.page_ids
    np.testing.assert_allclose(
        np.asarray(fi2.embs, dtype=np.float32), np.asarray(fi.embs, dtype=np.float32), atol=1e-3
    )


def test_self_match_top1():
    fi, embs = make_findex()
    scores = native_float_scores(fi, embs[3])
    assert scores.argmax() == 3
    assert rank_of(scores, fi.page_ids, "d3:1") == 1


def test_scores_are_true_maxsim():
    fi, embs = make_findex(n_pages=2)
    q = embs[0]
    doc = np.asarray(fi.page_tokens(1), dtype=np.float32)
    expected = (q @ doc.T).max(axis=1).sum() / q.shape[0]
    assert abs(native_float_scores(fi, q)[1] - expected) < 1e-2  # f16 saklama toleransı
