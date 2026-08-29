from pathlib import Path

import numpy as np
import pytest

from belge_gozu.index.store import PackedIndex, as_u64, binarize_pack


def test_binarize_pack_bits():
    v = np.zeros((1, 128), dtype=np.float32)
    v[0, 0] = 1.0  # ilk bit set → ilk byte 0b10000000
    v[0, 127] = 1.0  # son bit set → son byte 0b00000001
    packed = binarize_pack(v)
    assert packed.shape == (1, 16) and packed.dtype == np.uint8
    assert packed[0, 0] == 0x80 and packed[0, 15] == 0x01


def rand_embs(rng, n_pages: int) -> list[np.ndarray]:
    return [
        rng.standard_normal((rng.integers(5, 12), 128)).astype(np.float32) for _ in range(n_pages)
    ]


def test_roundtrip(tmp_path: Path):
    rng = np.random.default_rng(7)
    embs = rand_embs(rng, 4)
    idx = PackedIndex.build([f"p{i}" for i in range(4)], embs)
    assert idx.offsets[-1] == sum(e.shape[0] for e in embs)
    idx.save(tmp_path)
    loaded = PackedIndex.load(tmp_path)
    assert loaded.page_ids == idx.page_ids
    np.testing.assert_array_equal(loaded.page_tokens(2), idx.page_tokens(2))
    np.testing.assert_array_equal(loaded.page_vecs, idx.page_vecs)
    assert isinstance(loaded.tokens, np.memmap)


def test_build_rejects_length_mismatch():
    rng = np.random.default_rng(1)
    embs = [rng.standard_normal((4, 128)).astype(np.float32)] * 2
    with pytest.raises(ValueError, match="eşleşmiyor"):
        PackedIndex.build(["p0"], embs)


def test_build_rejects_empty():
    with pytest.raises(ValueError, match="boş korpus"):
        PackedIndex.build([], [])


def test_build_rejects_zero_token_page():
    rng = np.random.default_rng(1)
    embs = [
        rng.standard_normal((4, 128)).astype(np.float32),
        np.zeros((0, 128), np.float32),
    ]
    with pytest.raises(ValueError, match="p1"):
        PackedIndex.build(["p0", "p1"], embs)


def test_build_rejects_zero_rows():
    embs = [
        np.vstack(
            [
                np.ones((2, 128), dtype=np.float32),
                np.zeros((1, 128), dtype=np.float32),
            ]
        )
    ]
    with pytest.raises(ValueError, match="padding satırı sızmış: p:1"):
        PackedIndex.build(["p:1"], embs)


def test_manifest_roundtrip(tmp_path: Path):
    from tests.index.test_manifest import make_manifest

    embs = [np.random.default_rng(0).standard_normal((4, 128)).astype(np.float32)]
    idx = PackedIndex.build(["p:1"], embs, manifest=make_manifest(n_pages=1, n_tokens=4))
    idx.save(tmp_path)
    loaded = PackedIndex.load(tmp_path, mmap=False)
    assert loaded.manifest is not None and loaded.manifest.n_pages == 1


def test_legacy_index_loads_without_manifest(tmp_path: Path):
    embs = [np.random.default_rng(0).standard_normal((4, 128)).astype(np.float32)]
    PackedIndex.build(["p:1"], embs).save(tmp_path)
    assert PackedIndex.load(tmp_path, mmap=False).manifest is None


# --- score_all: T14'te retrieval/core.py'den buraya taşınan çekirdek --------


def _score_fixture(n_pages=6, tokens=8, seed=5):
    rng = np.random.default_rng(seed)
    embs = [rng.standard_normal((tokens, 128)).astype(np.float32) for _ in range(n_pages)]
    return PackedIndex.build([f"d{i}:1" for i in range(n_pages)], embs), embs


def test_score_all_matches_per_page_reference():
    """Referans: sayfa başına `binary_maxsim / n_q / 128` (normalize [-1,1]).

    128'e bölme T14'ün TEK matematik değişikliğidir: binary skoru
    int8/float16'nın dot-product bandına taşır, böylece tek bir eşik
    (Settings.min_score_threshold) üç temsilde de aynı anlama gelir."""
    from belge_gozu.retrieval.core import binary_maxsim

    idx, embs = _score_fixture()
    q = embs[2]
    scores = idx.score_all(q)
    qp = binarize_pack(q)
    for i in range(len(idx.page_ids)):
        expected = binary_maxsim(qp, np.asarray(idx.page_tokens(i))) / q.shape[0] / 128
        assert scores[i] == expected
    assert scores.argmax() == 2  # kendi sayfası top-1
    assert (scores >= -1.0).all() and (scores <= 1.0).all()


def test_score_all_respects_chunk_tokens():
    """chunk_tokens SONUCU DEĞİŞTİRMEZ — yalnız bellek tepe noktasını.

    Hem açık argüman hem instance override'ı (retrieval/core.py'nin
    CHUNK_TOKENS'ı buradan geçirir) aynı skorları vermeli."""
    idx, embs = _score_fixture()
    q = embs[4]
    base = idx.score_all(q)
    np.testing.assert_array_equal(idx.score_all(q, chunk_tokens=10), base)  # ~1-2 sayfa/chunk
    np.testing.assert_array_equal(idx.score_all(q, chunk_tokens=1), base)  # sayfa başına 1 chunk
    idx.CHUNK_TOKENS = 10  # instance override (test deseni)
    np.testing.assert_array_equal(idx.score_all(q), base)


def test_as_u64_rejects_wrong_width():
    """Sessiz sözleşme -> açık hata: (n,16) uint8 dışındaki her şey reddedilir."""
    with pytest.raises(ValueError, match=r"\(n,16\)"):
        as_u64(np.zeros((3, 8), dtype=np.uint8))
