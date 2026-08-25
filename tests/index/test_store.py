from pathlib import Path

import numpy as np

from belge_gozu.index.store import PackedIndex, binarize_pack


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
