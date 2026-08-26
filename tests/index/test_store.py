from pathlib import Path

import numpy as np
import pytest

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
