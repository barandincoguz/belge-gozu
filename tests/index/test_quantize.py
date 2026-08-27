import numpy as np

from belge_gozu.bench.oracle import FloatIndex, native_float_scores
from belge_gozu.index.quantize import Int8Index, derive_packed
from belge_gozu.index.store import binarize_pack
from tests.index.test_manifest import make_manifest


def make_findex(n_pages=6, tokens=5, seed=2):
    rng = np.random.default_rng(seed)
    embs = [rng.standard_normal((tokens, 128)).astype(np.float32) for _ in range(n_pages)]
    return FloatIndex.build([f"d{i}:1" for i in range(n_pages)], embs), embs


def test_derive_packed_matches_direct_binarize():
    fi, embs = make_findex()
    packed = derive_packed(fi)
    np.testing.assert_array_equal(
        np.asarray(packed.page_tokens(2)),
        binarize_pack(np.asarray(fi.page_tokens(2), dtype=np.float32)),
    )


def test_int8_scores_close_to_float():
    fi, embs = make_findex()
    i8 = Int8Index.derive(fi)
    f = native_float_scores(fi, embs[1])
    q = i8.score_all(embs[1])
    assert np.argmax(q) == np.argmax(f) == 1
    np.testing.assert_allclose(q, f, rtol=0.05, atol=0.5)


def test_int8_roundtrip(tmp_path):
    fi, _ = make_findex()
    i8 = Int8Index.derive(fi)
    i8.save(tmp_path)
    i82 = Int8Index.load(tmp_path, mmap=False)
    np.testing.assert_array_equal(np.asarray(i82.codes), np.asarray(i8.codes))


def test_int8_score_all_multichunk_matches_single_chunk():
    """CHUNK_TOKENS küçültülüp örnek çok chunk'a bölündüğünde tek-chunk koşumuyla
    aynı skorları üretmeli (sayfa-hizalı chunklama T5/ExhaustiveBinaryRetriever
    desenini mirror'lıyor -- bkz. retrieval/core.py test override deseni)."""
    fi, embs = make_findex(n_pages=6, tokens=8)
    i8_single = Int8Index.derive(fi)
    i8_multi = Int8Index.derive(fi)
    i8_multi.CHUNK_TOKENS = 10  # sayfa başına 8 token -> her chunk ~1-2 sayfa

    q = embs[3]
    single = i8_single.score_all(q)
    multi = i8_multi.score_all(q)
    # float32 matmul'ün chunk sınırına göre farklı toplama sırası ~1e-6 düzeyinde
    # gürültü üretir (native_float_scores'ta da aynı özellik); değer eşitliği
    # değil sayısal denklik bekleniyor.
    np.testing.assert_allclose(multi, single, rtol=1e-5, atol=1e-5)


def test_derive_packed_carries_manifest_with_sign_1bit_quantization():
    manifest = make_manifest(quantization="float16", n_pages=6, n_tokens=30)
    fi, _ = make_findex()
    fi.manifest = manifest
    packed = derive_packed(fi)
    assert packed.manifest is not None
    assert packed.manifest.quantization == "sign-1bit"
    # geri kalan alanlar taşınmış olmalı (yalnız quantization değişti)
    assert packed.manifest.model_dump(exclude={"quantization"}) == manifest.model_dump(
        exclude={"quantization"}
    )
