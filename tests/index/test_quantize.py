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
    # review R1 IMPORTANT-1: rtol=0.05/atol=0.5 tek başına per-token ölçek
    # sözleşmesini kanıtlamıyor (global ölçek, kesme, hatta 3-bit bir kuantizör
    # de bu toleransı geçer). İki ek sözleşme testi:
    # (a) her satır KENDİ max|x|'ine göre 127'ye satüre olmalı (per-token scale)
    assert (np.abs(i8.codes).max(axis=1) == 127).all()
    # (b) round-to-nearest dequant hatası satır başına scale/2'yi aşamaz
    src = np.asarray(fi.embs, dtype=np.float32)
    deq = i8.codes.astype(np.float32) * i8.scales[:, None]
    assert np.all(np.abs(deq - src) <= i8.scales[:, None] / 2 + 1e-4)

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
    # review R1 MINOR-7: yalnız codes'u doğrulamak scales/offsets/page_ids'teki
    # sessiz bir regresyonu (her int8 skoru bozan türden) kaçırırdı.
    np.testing.assert_array_equal(np.asarray(i82.scales), np.asarray(i8.scales))
    np.testing.assert_array_equal(np.asarray(i82.offsets), np.asarray(i8.offsets))
    assert i82.page_ids == i8.page_ids


def test_int8_derive_chunked_matches_single_shot():
    """review R1 IMPORTANT-2: `derive` chunk'lı yeniden yazıldı (bellek tepe
    noktası tüm korpusu float32'ye açan ~4 kopya yerine chunk boyutuna
    indirildi). Her satırın kuantizasyonu bağımsız olduğu için (score_all'daki
    reduceat'in aksine satırlar arası indirgeme yok) chunk sınırı sonucu HİÇ
    etkilememeli -- bit-birebir eşitlik bekleniyor."""
    fi, _ = make_findex(n_pages=6, tokens=8)
    single = Int8Index.derive(fi)
    chunked = Int8Index.derive(fi, chunk_tokens=10)  # sayfa başına 8 token -> çoklu chunk
    np.testing.assert_array_equal(chunked.codes, single.codes)
    np.testing.assert_array_equal(chunked.scales, single.scales)


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
