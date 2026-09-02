"""ColBERT geç-etkileşim kodlama sözleşmesi — saf mantık testleri (model AĞIRLIĞI YOK).

Sözleşme `moganai/Mogan-ColBERT-TR`'nin kendi `config_sentence_transformers.json`
dosyasından doğrulandı (2026-09-02): query_prefix "[unused0]", document_prefix
"[unused1]", query_length 32, document_length 512, do_query_expansion true,
attend_to_expansion_tokens false, 32 noktalama skiplist'i.

NEDEN bu kadar test: bu projede kodlama biçimini yanlış almak ölçülmüş bir
bedele sahip — görsel kodlayıcıda sorgu formatı hatası R@5'i 0,233 yerine 0,093
yapıyordu. pylate'in `encode()` fonksiyonunun `is_query` VARSAYILANI True, yani
belgeleri bayrak vermeden kodlamak sessizce her chunk'ı 32 token'a keser. O hata
sınıfı burada testle kapatılır.
"""

import numpy as np
import pytest

from belge_gozu.index.colbert_encode import (
    ColbertConfig,
    build_document_ids,
    build_query_ids,
    document_keep_mask,
    load_colbert_config,
    maxsim,
)

CFG = ColbertConfig(
    query_prefix="[unused0]",
    document_prefix="[unused1]",
    query_length=8,          # testte kısa; üretimde 32
    document_length=16,      # testte kısa; üretimde 512
    do_query_expansion=True,
    attend_to_expansion_tokens=False,
    skiplist_ids=frozenset({91, 92}),
)

CLS, SEP, MASK, PAD = 2, 3, 4, 0
QMARK, DMARK = 6, 7


# --------------------------------------------------------------------------
# config okuma — sözleşme tahmin EDİLMEZ, modelden okunur
# --------------------------------------------------------------------------


def test_load_colbert_config_reads_the_models_own_contract():
    raw = {
        "query_prefix": "[unused0]",
        "document_prefix": "[unused1]",
        "query_length": 32,
        "document_length": 512,
        "do_query_expansion": True,
        "attend_to_expansion_tokens": False,
        "skiplist_words": ["!", ",", "."],
    }
    cfg = load_colbert_config(raw, skiplist_to_ids={"!": 91, ",": 92, ".": 93})
    assert cfg.query_length == 32
    assert cfg.document_length == 512
    assert cfg.document_prefix == "[unused1]"
    assert cfg.skiplist_ids == frozenset({91, 92, 93})


def test_load_colbert_config_rejects_unknown_marker():
    """İşaret token'ı sözlükte yoksa UNK'a düşer ve getirim sessizce çöker."""
    raw = {"query_prefix": "[Q]", "document_prefix": "[D]", "query_length": 32,
           "document_length": 512, "do_query_expansion": True,
           "attend_to_expansion_tokens": False, "skiplist_words": []}
    with pytest.raises(ValueError, match="skiplist"):
        load_colbert_config(raw, skiplist_to_ids=None)


# --------------------------------------------------------------------------
# sorgu tarafı
# --------------------------------------------------------------------------


def test_query_marker_goes_at_index_one_not_prepended_as_text():
    ids, _ = build_query_ids([CLS, 10, 11, SEP], QMARK, MASK, CFG)
    assert ids[0] == CLS
    assert ids[1] == QMARK, "işaret CLS'ten SONRA, index 1'e girer"
    assert ids[2:4] == [10, 11]


def test_query_is_padded_to_query_length_with_mask_not_pad():
    ids, _ = build_query_ids([CLS, 10, SEP], QMARK, MASK, CFG)
    assert len(ids) == CFG.query_length
    assert ids[-1] == MASK, "sorgu genişletmesi [MASK] ile yapılır, [PAD] ile değil"
    assert PAD not in ids


def test_query_attention_is_zero_on_expansion_tokens():
    """attend_to_expansion_tokens=False: dikkat 0, ama vektörler KORUNUR."""
    ids, attn = build_query_ids([CLS, 10, SEP], QMARK, MASK, CFG)
    assert len(attn) == len(ids) == CFG.query_length
    # [CLS, 10, SEP] 3 token; index 1'e giren işaretle 4 gerçek token
    assert attn[:4] == [1, 1, 1, 1]
    assert set(attn[4:]) == {0}, "genişletme token'larında dikkat 0"


def test_query_truncates_to_length_minus_one_because_marker_costs_a_slot():
    long_ids = [CLS] + list(range(20, 40)) + [SEP]
    ids, _ = build_query_ids(long_ids, QMARK, MASK, CFG)
    assert len(ids) == CFG.query_length


# --------------------------------------------------------------------------
# belge tarafı
# --------------------------------------------------------------------------


def test_document_marker_differs_from_query_marker():
    d, _ = build_document_ids([CLS, 10, 11, SEP], DMARK, CFG)
    assert d[1] == DMARK != QMARK


def test_document_is_not_padded():
    """Belge tarafı doldurulmaz — sorgu gibi sabit uzunluğa çekilmez."""
    d, _ = build_document_ids([CLS, 10, 11, SEP], DMARK, CFG)
    assert len(d) == 5
    assert len(d) != CFG.query_length


def test_document_truncates_to_document_length():
    long_ids = [CLS] + list(range(20, 60)) + [SEP]
    d, _ = build_document_ids(long_ids, DMARK, CFG)
    assert len(d) == CFG.document_length


def test_document_encoded_as_query_length_is_the_known_failure_mode():
    """Belge tam olarak query_length vektör verirse sorgu gibi kodlanmıştır.

    pylate'te `encode()` varsayılanı `is_query=True`; bu sessiz hatanın imzası
    budur ve indeksleyici bunu yakalamak zorundadır.
    """
    d, _ = build_document_ids([CLS] + list(range(20, 60)) + [SEP], DMARK, CFG)
    assert len(d) != CFG.query_length


# --------------------------------------------------------------------------
# noktalama skiplist'i
# --------------------------------------------------------------------------


def test_document_keep_mask_drops_skiplist_and_padding():
    ids = [CLS, DMARK, 10, 91, 11, 92, SEP, PAD, PAD]
    keep = document_keep_mask(ids, CFG.skiplist_ids, pad_id=PAD)
    assert keep == [True, True, True, False, True, False, True, False, False]


def test_document_keep_mask_leaves_clean_text_untouched():
    ids = [CLS, DMARK, 10, 11, SEP]
    assert document_keep_mask(ids, CFG.skiplist_ids, pad_id=PAD) == [True] * 5


# --------------------------------------------------------------------------
# MaxSim
# --------------------------------------------------------------------------


def test_maxsim_is_sum_over_query_of_max_over_document():
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    d = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    # q0 en iyi eşleşme 1.0 (d0), q1 en iyi eşleşme 0.5 (d1) -> toplam 1.5
    assert maxsim(q, d) == pytest.approx(1.5)


def test_maxsim_is_sum_not_mean():
    q = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    d = np.array([[1.0, 0.0]], dtype=np.float32)
    assert maxsim(q, d) == pytest.approx(2.0), "ortalama olsaydı 1.0 olurdu"


def test_maxsim_empty_document_scores_zero():
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    assert maxsim(q, np.zeros((0, 2), dtype=np.float32)) == 0.0


# --------------------------------------------------------------------------
# içerik-yalnız sorgu vektörleri (exp3, KEPT)
#
# Ölçüldü (insan-doğrulanmış n=47): MaxSim toplamını yalnız GERÇEK sorgu
# token'ları üzerinden almak R@5'i 0,7021 -> 0,7447, R@20'yi 0,8723 -> 0,9149
# yükseltti; birincil metrik ve guardrail'ler değişmedi.
#
# Bu, ColBERT §3.2'nin "genişletme esastır" ifadesinden bir SAPMADIR ve modül
# başlığında öyle işaretlenmiştir. Genişletme KODLAMADA korunur (eğitim rejimi);
# yalnız SKORLAMA toplamından çıkarılır.
# --------------------------------------------------------------------------


def test_content_token_mask_marks_only_real_tokens():
    from belge_gozu.index.colbert_encode import content_token_mask

    _, attn = build_query_ids([CLS, 10, SEP], QMARK, MASK, CFG)
    mask = content_token_mask(attn)
    assert mask.tolist() == [True, True, True, True, False, False, False, False]


def test_content_token_mask_is_all_true_without_expansion():
    from belge_gozu.index.colbert_encode import content_token_mask

    cfg = ColbertConfig(**{**CFG.__dict__, "do_query_expansion": False})
    _, attn = build_query_ids([CLS, 10, SEP], QMARK, MASK, cfg)
    assert content_token_mask(attn).all()
