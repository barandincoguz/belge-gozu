"""Aday birleşimi — iki kanalın adaylarını reranker'a giden tek listede toplar.

Bu dosyanın var olma sebebi ölçülmüş bir hata: ilk uygulama BM25 listesinin
TAMAMINI yazıp ColBERT adaylarını arkasına ekliyordu. BM25 tarafı ~400 sayfa
uzunluğunda olduğu için ColBERT adayları hiçbir k'da top-k'ya giremiyordu ve
birleşim BM25'e ÖZDEŞ çıkıyordu — D2'nin +0,238'lik kazancı ölçümde görünmüyordu.

Skorlar KARIŞMAZ. G1.6'da üç skor füzyonu ölçülüp reddedildi (küresel RRF
0,674→0,395; doküman bölümleme 0,907→0,837; pencere-içi RRF 0,837→0,535). Burada
yapılan şey farklıdır: iki sıralama örülür, sıralamayı reranker çözer.
"""

from belge_gozu.retrieval.union import union_candidates


def test_union_interleaves_so_the_second_channel_reaches_top_k():
    """Asıl hata buydu: ikinci kanal listenin arkasına atılırsa top-k'ya giremez."""
    bm25 = [f"a{i}" for i in range(100)]
    colbert = ["z1", "z2", "z3"]
    out = union_candidates(bm25, colbert)
    assert "z1" in out[:5], "ikinci kanalın ilk adayı top-5'te olmalı"
    assert out.index("z1") < 10


def test_union_keeps_first_place_for_the_first_channel():
    """BM25 ilk sırayı korur — dondurulmuş reçete yerinden edilmez."""
    assert union_candidates(["a", "b"], ["z", "y"])[0] == "a"


def test_union_deduplicates_preserving_first_occurrence():
    assert union_candidates(["a", "b"], ["b", "c"]) == ["a", "b", "c"]


def test_union_covers_every_candidate_from_both_channels():
    bm25, colbert = ["a", "b", "c"], ["x", "y", "z"]
    out = union_candidates(bm25, colbert)
    assert set(out) == {"a", "b", "c", "x", "y", "z"}
    assert len(out) == len(set(out))


def test_union_handles_uneven_lengths_without_dropping_the_tail():
    out = union_candidates(["a"], ["x", "y", "z"])
    assert out == ["a", "x", "y", "z"]


def test_union_with_empty_second_channel_is_the_first_channel():
    assert union_candidates(["a", "b"], []) == ["a", "b"]


def test_union_with_empty_first_channel_is_the_second_channel():
    assert union_candidates([], ["x", "y"]) == ["x", "y"]


def test_union_recall_is_never_worse_than_either_channel_alone():
    """Değişmez: birleşim hiçbir k'da tek kanaldan kötü olamaz."""
    bm25 = ["a", "b", "c", "d"]
    colbert = ["x", "y", "a", "z"]
    out = union_candidates(bm25, colbert)
    for k in (1, 2, 4, 8):
        assert set(bm25[:k]) <= set(out[: 2 * k])
        assert set(colbert[:k]) <= set(out[: 2 * k])
