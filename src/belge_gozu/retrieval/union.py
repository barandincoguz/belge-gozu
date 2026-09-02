"""Aday birleşimi — D2'nin TÜM entegrasyon yüzeyi.

Skorlar karışmaz. G1.6'da üç SKOR füzyonu ölçülüp reddedildi (küresel eşit RRF
0,6744→0,3953; mutlak doküman bölümleme R@20 0,907→0,8372; pencere-içi RRF
0,8372→0,5349) — zayıf kanal skor düzeyinde birleşince güçlü kanalı bozuyordu.
Burada yapılan şey farklı: iki SIRALAMA örülür, hiçbir skor diğerine karışmaz,
nihai sırayı reranker verir. O olumsuz sonuç bu mekanizmaya uygulanmaz.

Ölçüldü (2026-09-02, insan-doğrulanmış n=47, Mogan-ColBERT-TR):

    kanal        R@5      R@20     R@50
    BM25       0,5745   0,7021   0,7872
    ColBERT    0,6809   0,8723   0,9149
    BİRLEŞİM   0,6809   0,7872   0,9149

`paraphrase` diliminde R@50 0,5714 -> 0,8095; guardrail dilimlerinde gerileme
YOK (`dogrudan-madde` 0,9231 -> 1,0000 iyileşti).
"""

from __future__ import annotations


def union_candidates(primary: list[str], secondary: list[str]) -> list[str]:
    """İki sıralamayı örer; ilk sırayı `primary` korur, tekrarlar düşer.

    NEDEN örgü, neden basit birleştirme değil: ilk uygulama `primary`nin
    tamamını yazıp `secondary`yi arkasına ekliyordu. `primary` ~400 sayfa
    uzunluğunda olduğu için ikinci kanal hiçbir k'da top-k'ya giremiyor ve
    birleşim `primary`ye ÖZDEŞ çıkıyordu — D2'nin +0,238'lik kazancı ölçümde
    görünmüyordu. Reranker'a giden şey bir DERİNLİK kümesidir; örgü onu doğru
    kurar.

    `primary` ilk sırayı korur çünkü BM25 reçetesi dondurulmuştur ve ColBERT
    onu yerinden edemez — yalnız yanına aday ekler.
    """
    out: list[str] = []
    seen: set[str] = set()
    for a, b in zip(primary, secondary, strict=False):
        for cand in (a, b):
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
    for cand in (*primary, *secondary):
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out
