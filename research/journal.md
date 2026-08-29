# Autoresearch günlüğü — Belge-Gözü retrieval kalite döngüsü

Program: `research/program.md` · Harness: `evaluate.py` (DONUK) · Deney yüzeyi: `retrieve.py`
Birincil metrik: canary-answerable (n=43) **R@5**. Sayı satırları: `results.jsonl`.

---

## #0 — Taban (baseline-visual-only)

- **Kurulum:** üretim davranışı — yalnız int8 görsel kanal (MaxSim), skor sırası.
- **Sayılar:** R@5 **0.2326** · R@1 0.093 · R@20 0.3023 · MRR 0.1487 · visual_R@5 0.375
  · chip1 rank 664 · chip2 rank 137
- **Sağlama:** A2 oracle int8 kolu R@5 0.233 (10/43) ile birebir; chip rank'ları
  bağımsız teşhisle (showcase-queries-diagnosis.json) birebir. Harness güvenilir.
- **Düzeltme:** programın ilk taslağı tabanı 0.116 yazmıştı — o sayı ESKİ 1-bit
  üretimindi; int8 geçişi tabanı zaten 2× yaptı. Program güncellendi.
- **Öğrenilen:** görsel kanal tavanı 0.2326 (oracle float=int8 → nicemleme değil,
  model sınırı). Hedef 0.30+ için ek kanal şart.

## #1 — exp1-bm25-only → KEPT

- **Hipotez:** Türkçe hukuk metninde birebir terim eşleşmesi (BM25, PDF metin
  katmanı) görsel kanaldan daha güçlü aday recall verir; kanal tek başına ölçülür
  (ilke 23: füzyondan önce kanal recall).
- **Değişiklik:** retrieve.py = saf BM25 (k1=1.5, b=0.75, tr_lower + \w+, ≥2 harf).
- **Sayılar:** R@5 **0.6744** (0.2326'dan) · R@1 0.5116 · R@20 0.7907 · MRR 0.6101
  · visual_R@5 0.75 · chip1 rank 8 · chip2 rank 2
- **Karar:** KEPT (tüm guardrail'ler de yükseldi).
- **Öğrenilen:** (a) metin kanalı BAŞAT kanal — görselin ~3×'i; (b) RG taramaları
  OCR katmanı taşıyor (4222'de 1 boş sayfa) → requires_visual sorular bile metinden
  bulunuyor (6/8); (c) hedef 0.30 daha füzyonsuz aşıldı — asıl soru artık füzyonun
  BM25-only'yi geçip geçemeyeceği. Dikkat: dogrudan-madde dilimi BM25 lehine
  (sorular madde diliyle örtüşür); paraphrase dilimi gerçek genelleme testi.

## #2 — exp2-rrf-visual-bm25-k60 → DISCARDED

- **Hipotez:** standart eşit-ağırlık RRF (k=60) iki kanalı birleştirince R@5 artar.
- **Sayılar:** R@5 0.3953 (0.6744'ten GERİLEME) · R@1 0.2093 · R@20 0.7442 · MRR 0.316
- **Karar:** DISCARDED (git checkout).
- **Öğrenilen + analiz (soru-bazlı çapraz tablo):** her ikisi @5: 8 · yalnız görsel: 2
  (c202, c205) · yalnız metin: 21 · hiçbiri: 12. Eşit-ağırlık RRF, zayıf kanalın
  (görsel R@5 0.2326) gürültüsünü — kapak/başlık sayfaları — metnin 21 tekil
  kazanımının üstüne bindirip 12'sini düşürüyor. Mükemmel füzyon tavanı 31/43=0.7209:
  füzyonun getirebileceği en fazla +2 soru. BM25 top-skoru güven geçidi olarak
  ayrıştırmıyor (vuran min 14.0 / ıskalayan maks 33.1). "Hiçbiri" listesinde 4 soru
  metin rank 6-8'de (c203:7, c212:6, c304:7, c001:8) → önce metin kanalını iyileştir.

## #3 — exp3-bm25-f5 → KEPT

- **Hipotez:** Türkçe eklemeli; F5 ön-ek kırpması (Can vd.) ek/çekim farklarını
  kapatıp paraphrase eşleşmesini güçlendirir. Tek değişken: tokenizasyon.
- **Sayılar:** R@5 **0.7674** (0.6744'ten, +4 soru) · R@1 0.4884 (−1 soru; veto dışı,
  not edildi) · R@20 0.8837 · MRR 0.6211 · visual_R@5 **1.0** · chip1 8 · chip2 3
- **Karar:** KEPT.
- **Öğrenilen:** kırpma kazancı recall tarafında (R@5/R@20); R@1'de küçük kayıp
  ön-ek çakışmalarının beklenen bedeli. requires_visual 8/8 → OCR katmanı + F5,
  "görsel" soruları tamamen kapsıyor.
