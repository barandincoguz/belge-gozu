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
