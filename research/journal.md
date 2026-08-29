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

## #4 — exp4-bm25-f5-bigram → DISCARDED

- **Hipotez:** unigram+bigram shingle, çok-kelimeli terimlere yakınlık sinyali ekler.
- **Sayılar:** R@5 0.6279 (0.7674'ten GERİLEME) · visual_R@5 0.5 · R@20 0.8372
- **Karar:** DISCARDED.
- **Öğrenilen:** F5-kırpılmış bigramlar ("bu_kanun", "türk_meden" gibi kalıplar)
  başlık/atıf sayfalarını şişiriyor ve sorgu uzunluğunu ikiye katlayıp gerçek
  içerik unigram'larının payını düşürüyor. Yakınlık istenirse ayrı kanal olarak
  denenmeli, aynı torbaya karıştırılmamalı.

## #5 — exp5-bm25-f5-stop → KEPT (ikincil-kanıt kuralıyla)

- **Hipotez:** yakın-ıska analizi soru-kalıbı kelimelerinin (göre, nasıl, bir,
  için...) eşleşmeyi bastığını gösterdi; sabit Türkçe işlev-kelimesi listesi
  (canary'ye ayarsız; "zaman"/"iş" gibi içerik-çakışanlar bilinçli dışarıda)
  gürültüyü keser.
- **Sayılar:** R@5 0.7674 (eşit) · R@1 0.5116 (+1) · R@20 **0.907** · MRR 0.6249
  · visual_R@5 1.0 · **chip1 rank 8→4 (top-5'e girdi)** · chip2 rank 2
- **Karar:** KEPT. Program kuralı güncellendi (şeffaf): R@5 eşitken R@20+MRR
  birlikte iyileşiyor ve guardrail gerilemiyorsa tut — kuralın amacı kanıtsız
  karmaşıklığı önlemekti; burada kanıt üç ikincil metrik + vaka analizi.
- **Öğrenilen:** stoplist R@5'i değil derin sıraları düzeltiyor (kalıp kelimeler
  en çok orta-sıra karışıklığı üretiyormuş); chip1 sınıfının (kanun adı + kalıp)
  ana ilacı bu oldu.

## #6 — exp6-bm25-f5-stop-docroute → DISCARDED (guardrail vetosu)

- **Hipotez:** kanun adını anan sorgularda o dokümanın sayfalarını öne bölümle.
- **Sayılar:** R@5 0.7907 (+1) AMA R@20 0.907→0.8372 (−3) ve visual_R@5 1.0→0.875 (−1)
  · chip1 4→2 · chip2 2
- **Karar:** DISCARDED — mutlak bölümleme, ad token'ları tesadüfen eşleşen
  sorgularda gold'u (özellikle RG/tarihi gold'lar) 20 dışına itiyor.
- **Öğrenilen:** yönlendirme sinyali gerçek (+1 @5, chip1 2) ama küresel
  uygulanamaz; aday kümesini DEĞİŞTİRMEYEN, yalnız yerel sırayı düzelten bir
  biçim gerekli → pencere-içi yeniden sıralama (exp7).

## #7 — exp7-docroute-window20 → KEPT

- **Hipotez:** exp6'nın yönlendirme sinyali, aday kümesini değiştirmeyen
  pencere-içi (top-20) yeniden sıralamayla guardrail'leri delmeden kazanılır.
- **Sayılar:** R@5 **0.8140** (35/43; exp5'ten +2) · R@1 0.5116 · R@20 0.907
  (yapısal olarak korunur) · MRR 0.6519 · visual_R@5 1.0 · **chip1 rank 2 ·
  chip2 rank 2**
- **Karar:** KEPT.
- **Öğrenilen:** kural sinyalleri (doküman adı) küresel bölümleme yerine
  pencere-içi sıralama düzeltmesi olarak güvenli; R@20 guardrail'i pencereyle
  hizalanınca gerileme sınıfı yapısal olarak kapanıyor.

## DÖNGÜ SONU — durma koşulu: hedef aşıldı

Taban 0.2326 → **0.8140** (3.5×). Hedef 0.30, esnek 0.40 — ikisi de aşıldı.
Seyir: +BM25 0.674 → +F5 0.767 → +stoplist (R@20/MRR/chip1) → +pencere-yönlendirme 0.814.
Atılanlar: eşit-RRF (0.395 — zayıf kanal gürültüsü), bigram (0.628), mutlak yönlendirme
(guardrail vetosu). Görsel kanalın füzyon katkısı bu bench'te SIFIR benzersiz @5 sorusu.

---

## Round 2 hazırlık notu (2026-08-29)

- P1 üretim entegrasyonu tamam (commit ded732b): reçete `src/belge_gozu/retrieval/{text,hybrid}.py`
  olarak portlandı; üretim bench teyidi R@5 **35/43 = 0.8140** (binary tanım; fractional recall 0.8023
  aynı koşumun farklı metrik tanımı). Canlı: chip'ler gerçek cevap + doğru atıf; anlamsız → abstain.
- research/ lint temizliği: retrieve.py reformatlandı (mantık AYNEN; sağlama koşumu "lint-sanity"
  results.jsonl'da — R@5 0.814, chip 2/2 birebir). retrieve_sha bundan sonra 32dd8055670b tabanlı.

## #8 — exp8-window50 → KEPT

- **Hipotez:** sağlamlık taraması w≥30'da +1 gösterdi (c214 txt 27); pencere 50'ye
  çıkınca yapısal R@20 garantisi kalkar ama ölçüm karar verir.
- **Sayılar:** R@5 **0.8372** (+1: c214) · R@20 **0.9302** (yükseldi!) · MRR 0.655
  · visual 1.0 · chip'ler 2/2
- **Karar:** KEPT — guardrail gerilemedi, aksine yönlendirme bir gold'u top-20'ye çekti.
- **Öğrenilen:** pencere büyümesi bu korpusta güvenli çıktı; üretim portu (WINDOW=20)
  için güncelleme adayı — rapor devrine not.
