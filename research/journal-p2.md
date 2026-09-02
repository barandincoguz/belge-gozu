# P2 deney günlüğü — semantik metin kanalı

Kurallar `research/program-p2.md`'de. Her deney: hipotez → değişiklik → sayılar
→ karar → öğrenilen. Başarısız deney de kayıttır.

---

## exp1 — Türkçe geç-etkileşim kanalı (Mogan-ColBERT-TR) · **KEPT**

**Hipotez.** BM25 sözlükseldir; `paraphrase` dilimi tam olarak sözlüksel
örtüşmenin olmadığı yerdir. Türkçe'de geç-etkileşim modelleri yoğun modelleri
alan-özgü görevlerde açık ara geçiyor (TurkColBERT, arXiv 2511.16528) ve MaxSim
altyapısı bu repoda zaten var — yanlış modele bağlı.

**Değişiklik.** `moganai/Mogan-ColBERT-TR` @ `ad90b4f`, `document_length=512`,
10.531 chunk üzerinde fp16 indeks. Aday düzeyinde örülmüş birleşim; skor füzyonu
yok. Yeni bağımlılık yok.

**Sayılar** (insan-doğrulanmış set, n=47, 60 ms/sorgu):

| kanal | R@1 | R@5 | R@20 | R@50 |
|---|---|---|---|---|
| BM25 | 0,4681 | 0,5745 | 0,7021 | 0,7872 |
| ColBERT | 0,5532 | 0,6809 | 0,8723 | 0,9149 |
| BİRLEŞİM | 0,4681 | 0,6809 | 0,7872 | **0,9149** |

**Birincil metrik: `paraphrase` R@50 0,5714 → 0,8095 (+0,2381), n=21.**

Guardrail'ler: `ayni-kanun-hard-negative`, `madde-numarali`, `tarihi-tarama`
1,0000'de kaldı; `dogrudan-madde` 0,9231 → 1,0000 **iyileşti**. Gerileme yok.

**Karar: KEPT.** Hedef 0,90'a ulaşılmadı, ara hedef 0,75 aşıldı.

**Öğrenilenler.**

1. **ColBERT tek başına birleşimden daha iyi sıralıyor** (R@20 0,8723 vs
   0,7872). Birleşim BM25'e ilk sırayı bıraktığı için R@1/R@20'de ondan geri
   kalıyor, R@50'de eşitleniyor. Reranker eklendiğinde bu fark kapanmalı —
   ama eğer kapanmazsa "BM25 ilk sırayı korusun" kararı yeniden ölçülmeli.
2. **Ölçüm hattındaki bir hata kazancı tamamen gizlemişti.** İlk birleşim
   uygulaması BM25'in ~400 sayfalık listesini yazıp ColBERT'i arkasına
   ekliyordu; birleşim BM25'e ÖZDEŞ çıkıyor, +0,238 hiç görünmüyordu. Ders:
   birleşim bir DERİNLİK kümesidir, liste birleştirme değil.
3. **Sözleşme modelden okunmalı.** `sentence_bert_config.json` Mogan'da
   `max_seq_length: 31` gönderiyor; miras alınsaydı her belge 31 token'a
   kesilirdi ve kanal sessizce çöp üretirdi.

**Kanıt.** `data/bench/results/d2-colbert-mogan.json`,
`data/index-colbert-mogan-f16/colbert.json`.

**Sıradaki.** exp2: reranker (`BAAI/bge-reranker-v2-m3`) — birleşimi sıralayıp
R@5'i R@50'ye yaklaştırmak, ve G1.3'ü ölçülmüşe çevirmek.
