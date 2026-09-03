# Belge-Gözü RAG Kalite Mimarisi v2 — Tasarım Dokümanı

**Tarih:** 2026-08-26 · **Durum:** Kullanıcı onayı bekliyor · **Yazar rolü:** ML Engineer + IR Engineer + RAG Systems Architect

> Bu doküman `docs/superpowers/specs/2026-08-25-belge-gozu-design.md`'nin retrieval ve
> yanıtlama kalitesiyle ilgili bölümlerini (§5, §6, §7) **supersede eder**; korpus hattı
> (§4), web/deploy (§8), mühendislik hijyeni (§9) ve telemetri tasarımı
> (`2026-08-26-telemetry-design.md`) yürürlükte kalır.
> `docs/superpowers/plans/2026-08-26-belge-gozu-plan2.md` bu tasarım ile birlikte
> **supersede edilmiştir**; task-bazlı eşleme tablosu master plan'dadır
> (`2026-08-26-belge-gozu-rag-quality-master.md` §8).

## 1. Neden v2: doğrulanmış kök nedenler

Aşağıdaki her ölçüm **2026-08-26'da bu repo'da, gerçek model ve gerçek indeksle yeniden
üretildi** (betik: scratchpad `verify_retrieval.py`; koşum çıktıları §1.1 tablosunda).
Hiçbiri README'den veya önceki oturum notundan körlemesine alınmadı.

### 1.1 Yeniden üretilen ölçümler

Sorgu A (uzun): *"Türk Medeni Kanunu'na göre yerleşim yeri nasıl tanımlanır?"* ·
Sorgu B (kısa): *"Yerleşim yeri nedir?"* · Altın sayfa: `k4721:4`
(TMK Madde 19: *"Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir."* —
sayfa görüntüsünden gözle doğrulandı, born-digital, okunaklı).

| Ölçüm | Sorgu A | Sorgu B |
|---|---|---|
| Sorgu token sayısı (n_q) | 40 | 21 |
| Altın sayfa binary skoru (per-token) | 54.45 | 68.57 |
| Stage-1 (mean-sign Hamming) sırası | **3127 / 4222** | **1768 / 4222** |
| Exhaustive binary MaxSim sırası | 1576 | **2** |
| Exhaustive binary top-1 | `k2918:91` (Karayolları Trafik) @ 60.55 — alakasız | `k6100:5` @ 69.52 |
| Mevcut hat (Stage-1 top-200 → MaxSim) top-1 | `k6362:55` (Sermaye Piyasası) @ 58.95 — alakasız | `k6102:28` @ 66.10 |
| Altın sayfa mevcut hattın top-5'inde mi | hayır | hayır |
| Stage-1 top-200 ∩ exhaustive top-200 | **%19.0** | **%11.5** |
| Yalnız TMK sayfaları içinde exhaustive sırası | 47 / 206 | 1 / 206 |

Ek doğrulamalar (aynı koşum):

- **Korpus:** 4222 sayfa, 50 kanun + 6 tarihî RG taraması; `k4721` 206 sayfa; indeks
  `page_ids` sırası `meta.parquet` ile birebir hizalı.
- **Yeniden-encode determinizmi:** `k4721:4` tek-görüntülük batch ile yeniden encode
  edildiğinde indeksteki bit'lerle **%100 eşleşiyor** (bit_match=1.0000).
- **Padding bug'ı:** indekste **3960 all-zero token satırı** var, **15 sayfaya** yayılmış
  (ör. `k6098:133-135`, `k5237:1-3`, `rg1928a:5`). Model padding embedding'lerini
  sıfırlıyor; `emb > 0` sign-packing sıfırı gerçek bir all-zero bit vektörüne çeviriyor
  ve bu satırlar Hamming-MaxSim'de gerçek token gibi skorlanıyor.
- **`pad_token_id` uyarısı:** model yüklenirken transformers şunu basıyor:
  *"pad_token_id must be `None` or an integer within the vocabulary (between 0 and
  31999), got 128002."* Kök neden olduğu kanıtlanmadı; reproducibility riski olarak
  manifest'e kaydedilir.
- **Query prefix:** kurulu `colpali-engine==0.3.18`'de `processor.query_prefix == ''`;
  `process_queries` yalnız `text + '<end_of_utterance>' * 10` üretiyor.
- **Manuel `"Query: "` prefix denemesi:** altın sayfa skoru A'da 54.45→58.33,
  B'de 68.57→73.83'e çıkıyor. Belge indeksi eski formatla üretildiğinden bu tam A/B
  değildir; geçerli test için sorgu + belge birlikte yeniden encode edilmelidir (P0).
- **Float-vs-binary nokta kontrolü (8 seçili sayfa, Sorgu A):** float ve binary
  sıralamalar birbirinden farklı (float top-1 `k6098:133`, binary top-1 `k2918:91`) →
  kuantizasyon kaybı gerçek. **Ama** altın sayfa bu zor-negatif alt kümesinde native
  float skorlamada da sonuncu (0.519 vs 0.57-0.589) → kuantizasyon, uzun sorgudaki
  başarısızlığın tek nedeni değil; modelin kendisi Türkçe paraphrase sorgusunda altın
  sayfayı ayırt edemiyor.

### 1.2 Kök neden hiyerarşisi

1. **Stage-1 mean-sign surrogate, MaxSim hedefiyle yapısal olarak uyumsuz.** Sayfanın
   ~1000 token embedding'ini ortalayıp tek sign vektörüne indirmek, late-interaction'ın
   "her sorgu tokenına en yakın belge tokenı" hedefini temsil etmez. Kanıt: Sorgu B'de
   exhaustive sıra 2 iken Stage-1 sırası 1768 — Stage-1, elindeki mükemmel sonucu yok
   ediyor. Top-200 kesişimi %11.5-19 → Stage-1 pratik olarak rastgeleye yakın bir filtre.
2. **Sorgu/belge prompt formatı checkpoint'in eğitim formatından sapmış.** Model kartı
   (birincil kaynak, §9.1): checkpoint `"Query: "` prefix + sondaki newline ile eğitildi;
   colpali-engine newline'ı 0.3.11'de, prefix'i 0.3.13'te düşürdü; image document prompt
   0.3.9 ve 0.3.11'de yeniden yazıldı. Kurulu 0.3.18 → hem sorgu hem belge tarafı eğitim
   formatının dışında.
3. **Model Türkçe için zero-shot ve zayıf.** Model kartı: retrieval eğitim seti *"fully
   English by design"*. Uzun Türkçe paraphrase sorgusunda TMK-içi aramada bile altın
   sayfa 47. sırada; native float nokta kontrolü de aynı yönde. **Görsel kanal tek başına
   ürün kalitesi veremez** → hibrit metin kanalı bir süs değil, zorunluluk (P1).
4. **1-bit sign kuantizasyonu ikincil ama gerçek kayıp.** Float büyüklük bilgisi
   atılıyor; sıralama değişimleri ölçüldü. ColBERTv2 kanıtı (§9.1): centroid + düşük-bit
   residual, 1-bit'te bile ≈0.7 MRR@10 kaybıyla kaliteyi korur — mevcut global sign'dan
   daha iyi bir sıkıştırma ailesi mevcut.
5. **Padding correctness bug'ı** (3960 satır, 15 sayfa) — hedef sorgunun ana nedeni
   değil ama düzeltilmesi zorunlu.
6. **Güven/iletişim hataları:** README'nin "exact MaxSim, not an approximation" iddiası
   yalnız binary uzayda doğru; UI skorları (`128 − 2·Hamming`, sorgu tokenı başına
   ortalama) kalibre edilmemiş benzerliktir; eşik 60.0 iki gözlemden seçilmiş ve yalnız
   top-1 raw skora bakıyor; eşiği düşürmek altın sayfayı aday havuzuna sokmaz (Stage-1
   sırası 3127 iken top-200'e girmesi imkânsız) — yalnız yanlış bağlamın answerer'a gitme
   olasılığını artırır; citation üretmeyen yanıt otomatik olarak 1. sayfaya bağlanıyor
   (`answer/gemini.py:81-82`) → false citation üretebilir.
7. **Testler kaliteyi ölçmüyor:** retrieval testleri sentetik/random embedding +
   self-match; README'nin kendi canlı kaydıyla 17 soruda 1 tam doğru yanıt.

## 2. Değişmez tasarım ilkeleri

Kullanıcıyla mutabık 25 ilke (brief'ten; bu spec bunların tamamına uyar). Özet — plan
kapılarında birebir referans verilir:

1. Threshold ayarı kök-neden düzeltmesi değildir.
2. Candidate recall ölçülmeden reranker eklenmez.
3. Altın sayfa aday havuzunda değilse hiçbir reranker onu kurtaramaz.
4. Mevcut Stage-1 ya kaldırılır ya da gold Recall@candidate ≥ %98 ile kanıtlanmış bir
   candidate generator ile değiştirilir.
5. 4222 sayfa ölçeğinde exhaustive binary/native oracle koşuları kalıcı kalite referansıdır.
6. Model/processor/prompt/dependency/render/mask/quantization/corpus künyesi bir
   **index manifest**'te saklanır.
7. Serve, uyumsuz model/processor/indeks kombinasyonuyla sessizce başlamaz (fail-fast).
8. Görsel retrieval korunur; metin sinyali yasaklanmaz.
9. İki açık mod: `visual-only` (araştırma hattı) ve `hybrid-production` (ürün kalitesi).
10. Born-digital PDF'te gömülü metin doğrudan; OCR yalnız fallback.
11. Retrieval atomu yalnız fiziksel sayfa değil; kanun → bölüm → madde → fıkra → sayfa
    ilişkisi korunur.
12-13. Query rewrite orijinali asla değiştirmez; varyantlar bağımsız koşar, fuse edilir.
14. İlk fusion RRF; learned fusion etiketli veriyle sonra.
15. Expansion/HyDE yalnız ölçülmüş kazançla açılır.
16. Reranking yalnız yüksek-recall candidate union üstünde.
17. Retrieval confidence ≠ answer evidence sufficiency.
18. Raw skor kullanıcıya güven/başarı yüzdesi gibi gösterilmez.
19. Auto-citation fallback kaldırılır.
20. Yanıt yalnız claim-level evidence verification geçerse sunulur.
21. Benchmark + mimari sabitlenmeden threshold kalibrasyonu yapılmaz.
22. Yeni katman yalnız ölçülmüş kazançla varsayılan açılır.
23. Korpus genişletme / agentic derin arama / LocalVLM, retrieval correctness'ten önce değil.
24. Eski dokümanlar silinmez; supersession açıkça yazılır.
25. Her katman feature flag + geri dönüş yoluyla gelir.

## 3. Hedef mimari

```
Kullanıcı sorgusu
  → metadata ayrıştırma (kanun no/adı, madde no, alıntı ifadeler)  [QueryFacets]
  → sorgu varyantları (ORİJİNAL her zaman + normalize + hukukî + metadata)
      ├→ BM25 / phrase / madde retrieval        (sayfa + madde düzeyi)
      ├→ multilingual dense retrieval           (madde düzeyi; model yerel benchmark'la seçilir)
      └→ visual late-interaction retrieval      (exhaustive binary MaxSim, Stage-1'siz)
  → candidate union → RRF (k=60) + dedup + metadata soft-boost (hard filter YOK)
  → [recall gate: union Recall@50 hedefte mi?] → text cross-encoder reranker
      └→ görsel kanıt gereken sayfalarda visual sinyal korunur
  → madde/sayfa evidence pack (komşu sayfa genişletmesi doğru sayfa bulunduktan SONRA)
  → evidence sufficiency verifier (claim segmentation + support/refute/insufficient)
      ├→ destek varsa: yanıt + claim-level citation
      └→ destek yoksa: kalibre edilmiş abstention
```

Model/kütüphane adları burada **kesinleştirilmez** (ilke: nihai seçim Belge-Gözü'nün
held-out hukuk benchmark'ına göre). Aday havuzu ve karşılaştırma planı §9.2'de.

### 3.1 v0'dan farkların gerekçesi

| v0 | v2 | Gerekçe (yerel kanıt) |
|---|---|---|
| Stage-1 mean-sign eleme | Kaldırılır; üretim yolu exhaustive binary MaxSim | Rank-2 sonucu 1768'e atıyor; top-200 kesişimi %11.5-19. Vektörize exhaustive 4222 sayfada **~1.2 s** ölçüldü (M4 Pro, tek çekirdek) — bu ölçekte eleme gereksiz. Geri dönüş isteği olursa PLAID-tarzı centroid interaction ancak Recall@candidate ≥ %98 kapısıyla girer. |
| Tek sorgu, tek kanal | Çok-varyant × çok-kanal + RRF | Model Türkçe zero-shot; TMK-içi aramada bile 47. sıra → görsel kanal tek başına yetmez. BM25+dense+visual union recall'u yükseltir (bBSARD: BM25 statü retrieval'da güçlü baseline; BGE-M3 MIRACL-tr 71.5 vs BM25 45.8). |
| Skor eşiği = güven | Retrieval gate ≠ evidence gate; kalibre selective answering | Eşik 2 gözlemden; raw Hamming benzerliği olasılık değil. |
| Sayfa atomu | Kanun→madde→fıkra→sayfa hiyerarşisi | Hukuk sorusunun doğal cevabı madde/fıkra; yapı-farkındalık statü-QA'da ölçülmüş kazanç (§9.1 SearchFireSafety). |
| Auto-citation fallback | Kaldırılır; claim-level verified citation | False citation üretme riski kodda doğrulandı. |

## 4. Faz yapısı ve kapılar (özet)

Ayrıntılar plan dosyalarında; buradaki kapılar bağlayıcıdır.

- **P0 — Retrieval correctness ve ölçülebilirlik.** Benchmark veri modeli + 30-50
  soruluk human-verified retrieval_eval set + stage-bazlı teşhis harness'ı + exhaustive/native
  oracle'lar + Stage-1'in kaldırılması + processor format A/B yeniden indeksleme +
  padding/mask düzeltmesi + index manifest + fail-fast + kuantizasyon ablasyonu +
  README/UI dürüstlük düzeltmeleri + kalite telemetri şeması.
  **Kapı:** `k4721:4` coverage kanıtlı; iki hedef sorgu kalıcı regression setinde;
  kullanılan candidate generator için gold Recall@candidate ≥ %98 (sağlanamıyorsa
  Stage-1 üretimde kapalı kalır — exhaustive yol %100 ile bunu trivially sağlar);
  oracle karşılaştırması her koşumda; indeks uyumsuzluğu fail-fast; padding satırları
  skorlanmıyor; kuantizasyon kaybı sayılandırılmış; baseline raporu üretilmiş.
- **P1 — Hybrid ve structure-aware retrieval.** Metin çıkarma + kalite dedektörü + OCR
  fallback + madde segmentasyonu + metadata indeksi + BM25 + dense + RRF + cross-encoder
  reranker (recall-gated) + iki mod + HF Space bütçeleri.
  **Kapı:** candidate-union Recall@50 ≥ %95 overall ve ≥ %90 her kritik dilimde;
  reranker kazancı CI'lı raporlanır; Sorgu A'nın dayanak sayfası final top-5'te
  (zorunlu regression); kazanç göstermeyen katman varsayılan açılmaz.
- **P2 — Selective answering, citation, kalibrasyon.** Claim-level verification +
  citation precision/completeness + auto-citation kaldırma + kalibre abstention +
  risk-coverage + outcome telemetry + koşullu fine-tuning alt planı.
  **Kapı:** unanswerable'da false supported-answer ≤ %2; claim-level citation support
  precision ≥ %98; kalibrasyon/test verisi ayrık; threshold'lar index/model revision'a
  versiyonlu; verifier geçmeyen yanıt kesin yanıt olarak gösterilmez.

P0 kapısı geçmeden P1'in default entegrasyonu ve P2 kalibrasyonu **başlamaz**; P1 kapısı
geçmeden P2 **başlamaz**.

## 5. Benchmark tasarımı — `belge-gozu-bench v2`

### 5.1 Ölçek ve aşamalama

- **RetrievalEval v1 (P0):** 30-50 soru, tamamı insan-doğrulamalı; iki hedef sorgu dahil.
- **Final (P1 sonu):** ~120 cevaplanabilir + ~30 cevaplanamaz/OOD.
- Agent/LLM taslakları doğrudan final'e girmez: `verification_status="draft"` →
  insan onayı → `"verified"`. `verified_by` alanı zorunlu.
- Gerçek kullanıcı sorguları (telemetri `events` tablosundan, izinli) ve insan
  paraphrase'leri tercih edilir.

### 5.2 Kayıt şeması (JSONL; tam sözleşme P0 planı Task 6'da)

`question_id, question, query_style, answerable, gold_doc_ids, gold_page_ids,
gold_article_ids, minimal_evidence_spans, reference_answer, slice, difficulty,
source_type, requires_visual, requires_multi_hop, unanswerable_reason, verified_by,
verification_status`

### 5.3 Dilimler

`dogrudan-madde` (born-digital doğrudan madde soruları) · `paraphrase` (günlük dil ↔
hukuk dili) · `madde-numarali` (kanun/madde numarası içeren) · `ayni-kanun-hard-negative`
· `capraz-kanun-terim` (aynı terimi kullanan farklı kanunlar) · `tablo-layout` ·
`tarihi-tarama` · `belirsiz-coklu-dayanak` · `multi-hop` · `korpus-disi` ·
`eksik-kanit` · `anlamsiz-ood`. Cevaplanamaz dilimlerin taksonomi kaynağı UAEval4RAG
(§9.1): out-of-database / underspecified / false-presupposition / nonsensical
uyarlaması `unanswerable_reason` alanında.

### 5.4 Leakage önleme

- **Law-grouped split:** train/dev/test ayrımı kanun (doc_id) bazında gruplanır; aynı
  kanunun çok benzer maddeleri train ve test'e dağılmaz. Fine-tuning aşamasında
  **entire-law held-out** zorunlu (P2 koşullu alt plan).
- Split dosyası (`data/bench/splits_v1.json`) benchmark versiyonuyla birlikte commit'lenir.
- Kalibrasyon verisi ile final test verisi kesin ayrık (P2).

## 6. Ölçüm seti

**Retrieval:** corpus coverage · candidate survival (aşama aşama) · Recall@{1,5,10,20,50,200}
· MRR · nDCG@5 · dilim-bazlı · kanun-bazlı · stage-wise oracle gap (üretim vs exhaustive
binary vs native float) · visual-only vs hybrid delta. Hesap yeri: `bench/metrics.py` +
`bench/harness.py`; her koşum `EvalReport` JSON'u (koşum künyesi: git sha + index
manifest + config) olarak `data/bench/results/` altına.

**Answering (P2):** answer accuracy · claim support · citation precision ·
citation recall/completeness · unsupported claim rate · answerable coverage ·
false-abstain · false-answer · risk-coverage. Hesap yeri: `answer/verify.py` çıktıları
üzerinde `bench/answer_eval.py`.

**Operational:** index build süresi/boyutu · peak memory · query encode / candidate
generation / rerank / answer gecikmeleri · p50/p95 uçtan uca · maliyet/1000 sorgu ·
cold-start. Hesap yeri: mevcut telemetri (`bg_stage_duration_seconds` + `events`) +
`bench/harness.py` latency alanları. Yeni stage adları mevcut telemetri kataloğuyla
çakışmadan eklenir (P0 Task 13).

## 7. Ablasyon matrisi

Her deney: tek değişen faktör · aynı benchmark split'i · aynı corpus/index revision ·
retrieval metriği (Recall@k, MRR, nDCG@5) · latency/memory maliyeti · bootstrap CI ·
karar. Koşum sırası ve karar kuralları master plan §6'da.

| # | Deney | Faz |
|---|---|---|
| A1 | Mevcut processor formatı (cpe-0.3.18: prefix'siz) — baseline | P0 |
| A2 | Training-compatible format (`"Query: "` + newline + eğitim doc prompt'u; sorgu+belge birlikte reindex) | P0 |
| A3 | Sentence Transformers MultiVectorEncoder yolu (resmî eğitim-format referansı) | P0 |
| B1 | Mean-sign Stage-1 açık (mevcut) vs kapalı (exhaustive) | P0 |
| B2 | Aday sayısı 200 / 500 / 1000 / exhaustive oracle | P0 |
| C1 | Native float16 MaxSim oracle | P0 |
| C2 | 1-bit sign (mevcut) vs int8 vs float16; residual (ColBERTv2-tarzı) koşullu uzantı | P0 |
| D1 | Query augmentation token'ları açık/kapalı | P0 |
| E1 | Original query only vs single-rewrite-replacement vs original+multi-variant fusion | P1 |
| F1 | BM25 only / dense only / visual only / BM25+dense / text+visual RRF | P1 |
| F2 | Metadata soft-boost açık/kapalı | P1 |
| F3 | OCR fallback açık/kapalı (tarihî dilimde) | P1 |
| G1 | Cross-encoder reranker açık/kapalı (recall-gated havuzda) | P1 |
| H1 | Evidence verifier açık/kapalı | P2 |
| H2 | Raw threshold vs kalibre selective answering | P2 |

## 8. Reproducibility sözleşmesi

- **Index manifest** (`data/index/manifest.json`): model adı + HF revision sha,
  colpali-engine/transformers/torch sürümleri, query format kimliği (prefix, suffix
  token, adet, newline), doc prompt kimliği, render (dpi/format/quality), mask
  politikası, quantization, corpus checksum (page_ids + meta içerik hash'i), sayfa/token
  sayıları, build zamanı, git sha.
- **Serve fail-fast:** açılışta manifest ↔ çalışan encoder/processor/korpus
  karşılaştırılır; uyumsuzlukta `IndexCompatibilityError` (bilinçli geçersiz kılma:
  `BG_ALLOW_INDEX_MISMATCH=true`, log'da kalıcı uyarı).
- **Her EvalReport koşum künyesi taşır:** git sha, index manifest özeti, config,
  benchmark sürümü. Belirsiz/yeniden üretilemeyen ölçüm rapora "unverified" etiketiyle
  girer, kapı kararlarında kullanılmaz.
- `pad_token_id` uyarısı manifest'e kaydedilir; format A/B koşumlarında değişip
  değişmediği izlenir (risk kaydı: master plan §9).

## 9. Research notes (birincil kaynaklar, 2026-08-26'da çekildi)

### 9.1 Doğrulanmış dayanaklar

- **ColSmol-500M model kartı** (huggingface.co/vidore/colSmol-500M): eğitim seti
  *"fully English by design"*; *"current colpali-engine no longer sends the query prefix
  and trailing newline that this checkpoint was trained with. The trailing newline went
  in 0.3.11 (illuin-tech/colpali#280) and the 'Query: ' prefix in 0.3.13
  (illuin-tech/colpali#339), and the image document prompt was rewritten in 0.3.9 and
  again in 0.3.11"*; *"The Sentence Transformers configuration in this repository
  reproduces the original training-time format"* (`MultiVectorEncoder`). Sınırlama notu:
  yüksek-kaynaklı diller dışında genelleme sınırlı olabilir.
- **colpali `add_model_family.md`**: padding embedding'leri model çıkışında sıfırlanır
  (*"Multiply by attention_mask... so padding tokens score as zero"*) — bu sözleşme
  dot-product MaxSim varsayar; **sign-binarizasyon bu sözleşmeyi kırar** (sıfır vektör →
  geçerli bit deseni). Ayrıca `query_prefix`/`query_augmentation_token` processor
  attribute'ları format restorasyonu için resmî kanca.
- **ColPali (arXiv 2407.01449, ICLR 2025):** ViDoRe ort. nDCG@5 81.3 vs en iyi metin
  hattı 67.0; sayfa başına 1024 patch, 128-dim; binary quantization ve token pooling
  meşru ~100x depolama kolları; query augmentation token ablasyonu İngilizce'de nötr ama
  **Fransızca'da kazanç** → Türkçe için augmentation token'ları koru, D1'de ölç.
- **ColBERTv2 (NAACL 2022):** centroid + residual (1-2 bit); b=1'de MRR@10 kaybı yalnız
  0.7 puan; cross-encoder distillation şablonu.
- **PLAID (arXiv 2205.09707):** centroid interaction ile aday üretimi; CPU'da 45×
  hızlanma, kalite paritesi → Stage-1 geri istenirse şablon bu, mean-sign değil.
- **BGE-M3 (arXiv 2402.03216):** dense+sparse+multi-vector tek modelde; 100+ dil,
  MIRACL-tr nDCG@10 71.5 vs BM25 45.8; 8192 token.
- **RRF (SIGIR 2009):** `RRFscore(d) = Σ 1/(k + r(d))`, k=60; skor kalibrasyonu
  gerektirmez; Condorcet/CombMNZ ve en iyi tekil sistemden %4-5 iyi.
- **Expansion failure (EACL Findings 2024):** genişletme zayıf retriever'lara yarar,
  güçlülere **zarar** — orijinal sorguyu korumanın ve expansion'ı ölçmeden açmamanın
  birincil dayanağı.
- **RAGTurk (SIGTURK 2026):** Türkçe RAG'de HyDE en yüksek doğruluk (%85) ama
  cross-encoder rerank + context augmentation Pareto-optimal (%84.6, çok daha ucuz);
  *"over-stacking generative modules can degrade performance by distorting morphological
  cues"* — Türkçe'de üretken katman yığmama uyarısı.
- **TR-TEB (LREC 2026):** 45 açık embedding modelini 47 Türkçe veri setinde ölçen ilk
  standart benchmark — dense model aday listesinin tarama kaynağı; nihai seçim yine
  yerel benchmark'la.
- **OCRTurk (SIGTURK 2026):** 7 OCR modeli; **PaddleOCR** Türkçe'de genel en iyi.
  Tarihî taramaların kapsanıp kapsanmadığı abstract'tan doğrulanamadı → P1 OCR
  benchmark'ı kendi tarihî dilimimizde koşulur.
- **bBSARD (COLING 2025 RegNLP):** statü-madde retrieval şablonu (1108 soru → madde
  ID'leri); **BM25, 300M parametre altındaki zero-shot dense modellerden iyi**;
  küçük dile-özgü fine-tuned model, proprietary embedding'leri yakalayabiliyor →
  P2 koşullu fine-tuning'in dayanağı.
- **LegalBench-RAG (arXiv 2408.10343):** gold kanıt = minimal span (belge/chunk değil);
  hassas granülerlik citation üretimi için önkoşul → `minimal_evidence_spans` alanının
  dayanağı.
- **RAGAS (EACL 2024) / ARES (NAACL 2024):** statement-decomposition faithfulness
  şablonu; ARES'in PPI'ı — birkaç yüz insan etiketiyle LLM-judge düzeltmesi →
  otomatik evaluator'lar yalnız insan-doğrulamalı subset ile kalibre edilmiş yardımcı.
- **UAEval4RAG (ACL 2025):** 6 kategorili unanswerable taksonomisi + acceptable/
  abstention_evalwered ratio metrikleri; prompt tasarımı tek başına abstention kalitesini ~%80
  oynatabiliyor.
- **SearchFireSafety (ACL 2026):** statü-merkezli QA'da hiyerarşi-farkındalı retrieval
  ölçülmüş kazanç; **domain-adapted modeller eksik kanıtta daha çok halüsinasyon** →
  fine-tuning kapısının abstention ölçümüyle birlikte değerlendirilme şartı.

Genelleme uyarısı: bu sonuçların hiçbiri Belge-Gözü'ne doğrudan taşınmaz; her öneri
yerel benchmark'ta doğrulanmadan varsayılan açılmaz (ilke 22).

### 9.2 Model aday havuzları (seçim yerel benchmark'la)

- **Dense (P1):** BGE-M3 (dense modu) · multilingual-E5 (small/base) · TR-TEB retrieval
  liderlerinden CPU-uygun 1-2 aday. Kısıt: HF Space CPU'da sorgu encode < ~1 s.
- **Cross-encoder (P1):** bge-reranker-v2-m3 · mmarco-mMiniLM sınıfı çok dilli CE ·
  TR-TEB/RAGTurk işaret ettiği adaylar. Kısıt: 50 aday × CPU'da toplam < ~5 s veya
  aday sayısı düşürülür (ölçülür).
- **OCR (P1):** PaddleOCR · Tesseract (tur) — kendi tarihî dilimimizde CER/etiket
  doğruluğuyla seçilir.
- **Verifier (P2):** Gemini structured-output (mevcut answerer istemcisi) baz; yerel
  NLI alternatifi ancak ölçümle.

## 10. Kapsam dışı (bu v2 turunda YAGNI)

Korpus genişletmesi (1475 dahil — master plan §8'de ertelenme gerekçesi) · agentic
"derin arama" · LocalVLM vitrini · ZeroGPU · Qdrant adaptörü · streaming/TTFT ·
fine-tuning (P2'de yalnız koşullu alt plan olarak kapısı tanımlanır, uygulanmaz).

## 11. Başarı kriterleri (v2'nin "bitti" tanımı)

1. P0/P1/P2 kapılarının tamamı sayısal kanıtla geçilmiş; kapı raporları
   `docs/research/findings/` altında commit'li.
2. Sorgu A ve B kalıcı regression setinde ve final hatta top-5 içinde (A: P1 kapısı,
   B: P0 kapısı).
3. `belge-gozu-bench v2` (≥120 answerable + ≥30 unanswerable, insan-doğrulamalı,
   law-grouped split'li) yayınlanabilir durumda.
4. README dürüst: "exact binary" düzeltilmiş, skor etiketi "uncalibrated similarity",
   sonuç tabloları CI'lı, kayıplar dahil.
5. Ablasyon matrisi (§7) doldurulmuş; varsayılan açık her katmanın sayısal gerekçesi var.
