# P2 gerçeklik denetimi — supersession + hazırlık raporu

- **Tarih:** 2026-08-30
- **Branch / HEAD:** `feat/p0-retrieval-correctness` @ `99b8364`
- **Denetlenen plan:** `docs/superpowers/plans/2026-08-26-belge-gozu-p2-selective-answering.md`
  (2026-08-26 yazıldı, 12 task) + master §5 kapıları G2.1-G2.8
- **Yöntem:** salt-okunur. Her supersession iddiası plan satırı + bugünkü kod/veri satırı
  ile çiftlenmiştir. Kod okundu, veri sayıldı, hiçbir şey değiştirilmedi.

## 0. Yönetici özeti

Plan 2026-08-26'da, **P1'in tam arayüz yüzeyini üreteceği varsayımıyla** yazıldı. Ruling R23
(2026-08-29) P1'i "ölçülmüş BM25 reçetesinin üretimleştirilmesi"ne daralttı
(`docs/superpowers/plans/2026-08-26-belge-gozu-p1-hybrid-retrieval.md:5-24`). Sonuç: P2'nin
tükettiği P1 tiplerinin çoğu **hiç yazılmadı**.

| Durum | Task'ler | Sayı |
|---|---|---|
| VALID (yazıldığı gibi koşar) | T7, T11 | **2** |
| STALE (bir varsayım kırıldı, değiştirilerek koşar) | T1, T2, T5, T6, T8, T12 | **6** |
| PARTIALLY DONE | T3, T9 | **2** |
| BLOCKED-ON-DATA | T4, T10 | **2** |

**En yük taşıyan bayat varsayım:** plan, P2'nin omurga tipi `EvidencePack`/`EvidenceUnit`'i
(`retrieval/evidence.py`, P1 T11) ile `QueryFacets`'i (`retrieval/query.py`, P1 T4-T5)
"birebir tüketeceğini" yazar (plan `:64`, `:150-151`, `:307`, `:312`). **Bu iki dosya
mevcut değil** — `src/belge_gozu/retrieval/` yalnız `core.py`, `hybrid.py`, `text.py`,
`types.py` içerir. Aynı boşluk `corpus/articles.py` (madde katmanı → `CitationRef.article_id`
kaynağı yok), `retrieval/rerank.py` (→ `ConfidenceFeatures.rerank_top1` kaynağı yok) ve
`data/bench/bench_v2.jsonl` (T12 final koşumunun girdisi) için de geçerli. Yani T1/T2/T4/T5
ve T12 hepsi var olmayan bir yüzeye yaslanıyor.

**En keskin, planda HİÇ geçmeyen tehlike:** `[Sk]` etiketleri ile sayfa görüntüleri
arasında **bağ yok**. `answer/gemini.py:54-55` beş görüntüyü etiketsiz bir dizi olarak,
ardından tek bir metin bloğu gönderir (`contents=[*parts, prompt]`); modelin k'ıncı
görüntüyü `[Sk]` ile eşlemesi tamamen konumsal çıkarımdır ve hiçbir yerde doğrulanmaz
(bulgu B14, `docs/research/findings/2026-08-29-e2e-review.md:241`; öncelik #18, `:294`).
**Bu düzeltilmeden ölçülecek "citation precision" konumsal şansı ölçer** — yani G2.2
sayısı üretilebilir ama geçersizdir. Düzeltme saatler sürer ve T3/T4'ten ÖNCE gelmelidir.

---

## 1. Task bazında supersession tablosu (T1-T12)

### T1 — Claim segmentasyonu + verifier (`answer/verify.py`) → **STALE**

| Plan varsayımı | Plan satırı | Bugünkü gerçek |
|---|---|---|
| "Consumes: P1 `EvidencePack/EvidenceUnit` (`retrieval/evidence.py`) — birebir" | `:64` | `src/belge_gozu/retrieval/evidence.py` **YOK** (R23 backlog, P1 planı `:22`) |
| Test `EvidencePack(question=…, facets=QueryFacets(…), units=[])` | `:150-151` | `retrieval/query.py` de **YOK** → test derlenmez |
| `CitationRef.article_id: str \| None` | `:74` | Madde katmanı yok (`corpus/articles.py` backlog) → alan daima `None` |
| "`GeminiClient`'a `generate_structured(...)` metodu eklenir" | `:164-166` | `answer/gemini.py:50` yalnız `generate()` sunuyor — ekleme gerçekten gerekli (bu satır VALID) |
| Önbellek anahtarı = sha256(soru + claim metinleri + unit kimlikleri) | `:104-106` | **Anahtar eksik:** model kimliği (`config.py:66`, 2.0-flash→3.6-flash zaten bir kez değişti) ve istem/şema sürümü yok → istem değişince bayat verdict sessizce yeniden kullanılır. `data/cache/` dizini de henüz yok |

**Kurtarılabilir:** `segment_claims` tamamen bağımsız ve VALID (deterministik, kota yakmaz).
**Değişiklik:** `EvidenceUnit`/`EvidencePack`'i `answer/verify.py` içinde YERELDE tanımla —
`list[PageHit]` (`retrieval/types.py:14`) + `<index_dir>/page_texts.parquet` metninden
kurulur. P1 T11 sonradan yazılırsa aynı ada bakan bir adapter yeterlidir.

### T2 — İki kapı: retrieval gate ↔ evidence sufficiency gate → **STALE**

Mimari (iki kapının ayrılması) sağlam; **üç arayüz varsayımı** kırık:

1. **Skor ölçeği.** Plan `:186-187`: "eski skor eşiği artık TEK kapı değil;
   `min_score_threshold` yalnız flag kapalıyken eski davranış olarak kalır." Plan
   yazıldığında o eşik görsel/binary ölçekteydi (60.0). Bugün `answer/base.py:47`
   (`hits[0].score < self.min_score`) **BM25 ölçeğinde** bir skoru karşılaştırıyor
   (`retrieval/hybrid.py:232`, `config.py:151` → 10.6). Plan'ın "eski davranış olarak
   kalır" ifadesi güvenli bir geri düşüşü ima ediyor; **ölçüm bunu çürütüyor**:
   cevaplanamaz 5 sorunun **4'ü eşiği geçiyor** (23.52 / 12.96 / 17.86 / 15.54; yalnız
   anlamsız c006 4.23 altta) — `tests/retrieval/test_semantic_retrieval_eval.py:202-221`
   `xfail(strict=True)` kilidi + `config.py:144-150`. G2.8 "güvenli fallback" metni bunu
   **açıkça yazmak zorunda**: flag'ler kapalıyken sistem *ölçülmüş biçimde ayırmayan* bir
   kapıya döner, güvenli bir kapıya değil.
2. **Dönüş tipi + status kanalı.** Plan `:180`: `ask(...) -> tuple[VerifiedAnswer, list[PageHit]]`.
   Bugün `answer/base.py:40-42` `tuple[Answer, list[PageHit]]` döner ve **status artık
   `app/main.py:604-608`'de hesaplanıyor** (`degraded`/`abstained`/`answered`), gövdenin üst
   düzeyinde dönüyor (`main.py:630`). Plan bu alanı bilmiyor. `abstain_reason` **paralel bir
   kanal icat edilerek değil, mevcut `status`'a bağlanarak** taşınmalı.
3. **`EvidencePack`** — T1 ile aynı boşluk.

### T3 — Auto-citation kaldırma + citation metrikleri → **PARTIALLY DONE**

- **YAPILDI (Step 1a + G2.7):** fallback `answer/gemini.py:79-84`'te silinmiş, yorum
  sınıfı açıkça "(P2 / G2.7)" diye anıyor; test TERSine çevrilmiş —
  `tests/answer/test_gemini.py::test_no_marker_means_no_citation` `a.citations == []`
  iddia ediyor. Plan `:224` ("sayfa-1'e bağlama davranışı `answer/gemini.py:81-82`'den
  kaldırılır") **kapandı**.
- **YAPILMADI:** `src/belge_gozu/bench/answer_eval.py` yok; `AnswerRecord`,
  `citation_precision/completeness`, `selective_metrics` yok.
- **GEÇERLİLİK ÖNKOŞULU (planda yok):** B14 — `[Sk]`↔görüntü bağı konumsal
  (`answer/gemini.py:55`). Bu düzeltilmeden `citation_precision` anlamlı değil.
- Gold tarafı hazır: `gold_page_ids` **ve** `gold_article_ids` bench şemasında var
  (`bench/dataset.py:38-39`) — ama `article_id` üretim tarafında yok, o yüzden G2.2
  eşleşmesi **`page_id` üzerinden** yapılmalı.

### T4 — Answerable/unanswerable koşum harness'ı → **BLOCKED-ON-DATA**

Kod yazılabilir; **üretmesi istenen dilim kırılımı bugünkü veriyle üretilemez.**
Plan `:273-274` üç ayrı unanswerable satır ister: `korpus-disi` / `eksik-kanit` /
`anlamsiz-ood`. Ölçülen dağılım (`data/bench/retrieval_eval_v1.jsonl`, n=48):

| dilim | soru | not |
|---|---|---|
| `korpus-disi` | **3** (c003, c004, c005) | |
| `anlamsiz-ood` | **2** (c006, c007) | hiçbir test bağlamıyor (B2, `e2e-review.md:200-202`) |
| `eksik-kanit` | **0** | şemada tanımlı (`bench/dataset.py:22`), örneklem yok (B28, `e2e-review.md:255`) |

Ayrıca `bench/harness.py:260` (`if not q.answerable: continue`) bugün 5 satırı **hiç
koşmuyor** — yani abstain precision/recall için tek bir taban sayı yok. CLI'de de karşılık
yok: `cli.py`'de yalnız `bench run` (`:522`) ve `bench oracle` (`:595`) var, `bench answers`
yok, `--split` bayrağı hiç yok (B7).

### T5 — Güven özellikleri (`ConfidenceFeatures`) → **STALE**

Plan'ın 7 özelliği, bugün istek başına gerçekten elde olana karşı:

| Özellik | Plan | Bugün | Karar |
|---|---|---|---|
| `rerank_top1` | `:298` | Reranker yok (P1 T10 backlog) → plan'ın kendi fallback'i "fused top1" = BM25 `score` | **var** (ama `score`'un kopyası) |
| `margin_1_2` | `:299` | `main.py:454` hesaplıyor, `RequestEvent.margin_1_2` kaydediyor (`schema.py:16`) | **var** — B36: "kalibrasyonun en ucuz girdisi elde duruyor" |
| `channel_agreement` | `:300` | `HybridRetriever.channel_rankings` **yok**; yalnız 2 kanal var (`hybrid.py:209-211`) ve görselin @5 benzersiz katkısı **ölçülmüş SIFIR** (`hybrid.py:10-11`), 3 füzyon biçimi reddedildi | **dejenere** — kurulabilir ama bilgi taşıması beklenmemeli |
| `exact_match` | `:301` | `QueryFacets.quoted_phrases` / `phrase_hits` yok (modül yok) | **YOK** |
| `law_match` | `:302` | `facets.doc_ids` yok **AMA** `HybridRetriever.routed_docs()` (`hybrid.py:175-178`) tam bunu hesaplıyor ve telemetride duruyor (`detail.retrieval.routed_docs`, `hybrid.py:220`) | **ikame edilebilir** |
| `article_match` | `:303` | Madde katmanı yok | **YOK** |
| `verifier_support_ratio` | `:304` | T1'den gelir | koşullu |

Yani **7'nin 2'si yok, 1'i dejenere**. `FEATURE_ORDER` yeniden türetilmeli.
**Planın bilmediği, bugün BEDAVA duran sinyaller:** `visual_top1` (`hybrid.py:219`),
kanal `bm25_top1` ile SERVİS EDİLEN top-1 arasındaki fark (yönlendirme yüzünden farklı
olabilir — `config.py:129-135`), `len(routed_docs)`, sorgu uzunluğu. Bunlar zaten
`detail.retrieval`'e yazılıyor.

**Telemetri geçmişi kalibrasyon verisi olarak KULLANILAMAZ:** `main.py:502`
`detail.hits` yalnız `{page_id, score}` tutuyor — hit başına `visual_score` kaydedilmiyor
(yalnız `visual_top1` toplu olarak). `channel_agreement` geçmişten geri kurulamaz, taze
koşum gerekir.

### T6 — Kalibratör + versiyonlu threshold → **STALE**

- `Calibrator` / `CostMatrix` / `choose_threshold` makinesi **VALID** (saf numpy runtime,
  `numpy>=2.0` zaten `pyproject.toml:7`'de).
- `eval` extra + `scikit-learn`: `pyproject.toml:24-29`'da yalnız `ml` ve `dev` extra'ları
  var → gerçekten eklenmesi gerekiyor (plan bu konuda doğru).
- **Kırık varsayım — versiyonlama anahtarı.** Plan `:360-362`:
  `calibration_dir(index_revision) -> data/calibration/<index_revision-guvenli-ad>/`,
  "`index_revision` P0 T13'teki dizedir" (plan `:31`). O dize bugün gerçekten var:
  `app/main.py:381-384` → `f"{corpus_checksum[:12]}/{query_format.format_id}/{quantization}"`.
  **Ama bu dize retrieval REÇETESİNİ kodlamıyor:** pipeline (`hybrid`/`exhaustive`), BM25
  parametreleri, F5 kırpma, pencere-50, ASCII aksan katlaması — hiçbiri içinde değil.
  Oysa eşiğin bağlı olduğu eksen tam olarak bunlar: `config.py:116-127` aksan katlamasından
  SONRA eşik bandının **yeniden ölçülmek zorunda kaldığını** yazıyor. Yani bugünkü anahtarla
  versiyonlanmış bir threshold, reçete değiştiğinde geçersizleşmez — sessizce yanlış kalır.
  **Anahtar `index_revision` + `retrieval_pipeline` + reçete sürümü olarak genişletilmeli.**
  (Dizede `/` var; plan'ın "güvenli ad" ifadesi bunu zaten öngörüyor.)

### T7 — Kalibrasyon metrikleri + risk-coverage → **VALID**

Planın **tek tam geçerli** task'i. `brier`/`ece`/`auroc`/`risk_coverage`/`conformal_threshold`
saf numpy; birim testleri sentetik veriyle kurulu (plan `:402-404`); eksik hiçbir arayüz
tüketmiyor, kota yakmıyor, veri beklemiyor. **İlk iş olarak koşulabilir.**
(Çıktısının *anlamlı* olması veriye bağlı — ama kod ve testleri değil.)

### T8 — Selective answering entegrasyonu + güvenli fallback → **STALE**

- Kendi mekanizması sağlam: `/healthz` var (`main.py:522-536`), `"calibrator": "missing"`
  alanı küçük bir ekleme; degraded yolu zaten `base.py:52-55` + `main.py:604`.
- **Kırık:** T2 ve T6'dan devraldığı varsayımlar. Ayrıca plan `:427` "Flag'ler kapalıyken
  P1 davranışı" diyor — bugün "P1 davranışı" ölçülmüş biçimde ayırmayan bir eşiktir
  (yukarıda T2/1). G2.8'in metni bunu saklayamaz.
- `config.py`'de `evidence_verifier_enabled` / `selective_answering_enabled` flag'leri
  henüz yok (master §4 tablosu `:86-87` ikisini de `False` varsayılanla listeliyor) —
  eklenmeleri gerekiyor, bu kısım VALID.

### T9 — UI claim-citation + outcome telemetry + feedback + drift → **PARTIALLY DONE**

**Zaten var:**
- UI **status-güdümlü**: `index.html:730-744` `data.status`'tan dallanıyor
  (`answered`/`abstained`/`degraded`), `degraded` kendi sınıfını taşıyor (N1),
  mühür yalnız `abstained`'da. Plan'ın varsaydığı ABSTAIN_TEXT dize karşılaştırması
  **kaldırılmış** (K12 kapandı).
- `bg_abstain` etiketli Counter (`prom.py:106`), bugün `reason` değerleri
  `degraded`/`threshold` (`prom.py:189-191`) — plan'ın istediği `evidence`/`confidence`
  değerlerini eklemek mekanizma değişikliği değil.
- Recorder ALTER migrasyon deseni kurulu (`recorder.py:65`), yeni nullable alanlar
  için hazır.
- Citation çipleri `index.html:762-766` — ama **sayfa düzeyinde**, claim düzeyinde değil.

**Yok:** `abstain_reason` API gövdesinde yok (`main.py:630` yalnız `status` döndürüyor);
claim-claim gösterim; `/feedback` uç noktası; `RequestEvent`'te `claims_total/claims_supported/
verifier_ms/calibrated_conf/decision` alanları (`schema.py:29-64`); `scripts/drift_report.py`.

**Planın bilmediği borç, T9'a girmeli:** dürüst-ıska (`honest_miss`) bugün `answered`
sayılıyor (`yazim-degismezlik-ve-vitrin.md:40`) ve UI'da **hiçbir yüzeyi yok**
(`e2e-review.md:396`). P2'nin freni tam olarak bu sinyale yaslandığı için ayrı ve
birinci-sınıf bir durum olması gerekir.

### T10 — İnsan-kalibreli LLM-judge (PPI) → **BLOCKED-ON-DATA**

Kod bağımsız ve yazılabilir. **Ama planın kendi kuralı bağlayıcı** (`:498-500`): "judge
çıktısı TEK BAŞINA kapı kararı olamaz; yalnız `human_pairs` ≥ **30** örnekle kalibre
edilmiş haliyle rapora girer." Bugün insan-doğrulanmış satır sayısı **3**
(`retrieval_eval_v1.README.md:11`, c307/c308/c314). Ayrıca `verify_retrieval_eval.py --review` kuyruğu
çalışmıyor: yalnız `verification_status == "draft"` alıyor ve dağılım `{verified: 48}`
(`retrieval_eval_v1.README.md:88-99`, K8). Yani insan çifti **üretilemiyor bile**.

### T11 — Koşullu fine-tuning alt plan kapısı → **VALID**

Doküman görevi, bağımlılığı yok. Bugünden görünen: **kapı koşulu 2 ZATEN SAĞLANIYOR** —
"`paraphrase` veya `dogrudan-madde` diliminde Recall@5 < %80" (plan `:521-522`); ölçüm
paraphrase **2/7 = %28.6** (`2026-08-30-p1-hybrid-uretim-ve-round2.md:75`). Koşul 1
(G1+G2 PASS) sağlanmıyor, koşul 3 (bBSARD yerel doğrulama) bilinmiyor. Yani alt plan
tetiklenmesi yalnız 1 ve 3'e kalmış.

### T12 — P2 kapı raporu + final koşum → **STALE**

- Plan `:552`: `--bench data/bench/bench_v2.jsonl --split test`. **`bench_v2.jsonl` yok**
  (P1 T12 backlog, İNSAN kapılı).
- `--split test` bugün **cevaplanabilir 0 soru** getirir (bkz. §3).
- `bench answers` alt komutu yok.
- Plan `:557` "risk-coverage eğrisi (tablo + `docs/research/figures/` PNG)" —
  `docs/research/figures/` dizini yok (yalnız `findings/`, `evidence/`).

---

## 2. Kapı bazında hazırlık (G2.1-G2.8)

| Kapı | Ne gerekiyor | Bugün ne var | Somut boşluk |
|---|---|---|---|
| **G2.1** unanswerable false supported-answer ≤ %2 | Verifier (T1) + answer_eval (T3/T4) + yeterli sayıda cevaplanamaz soru | **n=5** cevaplanamaz; harness onları hiç koşmuyor (`harness.py:260`); 4'ü eşiği geçiyor → LLM çağrılıyor; freni yalnız LLM'in dürüst-ıskası tutuyor, o da `main.py:463`'te `"bulamadım" in text.lower()` alt-dize sezgisi (S35/D3, `config-coupling-audit.md:79`; K27 `e2e-review.md:179`) | **Aritmetik olarak ölçülemez.** n=5'te bir hata %20. 0 hata için %95 üst sınırı ≈ 3/n → **≤%2 iddiası ~150 test-split cevaplanamaz soru ister.** Ayrıca `honest_miss` tespiti sağlamlaştırılmadan (tek `HONEST_MISS_MARKER` + `tr_lower`) G2.1'in sayacı hem eksik hem fazla sayar |
| **G2.2** claim-level citation support precision ≥ %98 | Verifier + gold eşleşmesi | `gold_page_ids` 43 satırda var; auto-citation kaldırıldı | **Geçerlilik önkoşulu B14** (`gemini.py:55` konumsal `[Sk]` bağı) — düzeltilmeden ölçülen precision konumsal şansı ölçer. Ayrıca `article_id` kaynağı yok → eşleşme `page_id` üzerinden tanımlanmalı. n=43'te %98 ⇒ en fazla 1 hatalı atıf |
| **G2.3** risk-coverage eğrisi raporlu | T7 (VALID kod) + (prob, label) çiftleri | Kod bedava, veri yok | Yalnız veri hacmi: 48 noktada eğrinin her adımı %2.1 coverage — çözünürlük yok. §3'teki veri planı 127-167 test noktasına çıkarır |
| **G2.4** kalibrasyon ↔ test ayrıklığı, kanıt = split dosyası | Dolu `splits_v1.json` | `data/bench/splits_v1.json` = **`{"dev_docs": [], "test_docs": []}`**. `bench/dataset.py::question_split` boş kümede "dev"e düşüyor → **43/43 cevaplanabilir soru dev**; cevaplanamazlar (gold_doc_ids boş) sha256 ile bölünüyor → c004,c005 dev / c003,c006,c007 test. **Test split'inde cevaplanabilir soru YOK.** `bench run`'da `--split` bayrağı da yok (B7, `e2e-review.md:220-222`) | Split dosyasını law-grouped doldur + `--split` bayrağını ekle. Mekanizma (`question_split`) zaten yazılı, yalnız verisi yok |
| **G2.5** threshold'lar revision'a versiyonlu | `data/calibration/<revision>/` | `index_revision` dizesi var (`main.py:381-384`); dizin yok | Anahtar **yetersiz**: pipeline/reçete kodlanmıyor (T6'ya bakınız). Genişletilmeli |
| **G2.6** verifier geçmeyen yanıt kesin yanıt olarak gösterilmiyor | T2 + T8 + app testi | Yok | Tam boşluk (T1→T2→T8 zinciri) |
| **G2.7** auto-citation fallback kaldırıldı, test kanıtlı | — | ✅ **ZATEN SAĞLANIYOR** | Kanıt: `src/belge_gozu/answer/gemini.py:79-84` (fallback silinmiş; yorum sınıfı "P2 / G2.7" diye anıyor) + `tests/answer/test_gemini.py::test_no_marker_means_no_citation` (`a.citations == []`). Master §5 bu kapı için tam olarak `tests/answer/test_gemini.py`'yi ölçüm aracı gösteriyor (`master:125`). **P2 raporunda tek satırla kapatılabilir** |
| **G2.8** güvenli fallback tanımlı + testli | Doküman + test | Kısmen: degraded yolu var (`base.py:52-55` + `test_base.py::test_ask_degrades_gracefully_when_answerer_fails`), `/ask` `status="degraded"` (`main.py:604`), rate-limit/429 yolu var | Eksik: kalibratör-yok fallback'i + `/healthz` alanı. **Ve dürüstlük yükümlülüğü:** "flag'ler kapalı → P1 davranışı" cümlesi, P1 davranışının *ölçülmüş biçimde ayırmayan* bir eşik olduğunu (4/5 cevaplanamaz geçiyor) yazmadan kullanılamaz |

---

## 3. Veri önkoşulu analizi (asıl soru)

### 3.1 Bugünkü durum, sayılarla

- `data/bench/retrieval_eval_v1.jsonl`: **48 satır = 43 cevaplanabilir + 5 cevaplanamaz.**
- Cevaplanamazların dilimi: `korpus-disi` **3**, `anlamsiz-ood` **2**, `eksik-kanit` **0**.
- Doğrulama: **45 model-cross-check + 3 insan** (`retrieval_eval_v1.README.md:9-13`). Hiçbir sayı
  "insan-doğrulanmış benchmark üzerinde ölçüldü" diye sunulamaz (README §3.1).
- Split: `splits_v1.json` boş → cevaplanabilirlerin **tamamı dev**.
- 43 cevaplanabilir soru üzerinde **13 deney iterasyonu** koşuldu (autoresearch round 1-3);
  projenin kendi kaydı bunu "geliştirme-kümesi uyum riski… nihai doğrulama tutulmuş
  (held-out) set ister" diye yazıyor (`2026-08-30-p1-hybrid-uretim-ve-round2.md:88-91`).
- Korpus: **56 doküman / 4222 sayfa**; `data/manifest/v0_manifest.csv` doc_id kümesi ile
  `data/index-traincompat-int8/page_ids.json` doc kümesi **birebir aynı** (doğrulandı,
  fark 0).

### 3.2 Seçenekler

**(a) Cevaplanamaz seti, kurulu ajan-taslak + model-çapraz-kontrol hattıyla büyütmek.**
*Bu seçeneğin kritik avantajı:* `korpus-disi` etiketi **makinece doğrulanabilir**. Korpusun
tamamı 56 doc_id'lik bir manifest'tir; manifest'te olmayan bir mevzuata dair soru, tanımı
gereği korpus-dışıdır — bu etiket için ne sayfa okumak ne model gerekir.
Bu, `retrieval_eval_v1.README.md:44-51`'deki "doğrulayan model taslağı yazan modelle aynı aileden,
**korelasyonlu kör noktalar** mümkün" itirazından **tamamen kaçar**: etiket bir okuma
kararına değil, bir küme üyeliğine dayanır.
*Sınır (dürüstlük):* mekanik kontrol "adı geçen mevzuat korpusta yok"u kanıtlar,
"korpustaki hiçbir belge bu soruyu cevaplamıyor"u kanıtlamaz. Bu yüzden ikinci, yine ucuz
bir tur gerekir: üretim getiricisini koşup top-10'u modele göstermek ve **yalnız** "bu 10
sayfadan biri soruyu cevaplıyor mu?" diye sormak (altın sayfa okuma rejimi değil).
*Maliyet:* düşük. *Geçerlilik:* `korpus-disi` için yüksek, `eksik-kanit` için düşük.

**(b) P1-T12 tarzı benchmark v2 (insan kapılı).** *Geçerlilik:* en yüksek. *Maliyet:*
öncelik #19 tahmini 3-5 gün, tek geliştirici (`e2e-review.md:295`); üstelik araç bozuk —
`--review` kuyruğu boş ve `apply_decision` `verification_kind` yazmıyor
(`retrieval_eval_v1.README.md:88-99`). **Bugün kullanılamaz; P2'yi bloke eder.**

**(c) Mevcut 48 üzerinde leave-one-out + dürüst kırılganlık raporu.** *Maliyet:* ~sıfır.
*Geçerlilik:* G2.1 için **kabul edilemez** — 5 negatif üzerinde LOO, %20'lik adımlarla bir
oran üretir; ayrıca 43 cevaplanabilir soru fiilen bir *geliştirme* kümesidir, üzerinde hem
kalibre edip hem rapor etmek master §6'nın kuralını (`master:145-146`) ihlal eder. **Yalnız
G2.3'ün eğrisini "gösterim amaçlı, ölçüm değil" etiketiyle üretmek için kullanılabilir.**

### 3.3 Önerilen asgari veri planı

**İlke:** cevaplanamaz etiketleri ucuz, cevaplanabilir etiketleri pahalıdır. G2.1/G2.3/G2.4
tam olarak cevaplanamaz tarafa ihtiyaç duyar. O yüzden **yalnız cevaplanamaz tarafı
ölçekle**; cevaplanabilir 43 olduğu gibi kalsın ve dürüstçe "dev seti" diye anılsın.

**`data/bench/abstention_eval_v1.jsonl` — 300 satır:**

| Dilim | n | Doğrulama rejimi | İnsan emeği |
|---|---|---|---|
| `korpus-disi` | **200** | **Mekanik**: `data/manifest/v0_manifest.csv`'ye karşı script kontrolü (yeni `verification_kind: "mechanical"` — insan ve model-cross-check'ten dürüstçe ayrı) + üretim top-10'u üzerinde dar model turu | ~0 |
| `anlamsiz-ood` | **60** | Ajan taslak + model-cross-check + insan toplu göz gezdirme (etiketler kendinden aşikâr, sayfa okuma yok) | ~30 dk |
| `eksik-kanit` | **40** | **Tam rejim**: ajan taslak + bağımsız model-cross-check + **40/40 insan doğrulaması**. B28'in sıfır-soru dilimini kapatır | ~yarım gün |

**Split şeması (G2.4'ü çözer):**
1. `splits_v1.json`'u law-grouped doldur: 56 doc_id'nin **22'si `test_docs`**, 34'ü
   `dev_docs` → 43 cevaplanabilirin ~17'si test, ~26'sı dev. (`question_split`
   mekanizması `bench/dataset.py`'de zaten yazılı, yalnız verisi yok.)
2. Cevaplanamazlara **açık `split` alanı** ekle (gold_doc_ids boş olduğu için bugünkü
   sha256 fallback'i opak); 50/50 → **150 test / 150 dev**.
3. `bench run` ve yeni `bench answers`'a `--split` bayrağı ekle (bugün yok).

**Sonuç:** test split = ~17 cevaplanabilir + **150 cevaplanamaz**; dev split = ~26 + 150.

**Kapı aritmetiği bu planla ne oluyor:**
- **G2.1:** 0/150 hata ⇒ %95 tek-yanlı üst sınır ≈ 3/150 = **%2.0** → "≤%2" iddiası tam
  olarak *ancak* sıfır hatayla savunulabilir hale gelir. **Tek bir hata bile (1/150 = %0.67,
  üst sınır %3.7) "PASS" olarak değil, aralık olarak raporlanmalıdır.** 200 korpus-dışı
  sorunun script-etiketli olması bu n'i ulaşılabilir kılan tek şeydir.
- **G2.3:** risk-coverage 167 test noktası üzerinde çizilir (48 yerine).
- **G2.4:** boş olmayan bir split dosyası denetlenebilir hale gelir.
- **G2.2 / answer accuracy:** 43 cevaplanabilir üzerinde kalır → bu sayılar **dev-set
  sayılarıdır** ve raporda öyle etiketlenmelidir; K23'ün dejenere bootstrap CI uyarısı
  (`e2e-review.md:168-170`) dilim bazında geçerliliğini korur.

**Sınıf dengesizliği notu (planda yok):** 43 pozitif / 300 negatif ile `selective_metrics`'in
tek bir `coverage` sayısı anlamsızlaşır — **sınıf başına** raporlanmalı; kalibratör fit'i
(T5/T6) sınıf ağırlığı kullanmalı veya dengeli bir dev alt-örneklemi üzerinde kurulup
tam dev üzerinde raporlanmalı.

**Yan kazanç:** `eksik-kanit`'in 40 insan-doğrulanmış satırı, T10'un ≥30 `human_pairs`
şartını da karşılar — T10 bu adımdan sonra BLOCKED olmaktan çıkar.

**Önkoşul araç düzeltmesi:** `scripts/verify_retrieval_eval.py`'nin iki bilinen kusuru
(`retrieval_eval_v1.README.md:88-99`) düzeltilmeden insan doğrulaması **kaydedilemez** — kuyruk
`model-cross-check` satırlarını da kapsamalı, `apply_decision` `verification_kind`'ı
`"human"` yapmalı.

---

## 4. Gemini bütçe planı

### 4.1 Hangi task'ler LLM çağırıyor

| Task | Çağrı | Not |
|---|---|---|
| T1 `GeminiVerifier.verify` | soru başına **1** | plan `:163` açıkça "tek API çağrısı (claim başına değil)" — doğru tasarım |
| T2 `decide_verdicts` retry dalı | +1 answerer, +1 verifier | yalnız `insufficient` varken, bir kez (plan `:195-197`) |
| T4 `run_answer_eval` | soru başına 1 answerer | **planın hesaba katmadığı asıl çarpan** |
| T5 `collect_calibration.py` | dev akışının tamamı | önbellek isabetliyse bedava |
| T10 `GeminiJudge` | cevaplanabilir-answered başına 1 | |
| T12 final test koşumu | test split'in tamamı, bir kez | |

**Planın gözden kaçırdığı nokta:** plan bütçeyi *verifier* üzerinden tartışıyor
(`master:191`, plan `:26-28`), ama **answerer da bir Gemini çağrısıdır** ve cevaplanamaz
soruların ~%80'i eşiği geçtiği için (4/5 ölçümü) 300 cevaplanamaz soru **240 answerer
çağrısı** demektir. Ölçek burada.

### 4.2 Önerilen veri boyutlarında çağrı tahmini

Geçiş oranları ölçümden: cevaplanabilir %97.7 (42/43), cevaplanamaz %80 (4/5).

| Koşum | Soru | Answerer | Verifier | Judge | Toplam |
|---|---|---|---|---|---|
| Dev pass (26 cev. + 150 cev.siz) | 176 | 145 | 145 | 26 | **316** |
| + retry dalı (~%20) | | +29 | +29 | | **+58** |
| Test pass (17 + 150) | 167 | 137 | 137 | 17 | **291** |
| + retry | | +27 | +27 | | **+54** |
| H1 (verifier aç/kapa) ikinci kol | | ~0 (önbellek) | 0 | 0 | ~0 |
| **Toplam taze çağrı** | | | | | **≈ 720** |

Token/maliyet (ölçülen: `tokens_in≈5671, tokens_out≈66` per /ask, `e2e-review.md:35`):
answerer ≈ $0.0006, metin-paketli verifier ≈ $0.0011 (5 sayfa × ~1.7k token), judge ≈ $0.0001
→ **toplam ≈ $0.7**. **Para kısıt değil; oran kısıtı kısıt.**
⚠️ Fiyat sabitleri (`config.py:155-156`) **doğrulanmamış ve gemini-2.0-flash dönemine ait**,
kullanılan model `gemini-3.6-flash` (B37, `e2e-review.md:266`) — bu dolar rakamı yalnız
mertebe göstergesidir.

### 4.3 Kota gerçekliği — planı bugün kıran sayı

Master §9 (`:191`) ve plan `:26` "≈20 çağrı/gün" varsayıyor. **720 taze çağrı ⇒ 36 gün.**
Bu, P2'yi fiilen imkânsız kılar. Üç çıkış, sırayla:

1. **Kotayı yeniden doğrula (İLK İŞ, dakikalar).** "≈20/gün" 2026-08-26 planlama
   varsayımıdır; `gemini-3.6-flash` için gerçek serbest-katman limiti büyük olasılıkla çok
   daha yüksektir. Bütün takvim bu tek sayıya bağlı: 20/gün ise 36 gün, 200/gün ise 4 gün.
   B37 zaten fiyat sabitlerinin bu model için bayat olduğunu gösteriyor.
2. **Verifier'ı dürüst-ıskada kısa devre yap (tasarım kazancı, planda yok).** Yanıt
   `segment_claims` sonrası 0 claim üretiyorsa veya dürüst-ıska ise verifier'a hiç gitme —
   verdict tanımı gereği "insufficient". 150 cevaplanamaz sorunun ~%90'ı dürüst-ıska ise
   **pass başına ~108 verifier çağrısı** düşer.
3. **Kota gerçekten darsa kapsamı daralt:** dev kalibrasyonunu 60 soruluk dengeli bir
   alt-örneklemle koş, test split'i tam koş → ~400 çağrı.

### 4.4 Önbellek tasarımı — nerede yaşamalı

Plan `:104-106`: `data/cache/verifier/<sha256>.json`, anahtar =
sha256(soru + claim metinleri + pack unit kimlikleri). `data/cache/` **henüz yok**.

**Onaylanan:** dosya-tabanlı sha256 JSON önbelleği doğru araç — koşumlar güne bölünebilir,
tekrar koşumlar bedava, `.gitignore`'a girer, CI'da stub kullanılır.

**Düzeltilmesi gereken iki kusur:**
1. **Anahtar model kimliğini içermiyor.** `gemini_model` bir config anahtarı
   (`config.py:66`) ve zaten bir kez değişti (2.0-flash → 3.6-flash). Model değişince bayat
   verdict'ler sessizce yeniden kullanılır.
2. **Anahtar istem/şema sürümünü içermiyor.** `SYSTEM` (`gemini.py:9-14`) veya verifier
   istemi değiştiğinde aynı sorun. `prompt_version` sabiti anahtara girmeli.

**Önerilen yer:** `src/belge_gozu/answer/cache.py` → `JsonFileCache(root)` +
`cache_key(*, kind, model, prompt_version, payload)`. Tüketiciler:
`GeminiVerifier` (`data/cache/verifier/`), `GeminiJudge` (`data/cache/judge/`) ve
**yalnız-bench** bir `CachedAnswerer` sarmalayıcısı (`data/cache/answerer/`).
`CachedAnswerer` **`bench/` altında** durmalı, `answer/` altında değil: servis yolunda bir
yanıt önbelleği doğruluk tehlikesidir (servis edilen yanıt bir replay olmamalı) — katman
ayrımı bu niyeti yapısal hale getirir.

**Aynı turda kapatılacak:** B13 — Gemini çağrısında timeout/retry yok
(`e2e-review.md:240`). 300 soruluk bir toplu koşumun hata sınırlaması yok; batch runner'a
çağrı başına timeout + sınırlı retry gerekiyor.

---

## 5. Önerilen yürütme sırası (P2-lite)

### Faz 0 — kota yakmayan, kapıları açan işler (paralel)

| # | İş | Plan karşılığı | Gerekçe |
|---|---|---|---|
| **1** | **T7 aynen** — `bench/calibration_metrics.py` | T7 **VALID** | Tek tam geçerli task; saf numpy, sentetik testler, sıfır kota, sıfır veri. G2.3'ün aletini hemen verir |
| **2** | **Veri + split (planda YOK, yeni task)** — `abstention_eval_v1.jsonl` (300), `splits_v1.json` law-grouped, `--split` bayrağı, retrieval_eval-kompozisyon regresyon testi (B28/K7), `verify_retrieval_eval.py --review` iki kusurunun düzeltilmesi | — | **Uzun kutup.** G2.1/G2.3/G2.4 bunsuz anlamsız. 1 ile paralel başlat |
| **3** | **B14 düzeltmesi** — `gemini.py:55` `contents=[*parts, prompt]` yerine her görüntünün ÖNÜNE `[Sk] <doc>, sayfa <n>` metin parçası; + test | — | **T3/T4'ten ÖNCE olmak zorunda**: aksi halde G2.2 konumsal şansı ölçer (öncelik #18). Saatler |
| **4** | **Kota doğrulaması** — `gemini-3.6-flash` gerçek limitini ölç, `config.py:155-156` fiyat sabitlerini güncelle (B37) | plan `:26` | Takvimin tamamı bu sayıya bağlı (§4.3) |

### Faz 1 — LLM'li, önbellekli

| # | İş | Değişiklik |
|---|---|---|
| **5** | **T1-modified** | `segment_claims` aynen. `EvidenceUnit`/`EvidencePack` **yerelde** tanımlanır (`list[PageHit]` + `page_texts.parquet`), `retrieval/evidence.py`'den import EDİLMEZ. `CitationRef.article_id` alanı ileri-uyum için kalır ama `None`; G2.2 eşleşmesi `page_id` üzerinden. `GeminiClient.generate_structured` eklenir. `answer/cache.py` anahtarına model + prompt_version girer (§4.4). Verifier dürüst-ıskada kısa devre |
| **6** | **T3b** | Yalnız `bench/answer_eval.py` metrikleri — T3a ve G2.7 zaten kapalı |
| **7** | **T2-modified** | İki kapı. `abstain_reason` **mevcut `status` alanına** bağlanır (`main.py:604-608`), paralel kanal icat edilmez. `Answer` alanları korunur. G2.8 metni "flag kapalı → ayırmayan eşik" gerçeğini yazar. `honest_miss` tek `HONEST_MISS_MARKER` + `tr_lower` ile sağlamlaştırılır (S35/D3) — G2.1'in sayacı buna bağlı |
| **8** | **T4-modified** | `bench answers` + `--split`. Dilim kırılımı adım 2'nin doldurduğu dilimlere göre; `harness.py:260`'ın cevaplanamaz atlaması bu yolda geçerli değil. `selective_metrics` **sınıf başına** raporlanır |
| **9** | **T5+T6-modified** | `FEATURE_ORDER` yeniden türetilir: `bm25_top1(served)`, `margin_1_2`, `law_match←routed_docs`, `visual_top1`, `len(routed_docs)`, `verifier_support_ratio`. `exact_match`/`article_match` **düşürülür** (kaynak yok), `channel_agreement` dejenere olduğu için ya düşürülür ya "ölçüldü, bilgi taşımıyor" diye raporlanır. `calibration_dir` anahtarı `index_revision` + `retrieval_pipeline` + reçete sürümü |
| **10** | **T8 → T9-modified → T12-modified** | T9: UI zaten status-güdümlü — **genişlet, yeniden kurma**; `abstain_reason` + claim çipleri + dürüst-ıska yüzeyi + `/feedback` + yeni prom `reason` etiketleri (`prom.py:106` mekanizması hazır). T12: `bench_v2.jsonl` yerine `retrieval_eval_v1 + abstention_eval_v1`, `docs/research/figures/` dizini oluşturulur |
| **11** | **T10** (artık mümkün) → **T11** (doküman) | T10 adım 2'nin 40 insan çiftinden sonra ≥30 şartını karşılar; düşük öncelik. T11'de koşul 2'nin **zaten sağlandığı** (paraphrase %28.6) yazılır |

### Bugünün ölçümünün DOĞRUDAN ÇELİŞTİĞİ plan ifadeleri

1. **plan `:30-31`** "final sayılar = test split" — test split'te **0 cevaplanabilir soru**
   var (`splits_v1.json` boş + `question_split` fallback'i).
2. **plan `:187`** `min_score_threshold`'ın flag-kapalı "eski davranış" olarak korunması —
   o kapı **ölçülmüş biçimde ayırmıyor** (4/5 cevaplanamaz geçiyor; `xfail(strict)` kilidi).
   "Güvenli fallback" dili bu ölçümü saklayamaz.
3. **plan `:300`** `channel_agreement`'ın bilgi taşıdığı varsayımı — görsel kanalın @5
   benzersiz katkısı **ölçülmüş SIFIR**, üç füzyon biçimi reddedildi (`hybrid.py:10-15`).
4. **plan `:552`** `data/bench/bench_v2.jsonl` — **yok** (P1 T12 backlog, insan kapılı).
5. **plan `:521-522`** FT kapısı koşul 2'nin gelecekte değerlendirileceği varsayımı —
   **bugün zaten doğru** (paraphrase 2/7 = %28.6 < %80).
6. **plan `:26`** "≈20 çağrı/gün" — önerilen veri planıyla 36 günlük bir takvim üretir;
   sayı yeniden doğrulanmadan hiçbir runbook adımı planlanamaz.

---

## 6. Formal not: G1 önkoşulu sağlanmadı — `p2-gate.md` ne açıklamak zorunda

**Gerçek:** master §2 (`:48`) P2'nin giriş koşulunu "**G1 PASS commit'i**" olarak tanımlar
ve P1'in çıkış koşulu `docs/research/findings/*p1-gate.md*`tir (`master:47`).
**Bu dosya yok** — `docs/research/findings/` dizini denetlendi: yalnız `2026-08-27-p0-gate.md`
var. P1 ruling R23 ile daraltıldı ve gate koşumu (P1 T13) backlog'da kaldı
(P1 planı `:22`). Yani **G1 hiç adjudike edilmedi.** P2, kullanıcı direktifiyle başlıyor.

Dürüst bir `p2-gate.md` şunları **açıkça** yazmak zorundadır:

1. **Sapmanın kendisi.** P2, sağlanmış bir G1 üzerine değil, **açık kullanıcı direktifi**
   üzerine başlatıldı. Bu, master §1'in kapı kuralından (`:38-40`: "bir sonraki fazın hiçbir
   default-açık entegrasyonu önceki kapı raporu commit'lenmeden yapılamaz") bir sapmadır ve
   bir gözden kaçma değil, kayda geçmiş bir karardır.
2. **Hangi G1 ölçütlerinin bilinen biçimde sağlanmadığı, sayılarıyla.**
   - **G1.2** (kritik dilimlerde R@50 ≥ %90 her biri): `paraphrase` dilimi **2/7 = %28.6**
     (R@5; `2026-08-30-p1-hybrid-uretim-ve-round2.md:75`) — bu dilim ölçülen haliyle
     kapıyı geçmiyor.
   - **G1.3** (reranker kazancı, bootstrap CI alt sınırı > 0): reranker **yok** (P1 T10
     backlog) → ölçülmedi.
   - **G1.1** (candidate-union R@50): bugünkü API ile hesaplanamıyor (B9,
     `e2e-review.md:228-230`).
   - **G1.5 / G1.6 / G1.7**: ölçütleri "rapor var → p1-gate.md" olduğu için, rapor
     olmadığı sürece **tanım gereği** sağlanamaz.
   - Sağlanan: **G1.4** (Sorgu A dayanak sayfası top-5 — uzun sorgu gold sırası 664→**2**).
3. **Bunun P2 sayılarına ne yaptığı.** Kalibre edilmiş bir güven, altındaki getirim
   katmanının zayıflığını "düşük güven" olarak kodlar. Bu davranışsal olarak dürüsttür,
   ama **P2'nin coverage rakamı onaylanmamış bir P1'in fonksiyonudur**: en kötü dilimi
   %28.6 recall'da olan bir hattın üstüne kurulan coverage, retrieval iyileştiğinde
   yeniden ölçülmek zorundadır. `paraphrase` dilimindeki abstain'ler bir kalibrasyon
   başarısı değil, bir retrieval borcudur.
4. **Kapı kuralının hâlâ koruduğu şey.** Master §1'in koruması yalnızca P2 flag'leri
   (`evidence_verifier_enabled`, `selective_answering_enabled`) **default-KAPALI** kaldığı
   sürece geçerlidir (master §4, `:86-87`). Rapor flag varsayılanlarını açıkça yazmalı ve
   G1+G2 raporlanana kadar default-açık entegrasyon yapılmadığını beyan etmelidir.
5. **Benchmark'ın kendisi.** Tüm P2 sayıları 48 satırın **45'i model-cross-check /
   3'ü insan** olan bir set (ve önerilen `abstention_eval_v1`'in mekanik+model etiketleri) üzerinde
   ölçülür. Hiçbir P2 rakamı "insan-doğrulanmış benchmark üzerinde ölçüldü" diye
   sunulamaz (`retrieval_eval_v1.README.md:38-43`). `verification_kind` başına ayrı sayı
   basılmalıdır — `mechanical` / `model-cross-check` / `human` üç ayrı sütun.
6. **G2.1'in aritmetik dürüstlüğü.** "≤%2" bir nokta tahmini olarak değil, gözlenen oran +
   %95 üst sınır (Clopper-Pearson/rule-of-three) olarak raporlanmalı; n=150 test
   cevaplanamazda **sıfır hata bile ancak %2.0 üst sınır** verir. Tek bir hata kapıyı
   aritmetik olarak PASS yapamaz.

---

## Ek: denetimde doğrulanan mevcut/eksik envanteri

**Var:** `answer/base.py` (AskService, tek kapı), `answer/gemini.py` (fallback'siz),
`retrieval/{core,hybrid,text,types}.py`, `bench/{dataset,harness,metrics,oracle}.py`,
`telemetry/{schema,recorder,prom,collect,export}.py`, `app/main.py` (status-güdümlü `/ask`,
`/healthz`, `/metrics`, `/stats`, rate limit), `app/static/index.html` (status-güdümlü UI),
`data/bench/retrieval_eval_v1.jsonl` (48), `data/manifest/v0_manifest.csv` (56 doc),
`data/index-traincompat-int8/` (4222 sayfa + `page_texts.parquet`).

**Yok:** `retrieval/evidence.py`, `retrieval/query.py`, `retrieval/rerank.py`,
`retrieval/dense.py`, `retrieval/fusion.py`, `corpus/articles.py`, `answer/verify.py`,
`answer/calibrate.py`, `answer/cache.py`, `bench/answer_eval.py`,
`bench/calibration_metrics.py`, `bench/judge.py`, `scripts/collect_calibration.py`,
`scripts/drift_report.py`, `data/bench/bench_v2.jsonl`, `data/calibration/`, `data/cache/`,
`docs/research/figures/`, `docs/research/findings/*p1-gate*`,
`pyproject.toml` `eval` extra (`scikit-learn`), `config.py` `evidence_verifier_enabled` /
`selective_answering_enabled`, `cli.py` `bench answers` / `calibrate fit` / `--split`.

**Boş:** `data/bench/splits_v1.json` = `{"dev_docs": [], "test_docs": []}`.
