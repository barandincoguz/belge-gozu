# Belge-Gözü — Proje Durum Envanteri (plan/spec ↔ kod gerçeklik denetimi)

**Tarih:** 2026-08-31 · **Dal:** `feat/p0-retrieval-correctness` · **HEAD:** `21c702c`
**Yöntem:** READ-ONLY. Her satır dosyadan/artefakttan doğrulandı; hiçbir sayı hafızadan
veya önceki rapordan körlemesine alınmadı. Ajan iddiasına dayanan noktalar açıkça
"(ajan raporu)" diye işaretlidir.

**Bu koşumda bağımsız olarak yeniden üretilenler:** `uv run pytest -q -m "not slow"` →
**666 passed, 6 deselected** (5,59 sn); `make lint` → **All checks passed / 111 files
formatted / pyright 0 errors**; `git remote` → **0 remote**; `git rev-list --count
main..HEAD` → **85**; `git rev-list --count HEAD` → **147**; üretim manifest'i
(`data/index-traincompat-int8/manifest.json`) alan alan okundu; `page_texts.parquet`
4222 satır / 1 boş metin; `data/bench/results/*.json` dilim-bazlı metrikler yeniden
hesaplatıldı; `data/calibration/**/calibrator.json` künyeleri okundu.

---

## 0. Tek bakışta

| Boyut | Durum |
|---|---|
| Kapı raporları | Yalnız **G0** var (`2026-08-27-p0-gate.md`, KOŞULLU GEÇTİ 8/9 → sonradan 9/9). **G1 raporu YOK**, **G2 raporu YOK**. |
| Üretim hattı | `retrieval_pipeline="hybrid"` (BM25 metin kanalı sıralar; görsel kanal koşar ama sıralamaz), `index_dir=data/index-traincompat-int8` |
| Test/lint | 666 test yeşil, 6 slow deselect; ruff + pyright temiz |
| Yayın | **GitHub'da değil (0 remote)**, HF Space **hiç oluşturulmadı** (PRO gerekiyor), Docker imajı **hiç build/deploy doğrulanmadı** |
| Benchmark | retrieval_eval 48 (3 insan + 45 model-cross-check) + abstention_eval_v1 330; **bench_v2 (120+30) YOK** |
| Dürüstlük | README + kapı raporları + veri README'leri örnek düzeyde dürüst (3/48 insan sınırını üç ayrı yerde tekrarlıyor). **İSTİSNA: canlı UI `index.html:454` "43 soruluk insan-doğrulamalı retrieval_eval" diyor — YANLIŞ** ve iç incelemede KRİTİK işaretlenmiş olmasına rağmen düzeltilmemiş |

---

## 1. Kapı durumu (G0 / G1 / G2)

### 1.1 G0 — rapor VAR

**Rapor:** `docs/research/findings/2026-08-27-p0-gate.md` (453 satır) ·
**Baseline eki:** `docs/research/findings/2026-08-27-p0-baseline.md` (945 satır)

| Kapı | Ölçüt | Hüküm | Kanıt (bugün doğrulandı) |
|---|---|---|---|
| G0.1 | `k4721:4` korpus kapsamı | **PASS** | `data/index-traincompat-int8/manifest.json` → `corpus_checksum=133444d8c235fb45…`, `n_pages=4222`; p0-gate §3 üç ayrı doğrulama; `tests/retrieval/test_semantic_retrieval_eval.py::test_retrieval_eval_gold_pages_covered` |
| G0.2 | Sorgu A + B kalıcı regression setinde | **PASS** | `tests/retrieval/test_semantic_retrieval_eval.py` (6 slow test); `tests/retrieval/retrieval_eval_expectations.json` cırcır: `hybrid.long_query_gold_rank_max=2`, `exhaustive=664` |
| G0.3 | Candidate generator gold Recall@candidate ≥ %98 | **PASS (tanım gereği)** | Exhaustive yol eleme yapmaz; Stage-1 ölçüldü ve **reddedildi** (c=200'de survival %9,3 — p0-gate Ek/B2) |
| G0.4 | Exhaustive binary + native float oracle karşılaştırılabilir | **PASS (araç sapması notlu)** | `bench oracle` (cli.py:606) üç kol; **ancak `EvalReport`'ta `oracle_gap` alanı YOK** — `harness.py:43-52` alanları: run_id, git_commit, index_manifest, config, missing_gold_pages, overall, per_slice, per_doc, diagnostics. Sapma p0-gate §4/3'te kayıtlı, **bugün hâlâ açık** |
| G0.5 | İndeks/processor uyumsuzluğu fail-fast | **PASS** | `index/compat.py` + `app/main.py:270-333` (6 eksen); `tests/app/test_compat.py` |
| G0.6 | Padding satırları skorlanmıyor | **PASS** | Üretim manifest'i `mask_policy="drop-padding"`, `n_tokens=3.759.994`; p0-gate §3 bayt taraması: v0'da 3.960 all-zero, yeni indekslerde 0 |
| G0.7 | Kuantizasyon kaybı sayılandırıldı | **PASS** | int8 = float16 her k'da; 1-bit R@20'de −7,0 puan. Karar **uygulandı** (P0'da ertelenmişti): üretim int8 (`config.py:60`, commit `b790f6c`) |
| G0.8 | Sorgu B üretim hattında top-5 | **PASS** | p0-gate §3: canlı `/search` → `k4721:4` rank 4; hibritte cırcır rank ≤ 2 |
| G0.9 | Baseline raporu | **PASS** (P0 kapanışında KOŞULLU idi) | `docs/research/findings/2026-08-27-p0-baseline.md` yazıldı → eksiklik kapandı |

**G0'ın kapanmamış artıkları (p0-gate §4'ten, bugün doğrulandı):**

| # | Konu | Bugünkü durum |
|---|---|---|
| 3 | `EvalReport.oracle_gap` alanı | **HÂLÂ AÇIK** — `bench/harness.py:43-52`'de yok |
| 4 | compat `model_name`/`mask_policy`/`corpus_checksum` dallarının birim testi | ajan raporu; bu denetimde tek tek doğrulanmadı |
| 8 | Gecikme ölçümlerinin ham artefaktı | **KAPANDI** — `data/bench/results/latency-by-representation.json` mevcut |
| — | RetrievalEval insan doğrulaması | **HÂLÂ AÇIK** — 3/48 insan (bkz. §6) |

### 1.2 G1 — rapor YOK; kriterlerin bir kısmı fiilen ÖLÇÜLDÜ ama adjudike EDİLMEDİ

**`docs/research/findings/*p1-gate*` dosyası yoktur.** Sebep: Ruling **R23** (master §11,
2026-08-29) P1 kapsamını "ölçülmüş reçetenin üretimleştirilmesi"ne daralttı
(T1kısmi + T6 + T8-revize); F1/E1/F2/F3/G1 ablasyon matrisi hiç koşulmadı.

R23 sonrası G1 satırlarının gerçek durumu — **sayılar bu denetimde
`data/bench/results/20260830-1611-6d5b345-hybrid.json`'dan yeniden hesaplandı**
(retrieval_eval, n=43 answerable, `only_verified=true`, pipeline=hybrid, int8):

| Kapı | Ölçüt (master §5) | Bugünkü hüküm | Ölçülen sayı / gerekçe |
|---|---|---|---|
| **G1.1** | Candidate-union Recall@50 ≥ %95 (verified full bench) | **ÖLÇÜLDÜ → FAIL** (ama resmî değil) | **R@50 = 0,9302** (< 0,95). İki katmanlı çekince: (a) "candidate-union" diye bir şey yok — tek kanal (BM25) sıralıyor, yani ölçülen sayı union değil **tek-kanal** recall'ü; (b) full bench (`bench_v2`) yok, ölçüm retrieval_eval n=43'te. **Not:** aynı metrik ASCII-katlama ÖNCESİ **0,9535** idi (`20260829-2115-3a031ca-hybrid.json`) — yani exp12 R@5'i 0,8256→0,8488'e çıkarırken R@50'yi 0,9535→0,9302'ye düşürdü. Bu takas hiçbir yerde adjudike edilmemiş. |
| **G1.2** | Kritik 4 dilimde Recall@50 ≥ %90 | **ÖLÇÜLDÜ → 3/4 PASS, 1 FAIL** | `dogrudan-madde` 1,0000 (n=13) · `madde-numarali` 1,0000 (n=6) · `ayni-kanun-hard-negative` 1,0000 (n=5) · **`paraphrase` 0,5714 (n=7) → FAIL**. Paraphrase R@5 yalnız 0,2857, MRR 0,104 — sözcüksel tavan (dense kanal yokluğunun doğrudan bedeli). |
| **G1.3** | Reranker kazancı, bootstrap CI alt sınırı > 0 | **ANLAMSIZ / ÖLÇÜLMEDİ** | Reranker yok (`retrieval/rerank.py` YOK, `config.py`'de `rerank_enabled` YOK). Ön koşul (R@20 = 0,9302) sağlanıyor, yani reranker havuzu recall-kapısını **geçer** — kapı ölçülebilir hale geldi ama katman yazılmadı. |
| **G1.4** | Sorgu A dayanak sayfası final top-5 (zorunlu regression) | **FİİLEN KARŞILANDI, RAPORA YAZILMADI** | `tests/retrieval/retrieval_eval_expectations.json` → `hybrid.long_query_gold_rank_max = 2` (ölçüm 2026-08-30, üretim indeksi + hibrit). Yani rank 2 ≤ 5. P0 devrindeki D3 (1221 → 664 → **2**) böylece kapandı. |
| **G1.5** | Hybrid vs visual-only delta + latency/memory raporlu | **KISMEN ÖLÇÜLDÜ, RAPOR YOK** | Delta ölçülü: visual-only (exhaustive/1-bit) R@5 **0,1163** vs hibrit **0,8488** (`verified-production-exhaustive.json` vs `20260830-1611-…-hybrid.json`); gecikme: görsel kanal ~0,24 sn/sorgu (int8, CPU), BM25 ms mertebesi (`retrieval/hybrid.py:20-21`); `latency-by-representation.json` artefaktı var. Bir p1-gate raporunda toplanmadı. |
| **G1.6** | Kazanç göstermeyen katman default kapalı | **FİİLEN UYULDU** | RRF üç biçimde ölçüldü ve **reddedildi** (0,674→0,395 küresel; 0,8372→0,5349 pencere-içi; `research/journal.md` #2/#6/#10) → üretime hiç girmedi. Görsel kanal koşuyor ama **sıralamaya girmiyor** (`retrieval/hybrid.py:249-266`). |
| **G1.7** | HF Space bütçeleri (indeks boyutu + peak RAM + p50/p95) | **ÖLÇÜLMEDİ** | Space hiç oluşturulmadı; cold-start / peak RAM ölçümü yok. |

**Kritik usul bulgusu:** P1 planının kendi kuralı — *"`retrieval_mode` default'u ancak
TÜM G1 satırları PASS ise `hybrid-production` olur"* — **çiğnendi**: `config.py:121`
bugün `retrieval_pipeline="hybrid"` ile üretimde. Kararın ölçüm gerekçesi güçlü
(R@5 0,2326 → 0,8605), ama resmî kapı adjudikasyonu yapılmadı. Projenin kendi belgesi
(`agent-reports/2026-08-30-p2-reality-audit.md` §6, ajan raporu) aynı sonuca bağımsız
olarak varmış ve "P2 kullanıcı talimatıyla başladı, G1 PASS ile değil" (R29) diye
kayda geçmiş — `docs/research/findings/2026-08-30-p2-baslangic.md:6`.

### 1.3 G2 — rapor YOK, koşum YOK

| Kapı | Ölçüt | Bugünkü hüküm | Kanıt |
|---|---|---|---|
| G2.1 | Unanswerable'da false supported-answer ≤ %2 | **ÖLÇÜLMEDİ (kapı anlamında)** | Yalnız **dev** ölçümü var: `data/calibration/…7b56eeeb7327/calibrator.json` → `false_answer_on_unanswerable = {rate: 0.0, n: 159, upper_bound_95: 0.0187, method: clopper_pearson}` — artefaktın kendisi *"DEV ÖLÇÜMÜ — G2.1 KAPI SAYISI DEĞİLDİR"* notunu taşıyor (eşik dev'de seçilip dev'de ölçüldü → iyimser) |
| G2.2 | Claim-level citation support precision ≥ %98 | **ÖLÇÜLMEDİ** | `bench/answer_eval.py` yok; citation precision hesaplayan koşum yok |
| G2.3 | Risk-coverage eğrisi raporlu | **ÜRETİLDİ, RAPORLANMADI** | `calibrator.json → kunye.dev_metrics.risk_coverage` (185 nokta), `tau=0,5037`, `coverage_at_tau=0,0216` (**%2,2**), `risk_at_tau=0,0`, `selective_accuracy_at_tau=1,0`, AUROC 0,7817, Brier 0,0859, ECE 0,0341. Figür/rapor yok. |
| G2.4 | Kalibrasyon ↔ test ayrıklığı | **KISMEN** | `data/bench/splits_v1.json` law-grouped; kalibratör künyesi `split="dev"` taşıyor. Test yakası hiç kullanılmadı (iyi), ama final test koşumu da yapılmadı. |
| G2.5 | Threshold'lar revision'a versiyonlu | **PASS** | `data/calibration/133444d8c235-train-compat-v1-int8__hybrid__<recipe_fp>/calibrator.json` — korpus+format+quant+pipeline+reçete parmakizi anahtarda |
| G2.6 | Verifier geçmeyen yanıt kesin yanıt olarak gösterilmiyor | **KOD VAR, VARSAYILAN KAPALI** | `answer/base.py:229-245` (`_apply_gate2` → `VERIFIER_DEMOTE_TEXT` + `abstained=True`); ama `config.py:216` `gate_verifier: bool = False` → bugün servis edilen yanıtlar **doğrulanmadan** sunuluyor |
| G2.7 | Auto-citation fallback kaldırıldı | **PASS** | `answer/gemini.py:661-664` (*"Atıf YOKSA atıf yok. Eskiden burada top-1 sayfayı otomatik atıf olarak ekleyen bir fallback vardı"*); commit `ef3971d`; test `tests/answer/test_gemini.py:67 test_no_marker_means_no_citation` |
| G2.8 | Güvenli fallback tanımlı + testli | **KISMEN** | Gate'ler `None` iken P1 davranışına dönüş `answer/base.py:146` ile sözleşmeleştirilmiş; `gate_calibrated` için artefakt yoksa fail-fast (config.py:206-209). T8'in tam entegrasyonu yapılmadı. |

---

## 2. Görev tablosu

### 2.1 P0 — Task 1..15 (plan: `2026-08-26-belge-gozu-p0-retrieval-correctness.md`)

| # | Görev | Durum | Kanıt |
|---|---|---|---|
| T1 | Index manifest modeli | **TAMAM** | `src/belge_gozu/index/manifest.py` (148 satır); `tests/index/test_manifest.py`; üretim `manifest.json` tüm alanları taşıyor |
| T2 | Maskeli + formatlı encoder | **TAMAM** | `src/belge_gozu/index/encode.py` (150); `tests/index/test_encode.py`, `test_encode_mask.py` (slow determinizm testi dahil) |
| T3 | PackedIndex v2 (padding reddi) | **TAMAM** | `src/belge_gozu/index/store.py` (125), `build`'de `padding satırı sızmış` ValueError; `tests/index/test_store.py` |
| T4 | Serve-time uyumluluk (compat) | **TAMAM** | `src/belge_gozu/index/compat.py` (53); `app/main.py:270-333`; `tests/app/test_compat.py` |
| T5 | ExhaustiveBinaryRetriever + pipeline seçimi | **TAMAM (+ genişletildi)** | `src/belge_gozu/retrieval/core.py` (164) — sonradan generic `ExhaustiveRetriever`'a çevrildi (int8 geçişi, `b790f6c`) |
| T6 | Benchmark veri modeli | **TAMAM** | `src/belge_gozu/bench/dataset.py` (219), 12 dilim + answerable/unanswerable + span alanları; `tests/bench/test_dataset.py` |
| T7 | Metrikler | **TAMAM** | `src/belge_gozu/bench/metrics.py` (37) — recall@k / mrr / ndcg / bootstrap_ci; `tests/bench/test_metrics.py` |
| T8 | Teşhis harness'ı + `bench run` | **TAMAM (bir eksik)** | `src/belge_gozu/bench/harness.py` (308); `cli.py:533`. **Eksik:** `EvalReport.oracle_gap` alanı hiç eklenmedi |
| T9 | Float oracle | **TAMAM (taşındı)** | `src/belge_gozu/bench/oracle.py` (38) + `index/float_store.py` (101) — D4 katman ihlali kapandı |
| T10 | RetrievalEval set v1 + semantic retrieval_eval testleri | **KISMİ** | `data/bench/retrieval_eval_v1.jsonl` (48) + 6 slow test var; **insan doğrulama kapısı (Step 2) hâlâ açık** — 3/48 insan |
| T11 | Processor format A/B + karar | **TAMAM** | A1/A2/A3 koşuldu; karar `train-compat-v1` (`config.py:98`), `data/bench/results/a{1,2}-*.json` |
| T12 | Kuantizasyon ablasyonu C1/C2 | **TAMAM (+ üretime alındı)** | `index/quantize.py` (128); int8 üretimde (`config.py:60`) |
| T13 | Kalite telemetrisi genişletmesi | **TAMAM** | `telemetry/schema.py`, `prom.py`; `docs/research/metrics-catalog.md` güncel (bg_verifier_verdicts, bg_llm_key_rotations dahil) |
| T14 | README/UI dürüstlüğü + baseline & gate raporu | **TAMAM** | `README.md`, `app/static/index.html:413/456/617`; `2026-08-27-p0-baseline.md` + `2026-08-27-p0-gate.md` |
| T15 | Hijyen borcu üçlüsü | **TAMAM** | `tests/corpus/test_manifest.py:40` (shipped-manifest + unique id), `:50` (`build_ssl_context`), `tests/index/test_store.py:116-135` (çok-chunk), `pyproject.toml:57-65` (uyarı filtresi) |

**P0 sayım: TAMAM 14 · KISMİ 1 (T10) · YAPILMADI 0.**

### 2.2 P1 — Task 1..13 (plan: `2026-08-26-belge-gozu-p1-hybrid-retrieval.md`, R23 daraltmalı)

| # | Görev | Durum | Kanıt / sapma |
|---|---|---|---|
| T1 | Sayfa metni + kalite dedektörü | **KISMİ** (R23'ün kendi etiketi "T1kısmi") | `src/belge_gozu/corpus/text.py`; `tests/corpus/test_text.py`. Plan `PageText` modeli + `text_source` + `quality` + OCR kancası istiyordu → gerçekte düz `page_id,text` DataFrame |
| T2 | OCR fallback + Türkçe OCR benchmark'ı | **YAPILMADI** (backlog) | `corpus/ocr.py` YOK. Gerekçe: 4221/4222 sayfa metin katmanlı (bu denetimde doğrulandı: 1 boş, 4 sayfa <50 karakter) |
| T3 | Madde segmentasyonu | **YAPILMADI** (backlog) | `corpus/articles.py` YOK → **ilke 11 ihlali** |
| T4 | Kanun alias + facet ayrıştırma | **YAPILMADI** (backlog) | `retrieval/query.py` YOK, `data/manifest/aliases.csv` YOK |
| T5 | Sorgu varyantları | **YAPILMADI** (backlog) | Aynı dosya yok; varyant/rewrite mekanizması hiç yazılmadı |
| T6 | BM25 metin kanalı | **TAMAM (ağır uyarlamayla)** | `src/belge_gozu/retrieval/text.py:249-306` (`BM25Index`); `tests/retrieval/test_text.py`. Sapma: `.scores()->ndarray` (plan `.search()->list`), `phrase_hits` YOK, **disk persistansı YOK** (her açılışta bellekte kuruluyor, ~0,4 sn). Plan'da olmayan eklemeler: ASCII katlama (exp12), QTF doygunluk tavanı, doküman-adı yönlendirmesi (plan bunu T8'e vermişti) |
| T7 | Dense metin kanalı | **YAPILMADI** (backlog, yeniden kapılandı) | `retrieval/dense.py` YOK. **Paraphrase diliminin R@5'i 0,2857 — dense'in doğrudan hedefi** |
| T8 | RRF füzyonu + HybridRetriever | **SÜPERSEDE** | `retrieval/fusion.py` YOK, `rrf_fuse` kodda **hiç yok**. `retrieval/hybrid.py:151-287` var ama mekanizma farklı: BM25 sırası + `route_window` (yumuşak, pencere-içi). RRF üç biçimde ölçülüp reddedildi |
| T9 | Kanal-düzeyi teşhis + recall gate | **KISMİ** (planın kendi notu) | `bench/harness.py:103-171` `HybridDiagnosticAdapter` — 2 kanal (plan 5 istiyordu). **G1.1'in "candidate-union"u bugünkü mimaride hesaplanamıyor** (tek kanal sıralıyor) |
| T10 | Cross-encoder reranker | **YAPILMADI** (backlog) | `retrieval/rerank.py` YOK; `config.py`'de `rerank_enabled`/`rerank_model`/`rerank_pool` YOK |
| T11 | Evidence pack + komşu sayfa | **YAPILMADI** (backlog) | `retrieval/evidence.py` YOK → P2 T1/T2 bunu `list[PageHit]` + `page_texts` ile yeniden tasarlamak zorunda kaldı |
| T12 | Tam benchmark v2 (120+30, insan kapılı) | **YAPILMADI** (backlog) | `data/bench/bench_v2.jsonl` YOK. `splits_v1.json` **var ama P2'nin abstention_eval işinden geldi**, P1 T12'den değil |
| T13 | HF Space bütçeleri + ablasyonlar + kapı raporu | **YAPILMADI** | `p1-gate.md` YOK; F1/E1/F2/F3/G1 matrisi koşulmadı |

**P1 sayım: TAMAM 1 (T6) · KISMİ 2 (T1, T9) · SÜPERSEDE 1 (T8) · YAPILMADI 9.**

**Plan ↔ gerçek mimari ayrışması (P1):** Plan beş kanallı (bm25-page, bm25-article,
phrase, dense, visual) + RRF + metadata soft-boost + `retrieval_mode` bayrağı öngörüyordu.
Gerçekte **iki kanal** var ve yalnız biri sıralıyor. `retrieval/` altında bulunanlar:
`core.py` (P0 görsel), `hybrid.py` (yeni), `text.py` (yeni), `types.py` (yeni).
BM25/routing parametreleri `Settings` alanı değil, `retrieval/text.py` modül sabiti
(`F5=5`, `WINDOW=50`, `QTF_CAP=2`, `K1/B`, `RECIPE_VERSION`) — env ile ayarlanamaz,
ama `recipe_fingerprint` üzerinden kalibrasyon anahtarına giriyor.

### 2.3 P2 — Task 1..12 (plan: `2026-08-26-belge-gozu-p2-selective-answering.md`)

| # | Görev | Durum | Kanıt / sapma |
|---|---|---|---|
| T1 | Claim segmentasyonu + verifier | **TAMAM** (bayrak-kapalı) | `answer/verify.py` (992); `tests/answer/test_verify.py` (35 test); iki tur inceleme → 15/15 RESOLVED. Sapma: `EvidencePack` yok → `list[PageHit]`+`page_texts`; verdict sözlüğü `supported/unsupported/belirsiz` (plan: supported/refuted/insufficient); `VerifiedAnswer` sınıfı kodda **yok** |
| T2 | İki kapı (retrieval ↔ evidence) | **KISMİ** | `answer/base.py:174-245`; `tests/answer/test_gate.py` (19). Sapma: `decide_verdicts()` **hiç yazılmadı** (3-yollu present/retry/abstain yerine ikili demote); `Answer`'da `abstain_reason` alanı yok |
| T3 | Auto-citation kaldırma + citation metrikleri | **KISMİ** | Kaldırma **TAM** (`answer/gemini.py:660-665` + test). Metrikler **YOK** — `bench/answer_eval.py` yok, `citation_precision` yok |
| T4 | Answerable/unanswerable koşum harness'ı | **YAPILMADI** | `bench answers` CLI alt komutu yok; `run_answer_eval`/`AnswerEvalReport` yok. (Tüketeceği veri `abstention_eval_v1.jsonl` hazır) |
| T5 | Güven özellikleri | **TAMAM** (özellik kümesi yeniden tanımlandı) | `answer/calibrate.py:74-183`; `tests/answer/test_calibrate.py` (49). Plan'ın 7 özelliği yerine ampirik 5 özellik (`served_top1, bm25_margin, matched_terms_top1, matched_frac, routed`) — çünkü reranker/facet/madde katmanı yok |
| T6 | Kalibratör + versiyonlu threshold | **KISMİ** | `answer/calibrate.py:229-540`; artefakt `data/calibration/…7b56eeeb7327/calibrator.json`. Sapma: **isotonic/Platt yok** (yalnız logistic, sklearn'siz full-batch GD); `CostMatrix` yok → `max_risk` bütçesi |
| T7 | Kalibrasyon metrikleri + risk-coverage | **TAMAM** | `bench/calibration_metrics.py` (328) — brier/ece/auroc/risk_coverage/conformal + Wilson/Clopper-Pearson; `tests/bench/test_calibration_metrics.py` (38) |
| T8 | Selective answering entegrasyonu + güvenli fallback | **KISMİ** | Kapı bağlı (`calibrate.py:706-742` `CalibratedRetrievalGate`); `tests/app/test_gates_api.py` (12). Sapma: `/healthz`'de `"calibrator"` alanı **yok**; artefakt eksikse plan "gate'i kapat + WARNING" diyordu, gerçekte **boot fail-fast**; kapı sırası retrieval→**confidence**→answerer→evidence (plan: …→evidence→confidence) — Gemini çağrısından tasarruf için bilinçli, ama kapı 1 `verifier_support_ratio`'yu yapısal olarak asla göremez |
| T9 | UI claim-citation + outcome telemetry + feedback + drift | **YAPILMADI** | `index.html`'de claim yok; `/feedback` yok; `RequestEvent`'e yeni alan eklenmedi; `scripts/drift_report.py` yok |
| T10 | İnsan-kalibreli LLM-judge (PPI) | **YAPILMADI** | `bench/judge.py` yok. Ön koşul (≥30 insan çifti) da yok |
| T11 | Koşullu fine-tuning alt plan kapısı | **YAPILMADI** | Alt plan dosyası yok; kapı koşulu 2 (paraphrase R@5 < %80) **zaten sağlanıyor** (0,2857) ama resmî değerlendirme yapılmadı |
| T12 | P2 kapı raporu + final koşum + README | **YAPILMADI** | `p2-gate.md` yok; test-split koşumu yok; README'de P2 sonuç bölümü yok |

**P2 sayım: TAMAM 3 (T1, T5, T7) · KISMİ 4 (T2, T3, T6, T8) · YAPILMADI 5 (T4, T9, T10, T11, T12).**

**Plan ↔ gerçek ayrışması (P2):** Planın 12 görevinden 6'sı (T1/T2/T5/T6/T8/T12) P1'in hiç
yazılmamış arayüzlerini (`EvidencePack`, `QueryFacets`, `bench_v2.jsonl`, reranker skoru)
tüketmek üzere tasarlanmıştı; hepsi `list[PageHit]` + `page_texts.parquet` üzerine yeniden
kuruldu. Master §3'ün tip sözlüğünden `VerifiedAnswer`, `ClaimVerdict`, `CitationRef`,
`CostMatrix` sınıflarının **hiçbiri kodda yok** — işlevsel karşılıkları farklı adlarla var
(`Claim`, `Verdict`, `ThresholdChoice`, düz dict). Ayrıca bayrak adları plandan farklı:
`evidence_verifier_enabled` → `gate_verifier`, `selective_answering_enabled` →
`gate_calibrated` (ikisi de `False`).

**Numaralı görev olmayan ama büyük emek:** `data/bench/abstention_eval_v1.jsonl` (330 satır
cevaplanamaz benchmark) taslak → çapraz-kontrol → checker-2 → yedek parti turlarıyla
üretildi; test yakasında 155 satırla Clopper-Pearson üst sınırı %1,914 < %2 (G2.1'in
ölçülebilirlik ön koşulu). **Henüz bir T4 harness'ı ya da T12 koşumu tarafından
tüketilmiyor** — bugün yalnız T5/T6 kalibrasyonunu besliyor.

---

## 3. 25 ilke uyumu

Spec §2'deki liste birebir; her satır bugünkü koda karşı doğrulandı.

| # | İlke | Uyum | Kanıt / not |
|---|---|---|---|
| 1 | Threshold ayarı kök-neden düzeltmesi değildir | **UYUYOR** | Eşik üç kez **mekanik ölçek taşımasıyla** hareket etti (60,0 → 0,58 → 10,6), her seferinde çalışma noktası sayıca yeniden üretildi; `config.py:122-167` bunu açıkça *"KALİBRASYON DEĞİL"* diye yazıyor. Kök nedenler yapısal olarak düzeltildi (Stage-1 kaldırma, format, int8, metin kanalı). |
| 2 | Candidate recall ölçülmeden reranker eklenmez | **UYUYOR (boşta)** | Reranker yok; ön koşul ölçüldü (R@20 = 0,9302) |
| 3 | Altın sayfa havuzda değilse reranker kurtaramaz | **UYUYOR (boşta)** | Aynı |
| 4 | Stage-1 kaldırılır ya da ≥%98 ile değiştirilir | **UYUYOR** | `two-stage` yalnız ablasyon kolu (`config.py:121` docstring); survival %9,3 ölçüldü |
| 5 | Exhaustive/native oracle kalıcı kalite referansı | **RİSK** | `bench oracle` + `bench/oracle.py` + `index/float_store.py` duruyor; **ama** üretim artık BM25 sıralıyor → oracle yalnız **görsel kanalı** ölçüyor, üretim hattının oracle referansı yok. Ayrıca `EvalReport.oracle_gap` alanı hiç eklenmedi (G0.4 sapması). |
| 6 | Index manifest künyesi | **UYUYOR** | `index/manifest.py`; üretim manifest'i model+revision+query_format+doc_prompt_sha256+render+mask_policy+quantization+corpus_checksum+n_pages/n_tokens+built_at+git_commit taşıyor |
| 7 | Serve fail-fast | **UYUYOR** | `index/compat.py` + `app/main.py:270-333`; `BG_ALLOW_INDEX_MISMATCH` bilinçli override (`config.py:179`) |
| 8 | Görsel korunur, metin yasaklanmaz | **UYUYOR** | `retrieval/hybrid.py:251-252` görsel her sorguda koşar; `PageHit.visual_score` ayrı alanda taşınır |
| 9 | İki açık mod (visual-only / hybrid-production) | **AD SAPMASI** | Fiilen var ama `retrieval_mode: "visual-only"\|"hybrid-production"` yerine `retrieval_pipeline: "hybrid"\|"exhaustive"\|"two-stage"` (`config.py:121`). Rollback tek satır ama **eşik de taşınmalı** — bu korkuluklu (`app/main.py:347`). |
| 10 | Born-digital metin doğrudan, OCR fallback | **KISMİ** | `corpus/text.py` gömülü metin çıkarıyor; **OCR yok** (`corpus/ocr.py` YOK). Bugün ölçüldü: 4222 sayfanın **1'i tamamen boş**, **4'ü <50 karakter** → bu sayfalar sıralayan tek kanal için görünmez. |
| 11 | Kanun → bölüm → madde → fıkra → sayfa hiyerarşisi | **İHLAL / EKSİK** | `corpus/articles.py` **YOK**. Retrieval atomu hâlâ fiziksel sayfa. `bench/dataset.py:47-48` `gold_article_ids` + `minimal_evidence_spans` alanlarını taşıyor ama **hiçbir şey üretmiyor/tüketmiyor**; P2 `CitationRef.article_id` kalıcı `None`. |
| 12-13 | Query rewrite orijinali değiştirmez; varyantlar fuse edilir | **BOŞTA (ihlal yok)** | `retrieval/query.py` YOK; varyant mekanizması hiç yazılmadı → tek sorgu dizesi var. ASCII katlama/F5 kırpma bir **analiz** adımıdır (sorgu ve belge tarafına simetrik uygulanır, `retrieval/text.py`), rewrite değil. |
| 14 | İlk fusion RRF | **LAFZEN UYULDU, SONUÇ SAPTI** | RRF **önce** denendi (exp2, k=60) ve ölçümle reddedildi (R@5 0,674→0,395); pencere-içi RRF de reddedildi (0,8372→0,5349). Kodda `rrf_fuse` **yok**. Üretimdeki "füzyon" = BM25 sırası + doküman-adı pencere-içi yönlendirme (hard filter değil). **İletişim riski:** sistem "hibrit" adını taşıyor ama sıralamayı tek kanal yapıyor. |
| 15 | Expansion/HyDE yalnız ölçülmüş kazançla | **UYUYOR** | Hiç yok |
| 16 | Reranking yalnız yüksek-recall union üstünde | **UYUYOR (boşta)** | Reranker yok |
| 17 | Retrieval confidence ≠ evidence sufficiency | **UYUYOR** | İki ayrı kapı: `answer/base.py:131-141` (`RetrievalGate` / `EvidenceGateProtocol`), sıra `_eval_gate1` → cevap üretimi → `_apply_gate2` |
| 18 | Raw skor güven yüzdesi gibi gösterilmez | **UYUYOR** | `app/static/index.html:413` (*"kaba, kalibre edilmemiş bir kesme noktası (güven ölçüsü değil)"*), `:456`, `:617` (*"güven yüzdesi DEĞİL"*). **Küçük risk:** çubuk genişliği yine de `score/maxV` yüzdesi olarak çiziliyor (`:565-566`). |
| 19 | Auto-citation fallback kaldırılır | **UYUYOR** | `answer/gemini.py:661-664` + `tests/answer/test_gemini.py:67` |
| 20 | Yanıt yalnız claim-level verification geçerse sunulur | **BUGÜN İHLAL (bilinçli, bayrak-kapalı)** | `config.py:216` `gate_verifier=False` → varsayılan serviste iddia doğrulaması **yok**. Gerekçe kayıtlı: master §1'in kapı kuralı G1 raporu olmadan default-açık entegrasyonu yasaklıyor (`config.py:196-202`). Yani ilke 20 ↔ master §1 arasında **çözülmemiş bir gerilim** var. |
| 21 | Benchmark + mimari sabitlenmeden kalibrasyon yok | **RİSK** | Eşik hareketleri mekanik (uyumlu). **Ama** bir kalibratör `bench_v2` yokken ve G1 adjudike edilmemişken fit edildi (`calibrator.json`, dev n=185, yalnız 22 pozitif). Bayrak kapalı + versiyonlu anahtar bunu meşru bir *hazırlık* yapıyor, ama mimari (dense kanal, reranker) değişirse artefakt geçersizleşir. |
| 22 | Yeni katman yalnız ölçülmüş kazançla default açılır | **UYUYOR** | Hibrit default-açık, gerekçesi ölçülü (0,2326→0,8605); verifier/kalibre kapı default-kapalı |
| 23 | Korpus genişletme / agentic / LocalVLM önce değil | **UYUYOR** | `corpus_checksum` altı indeksin hepsinde `133444d8c235…` — korpus donuk; 1475 eklenmedi; LocalVLM yok |
| 24 | Eski dokümanlar silinmez, supersession yazılır | **UYUYOR** | `git log --diff-filter=D -- '*.md'` → **hiç md silinmemiş**; supersession notları spec (:5-11), master (:12-14), P1 planı (:5-22) |
| 25 | Her katman feature flag + rollback | **UYUYOR** | `retrieval_pipeline`, `query_format_id` (enum), `gate_calibrated`, `gate_verifier`, `allow_index_mismatch`, rate-limit'ler, `log_query_text`; eski indeks dizinleri diskte duruyor (`data/index*` 7 dizin) |

**İhlal/risk taşıyan ilkeler (7):** 5 (oracle artık üretim hattını ölçmüyor) · 10 (OCR
fallback yok, 5 sayfa metinsiz) · **11 (madde hiyerarşisi hiç kurulmadı — en somut
ihlal)** · 14 (RRF reddedildi; "hibrit" adı sıralama gerçeğini abartıyor) · 18 (çubuk
yüzde gibi görünüyor — kozmetik) · **20 (varsayılan serviste claim doğrulaması yok)** ·
21 (kalibratör mimari sabitlenmeden fit edildi).

---

## 4. Açık bulgu envanteri (birleşik)

Kaynaklar: `2026-08-29-e2e-review.md` (K1-K36 + B1-B38) · `agent-reports/2026-08-30-e2e-review-2.md`
(Y1-Y45) · `2026-08-29-config-coupling-audit.md` §2 (A/B/C/D/E/F, 139 satırlık S/C/D
envanterinin triyajı) · `2026-08-30-edge-case-probe.md` (EC1-EC9) · vitrin sprint
review/re-review (M1-M2, L1-L5, N1-N3, NEW-1) · P2 T1/T2 rotation review/re-review
(H1-H3, M1-M4, L1-L8, N1-N2) · `p0-decision-log.md` "minor (deferred)" kayıtları.

**Ölçek:** ham bulgu kimliği sayısı ≈ **200+** (K36 + B38 + Y45 + EC9 + CA'nın 139 S/C/D
satırı + ~30 review nit'i), tekrarlar birleştirildikten sonra **yaklaşık 150 ayrık kalem**.
Bunların büyük çoğunluğu kaynak belgelerde zaten KAPALI işaretli (CA §2A'nın 21 satırı,
E2 §7'nin K-kalemi yeniden denetimi, vitrin/P2 re-review'larının 15/15 + 12/12 + 15/15
RESOLVED turları).

### 4.1 Spot-check — 15 kalem bugünkü koda karşı doğrulandı

| # | Kalem | Hüküm | Kanıt (bugünkü kod) |
|---|---|---|---|
| 1 | **Y1** BM25 sorgu-terimi tekrarı skoru şişiriyor (667 vs eşik 10,6) | **KAPANMIŞ** | `retrieval/text.py:109` `QTF_CAP = 2`; `scores()` `:290-294` `qw = min(qtf, QTF_CAP)`. exp14/R30: saldırı sorgusu 667,5 → 16,7; retrieval_eval R@5 değişmedi |
| 2 | **Y2** O(sorgu_token × 4222) doğrusal tarama; inverted index yok | **HÂLÂ AÇIK** | `retrieval/text.py:295` `for i, freqs in enumerate(self.doc_freqs):` — hâlâ tam tarama. Skor-şişirme kolu kapandı, karmaşıklık kapanmadı. `config.py:192-193` rate limit varsayılanı hâlâ 0 |
| 3 | **Y15 / K33** Gemini çağrısında zaman aşımı yok; lazy client yarışı | **KAPANMIŞ** | `answer/gemini.py:42-45` `GEMINI_TIMEOUT_S=15.0`, `GEMINI_TOTAL_BUDGET_S=35.0`; `_ensure_client` `:245-261` çift-kontrollü kilit (`threading.Lock`) |
| 4 | **K27** `honest_miss` Türkçe İ/I'de kırılıyor | **KAPANMIŞ** | `answer/base.py:64,74` artık `tr_lower()` kullanıyor (paylaşılan `retrieval.text` fonksiyonu) |
| 5 | **Y17/Y31** honest-miss API sözleşmesinde görünmez | **KAPANMIŞ** | `app/main.py:781` yanıtta `"honest_miss"` alanı; `index.html:805,811,818` ayrı CSS sınıfı |
| 6 | **Y20** `degraded` olaylarında hata sınıfı yok (114/114 NULL) | **KAPANMIŞ** | `answer/base.py:194-202` `error_type` hesaplanıp `annotate` ediliyor; `gemini.py:149-198` `classify_error()` taksonomisi |
| 7 | **Y28/K17** Enter/çip tıklaması çift-gönderim korumasını atlıyor | **KAPANMIŞ** | `index.html:728,737` `let inFlight = false` + tek giriş noktası |
| 8 | **Y29** UI'da "43 soruluk **insan-doğrulamalı** retrieval_eval" | **HÂLÂ AÇIK — KRİTİK (itibar)** | `app/static/index.html:454` birebir duruyor. Gerçek: 3/48 insan. E2 bunu KRİTİK/#7 diye işaretlemiş, düzeltme metnini de vermiş; uygulanmamış (`git blame` → `6d5b345`, sonrasında değişmemiş) |
| 9 | **Y30** "6 çipin hepsi retrieval_eval'den" iddiası | **HÂLÂ AÇIK — ÖNEMLİ (itibar)** | `index.html:347` birebir duruyor; 6 çipin 2'si için yanlış (biri ayarlama hedefi olmuş vitrin sorgusu, biri retrieval_eval'de olmayan bir sorunun ASCII varyantı) |
| 10 | **K18** `stage1_ms`/`stage2_ms` üretimde daima NULL | **HÂLÂ AÇIK** | `app/main.py:544-545` hâlâ `col.stages.get("stage1_hamming")` / `("stage2_maxsim")` okuyor; hibrit hat bu adları hiç yaymıyor (`query_encode`, `exhaustive_maxsim`, `text_bm25`, `route_fuse`) |
| 11 | **K9/K10** `candidate_survival` aslında Recall@200; `gold_ranks=-1` teşhis sinyalini yok ediyor | **HÂLÂ AÇIK** | `bench/harness.py:265` `-1 if g not in top_ids`; `:271` `candidate_survival` yalnız `record_top` penceresinde üyelik bakıyor |
| 12 | **K21** `hub.pull_index` hedef dizini temizlemiyor | **HÂLÂ AÇIK** | `index/hub.py:43-45` `mkdir(exist_ok=True)` + doğrudan `shutil.copy` döngüsü; `delete_patterns` yok → karma kuantizasyon artığı riski |
| 13 | **Y5** `/search` OOV sorguda "hepsi sıfır" listeyi geçerli sonuç gibi döndürüyor | **HÂLÂ AÇIK** | `app/main.py:713` hâlâ düz `{"hits": hits}`; `no_match`/`status` alanı yok |
| 14 | **K3** Eşik cevaplanabilir/cevaplanamazı ayırmıyor | **HÂLÂ AÇIK (bilinçli, kilitli)** | `tests/retrieval/test_semantic_retrieval_eval.py:202-235` `xfail(strict=True)` — hibrit/BM25 ölçeğinde de ayırmıyor. Çözüm P2 kapı 1 (varsayılan kapalı) |
| 15 | **NEW-1** `_evict_oldest` boş dict'te `ValueError` | **HÂLÂ AÇIK ama GEÇERSİZ (ulaşılamaz)** | `app/main.py:178-183` guard yok; ama `max_clients` Settings alanı değil, daima `RATE_LIMITER_MAX_CLIENTS=10_000` |

**Ek bağımsız spot-check'ler (bu denetimin kendi taraması):**

| Kalem | Hüküm | Kanıt |
|---|---|---|
| **C8** `query_format_id` enum'a çevrilsin | **KAPANMIŞ** | `config.py:98` `QueryFormatChoice` |
| **C9** CLI import-anı ValidationError | **KAPANMIŞ** | `cli.py:96-97` `try: _CLI_DEFAULTS = Settings() / except ValidationError` |
| **C12** CI `uv sync --locked` | **HÂLÂ AÇIK** | `.github/workflows/ci.yml` → `uv sync --extra dev` (locked yok, ml yok, docker build yok) |
| **C19** Test env izolasyonu autouse fixture | **HÂLÂ AÇIK** | `grep -rn "autouse=True" tests/` → **hiç yok**; `Settings` `.env`'i okur, kökte gerçek anahtarlı `.env` var |
| **C38** Docker `BG_LOG_QUERY_TEXT=false` | **KAPANMIŞ** | `Dockerfile` |
| **C41 / PDL-T9** `FloatIndex.build`'de padding invariantı | **HÂLÂ AÇIK** | `index/float_store.py`'da all-zero kontrolü yok (`store.py` ile asimetrik) |
| **C42** Ölü sabit `DEFAULT_MANIFEST` | **HÂLÂ AÇIK** | `cli.py:54` tanımlı, `src/` genelinde başka kullanım yok |
| **C5** Dockerfile `--pull` sessiz no-op | **HÂLÂ AÇIK** | `cli.py:1337` + Dockerfile'da `BG_HF_DATASET_REPO` hiç set edilmemiş (dört revizyonun hiçbirinde) |
| **C11** `HF_KEY` → `HF_TOKEN` | **HÂLÂ AÇIK** | `index/hub.py:8-9` `HfApi()` token'sız; `Settings`'te `hf_token` yok; `.env`'de anahtar adı `HF_KEY ` (sondaki boşlukla) |
| **S51/S52** `k` ve sorgu uzunluğu doğrulaması | **KAPANMIŞ** | `app/main.py:77-78` `max_length=MAX_QUERY_CHARS(500)`, `ge=1, le=MAX_K(50)` |
| **S33/S34/D1/D2** API'de abstain/degraded ayrımı | **KAPANMIŞ** | Yanıt `status` alanı: `"answered"|"abstained"|"degraded"` (`index.html:791-812`) |
| **C10** `write-manifest` mevcut manifest'i ezmesin | **KISMEN AÇIK** | `cli.py:418-448` yalnız `--legacy` zorunluluğunu kontrol ediyor, var olan `manifest.json`'ı reddetmiyor |

### 4.2 Hâlâ açık, gruplanmış özet

- **İtibar riski (2, en acil):** Y29 (`index.html:454` "insan-doğrulamalı"), Y30 (`:347` çip kaynağı iddiası).
- **Bench teşhis doğruluğu (3):** K9, K10 (`harness.py`), K18 (`main.py:544-545` ölü sütunlar), + `EvalReport.oracle_gap` yok.
- **Deploy zinciri (6):** C5 (Docker `--pull` no-op), C11 (HF token), K21 (pull atomikliği), C28 (uzak indeks bayat), B5/B6 (CUDA tekerleği, uid/HF_HOME), C12 (CI lock+ml+docker).
- **Ölçeklenme/istismar (2):** Y2 (BM25 doğrusal tarama), Y11 (`/stats`/`/metrics` kimlik doğrulamasız + tam tablo taraması).
- **UI dürüstlük/erişilebilirlik (~10):** Y33 (sahte aşama zamanlaması), Y34 (sabit "BM25" başlığı), Y35 (istemci eşik sabiti), Y36/K14 (sıfır-tabanlı çubuklar), Y37 (AbortController yok), Y38 (a11y), Y40 (enjeksiyon lavaboları), Y43, Y45.
- **Kalibrasyon-bağımlı (2):** K3/D-grubu (eşik ayırmıyor), S9/D16 (göreli margin).
- **Nit/teknik borç (~15):** C41, C42, C19, C10, S5, D21, PDL-T1/T2/T4/T15, NEW-1, N1 (bütçe ≤+2 taşma), N2 (bayat docstring), Y6, Y8, Y10, Y12, Y24, Y25.

---

## 5. Ürün / dağıtım boşlukları

| # | Alan | Bugünkü durum (kanıt) | Eksik |
|---|---|---|---|
| 5.1 | **Kaynak kodu yayını** | **`git remote -v` → 0 remote.** `main` dalı `d964b9c` (telemetri dönemi), HEAD'in **85 commit gerisinde**; toplam 147 commit. P0+P1+P2'nin tamamı yalnız yerel `feat/p0-retrieval-correctness` dalında | GitHub repo yok → **CI hiç koşmadı** (workflow `push:[main]` + `pull_request` tetikleyicili), portfolyo görünürlüğü sıfır. Bu, "işe alım sinyali" hedefli bir projede tek başına en büyük ürün boşluğu |
| 5.2 | **HF Space (barındırma)** | Space **hiç oluşturulmadı**. README:21-30 gerekçeyi yazıyor: HF, Docker/Gradio Space oluşturmak için (ücretsiz `cpu-basic` katmanında bile) **PRO aboneliği** istiyor, Hub API `402 Payment Required` döndürmüş (yazar iddiası; bu denetimde ağ çağrısı yapılmadı). Yalnız statik SDK ücretsiz | Space YAML frontmatter (`sdk: docker`, title, emoji…) README'de **yok**; Space repo'suna push eden kod yolu yok (`index/hub.py` yalnız `repo_type="dataset"`); Space secrets/variables şablonu (`.env.example`) yok |
| 5.3 | **Docker / compose** | `Dockerfile` (21 satır) var: python:3.12-slim, `uv export --frozen --extra ml`, `COPY src`, `BG_DEVICE=cpu`, rate-limit 10/60, `BG_LOG_QUERY_TEXT=false`, `CMD serve --pull --port 7860`. Uygulama düzeyinde compose **YOK** (yalnız `observability/docker-compose.yml`) | **İmaj bir kez bile build edilmedi** (CI'da adım yok, raporlarda kayıt yok). Açık kusurlar: (a) `CMD --pull` ama `BG_HF_DATASET_REPO` hiç set edilmiyor → `cli.py:1337` `if pull and s.hf_dataset_repo:` **sessiz no-op**; `COPY src` olduğu için `data/` de yok → boot'ta indeks bulunamaz; (b) `USER` yok (HF Docker Space uid 1000 ile koşar, `/app` root'a ait); (c) `HF_HOME` yok (~1 GB model indirme yeri tanımsız); (d) `EventRecorder.__init__` (`telemetry/recorder.py:75`) `mkdir`'i try/except dışında → izin hatası create_app'i çökertir; (e) `pyproject.toml`'de `[tool.uv]` CPU-only torch index yok → Linux'ta CUDA tekerlekleri (~6-8 GB açılmış); (f) `.dockerignore` yok |
| 5.4 | **HF hub push/pull güncelliği** | `index/hub.py` (52 satır): `push_index`/`pull_index`, `repo_type="dataset"`. **`revision=` parametresi hiçbirinde yok** → daima default ref; `HfApi()`'ye **token geçilmiyor**; `Settings`'te `hf_token` alanı yok (`.env`'de `HF_KEY ` var — hem yanlış ad hem sondaki boşluk; `extra="ignore"` sessizce yutar). `upload_folder`'da `delete_patterns` yok → farklı kuantizasyon dosyaları uzakta karışabilir | **Uzak indeks bayat.** README:252-258'in kendi ifadesi: yayınlanmış Hub kopyası P1 ÖNCESİ push edildi, `page_texts.parquet` **içermiyor** → varsayılan `hybrid` yolu `serve --pull` ile ayağa kalkmaz (fail-fast). Ayrıca 27-Ağu int8/train-compat yeniden inşasından beri **hiçbir push kaydı yok** (git log/findings/raporlar temiz — ajan raporu #28) → uzakta v0 (manifest'siz) indeks duruyorsa taze klon `compat.py` fail-fast'ine çarpar |
| 5.5 | **CI kapsamı** | `.github/workflows/ci.yml` (14 satır): checkout → setup-uv → `uv sync --extra dev` → `make lint` (ruff check + ruff format --check + pyright) → `make test` (`pytest -m "not slow"`) | (a) **`--locked`/`--frozen` yok** → lockfile disiplini denetlenmiyor (Dockerfile `uv export --frozen` kullanıyor → asimetri; drift ancak hiç yapılmamış imaj build'inde patlar); (b) `--extra ml` yok → torch/colpali CI'da hiç kurulmuyor, gerçek retriever/encoder yolu test edilmiyor (6 slow test daima atlanıyor); (c) **docker build adımı yok**; (d) bağımlılık denetimi / gizli-anahtar taraması yok |
| 5.6 | **Gizlilik / rate limit varsayılanları** | `log_query_text: bool = True` (`config.py:176`) → **ham sorgu metni varsayılan olarak `data/requests.sqlite`'a yazılıyor**; sha256 her koşulda yazılır. `rate_limit_ask_per_min = 0`, `rate_limit_search_per_min = 0` (kapalı, `config.py:192-193`). Dağıtım profili yalnız **Dockerfile'da**: 10/dk ask, 60/dk search, `log_query_text=false`. Limiter süreç-içi + IP başına, `X-Forwarded-For`'a güvenmiyor, 10.000 istemci tavanı + tahliye (`app/main.py:102-183`) | `make serve` (yerel/manuel dağıtım) yolunda **ne hız sınırı ne gizlilik varsayılanı** devrede — imaj dışına çıkan her dağıtım açıkta. `/metrics`, `/stats`, `/ask`, `/search`, `/pages/*` **kimlik doğrulamasız**. CORS middleware yok (aynı-origin UI için sorun değil, ayrı frontend için engel) |
| 5.7 | **Model / anahtar pinleri** | `colpali-engine==0.3.18` **kesin pin** (`pyproject.toml:29`, gerekçeli: prompt-format sözleşmesi). Model revizyonu manifest'te pinli (`model_revision=650243e9…`) ve compat'ta karşılaştırılıyor. `retriever_model` config'te revizyonsuz. `gemini_model="gemini-3.6-flash"` (`config.py:66`) — sürüm-tarihli pin değil. `torch>=2.4` gevşek. Çekirdek bağımlılıkların hepsi `>=` aralığı; `uv.lock` (380 KB, 29 Ağu) mevcut | `from_pretrained(revision=…)` build yolunda kullanılmıyor (audit C15); `transformers` açık bağımlılık değil (C13 kalanı); Gemini model adı sürüm-pinli değil → sağlayıcı yine emekliye ayırırsa aynı 404 tekrar eder; `/healthz` Gemini anahtarı yokken de `"ok"` diyor (`answerer_ready` alanı yok — C37) |


---

## 6. Veri / benchmark dürüstlük durumu

### 6.1 Ölçülen gerçek sayılar (JSONL'lerden bu denetimde hesaplandı)

**`data/bench/retrieval_eval_v1.jsonl` — 48 satır**

| Alan | Dağılım |
|---|---|
| `verification_status` | verified 48 (draft 0, rejected 0) |
| `verification_kind` | **model-cross-check 45 · human 3** |
| `verified_by` | `model-cross-check:claude-opus-5` 45 · `baran` 3 |
| answerable | 43 · unanswerable 5 |
| dilim | dogrudan-madde 13, paraphrase 7, madde-numarali 6, ayni-kanun-hard-negative 5, tablo-layout 4, tarihi-tarama 4, capraz-kanun-terim 4, korpus-disi 3, anlamsiz-ood 2 |

İnsan doğrulamalı 3 satır: `c307`, `c308`, `c314`. **Spec'in 12 diliminden 3'ü boş**
(multi-hop, belirsiz-coklu-dayanak, eksik-kanit — retrieval_eval'de yok).

**`data/bench/abstention_eval_v1.jsonl` — 330 satır (hepsi cevaplanamaz)**

| Alan | Dağılım |
|---|---|
| dilim | korpus-disi 230 · anlamsiz-ood 60 · eksik-kanit 40 |
| `verification_status` | verified 309 · rejected 21 |
| `verification_kind` | mechanical:manifest-absence 218 · model-cross-check 112 · **human 0** |
| dilim × durum | korpus-disi 218/12 · anlamsiz-ood 60/0 · eksik-kanit 31/9 |

230 `korpus-disi` satırın **113'ü hiçbir bağımsız denetçiden geçmedi** (yalnız
`script:validate_abstention_eval` mekanik etiketi).

**`data/bench/splits_v1.json`** — law-grouped, **22 test dokümanı / 34 dev dokümanı**,
kesişim boş. Bu denetimde `assign_split()` yeniden uygulanarak doğrulandı:

| | unanswerable | retrieval_eval-answerable |
|---|---|---|
| **dev** | 159 | 26 |
| **test** | 155 | 17 |

Soru-kimliği düzeyinde de ayrık (185 dev / 172 test, kesişim ∅). Kalibratörün n=185'i
**dev'in tamamına** eşit; `p2-calibration-dev-v1.json`'un 185 `per_question` satırının
hepsi bağımsız olarak dev'e düştü → **kalibrasyon ↔ test sızıntısı YOK** (G2.4 fiilen sağlanıyor).

### 6.2 Hangi yayınlanmış sayı hangi kümeden geliyor

| Yayınlanan sayı | Kaynak koşum | Küme |
|---|---|---|
| R@5 **0,8605** (37/43 ikili) / **0,8488** (kesirli) / R@20 **0,9302** | `20260830-1611-6d5b345-hybrid.json` | retrieval_eval_v1, n=43, hibrit, int8 |
| int8 = float16 her k'da; 1-bit −7,0 puan R@20 | `a2-traincompat-oracle.json` | retrieval_eval_v1, n=43 |
| B1/B2 baseline (two-stage R@5 0,000 / exhaustive 0,070) | `baseline-v0idx-{exhaustive,twostage}.json` | retrieval_eval_v1, v0 indeks |
| Aday süpürmesi 200/500/1000/exhaustive | `b2-traincompat-twostage-c*.json` | retrieval_eval_v1 |
| tau=0,5037, kapsama %2,2, AUROC 0,7817, ECE 0,0341 | `p2-calibration-dev-v1.json` + `calibrator.json` | retrieval_eval(dev 26) + abstention_eval(dev 159) = 185 |
| Görsel-only R@5 0,1163 | `verified-production-exhaustive.json` | retrieval_eval_v1, 1-bit |

**Hiçbir recall koşumu `abstention_eval_v1.jsonl` üzerinde koşmadı** — o dosya yalnız
kalibrasyon fitinde negatif örnek olarak kullanılıyor.

### 6.3 bench_v2 yokluğu

Spec §5.1 ≥120 answerable + ≥30 unanswerable, insan-doğrulamalı bir set istiyor.
`data/bench/bench_v2.jsonl` **hiçbir yerde yok**. Yerine geçen `retrieval_eval_v1 + abstention_eval_v1`
kombinasyonu **43 answerable** taşıyor (hedefin ~1/3'ü) ve 378 satırın **3'ü (%0,8)**
insan onaylı → spec'in "insan-doğrulamalı" şartını da karşılamıyor.

### 6.4 "İnsan doğrulaması" iddiası taraması (repo geneli)

| Yer | İddia | Hüküm |
|---|---|---|
| `README.md:358`, `:382`, `:161-169` | "3/48 rows human-verified, 45 model-cross-checked" | **DOĞRU** — üç ayrı yerde tekrarlanan çekince |
| `data/bench/retrieval_eval_v1.README.md` | "Bu set insan-doğrulanmış DEĞİLDİR… 3'ü insan" | **DOĞRU** |
| `data/bench/abstention_eval_v1.README.md` | "0'ı insan onayından geçmiştir" | **DOĞRU** (ama üstteki özet tablo bayat: 300/286/14 diyor, dosya bugün 330/309/21) |
| `2026-08-27-p0-gate.md:35-48, 449-452` | "insan-doğrulanmış olarak alıntılanamaz" | **DOĞRU** |
| `p0-decision-log.md:278` | "yalnız 3/48 insan onaylı" | **DOĞRU** |
| **`src/belge_gozu/app/static/index.html:454`** | **"43 soruluk insan-doğrulamalı retrieval_eval"** | **YANLIŞ — canlı serviste duruyor.** 43 answerable satırın yalnız 3'ü (aslında hepsi de değil: 3 insan satırının 3'ü de answerable) insan onaylı |
| `index.html:347` | "6 çipin hepsi retrieval_eval setinden ya da yazım varyantından" | **YANLIŞ** — 2 çip için doğru değil |

**Ek bulgu (bu denetimde ortaya çıktı):** `--only-verified` bayrağı bugünkü veride
**etkisiz**. `bench/dataset.py:114-132` yalnız `verification_status != "verified"`
filtreliyor, `verification_kind`'a bakmıyor; retrieval_eval'nin 48/48'i `verified` olduğundan
`--only-verified` hiçbir satırı elemiyor. Kanıt: `verified-production-exhaustive.json`
(only_verified=True) ile `a2-traincompat-1bit-exhaustive.json` (False) **birebir aynı**
overall sayıları veriyor. p0-gate'in "doğrulanmış set üzerinde sayılar DEĞİŞMEDİ"
gözlemi bu yüzden ampirik bir teyit değil, **totolojidir** — bu çerçeveleme başka bir
yerde tekrar kullanılmamalıdır.

---

## 7. "Kalan iş" önceliklendirmesi

Büyüklük: **S** ≤ yarım gün · **M** 1-3 gün · **L** > 3 gün (tek geliştirici).

### (a) P2'yi BİTİRMEK için zorunlu

| # | İş | Boyut | Bağımlılık |
|---|---|---|---|
| a1 | **T4: `bench/answer_eval.py` + `bench answers` CLI** — answerable/unanswerable koşum harness'ı; `AnswerRecord`, citation precision/completeness, false-answer/false-abstain | **M** | Yok (veri hazır: retrieval_eval 43 + abstention_eval 330) |
| a2 | **T3 kalanı: citation metrikleri** (G2.2'nin hesap yeri) | **S** | a1 |
| a3 | **G2 koşum kota planı** — 2×20 çağrı/gün ücretsiz kota vs ücretli katman; önbellek + güne bölme. **KULLANICI KARARI bekliyor** | **S** (karar) / **M** (koşum) | Kullanıcı |
| a4 | **T12: p2-gate.md + test-split final koşumu** (G2.1-G2.8 satır satır) | **M** | a1, a2, a3 |
| a5 | **T8 kalanı: `/healthz`'e `calibrator` alanı + `abstain_reason` API'de** (G2.6/G2.8'in test kanıtı) | **S** | Yok |
| a6 | **Kapı bayraklarının üretim kararı** — `gate_calibrated` bugünkü tau'da %2,2 kapsama veriyor; hangi çalışma noktasıyla açılacağı bir politika kararı | **S** (karar) | a4 |
| a7 | **T11: fine-tuning kapısının resmî değerlendirmesi** (koşul 2 zaten sağlanıyor: paraphrase R@5 0,2857 < %80) | **S** | a4 |
| a8 | *(G2 öncesi kaçınılmaz)* **G1'in adjudike edilmesi** — en azından dar bir `p1-gate.md`: bugün ölçülü olan G1.1/G1.2/G1.4/G1.5/G1.6'yı sayıyla yaz, G1.3/G1.7'yi "ölçülmedi + sebep" diye kaydet | **S** | Yok — veriler zaten mevcut |

### (b) Yayın / portfolyo değeri yüksek HIZLI kazanımlar

| # | İş | Boyut | Bağımlılık |
|---|---|---|---|
| b1 | **GitHub'a push** — repo oluştur, `feat/…` dalını yayınla, CI'yı ilk kez koştur. *Projenin en büyük tek boşluğu; 147 commit ve 666 test görünmez durumda* | **S** | Yok |
| b2 | **Y29 + Y30 düzeltmesi** (`index.html:454`, `:347`) — canlı UI'daki iki yanlış iddia; projenin kendi dürüstlük standardının ihlali | **S** | Yok |
| b3 | **CI sertleştirme**: `uv sync --locked`, docker build adımı, (opsiyonel) `--extra ml` ile slow testlerin nightly koşumu | **S** | b1 |
| b4 | **Docker imajını bir kez gerçekten build et + smoke test** — `BG_HF_DATASET_REPO` set/fail-fast, `USER`, `HF_HOME`, `[tool.uv]` CPU torch index, `.dockerignore` | **M** | b3 |
| b5 | **HF hub'a taze indeks push** (int8 + `page_texts.parquet`) + `revision=` pinleme + `hf_token` alanı + pull atomikliği (K21/C11/C28) | **M** | b4 |
| b6 | **Space kararı** — PRO alınacak mı? Alternatif: statik SDK ile "canlı olmayan" vitrin, ya da Render/Fly/HF olmayan bir CPU host. **KULLANICI KARARI** | **S** (karar) / **M** (uygulama) | b4, b5 |
| b7 | **K18 ölü telemetri sütunları + K9/K10 bench teşhis etiketleri** — sessizce yanlış rapor üreten üç yer | **S** | Yok |
| b8 | **README'ye p1/p2 sonuç tabloları + `.env.example`** | **S** | a4 |

### (c) Bilimsel derinlik

| # | İş | Boyut | Bağımlılık |
|---|---|---|---|
| c1 | **T7: dense metin kanalı** (BGE-M3 / multilingual-E5) — paraphrase dilimi R@5 0,2857'de sıkışmış; sözcüksel tavanın tek çıkışı. G1.2'nin tek FAIL'i | **L** | Yok (F1 ablasyon çerçevesi hazır) |
| c2 | **T10: cross-encoder reranker** — recall kapısı artık geçiliyor (R@20 0,9302); G1.3 ölçülebilir | **M** | c1 tercihen |
| c3 | **bench_v2 (120 answerable + 30 unanswerable)** — spec §5.1; law-grouped split hazır | **L** | Yok |
| c4 | **İnsan doğrulaması** — retrieval_eval'nin ≥30 satırı + abstention_eval örneklemi; `verify_retrieval_eval --review` kuyruğu kusurunun (K8) düzeltilmesi + `require_human` bayrağı | **L** | K8 fix (**S**) |
| c5 | **T3 (madde segmentasyonu)** — ilke 11'in tek gerçek ihlali; citation granülaritesi ve `gold_article_ids`'in aktifleşmesi | **M** | Yok |
| c6 | **T10 (LLM-judge + PPI)** — c4'ün ≥30 insan çifti ön koşulu | **M** | c4 |
| c7 | **Ablasyon matrisinin (spec §7) tamamlanması** — E1/F1/F2/F3/G1 satırları | **L** | c1, c2 |
| c8 | **T2 OCR fallback** — bugün yalnız 5 sayfayı etkiliyor (1 boş + 4 çok kısa); tarihî dilim R@50 zaten 1,0 → **düşük öncelik**, ilke 10 uyumu için kaydedilir | **M** | Yok |

### (d) Ertelenebilir / teknik borç

| # | İş | Boyut |
|---|---|---|
| d1 | Y2: BM25 inverted index (bugünkü ölçekte ~2-8 ms, acil değil) | M |
| d2 | `EvalReport.oracle_gap` alanı (G0.4 araç sapması) | S |
| d3 | C19 test env izolasyonu autouse fixture; C41 FloatIndex padding invariantı; C42 ölü sabit; C10/S5 CLI guard'ları | S |
| d4 | Y11 `/stats`/`/metrics` erişim kontrolü + tam tablo taraması | S |
| d5 | UI nit'leri: Y33 (sahte pacing), Y34, Y35, Y36, Y37, Y38 (a11y), Y40, Y43, Y45 | M (toplu) |
| d6 | Telemetri/katalog birleştirmeleri (D12, S22-S23, D11, S55, D14), vokabüler enum'ları (D4-D6, S36, D25) | M |
| d7 | `abstention_eval_v1.README.md` üstteki özet tablosunun güncellenmesi (300/286/14 → 330/309/21) | S |
| d8 | PDL "minor (deferred)" kalemleri (T1/T2/T4/T15), NEW-1, N1/N2 nit'leri | S |
| d9 | Korpus genişletmesi (1475), LocalVLM, agentic derin arama — spec §10 kapsam dışı, ilke 23 gereği P2 sonrası | L |

### 7.1 En kritik 5 eksik (tek cümlelik)

1. **Proje GitHub'da değil (0 remote, main 85 commit geride)** — 147 commit / 666 test / 3 faz görünmez; CI hiç koşmadı.
2. **Canlı UI'da yanlış iddia** (`index.html:454` "insan-doğrulamalı retrieval_eval") — projenin kendi dürüstlük standardını ihlal ediyor ve iç incelemede KRİTİK işaretlenmiş olmasına rağmen düzeltilmemiş.
3. **G1 hiç adjudike edilmemiş, G2 koşumu yok** — hibrit üretimde default açık; ölçülü sayılar (R@50 0,9302 < %95; paraphrase 0,5714 < %90) bir kapı raporunda kayıtlı değil.
4. **Dağıtım zinciri hiç doğrulanmamış** — Docker imajı bir kez bile build edilmedi, `--pull` sessiz no-op, HF'teki indeks P1 öncesi (metin artefaktı yok), Space PRO nedeniyle yok.
5. **P2'nin ölçüm ayağı eksik** — `bench/answer_eval.py` + `bench answers` yok; G2.1/G2.2 için hiçbir koşum yapılamaz, dolayısıyla verifier/kalibre kapıları meşru biçimde açılamıyor.
