# Belge-Gözü RAG Kalite v2 — Master Roadmap

**Tarih:** 2026-08-26 · **Durum:** Kullanıcı onayı bekliyor
**Spec:** `docs/superpowers/specs/2026-08-26-belge-gozu-rag-quality-v2-design.md`

> Bu roadmap üç sıralı, kapı-korumalı implementation planını yönetir:
>
> - **P0:** `docs/superpowers/plans/2026-08-26-belge-gozu-p0-retrieval-correctness.md`
> - **P1:** `docs/superpowers/plans/2026-08-26-belge-gozu-p1-hybrid-retrieval.md`
> - **P2:** `docs/superpowers/plans/2026-08-26-belge-gozu-p2-selective-answering.md`
>
> `2026-08-26-belge-gozu-plan2.md` **supersede edilmiştir** (eşleme: §8). Telemetri planı
> (`2026-08-26-telemetry.md`) tamamlanmıştır ve korunur; P0 Task 13 onun şemasını
> genişletir, değiştirmez.

## 1. Bağımlılık grafiği

```
P0 (retrieval correctness + ölçüm altyapısı)
 ├─ üretir: bench veri modeli + retrieval_eval set + harness + oracle'lar
 ├─ üretir: IndexManifest + fail-fast + maskeli encoder + format kararı (A1/A2/A3)
 ├─ üretir: ExhaustiveBinaryRetriever (üretim yolu), kuantizasyon kararı (C2)
 └─ KAPI G0 ──geçmeden──X──> P1 default entegrasyonu, P2 kalibrasyonu BAŞLAYAMAZ

P1 (hybrid + structure-aware retrieval)   [G0 geçti şartıyla]
 ├─ tüketir: P0 harness/oracle/manifest/format kararı
 ├─ üretir: PageText + Article + BM25 + Dense + RRF + reranker + EvidencePack
 ├─ üretir: visual-only / hybrid-production modları + full benchmark (120+30)
 └─ KAPI G1 ──geçmeden──X──> P2 BAŞLAYAMAZ

P2 (selective answering + citation + kalibrasyon)   [G1 geçti şartıyla]
 ├─ tüketir: P1 EvidencePack + full benchmark + ConfidenceFeatures kaynak sinyalleri
 ├─ üretir: verifier + claim-level citation + Calibrator + risk-coverage + outcome telemetry
 └─ KAPI G2 → yayın (README sonuç tabloları, bench yayını)
     └─ koşullu: fine-tuning alt planı (G2-FT kapısı ayrıca)
```

Kapı ihlali kuralı: bir sonraki fazın **hiçbir default-açık entegrasyonu** önceki kapı
raporu commit'lenmeden yapılamaz. Deneysel kod flag-kapalı olarak erken girebilir; bu
kural ürün yolunu korur.

## 2. Faz amaçları, giriş/çıkış koşulları

| Faz | Amaç | Giriş koşulu | Çıkış koşulu (kapı raporu) |
|---|---|---|---|
| P0 | Retrieval'ı ölçülebilir ve doğru yapmak: bozuk Stage-1'i kaldır, format/mask/manifest düzelt, oracle'ları kur | Bu roadmap'in kullanıcı onayı | `docs/research/findings/2026-XX-XX-p0-gate.md`: §5 G0 satırlarının tamamı sayıyla PASS |
| P1 | Aday havuzunu hibrit kanallarla ≥%95 Recall@50'ye çıkarmak, reranker'la top-5 kaliteyi kanıtlamak | G0 PASS commit'i | `.../p1-gate.md`: §5 G1 satırları PASS; Sorgu A top-5 regression yeşil |
| P2 | Yalnız kanıtlı yanıt sunmak: claim-level citation + kalibre abstention | G1 PASS commit'i | `.../p2-gate.md`: §5 G2 satırları PASS |

## 3. Ortak interface'ler (fazlar arası sözleşme)

Tam imzalar ilgili planların task'lerinde; burada fazlar arası taşınanlar:

| Interface | Tanımlandığı yer | Tüketen |
|---|---|---|
| `IndexManifest`, `QueryFormat` (`index/manifest.py`) | P0 T1 | P0 T4/T11, P1 tüm indeks kurucuları, P2 threshold versiyonlama |
| `Encoder.encode_query/encode_pages` (maskeli, formatlı) | P0 T2 | P0 T5/T9, P1 visual kanal |
| `PackedIndex` (manifest'li, padding'siz) | P0 T3 | P0 T5, P1 F1 ablasyonları |
| `ExhaustiveBinaryRetriever.search(query, k) -> list[PageHit]` | P0 T5 | app, P1 visual kanal, tüm harness koşumları |
| `BenchQuestion`, `load_bench` (`bench/dataset.py`) | P0 T6 | P0-P2 tüm eval'ler |
| `recall_at_k/mrr/ndcg_at_k/bootstrap_ci` (`bench/metrics.py`) | P0 T7 | P0-P2 |
| `run_retrieval_eval(...) -> EvalReport` + `QuestionDiagnostic/StageRecord` | P0 T8 | P0 T11/T12/T14, P1 T9/T13, P2 raporları |
| `FloatIndex` + `native_float_ranks` (`bench/oracle.py`) | P0 T9 | P0 T12, P1 oracle-gap raporları |
| `PageText`/`Article` parquet şemaları (`corpus/text.py`, `corpus/articles.py`) | P1 T1/T3 | P1 BM25/dense/rerank, P2 verifier (madde metni) |
| `QueryFacets`, `make_variants` (`retrieval/query.py`) | P1 T4/T5 | P1 fusion, P2 ConfidenceFeatures |
| `HybridRetriever.search(query, k) -> list[PageHit]` + `retrieve_evidence(query, k) -> EvidencePack` | P1 T8/T11 | app, P2 AskService |
| `EvidencePack`/`EvidenceUnit` (`retrieval/evidence.py`) | P1 T11 | P2 verifier/citation |
| `VerifiedAnswer`, `ClaimVerdict`, `CitationRef` (`answer/verify.py`) | P2 T1 | app/UI, outcome telemetry |
| `ConfidenceFeatures`, `Calibrator` (`answer/calibrate.py`) | P2 T5/T6 | P2 selective answering |

Tip tutarlılığı kuralı: bu tabloda geçen ad ve imzalar üç planda birebir aynıdır;
plan self-review'ları bunu doğrular.

## 4. Migration, feature flag ve rollback stratejisi

**Feature flag'ler** (tümü `config.py`'de, env ile geçersiz kılınabilir):

| Flag | Varsayılan | Faz | Rollback |
|---|---|---|---|
| `retrieval_pipeline: "exhaustive" \| "two-stage"` | `"exhaustive"` (P0 T5 sonrası) | P0 | `"two-stage"`'e döndür — eski kod silinmez, ablasyon-only konumda yaşar |
| `query_format_id: "cpe-0.3.18" \| "train-compat-v1"` | A/B kararına göre (P0 T11) | P0 | Manifest'i eski indeksi gösterecek şekilde değiştir; iki indeks dizini de diskte/HF'te sürümlü durur |
| `quantization: "sign-1bit" \| "int8" \| "float16"` | C2 kararına göre (P0 T12) | P0 | İndeks dizinleri sürümlü; config geri alınır |
| `retrieval_mode: "visual-only" \| "hybrid-production"` | `"visual-only"` (P1 kapısına kadar) | P1 | Mod anahtarı tek satır |
| `rerank_enabled: bool` | `False` (G1 kanıtına kadar) | P1 | Kapat |
| `ocr_fallback_enabled: bool` | `False` (tarihî dilim ölçümüne kadar) | P1 | Kapat |
| `evidence_verifier_enabled: bool` | `False` (P2 kanıtına kadar) | P2 | Kapat → sistem P1 davranışına döner (yanıt + sayfa-düzeyi citation + eski abstain) |
| `selective_answering_enabled: bool` | `False` | P2 | Kapat |

**Migration:** her yeniden indeksleme yeni bir dizine yazar
(`data/index-<format_id>-<quant>/`), manifest'iyle birlikte HF Datasets'e ayrı
revision olarak push edilir. `data/index/` her zaman "aktif" indekse işaret eden kopya/
symlink değil, **config'te seçilen dizindir** (`index_dir` zaten config'te). Eski indeks
silinmez; en az bir önceki üretim indeksi HF'te etiketli tutulur.

**Rollback ilkesi:** her kapı raporu, o fazın flag'lerini kapatınca sistemin hangi
davranışa döndüğünü tek cümleyle yazar. Uygulama sırasında bir task üretim yolunu
bozarsa ilgili flag kapatılır, fix ayrı commit'le gelir.

## 5. Quality gate tablosu

| Kapı | Ölçüt | Eşik | Ölçüm aracı |
|---|---|---|---|
| G0.1 | `k4721:4` corpus coverage (indekste, meta'da, checksum'da) | kanıtlı | `bench run` coverage bölümü |
| G0.2 | Sorgu A + B kalıcı regression setinde | evet | `tests/retrieval/test_semantic_retrieval_eval.py` (slow) |
| G0.3 | Üretim candidate generator gold Recall@candidate | ≥ %98 (exhaustive yol: %100 tanım gereği; Stage-1 varyantı ancak bu eşikle girebilir) | harness candidate_survival |
| G0.4 | Exhaustive binary + native float oracle her koşumda karşılaştırılabilir | evet | `bench oracle` + EvalReport oracle-gap |
| G0.5 | İndeks/processor uyumsuzluğu fail-fast | test kanıtlı | `tests/index/test_manifest.py`, `tests/app/test_compat.py` |
| G0.6 | Padding satırları skorlanmıyor (yeni indekste 0 all-zero satır) | 0 satır | `bench run` index-audit + birim test |
| G0.7 | Kuantizasyon kaybı sayılandırıldı (C1/C2 tablosu) | rapor var | p0-gate.md tablosu |
| G0.8 | Sorgu B (`Yerleşim yeri nedir?`) üretim hattında top-5 | evet | semantic retrieval_eval testi |
| G0.9 | Baseline raporu (mevcut mimari) üretildi | commit'li | `docs/research/findings/` |
| G1.1 | Candidate-union Recall@50 (overall, verified full bench) | ≥ %95 | harness |
| G1.2 | Kritik dilimlerde Recall@50 (`dogrudan-madde`, `paraphrase`, `madde-numarali`, `ayni-kanun-hard-negative`) | ≥ %90 her biri | harness per-slice |
| G1.3 | Reranker kazancı Recall@5 / MRR / nDCG@5, bootstrap CI ile | CI alt sınırı > 0 | harness + bootstrap_ci |
| G1.4 | Sorgu A dayanak sayfası final top-5 | evet (zorunlu regression) | semantic retrieval_eval testi |
| G1.5 | Hybrid vs visual-only delta + latency/memory maliyeti raporlu | rapor var | p1-gate.md |
| G1.6 | Kazanç göstermeyen katman default kapalı | denetim | p1-gate.md flag tablosu |
| G1.7 | HF Space bütçeleri: index boyutu + peak RAM + p50/p95 ölçülü | rapor var; p95 hedefi raporda gerekçeli | p1-gate.md |
| G2.1 | Unanswerable bench'te false supported-answer | ≤ %2 | answer_eval |
| G2.2 | Claim-level citation support precision | ≥ %98 | answer_eval |
| G2.3 | Risk-coverage eğrisi raporlu (coverage bedeli görünür) | rapor var | calibrate raporu |
| G2.4 | Kalibrasyon verisi ↔ final test ayrıklığı | kanıtlı (split dosyası) | splits_v1.json denetimi |
| G2.5 | Threshold'lar retriever/index/model revision'a versiyonlu | evet | `data/calibration/<revision>/` |
| G2.6 | Verifier geçmeyen yanıt kesin yanıt olarak gösterilmiyor | test kanıtlı | app testleri |
| G2.7 | Auto-citation fallback kaldırıldı | test kanıtlı | `tests/answer/test_gemini.py` |
| G2.8 | Hedef sağlanamadığında güvenli fallback davranışı tanımlı | dokümante + test | p2-gate.md |

## 6. Deneylerin çalıştırılma sırası

1. **P0-baseline:** mevcut indeks + mevcut format, retrieval_eval set → B1/B2 (Stage-1 açık/kapalı,
   aday sayısı taraması) + coverage. Çıktı: baseline EvalReport (G0.9).
2. **P0-format:** f16 master embedding'lerle A1 (mevcut format) reindex → C1 float oracle;
   A2 (train-compat) reindex → A1-vs-A2; A3 (ST MultiVectorEncoder) sorgu/belge çapraz
   kontrol. Karar: `query_format_id` + doc format.
3. **P0-quant:** kazanan formatın f16 master'ından 1-bit/int8 türet → C2 + D1. Karar:
   `quantization`. (Residual yalnız int8 de yetersizse; koşul p0 planında.)
4. **P1-kanallar:** F1 tek-kanal ve ikili kombinasyonlar (dev split) → dense model seçimi,
   BM25 varyant seçimi. Ardından E1 (varyant füzyonu) ve F2 (soft-boost).
5. **P1-rerank:** union recall G1.1/G1.2'yi dev'de geçiyorsa G1 reranker açık/kapalı;
   F3 OCR tarihî dilimde.
6. **P1-final:** verified full bench üzerinde kilitli konfigürasyonla tek koşum → G1 raporu.
7. **P2:** H1 verifier açık/kapalı; kalibrasyon fit (calibration split) → H2 raw-vs-calibrated;
   final test tek koşum → G2 raporu.

Kural: her karar deneyi **dev/calibration split'inde** koşar; **test split'i** yalnız kapı
raporlarında, faz başına bir kez kullanılır.

## 7. Benchmark ve index versioning

- Benchmark dosyaları: `data/bench/retrieval_eval_v1.jsonl` (P0), `data/bench/bench_v2.jsonl`
  (P1), `data/bench/splits_v1.json` (law-grouped). Şema/dilim değişikliği → yeni sürüm
  dosyası; eski dosya silinmez. Sonuçlar `data/bench/results/<run_id>.json`; `run_id` =
  `<tarih>-<git-kisa-sha>-<index-format>-<quant>`.
- İndeks sürümleri: `data/index-<format_id>-<quant>/` + `manifest.json`; HF Datasets'e
  revision'lı push. README/rapor her sayıyı `run_id` ile anar.
- Kalibrasyon artefaktları (P2): `data/calibration/<index_revision>/` altında model +
  threshold + fit raporu.

## 8. Eski Plan 2 eşleme tablosu

| Plan 2 task | Karar | Gerekçe / yeni yer |
|---|---|---|
| T1 Hijyen üçlüsü (manifest testi, TLS, çok-chunk hizalama) | **Aynen korunur** | P0 Task 15'e devralındı (bağımsız regresyon ağı) |
| T2 Bench veri modeli (CSV, 3 dilim) | **Değiştirildi** | P0 Task 6: JSONL, 12 dilim, answerable/unanswerable, span'lar, insan-doğrulama durumu, law-grouped split |
| T3 Metrikler + harness | **Değiştirildi** | P0 Task 7-8: + MRR, bootstrap CI, stage-bazlı teşhis, oracle-gap, koşum künyesi |
| T4 OCR + metin-RAG baseline (yalnız rakip) | **Değiştirildi (terfi)** | P1 T1-T7: metin sinyali artık üretim kanalı (`hybrid-production`); "rakip" ölçümü F1 ablasyonunun visual-only satırı olarak yaşar |
| T5 Sorgu yeniden yazımı (tek sorgu replacement) | **Değiştirildi** | P1 T5: orijinal daima korunur; deterministik varyantlar + RRF; LLM rewrite yalnız ölçülmüş kazançla (E1) |
| T6 VLM reranker (pointwise 0-10, top-20) | **Değiştirildi** | P1 T10: risk analizi + text cross-encoder birincil; recall-gate önkoşul (top-20'de gold yoksa rerank faydasız — Plan 2'nin ana kusuru) |
| T7 LocalVLM answerer | **Ertelendi** | Retrieval correctness öncesi vitrin özelliği (ilke 23); P2 sonrası backlog |
| T8 1475 sayılı Kanun korpus eki | **Ertelendi** | Korpus P0-P1 boyunca checksum'la donduruldu (benchmark bütünlüğü + A/B'lerde değişken izolasyonu). P1 kapısından sonra korpus rev v0.2 olarak, coverage ölçümüyle birlikte eklenir |
| T9 Benchmark taslak + kullanıcı doğrulama | **Kısmen korunur** | P0 Task 10 (retrieval_eval) + P1 Task 12 (full): ajan taslağı serbest ama final'e yalnız insan onayıyla; şema zenginleşti; leakage kuralları eklendi |
| T10 Kalibrasyon + ablasyon + tablolar | **Bölündü** | Retrieval ablasyonları P0/P1'e; threshold kalibrasyonu P2'ye (ilke 21: mimari sabitlenmeden kalibrasyon yok — Plan 2 bunu ihlal ediyordu) |

Plan 2'nin ayrıca eleştirilen noktaları: Stage-1 oracle-recall ölçümünün hiç olmaması
(P0'ın merkezi), model/processor contract + index manifest eksikliği (P0 T1-T4),
gerçek-model semantic retrieval_eval testlerinin yokluğu (P0 T10), pointwise VLM'in aday başına
API maliyeti/skor güvenilirliği (P1 T10 risk analizi).

## 9. Risk register

| Risk | Olasılık | Etki | Azaltma |
|---|---|---|---|
| Train-compat format da Türkçe'de yetersiz kalır (model İngilizce-eğitimli) | Orta-yüksek | P0 sonrası visual kanal hâlâ zayıf | Beklenen durum: P0 hedefi düzeltme değil doğru ölçüm; ürün kalitesi P1 hibride bağlandı. Visual kanal "requires_visual" dilimlerde katkısıyla değerlendirilir |
| Batch-vs-single encode determinizmi sağlanamaz (left-pad + pozisyon etkisi) | Orta | Reindex maliyeti artar | P0 T2 determinism testi ölçer; sapma varsa build batch=1'e düşer (yavaş ama doğru; süre raporlanır) |
| `pad_token_id` 128002 uyarısının bilinmeyen etkisi | Düşük-orta | Reproducibility | Manifest'e kayıt; format A/B'de sabit tutulur; davranış değişimi izlenirse ayrı incelenir |
| f16 master embedding üretimi uzun sürer (~1 saat/dizin × varyant) | Yüksek | Takvim | Varyant sayısı sınırlı (A1, A2); f16 tek kez üretilir, kuantizasyonlar ondan türetilir |
| HF Space: 2 vCPU'da exhaustive binary ~1.2 s (M4) → 3-6 s tahmini | Orta | UX gecikmesi | P1 G1.7 bütçe ölçümü; gerekirse int8 SIMD/chunk ayarı veya PLAID-tarzı centroid aday üretimi (Recall ≥ %98 kapısıyla) |
| Cross-encoder CPU'da yavaş | Orta | p95 hedefi | Aday sayısı/model boyutu ablasyonu; madde-metni kısaltma; rerank flag'i |
| Dense model bellek + indeks boyutu Space'i zorlar | Orta | Deploy | Madde-düzeyi embedding (sayfa değil) ~2-4k vektör; küçük model adayları; G1.7 |
| Benchmark insan-doğrulama darboğazı | Yüksek | Takvim | RetrievalEval 30-50 ile P0 ilerler; full set P1 sonuna kadar paralel doğrulanır; doğrulanmamış soru kapı kararına giremez |
| Gemini kota (≈20 çağrı/gün) P2 verifier'ı sıkar | Yüksek | P2 deney hızı | Verifier çağrıları önbellekli + toplu koşumlar güne bölünür; claim segmentation deterministik ön-parçalama ile çağrı sayısı düşürülür; bütçe her runbook'ta açık |
| LLM-judge güvenilirliği | Orta | P2 metrik geçerliliği | ARES-PPI yaklaşımı: yalnız insan-doğrulamalı subset ile kalibre edilmiş yardımcı |
| Tek geliştirici + uzun plan seti | Yüksek | Sürüklenme | Her task bağımsız commit + kapı raporları; fazlar kendi başına değerli çıktı bırakır |

## 10. HF Space kaynak riskleri (özet bütçe)

Ölçülen yerel referanslar (M4 Pro): packed 1-bit tokens 60.5 MB (4222 sayfa);
exhaustive binary ~1.2 s/sorgu; f16 master ~968 MB (Space'e inmez); int8 ~484 MB.
Space hedefi (free CPU, 16 GB RAM, 2 vCPU): indeks + metin + dense artefaktları
< 2 GB disk / < 4 GB RAM; sorgu encode + retrieval + rerank p95 raporda gerekçeli
hedefle (G1.7'de sayı verilecek; şimdiden taahhüt edilmez — ölçüm önce). Cold-start:
model + indeks yükleme süresi p1-gate raporunda ölçülür.

---

## 11. P0 sonrası devir notları (2026-08-27, ölçümle güncellendi)

P0 tamamlandı; kapı raporu `docs/research/findings/2026-08-27-p0-gate.md` (KOŞULLU GEÇTİ,
8/9). Aşağıdakiler P1'e devredilmiştir — her biri ölçülmüş bir gerekçeyle:

| # | Devredilen iş | Ölçülmüş gerekçe | Ruling |
|---|---|---|---|
| D1 | **int8 indeksin üretim yoluna bağlanması** (P1'in İLK işi olmalı) | int8, float16 ile her k'da birebir aynı; 1-bit'in R@20 kaybı 7.0 puan VE 1-bit 4.5× daha yavaş (1.08 s vs 0.24 s). Üretim hâlâ 1-bit çünkü `ExhaustiveBinaryRetriever` yalnız `PackedIndex` tüketiyor. | R16 |
| D2 | **Eşik/abstain kalibrasyonu** — P2'nin konusu ama P1'in kanal seçimini de etkiler | Eşik 60.0 ARTIK AYIRMIYOR: cevaplanamaz sorular 59.65-71.95, cevaplanabilirler 59.85-78.50 — dağılımlar örtüşüyor. `tests/retrieval/test_semantic_retrieval_eval.py`'de xfail(strict=True) ile kilitli. | — |
| D3 | **Uzun sorgu (c001) hâlâ 1221. sırada** — G1.4'ün hedefi | Format düzeltmesi 3127→1221 getirdi ama top-5 için hibrit metin kanalı şart. | — |
| D4 | `FloatIndex`'in `bench/` paketinden `index/`e taşınması (layering) | `index/quantize.py` hâlâ `bench.oracle`'dan import ediyor; `chunk_bounds`/`git_commit` taşındı, bu kaldı. | — |
| D5 | Benchmark insan doğrulaması (retrieval_eval 48 satır hâlâ `draft`) | Bütün kapı sayıları bu nedenle geçicidir; mekanizma kapıları (G0.1/G0.4/G0.5/G0.6) etkilenmez, recall tabanlı olanlar yeniden hesaplanmalı. | — |
| D6 | colpali-engine ↔ Sentence Transformers arası ~%0.8 sign farkı | CPU/fp32'de de sürüyor (dtype değil implementasyon kaynaklı); mean cosine ≥ 0.9995. Binary indeksin referans-sadakati açık soru. | — |

**Güncelleme (2026-08-29, commit `b790f6c`):** D1 ve D4 KAPANDI — üretim int8'e geçti
(`load_scorable_index` manifest dispatch'i, tek normalize skor ölçeği ≈[−1,1], generic
`ExhaustiveRetriever`; `FloatIndex` → `index/float_store.py`). D2 güncellendi: eşik 0.58'e
MEKANİK ölçek taşımasıyla geçirildi (kalibrasyon değil; binary@60 çalışma noktası birebir),
int8'de de ayırmıyor (answerable medyan 0.6250 / unanswerable 0.6550 — örtüşme temsilden
bağımsız), xfail kilidi int8 sayılarıyla duruyor. D3 güncellendi: uzun sorgu int8'de 1221→**664**
(cırcır `retrieval_regression_expectations.json` quantization anahtarıyla 664'e çekildi); top-5 için hibrit
kanal gereksinimi değişmedi. D5 güncellendi: 48/48 satır doğrulandı (3 insan + 45
model-cross-check — insan-doğrulanmış SAYILMAZ, künye `retrieval_eval_v1.README.md`). Eşlik eden
bağlaşım denetimi: `docs/research/findings/2026-08-29-config-coupling-audit.md`.

**Güncelleme 2 (2026-08-29 akşam):** Autoresearch döngüsü (Karpathy metodolojisi,
`research/`) metin kanalını ÖLÇTÜ: BM25+F5+stoplist+pencere-yönlendirme reçetesi retrieval_eval
R@5 0.2326→**0.8140**; eşit-RRF ölçümle reddedildi (0.395); görselin @5 benzersiz katkısı
F5 sonrası 0 soru. Bulgular: `docs/research/findings/2026-08-29-autoresearch-text-channel.md`.
Ruling R23: P1 kapsamı "ölçülmüş reçetenin üretimleştirilmesi" olarak daraltıldı
(T1kısmi+T6+T8-revize; kalanı backlog — P1 planındaki durum notuna bakınız). Hibrit eşik
mekanik taşıma: BM25 ölçeğinde **10.6** (band (10.528,10.712], 42/43+4/5 birebir; ayrım
yine yok — kalibrasyon P2). D3 (uzun sorgu) reçetede KAPANDI: gold 664→2 (üretim teyidi
entegrasyon koşumunda).

**P1 giriş koşulu güncellemesi:** P1'in F1 kanal ablasyonu, görsel kanalı **int8 üzerinde**
ölçmelidir (1-bit değil) — aksi halde görsel kanal kendi tavanının 7 puan altında
raporlanır ve hibrit füzyon kararı çarpıtılır.
