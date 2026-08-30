# Belge-Gözü Telemetri: Merkezî Metrik ve Gözlemlenebilirlik Sistemi — Tasarım

Tarih: 2026-08-26 · Durum: onaylı kararlar üstüne yazıldı (grilling 1. tur) · Spec sahibi: proje

## 1. Amaç ve kapsam

Demo'daki hataları, darboğazları ve davranışları **sayıya bağlamak**: her isteğin ham
olay kaydı (makale analizi için gerçeğin kaynağı) + Prometheus/Grafana canlı görünümü.
Plan 2'nin offline kalite benchmark'ından (NDCG/Recall) ayrı bir eksen: bu sistem
**çalışma-anı** davranışını ölçer (gecikme, verim, token hızı, abstain oranı, maliyet).
İkisi makalede birleşir: "kalite" Plan 2'den, "sistem davranışı" buradan.

Kapsam DIŞI (Faz 2 backlog, §13): streaming TTFT, Redis (embedding önbelleği /
rate-limit), OpenTelemetry trace, Grafana alerting, GPU/MPS örnekleme.

## 2. Onaylı kararlar (grilling 1. tur)

| # | Karar |
|---|---|
| Q1 | **(c)** Ham olay kaydı = gerçeğin kaynağı; dashboard onun üstünde görünüm |
| Q2 | **(A)** `prometheus-client` + `/metrics` endpoint'i; Prometheus+Grafana OrbStack compose ile lokal; uygulama tek başına da tam çalışır (HF Space tek-konteyner kısıtı) |
| Q3 | **(a)** `data/requests.sqlite` içinde yeni `events` tablosu; `belge-gozu metrics export` ile Parquet/CSV |
| Q4 | Soru metni **tam** kaydedilir; `BG_LOG_QUERY_TEXT=false` ile kapatılabilir (o durumda SHA-256 + uzunluk) |
| Q5 | Bulgular dizini: **`docs/research/`** |
| Q6 | **Akışsız kalınır**; `usage_metadata` yakalanır (token in/out, ort. token/sn, maliyet tahmini). TTFT/streaming Faz 2 |
| Q7 | **(a)** `scripts/loadgen.py`; varsayılan `/search` (Gemini kotası ≈20/gün korunur), `/ask` ancak açık bayrakla |

Bu tur bana bırakılan ikincil kararlar §4–§11'de gömülü ve **normatiftir**.

## 3. Mimari

```
istek → FastAPI route
          ├─ RequestTimer (toplam süre, inflight gauge)
          ├─ StageCollector (contextvar): retrieval/core + AskService içindeki
          │    with stage("query_encode") / ("stage1_hamming") / ("stage2_maxsim") / ("answerer")
          │    + annotate("tokens_in", …) (GeminiAnswerer'dan)
          ├─ olay birleştirme → EventRecorder.record()  → SQLite events (WAL)
          └─ PromMetrics.observe(event)                 → in-proc registry
                                                              ↓
                                              GET /metrics (Prometheus text format)
                                                              ↓ scrape (5s)
                                  OrbStack compose: Prometheus (9090) → Grafana (3000)
                                                       provisioned dashboard (repo'da JSON)
```

- **Telemetri asla isteği düşürmez** (mevcut ilke korunur): recorder ve olay
  birleştirme try/except ile sarılır; hata bir kez WARNING loglanır.
  **Öncelik netliği:** "istek asla düşmez" ilkesi "her istek bir satır yazar" beklentisinden
  önceliklidir — olay kurulumu (`build_event`) çökerse satır hiç yazılmaz, ama istek yine de
  başarılıdır (bkz. §14 kabul ölçütü 1).
- StageCollector **contextvar** tabanlıdır: kolektör yokken tüm çağrılar no-op
  (testler ve CLI yolları etkilenmez). FastAPI sync endpoint'lerinde contextvar,
  istek thread'ine taşınır; istek içi akış tek thread olduğundan güvenlidir.
- Eski `_log_db/_log_write/log` kaldırılır; `log` tablosu diskte kalır (migrasyon
  yok), `/stats` artık `events`'ten okur.

## 4. Yeni paket: `src/belge_gozu/telemetry/`

| Dosya | Sorumluluk |
|---|---|
| `schema.py` | `RequestEvent` (pydantic) — tek olayın tam sözleşmesi; `EVENTS_DDL` |
| `collect.py` | `StageCollector` + `stage(name)` ctx mgr + `annotate(k,v)` + contextvar |
| `recorder.py` | `EventRecorder`: WAL'lı sqlite, tek bağlantı + Lock, `busy_timeout=5000`, best-effort `record()` |
| `prom.py` | Registry, metrik tanımları, `observe(event)`, `render()` (`/metrics` gövdesi), startup gauge'ları |
| `export.py` | `events` → Parquet/CSV (pyarrow mevcut bağımlılık) |

Bağımlılık: `prometheus-client>=0.20` **ana** bağımlılıklara eklenir (saf Python,
Space imajına da girer).

## 5. Olay şeması (`events` tablosu)

Sabit kolonlar (SQL ile doğrudan sorgulanan alanlar) + `detail` JSON (geri kalan her şey):

```
id INTEGER PRIMARY KEY AUTOINCREMENT
ts TEXT               -- ISO-8601 UTC
endpoint TEXT         -- '/ask' | '/search'
status TEXT           -- /ask: 'answered' | 'abstained' | 'degraded' | 'error'
                      -- /search: 'ok' | 'error'
http_status INTEGER
total_ms REAL
encode_ms REAL        -- NULL olabilir (hata yolları)
stage1_ms REAL
stage2_ms REAL
answer_ms REAL        -- yalnız /ask + LLM çağrıldıysa
top_score REAL
margin_1_2 REAL       -- top1 − top2 (ayırt edicilik)
abstained INTEGER     -- 0/1, yalnız /ask
honest_miss INTEGER   -- 0/1 sezgisel: abstain DEĞİL ama yanıt 'bulamadım' içeriyor
                      -- (v0'ın ana bulgusu 13/17 idi; kataloğda 'heuristic' işaretli)
k INTEGER, candidates INTEGER
query_len INTEGER
query_text TEXT       -- BG_LOG_QUERY_TEXT=false ise NULL
query_sha256 TEXT     -- her durumda yazılır (dedup/korelasyon)
answer_len INTEGER
citations_n INTEGER
tokens_in INTEGER, tokens_out INTEGER
tokens_per_s REAL     -- tokens_out / (answer_ms/1000)
est_cost_usd REAL
error_type TEXT       -- exception sınıf adı (yalnız status='error')
detail TEXT           -- JSON: top-5 [{page_id,score}], model adları, device,
                      -- threshold, app_version — koşum künyesi (makale tekrarlanabilirliği)
```

İndeks: `(ts)`, `(endpoint, ts)`.

## 6. Prometheus metrik kataloğu (Faz 1 — tamamı bu inşada)

Adlandırma: `bg_` öneki, taban birim saniye, Prometheus adlandırma kuralları.

| Metrik | Tip | Etiketler | Ne söyler |
|---|---|---|---|
| `bg_http_requests_total` | Counter | `endpoint`, `status` | trafik + sonuç dağılımı (hata oranı = status='error'/toplam) |
| `bg_request_duration_seconds` | Histogram | `endpoint` | uçtan uca gecikme; p50/p95/p99 buradan |
| `bg_inflight_requests` | Gauge | `endpoint` | anlık eşzamanlılık / doyma |
| `bg_stage_duration_seconds` | Histogram | `stage` ∈ {query_encode, stage1_hamming, stage2_maxsim, answerer} | darboğaz ayrıştırması ("gecikme nerede?") |
| `bg_retrieval_top_score` | Histogram | — | skor dağılımı; eşik kalibrasyonu + drift |
| `bg_retrieval_score_margin` | Histogram | — | top1−top2; retrieval kararlılığı |
| `bg_abstain_total` | Counter | `reason` ∈ {threshold, degraded} | halüsinasyon freni sağlığı; kota hatası görünürlüğü |
| `bg_rate_limited_total` | Counter | `endpoint` | 429 sayısı (review L3, P1 fix round) — 422 kasıtlı dışarıda, framework düzeyi |
| `bg_honest_miss_total` | Counter | — | LLM 'sayfalarda bulamadım' dedi (sezgisel tespit) |
| `bg_llm_tokens_total` | Counter | `direction` ∈ {input, output} | hacim; maliyet tabanı |
| `bg_llm_tokens_per_second` | Histogram | — | ortalama üretim hızı (akışsız: out/answer_süresi) |
| `bg_llm_cost_usd_total` | Counter | — | kümülatif tahmini maliyet |
| `bg_index_pages` | Gauge | — | yüklü korpus boyutu (koşum künyesi) |
| `bg_app_info` | Info | `retriever_model`, `gemini_model`, `device`, `version` | hangi konfig koşuyor |
| `process_*`, `python_gc_*` | (varsayılan kolektörler) | — | RSS, CPU, GC — bedava sistem görünümü |

Bucket'lar (ölçülmüş v0 davranışına göre):
- request: `0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30`
- stage: `0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20`
- top_score: `45, 50, 55, 58, 60, 62, 65, 70, 75` · margin: `0, .5, 1, 2, 4, 8`
- tokens/s: `5, 10, 20, 40, 80, 160`

**Türetilmiş metrikler** (SQL/pandas — kataloğun analiz bölümü, kod değil):
abstain oranı zaman serisi; skor↔soru-uzunluğu ilişkisi; encode süresi↔soru uzunluğu;
eşzamanlılık↔throughput eğrisi (loadgen taraması); soru başına maliyet; honest-miss oranı.

**İstemci tarafı** (loadgen çıktısı, sunucudan bağımsız doğrulama):
achieved RPS, istemci p50/p95/p99, hata/timeout sayısı.

## 7. Enstrümantasyon noktaları (dokunulan mevcut dosyalar)

| Dosya | Değişiklik |
|---|---|
| `retrieval/core.py` | `search()` içine `with stage("query_encode")`, `search_embedding()` içine stage1/stage2 sarmaları (telemetry.collect no-op güvenli) |
| `answer/base.py` | `AskService.ask`: answerer çağrısı `with stage("answerer")`; abstain reason annotate |
| `answer/gemini.py` | `GeminiClient.generate` → `GenResult(text, tokens_in, tokens_out)` döner; `GeminiAnswerer` token'ları `annotate()` eder. Stub'lar `usage`sız çalışmaya devam eder (None → kolonlar NULL) |
| `app/main.py` | route'larda RequestTimer + olay birleştirme + `recorder.record` + `prom.observe`; `GET /metrics`; `/stats` events'ten (şekil geriye uyumlu + `p95_ms`, `abstain_rate`, `by_endpoint`); eski `_log_*` kaldırılır |
| `config.py` | `log_query_text: bool = True`, `gemini_price_in_usd_per_1m: float`, `gemini_price_out_usd_per_1m: float` (varsayılanlar inşa günü güncel fiyatla doğrulanır; runbook'ta 'tahmin, env ile geçersiz kıl' notu) |
| `pyproject.toml` | `prometheus-client>=0.20` |

## 8. Gözlemlenebilirlik stack'i (lokal, OrbStack)

```
observability/
  docker-compose.yml            # prometheus:9090 + grafana:3000, adlandırılmış volume'ler
  prometheus.yml                # scrape: host.docker.internal:7860, interval 5s
  grafana/provisioning/
    datasources/prometheus.yml  # otomatik datasource
    dashboards/provider.yml
    dashboards/belge-gozu.json  # commit'li dashboard (makale figürlerinin kaynağı)
```

Dashboard panelleri (tek ekran): RPS (endpoint bazlı) · p50/p95/p99 · aşama süresi
yığılı görünüm · abstain oranı % · top_score dağılımı (heatmap) · token/sn ·
kümülatif token & maliyet · inflight · process RSS/CPU.
Makefile: `obs-up`, `obs-down`. Uygulama compose'a **girmez** (host'ta koşar; Space
ile aynı topoloji).

## 9. Yük üreticisi: `scripts/loadgen.py`

- httpx (mevcut bağımlılık), asyncio; argümanlar: `--endpoint search|ask|mixed`
  (varsayılan **search**), `--concurrency`, `--duration` | `--requests`, `--out results.json`.
- `--endpoint ask|mixed` seçilirse kota uyarısı basar ve `--yes-burn-quota` bayrağı ister.
- Sorular `scripts/queries_sample.txt`'ten (≈30 çeşitli Türkçe mevzuat sorusu) örneklenir.
- Çıktı: istemci tarafı özet JSON (§6) + stdout tablosu. İstatistik fonksiyonu saf ve
  birim-testlidir; sunuculu koşum `slow` işaretlidir.

## 10. CLI ve dışa aktarma

- `belge-gozu metrics export --out data/exports/events.parquet [--csv]`
- `belge-gozu metrics summary` — terminalde hızlı özet (istek sayısı, p50/p95,
  abstain %, toplam token/maliyet) → bulgu notu yazarken kopyala-yapıştır.

## 11. `docs/research/` yapısı

```
docs/research/
  metrics-catalog.md   # §5–6'nın insan-okur sözlüğü: tanım, birim, etiket, 'neden', faz
  runbook.md           # stack'i kaldır, loadgen koş, bulgu notu yaz, figür export et;
                       # Gemini kota bütçesi uyarıları; fiyat varsayımlarını doğrulama adımı
  findings/
    2026-08-26-baseline.md   # İLK GERÇEK ölçüm oturumu (inşanın son task'ında doldurulur)
  figures/             # Grafana/pandas'tan export edilen PNG'ler (makaleye girecekler)
```

Kural: her ölçüm oturumu tarihli bir bulgu notu bırakır (koşum künyesi: config,
commit sha, korpus boyutu, yük parametreleri + gözlemler + figür referansları).

## 12. Test stratejisi (CI: ağ/GPU/model yok)

- `tests/telemetry/test_recorder.py` — olay yazımı, WAL, eşzamanlı yazım (thread'ler),
  asla-fırlatmaz (bozuk DB mock; mevcut `test_log_write_never_raises`'ın halefi)
- `tests/telemetry/test_collect.py` — stage iç içe/no-op/annotate; kolektörsüz sıfır etki
- `tests/telemetry/test_prom.py` — kayıtlı seriler; stub /ask & /search sonrası sayaç artışı;
  `/metrics` gövdesinde `bg_` isimleri
- `tests/telemetry/test_export.py` — events → parquet gidiş-dönüş
- `tests/app/test_api.py` — `/stats` genişletilmiş şekil (geriye uyumlu alanlar korunur)
- `tests/answer/` — stub GenResult ile token yakalama; usage'sız stub → NULL kolonlar
- `tests/test_loadgen.py` — istatistik birleştirici saf fonksiyon testi
- Compose/Grafana CI'da test edilmez; runbook'taki canlı doğrulama adımı son task'tadır

## 13. Faz 2 backlog (bilinçli erteleme)

streaming + TTFT (+ UI akışı) · Redis: sorgu-embedding önbelleği, rate-limit ·
OpenTelemetry trace (aşama span'ları hazır — stage adları korunur) · Grafana alerting ·
MPS/GPU örnekleme · oturum/kullanıcı kimliği (privacy tasarımıyla birlikte)

## 14. Kabul ölçütleri

1. Her `/ask` ve `/search` isteği `events`'e aşama kırılımıyla düşer; telemetri hatası
   isteği asla bozmaz (test kanıtlı).
2. `/metrics` `bg_*` serilerini döner; compose stack'i OrbStack'te kalkar; loadgen
   koşusu sırasında Grafana dashboard'u canlı veri gösterir (ekran görüntüsü bulgu notunda).
3. Token in/out, token/sn ve maliyet tahmini gerçek bir `/ask`'ta dolu gelir.
4. `metrics export` Parquet üretir, pandas ile okunur.
5. `-m "not slow"` süiti + ruff + pyright temiz; hiçbir test/CI adımı Gemini kotası yakmaz;
   loadgen varsayılanı `/search`.
6. `docs/research/` kurulmuş; `2026-08-26-baseline.md` GERÇEK sayılarla dolu ve commit'li.
