# Metrik Kataloğu

Kaynak: `docs/superpowers/specs/2026-08-26-telemetry-design.md` §5 (olay şeması) ve
§6 (Prometheus kataloğu). Bu belge o iki bölümün insan-okur sözlüğüdür — spec
normatif kaynak, burası referans/çeviri katmanıdır.

> **Not (nearest-rank p95):** Bu belgede ve kod tabanında geçen her p95/p50/p99
> ifadesi **en-yakın-sıra (nearest-rank, ceil)** formülünü kullanır:
> `sıralı[min(n-1, ceil(p·n) - 1)]`. Bu formül üç yerde birebir aynıdır:
> `/stats` (`src/belge_gozu/app/main.py`), `belge-gozu metrics summary`
> (`src/belge_gozu/cli.py`) ve `scripts/loadgen.py`. Grafana dashboard'undaki
> `histogram_quantile(...)` panelleri ayrı bir yöntemdir (Prometheus'un bucket
> interpolasyonu) — sayılar birebir örtüşmez, ikisi de p95'in geçerli
> tahminleridir, sadece farklı algoritmalarla.

## 1. Olay şeması — `events` tablosu (spec §5)

Her `/ask` ve `/search` isteği bu tabloya bir satır düşürür (WAL modlu SQLite,
`data/requests.sqlite`). Sabit kolonlar + `detail` JSON (koşum künyesi: top-5
[{page_id,score}], model adları, device, threshold, app_version).

| Kolon | Tip | Birim | Kaynak kod noktası | Neden önemli | Faz |
|---|---|---|---|---|---|
| `id` | INTEGER PK | — | `telemetry/schema.py` (`EVENTS_DDL`) | satır kimliği | 1 |
| `ts` | TEXT | ISO-8601 UTC | `app/main.py` (route handler) | zaman serisi analizleri, bulgu notu künyesi | 1 |
| `endpoint` | TEXT | — (`/ask`\|`/search`) | `app/main.py` | endpoint bazlı kırılım (RPS, gecikme, maliyet) | 1 |
| `status` | TEXT | — (`/ask`: answered\|abstained\|degraded\|error · `/search`: ok\|error) | `app/main.py` | sonuç dağılımı; hata oranı = error/toplam | 1 |
| `http_status` | INTEGER | HTTP kodu | `app/main.py` | client hatası mı server hatası mı ayrımı | 1 |
| `total_ms` | REAL | ms | `app/main.py` (RequestTimer) | uçtan uca gecikme — kullanıcının hissettiği | 1 |
| `encode_ms` | REAL, NULL olabilir | ms | `retrieval/core.py` `with stage("query_encode")` | darboğaz ayrıştırması: embedding aşaması | 1 |
| `stage1_ms` | REAL | ms | `retrieval/core.py` `with stage("stage1_hamming")` | darboğaz ayrıştırması: Hamming ön-eleme | 1 |
| `stage2_ms` | REAL | ms | `retrieval/core.py` `with stage("stage2_maxsim")` | darboğaz ayrıştırması: MaxSim yeniden sıralama | 1 |
| `answer_ms` | REAL, yalnız `/ask` + LLM çağrıldıysa | ms | `answer/base.py` `with stage("answerer")` | LLM çağrısının toplam süredeki payı | 1 |
| `top_score` | REAL | skor birimi (spec eşiği ile aynı ölçek) | `retrieval` → `app/main.py` birleştirme | eşik kalibrasyonu, skor driftini izleme | 1 |
| `margin_1_2` | REAL | skor birimi | `retrieval` → `app/main.py` birleştirme | top1−top2: retrieval kararlılığı/belirsizliği | 1 |
| `abstained` | INTEGER 0/1, yalnız `/ask` | bool | `answer/base.py` (`Answer.abstained`) | halüsinasyon freninin ne sıklıkla tetiklendiği | 1 |
| `honest_miss` | INTEGER 0/1, **sezgisel** | bool | `app/main.py:118` — `"bulamadım" in answer.text.lower()` | **HEURISTIC**: abstain DEĞİL ama LLM metninde "bulamadım" geçiyor; v0'ın ana bulgusuydu (13/17) | 1 |
| `k` | INTEGER | adet | `app/main.py` (istek gövdesi) | retrieval genişliği | 1 |
| `candidates` | INTEGER | adet | `retrieval/core.py` | aday havuzu boyutu | 1 |
| `query_len` | INTEGER | karakter | `app/main.py` | soru uzunluğu↔gecikme/skor ilişkisi | 1 |
| `query_text` | TEXT, NULL olabilir | — | `app/main.py`, `config.py: log_query_text` | ham metin (varsayılan açık); `BG_LOG_QUERY_TEXT=false` ile kapatılır | 1 |
| `query_sha256` | TEXT | hex hash | `app/main.py` | her durumda yazılır — dedup/korelasyon (query_text kapalıyken de) | 1 |
| `answer_len` | INTEGER | karakter | `app/main.py` | yanıt uzunluğu | 1 |
| `citations_n` | INTEGER | adet | `app/main.py` | atıf sayısı — yanıt zenginliği | 1 |
| `tokens_in` | INTEGER | token | `answer/gemini.py:76` (`annotate("tokens_in", ...)`) | maliyet tabanı, `usage_metadata`sız stub'larda NULL | 1 |
| `tokens_out` | INTEGER | token | `answer/gemini.py:78` (`annotate("tokens_out", ...)`) | üretim hacmi, maliyet tabanı | 1 |
| `tokens_per_s` | REAL | token/sn | `app/main.py` (`tokens_out / (answer_ms/1000)`) | akışsız ortalama üretim hızı | 1 |
| `est_cost_usd` | REAL | USD | `app/main.py:113-115` (`config.py` fiyat sabitleri) | çağrı başına tahmini maliyet | 1 |
| `error_type` | TEXT, yalnız `status='error'` | exception sınıf adı | `app/main.py` | hata sınıflandırması | 1 |
| `detail` | TEXT (JSON) | — | `app/main.py` | top-5 `[{page_id,score}]`, model adları, device, threshold, app_version — makale tekrarlanabilirliği için koşum künyesi | 1 |

İndeks: `(ts)`, `(endpoint, ts)` — `telemetry/schema.py: EVENTS_INDEXES`.

## 2. Prometheus metrik kataloğu (spec §6)

Adlandırma: `bg_` öneki, taban birim saniye. Registry ve tanımlar
`src/belge_gozu/telemetry/prom.py` (`PromMetrics.__init__`); her olay
`PromMetrics.observe(event)` ile buraya yansır; gövde `GET /metrics`
(`generate_latest`, Prometheus text format).

| Metrik | Tip | Etiketler | Birim | Kaynak kod noktası | Neden önemli | Faz |
|---|---|---|---|---|---|---|
| `bg_http_requests_total` | Counter | `endpoint`, `status` | istek | `prom.py: self.requests` | trafik + sonuç dağılımı; hata oranı = `status="error"`/toplam | 1 |
| `bg_request_duration_seconds` | Histogram | `endpoint` | saniye | `prom.py: self.duration`, bucket'lar `REQUEST_BUCKETS` | uçtan uca gecikme; p50/p95/p99 buradan (`histogram_quantile`) | 1 |
| `bg_inflight_requests` | Gauge | `endpoint` | adet | `prom.py: self.inflight_g` (context manager `inflight()`) | anlık eşzamanlılık / doyma sinyali | 1 |
| `bg_stage_duration_seconds` | Histogram | `stage` ∈ {query_encode, stage1_hamming, stage2_maxsim, answerer} | saniye | `prom.py: self.stage`, bucket'lar `STAGE_BUCKETS` | darboğaz ayrıştırması: "gecikme nerede birikiyor?" | 1 |
| `bg_retrieval_top_score` | Histogram | — | skor | `prom.py: self.top_score`, bucket'lar `SCORE_BUCKETS` | skor dağılımı; eşik kalibrasyonu + zaman içi drift | 1 |
| `bg_retrieval_score_margin` | Histogram | — | skor farkı | `prom.py: self.margin`, bucket'lar `MARGIN_BUCKETS` | top1−top2; retrieval kararlılığı | 1 |
| `bg_abstain_total` | Counter | `reason` ∈ {threshold, degraded} | adet | `prom.py: self.abstain` (`observe()` içindeki dallanma) | halüsinasyon freninin sağlığı; `degraded` = kota/servis hatası görünürlüğü | 1 |
| `bg_honest_miss_total` | Counter | — | adet | `prom.py: self.honest_miss` | **HEURISTIC üstüne kurulu sayaç** — bkz. §1 `honest_miss` satırı | 1 |
| `bg_llm_tokens_total` | Counter | `direction` ∈ {input, output} | token | `prom.py: self.tokens` | hacim; maliyet tabanı | 1 |
| `bg_llm_tokens_per_second` | Histogram | — | token/sn | `prom.py: self.tps`, bucket'lar `TPS_BUCKETS` | ortalama üretim hızı (akışsız: `tokens_out / answer_süresi`) | 1 |
| `bg_llm_cost_usd_total` | Counter | — | USD | `prom.py: self.cost` | kümülatif tahmini maliyet | 1 |
| `bg_index_pages` | Gauge | — | sayfa | `prom.py: self.pages` (`set_app_info`) | yüklü korpus boyutu (koşum künyesi) | 1 |
| `bg_app_info` | Info | `retriever_model`, `gemini_model`, `device`, `version`, `threshold` | — | `prom.py: self.info` (`set_app_info`) | hangi konfigürasyon koşuyor — tekrarlanabilirlik | 1 |
| `process_*`, `python_gc_*` | (varsayılan kolektörler) | — | RSS/CPU/GC | `prometheus_client` varsayılan koleksiyoncular (otomatik) | bedava sistem görünümü; `process_resident_memory_bytes` dashboard'da RSS paneli | 1 |

> Not: Prometheus text formatında Counter'lar `_total` soneki ile sunulur
> (ör. kod içinde `bg_http_requests` tanımlıysa `/metrics` çıktısında
> `bg_http_requests_total` görünür) — yukarıdaki tablo `/metrics`'te
> gözlemlenen adları kullanır (spec §6 ile birebir).

### Bucket'lar (ölçülmüş v0 davranışına göre)

| Histogram | Bucket sınırları |
|---|---|
| `bg_request_duration_seconds` | `0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30` (saniye) |
| `bg_stage_duration_seconds` | `0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20` (saniye) |
| `bg_retrieval_top_score` | `45, 50, 55, 58, 60, 62, 65, 70, 75` |
| `bg_retrieval_score_margin` | `0, 0.5, 1, 2, 4, 8` |
| `bg_llm_tokens_per_second` | `5, 10, 20, 40, 80, 160` (token/sn) |

Kaynak: `src/belge_gozu/telemetry/prom.py` — `REQUEST_BUCKETS`, `STAGE_BUCKETS`,
`SCORE_BUCKETS`, `MARGIN_BUCKETS`, `TPS_BUCKETS`.

### Türetilmiş metrikler (SQL/pandas — analiz katmanı, kod değil)

Bunlar Prometheus'ta değil, `events` tablosundan `belge-gozu metrics
export`/`summary` veya pandas ile hesaplanır:

- Abstain oranı zaman serisi (`abstained` + `status='degraded'` / toplam)
- Skor ↔ soru-uzunluğu ilişkisi (`top_score` vs `query_len`)
- Encode süresi ↔ soru uzunluğu ilişkisi (`encode_ms` vs `query_len`)
- Eşzamanlılık ↔ throughput eğrisi (loadgen taraması, `--concurrency` süpürmesi)
- Soru başına maliyet (`est_cost_usd` dağılımı)
- Honest-miss oranı (**sezgisel** — bkz. yukarıdaki not)

### İstemci tarafı metrikler (sunucudan bağımsız doğrulama)

`scripts/loadgen.py` çıktısı: achieved RPS, istemci p50/p95/p99 (nearest-rank,
yukarıdaki formülle — `scripts/loadgen.py: pct()`), hata/timeout sayısı
(`errors`). Bkz. `docs/research/runbook.md` — loadgen'in `requests` alanı
yalnızca **başarılı** (HTTP 200) çağrıları sayar; toplam denenen istek sayısı
`requests + errors`'tır.

## 3. Kapsam dışı (Faz 2 backlog — spec §13)

streaming TTFT · Redis (embedding önbelleği / rate-limit) · OpenTelemetry trace
(aşama span adları zaten hazır) · Grafana alerting · GPU/MPS örnekleme ·
oturum/kullanıcı kimliği.
