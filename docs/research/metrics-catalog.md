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
| `status` | TEXT | — (`/ask`: answered\|abstained\|degraded\|error · `/search`: ok\|error · her ikisi: **rejected**) | `app/main.py` | sonuç dağılımı; hata oranı = error/toplam; `rejected` = 422/429 (Y23) | 1 |
| `http_status` | INTEGER | HTTP kodu | `app/main.py` | client hatası mı server hatası mı ayrımı | 1 |
| `total_ms` | REAL | ms | `app/main.py` (RequestTimer) | uçtan uca gecikme — kullanıcının hissettiği | 1 |
| `encode_ms` | REAL, NULL olabilir | ms | `retrieval/core.py` `with stage("query_encode")` | darboğaz ayrıştırması: embedding aşaması | 1 |
| `stage1_ms` | REAL | ms | `retrieval/core.py` `with stage("stage1_hamming")` | darboğaz ayrıştırması: Hamming ön-eleme | 1 |
| `stage2_ms` | REAL | ms | `retrieval/core.py` `with stage("stage2_maxsim")` | darboğaz ayrıştırması: MaxSim yeniden sıralama | 1 |
| `answer_ms` | REAL, yalnız `/ask` + LLM çağrıldıysa | ms | `answer/base.py` `with stage("answerer")` | LLM çağrısının toplam süredeki payı | 1 |
| `top_score` | REAL | skor birimi (spec eşiği ile aynı ölçek) | `retrieval` → `app/main.py` birleştirme | eşik kalibrasyonu, skor driftini izleme | 1 |
| `margin_1_2` | REAL | skor birimi | `retrieval` → `app/main.py` birleştirme | top1−top2: retrieval kararlılığı/belirsizliği | 1 |
| `abstained` | INTEGER 0/1, yalnız `/ask` | bool | `answer/base.py` (`Answer.abstained`) | halüsinasyon freninin ne sıklıkla tetiklendiği | 1 |
| `honest_miss` | INTEGER 0/1, **yalnız `status='answered'` satırlarında**; diğerlerinde NULL | bool | `answer/base.py: is_honest_miss()` — `HONEST_MISS_MARKER in tr_lower(text)` | modelin KENDİ dürüst ıskası (getirim getirdi, model kanıt bulamadı). **Üç değer, üç anlam:** NULL = hesaplanmadı (LLM hiç konuşmadı — abstained/degraded/error ya da `/search`) · 0 = hesaplandı, ıska yok · 1 = hesaplandı, ıska var. P2 bu kolonu hedef değişken olarak okuyacağı için abstain/degraded satırlarına 0 YAZILMAZ. TEK hesap yolu: `/ask` gövdesindeki `honest_miss`, bu kolon ve `bg_honest_miss_total` aynı fonksiyondan. Mühür (`HONEST_MISS_MARKER`) Gemini SİSTEM istemine f-string ile GÖMÜLÜ, yani modele dayatılan ifade ile aranan ifade ayrışamaz (Y17/K27) | 1 |
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
| `error_type` | TEXT (`error`: exception sınıf adı · `degraded`/`rejected`: taksonomi) | timeout\|http_5xx\|http_429\|auth\|safety_block\|parse\|other\|validation\|rate_limited | `answer/base.py: ERROR_TYPES` + `app/main.py: _REJECT_REASONS` | "cevaplayamadı" ile "cevaplamamalı"yı ayırır; degraded satırlarda eskiden 114/114 NULL'du (Y20) | 1 |
| `score_scale` | TEXT, NULL olabilir | ölçek adı (`hybrid-bm25`\|`visual-normalized`) | `config.PIPELINE_SCORE_SCALE[pipeline]` -> `app/main.py` | `top_score`/`margin_1_2` HANGİ ölçekte — üretimde bu sütunda üç uyumsuz ölçek karışıktı (Y18). Migrasyon ÖNCESİ satırlar NULL KALIR: hangi ölçekte oldukları bilinmiyor, uydurulmaz | 1 |
| `detail` | TEXT (JSON) | — | `app/main.py` | koşum künyesi: `hits` (top-k `[{page_id,score}]`), `threshold`, `retriever_model`, `gemini_model`, `device`, `app_version`, `stages` (aşama adı -> ms; `_STAGE_COLS` dışındaki adlar YALNIZ burada), `retrieval` (`query_format`, `quantization`; hibrit yolda ayrıca `bm25_top1`, `visual_top1`, `routed_docs`). **P2 (yalnız bayrak açıkken):** `gate1` (`p`, `tau`, `passed`, `key`, `guarantee`, `features` — `BG_GATE_CALIBRATED=true`) ve `gate2` (`demoted`, `n_claims`, `n_verified`, `n_supported`, `truncated`, `llm_calls`, **`api_attempts`**, `cache_hits`, `budget_max_attempts`, `budget_used`, `budget_exhausted`, `claims[]` = iddia başına `{claim_id, verdict, gerekce, cited_sources, inherited_sources, cached, attempts}` — `BG_GATE_VERIFIER=true`; atlandıysa yalnız `{demoted, skipped}`). **`llm_calls` ≠ `api_attempts`:** ilki "kaç doğrulayıcı çağrısı başlatıldı", ikincisi "kaç HTTP denemesi gitti" (anahtar rotasyonu + retry dahil) — ücretsiz kota DENEME ile tükendiği için bütçe (`Settings.verifier_max_llm_calls`) ikincisini sayar. Bayraklar kapalıyken bu iki anahtar HİÇ yazılmaz, yani bayrak-kapalı olay şeması P1 ile birebir aynıdır | 1 |

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
| `bg_stage_duration_seconds` | Histogram | `stage` ∈ {query_encode, exhaustive_maxsim, text_bm25, route_fuse, stage1_hamming, stage2_maxsim, answerer} | saniye | `prom.py: self.stage`, bucket'lar `STAGE_BUCKETS` | darboğaz ayrıştırması: "gecikme nerede birikiyor?" | 1 |
| `bg_retrieval_top_score` | Histogram | `quantization` | skor (normalize ~[-1,1]) | `prom.py: self.top_score`, bucket'lar `SCORE_BUCKETS` | skor dağılımı; eşik kalibrasyonu + zaman içi drift — **yalnız görsel-ölçek pipeline'ları** (`exhaustive`/`two-stage`) | 1 |
| `bg_retrieval_score_margin` | Histogram | `quantization` | skor farkı (aynı ölçek) | `prom.py: self.margin`, bucket'lar `MARGIN_BUCKETS` | top1−top2; retrieval kararlılığı — yalnız görsel-ölçek pipeline'ları | 1 |
| `bg_retrieval_top_score_bm25` | Histogram | — | skor (BM25 birimi, üst sınırsız; ölçülen bant ~4-70) | `prom.py: self.top_score_bm25`, bucket'lar `BM25_SCORE_BUCKETS` | hibrit (P1) pipeline'ın skor dağılımı; eşik 10.6 bucket sınırı olarak var, çalışma noktası doğrudan okunur | 1 |
| `bg_retrieval_score_margin_bm25` | Histogram | — | skor farkı (BM25 birimi) | `prom.py: self.margin_bm25`, bucket'lar `BM25_MARGIN_BUCKETS` | hibrit pipeline'da top1−top2 | 1 |
| `bg_abstain_total` | Counter | `reason` ∈ {threshold, gate1, gate2_demote, degraded} | adet | `prom.py: ABSTAIN_REASONS` + `abstain_reason(ev)` — TEK karar yolu | halüsinasyon freninin sağlığı, **hangi fren olduğu ayrımıyla**: `threshold` = eski skor eşiği · `gate1` = kalibre güven kapısı (`p < tau`) · `gate2_demote` = kanıt kapısı yanıtı düşürdü · `degraded` = kota/servis hatası. Üçü tek etikette toplanınca (P2 öncesi) operatör artışı eşiğe mi kalibratöre mi doğrulayıcıya mı atfedeceğini metrikten çıkaramıyordu; kapı 1'in ölçülen dev kapsaması %2.2 olduğu için bu seri bayrak açıldığında patlar. Bayraklar KAPALIYKEN yalnız `threshold`/`degraded` üretilir (olayda kapı bloğu yok) | 1 |
| `bg_rate_limited_total` | Counter | `endpoint` | adet | `prom.py: self.rate_limited`, `app/main.py::enforce_rate_limit` doğrudan artırır | 429 hız sınırı reddi — "hangi uç nokta sınırlanıyor?" | 1 |
| `bg_rejected_total` | Counter | `reason` ∈ {validation, rate_limited} | adet | `prom.py: self.rejected`, `app/main.py::record_rejection` | reddedilen isteklerin BİRLEŞİK görünümü — "istekler NEDEN reddediliyor?". 422'ler P2 için "cevaplanmaması gereken"in en temiz sınıfıdır (Y23). Bir 429 hem burada hem `bg_rate_limited_total`ta görünür (farklı eksenler), TEK seri içinde iki kez sayılmaz | 1 |
| `bg_honest_miss_total` | Counter | — | adet | `prom.py: self.honest_miss` (olayın `honest_miss` alanı) | modelin dürüst ıskası; §1'deki TEK hesap yolundan besleniyor — ayrı bir sezgi tutmaz | 1 |
| `bg_verifier_verdicts_total` | Counter | `verdict` ∈ {supported, unsupported, belirsiz} | iddia | `prom.py: self.verifier_verdicts` (olayın `detail.gate2.claims[]`) | P2 kanıt kapısının İDDİA BAŞINA karar dağılımı. Etiket kümesi `answer/verify.VERDICTS`ten gelir (kopya yok, kardinalite kapalı). `BG_GATE_VERIFIER=false` iken seri BOŞTUR — bayrak kapalıyken olaylarda `detail.gate2` hiç oluşmaz. Yorum: `unsupported+belirsiz` payı, yanıtların kanıtlanabilirliğinin doğrudan ölçüsüdür; `belirsiz` DESTEKLENMEMİŞ sayılır (şüphede-reddet) | 1 |
| `bg_llm_tokens_total` | Counter | `direction` ∈ {input, output} | token | `prom.py: self.tokens` | hacim; maliyet tabanı | 1 |
| `bg_llm_tokens_per_second` | Histogram | — | token/sn | `prom.py: self.tps`, bucket'lar `TPS_BUCKETS` | ortalama üretim hızı (akışsız: `tokens_out / answer_süresi`) | 1 |
| `bg_llm_cost_usd_total` | Counter | — | USD | `prom.py: self.cost` | kümülatif tahmini maliyet | 1 |
| `bg_llm_key_rotations_total` | Counter | `from_key` ∈ {key1, key2} | adet | `prom.py: self.key_rotations` (olayın `detail.llm.rotations[] = {from, error_type}`) | API anahtarı rotasyonu: HANGİ anahtarda hata alınıp öbürüne geçildi (`answer/gemini.py::RotatingGeminiClient`). Etiket kümesi `answer/gemini.KEY_LABELS`ten gelir; anahtar DEĞERİ ne metriğe ne loga ne de olaya yazılır. Hata SINIFI etiket değil, olay alanıdır (`rotations[].error_type`) — kardinalite taksonomi kadar büyümesin. Tek anahtarlı dağıtımda (`GOOGLE_API_KEY_2` boş) seri BOŞTUR. Yorum: rotasyon başarılıysa istek `answered` biter ve `events.error_type` NULL kalır — birincil anahtarın kotasının bittiğini söyleyen TEK erken uyarı bu seridir (`bg_abstain_total{reason="degraded"}` daha artmamışken) | 2 |
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
| `bg_request_duration_seconds` | `0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 45, 60` (saniye) |
| `bg_stage_duration_seconds` | `0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 35, 60` (saniye) |
| `bg_retrieval_top_score` | `0.30, 0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.65, 0.70, 0.80` |
| `bg_retrieval_score_margin` | `0.0, 0.005, 0.01, 0.02, 0.04, 0.08` |
| `bg_retrieval_top_score_bm25` | `0, 5, 10, 10.6, 15, 20, 30, 45, 70, 100` |
| `bg_retrieval_score_margin_bm25` | `0, 0.5, 1, 2, 5, 10, 20, 40` |
| `bg_llm_tokens_per_second` | `5, 10, 20, 40, 80, 160` (token/sn) |

Gecikme bucket'larının üst uçları LLM toplam bütçesine bağlıdır (`GEMINI_TOTAL_BUDGET_S`, 2026-08-31: 35 -> 50 sn); tavan bucket bütçenin üstünde tutulur, aksi halde bütçeye yakın istekler `+Inf`e düşüp p95/p99'u körleştirir.

Kaynak: `src/belge_gozu/telemetry/prom.py` — `REQUEST_BUCKETS`, `STAGE_BUCKETS`,
`SCORE_BUCKETS`, `MARGIN_BUCKETS`, `TPS_BUCKETS`.

Skor/marj bucket'ları T14'te normalize [-1,1] ölçeğine taşındı; geçiş öncesi seriler ve `events` satırları eski binary ölçeğindedir (0-128) ve `bg_app_info`'nun `index_revision` etiketi (olay tablosunda `index_revision` kolonu) ile bu iki histogramın `quantization` etiketinden ayırt edilir.

**BM25 ölçeği (P1).** Hibrit pipeline'da sıralamayı ve dolayısıyla `top_score`'u BM25 metin kanalı üretir: kalibre edilmemiş, ÜST SINIRSIZ birim. Canary'de **servis edilen** (yani eşiğe giren) top-1'ler min 10.53 / medyan 24.02 / maks 69.30; kanalın kendi top-1 medyanı 26.05'tir (`detail.retrieval.bm25_top1`) — doküman-adı yönlendirmesi sıralamanın birincisini skora göre seçmediği için ikisi ayrışır ve eşik/çalışma noktası **servis edilen** skordan ölçülmüştür. Bu örnekler normalize [-1,1] serilerinde toplansaydı hepsi son bucket'a düşer ve quantile'lar anlamsızlaşırdı, bu yüzden `PromMetrics.observe` olayın `pipeline` künyesine bakıp AYRI `*_bm25` serilerine yönlendirir. Yönlendirme kümesi (`prom.py: BM25_SCALE_PIPELINES`) `config.PIPELINE_SCORE_SCALE`'den **türetilir**, kopya sabit tutulmaz. `quantization` etiketi bu serilerde YOKTUR: BM25 skoru metin katmanından gelir, indeks temsiline bağlı değildir. Aşama serisine (`bg_stage_duration_seconds`) hibritin `text_bm25`/`route_fuse` adları `detail.stages` fallback'iyle kendiliğinden akar — prom.py'de aşama adı listesi tutulmaz.

**`rejected` satırları ve gecikme toplamları.** `/stats` (`avg_ms`, `p95_ms`) ve
`belge-gozu metrics summary` `WHERE status <> 'rejected'` ile hesaplanır: ret satırlarının
`total_ms`'i sub-ms'dir (ölçüm: 0.0136 / 0.0213 ms) ve bir 429 dalgasında son-10000
penceresini domine edip p95'i sistem kötüye kullanılırken mükemmel gösterirdi — aynı
gerekçeyle `record_rejection` da `prom.observe`'u çağırmaz. `requests` ve `by_endpoint`
KASITLI olarak filtrelenmez: reddedilen istek de gelmiş bir istektir.

**Reddedilen istekler (Y23).** `require_searchable` (422) ve hız sınırı (429)
artık `status='rejected'` ile MİNİMAL bir olay satırı yazar: `endpoint`,
`error_type` (`validation`/`rate_limited`), `total_ms`, sorgu kimliği. Skor/aşama
alanları NULL kalır — getirici hiç çağrılmadı, doldurulsalardı P2'nin okuyacağı
tabloya sahte sıfırlar girerdi. **Kapsam sınırı, dürüstçe:** pydantic düzeyinde
reddedilen istekler (gövde `max_length`, `k` aralığı) uç nokta gövdesine hiç
ULAŞMADAN 422 döner ve olay YAZILMAZ; `bg_rejected_total` da onları saymaz.

**BM25 skor bandı ve sorgu-terim doygunluğu (Y1).** `bg_retrieval_top_score_bm25`
için önceden yazılı "ölçülen bant ~4-70" ifadesi ÜRETİMDE ÇÜRÜTÜLMÜŞTÜ: aynı
terimi tekrar eden bir sorgu skoru doğrusal şişiriyordu (canlı ölçüm: 667.5;
tarihsel maksimum 1053.47) ve tek bir istek histogramın `_sum`'ını ele
geçirebiliyordu. `BM25Index.scores` artık sorgu terimlerini `min(qtf, 2)` ile
ağırlıklandırıyor (`retrieval/text.py: QTF_CAP`), yani sorgu tarafı katkısı
sabit bir çarpanla sınırlı. Ölçek hâlâ ÜST SINIRSIZ (doküman tarafı), ama band
artık sömürülebilir değil.

**Grafana panosu.** `observability/grafana/.../belge-gozu.json` iki skor paneli taşır: "Top skor dağılımı — BM25 (hibrit, varsayılan)" (`bg_retrieval_top_score_bm25_bucket`) ve "Top skor dağılımı — görsel ölçek" (`bg_retrieval_top_score_bucket`, `quantization` kırılımlı). Etkin pipeline'a göre biri dolu, diğeri BOŞ olur; bu beklenen davranıştır, telemetri arızası değildir.

### Türetilmiş metrikler (SQL/pandas — analiz katmanı, kod değil)

Bunlar Prometheus'ta değil, `events` tablosundan `belge-gozu metrics
export`/`summary` veya pandas ile hesaplanır:

- **Eşik abstain oranı** zaman serisi (`AVG(abstained)` WHERE `endpoint='/ask' AND status <>
  'degraded'` — `belge-gozu metrics summary`, `/stats`'ın `abstain_rate`'i): `degraded` (kota/servis
  hatası) satırları hem paydan hem paydadan hariç tutulur, yalnızca eşik altı kalıp LLM'e hiç
  gitmeyen istekleri yansıtır. Bunu Prometheus'taki `bg_abstain_total` (§2) ile karıştırmayın —
  o sayaç `reason ∈ {threshold, degraded}` ile **toplam fren aktivasyonunu** sayar (degraded
  dahil); ikisi farklı sorulara cevap verir: SQL "eşik yüzünden ne kadar sıklıkla LLM'e hiç
  gidilmedi", Prometheus "hangi nedenle olursa olsun fren kaç kez tetiklendi".
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
