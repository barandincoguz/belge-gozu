# Vitrin sprinti — düzeltme turu re-review (commit `71e5860`)

**Tarih:** 2026-08-30 · **Kapsam:** `71e5860` (`6d5b345`'in düzeltme turu) · **Kip:** salt-okunur
(pytest / lint / grep / curl / git ls-files)

**Okunanlar:**
- Orijinal bulgular: `vitrin-sprint-review.md` (commit `6d5b345`, APPROVE, 2 MEDIUM / 5 LOW / 3 NIT / 1 INFO)
- Düzeltme özeti: `.superpowers/sdd/2026-08-26-belge-gozu-p0-retrieval-correctness/vitrin-sprint-report.md` §E
- Düzeltme diff'i: `review-fix-71e5860.diff` (8 dosya, +244/−16, tam okundu)

**VERDİKT: TÜMÜ ONAYLANDI.** 10/10 aktif bulgu (M1-M2, L2-L5, N1-N3) gerçek koda karşı doğrulandı
ve iddiayla birebir örtüşüyor; L1 kontrolcü kararıyla bilerek düzeltilmedi (docstring'e
gerekçe eklendi); I1 zaten "aksiyon gerekmez" statüsündeydi ve dokunulmamış kalması doğru.
Testler (338/338), lint (temiz) ve canlı `/healthz` bağımsız olarak yeniden üretildi. Bir
adet YENİ, düşük şiddetli, bugün ULAŞILAMAZ savunma boşluğu bulundu (aşağıda NEW-1).

---

## 1. Bulgu bazlı doğrulama

| # | Bulgu | Şiddet | Durum | Kanıt |
|---|---|---|---|---|
| M1 | `RateLimiter._hits` sınırsız büyüyor (IP başına kalıcı girdi) | MEDIUM | **RESOLVED** | `main.py:99,131-175`: `_evict_expired` her `check()` başında penceresi tamamen dolmuş istemcileri siler, `max_clients=10_000` + `_evict_oldest` tavanı LRU-tarzı düşürür; disabled yol (`per_min<=0`) hâlâ `_hits`'e hiç dokunmuyor. Grep ile kaynakta teyit edildi; 3 yeni test (`evicts_clients_whose_window_fully_expired`, `caps_tracked_clients`, `disabled_path_holds_no_state`) 338'lik koşumda yeşil. |
| M2 | `query_encode` aşama süresi semafor kuyruğunu da ölçüyor | MEDIUM | **RESOLVED** | Üç çağrı yeri de (`hybrid.py:204`, `core.py:67`, `core.py:142`) `with ENCODE_LIMIT, stage("query_encode"):` sırasına çevrilmiş — grep ile doğrulandı. Bağlam yöneticisi giriş/çıkış sırası doğru (semafor önce girer/sonra çıkar), istisna yolunda da semafor serbest kalır (nested `with` semantiği). |
| L1 | Boş-sorgu kapısı görsel-yalnız pipeline'lara da BM25 kuralı uyguluyor | LOW | **DISPUTED — kabul edildi, kod değişmedi** | `require_searchable` docstring'ine kontrolcü gerekçesi eklendi ("içeriksiz sorgu reddi ürün-düzeyi, pipeline'dan bağımsız"); fonksiyon gövdesi (`if not tokenize(text): raise HTTPException(422, ...)`) birebir aynı — diff'te yalnız docstring hunk'ı var. Brief'in kendisi bunu "controller ruling" olarak zaten çerçeveliyor; yeniden açılacak bir teknik itiraz bulunmadı. |
| L2 | 422'ler bedava, boş-içerik reddi kota jetonu harcıyordu | LOW | **RESOLVED** | `/search` ve `/ask`'te sıra `require_searchable(...)` → `enforce_rate_limit(...)` olarak değişti (önceden tersti). Yeni parametrized test `test_rejected_request_does_not_consume_rate_limit_token` (`/search`+`/ask`, tavan=1): geçersiz istek 422 + jeton harcanmıyor, ardından gelen geçerli istek 200. |
| L3 | 422/429 telemetriye hiç düşmüyor (`/metrics` kör) | LOW | **RESOLVED** | `telemetry/prom.py`: yeni `Counter("bg_rate_limited", …, ["endpoint"])`; `enforce_rate_limit` 429 anında `prom.rate_limited.labels(endpoint=endpoint).inc()` çağırıyor (record_event yolunun dışında, doğrudan). Canlı `:7860/metrics`'te `bg_rate_limited_total` HELP/TYPE satırları görüldü; `test_rate_limited_requests_are_counted_in_metrics` tam metin eşleşmesini (`bg_rate_limited_total{endpoint="/search"} 1.0`) assert ediyor ve geçiyor. 422'ler kasıtlı olarak bu sayacın dışında (dokümante edilmiş). |
| L4 | Pencere sona ermesi ve X-Forwarded-For test edilmiyor | LOW | **RESOLVED** | İki yeni test: `test_rate_limiter_window_expiry_unblocks_client` (`window_s=0.05`, sleep 0.06, 3. istek sonrası serbest) ve `test_rate_limit_ignores_x_forwarded_for_header` (iki farklı XFF değeri, aynı gerçek istemci → 2. istek yine 429). İkisi de 338'lik koşumda yeşil. |
| L5 | Sevkiyat sayılarının (0.8605/0.8488) repoda bench artefaktı yok | LOW | **RESOLVED** | `git ls-files \| grep 20260830` → `data/bench/results/20260830-1611-6d5b345-hybrid.json` izleniyor. README'de dosyaya bağlantı + yeniden-üretim komutu eklendi. **Bağımsız sayısal doğrulama:** dosyanın `overall.recall_at["5"]=0.8488372…` alanı iddiayla eşleşiyor; `diagnostics[]` (43 soru, `route_fuse` aşaması, 1-indeksli rank) üzerinden elle yeniden hesapladığımda binary recall@5 = **37/43 = 0.8605** ve fractional = **0.8488** — ikisi de README'nin sayılarıyla birebir. |
| N1 | `degraded` kartı `class="abstained"` ödünç alıyor | NIT | **RESOLVED** | `index.html`: üçlü dallanma `status === "answered" ? "" : status === "abstained" ? " abstained" : " degraded"`. Test hem yeni sınıfın varlığını hem eski hatalı ifadenin YOKLUĞUNU assert ediyor. |
| N2 | Bozulmada eski yanıt metni DOM'da kalıyor | NIT | **RESOLVED** | `$("answer-text").innerHTML = ""` iki noktada eklendi: istek başlarken ve `degraded` dalı içinde (savunma amaçlı ikinci temizlik). Test `count('$("answer-text").innerHTML = "";') >= 2` assert ediyor. |
| N3 | `friendly422` her liste-şekilli hatayı "uzunluk" sanıyor | NIT | **RESOLVED** | İlk hatanın `type`ına göre 3 dallı mantık: `string_too_long` → uzunluk mesajı, `greater_than_equal`/`less_than_equal` → k-sınırı mesajı, bilinmeyen → genel Türkçe cümle. Test (sunulan HTML/JS üzerinde sözleşme testi, N1/N2 ile aynı stil) tip string'lerinin kaynakta varlığını assert ediyor. |
| I1 | Pipeline temposu istemci tarafı tahmin | INFO | **N/A — aksiyon gerekmiyor** | Orijinal review zaten "kayıt için not, düzeltme talebi değil" diyordu. Diff'te `index.html:489` (`const pacing = […]`) civarına dokunulmamış — beklenen ve doğru davranış. |

---

## 2. YENİ bulgular

### NEW-1 (NIT, savunmacı programlama — bugün ULAŞILAMAZ)

**Yer:** `src/belge_gozu/app/main.py::RateLimiter._evict_oldest` (main.py:170-175)

`_evict_oldest`, `self._hits` BOŞKEN çağrılırsa `min(self._hits, key=…)` boş dizi üzerinde
`ValueError` fırlatır. Bu yalnız `max_clients <= 0` iken tetiklenebilir: `check()` içinde
`if len(self._hits) >= self.max_clients: self._evict_oldest()` koşulu, boş sözlükte bile
`0 >= 0` (ya da negatif tavanlarda her zaman) doğru olur.

**Bugünkü risk: sıfır.** Production'daki iki kuruluş noktası da (`main.py:425-426`,
`search_limiter = RateLimiter(s.rate_limit_search_per_min)` / `ask_limiter = RateLimiter(s.rate_limit_ask_per_min)`)
yalnızca `per_min` veriyor — `max_clients` her zaman modül varsayılanı `RATE_LIMITER_MAX_CLIENTS = 10_000`.
`Settings`'te (`config.py`) `max_clients`'a karşılık gelen bir alan da yok (grep temiz). Yeni
testler de yalnız `max_clients=3` ve varsayılanı kullanıyor, `0`'ı hiç sondalamıyor.

**Öneri (bir satır, isteğe bağlı backlog):** `_evict_oldest` başına `if not self._hits: return`
savunma satırı — `max_clients` ileride konfigüre edilebilir hale gelirse ya da bir testte
doğrudan `max_clients=0` verilirse sessizce no-op olur, 500'e çıkmaz.

### Değerlendirilip bulgu SAYILMAYAN bir gözlem

L2'nin sıra değişimi (`require_searchable` → `enforce_rate_limit`), boş/aşırı-uzun sorgu
selini sınırlayıcıdan TAMAMEN muaf tutuyor (öncesinde en azından kota jetonu tüketip
nihayetinde 429'a çarpardı). Bilerek incelendi ve bulgu sayılmadı: `require_searchable`
yalnız CPU'da `tokenize()` çalıştırıyor, sınırlayıcının korumaya çalıştığı kaynağa
(LLM kotası / GPU, `RateLimiter` docstring'inde açık) hiç dokunmuyor; L3 zaten 422'lerin
"kasıtlı olarak" sayaç/telemetri dışında kaldığını dokümante ediyor. Aynı mantığın devamı.

---

## 3. Bağımsız doğrulama komutları

| Komut | Beklenen | Sonuç |
|---|---|---|
| `uv run pytest -q -m "not slow"` | 338 passed | **338 passed, 6 deselected in 2.85s** ✅ |
| `make lint` (ruff check + format --check + pyright) | temiz | **All checks passed! 100 files already formatted. 0 errors, 0 warnings, 0 informations** ✅ |
| `git log -1 --oneline` | HEAD = `71e5860` | `71e5860 fix(review): vitrin sprint bulguları…` ✅ (çalışma ağacı temiz, yalnız incelemeyle ilgisiz untracked dosyalar) |
| `git ls-files \| grep 20260830` | bench artefaktı izleniyor | `data/bench/results/20260830-1611-6d5b345-hybrid.json` ✅ |
| limiter grep (`_evict_expired`, `_evict_oldest`, `max_clients`) | M1 kodu mevcut | `main.py:99,131-175` bulundu ✅ |
| semafor/timer grep (`ENCODE_LIMIT, stage("query_encode")`) | 3 çağrı yeri | `hybrid.py:204`, `core.py:67`, `core.py:142` ✅ |
| `curl -s localhost:7860/healthz` | 200 ok | `{"status":"ok","pages":4222,…}` HTTP 200 ✅ |
| `curl -s localhost:7860/metrics \| grep rate_limited` | yeni sayaç kayıtlı | `# HELP bg_rate_limited_total 429 hız sınırı reddi` + `# TYPE … counter` ✅ (per_min=0 olduğundan canlı sunucuda değer artışı yok — birim testiyle ayrıca kilitli) |
| bench json bağımsız yeniden hesap (Python, `diagnostics[]`) | 37/43=0.8605, frac=0.8488 | **birebir eşleşti** ✅ |

---

## 4. Sonuç

11 maddenin 9'u tam RESOLVED, 1'i (L5) hem RESOLVED hem bağımsız sayısal doğrulamayla
GÜÇLENDİRİLMİŞ, 1'i (L1) bilinçli DISPUTED-kabul, 1'i (I1) zaten aksiyon gerektirmiyordu.
Hiçbir düzeltme yeni bir davranış regresyonu getirmedi. Tek YENİ bulgu (NEW-1) bugün
ulaşılamaz bir savunma boşluğu — engelleyici değil, backlog'a düşer.
