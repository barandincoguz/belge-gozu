# P2 Faz-0 — Yeniden inceleme (fix turu)

**Tarih:** 2026-08-30 · **Mod:** salt-okunur · **Hedef commit:** `d89cee7`
(inceleme turu 1: `p2-faz0-review.md`, hedefi `3851c2c` idi — 3 Orta + 4 Düşük)

**Kapsam notu:** `src/belge_gozu/bench/dataset.py` ve `data/bench/*`'teki commit-dışı
değişiklikler (paralel bir ajanın işi) bu incelemenin dışında bırakıldı; `HEAD == d89cee7`
doğrulandı ve dokunulan 9 dosyanın hiçbirinde commit-dışı fark yok (`git diff HEAD --
<9 dosya>` boş).

**Karar: APPROVE.** Yedi kalemin yedisi de **RESOLVED**. Bir bilgi-amaçlı nit bulundu
(aşağıda N1) — düzeltme gerektirmiyor, kaydı için not edildi.

---

## 0. Doğrulama tabanı

| Koşum | Sonuç |
|---|---|
| `uv run pytest tests/answer tests/telemetry tests/app -q` | **197 passed** (fix öncesi 183 — 14 yeni test) |
| `uv run pytest -q -m "not slow"` | **443 passed, 6 deselected** — rapordaki sayıyla **birebir** |
| `make lint` (ruff check + format + pyright) | **All checks passed** · 102 files formatted · 0 errors |
| Hedefli `-v` koşum (7 kalemin testleri, 39 test) | hepsi **PASSED**, tekil isimlerle doğrulandı |
| L4 doğrulama: `cd tests/telemetry && pytest test_prom.py -k catalog` | **1 passed** — göreli yol artık alt dizinden çalışıyor |
| Canlı `/healthz` (:7860) | `pages=4222 · threshold=10.6 · pipeline=hybrid · int8` |
| Canlı `/stats` (:7860) | `requests=2902 · avg_ms=323.9 · p95_ms=275.4 · by_endpoint={"/ask":38,"/search":2864}` — sunucu ayakta ve tutarlı |

---

## 1. Bulgu bazında sonuç

| # | Bulgu | Durum | Kanıt (bir satır) |
|---|---|---|---|
| **M1** | 15 sn faz-başı sınırı; "≤35 sn toplam" enforce edilmiyordu | **RESOLVED** | `gemini.py:231,246-256`: `started=time.monotonic()` döngüden önce; her retry'den önce `elapsed+backoff_s+timeout_s>total_budget_s` iken `annotate("gemini_retry_skipped_budget", True)` ile retry atlanıyor; yorum garantiyi dürüstçe sınırlıyor ("toplam ≤35 sn değil, bütçe aşılmışken üstüne binmez"); `test_slow_client_degrades_with_timeout_taxonomy_within_budget` gerçek `AskService.ask()` üzerinden `calls==1 · degraded · error_type=timeout` ile skip yolunu uçtan uca tetikliyor. |
| **M2** | `parse` dalı ölü kod, `UnknownApiResponseError` `APIError` dalının içinde kontrol ediliyordu | **RESOLVED** | `gemini.py:122-123`: kontrol artık `APIError` dalının **dışında ve en başta**; `issubclass(UnknownApiResponseError, APIError)` `False` olduğu doğrulandı. İkincil `auth` yanlış-sınıflandırması da kapandı: mesajı `"Invalid API key"` içeren gerçek `genai_errors.UnknownApiResponseError` örneği `test_parse_error_is_not_misclassified_as_auth`'ta `"parse"` dönüyor (eskiden zincirin sonundaki `ValueError`+`_API_KEY_MSG` düşüşüne takılıp `auth` dönerdi). Üç test de **gerçek** `google.genai.errors` sınıfını fırlatıyor, mock değil. |
| **M3** | `rejected` satırları `/stats` ve CLI gecikme istatistiklerini aşağı çekiyordu | **RESOLVED** | `main.py:760-770` ve `cli.py`: yalnız `avg_ms`/`p95_ms` sorguları `WHERE status <> 'rejected'`; `requests` sayımı ve `by_endpoint` (`GROUP BY endpoint`, satır değişmedi) **kasıtlı filtresiz** kaldı. İki test bunu KARŞILIKLI kilitliyor: `test_stats_latency_excludes_rejected_rows` (52 karışık satır → `avg=1500,p95=2000` FİLTRELİ ama `requests=52` FİLTRESİZ) ve `test_stats_requests_and_by_endpoint_still_count_rejections` (tek ret satırı → `requests=1`, `by_endpoint={"/ask":1}` — ret satırı sayılıyor). |
| **L1** | `honest_miss` abstained/degraded satırlarda `0` yazıyordu (NULL değil) | **RESOLVED** | `main.py:482`: `honest_miss = is_honest_miss(answer) if status == "answered" else None`; DDL/model kolonu zaten nullable `INTEGER`/`bool \| None`, şema göçü gerekmedi (`_ADDED_COLUMNS`'taki no-op ALTER önceden vardı, dokunulmadı). Katalog satırı üç değerin anlamını yazıyor (NULL/0/1). Üç test `degraded`/`abstained`/`/search` satırlarında `honest_miss IS NULL`'ı doğruluyor. `/ask` gövdesindeki bağımsız `honest_miss` alanı (line 695) ve `prom.observe`'un `if ev.honest_miss` sayacı bu değişiklikten ETKİLENMEDİ — ikisi de `is_honest_miss()`'in abstained için zaten `False` döndürdüğü davranışa dayanıyor, regresyon yok. |
| **L2** | Arayüzün boş-kabuk savunması `data.hits`'i kapsamıyordu | **RESOLVED** | `index.html:790,880-881`: `const hits = Array.isArray(data.hits) ? data.hits : [];` her iki çağrı yerinde (`renderChart(hits)`, `renderHits(hits)`) kullanılıyor; dosyada başka hiçbir korumasız `data.hits` referansı kalmadı (grep doğrulandı). |
| **L3** | Paralel kolon listesi (4 yer) hiçbir testle kilitlenmemişti | **RESOLVED** | `tests/telemetry/test_schema.py::test_column_lists_stay_in_sync`: `_COLUMNS == ddl_cols[1:]` (**sıra dahil**), `set(_COLUMNS) == set(RequestEvent.model_fields)`, `_ADDED_COLUMNS ⊆ ddl_cols` — üçü birden geçiyor. |
| **L4** | Katalog testi çalışma dizinine bağımlıydı | **RESOLVED** | `tests/telemetry/test_prom.py`: yol artık `Path(__file__).resolve().parents[2] / "docs/research/metrics-catalog.md"`; bağımsız doğrulama: `tests/telemetry` alt dizininden çalıştırıldığında test **geçiyor** (fix öncesi `FileNotFoundError` verirdi). |

**Tally: 7/7 RESOLVED, 0 PARTIAL, 0 NOT.**

---

## 2. Yeni bulgular

**N1 (bilgi amaçlı, düzeltme gerektirmiyor).** `tests/answer/test_gemini.py::test_slow_client_degrades_with_timeout_taxonomy_within_budget`
docstring'i "gerçek duvar saati bütçenin çok altında" diyor ve bunu `real_t0 =
time.monotonic()` ile ölçüyor — ama testin başında `monkeypatch.setattr
("belge_gozu.answer.gemini.time.monotonic", ...)` **stdlib `time` modülünü global
olarak** yamalıyor (Python'da modüller singleton; `gemini.py`'nin `import time`'ı ile
testin kendi `import time`'ı aynı nesneyi paylaşıyor). Yani `real_t0` de sahte saati
okuyor, gerçek duvar saatini değil; assert (`< GEMINI_TOTAL_BUDGET_S`) sahte saat
üzerinden trivially geçiyor. Testin asıl yükünü taşıyan assertion'lar (`models.calls==1`,
`col.notes["degraded"] is True`, `error_type=="timeout"`) gerçek ve anlamlı — M1'in skip
yolunu doğru doğruluyorlar. Sorun yalnız bir docstring/isimlendirme netliği meselesi,
davranışta bir kusur değil; M1'in RESOLVED durumunu etkilemiyor.

---

## 3. Sonuç

M1–M3 (Orta) ve L1–L4 (Düşük) hepsi iddia edilen senaryoyu kapatacak şekilde
düzeltilmiş; hiçbirinde yeni bir regresyon ya da yarım-bırakılmış komşu-katman
tutarsızlığı bulunmadı. Test sayıları (`197`/`443`) rapordaki rakamlarla birebir
örtüşüyor, lint temiz, canlı `/healthz` + `/stats` sağlıklı. Bu faz **APPROVE**.
