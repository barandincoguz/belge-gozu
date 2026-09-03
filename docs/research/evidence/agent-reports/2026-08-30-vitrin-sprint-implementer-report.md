# Vitrin sprinti — yazım-değişmez retrieval + API sertleştirme + kanal-şeffaf arayüz

**Tarih:** 2026-08-30
**Dal:** `feat/p0-retrieval-correctness`
**BASE (sprint öncesi HEAD):** `a9b1eaae9b7a230bb7a75cdb536872cbb8c489a3`
**Commit:** tek commit — `feat(vitrin): yazım-değişmez retrieval + API sertleştirme + kanal-şeffaf profesyonel arayüz`

---

## A. Retrieval — ascii-fold (exp12) üretime portu

`research/retrieve.py`'den **birebir** taşındı (`src/belge_gozu/retrieval/text.py`):

* `_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")`
* `ascii_fold(s)`
* `_STOP_FOLDED = frozenset(ascii_fold(w) for w in STOPWORDS)`
* yeni `tokenize`: `\w+` → `tr_lower` → `len>1` → **katlama** → katlanmış uzayda stopword eleme → F5

Port doğrulaması: üretim `tokenize` ile `research/retrieve.py::tokenize` 8 örnek girdide
karakter-karakter aynı çıktıyı verdi. Doküman-adı çıkarımı aynı `tokenize`'ı çağırdığı için
katlanmış uzaya kendiliğinden geçti.

### Bilerek "düzeltilmeyen" bir ayrıntı (raporlanabilir)

`_GENERIC = {"kanun", "türk", "türki", "cumhu"}` **aksanlı** kaldı — reçetede de öyle.
Katlamadan sonra `tokenize` `"turk"/"turki"` üretiyor, dolayısıyla bu iki girdi artık hiçbir
token'la eşleşmiyor ve pratikte yalnız `"kanun"/"cumhu"` eleniyor. Sonuç: "Türk ..." ile
başlayan kanunların ad kümesinde `"turk"` KALIYOR, yani o kanunlar ancak sorguda "türk/turk"
da geçerse yönlendiriliyor (**daha dar** bir yönlendirme). 0.8605 tam olarak bu davranışla
ölçüldü; listeyi katlamak reçeteyi ölçülmemiş bir varyanta çevirirdi. Karar: **portu bozma**,
`text.py` içine gerekçeli bir yorum + iki teste açık not yazıldı. Değiştirmek isteyen önce
bench'i yeniden koşmalı.

### Eşik bandı — fold sonrası YENİDEN ÖLÇÜLDÜ (bu sprintin ledger'ı)

Metin kanalı deterministik olduğu için ölçüm model gerektirmeden, servis edilen top-1
(yani `AskService`in eşikle karşılaştırdığı gerçek skor) üzerinde tekrarlandı:

```
sayfa: 4222   adı çıkarılan doküman: 50
cevaplanabilir n=43   min 10.5265   2. en küçük 10.7115   medyan 23.7780   maks 66.6822
bant: (10.5265, 10.7115]          -> 10.6 bandın İÇİNDE, eşik DEĞİŞMEDİ
eşik 10.6 -> cevaplanabilir geçen: 42/43
cevaplanamaz top-1: c006 4.23, c004 12.96, c007 15.54, c005 17.86, c003 23.52
cevaplanamaz geçen: 4/5
kanal (servis edilen değil) top-1 medyanı: 26.05  (değişmedi)
```

Brief'te verilen bant `(10.5265, 10.7115]` ve çalışma noktası `42/43 + 4/5` **birebir**
yeniden üretildi.

### Güncellenen künyeler

* `config.py` — eşik yorumu (fold sonrası bant, ölçüm kaynağı: journal #12 + bu ledger),
  `retrieval_pipeline` yorumundaki R@5 zinciri `0.2326 → 0.8140 → 0.8372 → 0.8605`.
* `config.py` — `query_format_id` artık `QueryFormatChoice` enum (audit C8).
* `retrieval/text.py` modül docstring'i — exp12 satırı + exp13 (DISCARDED) satırı +
  yazım-değişmezlik paragrafı.
* `retrieval/hybrid.py` docstring — `0.2326 -> 0.8372 -> 0.8605`.
* slow retrieval_eval docstring'leri — kısa sorgu (rank 1 / 10.71, katlama sonrası aynı), cırcır
  (hâlâ 2/4222), xfail gerekçesi (23.52/12.96/17.86/15.54, bant `(10.5265, 10.7115]`).
* README — aşağıda.

### Birim testleri

* `tokenize` bilinen çıktıları katlanmış uzaya güncellendi (`"iş"→"is"`, `"görev"→"gorev"`).
* YENİ `test_ascii_fold_maps_the_measured_character_set` (tablo + uzunluk korunumu).
* YENİ `test_tokenize_is_writing_invariant` — aksanlı/aksansız tam kesişim.
* YENİ `test_tokenize_folds_function_words_too` — `"ne icin nasil kac"` = `[]`.
* BM25 elle-hesap testi: sözlük anahtarı `"borç"→"borc"`; **sayısal değerler değişmedi**
  (0.578466 / 0.470004 aynen), çünkü katlama df/dl/idf hesabına değil yalnız token
  kimliğine dokunuyor. Test bunu açıkça assert ediyor (`"borç" not in idx.idf`).
* Doküman-adı testleri `_GENERIC` etkileşimi için dürüstçe güncellendi
  (`{"turk","meden"}`, `"CUMHURİYET KANUNU"` örneği).
* YENİ slow test `test_accentless_query_ranks_identically` — üretim indeksinde iki yazımın
  tam korpus sırası birebir eşit.

**İDDİANIN KAPSAMI (bulundu ve daraltıldı):** ilk yazdığım slow test
`"Turk Medeni Kanununa gore..."` (apostrofsuz) kullanıyordu ve **kırıldı**. Sebep katlama
değil noktalama: `"Kanunu'na"` → `["kanun","na"]`, `"Kanununa"` → `["kanun"]`. Yani
yazım-değişmezlik iddiası "aynı metnin aksanları katlanmış hâli" için geçerli; apostrof
düşürmek ayrı bir yazım değişikliği ve kapsam dışı. Test `"Turk Medeni Kanunu'na gore
yerlesim yeri nasil tanimlanir?"` ile düzeltildi ve kapsam docstring'e yazıldı.
(Yan gözlem: apostrofsuz varyantta gold rank 2 yerine 1 oluyor.)

---

## B. API sertleştirme

| # | İş | Durum |
|---|---|---|
| B1 | `SearchBody.k: int \| None = Field(None, ge=1, le=50)`, `query`/`question` `max_length=500` | ✅ 422, `detail` = pydantic LİSTESİ |
| B2 | Boş-içerik kapısı: `tokenize()` boşsa 422 + `"sorgu boş ya da yalnız işlev kelimeleri içeriyor"`, **iki uç noktada** | ✅ `detail` = DÜZ DİZE |
| B3 | `/ask` gövdesine üst düzey `"status"` (`answered`/`abstained`/`degraded`) | ✅ telemetriye yazılan değerin aynısı |
| B4 | `PageHit.visual_score: float \| None`, hibritte dolu, görsel kollarda `None` | ✅ `/search` + `/ask` |
| B5 | Hız sınırı (varsayılan KAPALI), IP başına kayan pencere, 429 + `Retry-After` | ✅ Dockerfile: ask 10/dk, search 60/dk, `BG_LOG_QUERY_TEXT=false` |
| B6 | `threading.Semaphore(4)` encode çevresinde (hibrit + exhaustive + two-stage) | ✅ `index/encode.py::ENCODE_LIMIT` |
| B7 | `query_format_id` enum + `cli.py` `ValidationError` → Türkçe mesaj + `SystemExit(2)` | ✅ |

Notlar:

* Sınırlayıcı **uygulama başına** (global değil) — testler ve aynı süreçteki ikinci bir app
  birbirinin sayacını görmüyor.
* İstemci kimliği `request.client.host`; `X-Forwarded-For` **bilerek okunmuyor**
  (doğrulanmamış başlığa güvenmek sınırı tek satırla atlatılabilir kılar). Bedeli dürüstçe
  README'ye yazıldı: ters vekil arkasında sınır küresel bir tavana dönüşür.
* `ENCODE_LIMIT` tek yerde (`index/encode.py`) tanımlı, üç getirici de aynı semaforu
  kullanıyor; yorumda "savunmacı sınır, ölçüm: 40@c=8 sağlıklı" geçiyor.
* `visual_score` `score` ile ASLA karışmıyor — ayrı alan, ayrı ölçek, `types.py` docstring'i
  bunu gerekçelendiriyor.

`belge-gozu --help` doğrulaması:
```
$ BG_QUERY_FORMAT_ID=bogus uv run belge-gozu --help ; echo $?
Yapılandırma hatası: ortam değişkenleri (BG_*) ya da .env dosyası geçersiz.
  - query_format_id: Input should be 'cpe-0.3.18' or 'train-compat-v1'
Düzeltip tekrar deneyin (örn. `unset BG_QUERY_FORMAT_ID`).
2
```
Traceback yok; temiz ortamda `--help` normal çalışıyor.

---

## C. Arayüz (frontend-design skill'i UI'dan ÖNCE yüklendi)

Kimlik korundu ve rafine edildi — gazete manşeti, pipeline şeridi, skor grafiği, mühür
hepsi yerinde. Yeni tek renk kararı: **`--scan: #8A6D3B`** (sepya fotokopi mürekkebi) —
ikinci kanalın kendi mürekkebi. İki kanal iki ölçekte skorluyor, bu yüzden aynı maviyle
gösterilmiyorlar; renk burada dekorasyon değil, ölçek ayrımının taşıyıcısı.

1. **Durum güdümlü haller** — `data.status` üzerinden:
   * `abstained` → mühür (döner, kırmızı, çift cetvelli, sağ üstte) — bugünkü gibi;
   * `degraded` → **YENİ** "SERVİS NOTU" bandı: düz, taralı, gri, akışın içinde, sol
     kenar çubuklu. Mühürle **her eksende** zıt olacak şekilde tasarlandı; sayfalar
     gösterilmeye devam ediyor, pipeline'ın 4. adımı "Gemini yanıt veremedi" (sönük)
     oluyor. Metnin kaynağı sunucu (`SERVICE_ERROR_TEXT`), arayüz yalnız sunum ekliyor.
   * `422` → girdi kutusunun ALTINDA satır içi kırmızı mesaj + `aria-invalid`, pipeline
     gizleniyor. Sunucunun iki farklı 422 şekli de karşılanıyor (düz dize → Türkçe
     detay + eylem cümlesi; pydantic listesi → "en fazla 500 karakter" cümlesi).
   * `429` → `Retry-After` saniyesiyle Türkçe mesaj.
   * ağ hatası → "Sunucuya ulaşılamadı. Servisin çalıştığından emin olup tekrar deneyin."
   * Hiçbir yolda ham JSON gösterilmiyor; `err.message` de artık basılmıyor.
2. **Kanal şeffaflığı** — skor grafiği 4 sütunlu bir künye tablosuna dönüştü:
   `belge · sayfa | skor dağılımı | BM25 | görsel`, başlık satırıyla. İki skor **ayrı
   sütunlarda** çünkü ayrı ölçeklerde; tek eksende çizmek onları karşılaştırılabilirmiş
   gibi gösterirdi (T14'ün ayıkladığı hata sınıfı). Eşik etiketi artık başlık satırının
   track hücresine sabit — hizalama yüzdeyle değil grid'in kendisiyle garantili. Sayfa
   kartlarının altyazısında da `görsel 0,71`. Dipnot iki cümle: biri BM25'in kalibre
   edilmemişliği, biri "görsel koşar ama sıralamaya girmez". `visual_score === null`
   (görsel kollar) → sütun ve etiketler tamamen düşüyor.
3. **6 chip** — arşiv fişi biçiminde, her birinde mono dilim etiketi. Seçim ve **canlı**
   sıralar aşağıda.
4. **"Nasıl çalışır"** — ilk yüklemede GÖRÜNÜR yeni bölüm: üç sütun
   (`metin kanalı · sıralar` / `yönlendirme · yeniden sıralar` / `görsel kanal · sıralamaz`
   — numara yok, çünkü görsel kanal aslında sırada değil; etiketler bunu söylüyor) +
   künye satırı (R@5 0.8605 · R@20 0.9302 · 43 soruluk retrieval_eval · 4.222 sayfa / 50 kanun ·
   aksanlı=aksansız) + kırmızı `uyarı:` satırı (kalibre değil, eşik çalışma noktası,
   kalan 6 soru sözcüksel tavan).
5. **Meta + a11y + mobil** — Türkçe `<title>` + description + `og:title`/`og:description`
   (dış görsel yok); skor satırları `<div>` yerine gerçek `<button>` (klavyeyle
   erişilebilir) + açıklayıcı `aria-label`; chip'lerde `aria-label` soru metni, süs
   span'leri `aria-hidden`; `#q-msg` `role="alert"`; `:focus-visible` her etkileşimli
   öğede; girdi kutusunda `maxlength="500"` (sunucu 422'si arkada emniyet olarak duruyor);
   375 px'te tam düzen doğrulandı (yatay taşma yok — gerçek 375 px viewport'ta iframe ile
   ölçüldü, headless pencere genişliği macOS'ta clamp'leniyor); `prefers-reduced-motion`
   hem mühür hem pipeline hem chip geçişlerinde.
6. **Pipeline şeridi** — aşamalar artık dürüst: `sorgu kodlanıyor — görsel kanal` →
   `4.222 sayfada BM25 metin araması + doküman yönlendirme → ilk 5` → `eşik kontrolü
   (10,60)` → `Gemini ...`. Tempo ölçüme yakın: `[0, 1150, 1260, 1340]` ms — encode ~1,1 sn
   (yükün büyük kısmı), BM25+yönlendirme+eşik milisaniyeler, kalan süre LLM'in.

Görsel doğrulama: dört durum (answered / abstained / degraded / 422) stub'lanmış `fetch`
ile aynı HTML+JS üzerinde 1100 px'te, ayrıca gerçek sunucuda masaüstü ve 375 px mobil
ekran görüntüsüyle kontrol edildi.

---

## D. Doğrulama (verbatim)

### D1 — hızlı testler + lint
```
$ uv run pytest -q -m "not slow"
327 passed, 6 deselected in 2.81s

$ make lint
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
100 files already formatted
0 errors, 0 warnings, 0 informations
```

### D2 — slow testler
```
$ uv run pytest -m slow -v
tests/index/test_encode_mask.py::test_batch_vs_single_sign_determinism PASSED
tests/retrieval/test_semantic_retrieval_eval.py::test_retrieval_eval_gold_pages_covered PASSED
tests/retrieval/test_semantic_retrieval_eval.py::test_short_query_gold_in_top5 PASSED
tests/retrieval/test_semantic_retrieval_eval.py::test_long_query_rank_ratchet PASSED
tests/retrieval/test_semantic_retrieval_eval.py::test_accentless_query_ranks_identically PASSED
tests/retrieval/test_semantic_retrieval_eval.py::test_out_of_corpus_retrieval_eval_scores_below_threshold XFAIL

================ 5 passed, 327 deselected, 1 xfailed in 21.08s =================
```
XPASS yok.

### D3 — bench
```
$ uv run belge-gozu bench run --only-verified --out <scratchpad>/bench-postfold.json
bench modu: yalnız doğrulanmış (n=48)
recall@5=0.849 mrr=0.632 ndcg5=0.680 n=43 ci_recall5=(0.7438953488372093, 0.9418604651162791)
```
`overall`: `R@1 0.4535 · R@5 0.8488 · R@10 0.8837 · R@20 0.9302 · R@50 0.9302 · R@200 0.9767 · MRR 0.6320`

* **İkili (binary) R@5 = 37/43 = 0.8605** — beklenenle **birebir** aynı (fark 0.0000).
* **Kesirli (fractional) R@5 = 0.8488** (bench `recall@5=0.849` olarak basıyor).
* MRR 0.632 ve R@20 0.9302 journal #12'nin exp12 sayılarıyla birebir örtüşüyor.
* `paraphrase` dilimi tek başına R@5 0.286 (kalan 6 ıskanın hepsi orada).
* Rapor bilerek scratchpad'e yazıldı — `data/` altına hiçbir şey eklenmedi.

### D4 — canlı (:7860 yeniden başlatıldı, AYNI komut)

Yeniden başlatma: `uv run --directory /Users/barandincoguz/Desktop/project-delta belge-gozu serve --port 7860`
(eski PID 66005/66009 kapatıldı; yeni süreç **çalışır durumda bırakıldı**).

| # | Kontrol | Sonuç |
|---|---|---|
| a | aksansız `/search` "Is Kanunu'na gore yillik ucretli izin suresi ne kadardir?" | ✅ **k4857:28 sıra 2** (`k4857:31 19.91 / k4857:28 19.39 / k4857:29 16.02`), her hit'te `visual_score` dolu (0.67 / 0.70 / 0.71) |
| b | aksansız `/ask` aynı soru | ✅ `status="answered"`, `abstained=false`, **atıf `k4857:28`** (izin çizelgesi), gövde 14/20/26 gün kademelerini doğru veriyor |
| c | `k=100000` | ✅ **422**, `{"detail":[{"type":"less_than_equal","loc":["body","k"],"msg":"Input should be less than or equal to 50",...}]}` |
| d | boş sorgu (`""`, `"   "`, `"bu ne için"`) | ✅ **422** × 6 (her iki uç nokta × üç biçim), `{"detail":"sorgu boş ya da yalnız işlev kelimeleri içeriyor"}` |
| e | 3000 karakterlik sorgu | ✅ **422** ikisinde de, `"String should have at most 500 characters"` |
| f | 6 chip sorusu `/ask` | ✅ 6/6 `status="answered"`, hepsi doğru atıfla (aşağıdaki tablo) |
| g | `/healthz` | ✅ **şekil değişmedi**: `{"status","pages":4222,"threshold":10.6,"top_k":5,"pipeline":"hybrid","index":{"quantization":"int8","revision":"133444d8c235/train-compat-v1/int8"}}` — alan eklenmedi/çıkarılmadı, geriye dönük uyumlu |

### D5 — arayüz grep'leri
```
$ grep -n "ABSTAIN_TEXT\|a.text ===\|dayanak bulamadım" index.html   -> YOK (temiz)
$ grep -c 'class="chip"' index.html                                  -> 6
$ grep -c "visual_score" index.html                                  -> 4
status dallanması: data.status + status === "abstained"/"degraded"/"answered" (11 satır)
```

---

## Seçilen chip'ler ve CANLI sıraları (yeniden başlatma sonrası, post-fold)

| # | Etiket | qid | Soru | Gold | Canlı `/search` sırası | `/ask` |
|---|---|---|---|---|---|---|
| 1 | doğrudan madde | **c001** | Türk Medeni Kanunu'na göre yerleşim yeri nasıl tanımlanır? | `k4721:4` | **2** | ✅ atıf `k4721:4` |
| 2 | madde · çizelge | (vitrin chip 2, retrieval_eval'de yok) | İş Kanunu'na göre yıllık ücretli izin süresi ne kadardır? | `k4857:28` | **2** | ✅ atıf `k4857:28` |
| 3 | tablo · tarife | **c302** | 492 sayılı Kanun (9) sayılı tarife, B sınıfı sürücü belgesi harcı kanunla getirilen miktar nedir? | `k492:69` | **1** | ✅ atıf `k492:69` |
| 4 | tarihî tarama | **c307** | 10 Kasım 1975 tarihli Resmî Gazete'de yayımlanan 7/10445 sayılı Kararname hangi konudadır? | `rg1975a:1` | **2** | ✅ atıf `rg1975a:33, rg1975a:1` |
| 5 | kanun adı geçmiyor | **c110** | Onbir yaşındaki bir çocuk suç sayılan bir eylemde bulunursa hakkında ceza davası açılabilir mi? | `k5237:8` | **3** | ✅ atıf `k5237:8` |
| 6 | aksansız yazım | (yazım varyantı) | yillik ucretli izin suresi ne kadar? | `k4857:28` | **2** | ✅ atıf `k4857:28` |

Seçim yöntemi: `tablo-layout` (4 aday), `tarihi-tarama` (4 aday) ve `paraphrase` (7 aday)
dilimlerinin TAMAMI önce deterministik metin kanalıyla tarandı, gold'u top-3'te olanlar
arasından kısa ve okunur olanlar seçildi. `paraphrase` diliminde top-3'e giren **tek**
soru c110'du (c208 rank 4, geri kalanlar 17-256) — dilimin sözcüksel tavanı zaten
raporlanan zayıflık.

---

## Atlananlar / uyarılar

* **Atlanan iş yok** — A, B, C, D'nin tüm maddeleri tamamlandı.
* `research/` ve `docs/research/findings/` **hiç değiştirilmedi** (yalnız okundu).
* `data/` altına hiçbir dosya eklenmedi; bench raporu scratchpad'e yazıldı.
* Commit'e girmeyenler: `.agents/` ve `skills-lock.json` (sprint kapsamı dışı, önceden de
  untracked'di); `.superpowers/` zaten `.gitignore`'da.
* Sunucu :7860 üzerinde **ÇALIŞIR** bırakıldı.

---

## E. Düzeltme turu — bağımsız inceleme bulguları (2026-08-30)

**Kaynak:** `.superpowers/sdd/2026-08-26-belge-gozu-p1-hybrid-retrieval/vitrin-sprint-review.md`
(commit `6d5b345`, VERDİKT: APPROVE, 2 MEDIUM / 5 LOW / 3 NIT). Çalışma **tek başına**
(alt-ajansız), `research/` ve `docs/research/` **hiç dokunulmadı**.

| # | Bulgu | Yer | Düzeltme |
|---|---|---|---|
| M1 | `RateLimiter` sözlüğü sınırsız büyüyordu (IP başına kalıcı girdi) | `app/main.py` | Her `check()` ÖNCE tüm sözlüğü tarar, penceresi tamamen dolmuş (en son isteği bile `window_s`'ten eski) istemcileri siler (`_evict_expired`); ayrıca `max_clients=10_000` tavanı, dolduğunda en son etkinliği en eski istemciyi düşürür (`_evict_oldest`) — sahte-IP seli de kapatılır. Varsayılan-kapalı yol (`per_min<=0`) hâlâ hiç durum tutmuyor. |
| M2 | `query_encode` aşama süresi semafor kuyruğunu da ölçüyordu | `retrieval/hybrid.py:203`, `retrieval/core.py` (iki getirici) | Bağlam sırası çevrildi: `with ENCODE_LIMIT, stage("query_encode"):` — semafor ÖNCE alınır, kuyruk beklemesi artık `encode_ms`/`bg_stage_duration_seconds{stage="query_encode"}`e karışmaz. Üç çağrı yeri de (hybrid + iki-aşamalı + exhaustive) aynı düzeltmeyi aldı. |
| L1 | Boş-sorgu kapısı görsel-yalnız pipeline'lara da BM25 kuralı uyguluyor | `app/main.py::require_searchable` | **DİSPUTED — davranış DEĞİŞTİRİLMEDİ.** Gerekçe docstring'e eklendi: içeriksiz sorgu reddi ürün-düzeyi bir kuraldır, pipeline'dan bağımsız (kontrolcü kararı). |
| L2 | 422'ler bedava, boş-içerik reddi kota jetonu harcıyordu | `app/main.py` `/search`, `/ask` | Sıra çevrildi: `require_searchable` ÖNCE, `enforce_rate_limit` SONRA — her iki uç noktada. |
| L3 | 422/429 telemetriye hiç düşmüyordu (`/metrics` kör) | `telemetry/prom.py`, `app/main.py::enforce_rate_limit` | Yeni `bg_rate_limited_total{endpoint}` Counter, 429 anında (record_event yolunun DIŞINDA, doğrudan) artırılıyor. 422'ler kasıtlı olarak dışarıda (framework düzeyi). Metrik kataloğuna satır eklendi (`docs/superpowers/specs/2026-08-26-telemetry-design.md`). |
| L4 | Pencere sona ermesi ve X-Forwarded-For kararı test edilmiyordu | `tests/app/test_api.py` | `RateLimiter(window_s=0.05)` ile pencere-geçince-serbest testi; `X-Forwarded-For` başlığı farklı gönderilse de sınırın hâlâ geçerli olduğunu doğrulayan uçtan-uca test. |
| L5 | Sevkiyat sayılarının (0.8605/0.8488) repoda bench artefaktı yoktu | `README.md`, `data/bench/results/` | `uv run belge-gozu bench run --only-verified` HEAD `6d5b345` üzerinde yeniden koşuldu (aynı sonuç: `recall@5=0.849 mrr=0.632 ndcg5=0.680 n=43`), çıktı `data/bench/results/20260830-1611-6d5b345-hybrid.json` olarak commit edildi (diagnostics'ten 37/43=0.8605 ikili değeri elle doğrulandı). README'nin "Which Recall@5?" notuna dosya adı + yeniden-üretim komutu eklendi. |
| N1 | `degraded` kartı `class="abstained"` taşıyordu | `app/static/index.html` | Kendi sınıfı (`" degraded"`) — üçlü dallanma, görsel tasarım DEĞİŞMEDİ (zaten `#answer-text` o durumda gizliydi). |
| N2 | Bozulmada eski yanıt metni DOM'da kalıyordu | `app/static/index.html` | `#answer-text` hem yeni istek BAŞLARKEN hem bozulma dalı içinde açıkça temizleniyor. |
| N3 | `friendly422` her liste-şekilli hatayı "500 karakter" sanıyordu | `app/static/index.html::friendly422` | İlk hatanın `type`ına göre dallanma: `string_too_long` → Türkçe uzunluk mesajı, `greater_than_equal`/`less_than_equal` → k-sınırı mesajı, bilinmeyen → genel Türkçe cümle. |

### Testler

11 yeni test eklendi (`tests/app/test_api.py`): M1 ×4 (süpürme, tavan, pencere-geçince-serbest,
kapalı-yol-durumsuz), L2 ×2 (parametrized, `/search`+`/ask`), L3 ×1, L4 ×1 (X-Forwarded-For),
N1/N2/N3 ×3 (sunulan HTML/JS üzerinde sözleşme testleri).

```
$ uv run pytest -q -m "not slow"
338 passed, 6 deselected in 3.02s        # 327 + 11 yeni

$ make lint
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
100 files already formatted
0 errors, 0 warnings, 0 informations
```

### Canlı doğrulama

* :7860 durduruldu ve `BG_DEVICE=mps nohup uv run belge-gozu serve --port 7860 ...` ile
  yeniden başlatıldı; `/healthz` ~8 sn içinde `ok` döndü (şekil değişmedi).
* Chip `/ask` ("İş Kanunu'na göre yıllık ücretli izin süresi ne kadardır?"): ilk denemede
  Gemini API'den geçici bir `503 UNAVAILABLE` ("high demand") geldi ve servis **tam olarak
  tasarlandığı gibi** `status="degraded"` ile 200 döndü (sayfalar hâlâ geçerliydi) — bu
  fix turunun DEĞİŞTİRDİĞİ hiçbir şeyle ilgisi yok, dış API'nin geçici arızası. Tekrar
  denemede `status="answered"`, atıf `k4857:28` (beklenen gold).
* Boş sorgu (`"bu ne için"`): **422**, `{"detail":"sorgu boş ya da yalnız işlev kelimeleri
  içeriyor"}`.
* Tek kullanımlık ikinci süreç, port **7861**, `BG_RATE_LIMIT_SEARCH_PER_MIN=2`: 1. ve 2.
  `/search` 200, **3.** `/search` → **429**, `Retry-After: 59`, ve o sürecin kendi
  `/metrics`'inde `bg_rate_limited_total{endpoint="/search"} 1.0` görüldü. 7861 tamamen
  durduruldu (port serbest, kalıntı süreç yok); :7860 boyunca kesintisiz **ÇALIŞIR**
  durumda kaldı (`/stats` ve `/metrics` sonrasında da sağlıklı yanıt verdi).
* Sunucu :7860 üzerinde **ÇALIŞIR** bırakıldı.

### Commit

Tek commit: `fix(review): vitrin sprint bulguları — limiter tahliyesi, encode zamanlama,
429 telemetrisi` (dosyalar: `src/belge_gozu/app/main.py`,
`src/belge_gozu/app/static/index.html`, `src/belge_gozu/retrieval/{core,hybrid}.py`,
`src/belge_gozu/telemetry/prom.py`, `tests/app/test_api.py`, `README.md`,
`docs/superpowers/specs/2026-08-26-telemetry-design.md`,
`data/bench/results/20260830-1611-6d5b345-hybrid.json`).
