# P1 hibrit getirim — spec uyumu + kod kalitesi incelemesi

- **Commit:** `ded732b` (BASE `1952b48`), branch `feat/p0-retrieval-correctness`
- **İnceleme tarihi:** 2026-08-29
- **Verdict:** **APPROVE with fixes** — CRITICAL yok. Reçete birebir, korkuluklar spec'e
  harfi harfine uyuyor, çalışma noktası commit edilmiş bench raporundan yeniden
  üretilebiliyor. Bulguların tamamı çevre yüzeylerde (Grafana panosu, README, bir
  kopyalanmış sabit, bir CLI reddi eksiği).

> Not: `research/*.py` ile üretim tokenizer'ı arasındaki kopyalama BİLİNÇLİ
> (araştırma provenance olarak donduruldu) ve görev talimatı gereği bulgu olarak
> raporlanmadı.

---

## 0. Kendi koştuğum doğrulamalar

| Koşum | Sonuç |
|---|---|
| `uv run pytest tests/retrieval/test_text.py tests/retrieval/test_hybrid.py tests/app/test_compat.py -q` | **53 passed** |
| `uv run pytest -q -m "not slow"` | **282 passed, 5 deselected** |
| `uv run ruff check src tests` / `ruff format --check` | temiz / 84 dosya biçimli |
| `uv run pyright` | 0 errors, 0 warnings |
| `GET /healthz` (canlı 7860) | `{"pipeline":"hybrid","threshold":10.6,"pages":4222,"index":{"quantization":"int8",...}}` |
| `POST /ask {"question":"asdf qwerty zxcvbn madde hukuk?"}` | **`abstained: true`**, atıf yok, top1 `k2547:98` = 4.23 |
| `POST /search` TMK sorgusu | `k4721:1 16.70 → k4721:4 13.82 → k4721:20 11.61 → k6102:254 16.26 → k6100:2 16.00` (gold **rank 2**, liste beklendiği gibi MONOTON DEĞİL) |

**Reçete denklik testi (mekanik).** `research/retrieve.py` modülünü ve
`belge_gozu.retrieval.text`'i yan yana yükleyip karşılaştırdım:

- `STOPWORDS` küme-eşit (46 = 46, simetrik fark boş); `F5=5`, `WINDOW=20`,
  `_GENERIC` eşit; `_TITLE_LINE.pattern` ve `_WORD.pattern`+`flags` birebir aynı.
- `tokenize`: 20.000 rastgele Türkçe-alfabe dizisinde **0 fark**; ayrıca tüm
  stopword'ler + `İş/Kanunu/göre/görev/GÖREV/Iğdır/ışık/ANAYASA/m.19/5651/a/ı/İ/I`
  kenar durumlarında birebir. **Sıra doğru:** `len(t) > 1` ve stopword elemesi
  TAM KELİME üzerinde, F5 kırpması SONRA (`text.py:79-80` == `retrieve.py:40-41`).
- `BM25Index.idf` sözlüğü `BM25.idf` ile eşit; `scores()` çıktısı 5 sorguda
  `max|Δ| = 0.0`; `avgdl` aynı.
- **Uçtan uca sıra:** sentetik 3-doküman/9-sayfa korpusta `rank_pages(q)` ile
  `HybridRetriever.rank_all(q)` 5 sorguda **birebir aynı tam sıralama**
  (yönlendirmenin tetiklendiği ve tetiklenmediği kollar dahil); çıkarılan
  doküman-adı sözlükleri de aynı.
- Üretim portundaki tek fark **katı ek korumalar**: `extract_doc_name_tokens`'ta
  `zip(..., strict=True)` ve `BM25Index.__init__`'te uzunluk/boşluk doğrulaması.
  Davranışsal sapma yok. (`research/retrieve.py` `1a0624e`→HEAD arasında yalnız
  kozmetik değişti: bir ara değişken + `strict=True`.)

**Eşik bandının doğrulanması (commit edilmiş artefakttan).**
`data/bench/results/20260829-2042-1952b48-hybrid.json` içindeki 43 cevaplanabilir
sorunun `route_fuse` aşama skorlarından:

| | servis edilen top-1 | ham BM25 top-1 |
|---|---|---|
| min | **10.5284** (c205) | 10.5284 |
| medyan | 24.0215 | **26.0520** |
| maks | 69.2982 | 69.2982 |
| ≥ 10.6 | **42/43** | 42/43 |

İkinci en küçük değer **10.7117** → bandın `(10.528, 10.712]` olduğu iddiası
**birebir doğrulandı**, 10.6 bandın içinde. Ayrıca ikili (binary) tanımla
R@5 = **35/43 = 0.8140**, kesirli tanımla **0.8023** — raporun açıklaması doğru.

**Gecikme iddiaları (gerçek korpus, 4222 sayfa).** BM25 kurulumu **0.39 sn** +
doküman-adı çıkarımı 0.00 sn = **0.40 sn** (README "~0.4 s" ✅); sorgu başına
**2.0–5.1 ms** (README "~2-8 ms" ✅); metin katmanı boş sayfa **1/4222** ✅;
50/56 dokümanın adı çıkarılabiliyor.

---

## 1. Spec kontrol listesi (13 teslimat)

| # | Teslimat | Durum |
|---|---|---|
| 1 | `corpus/text.py` — `extract_page_texts`, 1-tabanlı, satır hizası | ✅ |
| 2 | `retrieval/text.py` — reçetenin VERBATIM portu | ✅ (mekanik olarak doğrulandı) |
| 3 | `retrieval/hybrid.py` — `HybridRetriever`, görsel kanal telemetride | ✅ |
| 4 | CLI `index build-text` | ⚠️ **kısmi** — reddetme kapsamı eksik (M3) |
| 5 | Serve wiring + fail-fast (parquet varlığı + page_id eşitliği) | ✅ |
| 6 | Eşik 10.6, bant `(10.528, 10.712]` | ✅ (artefakttan yeniden üretildi) |
| 7 | Pipeline-duyarlı ölçek korkuluğu (hybrid `(0,1.5]` ve `>200`; görsel `>1.5`; negatif serbest) | ✅ (sınırlar harfi harfine) |
| 8 | `/healthz` `pipeline` alanı | ✅ (tam-eşitlik testiyle kilitli) |
| 9 | `bg_retrieval_top_score_bm25` + `bg_retrieval_score_margin_bm25` | ⚠️ **kısmi** — Grafana panosu güncellenmedi (H1), yönlendirme sabiti kopya (M1) |
| 10 | UI footer + skor dipnotu + JS yedeği 10.6 | ⚠️ **kısmi** — `st-scan` etiketi hâlâ yanlış (M4) |
| 11 | Pipeline-anahtarlı cırcır (hybrid 2 / exhaustive 664) | ✅ (fixture ETKİN pipeline'ı okuyor) |
| 12 | Abstain xfail BM25 sayılarıyla yeniden yazıldı, XPASS yok | ✅ (slow suite controller tarafından koşuldu) |
| 13 | `bench run` hibrit dalı + commit edilmiş rapor + README | ⚠️ **kısmi** — README quickstart sırası çalışmıyor (H2), recall tanımı belirtilmemiş (M2) |

---

## 2. Bulgular

### CRITICAL — yok

Reçete sapması aranan yerde **hiçbir sapma yok** (§0). Korkuluk sınırları,
`AskService` eşik karşılaştırması, telemetri ölçek ayrımı ve fail-fast yolları
spec'e uyuyor.

---

### HIGH

#### H1 — Grafana "Top skor dağılımı" paneli varsayılan pipeline'da KALICI BOŞ
`observability/grafana/provisioning/dashboards/belge-gozu.json:307`

```
"expr": "sum by (le, quantization) (rate(bg_retrieval_top_score_bucket[5m]))"
```

Commit skor serisini pipeline'a göre ikiye ayırdı (`prom.py:162-175`) ve katalog +
README'yi güncelledi, ama **README'nin kendi tanıttığı** provizyonlu panoyu
(`make obs-up` → "dashboard `belge-gozu` pre-provisioned") güncellemedi. Varsayılan
`hybrid` yolunda `bg_retrieval_top_score` serisine **hiç örnek düşmez** — nitekim
`tests/telemetry/test_prom.py::test_hybrid_scores_go_to_the_bm25_series` bunu
açıkça iddia ediyor (`assert "bg_retrieval_top_score_bucket" not in text`).

**Hata senaryosu:** operatör `make obs-up` yapar, üretim trafiği akar, skor paneli
sonsuza dek boş kalır; okuma ya "telemetri bozuk" ya da (daha kötüsü) "skor
kaydedilmiyor" olur — oysa veri `bg_retrieval_top_score_bm25`'te durmaktadır.
Diğer paneller etkilenmiyor: "Aşama süreleri" `sum by (le,stage)` olduğu için
`text_bm25`/`route_fuse`'u kendiliğinden alıyor ✅.

**Düzeltme:** panele ikinci bir target (`bg_retrieval_top_score_bm25_bucket`)
eklemek ya da paneli pipeline'a göre ikiye bölmek.

#### H2 — README quickstart yazıldığı sırayla ÇALIŞMIYOR
`README.md:193-204` (+ `src/belge_gozu/index/hub.py:44-45`)

```bash
# serve straight from the published index + images (no local corpus needed)
BG_HF_DATASET_REPO=... uv run belge-gozu serve --pull     # <-- ilk komut, BLOKLAYICI

# the hybrid (default) pipeline also needs the BM25 text-channel artifact...
uv run belge-gozu corpus download
uv run belge-gozu index build-text
```

Üç ayrı sorun:

1. `serve --pull` **ilk** komut ve süreç bloklar; `index build-text` adımlarına
   sırayla ilerleyen bir kullanıcı hiç ulaşamaz.
2. Yayınlanmış HF indeksi P1'den önce push edildiği için `page_texts.parquet`
   İÇERMEZ → varsayılan `hybrid` pipeline'da o ilk komut `IndexCompatibilityError`
   ile **fail-fast** eder. (Doğru davranış — ama README onu ilk adım olarak
   gösteriyor.)
3. `# no local corpus needed` yorumu artık varsayılan yol için **yanlış**:
   metin kanalı 56 PDF'in tamamını (`corpus download`) gerektiriyor.

`pull_index` üst düzey dosyaları kopyaladığı için (`hub.py:44-45`), indeks
`build-text` sonrası yeniden push edilirse `--pull` artefaktı da getirir; asıl
düzeltme bu + adım sırasının düzeltilmesi.

---

### MEDIUM

#### M1 — `BM25_SCALE_PIPELINES`, `PIPELINE_SCORE_SCALE`'i TÜRETMİYOR (ikinci hakikat kaynağı)
`src/belge_gozu/telemetry/prom.py:34-36` vs `src/belge_gozu/config.py:16-23`

`config.py`'nin kendi yorumu şunu vaat ediyor: *"tek yerde tutulur ki korkuluk,
uyarı ve telemetri yönlendirmesi aynı kaynağa baksın"*. Telemetri **o kaynağa
bakmıyor**; elle yazılmış bir kopya tutuyor:

```python
BM25_SCALE_PIPELINES = frozenset({"hybrid"})   # prom.py:36
```

**Hata senaryosu:** ileride `PIPELINE_SCORE_SCALE`'e `"hybrid-v2": "hybrid-bm25"`
eklenir. Korkuluk doğru davranır, taşınabilirlik uyarısı sessiz kalır (ölçek
eşleşir), `Literal` genişletilir, `test_config.py:43`'teki anahtar kümesi
güncellenir — **ama BM25 skorları normalize `[-1,1]` histogramına dökülmeye
başlar**, yani commit'in önlemek için yazıldığı T14 hatasının aynısı geri gelir.
İki sabiti bağlayan hiçbir test yok.

**Tek satırlık düzeltme:**
```python
BM25_SCALE_PIPELINES = frozenset(p for p, s in PIPELINE_SCORE_SCALE.items() if s == "hybrid-bm25")
```

#### M2 — README 0.814 diyor, commit edilmiş bench raporu 0.8023; fark README'de HİÇ geçmiyor
`README.md:9, 85-86, 255` vs `data/bench/results/20260829-2042-1952b48-hybrid.json`

Ben iki sayıyı da aynı rapordan yeniden hesapladım: ikili (herhangi bir gold
top-k'da) tanımıyla **35/43 = 0.8140**, üretim harness'ının kesirli
(`|rel ∩ top-k| / |rel|`) tanımıyla **0.8023**. Fark yalnızca 2 çok-gold'lu
soruda doğuyor ve **getirim farkı değil, metrik TANIMI farkı**.

Implementer raporu (§4) bunu örnek biçimde açıklıyor; **README açıklamıyor.**
README'yi okuyup `uv run belge-gozu bench run` çalıştıran biri 0.802 görür ve
manşet tabloyla çelişkiye düşer — hiçbir açıklama bulamaz. Projenin bütün duruşu
"ölçüm dürüstlüğü" olduğu için bu tek cümlelik eksik orantısız zarar veriyor.

#### M3 — `index build-text` sessizce BOZUK bir artefaktı kabul ediyor
`src/belge_gozu/cli.py:259-294`

Komut şunları reddediyor: `page_ids.json` yok (`:277`), `manifest.json` yok
(`:281`), `data/pdf` yok (`:286`). Reddetmediği durum: **çıkarımın büyük kısmının
boş çıkması**. `:293-294` boş sayıyı ekrana basıyor ama exit 0 veriyor.

**Hata senaryosu:** `corpus download` yarıda kesilmiş (ağ hatası / Ctrl-C) →
`data/pdf/` içinde 56 PDF'in 20'si var. `build-text` 4222 satırlık, **satır-hizalı**
ve bu yüzden serve'ün hizalama kontrolünden geçen (`app/main.py:88`, yalnız
`page_id` listesini karşılaştırıyor) bir parquet üretir; 2500 sayfanın metni boş
string olur. Servis açılır, hibrit "çalışır", ama korpusun yarısı BM25 tarafından
hiç görülmez. Kısmi bozulma **tamamen sessizdir** (tam bozulmada her sorgu skor 0
alıp abstain'e düştüğü için en azından gürültülüdür).

**Düzeltme:** boş oran bir eşiği aşarsa (ör. >%5; sağlıklı ölçüm **1/4222**)
`typer.BadParameter` ya da açık bir `--allow-empty` bayrağı.

#### M4 — UI aşama etiketi hâlâ "exhaustive MaxSim → ilk 5" diyor
`src/belge_gozu/app/static/index.html:235` (statik yedek) ve `:293` (healthz sonrası)

```
4.222 sayfada exhaustive MaxSim → ilk 5
```

Footer (`:264`) ve skor dipnotu (`:390`) doğru güncellendi, bu satır güncellenmedi.
Varsayılan yolda "ilk 5"i **BM25 seçiyor**; görsel kanal hiç sıralamaya girmiyor.
Kullanıcıya gösterilen tek "sistem nasıl çalışıyor" anlatısı bu üç öğe ve biri
artık yanlış. Implementer bunu §6.2'de bildirip onay bekletmiş — onaylanmalı.

---

### LOW

#### L1 — Eşiğe giren skor ile alıntılanan medyan farklı sayfadan
`README.md:141-142`, `config.py:90-92`, `docs/research/metrics-catalog.md:102`

Üçü de "min 10.53 / medyan **26.05** / maks 69.30" diyor ve README bunu *"the score
that reaches the abstain gate"* diye çerçeveliyor. 26.05 **kanalın** top-1'i
(`text_bm25` aşaması); kapıya giden servis edilen top-1'in medyanı **24.02**
(min/maks aynı). Çalışma noktası etkilenmiyor (10.6'da her iki tanımla da 42/43 —
ikisini de hesapladım), ama cümle "kapıya giden skor" dediği için teknik olarak
yanlış sayfayı alıntılıyor.

#### L2 — UI: abstain kararı `hits[0]`'da, çubuk renkleri satır satır
`index.html:345` ve `:385` (`const above = h.score >= THRESHOLD`)

Yönlendirme listeyi monoton olmaktan çıkardığı için (canlı: 16.70, 13.82, 11.61,
16.26, 16.00) ileride bir eşik "abstain edildi" mesajıyla **yeşil çubukları** yan
yana gösterebilir. Bugünkü 10.6 ile tetiklenmiyor.
**Çubuk matematiği DOĞRU:** `maxV = Math.max(...hits.map(h => h.score), THRESHOLD, 0.01) * 1.08`
(`:338`) ilk değeri değil **maksimumu** alıyor ve `pct()` 0–100'e clamp'liyor
(`:339`) — monoton olmayan ve negatif değerlerde de bozulmuyor. ✅

#### L3 — Bir app testi hiçbir şey iddia etmiyor
`tests/app/test_api.py::test_hybrid_search_records_channel_tops_and_routing`

```python
# servis edilen top-1 BM25 ölçeğinde (görsel normalize banda değil)
assert row[1] == detail["hits"][0]["score"]
```

`build_event` (`app/main.py:326`) `top = hits[0].score` ile türetiyor, `detail.hits`
de aynı listeden doluyor — assertion **totolojik**, ölçek hakkında hiçbir şey
söylemiyor. Gerçek ölçek kilidi başka yerde ve sağlam:
`tests/retrieval/test_hybrid.py::test_page_hit_score_is_bm25_scale` (`> 1.5`).

#### L4 — Korkuluk SINIRLARI test edilmiyor
`tests/app/test_compat.py:264-327`

Testler 0.58 / 10.6 / 5000 / 60 / -1e9 kullanıyor; **1.5, 200 ve 0** hiç
denenmiyor. `app/main.py:209`'daki `<=`'yi `<` yapmak ya da `:216`'daki `>`'yi
`>=` yapmak paketi yeşil bırakır. Ayrıca
`test_negative_threshold_allowed_on_every_pipeline` adına rağmen yalnız
`["hybrid", "exhaustive"]` ile parametrize — `two-stage` kolu kapsam dışı.

*(Sınır mantığının kendisi spec'e UYUYOR: hybrid'de 0 serbest, `(0,1.5]` red,
200 serbest, `>200` red; görselde `>1.5` red, 1.5 serbest; negatif her yerde
serbest. Kontrol ettim.)*

#### L5 — Pencere-küme değişmezliği ÖRNEK testiyle kilitli, property testiyle değil
`tests/retrieval/test_text.py:2992-3023` (diff satırları)

Tek bir sıralama, tek bir `routed_docs`, tek bir pencere. Görev "window-set-invariance
property test" istiyordu. Yük taşıyan davranış (`exp6` vetosu: pencere DIŞINDAN
sayfa çekmeme) ayrı bir örnekle kapsanmış (`test_route_window_does_not_pull_pages_into_the_window`),
yani risk düşük.

#### L6 — Metin artefaktı kontrolü VLM + 476 MB indeks yüklendikten SONRA koşuyor
`app/main.py:249-263` → `:164` → `build_text_channel:78-101`

Eşik korkuluğu tam da bu gerekçeyle `create_app`'in en başına alınmıştı
(`:204-206` yorumu). Parquet varlık/hizalama kontrolü de saf dosya sistemi
kontrolü — aynı yere alınabilirdi. Pratik bedel: tek satırlık "`index build-text`
çalıştır" mesajını almak için dakikalarca model yüklemesi.

#### L7 — Bench adapter'ı sıralama kompozisyonunu YENİDEN kuruyor + `visual` gecikmesi encode'u içeriyor
`src/belge_gozu/bench/harness.py:126-160`

`HybridDiagnosticAdapter.run` `HybridRetriever._rank`'i çağırmıyor; `argsort` +
`routed_docs` + `route_window` üçlüsünü elle diziyor (`:157`). Bugün denk, ama
üretim sıralaması değişirse bench sessizce başka bir şey ölçer — bu commit'in
`bench run --pipeline` varsayılanını Settings'e bağlarken önlediği hata sınıfının
aynısı. Ayrıca `visual_rec.latency_ms = (t1-t0)` **`encode_query`'yi içeriyor**
(`:131-134`), üretimde ise `query_encode` ayrı bir aşama — bench'in "visual"
gecikmesi üretimin `exhaustive_maxsim`'i ile karşılaştırılabilir değil, hiçbir
yerde de not düşülmemiş.

#### L8 — Katmanlama: `bench`/CLI artık `app.main`'e bağımlı
`src/belge_gozu/cli.py:498` — `from belge_gozu.app.main import build_text_channel`

Bench koşumu FastAPI uygulama modülünü (ve dolayısıyla answerer/telemetri
importlarını) çekiyor. Gerekçe ("serve ile AYNI kurulum") sağlam ve yorumla
belgelenmiş, ama fonksiyonun doğru evi `retrieval/` ya da `index/`; oradan
hem serve hem bench çağırabilirdi.

#### L9 — `canary_expectations.json` hybrid bloğunda ölü anahtar
`tests/retrieval/canary_expectations.json`

`"pipeline": "hybrid"` hiçbir yerde okunmuyor (exhaustive bloğunun
`"quantization"`'ı okunuyor ve assert ediliyor — `test_semantic_canary.py:141`).
Ya assert edilmeli ya kaldırılmalı; şu hali "kontrol ediliyor" izlenimi veriyor.

#### L10 — `detail` kolonunun katalog satırı yeni anahtarları saymıyor
`docs/research/metrics-catalog.md:50`

Satır hâlâ "top-5 `[{page_id,score}]`, model adları, device, threshold,
app_version" diyor; `retrieval.bm25_top1` / `visual_top1` / `routed_docs` (ve
zaten eksik olan `stages`) yok. §2'deki BM25 notu ayrıntılı ve doğru, eksik olan
§1 tablosu.

---

## 3. Öncelik başlıklarına göre kapanış

1. **Reçete sadakati** — ✅ **temiz**. Sabit sabit, dal dal karşılaştırıldı ve
   ayrıca uçtan uca sıralama denkliğiyle doğrulandı. Sessiz sapma **yok**.
2. **Eşik / korkuluk doğruluğu** — ✅ sınırlar spec'e harfi harfine uyuyor
   (0 serbest, `(0,1.5]` red, 200 serbest, `>200` red; görselde `>1.5` red;
   negatif her kolda serbest). `THRESHOLD_CALIBRATED_ON` ↔ `PIPELINE_SCORE_SCALE`
   bağı testle kilitli. `AskService` (`answer/base.py:47`) değişmedi ve servis
   edilen `PageHit.score` (BM25) ile karşılaştırıyor — **çalışma noktası tam da
   bu değer üzerinde ölçülmüş** (servis edilen min 10.5284, ikinci 10.7117,
   42/43); tutarlı. Eksik: sınır değerlerinin testi (L4).
3. **Fail-fast bütünlüğü** — ✅ parquet yok / hizasız / two-stage-int8 / manifest
   yok kolları hepsi test edilmiş; **uyumluluk kontrolü hibritte de koşuyor**
   (`build_retriever:148` pipeline dallanmasından ÖNCE, ve `test_create_app_fails_fast_on_mismatch`
   artık varsayılan hibrit altında koşuyor) — görsel indeks kimliği hâlâ zorunlu ✅.
   Eksik: `build-text`'in kısmi-boş artefaktı reddetmemesi (M3), kontrolün model
   yüklemesinden sonraya kalması (L6).
4. **Telemetri yönlendirmesi** — ✅ hibrit top_score/margin YALNIZ `*_bm25`
   serilerine, görsel kollar eskisinde; aşama adları `detail.stages` fallback'iyle
   akıyor ve uçtan uca testle kilitli; katalog satırları doğru. Eksik: Grafana
   panosu (H1) ve kopya sabit (M1).
5. **Sıralama / PageHit semantiği** — ✅ `score` = O SAYFANIN BM25'i, yönlendirme
   sonrası da; `k` dilimlemesi yönlendirmeden SONRA (`hybrid.py:138`); UI
   `maxV` maksimumu alıyor ve clamp'liyor (L2'de detay); `last_retrieval_meta`
   `getattr` korumalı ve ContextVar ile istek-yerel (test var).
6. **Testler iddiaları kilitliyor mu** — çoğunlukla evet; L3 (totolojik assert),
   L4 (sınır değerleri + two-stage), L5 (property yerine örnek), L9 (ölü anahtar).
   Cırcır ETKİN pipeline'ı okuyor (`test_semantic_canary.py:129`) ve fixture aynı
   `get_settings()` + `build_retriever` yolundan geçiyor ✅. conftest parquet
   fixture'ı tiny indeksin `ids` listesini birebir kullanıyor ✅.
7. **Doküman dürüstlüğü** — büyük ölçüde iyi (eşiğin kalibrasyon DEĞİL oluşu,
   örtüşen dağılımlar, 8 ıskanın adlarıyla sayılması). Eksikler: recall tanımı
   (M2), quickstart sırası (H2), medyan alıntısı (L1), `detail` katalog satırı (L10).
   Gecikme iddiaları **ölçüldü ve tuttu**.
8. **Kod kalitesi** — ölü kod yok, adlandırma tutarlı, üretim/araştırma kopyası
   bilinçli (bayrak yok). Gerçek kalite bulguları: M1 (ikinci hakikat kaynağı),
   L7 (bench kompozisyon kopyası + gecikme atfı), L8 (katmanlama).

## 4. Birleştirmeden önce önerilen minimum

1. **H1** panosuna `bg_retrieval_top_score_bm25_bucket` target'ı.
2. **H2** README quickstart sırası + indeksin `build-text` sonrası yeniden push'u.
3. **M1** `BM25_SCALE_PIPELINES`'ı `PIPELINE_SCORE_SCALE`'den türet (1 satır).
4. **M2** README'ye ikili/kesirli recall farkını söyleyen bir cümle.
5. **M4** `st-scan` etiketi (implementer'ın onay beklettiği tek satır).

M3 ve LOW'lar takip işine bırakılabilir; hiçbiri ölçülen davranışı bugün
değiştirmiyor.
