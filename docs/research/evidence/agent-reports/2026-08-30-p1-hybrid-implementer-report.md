# P1 — Hibrit getirim üretime alındı (BM25 metin kanalı + doküman-adı yönlendirmesi)

- **Tarih:** 2026-08-29
- **Branch:** `feat/p0-retrieval-correctness` (BASE `1a0624e`)
- **Commit:** `feat(retrieval): P1 hibrit üretimde — BM25 metin kanalı + doküman yönlendirme, eşik 10.6 mekanik taşıma`
- **Girdi:** autoresearch exp7 reçetesi (`research/retrieve.py` @ `1a0624e`), findings
  `docs/research/findings/2026-08-29-autoresearch-text-channel.md`

## 1. Ne yapıldı

Ölçülmüş reçete üretim koduna **birebir** portlandı ve varsayılan getirim yolu oldu.

| Yeni/değişen | Ne |
|---|---|
| `src/belge_gozu/corpus/text.py` (yeni) | `extract_page_texts(pdf_dir, page_ids)` — pymupdf, 1-tabanlı sayfa; eksik PDF/sayfa → boş string, satır sırası `page_ids` ile birebir |
| `src/belge_gozu/retrieval/text.py` (yeni) | `tr_lower`, `STOPWORDS`, `F5=5`, `tokenize`, `BM25Index` (k1=1.5, b=0.75), `extract_doc_name_tokens`, `route_window(window=20)` — sabitler ve skorlama ifadesi `research/retrieve.py`'den VERBATIM |
| `src/belge_gozu/retrieval/hybrid.py` (yeni) | `HybridRetriever` — `ExhaustiveRetriever` ile aynı sözleşme; aşamalar `query_encode` → `exhaustive_maxsim` (görsel, telemetri) → `text_bm25` → `route_fuse`; `PageHit.score` = BM25; `last_retrieval_meta` (ContextVar, istek-yerel) |
| `config.py` | `retrieval_pipeline` Literal'a `"hybrid"` eklendi ve VARSAYILAN oldu; `min_score_threshold` 0.58 → **10.6**; `THRESHOLD_CALIBRATED_ON` → `"hybrid-bm25"`; yeni `PIPELINE_SCORE_SCALE` |
| `app/main.py` | pipeline-duyarlı ölçek korkuluğu; `build_text_channel()` (artefakt + hizalama doğrulaması + BM25 kurulum süresi logu); hibrit dalı; `/healthz` → `pipeline`; `detail.retrieval`'e getirici künyesi (guarded `getattr`) |
| `cli.py` | `belge-gozu index build-text`; `Pipeline` enum'a `hybrid`; `bench run --pipeline` varsayılanı Settings'ten (`DEFAULT_PIPELINE`) |
| `bench/harness.py` | `HybridDiagnosticAdapter` — üç kanalı ayrı `StageRecord`'a yazar (`visual`, `text_bm25`, `route_fuse`) |
| `telemetry/prom.py` | `bg_retrieval_top_score_bm25` + `bg_retrieval_score_margin_bm25`; `observe()` olayın `pipeline` künyesine göre yönlendirir |
| `app/static/index.html` | footer + skor dipnotu; `/healthz` erişilemezse kullanılan THRESHOLD yedeği 0.58 → 10.6 |
| `README.md`, `docs/research/metrics-catalog.md` | reçete + ölçüm tablosu, quickstart `index build-text`, eşik taşıma cümlesi, yeni metrik satırları |

**Görsel kanal serviste KALDI** ama sıralamaya girmiyor: ölçüm (bulgu 3) F5 sonrası
görselin top-5'e benzersiz katkısının SIFIR soru olduğunu, eşit-ağırlık RRF'in ise
zarar verdiğini (0.674 → 0.395) gösteriyor. Her istekte iki kanalın top-1'i yan yana
kaydediliyor (`detail.retrieval.bm25_top1` / `visual_top1`) — P2 kalibrasyonunun girdisi.

## 2. Eşik: mekanik ölçek taşıması, kalibrasyon değil

Ölçüm (retrieval_eval, üretim int8 indeksi + hibrit yol, bu görevde yeniden doğrulandı):

| | değer |
|---|---|
| cevaplanabilir n=43 top-1 | min **10.53** / medyan 26.05 / maks **69.30** |
| korpus-dışı top-1'ler | c003 **23.53**, c004 **12.96**, c005 **17.86**, c007 **15.54** |
| anlamsız kontrol | c006 **4.23** |
| çalışma noktası bandı | **(10.528, 10.712]** → seçilen eşik **10.6** |
| eşiği geçen | **42/43** cevaplanabilir + **4/5** cevaplanamaz — binary@60 / int8@0.58 ile AYNI |

Yani 10.6 bir birim dönüşümüdür, yeni bir karar değil. **Dağılımlar hâlâ iç içe**: üç
gerçek korpus-dışı soru eşiğin üstünde; eşiği yükseltmek cevaplanabilir bandın alt ucunu
(10.53) keser. Kalibrasyon P2'nin işi ve durum `xfail(strict=True)` ile kilitli.

## 3. Ölçek korkuluğu (iki yönlü)

`create_app` en başta, indeks/model yüklenmeden:

- **hybrid**: `0 < eşik ≤ 1.5` → RED ("görsel-ölçek kalıntısı; bm25 ölçeği ~5-70"); `eşik > 200` → RED.
- **exhaustive/two-stage**: `eşik > 1.5` → RED (mevcut binary-kalıntı mesajı).
- **negatif eşikler her kolda SERBEST** — "her zaman cevapla" bilinçli bir kapatmadır,
  ölçek kalıntısı değil (testlerin `-1e9`'u). Bu davranış testle kilitlendi.

Taşınabilirlik uyarısı kuantizasyon ekseninden **pipeline** eksenine taşındı: etkin
pipeline'ın ölçeği `THRESHOLD_CALIBRATED_ON`'dan farklıysa WARNING; hibritte sessiz.

## 4. Doğrulama (hepsi bu görevde koşuldu)

**Testler + lint**

```
uv run pytest -q -m "not slow"   ->  282 passed, 5 deselected in 1.75s
uv run ruff check src tests      ->  All checks passed!
uv run ruff format --check ...   ->  84 files already formatted
uv run pyright                   ->  0 errors, 0 warnings, 0 informations
```

`make lint` ise KIRMIZI — ama yalnızca **BASE'de zaten kırmızı olan** 8 hata yüzünden:
`research/evaluate.py` (4), `research/retrieve.py` (2), `research/prepare.py` (2).
`git stash -u` ile doğrulandı: BASE `1a0624e`'de de aynı 8 hata var. Bu dosyalara
DOKUNULMADI — (a) `research/program.md` bu oturum sırasında başka bir ajan tarafından
değiştirildi (round-2 autoresearch açık), (b) `research/evaluate.py` `retrieve.py`'nin
sha256'sını `results.jsonl`'e künye olarak yazıyor; biçimlendirme o künyeyi değiştirip
defterdeki satırları karşılaştırılamaz hale getirirdi. Kararı controller'a bırakıyorum
(seçenekler: research/*.py'yi formatla, ya da `[tool.ruff] exclude`'a `research` ekle).

**Slow suite**

```
BG_DEVICE=mps uv run pytest -m slow -v
  tests/index/test_encode_mask.py::test_batch_vs_single_sign_determinism        PASSED
  tests/retrieval/test_semantic_retrieval_eval.py::test_retrieval_eval_gold_pages_covered       PASSED
  tests/retrieval/test_semantic_retrieval_eval.py::test_short_query_gold_in_top5        PASSED
  tests/retrieval/test_semantic_retrieval_eval.py::test_long_query_rank_ratchet         PASSED
  tests/retrieval/test_semantic_retrieval_eval.py::test_out_of_corpus_..._threshold     XFAIL
  ================ 4 passed, 282 deselected, 1 xfailed in 20.28s ================
```

XPASS yok. Cırcır artık pipeline'a anahtarlı: hibrit bloğu `long_query_gold_rank_max: 2`
(ölçüldü), exhaustive bloğu 664 olarak KORUNDU.

**Bench (üretim teyidi)**

```
BG_DEVICE=mps uv run belge-gozu bench run --pipeline hybrid --all
bench modu: TÜMÜ (taslak dahil, n=48)
recall@5=0.802 mrr=0.652 ndcg5=0.681 n=43 ci_recall5=(0.6744, 0.9070)
rapor -> data/bench/results/20260829-2042-1952b48-hybrid.json
```

| metrik | değer |
|---|---|
| R@1 | 0.500 |
| **R@5** | **0.8023** |
| R@20 | **0.9070** |
| R@50 / R@200 | 0.9535 / 0.9767 |
| MRR | 0.6519 |

**0.802 vs araştırmanın 0.8140'ı — fark 0.0116 (<0.02 toleransı) ve NEDENİ ÖLÇÜLDÜ:
metrik TANIMI farklı, getirim değil.** `research/evaluate.py`'nin `r_at(k)`'sı
"gold'lardan HERHANGİ biri top-k'da mı" (ikili) sayarken üretim harness'ı
`bench/metrics.py::recall_at_k` kesirli recall kullanıyor (`|rel ∩ top-k| / |rel|`), yani
2 gold'lu bir soruda 1 tanesi gelirse 0.5 sayıyor. Aynı rapordan ikili tanımla
yeniden hesaplandığında **35/43 = 0.8140** — araştırma sayısının BİREBİR aynısı.
R@20 (0.907) ve MRR (0.652) zaten birebir örtüşüyor.

**Canlı servis** (`BG_DEVICE=mps ... serve --port 7860`, ÇALIŞIR BIRAKILDI)

`/healthz` → `{"status":"ok","pages":4222,"threshold":10.6,"top_k":5,"pipeline":"hybrid",`
`"index":{"quantization":"int8","revision":"133444d8c235/train-compat-v1/int8"}}`

| çağrı | sonuç |
|---|---|
| `/search` "İş Kanunu'na göre yıllık ücretli izin süresi ne kadardır?" | top-5: k4857:31 (19.94), **k4857:28 (19.42)**, k4857:29, k4857:26, k4857:50 — gold **rank 2** |
| `/ask` aynı soru | **GERÇEK YANIT**, abstain YOK, atıf `k4857:28` — 14/20/26 gün cetvelini, yer altı işleri +4 gün ve 18/50 yaş istisnasını doğru veriyor |
| `/ask` "Türk Medeni Kanunu'na göre yerleşim yeri nasıl tanımlanır?" | **GERÇEK YANIT**, atıf `k4721:4` — "sürekli kalma niyetiyle oturduğu yer" + birden çok yerleşim yeri olamaz (m.19) |
| `/ask` "asdf qwerty zxcvbn madde hukuk?" | **ABSTAIN** (top1 `k2547:98` = 4.23 < 10.6), atıf yok |

`/metrics`: `bg_retrieval_top_score_bm25_count 4.0` (bucket `le="10.6"` = 1.0 — anlamsız
sorgu), `bg_stage_duration_seconds_count{stage="text_bm25"|"route_fuse"|"exhaustive_maxsim"} 4.0`.

**Korkuluk canlı denemesi** (stub encoder, model yüklemeden):

```
hybrid+0.58     -> IndexCompatibilityError: ... görsel-ölçek kalıntısı ... bm25 ölçeği ~5-70 ...
exhaustive+10.6 -> IndexCompatibilityError: ... eski binary ölçeği (0-128) ya da bm25 ölçeği kalıntısı ...
hybrid+5000.0   -> IndexCompatibilityError: ... bm25 ölçeğinin çok üstünde ...
```

## 5. Artefaktlar ve gecikme

- `data/index-traincompat-int8/page_texts.parquet` — **5.5 MB**, 4222 sayfa, **1** sayfada
  metin katmanı yok (findings'teki 4221/4222 ile birebir). `.gitignore` altında (`data/*`),
  **COMMIT EDİLMEDİ**; `uv run belge-gozu index build-text` ile ~9 sn'de yeniden üretilir.
  `corpus_checksum` yalnız `page_ids.json` + `meta.parquet` okuduğu için manifest'i
  geçersizleştirmiyor (testle kilitlendi).
- Gecikme: BM25 sorgu başına **~2-8 ms** (4222 sayfa); başlangıçta BM25 indeks kurulumu
  **~0.4 sn**. Görsel kanal aynen ~0.24 sn/sorgu ekliyor (telemetri için koşuyor).

## 6. Notlar / controller kararına bırakılanlar

1. **`make lint`** — yukarıdaki §4; `research/*.py`'ye dokunulmadı.
2. **UI `st-scan` etiketi** hâlâ "N sayfada exhaustive MaxSim → ilk 5" diyor. Görev
   "footer + dipnot, başka bir şey yok" dediği için DEĞİŞTİRİLMEDİ; kısmen doğru (görsel
   kanal gerçekten tüm korpusu tarıyor) ama "→ ilk 5" artık BM25'in işi. Tek satırlık
   düzeltme, onay bekliyor.
3. **UI THRESHOLD yedeği** 0.58 → 10.6 yapıldı ("başka bir şey yok"un dışında bir
   değişiklik): dosyanın kendi yorumu yedeğin aynı bantta olmasını şart koşuyor, 0.58
   kalsaydı `/healthz` erişilemediğinde grafik eşik çizgisini %5'e koyup TÜM çubukları
   "eşik üstü" gösterirdi.
4. **`bg_retrieval_score_margin_bm25`** görevde adı geçmiyordu; eklendi. Aksi halde
   hibritin BM25-ölçekli marjları normalize `[-1,1]` marj histogramına karışırdı — T14'ün
   ayıkladığı hatanın aynısı. İki satır katalog notu ile belgelendi.
5. **Hit listesi skor sırasında DEĞİL** (yönlendirme pencere içini yeniden sıralıyor):
   TMK sorusunda `k4721:1 (16.70) → k4721:4 (13.82) → k4721:20 (11.61) → k6102:254 (16.26)`.
   Reçetenin doğru davranışı, ama UI çubukları bu yüzden monoton azalmıyor. Bilgi amaçlı not.
6. **`bench run --pipeline` varsayılanı** Settings'ten okunur hale getirildi (repo'nun
   CRITICAL-1 deseni): sabit kalsaydı üretim hibrite geçince bench sessizce eski yolu
   ölçmeye devam ederdi.
7. `research/program.md` çalışma ağacında değişik görünüyor — **benim değişikliğim değil**
   (eşzamanlı round-2 ajanı). Commit'e dahil edilmedi.

## 7. Yeni testler

`tests/corpus/test_text.py` (4), `tests/retrieval/test_text.py` (15),
`tests/retrieval/test_hybrid.py` (11), `tests/app/test_compat.py` (+8: iki yönlü korkuluk,
negatif eşik parametrik, eksik/hizasız artefakt, pipeline-tabanlı uyarı),
`tests/app/test_api.py` (+4: healthz pipeline, hibrit aşamalar, kanal künyesi),
`tests/telemetry/test_prom.py` (+3: bm25 yönlendirmesi iki yönde + uçtan uca),
`tests/test_cli.py` (+5: build-text), `tests/test_config.py` (+1: ölçek künyesi tutarlılığı).
Hiçbir mevcut iddia gevşetilmedi; görsel-kol testleri açık `retrieval_pipeline="exhaustive"`
ile korundu.

---

# Fix round 1 — inceleme bulguları + pencere 20→50 (2026-08-30)

Girdi: `.superpowers/sdd/2026-08-26-belge-gozu-p1-hybrid-retrieval/p1-hybrid-review.md`
(APPROVE with fixes — 0 Critical / 2 High / 4 Medium / 10 Low) + koordinatör
addendum'u (autoresearch round 2, exp8: yönlendirme penceresi 20 → 50).
**Tüm bulgular düzeltildi; itiraz edilen bulgu YOK.**

## Bulgu bazında

| # | Düzeltme |
|---|---|
| **H1** | Grafana panosuna BM25 paneli. Ana slot (x0,y16) artık `bg_retrieval_top_score_bm25_bucket` — varsayılan pipeline oraya yazıyor; görsel-ölçek paneli id 10 olarak boş alt-sağ slota (x12,y32) taşındı, `sum by (le, quantization)` deseni korundu. Hiçbir panel yeniden akıtılmadı; JSON doğrulandı (id'ler tekil, grid çakışması yok). İki panele de "hangi kolda dolar, boş olması NEDEN doğrudur" açıklaması eklendi. |
| **H2** | Quickstart yeniden sıralandı: `index pull` → `corpus download` → `index build-text` → `serve` (blocking komut EN SONDA). Yayınlanmış Hub indeksinin P1 öncesi push edildiği ve `page_texts.parquet` İÇERMEDİĞİ, bu yüzden metin kanalının PDF'leri gerektirdiği ve `serve --pull`ın tek başına fail-fast edeceği açıkça yazıldı; `index build-text` sonrası yeniden push edilirse artefaktın `--pull` ile geleceği de. Yalnız-görsel kaçış yolu (`BG_RETRIEVAL_PIPELINE=exhaustive` + eşiği taşı) not edildi. |
| **M1** | `BM25_SCALE_PIPELINES` artık `config.pipelines_on_scale(BM25_SCALE)` ile TÜRETİLİYOR. Ölçek adları (`BM25_SCALE`, `VISUAL_SCALE`) tek sabite indirildi; `THRESHOLD_CALIBRATED_ON = BM25_SCALE`. İki sabiti bağlayan test: `test_bm25_routing_set_is_derived_from_the_single_scale_map`. |
| **M2** | README'de R@5'in geçtiği her yere metrik tanımı: ikili (any-gold-in-top-5) **36/43 = 0.8372** vs kesirli **0.8256** — aynı koşum, iki konvansiyon; R@20 iki tanımda da 0.9302. Rapor yolu da yazıldı. |
| **M3** | `index build-text` artık page_ids'te geçen bir dokümanın PDF'i yoksa **listeyle reddediyor** (yüzde eşiği YOK — koordinatör talimatı); `--allow-missing` kaçış yolu var ve uyarı basıyor. Her koşumda doküman başına boş-sayfa kırılımı yazılıyor (1/4222 sağlıklı ile 2500/4222 yarım korpus artık toplama bakınca ayrılıyor). 3 yeni test. |
| **M4** | `st-scan` etiketi düzeltildi: statik yedek "BM25 + doküman yönlendirme → ilk 5"; `/healthz`ten sonra ETKİN pipeline'a göre (hybrid / exhaustive / two-stage) ve `TOP_K`'ya duyarlı yazılıyor. |
| **L1** | Eşik gerekçesindeki tüm sayılar **SERVİS EDİLEN** top-1'e taşındı (medyan 26.05 → **24.02**; min/maks aynı) — README, `config.py`, metrik kataloğu. Üçüne de "yönlendirme daha düşük BM25'li bir sayfayı 1. sıraya koyabilir; bant servis edilen skordan ölçüldü" cümlesi eklendi; kanal top-1 medyanı 26.05 ayrıca `detail.retrieval.bm25_top1` olarak anıldı. |
| **L2** | UI: eşiğin YALNIZ 1. satıra uygulandığı `sec-sub`'ta yazıldı + 1. satıra "eşik bu satıra uygulanır" tooltip'i; listenin skora göre monoton olmayabileceği de açıklandı. Çubuk matematiği (inceleme ✅ demişti) değiştirilmedi. |
| **L3** | Totolojik assert kaldırıldı: `top_score` artık fikstürün parquet'inden BAĞIMSIZ kurulan bir `BM25Index` ile karşılaştırılıyor + `> 1.5` bant kilidi. |
| **L4** | 9 vakalık sınır tablosu: hybrid 0 / 0.0001 / 1.5 / 1.5001 / 200 / 200.0001, görsel 1.5 / 1.5001, two-stage 1.5001. Negatif-eşik testi `two-stage` ile de parametrize edildi. |
| **L5** | Pencere değişmezliği artık **property testi**: 300 rastgele (sıralama, yönlendirilen küme, pencere) üçlüsü — pencere ∈ {0,1,2,5,20,VARSAYILAN,n,n+10}; küme korunumu, kuyruk dokunulmazlığı ve grup-içi sıra korunumu birlikte. |
| **L6** | Artefakt VARLIK kontrolü `require_text_artifact` olarak ayrıldı ve `create_app`in en başına, eşik korkuluğunun yanına alındı — VLM + 474 MB indeks yüklenmeden. HİZALAMA kontrolü (page_ids gerektirdiği için) `load_text_channel`'da kalıyor ve orada da koşuyor. |
| **L7** | `HybridDiagnosticAdapter` artık kompozisyonu yeniden kurmuyor, `HybridRetriever.rank` (public yapıldı) çağırıyor. `visual.latency_ms` `encode_query`'yi ARTIK İÇERMİYOR (üretimde ayrı aşama) ve bu docstring'de yazılı. |
| **L8** | `build_text_channel` `app/main.py`'den `retrieval/hybrid.py`'ye taşındı (`load_text_channel` + `require_text_artifact`). `cli.py` artık `belge_gozu.app.main`'i import ETMİYOR; serve ve bench aynı getirim-katmanı fonksiyonunu çağırıyor. |
| **L9** | `retrieval_eval_expectations.json`'daki `pipeline` anahtarı artık assert ediliyor (blok künyesi ↔ sözlük anahtarı tutarlılığı). |
| **L10** | Katalog §1'deki `detail` satırı gerçek şemayı sayıyor: `hits`, `threshold`, model/device/version, `stages`, `retrieval` (+ hibritte `bm25_top1`/`visual_top1`/`routed_docs`). Ayrıca Grafana iki-panel notu eklendi. |

**İtiraz edilen bulgu yok.** Round 1'de "controller kararına bırakılan" `make lint`
kırmızısı da kapandı: eşzamanlı autoresearch ajanı `research/*.py`'yi biçimlendirdi
(journal #7 notu), `make lint` artık **yeşil**.

## Addendum: yönlendirme penceresi 20 → 50 (exp8)

`WINDOW = 50` (`retrieval/text.py`). Sabitin yorumu yeniden yazıldı: pencere 20'de
R@20 guardrail'i YAPISAL olarak korunuyordu; 50'de bu garanti kalkıyor, yerine ÖLÇÜM
geçiyor ve R@20 korunmakla kalmayıp **yükseliyor** (0.907 → 0.9302). Pencere-İÇİ
yeniden sıralama sözleşmesi (küme değişmez, kuyruk dokunulmaz) `window` değerinden
bağımsız olarak yapısal ve property testiyle kilitli.

**Eşik bandı DEĞİŞMEDİ** (kontrol edildi, servis edilen skorlar üzerinde):
min 10.5284, ikinci 10.7117, medyan 24.0215, maks 69.2982 — yani `(10.528, 10.712]`
ve 42/43 + 4/5 çalışma noktası aynen geçerli; korpus-dışı top-1'ler de birebir aynı
(23.53 / 12.96 / 17.86 / 15.54 / 4.23). Cırcır `long_query_gold_rank_max: 2` aynı kaldı.

## Doğrulama

```
uv run pytest -q -m "not slow"     -> 296 passed, 5 deselected
make lint                          -> All checks passed! / 100 files already formatted / 0 errors (pyright)
BG_DEVICE=mps uv run pytest -m slow -v
   test_retrieval_eval_gold_pages_covered                         PASSED
   test_short_query_gold_in_top5                          PASSED   (kısa sorgu gold rank 1)
   test_long_query_rank_ratchet                           PASSED   (rank 2 <= cırcır 2)
   test_out_of_corpus_retrieval_eval_scores_below_threshold       XFAIL    (XPASS yok)
   ================ 4 passed, 296 deselected, 1 xfailed ================
```

**Bench (pencere 50, yeniden koşuldu):**

```
BG_DEVICE=mps uv run belge-gozu bench run --pipeline hybrid --all
recall@5=0.826 mrr=0.655 ndcg5=0.690 n=43 ci_recall5=(0.7093, 0.9302)
rapor -> data/bench/results/20260829-2115-3a031ca-hybrid.json
```

| | pencere 20 (önceki commit) | pencere 50 (şimdi) | journal #8 |
|---|---|---|---|
| R@5 ikili (any-gold) | 35/43 = 0.8140 | **36/43 = 0.8372** | 0.8372 ✅ |
| R@5 kesirli (harness) | 0.8023 | **0.8256** | — |
| R@20 (ikili = kesirli) | 0.9070 | **0.9302** | 0.9302 ✅ |
| MRR | 0.6519 | **0.6550** | 0.655 ✅ |
| chip1 / chip2 / kısa sorgu | 2 / 2 / 1 | **2 / 2 / 1** | 2 / 2 ✅ |

Her iki rapor da commit'te duruyor (pencere-20 koşumu `20260829-2042-1952b48-hybrid.json`
provenance olarak korunuyor).

**Canlı servis** (runtime Python değişti → yeniden başlatıldı, ÇALIŞIR BIRAKILDI):
`/healthz` → `{"pages":4222,"threshold":10.6,"top_k":5,"pipeline":"hybrid","index":{"quantization":"int8",...}}`;
`/search` "İş Kanunu'na göre yıllık ücretli izin süresi ne kadardır?" → `k4857:31 19.94 →
k4857:28 19.42 → k4857:29 → k4857:26 → k4857:50` (gold rank 2); `/metrics` →
`bg_retrieval_top_score_bm25_count 1.0`, `bg_stage_duration_seconds_count{stage="text_bm25"|"route_fuse"} 1.0`.
