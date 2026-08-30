# P2 Faz-0 — Üretim güvenliği + ölçüm tesisatı

**Tarih:** 2026-08-30 · **Dal:** `feat/p0-retrieval-correctness` · **BASE:** `99b8364`
**Kapsam:** e2e-review-2 Y-kalemleri (Y1, Y2, Y15, Y17, Y18, Y20, Y22, Y23, Y28, Y32) +
p2-reality-audit B14. Kalibre selective answering'in (P2) yaslanacağı tesisat.

**Dokunulmayanlar (paralel ajan / kapsam dışı):** `bench/calibration_metrics.py`,
`tests/bench/test_calibration_metrics.py`, `docs/research/findings/`, `docs/superpowers/`.

---

## 1. Ne yapıldı — kalem kalem

### Y15 — Gemini zaman aşımı, tek retry, hata taksonomisi · `answer/gemini.py`

Çağrı senkron uç noktadan yapılıyor, yani Starlette'in iş parçacığı havuzundan bir iş
parçacığı **tutuyor**. Zaman aşımı yokken asılı bir TCP bağlantısı onu süresiz tutuyordu:
40 istek sonrası `/healthz` dahil her senkron uç nokta yanıt veremez hâle geliyordu.

- `genai.Client(..., http_options=types.HttpOptions(timeout=15_000))` — SDK sözleşmesi
  **milisaniye**; alttaki httpx istemcisine connect+read olarak geçer.
- **Bütçe:** 2 deneme × 15 sn + 0.5 sn backoff = **30.5 sn ≤ 35 sn** tavanı; aritmetik
  `test_timeout_budget_stays_under_the_declared_ceiling` ile kilitli.
- **Tek retry, yalnız `timeout`/`http_5xx`.** 429 bilinçle dışarıda: kota aşımında ikinci
  istek durumu yalnız kötüleştirir ve faturayı büyütür. `auth`/`safety_block`/`parse`
  deterministiktir.
- **Taksonomi** (`answer/base.py: ERROR_TYPES`): `timeout · http_5xx · http_429 · auth ·
  safety_block · parse · other`. Sınıflandırma **isimle değil davranışla** (HTTP kodu,
  timeout sınıfı): SDK'nın iç sınıf adları sürümlerle değişir, "5xx = sağlayıcı kesintisi,
  retry et" kararı değişmez.
- `AnswererError(error_type)` — `AskService` bunu `annotate("error_type", ...)` ile
  degraded yoluna taşır.
- **K33 kapandı:** `_ensure_client` artık `threading.Lock` ile korumalı (çift kurulum +
  istek yolunda çakışan yavaş SDK import'u).
- İstemci kurulum hatası da taksonomiye giriyor: anahtarsız dağıtım artık `other` değil
  **`auth`** raporluyor.

### Y20 — `events.error_type` degraded satırlarda dolu

**Sebep (114/114 NULL):** `AskService` istisnayı yutup yalnızca `annotate("degraded", True)`
yazıyordu; `app/main.py`'deki `error_type` kaydı ise **yalnız istisna `service.ask`'ten
kaçarsa** çalışıyordu — yanıtlayıcı hataları için asla kaçmaz. Bağlantı kuruldu:
`build_event` artık `error_type or col.notes["error_type"]` okuyor.

Sınıflandırılamayan hata `type(exc).__name__` **değil** `"other"` olur: telemetriyi SDK'nın
iç sınıf adlarına bağlamak operatöre eylem söylemez.

### Y1 — BM25 sorgu-terim doygunluğu (qtf ≤ 2) · `retrieval/text.py` + `research/retrieve.py`

`scores()` sorgu token'ları üzerinde döngü kurup her tekrar için aynı terimi yeniden
topluyordu; klasik BM25'in `k3` doygunluk çarpanı bu portta yoktu çünkü **canary'de hiçbir
sorgu terim tekrar etmiyor** — ölçüm uzayı bu sınıfı hiç görmemişti. `MAX_QUERY_CHARS=500`
bunu kapatmıyordu, yalnız ölçekliyordu.

Yeni: `Counter(tokenize(query))` üzerinde gezilir, ağırlık `min(qtf, QTF_CAP=2)`. Aynı
değişiklik `research/retrieve.py`'ye **birebir** uygulandı (reçete paritesi) ve docstring'e
ölçüm satırı eklendi.

**Parite koşumu — `uv run python research/evaluate.py exp14-qtf-cap2-parity`:**

```
[exp14-qtf-cap2-parity] R@5=0.8605  R@1=0.4651  R@20=0.9302  MRR=0.632  visual_R@5=1.0 (n=43, görsel 8)
  vaka: chip1-uzun rank=2  chip2-izin rank=2
  -> research/results.jsonl (retrieve_sha 237ff2ed9ccd)
```

exp12 (ölçülmüş taban) ile **her metrikte ve her iki vaka sırasında birebir aynı** — yani
tavan ölçülen uzayda hiçbir şeyi değiştirmedi, ölçülmeyen uzayda saldırıyı kapattı.

Testler: 80× tekrar tek-geçişin **≤ 2.05×**'i (canlı: 667.5 → 16.7), 80 tekrar = 2 tekrar,
ve tekrarsız sorguda elle hesapla birebir eşitlik.

### Y18 — Olay hijyeni: `pipeline` + `score_scale`

**(a) `pipeline` neden %96 NULL — bulgu:** kod yolu hatası **değil**. `build_event` her
zaman `pipeline=s.retrieval_pipeline` yazıyor ve **her iki uç nokta da** bu yoldan geçiyor.
2761 NULL satırın tamamı `2026-08-26T17:53–18:22` aralığından, yani kolonun `ALTER TABLE`
ile eklenmesinden **önceki** trafikten (çoğu loadgen/bench). Bir migrasyon o satırların
hangi pipeline'da üretildiğini bilemez. Doğrulama: `/ask` ve `/search` için ayrı ayrı
parametrik test.

**(b) `score_scale` TEXT kolonu** eklendi (migrasyon + DDL), değeri **tek kaynaktan**:
`config.PIPELINE_SCORE_SCALE[pipeline]` (vitrin sprintinin haritası). İkinci elle yazılmış
harita yok — bir test kolon değerini doğrudan o sözlükle karşılaştırıyor. Geçmiş satırlar
**NULL kalır** (dürüstlük): hangi ölçekte oldukları bilinmiyor, uydurulmaz.

### Y17 + Y32 + K27 — Dürüst-ıska birinci sınıf

- `answer/base.py: HONEST_MISS_MARKER = "verilen sayfalarda bulamadım"` — **tek kaynak**.
- Gemini SYSTEM istemi bunu **f-string ile gömüyor** (S35/D3 borcu kapandı): modele
  dayatılan ifade ile sunucunun aradığı ifade artık ayrışamaz.
- `is_honest_miss(answer)` = `HONEST_MISS_MARKER in tr_lower(text)` — `tr_lower`
  `retrieval/text.py`'den **yeniden kullanılıyor**, kopyalanmıyor. `str.lower()`
  "BULAMADIM"ı "bulamadim" yapardı (I→i) ve işaret eşleşmezdi.
- Tam ifade aranıyor, çıplak "bulamadım" değil: *"…bir istisna bulamadım ama m.45'te
  düzenlenmiştir"* eski sezgide yanlış pozitifti.
- **Tek kod yolu:** `/ask` gövdesindeki `honest_miss`, `events.honest_miss` kolonu ve
  `bg_honest_miss_total` üçü de bu fonksiyondan besleniyor.
- `/ask` yanıtı üst düzey `honest_miss: bool` kazandı. `status`'a **dördüncü değer
  eklenmedi**: dürüst ıska bir `answered` alt durumudur ve yeni bir `status` değeri her
  mevcut istemciyi sessizce yanlış dala düşürürdü.

**Canlı sondaj sırasında bulunan ve düzeltilen ek madde:** yumuşak yönerge ("açıkça '…'
de") ile model ıskayı kendi sözcükleriyle yazıyordu (*"…bilgi bulunmamaktadır"*) ve mühür
araması — eskisi de yenisi de — bunu **kaçırıyordu**. İstem "yanıtında TAM OLARAK şu
ifadeyi kullan" biçimine çevrildi. Uyumun garanti olmadığı (S35 borcunun kalan yarısı)
hâlâ doğru ve raporun §3'ünde açık bırakıldı.

**Arayüz (Y32):**
- `status` anahtarı açık ve **kapalı** bir küme: `KNOWN_STATES = [answered, abstained,
  degraded]`. Tanınmayan durum nötr bir **"bilinmeyen durum"** kartına düşer — `#answer-text`
  ve atıf çipleri **gizlenir ve DOM'a hiç girmez**, aşama şeridi "Gemini sayfaları okudu"
  **diyemez** ("son aşama bilinmiyor"). Eskiden üçlü operatörün varsayılan kolu sınıf
  veriyordu ama alttaki anahtarlar katı eşitlikle çalıştığı için bilinmeyen durum **tam
  dayanaklı yanıt gibi** çiziliyordu — yani Y17'yi Y32 olmadan eklemek arayüzü sessizce
  yalancı yapardı.
- Dürüst ıska (`answered` + `honest_miss=true`) kendi yumuşak durumunu aldı:
  **"sayfalarda bulunamadı"** bandı. Mühürden **her eksende farklı** (dönmez, kırmızı değil,
  kutu değil bant, sağ üstte değil akış içinde, sepya = görsel-kanal mürekkebi) çünkü
  **anlamı farklı**: mühür "eşik geçilemedi, LLM hiç çağrılmadı" der; bu "sayfalar getirildi,
  model onlarda kanıt bulamadı" der. Yanıt metni gizlenmez, yalnız yumuşatılır.

### Y23 — 422/429 artık görünür

- `guard(endpoint, text, request, limiter)`: doğrulama + hız sınırı + **ret olayı** tek
  yerde. L2 sırası (doğrulama önce, sınırlayıcı sonra) korundu.
- Minimal olay satırı: `status='rejected'`, `error_type ∈ {validation, rate_limited}`,
  `endpoint`, `total_ms`, sorgu kimliği. **Skor/aşama/atıf alanları NULL** — getirici
  çağrılmadı, doldurulsalardı P2'nin okuyacağı tabloya sahte sıfırlar girerdi.
  `log_query_text=false` ret satırlarında da geçerli (sha256 her koşulda yazılır).
- `bg_rejected_total{reason}` sayacı. `bg_rate_limited_total{endpoint}` **kaldırılmadı**:
  ekseni farklı ("hangi uç nokta sınırlanıyor?" ↔ "istekler neden reddediliyor?"). Bir 429
  iki seride görünür ama **tek seri içinde iki kez sayılmaz**.
- `prom.observe` ret satırlarında **kullanılmıyor**: sub-ms `total_ms` gecikme
  histogramına karışıp uç nokta p95'ini aşağı çekerdi.
- **Kapsam sınırı, dürüstçe:** pydantic düzeyinde reddedilenler (gövde `max_length`, `k`
  aralığı) uç nokta gövdesine hiç ulaşmadan 422 döner, olay yazılmaz. Testle kilitli,
  katalogda yazılı.

### Y22 — Telemetri yazma hataları: kısıtlı ama canlı

`_warned: bool` (ilk hata WARNING, sonrası **sonsuza dek** sessiz) yerine
`WRITE_ERROR_LOG_INTERVAL_S = 60` hız-sınırlı log + `write_failures` sayacı + **iki log
arasında yutulan hata sayısı** satırda raporlanıyor. Disk dolduğunda sistem hizmet vermeye
devam ediyor ve `/metrics` normal görünüyor; delik artık sessiz değil. Testler
monkeypatch'lenmiş saat + patlayan yazıcı ile.

### Y28 — Arayüz çift-gönderim

`let inFlight` + `setBusy(on)`: düğme, **Enter tuşu ve çipler aynı kapıdan** geçiyor
(`if (inFlight) return;`). Görsel devre dışı durum: düğme `disabled`, `#chips` `.busy`
(opacity + `pointer-events: none` + `aria-busy`). Bayrak `finally`'de bırakılıyor — başarı,
hata ve 422/429 erken dönüşlerinin **tek ortak çıkışı**. Eski yalnız-düğme koruması
kaldırıldı (test bunu da kilitliyor).

### B14 — `[Sk]` ↔ görüntü açık bağlama (G2.2 önkoşulu)

`contents=[*parts, prompt]` idi: önce beş **etiketsiz** görüntü, sonra tek metin bloğu.
Modelin k'ıncı görüntüyü `[Sk]` ile eşlemesi tamamen konumsal çıkarımdı — bu düzeltilmeden
ölçülecek `citation_precision` **konumsal şansı ölçerdi**.

Yeni `build_contents`: `"[S1] <doc> sayfa <n>"` → görüntü₁ → `"[S2] …"` → görüntü₂ → … →
istem. `build_prompt` künye listesini **bıraktı** (ikinci bir kopya sessizce ayrışabilirdi)
ve yerine kuralı söylüyor: *"bir etiket, ondan SONRA gelen görüntüye aittir"*. Testler yapı
düzeyinde: kurulan `contents` listesi stub SDK istemcisiyle inceleniyor, **canlı çağrı yok**.

### Y2 — Karar: kod değişikliği YOK

README dağıtım cümlesi eklendi: kamu dağıtımında hız limiti açık olmalı (Docker öntanımı
zaten açık); BM25 tek-işlem GIL sınırı **bilinen bir ölçek sınırıdır**, arıza değil
(9.4 ms normal sorgu → 119.2 ms 500 karakterlik sorgu). Başka hiçbir şey yapılmadı.

### Yan ürün: metrik kataloğu

`bg_rate_limited_total` **katalogda hiç geçmiyordu** (Y21). Artık bir test kataloğu
registry'den türetilen seri kümesiyle karşılaştırıyor — yeni seri eklemek kataloğu
güncellemeye **zorluyor**. Eklenen/düzeltilen satırlar: `bg_rate_limited_total`,
`bg_rejected_total`, `score_scale`, `honest_miss` (ölü satır referansı `main.py:118` →
gerçek kaynak), `error_type` (taksonomi), `status` (`rejected`), + BM25 bandı ve ret
kapsamı notları.

---

## 2. Doğrulama

### Süit + lint

```
uv run pytest -q -m "not slow"   -> 429 passed, 6 deselected            (BASE: 338)
make lint                        -> ruff check OK · ruff format OK · pyright 0 errors
BG_DEVICE=mps uv run pytest -m slow -v
                                 -> 5 passed, 429 deselected, 1 xfailed   (XPASS yok)
                                    xfail = test_out_of_corpus_canary_scores_below_threshold
                                    (K3 hâlâ dürüstçe kilitli — P2'nin var oluş sebebi)
```

### Reçete paritesi (exp14)

```
[exp14-qtf-cap2-parity] R@5=0.8605  R@1=0.4651  R@20=0.9302  MRR=0.632  visual_R@5=1.0 (n=43, görsel 8)
  vaka: chip1-uzun rank=2  chip2-izin rank=2
  -> research/results.jsonl (retrieve_sha 237ff2ed9ccd)
```

exp12 satırı ile karşılaştırma: `R@1 0.4651 · R@5 0.8605 · R@20 0.9302 · MRR 0.632 ·
visual_R@5 1.0 · chip1 2 · chip2 2` — **yedi alanın yedisi de birebir aynı.**

### Canlı sondajlar (:7860, `BG_DEVICE=mps`, restart sonrası)

`/healthz` → `pages=4222, threshold=10.6, pipeline=hybrid, index=int8,
revision=133444d8c235/train-compat-v1/int8`

**(a) Normal çip `/ask` — yeni interleaved bağlama ile atıf sağlığı**

```
soru : "Türk Medeni Kanunu'na göre yerleşim yeri nasıl tanımlanır?"
->     status=answered  honest_miss=False  citations=['k4721:4']
hits : k4721:1 16.69 · k4721:4 13.82 · k4721:20 11.60 · k4721:39 9.74 · k4721:203 9.20
metin: "…yerleşim yeri, bir kimsenin sürekli kalma niyetiyle oturduğu yerdir [S2]. …
        bir kimsenin aynı zamanda birden çok yerleşim yeri olamaz… [S2]."
```

Atıf sağlığı: model **top-1'i değil `[S2]`'yi** gösterdi ve `[S2]` = `k4721:4`, yani TMK'nın
yerleşim yeri maddesinin (m.19) bulunduğu sayfa. İçerik gerçekten o sayfadan; beş sayfanın
hepsi doğru dokümandan (yönlendirme çalışıyor). Eski konumsal bağlamada bu eşleşme
**doğrulanamaz** bir varsayımdı.

**(b) Korpus-dışı `/ask` → dürüst ıska sözleşmede**

```
soru : canary c003 (Türk Vatandaşlığı Kanunu — korpusta YOK)
top-1: 23.52  (eşik 10.6 -> fren GEÇİRDİ, ücretli çağrı yapıldı: K3 hâlâ açık)
->     status=answered  honest_miss=True  citations=[]
metin: "verilen sayfalarda bulamadım"
```

Arayüz tarafı: `honest_miss=true` + `status=answered` → `.answer-card.honest-miss` +
"sayfalarda bulunamadı" bandı (mühür DEĞİL, servis notu DEĞİL). Aynı sorunun **istem
düzeltmesinden önceki** koşumu tabloda `id=2892, honest_miss=0` olarak duruyor — model
"…bilgi bulunmamaktadır" yazmıştı; düzeltme sonrası `id=2894, honest_miss=1`.

**(c) `"ihbar"×80` (479 karakter) → doygunluk tavanı**

```
top-1 = 16.69  (k6100:14, Hukuk Muhakemeleri Kanunu)
tek geçiş "ihbar" -> top-1 = 8.34  (aynı sayfa)   =>  oran tam 2.00x
sonuçlar: HMK s.14 · VUK s.14 · CMK s.64 · TTK s.198 · VUK s.15  — hepsi TEK konu (ihbar)
```

Bulgudaki ölçüm **667.50**'ydi (eşiğin 63 katı). Şimdi 16.69 — eşiğin 1.6 katı, ölçülen
bandın içinde ve tek bir istek artık `bg_retrieval_top_score_bm25`'in `_sum`'ını ele
geçiremiyor (`_count=4, _sum=65.25`).

**(d) sqlite — yeni satırların kolonları**

```
id    endpoint  status    pipeline  score_scale  honest_miss  error_type  top1
2893  /ask      answered  hybrid    hybrid-bm25  0                        16.69   <- (a)
2894  /ask      answered  hybrid    hybrid-bm25  1                        23.52   <- (b)
2895  /search   ok        hybrid    hybrid-bm25  (NULL)                   16.69   <- (c)
2896  /search   ok        hybrid    hybrid-bm25  (NULL)                    8.34
2897  /ask      rejected  hybrid    hybrid-bm25  (NULL)       validation  (NULL)  <- 422
```

`/metrics`: `bg_rejected_total{reason="validation"} 1.0` · `bg_honest_miss_total 1.0`.
`bg_rate_limited_total` yok çünkü hız sınırı yerelde varsayılan **kapalı** (Y2 kararı) ve
sondajda hiç 429 üretilmedi — beklenen davranış.

Sunucu **çalışır bırakıldı** (:7860).

---

## 3. Açık bırakılanlar (bilinçli)

- **Mühür uyumu garanti değil.** İstem tam ifadeyi dayatıyor ama modelin ona uyacağının
  garantisi yok — S35 borcunun kalan yarısı. Yapısal vekil (`not abstained and
  citations == []`) hâlâ elde ve P2 için daha sağlam olabilir; bu commit onu **etiket
  olarak seçmedi**, yalnız ikisini de kaydediyor.
- **Y16** (boş `resp.text` "answered" sayılıyor) kapsamda değildi; yalnız güvenlik bloğu
  hâlinde `safety_block` fırlatılıyor, diğer boş yanıtlar davranışça değişmedi.
- **Y26** (`degraded` de `abstained=True` yazıyor) kapsamda değildi; ayrım `status` ve artık
  `error_type` üzerinden yapılabiliyor.
- **Y3/Y19** (sıralama monotonluğu, bucket taşması), **Y5** (OOV `/search`), **Y11**
  (`/stats`+`/metrics` korumasız), **Y13** (metin artefaktı checksum dışı) ve arayüzün
  kalan kalemleri (Y29/Y30 itibar, Y33 sahte tempo, Y35 sabit eşik, Y37 AbortController,
  Y38 a11y) bu fazın kapsamında değildi.
- **Geçmiş satırlar NULL kalıyor.** `pipeline`/`score_scale`/`honest_miss` geriye dönük
  doldurulmadı — P2 verisi `pipeline IS NOT NULL` ile çekilmeli.

---

## §fix — İnceleme turu 1 (`p2-faz0-review.md` → 3 Orta, 4 Düşük)

İncelemenin teşhisi kabul edildi ve yedi kalemin hepsi kapatıldı. Ortak desen — incelemenin
kendi ifadesiyle *"bir katmanda doğru düşünülmüş, komşu katmanda uygulanmamış"* — üç Orta
bulgunun da tam olarak açıklaması:

| # | Bulgu | Ne yapıldı |
|---|---|---|
| **M1** | 15 sn httpx'in **faz başına** sınırı; "≤35 sn toplam" enforce edilmiyordu | `generate()` artık döngüden önce `time.monotonic()` alıyor ve **her retry'den önce** `elapsed + backoff + timeout > total_budget_s` ise retry'yi **atlıyor** (`annotate("gemini_retry_skipped_budget")`). `total_budget_s` yapıcı parametresi oldu. Yorum httpx'in `Timeout(connect/read/write/pool)` semantiğini **dürüstçe** anlatıyor ve garantinin sınırını yazıyor: *"toplam ≤ 35 sn" değil, "bütçe aşılmışken ÜSTÜNE bir deneme daha BİNMEZ"* — sert duvar-saati kesmesi ayrı bir iptal mekanizması ister ve bu fazın kapsamında değil. |
| **M2** | `parse` dalı ölü kod (`UnknownApiResponseError` `ValueError`dan türer, `APIError`dan değil) | Kontrol `APIError` dalının **dışına ve en başa** alındı. Konum ayrıca incelemenin işaret ettiği ikincil riski kapatıyor: bu istisna ham gövdeyi mesajına gömdüğü için "API key" geçen bir hata sayfası `_API_KEY_MSG` desenine takılıp **`auth`** raporluyordu. Üç test (`ValueError` mirası + `parse` sınıflaması + `auth`'a düşmediği + uçtan uca). |
| **M3** | Ret satırları `/stats` ve CLI gecikme sayılarını aşağı çekiyordu | `/stats`'ın `avg_ms`/`p95_ms`'i ve `metrics summary`'nin aynı ikilisi `WHERE status <> 'rejected'`. `requests` ve `by_endpoint` **kasıtlı olarak filtrelenmedi** — reddedilen istek de gelmiş bir istektir; filtrelenen yalnız gecikme istatistiği. Karışık fixture DB ile üç test (50 sub-ms ret + 2 gerçek satır → `avg=1500, p95=2000, requests=52`). Katalog notu eklendi. |
| **L1** | `honest_miss` abstained/degraded satırlarda artık `0` (NULL değil) | Kolon **yalnız `status='answered'` satırlarında** 0/1; diğerlerinde NULL. Üç değerin üç anlamı katalogda ve kodda yazılı: NULL = hesaplanmadı · 0 = hesaplandı, ıska yok · 1 = hesaplandı, ıska var. Bu, incelemenin işaret ettiği "`WHERE honest_miss IS NOT NULL` sessizce farklı bir popülasyon seçer" sorununu kapatıyor. Üç test (degraded/abstained/`/search`). |
| **L2** | Arayüzün boş-kabuk savunması `data.hits`'i kapsamıyordu | `const hits = Array.isArray(data.hits) ? data.hits : [];` ve `renderChart`/`renderHits` onu kullanıyor — yorumun **adlandırdığı** senaryo artık gerçekten kapalı. |
| **L3** | Paralel kolon listesi borcu (dört liste, testsiz) | `test_column_lists_stay_in_sync`: `_COLUMNS == DDL[1:]` (sıra dahil) · `set(_COLUMNS) == set(RequestEvent.model_fields)` · `_ADDED_COLUMNS ⊆ DDL`. INSERT'e girmeyen kolonun **sessizce NULL kalması** sınıfı kapandı. |
| **L4** | Katalog testi çalışma dizinine bağımlıydı | Yol `Path(__file__).resolve().parents[2]` ile depo köküne sabitlendi; alt dizinden koşum doğrulandı. |

**Nitler:** N1–N8 düzeltme beklemiyordu ve bilinçli olarak dokunulmadı. N1 (`_ADDED_COLUMNS`
içindeki `honest_miss` no-op ALTER) artık L3 testinin `⊆ DDL` iddiasıyla en azından
**hizada tutuluyor**.

### Doğrulama (fix turu)

```
uv run pytest -q -m "not slow"        -> 443 passed, 6 deselected   (fix öncesi 429)
make lint                             -> All checks passed · 0 errors
BG_DEVICE=mps uv run pytest -m slow -q -> 5 passed, 1 xfailed        (XPASS yok)
pytest tests/ alt dizininden          -> katalog testi geçiyor (L4)
```

Canlı (:7860, runtime Python değiştiği için yeniden başlatıldı):

- **Çip `/ask`** (c302, harç tarifesi) → `status=answered · honest_miss=False ·
  citations=['k492:69']`, `[S1]` = servis edilen top-1 (25.61) ve yanıt gerçekten o
  tarifeden ("150,00 YTL / 6.754,60 TL"). İnterleaved bağlama sağlam.
- **`/stats` ret hijyeni** (M3, ölçülmüş):

  ```
  ask öncesi   requests=2899  avg_ms=320.6  p95_ms=273.6
  ask sonrası  requests=2900  avg_ms=321.0  p95_ms=274.6
  422 sonrası  requests=2901  avg_ms=321.0  p95_ms=274.6   <- DEĞİŞMEDİ
  ```

  Ret satırı (id 2901, `total_ms=0.0192 ms`) `requests`'e girdi ama gecikme
  istatistiklerine **girmedi** — istenen davranış tam olarak bu.

- **L1 canlı geçişi** aynı tabloda görünüyor: incelemenin gösterdiği fix-öncesi degraded
  satırı `honest_miss=0` iken, fix sonrası degraded satırı (id 2900, `error_type=http_429`)
  **NULL**. `SELECT status, honest_miss, COUNT(*)` kırılımı: `degraded|NULL|1` (yeni) ve
  `degraded|0|1` (eski).

- **Y15/Y20 zinciri yine üretimde doğrulandı:** id 2900 `degraded · http_429 ·
  total_ms=1673 ms`, **tek çağrı** — 429 retry edilmiyor, bütçe yolu tetiklenmiyor.
  (İnceleme sırasında tükenmiş olan Gemini kotası sondaj sırasında toparlandı.)

Sunucu **çalışır bırakıldı**.
