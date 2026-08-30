# Vitrin sprinti — bağımsız inceleme (commit `6d5b345`)

**Tarih:** 2026-08-30 · **BASE:** `a9b1eaa` · **İnceleme kapsamı:** `6d5b345` (docs commit
`fc351a6` yalnız bağlam olarak okundu) · **Kip:** salt-okunur (pytest / ruff / curl)

**VERDİKT: APPROVE** — spec'in A/B/C maddelerinin tamamı karşılanmış, iddialar ölçümle
örtüşüyor, hiçbir doğruluk ya da dürüstlük kusuru yok. Bulunan 2 MEDIUM'un biri (M1) bu
commit'in kendi Dockerfile'ının AÇTIĞI bir kod yolunda sınırsız bellek büyümesi — canlı
demo yayına alınmadan ÖNCE kapatılmalı; ikisi de birkaç satırlık düzeltme.

---

## 1. Doğrulama (bu incelemede koşuldu)

| # | Komut / kontrol | Sonuç |
|---|---|---|
| V1 | `uv run pytest tests/app/test_api.py tests/retrieval/test_text.py -q` | **64 passed** in 1.08s |
| V2 | `uv run pytest -q -m "not slow"` | **327 passed**, 6 deselected in 2.86s |
| V3 | `uv run ruff check .` + `ruff format --check .` | All checks passed · 100 files formatted |
| V4 | canlı `/ask` chip 6 (aksansız): `"yillik ucretli izin suresi ne kadar?"` | `status="answered"`, gold **`k4857:28` sıra 2** (16.76), `visual_score=0.6997`, atıf `k4857:28`, üst düzey anahtarlar `['status','answer','hits']` |
| V5 | canlı `/ask {"question":"bu ne için"}` | **422** `{"detail":"sorgu boş ya da yalnız işlev kelimeleri içeriyor"}` (DÜZ DİZE) |
| V6 | canlı `/search {"k":100000}` | **422**, `detail` LİSTE (`less_than_equal`, `le:50`) |
| V7 | `RateLimiter` doğrudan sondalandı (pencere / bellek / kapalı yol) | pencere budama **doğru**; bellek **sınırsız** (bkz. M1); kapalıyken **hiç durum tutmuyor** |
| V8 | Port karşılaştırması `research/retrieve.py:38-50` ↔ `retrieval/text.py:99-139` | **karakter-karakter aynı** |

Kontrolcünün raporladığı sayılar (327 / lint temiz / canlı aksansız sıra 2 / k=100000→422 /
boş→422) bağımsız olarak yeniden üretildi.

---

## 2. Spec kontrol listesi

### A — ascii-fold portu (yazım-değişmezlik)

| # | Madde | Durum |
|---|---|---|
| A1 | `_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")` birebir | ✅ `text.py:103` = `retrieve.py:38` |
| A2 | `ascii_fold` birebir | ✅ `text.py:106-113` |
| A3 | `_STOP_FOLDED = frozenset(ascii_fold(w) for w in STOPWORDS)` | ✅ `text.py:119` |
| A4 | `tokenize` SIRASI: tr_lower → len>1 → fold → katlanmış-uzayda stopword → F5 | ✅ `text.py:138-139`, iki satır da `retrieve.py:49-50` ile aynı |
| A5 | `_GENERIC` AKSANLI kalmış, reçeteyle aynı, gerekçesi yazılmış | ✅ her iki dosyada `{"kanun","türk","türki","cumhu"}`; `text.py:77-84` sonucu (daha DAR yönlendirme) açıkça anlatıyor, `tests/retrieval/test_text.py` docstring'i tekrar ediyor |
| A6 | Eşik bandı fold SONRASI yeniden ölçüldü, 10.6 değişmedi | ✅ `config.py` yorumu `(10.5265, 10.7115]`, 42/43 + 4/5; xfail reason da güncellenmiş |
| A7 | Diğer sabitler (STOPWORDS, F5=5, WINDOW=50, k1/b, `_TITLE_LINE`) korunmuş | ✅ |

**Sıra sözleşmesinin ince yeri de doğru taşınmış:** `len(t) > 1` filtresi katlama ÖNCESİNDEKİ
token üzerinde çalışıyor (katlama uzunluğu değiştirmediği için sonuç aynı, ama port yine de
reçetenin ifade sırasını bozmamış). Stopword elemesinin F5'ten önce ama katlamadan sonra
olması — yani "göre"/"gore" ikisinin de düşmesi, "görev"in kalması — hem kodda hem testte
(`test_tokenize_folds_function_words_too`) kilitli.

**`_GENERIC` notu doğrulandı ve dürüst.** `research/retrieve.py:92` de aksanlı. Katlama
sonrası `tokenize("TÜRK MEDENİ KANUNU")` = `["turk","meden","kanun"]`, `_GENERIC` yalnız
`"kanun"`u eliyor → ad kümesi `{"turk","meden"}`. Yani "Türk …" ile başlayan kanunlar artık
sorguda "türk/turk" da geçmeden yönlendirilmiyor. Bu **daraltıcı** bir davranıştır (asla
genişletici değil), 0.8605 tam olarak bununla ölçülmüş, ve "düzeltmek" ölçülmemiş bir
varyant üretirdi. Portu bozmama kararı **doğru**; belgeleme yeterli.

### B — API sertleştirme

| # | Madde | Durum |
|---|---|---|
| B1 | `k: Field(None, ge=1, le=50)` | ✅ `main.py:70`; şema düzeyinde (uç nokta gövdesine HİÇ ulaşmıyor); V6 canlı |
| B2 | `query`/`question` `max_length=500` | ✅ `main.py:69,74`; sınır kapsayıcı (500 geçer, 501 düşer — test var) |
| B3 | Boş-içerik 422, İKİ uç noktada, ÜRETİM tokenleştiricisiyle | ✅ `main.py:77-85` doğrudan `retrieval.text.tokenize`'ı import ediyor — ikinci bir "boş mu?" sezgisi YOK; V5 canlı |
| B4 | `/ask` üst düzey `status` = telemetriye yazılan değerin AYNISI | ✅ `main.py:536-541` tek değişken, hem `record_event(status=…)` hem gövde |
| B5 | `visual_score` yalnız hibritte dolu, diğerlerinde `None` | ✅ `hybrid.py:214,235` + `types.py:27` (`= None` varsayılan) |
| B6 | Hız sınırı varsayılan KAPALI, IP başına, 429 + `Retry-After`, Docker 10/60 + `BG_LOG_QUERY_TEXT=false` | ✅ davranış doğru — **M1** (bellek) ve **L2/L4** (muhasebe/test) ile |
| B7 | `Semaphore(4)` YALNIZ encode çevresinde, üç getiricide de | ✅ kapsam doğru, kilitlenme yolu yok — **M2** (ölçüm kirlenmesi) ile |
| B8 | `query_format_id` → enum | ✅ `config.py`, test `test_invalid_query_format_id_fails_at_config_time` |
| B9 | `_CLI_DEFAULTS` ValidationError koruması | ✅ `cli.py`, alt-süreç testiyle kilitli (`Traceback not in stderr`, exit 2) |

### C — Arayüz

| # | Madde | Durum |
|---|---|---|
| C1 | 6 chip + dilim etiketi | ✅ `index.html` chips grid; `aria-label` gerçek soru, süs span'leri `aria-hidden` |
| C2 | Durum güdümlü haller (mühür yalnız `abstained`, ayrı `degraded` bandı, 422/429/ağ) | ✅ `index.html:718-731`, `pipelineFinish` üç dallı |
| C3 | Hit başına görsel mini-gösterge + `null` işleme | ✅ `hasScan` (520) + üç ayrı `typeof … === "number"` kontrolü (553, 581, 605) |
| C4 | "Nasıl çalışır" hibrit tazelemesi | ✅ üç sütun + künye + kırmızı `uyarı:` satırı |
| C5 | Meta / OG, uydurma URL yok | ✅ `og:type/title/description`; `og:url` ve `og:image` **yok** (doğru — henüz yayın yok) |
| C6 | a11y: focus, aria, role="alert", gerçek `<button>` | ✅ `:focus-visible` global; skor satırları `<div>`→`<button>`; `aria-invalid` + `aria-describedby` |
| C7 | 375 px duyarlılık | ✅ CSS incelendi: `@media (max-width:560px)` 4→3 kolon, `.name` tam satır, chips `auto-fit minmax(238px,1fr)` 343 px'te tek kolona düşüyor, `input{min-width:0}` taşmayı kapatıyor |
| C8 | Pipeline şeridi aşama dürüstlüğü | ✅ hibritte "sorgu kodlanıyor — görsel kanal", non-hybrid'de sadeleşiyor (474) — **INFO-1** ile |
| C9 | `prefers-reduced-motion` | ✅ 287-292 (`.stamp.inked`, `.stage.active .dot`, `.hit, .chip`) + JS `reduced` dalı |
| C10 | ABSTAIN_TEXT dize karşılaştırması kalıntısı yok | ✅ grep temiz, sunucu tarafı test bunu ASSERT ediyor |

### Öne çıkan doğrulamalar (yeniden litige edilmedi ama kontrol edildi)

**`status` dürüstlüğü — tam.** `degraded` `AskService`in `except` kolundaki
`annotate("degraded", True)`den (`answer/base.py`), `abstained` eşik kolundan geliyor.
Gemini yanıtlayıcısı **hiçbir yerde** `Answer(abstained=True)` üretmiyor (`gemini.py:84`),
yani `status=="abstained"` ⟺ LLM gerçekten çağrılmadı ⟺ arayüzün "Gemini çağrılmadı —
eşik altı" cümlesi **doğru**. Dürüst-ıska `answered` altında kalıyor (`honest_miss`
`main.py:398-400` yalnız `not abstained` iken hesaplanıyor). Mühür tek koşula bağlı:
`status !== "abstained"`.

**`visual_score` sızıntısı yok.** Eşik `hits[0].score` ile karşılaştırılıyor
(`answer/base.py:47`); `rank()`/`route_window()` görsel diziye hiç bakmıyor; olay
`detail.hits`'e yalnız `page_id`+`score` yazıyor (`main.py:438`) — telemetri şeması
DEĞİŞMEDİ. Ölçek etiketi ("normalize [-1,1]") `config.VISUAL_SCALE` ile tutarlı.

**XSS yüzeyi temiz.** Tüm `innerHTML` yazımları ya sabit, ya sayı, ya `esc()`'ten geçiyor;
`mdLite` ÖNCE `esc` sonra iki markdown kuralını uyguluyor; bozulma metni ve tüm hata
cümleleri `textContent`; sorgu `encodeURIComponent` ile yazılıp `.value` ile okunuyor
(`urlQ.slice(0, MAX_CHARS)`). Ham JSON hiçbir yolda basılmıyor, `err.message` de kaldırılmış.

**Semafor kapsamı doğru.** `with stage(...), ENCODE_LIMIT:` yalnız `encode_query` çağrısını
sarıyor; skorlama/BM25 dışarıda. `threading.Semaphore` bağlam yöneticisi istisnada da
salıyor. İç içe alım yok (`AskService` → `search` tek seviye) → kilitlenme yolu yok. Üç
getiricide de var (`core.py` ×2, `hybrid.py`).

**Dokümantasyon doğruluğu.** README'nin 0.8605 (37/43 ikili) / 0.8488 (kesirli) / MRR
0.632 / R@20 0.9302 / exp11 0.5814 sayılarının **hepsi** `research/journal.md` #11-#13 ile
birebir örtüşüyor. Kalan eski sayılar (0.8372, 0.655, 36/43) yalnız *karşılaştırma* satırı
ya da round-2 tarihçesi olarak duruyor — bayat bir sevkiyat iddiası yok. Dockerfile'ın üç
env'i de gerçek `Settings` alanlarına karşılık geliyor (`env_prefix="BG_"`).

---

## 3. Bulgular (şiddet sıralı)

### MEDIUM-1 — `RateLimiter` sözlüğü sınırsız büyüyor: her ayrı istemci IP'si KALICI bir girdi

**Yer:** `src/belge_gozu/app/main.py:106` (`self._hits = defaultdict(deque)`), `:109-121` (`check`)

Pencere içindeki zaman damgaları budanıyor (`:116-117`) ama **anahtarın kendisi asla
silinmiyor**. Deque boşaldığında bile `self._hits[client]` girdisi sonsuza kadar kalıyor;
hiçbir süpürme, TTL ya da tavan yok.

Ölçüldü (bu incelemede, `RateLimiter(5)` üzerinde doğrudan):

```
50.000 ayrı IP -> 50.000 kalıcı sözlük girdisi (~40 MB)
kapalı (per_min=0) 1.000 çağrı -> 0 girdi        # varsayılan-kapalı GERÇEKTEN bedava
```

**Kırılma senaryosu:** Bu commit'in Dockerfile'ı sınırı **herkese açık dağıtım için
AÇIYOR** (`BG_RATE_LIMIT_ASK_PER_MIN=10`, `SEARCH=60`). Ters vekilsiz açılan bir demoda her
farklı kaynak IP kalıcı bellek maliyeti yaratır; IPv6'da ayrı kaynak adres üretmek bedavaya
yakındır. 476 MB'lık mmap indeksi tutan bir konteynerde bu yavaş bir OOM'dur — ve sınırın
var oluş amacı (kaynağı korumak) tam da bu yolda tersine döner. Anahtar saldırgan
kontrolündedir; bu, "sınırsız büyüyen sözlük" kalıbının klasik biçimi.

**Düzeltme (birkaç satır):** `check` içinde budamadan sonra deque boşsa anahtarı düşür
(`if not q: del self._hits[client]` — geçiş kolunda tekrar oluşacağı için `defaultdict`
davranışı bozulmaz), ya da N çağrıda bir pencereden eski anahtarları süpür, ya da sözlüğe
bir tavan koy (LRU). Varsayılan-kapalı yol zaten etkilenmiyor.

### MEDIUM-2 — `query_encode` aşama süresi artık semafor kuyruğunu da ölçüyor

**Yer:** `src/belge_gozu/retrieval/hybrid.py:203`, `src/belge_gozu/retrieval/core.py:57` ve `:132`
— hepsi `with stage("query_encode"), ENCODE_LIMIT:`

Bağlam yöneticileri soldan sağa girildiği için **zamanlayıcı semafordan ÖNCE başlıyor**.
Sonuç: `encode_ms` (SQLite `events`) ve Prometheus `bg_*` aşama histogramı, c>4'te
kuyruk-bekleme + hesap toplamını raporluyor; saf model süresini değil.

**Kırılma senaryosu:** README'nin "stage-by-stage latency — query encode" vaadi ile
"40 istek @ c=8, p50 1.34 s" ölçümü aynı seriyi kullanıyor. Yük altında bu seri şişer ve
operatör bunu "model yavaşladı" diye okur; oysa değişen tek şey kuyruk derinliğidir. Bu,
projenin en çok değer verdiği şeyi — ölçümün ne anlama geldiğini — bulanıklaştırıyor.

**Düzeltme:** sırayı çevir (`with ENCODE_LIMIT, stage("query_encode"):`) ya da beklemeyi
ayrı bir aşama olarak ölç (`encode_wait`) — ikincisi kuyruk derinliğini de görünür kılar.

### LOW-1 — Boş-sorgu kapısı, BM25 kullanmayan pipeline'lara da BM25 kurallarını uyguluyor

**Yer:** `src/belge_gozu/app/main.py:77-85`, çağrılar `:478` ve `:518`

`retrieval_pipeline=exhaustive|two-stage` kollarında sıralamayı görsel kanal kuruyor ve
`tokenize` hiç çalışmıyor; ama 422 kapısı yine de metin kanalının işlev-kelime/uzunluk
kurallarına göre reddediyor. Fonksiyonun kendi docstring'i gerekçe olarak "eleme kuralları
değişirse bu kapı da onunla değişsin" diyor — bu gerekçe yalnız hibrit kolda geçerli.
**Senaryo:** ablasyon koşumu HTTP üzerinden görsel kolda "bu ne için" sorarsa, görsel
kanalın skorlayabileceği bir sorgu 422 alır. Etkisi düşük (ablasyon kolları üretimde
kullanılmıyor), ama sözleşme metni gerçekten anlattığı şeyden daha geniş.

### LOW-2 — 422'lerin hız-sınırı muhasebesi tutarsız

**Yer:** `src/belge_gozu/app/main.py:477-478`, `:517-518`

`enforce_rate_limit` uç nokta gövdesinin İLK satırı, ama pydantic doğrulaması gövde hiç
çalışmadan koşuyor. Sonuç: aşırı uzun bir sorgu ya da `k=100000` **bedava**, buna karşılık
`"bu ne için"` bir kota jetonu **harcıyor**. Zararsız ama asimetrik; ucuz olan reddin pahalı
olandan daha çok maliyetlenmesi mantığın tersi.

### LOW-3 — 422 ve 429 telemetriye hiç düşmüyor

**Yer:** `main.py:85` ve `:126-133` — ikisi de `collecting()` / `record_event` bloğundan
ÖNCE `HTTPException` fırlatıyor.

Bu bilinçli (test `test_empty_query_does_not_reach_the_retriever` 0 olay assert ediyor) ve
getiricinin çağrılmaması doğru. Ama sonuç şu: **sınırlayıcının var olma sebebi olan trafik,
`/stats`'ta da `/metrics`'te de `events` tablosunda da görünmez.** Bir kötüye kullanım
dalgası tamamen sessiz geçer. En azından iki sayaç (`bg_rejected_total{reason="empty"|"rate_limit"}`)
bu boşluğu kapatır.

### LOW-4 — Sınırlayıcının pencere-süresi ve X-Forwarded-For kararı TEST EDİLMİYOR

**Yer:** `tests/app/test_api.py` (hız sınırı bloğu)

Var olan üç test 429'u, `Retry-After`ı, uç-nokta ayrımını ve varsayılan-kapalıyı kilitliyor.
Kilitlenmeyen iki şey:

1. **Pencere sona ermesi.** `while q and now - q[0] >= self.window_s: q.popleft()` satırının
   hiçbir testi yok. `>=` yerine `>`, ya da `time.monotonic` yerine duvar saati kullanan bir
   ileride-değişiklik, bir istemciyi **kalıcı olarak** kilitler ve testlerin hepsi yeşil
   kalır. (Bu incelemede elle sondalandı: `RateLimiter(2, window_s=0.5)` doğru davranıyor —
   ama bu davranışı kilitleyen bir şey repoda yok.) Ucuz düzeltme: `window_s=0.05` ile bir test.
2. **X-Forwarded-For'a güvenmeme kararı.** README bunu açık bir güvenlik özelliği olarak
   ilan ediyor ("deliberately does not trust X-Forwarded-For"); kod yorumunda da gerekçesi
   var. Ama bir gün "vekil-farkındalığı ekleyelim" diyen bir yama hiçbir testi kırmaz.
   Tek satırlık bir test (`headers={"X-Forwarded-For": "1.2.3.4"}` ile 429 hâlâ gelmeli)
   iddiayı kilitler.

### LOW-5 — Vitrin sayılarının repoda karşılığı olan bir bench artefaktı yok

**Yer:** `README.md` ölçüm tablosu ve "Which Recall@5?" notu

Önceki sürüm `data/bench/results/20260829-2115-3a031ca-hybrid.json`'a atıf veriyordu; bu
satır kaldırılmış — **doğru bir düzeltme**, çünkü o dosya repoda yok (`data/bench/results/`
listelendi: yalnız 27-29 Ağustos artefaktları var). Ama sonuç olarak sevkiyat sayıları
(0.8605 / 0.8488 / MRR 0.632) için repoda hiçbir rapor dosyası kalmadı; post-fold bench
bilinçli olarak scratchpad'e yazıldı. Metin kanalı deterministik olduğu için
`uv run belge-gozu bench run --only-verified` bunları yeniden üretir — yani **doğrulanabilir**,
ama okuyucunun koşması gerekiyor. "Her sayının bir kaydı var" duruşuna sahip bir vitrin
için ya post-fold raporun `data/bench/results/`'a eklenmesi (mevcut dosyalarla aynı
büyüklük mertebesinde) ya da tablonun yanına tam yeniden-üretim komutunun yazılması
tutarlı olurdu.

### NIT-1 — `degraded` yanıt kartı `class="abstained"` taşıyor

`index.html:718` — `status === "answered" ? "" : " abstained"`, yani bozulma durumu da
abstain sınıfını alıyor. Tek etkisi `#answer-text`in soluk/dar stili ve o eleman bozulmada
zaten gizli — görsel sonuç yok. Ama sınıf adı artık durumu yanlış söylüyor; `.degraded`
ayrı bir sınıf olsa hem CSS hem okuma dürüst kalırdı.

### NIT-2 — Bozulmada eski yanıt metni DOM'da kalıyor

`index.html:721-727` — bozulma kolunda `#answer-text` gizleniyor ama içeriği
temizlenmiyor, dolayısıyla bir önceki sorunun yanıtı orada duruyor. Bugün görünmüyor;
`hidden` üzerinde tek bir regresyon, "servis notu" bandının yanında **başka bir sorunun**
yanıtını gösterir. `$("answer-text").innerHTML = ""` bir satır.

### NIT-3 — `friendly422` pydantic-şekilli her hatayı "uzunluk" sanıyor

`index.html:659-666` — `detail` dize değilse cümle her koşulda "Soru en fazla 500 karakter
olabilir." Bugün `/ask`ta `question`ın tek kısıtı `max_length` olduğu için **doğru**; ikinci
bir kısıt (ör. `min_length`, tip değişimi) eklendiği gün arayüz sessizce yanlış cümleyi
kurar ve hiçbir test kırılmaz. `payload.detail[0].type` üzerinden dallanmak ya da en azından
bilinmeyen tipe genel bir cümle vermek dayanıklı olurdu.

### INFO-1 — Pipeline şeridinin temposu istemci tarafı bir tahmin

`index.html:489` — `const pacing = [0, 1150, 1260, 1340];`. Aşama ilerlemesi sunucu
olaylarıyla değil sabit zamanlayıcılarla sürülüyor; yalnız SIRA ve nihai geçen süre gerçek.
Kod yorumu bunu dürüstçe söylüyor ("SIRA gerçeği, tempo ölçüme yakın bir tahmindir") ama
kullanıcı tarafında bir ayrım yok. Bu davranış bu commit'te GELMEDİ (yalnız tempo
[0,900,2100,2600] → [0,1150,1260,1340] ile ölçüme yaklaştırıldı, ki bu bir iyileşme) —
kayıt için not, düzeltme talebi değil.

---

## 4. Kalite hükmü

Bu commit, bir portföy projesinde nadiren görülen bir şeyi yapıyor: **ölçümü koda taşırken
ölçümün geçerliliğini bozmamayı, "daha temiz görünen" bir varyanta kaymaya tercih ediyor.**
`_GENERIC`'in aksanlı bırakılması, gerekçesinin koda ve iki teste yazılması, ve
"değiştirmek isteyen önce bench'i koşsun" uyarısı — bu, reçete portlarının en sık kaybettiği
disiplin.

Aynı şekilde: `status` alanının telemetriyle **aynı değişken** olması (kopya değil), iki
skorun **ayrı kolonlarda** durması (ve UI yorumunun bunu neden yaptığını söylemesi),
X-Forwarded-For bedelinin README'ye **kendi aleyhine** yazılması, ve slow testin kapsamının
apostrof bulgusuyla **daraltılması** (genişletilmesi değil) — hepsi iddia disiplininin
işaretleri.

Kod kalitesi tarafında iki gerçek boşluk var ve ikisi de aynı sınıftan: **savunma
mekanizmalarının kendileri ölçülmemiş/sınırsız.** M1'de sınırlayıcı korumaya çalıştığı
kaynağı kendi tüketiyor; M2'de sınırlayıcı, ölçtüğünü sandığımız şeyi kirletiyor; L3/L4'te
sınırlayıcının çalıştığına dair ne telemetri ne test var. Bu üçü birlikte kapatılırsa
sertleştirme katmanı da sistemin geri kalanıyla aynı standarda gelir.

**APPROVE.** M1 canlı demo yayına alınmadan önce, M2 bir sonraki telemetri dokunuşuyla,
L1-L5 backlog. Hiçbiri bu commit'i geri çevirmeyi gerektirmiyor.

---

## 5. Bulgu özeti

| # | Şiddet | Başlık | Yer |
|---|---|---|---|
| M1 | MEDIUM | `RateLimiter` sözlüğü sınırsız büyüyor (IP başına kalıcı girdi) | `app/main.py:106,109-121` |
| M2 | MEDIUM | `query_encode` aşama süresi semafor kuyruğunu da içeriyor | `retrieval/hybrid.py:203`, `retrieval/core.py:57,132` |
| L1 | LOW | Boş-sorgu kapısı görsel-yalnız pipeline'lara metin kuralı uyguluyor | `app/main.py:77-85,478,518` |
| L2 | LOW | 422 muhasebesi tutarsız (pydantic bedava, tokenize kapısı jeton harcıyor) | `app/main.py:477-478,517-518` |
| L3 | LOW | 422/429 telemetriye hiç düşmüyor (`/stats`, `/metrics`, `events` kör) | `app/main.py:85,126-133` |
| L4 | LOW | Pencere sona ermesi ve X-Forwarded-For kararı test edilmiyor | `tests/app/test_api.py` |
| L5 | LOW | Sevkiyat sayılarının repoda bench artefaktı yok | `README.md`, `data/bench/results/` |
| N1 | NIT | `degraded` kartı `class="abstained"` taşıyor | `app/static/index.html:718` |
| N2 | NIT | Bozulmada eski yanıt metni DOM'da kalıyor | `app/static/index.html:721-727` |
| N3 | NIT | `friendly422` pydantic-şekilli her hatayı "uzunluk" sanıyor | `app/static/index.html:659-666` |
| I1 | INFO | Pipeline temposu istemci tarafı tahmin (bu commit'te gelmedi) | `app/static/index.html:489` |

**CRITICAL: 0 · HIGH: 0 · MEDIUM: 2 · LOW: 5 · NIT: 3 · INFO: 1**
