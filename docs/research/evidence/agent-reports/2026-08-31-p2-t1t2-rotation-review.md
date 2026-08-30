# Review — P2 T1+T2 (iddia doğrulayıcı + iki kapı) ve Gemini anahtar rotasyonu

- **Tarih:** 2026-08-31
- **Aralık:** `af6cc46..348fb63` — `8d4627d` (bağlam), **`d051918`** (T1+T2), **`348fb63`** (rotasyon)
- **Kaynaklar:** `review-t1t2-rotation.diff` (172 KB, tamamı), `p2-t1t2-report.md`,
  `key-rotation-report.md`, plan T1-T2 (`docs/superpowers/plans/2026-08-26-belge-gozu-p2-selective-answering.md:57-176`)
- **Koşulan doğrulamalar:** `uv run pytest -q -m "not slow"` → **650 passed, 6 deselected**;
  `make lint` → ruff check + ruff format (111 dosya) + pyright **0 hata**;
  4 adet hedefli davranış sondası (kod çalıştırılarak, ağsız); **1 canlı `/ask`** (`:7860`)
  + `/metrics` + sqlite okuması. Yazma yapılmadı.

---

## VERDİKT

**FIX REQUIRED — `BG_GATE_VERIFIER=true` AÇILMADAN ÖNCE.**
Bayrak-KAPALI davranış için **APPROVE**: bayraklar kapalıyken davranış gerçekten
P1 ile birebirdir ve bu commit'in üretimde bugün değiştirdiği tek şey rotasyondur
(o da doğru çalışıyor, canlıda doğrulandı). Aşağıdaki **H1/H2/H3**'ün üçü de
YALNIZ `gate_verifier` açıkken tetiklenir — yani commit'in bayrak-arkası inmesini
engellemezler, ama kapı 2'nin ilk gerçek koşumundan önce kapatılmaları gerekir.
Rotasyon commit'i (`348fb63`) tek başına **APPROVE**.

---

## Kontrol listesi 1 — T1+T2 spec uyumu

| # | Spec maddesi | Durum | Kanıt |
|---|---|---|---|
| 1 | Türkçe-farkında iddia bölümleme (`m.19`, `2.806,50`, `320.`) | ✅ | 27 karşı-örnekli sonda; `m./md./Av./vb./vs./no./s./Bkz./Dr./Prof./T.C./01.01.2020/19/1` hepsi tek iddia |
| 2 | `[Sn]` bağlama + paragraf devralma | ⚠️ | Çalışıyor; devralma **paragrafın TÜM kaynaklarını** verir → L4 |
| 3 | Yalnız-metin doğrulama, atıf yapılan sayfaların metnine karşı | ✅ | `verify.py:625-641`; canlı `page_texts.parquet` metni istemde (`test_gate2_verifier_reads_the_served_page_text`) |
| 4 | sha256 önbellek = model + istem sürümü + iddia + kanıt sha, **API anahtarı YOK** | ✅ | `verify.py:344-357`; `\x00` ayraçlı; imza testi `test_verifier_client_shares_the_rotating_pool_and_a_key_agnostic_cache` |
| 5 | İstem sürümü değişince anahtar değişir | ⚠️ | Elle tutulan sabit, istem metnine mekanik bağı YOK → **M3** |
| 6 | `VerifierBudget` sert tavan | ❌ | Yalnız CLI'de kuruluyor; **serve yolunda hiç yok** → **H1**; ayrıca çağrı≠deneme → **M1** |
| 7 | İki kapı da bayrak-KAPALI, kapalıyken BİREBİR uyum | ✅ | 650 test yeşil, `test_api.py` ellenmemiş, canlı gövde anahtarları `{status,honest_miss,answer,hits}`, olay `detail`inde kapı bloğu yok, `/metrics`te örnek yok |
| 8 | Kapalıyken tembel: kalibratör/`page_texts`/doğrulayıcı istemcisi kurulmaz | ✅ | `verify.py:700` erken dönüş; `test_build_gates_loads_nothing_when_both_flags_are_off` |
| 9 | Kapı 1 = `p < tau` çekimser, artefakt **fail-fast** | ✅ **başlangıçta** | `main.py:405` (`create_app`), istek başına DEĞİL; `test_missing_calibration_artifact_stops_startup`. Zamanlaması geç → L1 |
| 10 | Kapı 2 = tek desteklenmeyen iddia → `abstained` + `VERIFIER_DEMOTE_TEXT` + `detail.gate2` | ✅ | `base.py:209-231`; sonda: `status=abstained`, `citations=[]`, metin ABSTAIN_TEXT'ten ayrı |
| 11 | `belirsiz` = desteklenmemiş (şüphede-reddet) | ✅ | `verify.py:130` `SUPPORTED` karşılaştırması; parametrik test |
| 12 | Dürüst-ıska / çekimser / atıfsız → kapı 2 ATLANIR | ✅ | `gate2_skip_reason` (`verify.py:646-658`); 4 test + stub istemciye sıfır istem |
| 13 | CLI `verify run` + **zorunlu** `--max-llm-calls` | ✅ | `cli.py:1175` (`...`), negatif reddi, `test_verify_run_requires_an_explicit_llm_budget` |
| 14 | `status` sözlüğü genişlemedi | ✅ | Düşürme de `abstained`; `test_gate2_demote_keeps_the_status_vocabulary_and_flags_it_in_detail` |

## Kontrol listesi 2 — Rotasyon spec uyumu

| # | Spec maddesi | Durum | Kanıt |
|---|---|---|---|
| 1 | `google_api_key_2` Settings alanı (+ alias'lar) | ✅ | `config.py:70-84`; 3 test (`.env`, ortam, boş varsayılan) |
| 2 | Süreç düzeyinde **yapışkan** gösterge, kilitli | ✅ | `gemini.py:390-425`; canlı: 2. istek doğrudan key2, `answer_ms` 10179→5606 |
| 3 | **Kilit kapsamı:** API çağrısı kilidin İÇİNDE mi? | ✅ **DIŞINDA** | `current()`/`set()` (`gemini.py:412,417`) kilidi yalnız bir `int` okuma / bir `int` yazma için tutar; `_generate` çağrısı `_run` içinde (`gemini.py:483-522`) kilidin tamamen dışında. 30 sn'lik bir çağrı hiçbir isteği serileştirmez. İki ayrı `acquire` olduğu için iyi-huylu bir TOCTOU penceresi var (belgelenmiş + iki-iş-parçacıklı testle kilitli) |
| 4 | `parse` HARİÇ her API hatasında rotasyon | ✅ | `NON_ROTATABLE_ERROR_TYPES = {"parse"}` (`gemini.py:362`); `test_parse_error_does_not_rotate` |
| 5 | `safety_block` rotasyona girer (zararsız) | ✅ | `_generate` içinden `AnswererError("safety_block")` → `_may_rotate` geçirir |
| 6 | ≤1 rotasyon/istek, toplam ≤3 deneme | ✅ | 1. anahtar `max_attempts=1`, 2. anahtar varsayılan 2 → 3; `test_the_ladder_never_exceeds_three_attempts` |
| 7 | Hepsi TEK 35 sn bütçe altında, **HER** ek denemeden önce kontrol | ✅ | `started` paylaşılıyor; rotasyon öncesi `_may_rotate`→`budget_fits` (`gemini.py:534`), son-anahtar retry öncesi `budget_fits(started, backoff)` (`gemini.py:346`) |
| 8 | İki anahtar da düşerse → `degraded` + SON taksonomi + `keys_tried` | ✅ | `gemini.py:511-522`; uçtan uca test |
| 9 | `detail.llm.key` + `rotations[].error_type` | ✅ | Canlı sqlite: `detail.llm={'key':'key2'}`; `/search` satırları taşımıyor |
| 10 | `bg_llm_key_rotations_total{from_key}`, sınırlı etiket | ✅ | `prom.py:265-268`, `KEY_LABELS` süzgeci; bozuk-satır testi; canlı: `{from_key="key1"} 1.0` |
| 11 | Tek anahtarda birebir uyum | ⚠️ | Deneme/backoff/taksonomi birebir (test); **notlar birebir DEĞİL** (`detail.llm` yeni) → L7 |
| 12 | Anahtar DEĞERİ hiçbir yerde yok | ✅ | `.env`'deki iki değerin de depo+SDD dizini genelinde **0** eşleşmesi; `.env` gitignore'lu; kodda/logda/testte/raporda yalnız `"key1"/"key2"` |

---

## Bulgular (şiddet sırasıyla)

### H1 — Üretim (serve) yolunda `VerifierBudget` HİÇ YOK; kapı 2 açıkken sert tavan yok
`src/belge_gozu/app/main.py:405` — `build_gates(s, retriever, index_revision=index_revision)`
`budget` argümanını **hiç geçmez**, yani `verify.py:711-716`'daki `ClaimVerifier`
`budget=None` ile kurulur. `VerifierBudget(...)` deposu genelinde **yalnız**
`cli.py:1214` ve bir testte kuruluyor. Üstüne `Gates.budget` (`verify.py:680`) alanı
**hiçbir yerde okunmuyor** (M4) — yani bütçenin `build_gates` üzerinden aktığı izlenimi
API'de var, gerçekte yok.

**Arıza senaryosu (ölçülmüş sayılarla):** `BG_GATE_VERIFIER=true` ile `serve`;
`verifier_max_claims=8` (varsayılan). Rapor §5 ölçtü: *tipik bir yanıt 6-7 iddiaya
bölünüyor* ve *ücretsiz kota = 20 çağrı/gün/anahtar* (API'nin kendi mesajıyla
doğrulanmış). Rotasyon merdiveni her doğrulayıcı çağrısını **3 API denemesine kadar**
çarpabilir (M1). Yani **tek bir `/ask`**: 8 iddia × 3 = 24 doğrulayıcı denemesi + 3
yanıtlayıcı denemesi = **27** — iki anahtarın toplam günlük kotasının (40) yarısından
fazlası, tek istekte. `rate_limit_ask_per_min` varsayılanı `0` (kapalı) olduğu için
süreç içinde bunu durduracak başka hiçbir fren yok.

**Öneri:** `create_app` içinde `Settings`ten okunan bir süreç-ömürlü bütçe kur
(`verifier_max_llm_calls_per_process` ya da istek başına bir tavan) ve `build_gates`e
geçir; ya da `Gates.budget` alanını kaldırıp bütçeyi zorunlu argüman yap ki "serve
bütçesiz" hâli tip düzeyinde imkânsız olsun.

---

### H2 — Önbellek zehirlenmesi: ayrıştırılamayan/boş 200 yanıtı KALICI `belirsiz` olarak yazılıyor
`src/belge_gozu/answer/verify.py:522-536` — `parse_verdict` "ayrıştırılamadı" dalına
düştüğünde de (`verify.py:311-312`, `("belirsiz", "model çıktısı ayrıştırılamadı")`)
sonuç **koşulsuz** `cache.put` ediliyor. Önbellekte TTL yok, geçersizleştirme yok;
tek çıkış yolu `PROMPT_VERSION` bumpı (ve o da elle — M3).

**Çalıştırılarak doğrulandı:**
```
1. cagri: verdict='belirsiz' llm_called=True  -> onbellege YAZILDI (verdict='belirsiz')
2. cagri: verdict='belirsiz' cached=True      (istemci cagri sayisi = 1)
>>> KALICI ZEHIRLENME: True
```

**Arıza senaryosu:** Gemini `finish_reason=MAX_TOKENS` ile **boş gövde** döndürür.
`_generate` bunu `safety_block` saymaz (`gemini.py:365`: blok sebebi yok) ve
`GenResult(text="")` döner → `parse_verdict("")` → `belirsiz` → **önbelleğe yazılır**.
O andan sonra aynı (model, istem sürümü, iddia, kanıt) üçlüsü **sonsuza kadar**
`belirsiz` döner, sıfır çağrıyla, ve o iddiayı içeren her yanıt kapı 2'de düşer.
Aynı yol geçici bir gövde bozulması, kısmi akış, ya da SDK'nın yeni bir sarmalama
biçimi için de geçerli. Bu, önbelleğin var olma amacını (deterministik yargıyı
kalıcılaştırmak) tam tersine çevirir: **geçici bir arızayı** kalıcılaştırır.

**Öneri:** `parse_verdict`'i `(verdict, gerekce, parsed: bool)` yap (ya da ayrıştırma
başarısızlığını ayrı bir sentinel ile bildir) ve `parsed=False` iken `cache.put`
ÇAĞIRMA. `verify.py:311`'deki dal zaten tam olarak bu durumu biliyor.

---

### H3 — Cümle-BAŞI kısa parça kendi iddiası oluyor: kota yakıyor VE doğru yanıtı düşürüyor
`src/belge_gozu/answer/verify.py:204-212` — `_merge_fragments` `MIN_CLAIM_CHARS`
altındaki parçayı **yalnız ÖNCEKİ** cümleye ekler (`if out and ...`). Parça
**birinci** sırada ise ekleyecek önceki cümle yoktur ve parça kendi başına bir iddia
olarak kalır. `segment_claims`'in tek filtresi "alfanümerik içeriyor mu" olduğu için
(`verify.py:238`) süzülmez.

**Çalıştırılarak doğrulandı:**
```
'Evet. Yillik ucretli izin suresi on dort gundur [S1].'
  -> [('c1', 'Evet.'), ('c2', 'Yillik ucretli izin suresi on dort gundur.')]
'Yarg. HGK karari ...'  -> [('c1', 'Yarg.'), ('c2', 'HGK karari ...')]
```
`EvidenceGate` ile:
```
n_claims=2  n_supported=1  llm_calls=2  demoted=True
kararlar: [('c1','belirsiz'), ('c2','supported')]
>>> ICERIGI DOGRU olan yanit 'Evet.' yuzunden dusuruldu
```

**Arıza senaryosu:** Türkçe soru-cevapta "Evet." / "Hayır." açılışı olağandır;
`SYSTEM` istemi (`gemini.py`) bunu yasaklamaz. Böyle bir yanıtta (a) anlamsız
parçaya **bir kota çağrısı** harcanır (20/gün'lük bütçede ~%5), (b) yargıç izole
"Evet."i kanıtta bulamaz → `belirsiz` → **şüphede-reddet ilkesi doğru çalışır ama
yanlış girdi üzerinde**: içeriği tamamen desteklenen yanıt düşürülür. `_ABBREVS`
listesinde olmayan bir kısaltmayla başlayan cümle (`Any.`, `Yarg.`) aynı sonucu verir
— ve kısaltma listesi doğası gereği eksik kalacaktır.

**Öneri:** `_merge_fragments`'ı çift yönlü yap — baştaki kısa parça bir SONRAKİ
cümleye eklensin (`out` boşken parçayı tampona al, ilk normal cümlenin başına ekle).
Tek satırlık düzeltme, mevcut geriye-birleştirme davranışını bozmaz.

---

### M1 — `VerifierBudget` "doğrulayıcı çağrısı" sayıyor, "API denemesi" değil; rotasyon 3× çarpıyor
`verify.py:499-501` — `budget.consume()` `client.generate_json` **öncesinde bir kez**
çağrılır; retry ve anahtar rotasyonu istemcinin İÇİNDE olduğu için bütçe onları görmez.

**Çalıştırılarak doğrulandı** (key1 429 → key2 servis etti):
```
budget.used=1   GERCEK API denemesi=2 (key1=1, key2=1)
```
Merdivenin tamamı kullanıldığında oran 1:3'e çıkar.

Görev tanımının duruşu ("bir doğrulayıcı ÇAĞRISI, iki anahtar denemesi olsa da tek
sayılmalı") **kod tarafından karşılanıyor** — sorun muhasebede değil, **isimde ve
sözleşmede**: `--max-llm-calls` bayrağının gerekçesi açıkça kota koruması
(`cli.py:1772-1774`) ve kota **denemeyle** sayılır. `--max-llm-calls 20`, ölçülmüş
20/gün kotasına karşı 60 denemeye kadar izin verebilir. Aynı sapma raporlamaya da
geçiyor: `summary.verifier_llm_calls` ve `detail.gate2.llm_calls` gerçek kota
yükünü 3'e kadar **eksik** gösterir. En az bir isim/belge düzeltmesi, tercihen
denemeleri sayan ikinci bir sayaç (`RotatingGeminiClient` zaten `rotations`ı biliyor).

### M2 — `bg_abstain_total{reason}` üç ayrı sebebi tek etikette topluyor
`src/belge_gozu/telemetry/prom.py:220-223` — `reason` yalnız `{degraded, threshold}`.
Kapı 1 çekimseri de, kapı 2 düşürmesi de `abstained=True` olduğu için
`reason="threshold"` sayılır.

**Çalıştırılarak doğrulandı:** üç ayrı olay (eşik altı / `gate1.passed=False` /
`gate2.demoted=True`) → `bg_abstain_total{reason="threshold"} 3.0`.

Bu, `base.py:28-38`'de `VERIFIER_DEMOTE_TEXT` için yazılan ilkenin ("ikisi farklı şey
söyler ve ikisi de doğru olmak zorunda") Prometheus katmanında bozulmasıdır. Somut
bedel: kapı 1 açıldığında dev kapsaması **%2.2** ölçüldü (`p2-t1t2-report.md §4`) —
yani bu seri patlayacak ve operatör artışı eşiğe mi, kalibratöre mi, doğrulayıcıya mı
atfedeceğini metrikten **çıkaramayacak**. Olay satırında bilgi var (`detail.gate1/gate2`),
metrikte yok. Öneri: `reason` kümesini `{threshold, gate_calibrated, gate_verifier,
degraded}` yap; olay `detail`i zaten ayrımı taşıyor.

### M3 — `PROMPT_VERSION` elle tutulan bir sabit; istem metnine mekanik bağı yok
`verify.py:255-259`. Yorum doğru uyarıyı yapıyor ("İstem metni ya da şema değişirse BU
DİZE DE değişmek zorundadır") ama bunu **hiçbir şey zorlamıyor**: `VERIFIER_PROMPT` ya
da `VERDICT_SCHEMA` düzenlenip sürüm unutulursa eski istemin kararları yeni istemin
kararıymış gibi sessizce yeniden kullanılır — H2'nin yanında bu, önbelleğin ikinci
sessiz yanlışlık kaynağıdır. Depo bu sınıf problem için **zaten daha güçlü bir
desene sahip**: `retrieval/text.py:205` `recipe_fingerprint()` sabitlerden sha256
hesaplar ve tam olarak "reçete değişince sessizce yanlış kalmasın" diye yazılmış.
Öneri: `PROMPT_VERSION = "verify-v1-" + sha256(VERIFIER_PROMPT + json.dumps(VERDICT_SCHEMA, sort_keys=True))[:8]`,
ya da en azından istem hash'ini sabitleyen bir test.

### M4 — `Gates.budget` ölü alan
`verify.py:680` tanımlı, `build_gates` dolduruyor (`cli.py:1157` üzerinden),
**hiçbir yerde okunmuyor** (`grep` ile doğrulandı). `cli.py::verify_run` kendi yerel
`budget` değişkenini kullanır. H1'i maskeliyor: API'ye bakan biri bütçenin kapılara
aktığını sanır.

---

### L1 — Kalibrasyon fail-fast'i ağır yüklemeden SONRA
`main.py:363` `require_text_artifact` bilinçle **erken** çağrılıyor (review L6:
"tek satırlık mesaj için dakikalarca model yüklemek anlamsız"). Aynı gerekçe
kalibrasyon artefaktı için uygulanmamış: `build_gates` `main.py:405`, yani
`ColSmolEncoder` (`:369`) + 474 MB indeks (`:380`) yüklendikten sonra patlıyor.
`BG_GATE_CALIBRATED=true` + eksik artefakt = dakikalarca bekleyip hata.

### L2 — `VerifierCache.put` atomik değil, depo deseniyle tutarsız
`verify.py:429-440` düz `write_text` kullanıyor. Aynı önbellek dizinini `serve` ve
`verify run` **eşzamanlı** paylaşabilir; okuyucu yarım yazılmış dosya görebilir.
Sonuç iyi huylu (`get` `JSONDecodeError`ı yutuyor → fazladan bir çağrı, bozulma yok),
ama depoda zaten `corpus/download.py:70`'te `os.replace` deseni var. Tutarlılık için
tmp+`os.replace`.

### L3 — Bütçe yanıtın ORTASINDA biterse `llm_calls` raporu ile `budget.used` çelişir
`verify.py:610` liste kavramasının ortasında `VerifierBudgetExceeded` fırlarsa
`EvidenceGate.evaluate` hiç dönmez; `base.py:219-222` `{"demoted": True, "error":
"VerifierBudgetExceeded"}` yazar — `claims` ve `llm_calls` **yok**. Sonda:
```
gate2 notu = {'demoted': True, 'error': 'VerifierBudgetExceeded'}
GERCEK istemci cagrisi=1  budget.used=1   detail'de llm_calls: False
```
`cli.py:1850` `summary.verifier_llm_calls`'ı `g2.get("llm_calls", 0)` üzerinden
topladığı için o soruda harcanan çağrılar rapordan düşer, `budget.used` ise sayar —
aynı JSON'daki iki sayı birbirini yalanlar.

### L4 — Devralınan atıflar paragrafın TÜMÜNÜ alıyor: en zayıf iddiaya en geniş kanıt
`verify.py:245` `cited_sources=own or para_sources`. Sonda:
```
c1: src=[1, 2, 3] inherited=True  :: 'Bu genel bir degerlendirme cumlesidir...'
c2: src=[1] ... c3: src=[2] ... c4: src=[3]
```
Kendi atfı olmayan (yani en zayıf gerekçeli) cümle, üç sayfanın **birleşimine** karşı
yargılanır: hem "supported" çıkması en kolay iddia olur (kapının yönüne ters), hem de
istemi 3× büyütüp `EVIDENCE_CHAR_LIMIT` (12 k) kırpmasına en yakın iddia olur.
Devralmayı 1-2 kaynakla sınırlamak ya da `inherited_sources=True` iddialarında daha
katı bir eşik uygulamak düşünülmeli. (Bugün en azından **görünür**: `detail.gate2.claims[].inherited_sources`.)

### L5 — Katman: telemetri artık `answer/`e bağımlı; `base.py` ↔ `verify.py` gerçek bir döngü
`telemetry/prom.py:14-15` modül düzeyinde `answer.gemini.KEY_LABELS` ve
`answer.verify.VERDICTS` ithal ediyor — kesişen bir altyapı modülü alan modülüne
bağlanıyor (`answer/gemini.py` ise `telemetry.collect`e bağlı). Bugün döngü yok ama
yön ters. Ayrıca `base.py:211` `gate2_skip_reason`ı **fonksiyon içinde** ithal ediyor;
bu, `base ↔ verify` döngüsünü kırmak için gereken bir kaçamak. `gate2_skip_reason`
yalnız `Answer`a dokunuyor — `base.py`ye taşınırsa döngü kaybolur ve istek başına
tekrarlanan `import` ifadesi de gider. Benzer şekilde `build_gates` **her iki kapının**
(kalibrasyon kapısı dahil) kompozisyon kökü olduğu hâlde kanıt-doğrulayıcı modülünün
içinde duruyor; nötr bir yer (`answer/gates.py`) daha doğru olurdu.

### L6 — `RotatingGeminiClient` korumalı üyelere uzanıyor
`gemini.py:490,500,510,516` `client._generate(...)`, `gemini.py:481`
`self._slots[0].client._json_config(...)`. Ayrıca içerik/yapılandırma kurulumu
yapışkan slottan değil **daima `_slots[0]`**'dan yapılıyor (`gemini.py:475,481`) —
bugün zararsız (ikisi de durumsuz) ama sözleşme yazılı değil. `_generate`/`_json_config`
sarmalayıcı katmanın gerçek arayüzü olduğu için ya adları alt çizgisiz olmalı ya da
`GeminiClient` üzerinde açık bir "iç API" bölümü olarak belgelenmeli.

### L7 — Rapor ifadesi: tek anahtarda "notlar BİREBİR eskisi" doğru değil
`key-rotation-report.md` §1/§6 ve `gemini.py:576-580` docstring'i tek anahtarlı
davranışın "deneme sayısı, backoff, taksonomi, **notlar**" bakımından birebir eski
olduğunu söylüyor. Deneme/backoff/taksonomi birebir; **notlar değil**: `_served`
(`gemini.py:545`) tek anahtarlı BAŞARILI çağrıda da `merge_note("llm", key="key1")`
yazar ve bu, `events.detail`e yeni bir `llm` bloğu olarak girer (canlıda görüldü).
Zararsız ve faydalı bir ekleme — ama testin kendisi bunu söylüyor
(`test_single_key_success_still_records_which_key_served`), rapor söylemiyor.

### L8 — `_GEREKCE_RE` kesme işaretinde gerekçeyi kırpıyor
`verify.py:295` `r"[\"']?gerekce[\"']?\s*[:=]\s*[\"']([^\"']*)[\"']"` — regex geri
düşüş yolunda `"Kanun'un 19. maddesi..."` gibi Türkçede çok yaygın bir gerekçe
`Kanun` diye kesilir. Yalnız JSON ayrıştırması başarısız olduğunda devreye girer, ve
`verdict` etkilenmez; kozmetik ama telemetriye yazılan bir alan.

### Nit
- `verify.py:503-504` `except VerifierBudgetExceeded: raise` erişilemez (bütçe
  `try`ın dışında tüketiliyor); `pragma: no cover` bunu kabul ediyor — silinebilir.
- `base.py:74-83` `RetrievalGate`/`EvidenceGateProtocol` protokolleri dekoratif:
  `main.py:413-414` ve `cli.py:1162-1163` `pyright: ignore` ile geçiyor, yani
  protokoller hiçbir çağrı yerinde tip denetimi yapmıyor.

---

## İyi çalışan, doğrulanmış taraflar

- **Bayrak-kapalı değişmezlik gerçek.** `tests/app/test_api.py` ellenmemiş; 650 test
  yeşil; canlı `/ask` gövde anahtarları tam olarak `{answer,hits,honest_miss,status}`;
  sqlite `detail`inde `gate*` yok; `/metrics`te `bg_verifier_verdicts_total` yalnız
  `# HELP` satırı, hiçbir örnek yok. Bayrak açıkken hiçbir şeyin kazara yüklenmediği
  de `build_gates` erken dönüşüyle ve testle kilitli.
- **Düşürme yolu spec'e uygun:** `status=abstained` (sözlük genişlemedi),
  `VERIFIER_DEMOTE_TEXT` ABSTAIN_TEXT'ten ayrı, `citations=[]`, sayfalar korunuyor,
  ayrım `detail.gate2.demoted`de. `belirsiz` desteklenmemiş sayılıyor. Kapı 2 arızası
  da (kota/disk) düşürüyor — şüphede-reddet istisna yolunda da tutarlı.
- **Atlamalar doğru ve ücretsiz:** dürüst-ıska / çekimser / degraded / atıfsız yollarda
  doğrulayıcıya sıfır istem gidiyor (testlerde `client.prompts == []`).
- **Rotasyon merdiveni doğru kurulmuş.** ≤3 deneme, tek bütçe, her ek denemeden önce
  `budget_fits`, `parse` istisnası, son anahtara bırakılmış aynı-anahtar retry'si —
  hepsi hem kodda hem testte tutarlı. Canlı: key1 → 429 → key2 servis etti (`id=2909`),
  sonraki istekler doğrudan key2 (`id=2910/2911/2912`, `detail.llm={'key':'key2'}`,
  rotasyon sayacı artmadı). Yapışkanlığın ölçülen kazancı ~4,6 sn/istek.
- **Kilit kapsamı doğru** (bkz. kontrol listesi 2 / #3): API çağrısı kilidin dışında.
- **Anahtar gizliliği kusursuz:** `.env`'deki iki değer de depo + SDD dizininde 0 kez
  geçiyor; `.env` gitignore'lu; kodda/logda/istisna mesajında/metrikte/testte yalnız
  `key1`/`key2` etiketleri.
- **Sayaç etiketleri kapalı kümede** ve bozuk olay satırına karşı test edilmiş
  (`test_key_rotations_counter_ignores_unknown_labels`).
- **Katalog satırları dürüst:** `metrics-catalog.md:76,80` iki seriyi de, bayrak-kapalı
  boşluk davranışını da, "anahtar değeri yazılmaz" güvencesini de doğru anlatıyor.
- **Önbellek anahtarı bileşimi doğru:** model + istem sürümü + iddia + kanıt **içerik**
  sha'sı, `\x00` ayraçlı, API anahtarı YOK — rotasyon bir isabeti geçersizleştirmiyor
  (imza testiyle kilitli).
- **Canlı-sondanın yakaladığı hata (`Verdict.llm_called`) gerçekten regresyon testli:**
  `test_llm_calls_counts_actual_api_calls_not_just_cache_misses`.
- **Bölümleme Türkçe-farkındalığı sağlam:** 27 karşı-örneklik sondada yalnızca
  `_ABBREVS` dışı kısaltmalar (H3) ve sayı ile biten cümle sınırı (bilinçli, güvenli
  yönde eksik bölme) sapıyor.

---

## Önerilen kapatma sırası

1. **H2** (tek satırlık koşul — ayrıştırılamayan yanıtı önbelleğe yazma)
2. **H3** (tek fonksiyon — `_merge_fragments` çift yönlü)
3. **H1 + M4** (serve'e bütçe bağla, `Gates.budget`ı ya kullan ya kaldır)
4. **M1** (bayrak/rapor adlandırması: "doğrulayıcı çağrısı" ≠ "API denemesi"; deneme sayacı)
5. **M3** (istem sürümünü istem metninden türet), **M2** (`abstain` reason ekseni)
6. L1-L8 fırsat buldukça; L5 bir sonraki dokunuşta modül taşımasıyla.
