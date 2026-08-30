# P2 T1+T2 — iddia doğrulayıcı + iki kapı (bayrak-kapalı)

- **Tarih:** 2026-08-30
- **Branch:** `feat/p0-retrieval-correctness`
- **Kapsam:** plan T1 (`answer/verify.py`) + T2 (iki kapı, `answer/base.py`), İKİSİ DE
  varsayılan-KAPALI bayrakların arkasında.
- **Testler:** 627 geçti (taban 557 → **+70 yeni test**), `make lint` (ruff + ruff format
  + pyright) temiz.
- **Canlı Gemini çağrısı:** **6** (bütçe ≤6). Ayrıntı §5.

---

## 1. Ne yapıldı

| Parça | Dosya | Özet |
|---|---|---|
| İddia bölümleme | `src/belge_gozu/answer/verify.py` | `segment_claims` — deterministik, Türkçe-farkında; `[Sn]` atıflarını iddiaya bağlar |
| Doğrulayıcı | `answer/verify.py` | `verify_claim` — iddiayı ATIF YAPTIĞI sayfaların METNİNE karşı yargılar; sha256 önbellek + bütçe + katı ayrıştırma |
| Kanıt kapısı (kapı 2) | `answer/verify.py::EvidenceGate` | desteklenmeyen tek iddia yanıtı DÜŞÜRÜR (ilke 20) |
| Kalibre getirim kapısı (kapı 1) | `answer/calibrate.py::CalibratedRetrievalGate` | artefaktın seçtiği `tau` ile `p < tau` → çekimser |
| İki kapının servise takılması | `answer/base.py::AskService` | `gate1`/`gate2` opsiyonel; **ikisi de None → P1 davranışı** |
| Kapı kurulumu (fail-fast) | `answer/verify.py::build_gates` | serve + CLI ortak yolu; artefakt yoksa `IndexCompatibilityError` + `calibrate fit` ipucu |
| Yapılandırılmış çıktı | `answer/gemini.py::GeminiClient.generate_json` | `response_schema` + `temperature=0`; `generate()` ile AYNI retry/bütçe invariantı |
| Tek istemci kurulum noktası | `answer/gemini.py::build_gemini_client` | yanıtlayıcı ve doğrulayıcı AYNI fonksiyondan geçer (anahtar-rotasyon katmanı tek noktadan sarmalayabilsin) |
| BM25 yeniden kullanımı | `retrieval/hybrid.py::last_bm25_scores` | kapı 1 istek başına İKİNCİ bir korpus taraması eklemez |
| Sayfa metni yüzeyi | `retrieval/hybrid.py::load_page_texts` | doğrulayıcı, BM25'in skorladığı AYNI `page_texts.parquet`'i okur |
| Bayraklar | `config.py` | `gate_calibrated=False`, `gate_verifier=False`, `verifier_max_claims=8` |
| Telemetri | `app/main.py`, `telemetry/prom.py`, `docs/research/metrics-catalog.md` | olay `detail.gate1`/`detail.gate2` (yalnız bayrak açıkken) + `bg_verifier_verdicts_total{verdict}` + katalog satırları |
| Koşum harness'ı | `cli.py::verify run` | bench + split; `--max-llm-calls` **ZORUNLU**; künyeli JSON |

## 2. Plandan sapmalar (gerekçeli)

1. **`EvidencePack`/`EvidenceUnit` tüketilmedi** (plan `:64`). O dosyalar hiç yazılmadı
   (R23; `p2-reality-audit.md:49`). Kanıt yüzeyi bugünkü gerçek arayüz:
   `list[PageHit]` + `page_texts` eşlemesi.
2. **Görüntü kullanılmıyor** (plan `:163` "metin yoksa sayfa görüntüsü eklenir").
   Gerekçe: (a) kota — beş WebP bir doğrulama çağrısını answerer kadar pahalı yapar ve
   doğrulayıcı iddia BAŞINA çağrılır; (b) determinizm — önbellek anahtarı metnin
   sha256'sıdır. **Bedeli dürüstçe:** metin katmanı BOŞ (taranmış) sayfalarda karar
   `belirsiz` olur ve yanıt düşer — sistem orada sessizce değil, görünür biçimde
   çekimser kalır.
3. **İddia başına çağrı** (plan `:163` "tek API çağrısı"). Toplu istemde tek bir cümle
   değişince TÜM iddialar yeniden ödenir; iddia bazlı anahtar tekrar koşumlarda
   değişmeyen iddiaları bedava yapar. Tavan `verifier_max_claims`.
4. **Karar sözlüğü** `supported | unsupported | belirsiz` (plan: `refuted`/`insufficient`).
   Kontrolcü sözleşmesi. `belirsiz` DESTEKLENMEMİŞ sayılır (şüphede-reddet, G2.1 yönü).
5. **`decide_verdicts` retry dalı YAZILMADI** (plan `:192-198`: `insufficient` varken
   kısıtlı istemle bir kez yeniden üretim). Kapsam dışı bırakıldı: her retry +1 answerer
   +N doğrulayıcı çağrısıdır ve günlük kota **20** ölçüldü (§5). T8'e borç.
6. **`status` sözlüğü GENİŞLEMEDİ.** Düşürülen yanıt da `abstained`tır; ayrım
   `detail.gate2.demoted`. Dördüncü bir değer arayüzün `KNOWN_STATES` kilidini
   ("bilinmeyen durum" kartı) sessizce tetiklerdi.

## 3. Bayrak-kapalı değişmezlik — kanıt

- `tests/app/test_api.py`, `tests/answer/test_base.py`, `tests/answer/test_gemini.py`
  **bu commit'te HİÇ DEĞİŞMEDİ** (`git diff --stat` boş) ve hepsi geçiyor. Kilit budur.
- `tests/answer/test_gate.py::test_flags_off_is_byte_identical_to_p1` — kapı yokken
  `col.notes` içinde `gate1`/`gate2` YOK.
- `tests/app/test_gates_api.py::test_flags_off_body_has_no_detail_key_and_event_has_no_gate_blocks`
  — `/ask` gövde anahtarları tam olarak `{status, honest_miss, answer, hits}`; olay
  `detail`'inde kapı bloğu yok.
- Canlı doğrulama (§6): yeniden başlatılan `:7860`'ta gövde anahtarları
  `['answer', 'hits', 'honest_miss', 'status']`, `detail` YOK.
- **Tek katkı yüzeyi:** `/metrics` iki satır kazanır (`# HELP`/`# TYPE
  bg_verifier_verdicts_total`); bayrak kapalıyken seride hiçbir ÖRNEK yoktur
  (testle kilitli).

## 4. Ölçülen davranış (kota yakmayan)

Üretim kalibratörü (`133444d8c235-train-compat-v1-int8__hybrid__7b56eeeb7327`,
tau = **0.5037**) altı arayüz çipinin hepsini **kapı 1'de çekimsere düşürüyor**:

| çip | p | sonuç |
|---|---|---|
| TMK yerleşim yeri | 0.1439 | abstain |
| İş Kanunu yıllık izin | 0.1729 | abstain |
| 492 s.K. harç tarifesi | 0.2434 | abstain |
| RG 7/10445 kararname | 0.1795 | abstain |
| Onbir yaşındaki çocuk | 0.1632 | abstain |
| Aksansız yazım | 0.3238 | abstain |
| (karş.) canary c103 | 0.7587 | **geçer** |

Bu, `2026-08-30-p2-baslangic.md §4`'teki **%2.2 kapsam** bulgusunun servis tarafındaki
birebir karşılığıdır: kapı 1 tek başına AÇILAMAZ. Sayı bir arıza değil, ölçüm.

## 5. Canlı sonda (port 7862, bayraklar AÇIK, `verifier_max_claims=2`)

| sonda | soru | kapı 1 | kapı 2 | LLM çağrısı |
|---|---|---|---|---|
| 1 | çip: "TMK'ya göre yerleşim yeri…" | p=**0.14389** < tau=**0.50368** → ABSTAIN | koşmadı | **0** |
| 2 | canary c103 | p=**0.75873** ≥ tau → GEÇER | 6 iddia, 2 doğrulandı (tavan), c2=**supported** | 1 answerer + 2 doğrulayıcı = **3** |
| 3 | canary c103 (tekrar) | p=0.75873 → GEÇER | c2=belirsiz (**429**) | 1 answerer + 1 doğrulayıcı = **2** |
| 4 | `:7860` çip (bayrak KAPALI) | — | — | 1 answerer (**429** → degraded) = **1** |
| | | | **TOPLAM** | **6 / 6** |

**Sonda 1-2 doğrulamaları:**
- Çevrimiçi `p` değerleri çevrimdışı hesapla **birebir aynı** (0.1438870314054071 /
  0.7587332259802443) — yani kapı 1, getiricinin ZATEN hesapladığı BM25'i kullanarak
  kalibratörün fit edildiği özelliklerin aynısını üretiyor.
- Doğrulayıcı gerçekten sayfa metnini okudu. c2 gerekçesi: *"Kanıt metnindeki 24.
  maddede, hukuka aykırı olarak kişilik hakkına saldırılan kişinin hâkimden saldırıda
  bulunanlara karşı korunmasını isteyebileceği açıkça ifade edilmektedir."* (atıf
  sayfası `k4721:5`).
- c1 = `belirsiz` **çağrı yapılmadan**: modelin ilk cümlesi atıfsızdı ve paragrafında
  devralınacak `[Sn]` yoktu.
- Yanıt **düşürüldü** (`status="abstained"`, `detail.gate2.demoted=true`,
  metin `VERIFIER_DEMOTE_TEXT`): 6 iddiadan yalnız 2'si doğrulandı (tavan) →
  kırpma da bir "kanıtlanamadı" hâlidir.

**KOTA — bu sondanın en önemli bulgusu.** Sonda 3'ün doğrulayıcı çağrısı 429 döndü ve
API'nin kendi mesajı sayıyı YAZDI:
`quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue: '20',
model: gemini-3.6-flash`. Yani master §9'un "≈20 çağrı/gün" varsayımı bir tahmin değil,
**doğrulanmış bir limit** (`p2-baslangic.md §6`'nın "kotayı yeniden doğrula" maddesi
kapandı). Ölçülen maliyet profili: **tipik bir yanıt 6-7 iddiaya bölünüyor**, yani
varsayılan `verifier_max_claims=8` ile bir `/ask` **6-7 doğrulayıcı çağrısı** demek →
günlük kotayla **istek başına ~3 soru**. Kapı 2'nin ölçekli koşumu ücretli katman
kararına bağlıdır (KULLANICININ kararı).

**Arıza davranışı doğru:** 429 alan doğrulayıcı isteği DÜŞÜRMEDİ; `belirsiz` sayıp
yanıtı düşürdü (şüphede-reddet) ve olay `detail.gate2.claims[].gerekce` alanında
sebebi taşıyor.

**Önbellek isabeti — kanıt (0 API çağrısı).** Sonda 3'te model metni birebir tekrar
etmediği için canlı bir isabet oluşmadı; isabet, **canlı sunucunun yazdığı gerçek
önbellek dosyası** üzerinde kanıtlandı: `data/cache/verifier/b9339922….json`
(`model=gemini-3.6-flash`, `prompt_version=verify-v1`, `verdict=supported`).
Aynı iddia + aynı kanıtla `verify_claim`, **çağrıldığında AssertionError atan** bir
istemciyle yeniden koşuldu:

```
evidence sha eşleşiyor mu: True
anahtar yeniden türetildi ve dosya adıyla eşleşiyor mu: True
verdict=supported  cached=True  llm_called=False
=> 0 API ÇAĞRISI, karar birebir aynı.
```

(Sunucu bu arada yeniden başlatılmıştı — yani isabet süreçler arasında da kalıcı.)
Önbellek dizini `.gitignore`'un `data/*` kuralıyla kapsanıyor (`git check-ignore -v`
ile doğrulandı).

**Sondanın yakaladığı bir hata (düzeltildi):** `llm_calls` sayacı "önbellekten
gelmeyen" verdict'leri sayıyordu; atıfsız bir iddia ne önbellekten gelir ne API'ye
gider, yani sayaç onu bir ÇAĞRI gibi raporluyordu. `Verdict.llm_called` eklendi ve
muhasebe gerçek çağrılara bağlandı (`test_llm_calls_counts_actual_api_calls_not_just_cache_misses`).

## 6. `:7860` geri yüklendi (bayraklar KAPALI)

```
ÖNCE  (eski kod): {"status":"ok","pages":4222,"threshold":10.6,"top_k":5,"pipeline":"hybrid",
                   "index":{"quantization":"int8","revision":"133444d8c235/train-compat-v1/int8"}}
SONRA (yeni kod): (BİREBİR AYNI)
/search  → k4721:1(16.69), k4721:4(13.82), k4721:20(11.60), k4721:39(9.74), k4721:203(9.20)
/ask     → gövde anahtarları ['answer','hits','honest_miss','status'];  detail YOK
           status="degraded" (error_type=http_429 — GÜNLÜK KOTA TÜKENDİ, §5)
/metrics → bg_verifier_verdicts_total yalnız HELP/TYPE, hiçbir örnek yok
```

`degraded` bir regresyon DEĞİLDİR: aynı 429 bayraklardan bağımsız olarak P1 yolunun
da sonucudur (`answer/base.py`'nin değişmemiş degradasyon koruması) ve kota yarın
sıfırlanınca `answered`a döner. Sunucu **ÇALIŞIR halde bırakıldı** (port 7860).

## 7. Yapılmayanlar / borçlar

- **Arayüz T9'a bırakıldı** (kontrolcü izniyle). Bugün düşürülen yanıt mühürlü
  `abstained` kartında `VERIFIER_DEMOTE_TEXT` ile görünür — doğru ama iddia-düzeyi
  atıf çipleri ve "kanıt doğrulamasından geçemedi" ayrımı yok.
- **`verify run` gerçek bileşenlerle koşulmadı** (kota). Stub istemciyle 6 testi var
  (künyeli rapor, düşürme, bütçe tavanı, ikinci koşumun bedava olması, `--split test`
  bariyeri, zorunlu `--max-llm-calls`).
- **Retry dalı (plan `:192-198`) yok** — §2.5.
- Doğrulayıcı token/maliyet telemetriye YAZILMIYOR (`bg_llm_tokens_total` yalnız
  answerer'ı sayıyor); kapı 2 ölçekli koşulacaksa kapatılmalı.
- Kalibratörün `statistical_guarantee` alanı hâlâ `"none"` (n=4, CP üst %52.7) —
  kapı 1 bu haliyle bir KAPI KOŞUMU dayanağı değildir; her olayda
  `detail.gate1.guarantee` olarak dürüstçe taşınıyor.
- Yan gözlem (bu işle ilgisiz): Çarşamba'dan kalma bir `belge-gozu serve` süreci
  (pid 42801) %100 CPU'da dönüyor ve 7860'ı dinlemiyor; dokunulmadı.

---

# §fix — Review turu 1 (2026-08-31)

Kaynak: `p2-t1t2-rotation-review.md` (FIX REQUIRED, 3 High / 4 Medium / 8 Low).
Bayrak-kapalı iniş APPROVED'du; aşağıdakiler `BG_GATE_VERIFIER=true` açılmadan
önce kapatılması gerekenlerdi. **Testler: 666 geçti** (fix öncesi 650 → **+16**),
`make lint` temiz.

## Bulgu bazında sonuç

| # | Bulgu | Ne yapıldı |
|---|---|---|
| **H1** | serve yolunda bütçe HİÇ YOK | `Settings.verifier_max_llm_calls = 10` (İSTEK başına, `BG_VERIFIER_MAX_LLM_CALLS`); `build_gates` bütçeyi `EvidenceGate`e bağlar; `EvidenceGate.evaluate` bütçesizken HER istek için taze bütçe kurar. Tükendiğinde kalan iddialar çağrı yapılmadan `belirsiz` + `detail.gate2.budget_exhausted=true` → düşürme |
| **H2** | ayrıştırılamayan 200 KALICI `belirsiz` olarak önbelleğe yazılıyordu | `parse_verdict` artık `ParsedVerdict(verdict, gerekce, parsed)`; **yalnız `parsed=True` önbelleğe yazılır**. Transport hatası da yazılmaz. Test: bozuk-200 → belirsiz + **sıfır önbellek dosyası**; düzelmiş stub ikinci çağrıda `supported` + 1 dosya |
| **H3** | cümle-BAŞI kısa parça kendi iddiası oluyordu | `_merge_fragments` çift yönlü: `_is_leading_fragment` (TEK KELİME + kısa) bir SONRAKİ cümleye eklenir. Ölçüt karakter değil KELİME sayısı — "Kural budur." (2 kelime) kendi başına kalır, mevcut test bozulmaz. Uçtan uca regresyon: "Evet. Yıllık ücretli izin…" → 1 iddia, `n_supported=1`, **demote YOK**, 1 çağrı |
| **M1** | bütçe "çağrı" sayıyordu, "API denemesi" değil (rotasyon 3× çarpar) | `gemini._API_ATTEMPTS` ContextVar sayacı GERÇEK `generate_content` denemesinin hemen öncesinde artar; `verify.client_attempts(client)` onu okur; `VerifierBudget.max_attempts`/`charge(n)` denemeyle yüklenir (sayamayan stub istemcide taban 1). `detail.gate2.api_attempts` + `summary.verifier_api_attempts` eklendi; CLI bayrağı `--max-llm-attempts` (eski ad alias) |
| **M2** | üç fren tek `reason="threshold"` etiketinde | `prom.ABSTAIN_REASONS = (degraded, threshold, gate1, gate2_demote)` + tek karar yolu `abstain_reason(ev)`; katalog satırı güncellendi; üç kollu app testi |
| **M3** | `PROMPT_VERSION` elle tutulan sabit | `prompt_fingerprint()` = sha256(istem + şema)[:12], `recipe_fingerprint()` deseni; `PROMPT_VERSION = "verify-v1-<fp>"`. İstem düzenlenince önbellek KENDİLİĞİNDEN geçersizleşir (test) |
| **M4** | `Gates.budget` ölü alan | **KALDIRILDI**. `ClaimVerifier.budget` alanı da kaldırıldı; bütçe artık `verify(..., budget=...)` argümanı — tek mekanizma, ömrü çağırana ait (`serve`: istek, CLI: koşum) |
| **L1** | kalibrasyon fail-fast'i ağır yüklemeden SONRA | `load_gate1_artifact()` ayrıldı; `create_app` onu `require_text_artifact` ile aynı yerde, encoder/indeks yüklenmeden ÖNCE çağırır. Test: patlayan encoder ile bile `calibrate fit` hatası alınır |
| **L2** | önbellek yazımı atomik değil | tmp + `os.replace` (`corpus/download.py` deseni) |
| **L3** | bütçe ortada bitince `llm_calls` raporsuz kalıyordu | H1 tasarımıyla yapısal olarak çözüldü: `evaluate` artık yarıda kesilmez (fırlatma yok), `claims`/`api_attempts`/`budget_used` HER ZAMAN raporda; `budget.used` ile `summary.verifier_api_attempts` aynı sayı |
| **L4** | devralınan atıf paragrafın TÜMÜNÜ alıyordu | **EN YAKIN ÖNCEKİ** `[Sn]`e daraltıldı; önünde işaretli cümle yoksa iddia ATIFSIZ kalır. İki test (devralma + atıfsız kalma). Canlı sondada doğrulandı: c1 `src=[]` |
| **L5** | katman/döngü | `gate2_skip_reason` → `answer/base.py` (yalnız `Answer`a dokunuyor); `AskService`teki FONKSİYON-İÇİ import kaldırıldı. **Telemetri→answer import yarısı DISPUTED** (aşağıda) |
| **L6** | rotasyon sarmalayıcısı korumalı üyelere uzanıyor | `GeminiClient` docstring'ine açık **"İÇ API"** bölümü: `_generate(contents, config, *, started, max_attempts)` ve `_json_config(schema)` sözleşme olarak yazıldı; `_json_config`/`build_contents`ın DURUMSUZ olduğu (bu yüzden daima `_slots[0]`) belgelendi. Davranış değişmedi |
| **L7** | "tek anahtarda notlar birebir" iddiası yanlış | `RotatingGeminiClient` docstring'i düzeltildi: deneme/backoff/taksonomi birebir, **notlar değil** (`detail.llm.key` yeni). `key-rotation-report.md` rotasyon görevinin raporu olduğu için ellenmedi |
| **L8** | `_GEREKCE_RE` kesme işaretinde kırpıyordu | Çift tırnak + kaçış dizisi deseni önce, tek tırnak ikinci; test `"Kanun'un 19. maddesi bunu yazar"` |
| Nit | erişilemez `except VerifierBudgetExceeded` | Sınıfın kendisi kaldırıldı (bütçe artık fırlatmıyor) |
| Nit | protokoller dekoratifti | `Gates.retrieval/evidence` artık `RetrievalGate`/`EvidenceGateProtocol` tipli; `app/main.py`deki iki `pyright: ignore` **silindi** |
| — | (fırsat) `index_revision` kopyası | `app/main.py` elle kurulmuş f-string yerine `index/manifest.index_revision()` çağırıyor — servis ile CLI'nin kalibrasyon anahtarı artık tek fonksiyondan |

## Tek itiraz (DISPUTED)

**L5'in ikinci yarısı — `telemetry/prom.py`'nin `answer.*`ten ithali.** Bu ithal
tek amaçla var: sayaç etiketlerinin KAPALI KÜMESİ üreticiyle AYNI kaynaktan
gelsin (`VERDICTS`, `KEY_LABELS`). Kopyalamak, deponun kendi denetiminin
(`2026-08-29-audit-duplicated-contracts.md`) adını koyduğu hatayı geri getirir;
vokabülerleri `config.py`ye taşımak ise alan semantiğini ayar modülüne sokar.
Yön eleştirisi haklı ama bugünkü seçeneklerin ikisi de daha kötü; döngü YOK
(`collect.py` hiçbir şey ithal etmiyor). `base ↔ verify` yarısı FİİLEN ÇÖZÜLDÜ.

## Canlı bütçe-tavanı sondası (port 7862, bayraklar AÇIK, `BG_VERIFIER_MAX_LLM_CALLS=3`)

Soru: canary c103. Sonuç `status="abstained"`, `detail.gate2`:

```
demoted=true  n_claims=6  n_verified=6  n_supported=3  truncated=false
api_attempts=3  budget_max_attempts=3  budget_used=3  budget_exhausted=TRUE
  c1 belirsiz att=0 src=[]   -> atıfsız (L4: önünde işaretli cümle yok)
  c2 supported att=1 src=[1] -> "Kanun metninin 24. maddesinde ..."
  c3 supported att=1 src=[1] -> "Kanunun 25. maddesinde ..."
  c4 supported att=1 src=[1] -> "Kanıt metnindeki 25. maddede ..."
  c5 belirsiz att=0 -> "doğrulayıcı bütçesi doldu (3/3 API denemesi)"
  c6 belirsiz att=0 -> "doğrulayıcı bütçesi doldu (3/3 API denemesi)"
bg_abstain_total{reason="gate2_demote"} 1.0     <- M2 canlıda
bg_verifier_verdicts_total{supported}=3 {belirsiz}=3
```

**Kanıtlananlar:** (a) bütçe GERÇEKTEN kesiyor — 6 iddianın yalnız 3'ü çağrıya
gitti, c5/c6 sıfır denemeyle `belirsiz`; (b) `api_attempts == budget_used == 3`,
yani L3'ün "iki sayı birbirini yalanlıyor" hâli kapandı; (c) `reason` ekseni
ayrışıyor; (d) L4 canlıda görünür (c1 `src=[]`).

**Ölçülen canlı API denemesi = 5, planlanan 4.** Fark bir rotasyondan geldi:
yanıtlayıcının ilk denemesi `key1` ile **http_429** aldı (dünkü günlük kota),
merdiven `key2`ye döndü ve o servis etti — `detail.llm =
{"rotations":[{"from":"key1","error_type":"http_429"}],"key":"key2"}`,
`bg_llm_key_rotations_total{from_key="key1"} 1.0`. Yani 1 planlanan yanıtlayıcı
denemesi 2 oldu; doğrulayıcı tarafı tavanına (3) sadık kaldı. Bu tam olarak
M1'in ölçtüğü çarpandır ve bütçenin neden DENEME sayması gerektiğinin canlı
kanıtıdır.

**Önbellek:** M3 sonrası `PROMPT_VERSION` istem hash'ini taşıdığı için dünkü tek
önbellek girdisi (eski sürüm) artık isabet etmez — beklenen ve istenen davranış
(istem değişti, kararlar kendi isteminde kaldı).

## `:7860` (bayraklar KAPALI) — ÇALIŞIR bırakıldı

```
/healthz → {"status":"ok","pages":4222,"threshold":10.6,"top_k":5,"pipeline":"hybrid",
            "index":{"quantization":"int8","revision":"133444d8c235/train-compat-v1/int8"}}
/search  → k4721:1(16.69), k4721:4(13.82), k4721:20(11.60), k4721:39(9.74), k4721:203(9.20)
/metrics → `bg_verifier_verdicts_total{...}` ve `bg_abstain_total{...}` ÖRNEK SAYISI = 0
```

`/ask` bilinçli olarak koşulmadı (kota; kapı-kapalı gövde şekli §3'te zaten
kilitli ve `tests/app/test_gates_api.py` her koşumda doğruluyor).
