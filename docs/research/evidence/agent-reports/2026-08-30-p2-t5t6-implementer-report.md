# P2 T5+T6 — Güven özellikleri + kalibratör (dev koşumu)

- **Tarih:** 2026-08-30
- **Branch / parent HEAD:** `feat/p0-retrieval-correctness` @ `db6e7bd`
  (görev `d1087fb`'de başladı; paralel veri taslakçısı arada `6f26d76`+`db6e7bd`'yi
  landing yaptı — yalnız `data/bench/` + `scripts/validate_unans.py`, kod değişmedi)
- **Kapsam:** T5 (özellik çıkarımı) + T6 (kalibratör, versiyonlu artefakt, eşik seçimi)
  + ilk offline dev kalibrasyon koşumu
- **Model/ağ/kota:** HİÇBİRİ. Tüm döngü metin-yanı ve CPU: BM25 skorları + tokenleştirme
  + saf-numpy lojistik regresyon. Görsel indeks (`codes.npy`, 481 MB) hiç yüklenmez.
- **Testler:** 542 passed (taban 490 → **+52**), 6 deselected. `make lint` yeşil
  (ruff check + ruff format + pyright 0 hata).

---

## 1. Ne yapıldı

| Dosya | Durum | Ne |
|---|---|---|
| `src/belge_gozu/answer/calibrate.py` | **yeni** | 5 özellik + saf-numpy kalibratör + eşik seçimi + versiyonlu artefakt + offline veri kümesi |
| `src/belge_gozu/retrieval/text.py` | değişti | `recipe_fingerprint()`, `routed_docs()`, `K1`/`B`/`MIN_TOKEN_CHARS`/`RECIPE_VERSION` sabitleri |
| `src/belge_gozu/retrieval/hybrid.py` | değişti | `HybridRetriever.routed_docs` artık `text.routed_docs`'a DELEGE eder (davranış birebir aynı) |
| `src/belge_gozu/index/manifest.py` | değişti | `index_revision(manifest)` — `app/main.py`'deki satır içi kopyanın ortak evi (T8 oraya çevirecek) |
| `src/belge_gozu/cli.py` | değişti | `belge-gozu calibrate fit` / `calibrate eval` |
| `tests/answer/test_calibrate.py` | **yeni** | 40 test |
| `tests/retrieval/test_text.py` | değişti | 12 test (parmak izi kapsamı + yönlendirme yüklemi) |
| `data/bench/results/p2-calibration-dev-v1.json` | **yeni** | künyeli koşum raporu (COMMIT'LENİR) |

**Commit'lenmeyen:** `data/calibration/**` — `.gitignore:3` (`data/*`) kapsıyor.
Artefakt yeniden üretilebilir; kimliği rapordaki anahtar + künyedir.

---

## 2. Özellikler (T5)

`extract_features(query, text, doc_names, *, bm25=None, window=WINDOW) -> dict[str, float]`

| # | özellik | tanım | ölçüm anı |
|---|---|---|---|
| 1 | `served_top1` | **yönlendirme SONRASI** rank-1 sayfanın BM25 skoru | servis edilen skor (eşiğin gördüğü) |
| 2 | `bm25_margin` | yönlendirme ÖNCESİ top1 − top2 | kanal kararlılığı |
| 3 | `matched_terms_top1` | sorgunun BENZERSİZ token'larından kaçı ön-yönlendirme top-1 sayfasının token kümesinde | sözcüksel örtüşme |
| 4 | `matched_frac` | (3) ÷ benzersiz sorgu token sayısı | örtüşme oranı |
| 5 | `routed` | doküman-adı yönlendirmesi tetiklendi mi (0/1) | kural sinyali |

Ruling gereği DIŞARIDA: `q_len` (veri kümesi artefaktı) ve **tüm görsel özellikler**
(ölçülmüş ters yön, AUC .34).

**`served_top1` vs kanal `bm25_top1` ayrımı yük taşıyor** ve testte kilitli
(`test_served_top1_is_post_routing_and_can_be_below_channel_top1`): pencere-içi
yönlendirme rank-1'i skora göre değil sorguda adı geçen kanuna göre seçebilir, yani
servis edilen skor kanalın en yükseği OLMAYABİLİR (config.py review L1).

**Çevrimiçi kullanılabilirlik (T8 sözleşmesi):** fonksiyon yalnız sorgu metni +
serve'ün zaten kurduğu `BM25Index`/`doc_names`'i alır; bench'e özgü hiçbir girdi yok.
`bm25=` ile önceden hesaplanmış skorlar geçilebildiği için `HybridRetriever.search`
içinden çağrıldığında **ek korpus taraması maliyeti sıfırdır**
(`test_precomputed_scores_give_identical_features`).

**Yönlendirme yüklemi tek yerde:** `retrieval/text.routed_docs`. `hybrid.py` ona
delege eder — iki kopya olsaydı eğitim-zamanı özellik ile servis-zamanı davranış
sessizce ayrışabilirdi ve `recipe_fingerprint()` bunu göremezdi.

---

## 3. Etiket (LLM'siz, dürüst)

```
safe_to_answer = 1  <=>  answerable=True  VE  gold sayfa BM25+yönlendirme top-5'inde
safe_to_answer = 0  <=>  cevaplanamaz HER soru
                     VEYA cevaplanabilir ama getirim ıskaladı  (kanıtsız yanıt = risk bölgesi)
```

Ölçtüğü şey **"yanıt doğru mu" değil, "yanıt vermek güvenli miydi"**. Model çağrısı yok,
judge yok, kota yok. `bench.dataset.assign_split` + `load_splits` kullanılır (hukuk-gruplu
bölme); `verification_status != "verified"` olan HER satır atılır — yani `draft` ile
birlikte **`rejected` de dışarıda** (`load_bench`'in `only_verified` filtresiyle birebir;
`test_load_rows_drops_draft_and_rejected` kilitler). unans_v1'deki 14 `rejected` satır
veri kümesine girmedi.

---

## 4. Kalibratör + eşikler (T6)

- **Saf numpy lojistik regresyon.** sklearn YOK (plan `eval` extra'sı öneriyordu;
  vazgeçildi — çalışma anı zaten saf numpy olmak zorundaydı, fit'i de aynı kodla yapmak
  "eğitilen ile servis edilen aynı mı?" sorusunu ortadan kaldırır). 5 özellik,
  standartlaştırılmış; tam-toplu gradyan inişi, başlangıç **sıfır**, RNG **hiç yok** →
  determinizm tohuma değil YAPIYA dayanır (`test_fit_is_deterministic`).
- **L2 = 1.0 (bias hariç, `l2/n` ölçekli).** Kapalı bırakılmadı: etiket neredeyse
  ayrılabilir olabilir ve düzenlileştirmesiz ağırlıklar sınırsız büyüyüp olasılıkları
  0/1'e yapıştırır, risk-coverage taraması tek bir noktaya çökerdi. Sıralama (AUROC)
  etkilenmez.
- **Eşik #1 (seçilen): risk bütçesi.** dev taramasında `risk <= 0.05` kısıtı altında
  kapsamayı en büyükleyen tau (`bench.calibration_metrics.risk_coverage` üzerinden).
  Bütçeyi sağlayan nokta yoksa bütçe GEVŞETİLMEZ, tam-çekimser eşik döner.
- **Eşik #2 (aday): conformal**, alpha=0.05 — **n-yeterliliği ÖNCE zorunlu**
  (`conformal_candidate`): `n_hata < ceil(1/alpha) - 1 = 19` ise sayı yerine
  `"conformal: n yetersiz"` kaydedilir. T7 review nit'i kapandı.
  Bu koşumda n_hata=151 ≥ 19 → conformal gerçekten hesaplandı.
- Her iki aday **ve** seçilen eşik gerekçesiyle birlikte artefakta yazılır.

### Versiyonlu artefakt anahtarı (denetim bulgusu)

```
data/calibration/<index_revision-güvenli>__<pipeline>__<recipe_fp>/calibrator.json
                 └ 133444d8c235-train-compat-v1-int8 ┘ └hybrid┘ └e896992bedcc┘
```

Plan'ın anahtarı yalnız `index_revision`dı; o dize getirim REÇETESİNİ kodlamaz
(BM25 parametreleri, F5, stopword listesi, pencere, aksan katlaması) — oysa eşiğin
bağlı olduğu eksen tam olarak odur. `recipe_fingerprint()` sha256'nın ilk 12 hanesi;
kapsamı testle kilitli: `F5`, `WINDOW`, `QTF_CAP`, `K1`, `B`, `MIN_TOKEN_CHARS`,
`STOPWORDS`, `_GENERIC`, `_FOLD`, `_WORD`, `_TITLE_LINE`, `RECIPE_VERSION` —
her birinin değişimi parmak izini değiştirir (parametrize test).
Kapsamadığı şey açıkça yazılı: **algoritmik biçim** (adım sırası) otomatik yakalanamaz,
`RECIPE_VERSION` elle artırılır.

`load_calibrator(path, expected_key)` anahtar uyuşmazlığında **fail-fast**
(`CalibrationKeyMismatch`), ayrıca `feature_names != FEATURE_ORDER` durumunda da —
ağırlıkların yanlış özelliklere hizalanması sessiz bir felaket olurdu.

### Katman disiplini

`answer/calibrate.py` modül düzeyinde `belge_gozu.bench`'i **import etmez**; fit/eval
importları fonksiyon içindedir. Üretim yolunun bench paketine bağlanmaması bu projede
yerleşik disiplindir (`provenance.py` tam bu yüzden ayrılmıştı). Alt süreçte doğrulanır:
`test_runtime_import_does_not_pull_the_bench_package`.

---

## 5. Dev koşumu — VERBATIM çıktı

```
$ uv run belge-gozu calibrate fit --unans 'data/calibration/_pins/unans_v1@d1087fb.jsonl' --note "..."
bölme=dev n=173 (pozitif=22, negatif=151)
  cevaplanabilir=26 (gold@5=22, ıska=4) cevaplanamaz=147
fit: iter=1394 converged=True nll=0.3130
  w[served_top1] = -0.0361
  w[bm25_margin] = +0.3372
  w[matched_terms_top1] = -0.1084
  w[matched_frac] = +0.8104
  w[routed] = +0.5423
  bias = -2.3519
tau(risk_budget)=0.538128 coverage=0.023 risk=0.000
  gerekçe: dev taramasında risk<=0.050 kısıtı altında kapsamayı en büyükleyen tau (coverage=0.023, risk=0.000)
  conformal: split-conformal, alpha=0.05, hata n=151
dev: auroc=0.7809 brier=0.0912 ece=0.0333 aurc=0.7319
     DEV yanlış-yanıt (cevaplanamaz): 0.0000 (0/147, %95 üst sınır 0.0202) — G2.1 KAPI SAYISI DEĞİL
risk-coverage (kapsama azalan, ilk 8):
  tau      coverage  risk
  0.009827  1.0000    0.8728
  0.010604  0.9942    0.8721
  0.010659  0.9884    0.8713
  0.010863  0.9827    0.8706
  0.013512  0.9769    0.8698
  0.014431  0.9711    0.8690
  0.015474  0.9653    0.8683
  0.015724  0.9595    0.8675
artefakt -> data/calibration/133444d8c235-train-compat-v1-int8__hybrid__e896992bedcc/calibrator.json (calibrator.json gitignore'da; yeniden üretilebilir)
rapor -> data/bench/results/p2-calibration-dev-v1.json
```

### 5.1 Veri sabitlemesi (ÖNEMLİ)

`data/bench/unans_v1.jsonl` bu görev sırasında paralel bir taslakçı tarafından
**300 → 330 satıra çıkarıldı ve commit'lendi** (`6f26d76` u301-u330 partisi,
`db6e7bd` validator beklentisi). Koşum, controller'ın verdiği pin'e uyularak
`d1087fb`'deki içerikle yapıldı — böylece sayılar tekrarlanabilir ve taslakçıyla
yarışmıyor. **Bu fit, unans_v1 v1.1 (330 satır) ile YENİDEN KOŞULMALIDIR** (§7).
Paralel commit'ler yalnız `data/bench/` ve `scripts/validate_unans.py`'yi değiştirdi,
hiçbir üretim kodunu değil — yani künyedeki `git_commit=db6e7bd` kod durumunu doğru
tanımlıyor.

```
sha256(unans girdisi) = 2955d59e48e005e71a89a630efd0bcd5d3e07806df7d190891bcd6e8d32940ff
                      = sha256(git show d1087fb:data/bench/unans_v1.jsonl)     [doğrulandı]
canary_v1.jsonl       = 1676bb46...5a6e  (48 satır, koşum boyunca değişmedi)
splits_v1.json        = 3a0217f4...8b65  (v1, seed "belge-gozu-splits-v1", hukuk-gruplu)
```

Yeniden üretmek:
`git show d1087fb:data/bench/unans_v1.jsonl > 'data/calibration/_pins/unans_v1@d1087fb.jsonl'`
(dizin gitignore'da). Rapor künyesindeki `note` alanı bunu de yazıyor; `--note` bayrağı
tam bu sebeple eklendi — koşumun kimliği dosya YOLU değil, o anki İÇERİĞİdir (sha256).

Künyedeki `git_commit` = `d1087fb`, yani koşumun üstüne yapıldığı **parent** commit
(`bench run` ile aynı konvansiyon).

### 5.2 Sınıf başına n (dürüst)

| | n |
|---|---|
| dev toplam | **173** |
| pozitif (`safe_to_answer=1`) | **22** |
| negatif | **151** |
| — cevaplanabilir, gold@5 | 22 |
| — cevaplanabilir, getirim ıskası | 4 |
| — cevaplanamaz | 147 |
| cevaplanabilir toplam | 26 |

Kaynak kırılımı: canary 28, unans 145.
Cevaplanamaz gerekçe kırılımı: `korpus-disi` 102, `anlamsiz` 30, `eksik-kanit` 15.

Eşikler (cevaplanabilir ≥ 20, cevaplanamaz ≥ 40) **aşıldı** — prominent uyarı
gerekmedi. Ama **taban oran %87.3 negatiftir** ve bu, aşağıdaki en önemli kırılganlık.

### 5.3 Özellik istatistikleri (dev)

| özellik | ort | std | min | maks | ort(poz) | ort(neg) | tek-değişkenli AUC |
|---|---|---|---|---|---|---|---|
| `served_top1` | 24.04 | 10.89 | 4.36 | 66.68 | 28.98 | 23.32 | 0.604 |
| `bm25_margin` | 3.12 | 3.86 | 0.02 | 21.99 | 5.36 | 2.79 | 0.644 |
| `matched_terms_top1` | 6.00 | 2.77 | 1 | 15 | 7.68 | 5.76 | 0.677 |
| `matched_frac` | 0.593 | 0.176 | 0.167 | 1.0 | 0.725 | 0.574 | 0.722 |
| `routed` | 0.295 | 0.456 | 0 | 1 | 0.591 | 0.252 | 0.670 |

**BEŞ ÖZELLİĞİN DE sınıf ortalaması ve tek-değişkenli AUC'si DOĞRU YÖNDE** (poz > neg,
AUC > 0.5). Ama araştırma ölçümünden **çok daha zayıf**:

| özellik | AUC (canary: 43 poz / **5** neg) | AUC (dev: 22 poz / **151** neg) |
|---|---|---|
| `matched_terms_top1` | .937 | **.677** |
| `matched_frac` | .863 | **.722** |
| `served_top1` | .819 | **.604** |
| `bm25_margin` | .679 | **.644** |
| `routed` | (ölçülmemişti) | **.670** |

Sebep negatif sınıftır: araştırma ölçümü **5** cevaplanamaz soruyla yapılmıştı, buradaki
negatif sınıf **151** kişilik ve çok daha zor (özellikle `eksik-kanit`: konu korpusta VAR,
BM25 ilgili kanunu güvenle getiriyor). Bu bir gerileme değil, **ilk gerçekçi ölçüm** —
ve özellik seçiminin dayandığı sayıların iyimser olduğunu gösteriyor.

### 5.4 Ağırlık işaretleri — sağlamlık kontrolü

İki ağırlık negatif: `served_top1` **−0.036** ve `matched_terms_top1` **−0.108**.
Bu bir işaret ters dönmesi DEĞİL, **eşdoğrusallığın kısmi etkisi**:

```
korelasyon (dev)      served  margin  m_terms  m_frac  routed
served_top1            1.000   0.609    0.723   0.386   0.260
matched_terms_top1     0.723   0.431    1.000   0.694   0.375
```

`matched_terms_top1` ile `matched_frac` neredeyse aynı şeyin iki biçimi (sayı vs
sayı/uzunluk, r=0.694): çok değişkenli fit'te sözcüksel-örtüşme sinyalini `matched_frac`
(+0.810) soğuruyor, ham sayıya oranı SABİT TUTULDUĞUNDA kalan kısmi etki negatif oluyor
(oran sabitken daha büyük sayı = daha uzun sorgu = token başına daha zayıf kanıt).
Aynı ilişki `served_top1` ↔ `bm25_margin` (r=0.609) çiftinde. **Her iki negatif ağırlık
da pozitiflerin yanında küçük** (|−0.11| vs +0.81 / +0.54). Yine de bu, v2'de ele
alınması gereken bir tasarım borcudur (aşağıda §7).

### 5.5 Seçilen eşik + çalışma noktası

| | tau | answered | coverage | risk |
|---|---|---|---|---|
| **seçilen (risk_budget)** | **0.538128** | **4/173** | **0.0231** | **0.000** |
| conformal (alpha=.05, n_hata=151) | 0.331615 | 14/173 | 0.0809 | 0.500 |
| tam kapsama (referans) | 0.009827 | 173/173 | 1.000 | 0.8728 |

Gerekçe (artefakta yazılı): *"dev taramasında risk<=0.050 kısıtı altında kapsamayı en
büyükleyen tau"*. Seçilen tau'da:
- cevaplanamazlardan geçen: **0/147**
- cevaplanabilirlerden geçen: **4/26** → cevaplanabilir kapsama **%15.4**

**Conformal NEDEN seçilmedi (ve neden yine de kayıtlı):** conformal eşiği *hata
tarafının* niceliğini sınırlar — "yeni bir hatanın eşiği geçme olasılığı ≤ alpha".
Ölçüldü ve TUTUYOR: 7/151 = **0.0464 ≤ 0.05**. Ama bu, *answered içindeki risk*
DEĞİLDİR; %87.3 negatif taban oranında ikisi keskin biçimde ayrışıyor (aynı tau'da
answered riski **0.500**). Yani conformal garantisi geçerli ama bu görev için yanlış
nicelik; risk bütçesi doğrudan istenen şeyi optimize ediyor.

### 5.6 Dev metrikleri

| metrik | değer | not |
|---|---|---|
| AUROC | **0.7809** | rastgeleden (0.5) belirgin yukarıda; ayırt ediyor |
| Brier | 0.0912 | |
| ECE | 0.0333 | kalibrasyon fena değil |
| **AURC** | **0.7319** | DÜŞÜK İYİDİR. Taban (rastgele sıralayıcı) ≈ 0.8728 → kazanç var ama küçük; sayı büyük çünkü eğrinin tamamı %87 hata tabanının üstünde yaşıyor |
| DEV yanlış-yanıt (cevaplanamaz) | **0.0000** (0/147) | **%95 üst sınır 0.0202** (Clopper-Pearson) |

**DEV yanlış-yanıt oranı G2.1 KAPI SAYISI DEĞİLDİR** ve rapor JSON'ında da bu ibare
gömülü: eşik dev'de seçildi ve aynı dev'de ölçüldü → iyimser. Kapı, faz sonunda test
bölmesinde TEK koşumla ölçülür (`--split test` + `--yes-final-gate`, gürültülü kırmızı
banner ile korunuyor).

---

## 6. Dürüst kırılganlık beyanı

1. **Taban oran gerçekçi değil.** dev'in %87.3'ü negatif, çünkü unans_v1 (300 satır)
   canary'yi (48) eziyor. Üretim trafiğinde cevaplanamaz soru oranı bu değildir. Risk,
   kapsama ve AURC'nin MUTLAK değerleri bu prior'a bağlıdır; AUROC bağlı değildir
   (sıralama ölçüsü) — bu yüzden **AUROC 0.781 bu koşumun en taşınabilir sayısıdır**.
2. **Pozitif n = 22.** Yirmi iki pozitifle seçilen bir tau kırılgandır; tek bir sorunun
   kayması çalışma noktasını gözle görülür kaydırır. CI'lar geniştir.
3. **Kapsama çok düşük (%2.3 / cevaplanabilirlerin %15.4'ü).** Risk bütçesi %5 çok
   sıkı; %87 negatif tabanda o bütçe ancak en tepedeki 4 soruyla tutuluyor. Bu bir
   *bug* değil, ölçümün söylediği şey: **bugünkü beş metin özelliği, bu zorluktaki
   cevaplanamaz kümesini yüksek kapsamada güvenle ayıramıyor.**
4. **`eksik-kanit` en zor sınıf** (dev'de 15 örnek): konu korpusta var, BM25 güvenle
   getiriyor, sözcüksel özellikler "kanıt var" diyor. Kalibratörün asıl düşmanı bu ve
   örneklem küçük.
5. **Veri akışkan — fit ZATEN BAYAT.** unans_v1 bu görev sırasında 300 → **330** satıra
   çıktı ve commit'lendi (`6f26d76`). Bu koşum d1087fb'ye sabitli, yani **yeni 30 soruyu
   (u301-u330) GÖRMEDİ**. Sayılar v1.1'de değişecektir; `calibrate fit`'i yeniden koşmak
   tek komutluk bir iştir ve §7'nin ilk maddesidir.

---

## 7. Sonraki adımlar (öneri, bu görevde YAPILMADI)

- **ÖNCE: fit'i unans_v1 v1.1 (330 satır) ile yeniden koş.** Tek komut, model/kota yok:
  `uv run belge-gozu calibrate fit --note "unans_v1 v1.1 @ <commit>"`. Bu rapordaki tüm
  dev sayıları o koşumla güncellenmelidir.
- **T8 (serve entegrasyonu):** `extract_features` + `load_calibrator(expected_key)` hazır
  ve çevrimiçi kullanılabilir. `app/main.py`'nin satır içi `index_revision` kopyası
  `index/manifest.index_revision`'a çevrilmeli (helper eklendi, serve'e DOKUNULMADI).
  `/healthz`'e `"calibrator": missing|ok|key-mismatch` eklenmeli.
- **Eşdoğrusallık borcu:** `matched_terms_top1` ile `matched_frac` bir arada tutulacaksa
  ya biri düşürülmeli ya da ortogonalleştirilmeli; aksi halde ağırlık işaretleri
  yorumlanamaz kalır (tahmin gücü etkilenmez, açıklanabilirlik etkilenir).
- **Kapsama sorunu:** %5 risk bütçesinde kapsama kabul edilemez düşük. İki gerçek yol var
  ve ikisi de yeni SİNYAL gerektiriyor (eşik ayarı değil): (a) T1 verifier'ın
  `verifier_support_ratio`'su, (b) madde katmanı / `eksik-kanit`e özgü bir sinyal.
  Bugünkü beş özellikle bütçeyi gevşetmeden kapsamayı büyütmek mümkün değil.
- **Veri:** unans_v1 v2'ye ulaştığında `calibrate fit`'i yeniden koş; `--note` ile yeni
  pin'i yaz.

---

## 8. Doğrulama

```
$ uv run pytest -q -m "not slow"
542 passed, 6 deselected in 4.79s          # taban 490 -> +52

$ make lint
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
107 files already formatted
0 errors, 0 warnings, 0 informations
```

`calibrate eval --split dev` fit ile **birebir aynı** sayıları üretir (aynı `evaluate`
kodu; `test_cli_calibrate_eval_recomputes_from_artifact` kilitler).

---

# §fix — Review turu 1 (`437ba6e` üzerine)

- **Review:** `p2-t5t6-review.md` — VERDICT **APPROVE with findings**. Matematik bağımsız
  olarak yeniden hesaplanmış ve **birebir** tutmuş (AUROC 16 hanede aynı); bulgular
  dürüstlük/yeniden-üretilebilirlik borcu.
- **Fix commit:** `fix(review): t5+t6 bulguları — eşik belirsizlik künyesi, parmakizi
  tamamlığı, yeniden-üretilebilir kayıt`
- **Testler:** 542 → **556** (fix turunda +14; T5+T6 toplamı taban 490'a göre **+66**).
  `make lint` yeşil (ruff + format + pyright 0 hata).

## Bulgu bazında sonuç

| # | Bulgu | Sonuç |
|---|---|---|
| **J1** | Seçilen eşiğin `risk: 0.0`ı belirsizlik taşımıyor (4 satır) | **DÜZELTİLDİ** — aşağıda |
| **M2** | İki literal parmak izinden kaçıyor | **DÜZELTİLDİ** — `_TR_LOWER_MAP`, `_TITLE_KEYWORDS` |
| **M3** | Rapordaki sayılar depoda yeniden üretilemiyor | **DÜZELTİLDİ** — `per_question` + `feature_stats.auc` + `feature_correlations` |
| **M4** | Kayıt varsayılan bayraklarla üretilemiyor | **DÜZELTİLDİ** — `calibrate fit` (bayraksız) ile yeniden koşuldu + `git_blob` pin |
| m5 | Rapor `git_commit` çelişkisi | **DÜZELTİLDİ** (bu bölüm eski §5.1'i geçersiz kılar) |
| m6 | Parametrize kapsamı abartılmış | **DÜZELTİLDİ** — `_WORD`/`_TITLE_LINE` listeye eklendi (iddia artık doğru) |
| m7 | `--split test` MUTLU YOL testsiz | **DÜZELTİLDİ** — sentetik test bölmesiyle |
| m8 | `n_iter` bir fazla | **DÜZELTİLDİ** — `n_iter`=güncelleme, `n_gradient_evals` ayrı |
| m9 | `std` son ulp'te ayrışıyor | **DÜZELTİLDİ** — tek `X.std(axis=0)` çağrısı |
| m10 | `risk_budget` anahtarı içeriğiyle çelişebiliyor | **DÜZELTİLDİ** — aday kendi adıyla anahtarlanır |
| m11 | argsort ikilemesi | **DÜZELTİLDİ** — `text.rank_order()`; T8 imza borcu docstring'e yazıldı |
| m12 | Ölü eşitlik-bozma dalı | **DÜZELTİLDİ** — kaldırıldı, ulaşılamazlığı yorumda |

**İtiraz yok** — on iki bulgunun tamamı kabul edildi.

## J1 — eşiğin belirsizlik künyesi

`ThresholdChoice` artık `n_answered`, `errors`, `risk_point`, `risk_cp_upper_95` ve
`statistical_guarantee` (`"none"` | `"cp_upper<=target"`) taşır. Bayrak `"none"` iken CLI
sarı, kalın bir uyarı basar. **Seçim ölçütü DEĞİŞMEDİ** (controller kararı): eşik hâlâ
nokta-tahmini riske göre seçilir; CP-üst ölçütü bugünkü n'de feasible kümeyi boşaltıp
sistemi tam-çekimsere iterdi ve o karar verifier sinyali gelince verilmelidir. Düzeltme
seçim değil, **dürüst etiketlemedir** — conformal dalının "n yetersiz" korumasının
seçilen eşikteki simetriği.

## M2 — parmak izi tamamlığı → ANAHTAR DÖNDÜ

`tr_lower`ın İ/I eşlemesi ve `extract_doc_name_tokens`ın "KANUN"/"ANAYASA" kapısı gövde
literaliydi; ikisi de artık modül sabiti ve parmak izinde. Sonuç **beklenen ve istenen**:

```
recipe_fingerprint:  e896992bedcc  ->  7b56eeeb7327
artefakt anahtarı:   ...__hybrid__e896992bedcc  ->  ...__hybrid__7b56eeeb7327
```

Eski artefakt `load_calibrator` tarafından artık **reddedilir** (`CalibrationKeyMismatch`)
— mekanizmanın tam olarak yapması gereken şey. Yeniden fit zorunluydu ve yapıldı (M4).

## M4 — kanonik yeniden koşum (330 satır), VARSAYILAN BAYRAKLARLA

M4'ün asıl şikâyeti şuydu: `uv run belge-gozu calibrate fit` (bayraksız) kaydı
üretmiyordu, çünkü kayıt 300 satırlık bir pin dosyasına dayanıyordu. **Artık üretiyor.**

Koşum sırası önemliydi: fix turu boyunca checker-2 `unans_v1.jsonl` üzerinde çalışıyordu
(commit edilmemiş, 112 satırda etiket/kind değişimi). Ara bir koşum HEAD `db6e7bd`'ye
sabitlendi; sonra checker-2 `c6b68c3` ile landing yaptı ve çalışma ağacı HEAD ile
eşitlendi. Kanonik kayıt bu son durumda, **hiçbir `--unans`/`--canary`/`--splits`
bayrağı olmadan** üretildi:

```
$ uv run belge-gozu calibrate fit --note "..."
git_commit  : c6b68c3
unans       : data/bench/unans_v1.jsonl   330 satır / 309 verified
              sha256 2ba2ee70...aec9   git blob 553747756cdb87d40a68008a97fd24ff77391a05
canary      : data/bench/canary_v1.jsonl  48 satır / 48 verified
              sha256 1676bb46...5a6e   git blob 82ba969854d1c92f05d0c187d65ff7579fc5823d
```

Künye artık `sha256`ın YANINDA `git_blob` taşır: sha256 içeriği *kimliklendirir*, blob
onu *geri getirir* — `git cat-file -p 5537477` (doğrulandı, 330 satır). M4'ün "prosa
değil oynatılabilir referans" isteği bu şekilde karşılandı; `n_lines` + `n_verified` de
kayda girdi.

**checker-2'nin `c6b68c3`'ü dev sayılarını DEĞİŞTİRMEDİ** (185 satır, 159 cevaplanamaz,
aynı tau/AUROC): commit "test yakası korpus-dışı tam doğrulama" olduğu için düşen 7
`verified` satırın tamamı TEST bölmesindeydi. Yani controller'ın öngördüğü ek tazeleme
turu **gerekmedi** — kayıt zaten checker-2 sonrası HEAD'e ait.

**Golden-data:** 300 satırlık önceki kayıt silinmedi, `git mv` ile
`data/bench/results/p2-calibration-dev-v1-pin300.json` olarak korundu.

## Yeni dev koşumu — VERBATIM

```
bölme=dev n=185 (pozitif=22, negatif=163)
  cevaplanabilir=26 (gold@5=22, ıska=4) cevaplanamaz=159
fit: iter=1447 converged=True nll=0.2991
  w[served_top1] = +0.0032
  w[bm25_margin] = +0.3026
  w[matched_terms_top1] = -0.1987
  w[matched_frac] = +0.9037
  w[routed] = +0.5381
  bias = -2.4474
tau(risk_budget)=0.503679 coverage=0.022 risk=0.000 (nokta tahmini)
  belirsizlik: n_answered=4 hata=0 %95 CP üst sınır=0.527 guarantee=none
  UYARI: v1 eşiği NOKTA TAHMİNİDİR, n=4, CP üst %52.7 — İSTATİSTİKSEL GÜVENCE YOK; kapı koşumu verifier sinyali olmadan yapılmayacak
  gerekçe: dev taramasında risk<=0.050 kısıtı altında kapsamayı en büyükleyen tau (coverage=0.022, risk=0.000); DİKKAT: bu bir NOKTA TAHMİNİDİR (n=4, hata=0), %95 CP üst sınırı 0.527 > bütçe 0.050 — İSTATİSTİKSEL GÜVENCE YOK
  conformal: split-conformal, alpha=0.05, hata n=163
dev: auroc=0.7817 brier=0.0859 ece=0.0341 aurc=0.7444
     DEV yanlış-yanıt (cevaplanamaz): 0.0000 (0/159, %95 üst sınır 0.0187) — G2.1 KAPI SAYISI DEĞİL
risk-coverage (kapsama azalan, ilk 8):
  tau      coverage  risk
  0.007938  1.0000    0.8811
  0.008792  0.9946    0.8804
  0.008817  0.9892    0.8798
  0.008956  0.9838    0.8791
  0.011344  0.9784    0.8785
  0.012221  0.9730    0.8778
  0.013257  0.9676    0.8771
  0.013305  0.9622    0.8764
artefakt -> data/calibration/133444d8c235-train-compat-v1-int8__hybrid__7b56eeeb7327/calibrator.json (calibrator.json gitignore'da; yeniden üretilebilir)
rapor -> data/bench/results/p2-calibration-dev-v1.json
```

### 300-satır kaydına göre değişim

| | pin300 (v1) | HEAD/330 (kanonik) |
|---|---|---|
| girdi | `unans` @ d1087fb, 300 satır | `unans` @ c6b68c3, **330 satır** |
| yeniden üretme | `--unans <pin dosyası>` (elle kurulmalı) | **bayraksız `calibrate fit`** |
| dev n | 173 (22 / 151) | **185 (22 / 163)** |
| cevaplanamaz | 147 | **159** |
| tau | 0.538128 | **0.503679** |
| coverage | 0.0231 (4/173) | **0.0216 (4/185)** |
| risk (nokta) | 0.000 | **0.000** |
| **güvence** | (kaydedilmemişti) | **`none` — n=4, CP üst 0.527** |
| AUROC | 0.7809 | **0.7817** |
| Brier / ECE | 0.0912 / 0.0333 | **0.0859 / 0.0341** |
| AURC | 0.7319 | **0.7444** |
| DEV yanlış-yanıt | 0/147, üst 0.0202 | **0/159, üst 0.0187** |

30 yeni cevaplanamaz soru **tabloyu değiştirmedi**: aynı 4 soru yanıtlanıyor, AUROC
kıl payı yükseldi, taban risk 0.8728 → 0.8811'e çıktığı için AURC de yükseldi. §6'daki
kırılganlık beyanının tamamı **aynen geçerli** ve J1 ile artık artefaktın kendisinde.

`served_top1` ağırlığı bu koşumda **+0.0032** (pin300'de −0.0361), yani eşdoğrusallık
kaynaklı işaret oynaklığının küçük ve kararsız olduğu doğrulanmış oldu — §5.4'ün kısmi-etki
açıklaması güçleniyor. `matched_terms_top1` −0.1987 ile hâlâ negatif kısmi etkide
(korelasyon `matched_frac` ile .694).

### Kayıt artık kendi kendini kanıtlıyor (M3)

Yeni JSON 185 `per_question` satırı taşıyor (`qid, source, split, answerable,
unanswerable_reason, gold_in_topk, label, prob, answered_at_tau, features`). Yalnız bu
dosyadan yeniden hesaplandı ve **birebir** tuttu:

```
AUROC  recomputed 0.7816508645  committed 0.7816508645
Brier  recomputed 0.0858993207  committed 0.0858993207
ECE    recomputed 0.0340703423  committed 0.0340703423
AURC   recomputed 0.7444462537  committed 0.7444462537
tau: n_answered 4 == 4, errors 0 == 0
```

Ayrıca `kunye.feature_stats[*].auc` (§5.3'ün AUC sütunu) ve `kunye.feature_correlations`
(§5.4'ün eşdoğrusallık argümanı) artık kayıtta — ad-hoc betikte değil. Dosya 103 KB.

### Düzeltmeler (m5/m6)

- **m5:** §5.1'in "künyedeki `git_commit` = `d1087fb`" satırı YANLIŞTI; doğrusu
  `db6e7bd`'ydi (JSON hep doğruydu). Kanonik koşumun künyesinde `git_commit` = **`c6b68c3`**
  (koşumun üstüne yapıldığı parent commit; fix commit'i onun çocuğudur).
- **m6:** §3'ün "testle kilitli" listesi `_WORD` ve `_TITLE_LINE` için fazlaydı (hash'te
  vardılar, parametrize listesinde yoktular). İkisi listeye eklendi — iddia artık doğru;
  liste 9 → 13 vaka (`_TR_LOWER_MAP` ve `_TITLE_KEYWORDS` ile birlikte).
