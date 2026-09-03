# T14 — int8 üretim geçişi, tek skor ölçeği ve temsil-farkında korkuluklar

**Dal:** `feat/p0-retrieval-correctness` · **Commit:** `b790f6c` (30 dosya, +1111/−302)
**Tarih:** 2026-08-29

Üretim yolu artık ölçümün kazananını (int8) servis ediyor. Üç indeks temsili
(sign-1bit / int8 / float16) tek bir normalize skor ölçeğinde ([-1,1], sorgu
jetonu başına ortalama MaxSim) buluştu; eşik 60.0 → 0.58 **mekanik ölçek
taşımasıyla** aynı çalışma noktasına taşındı (kalibrasyon DEĞİL).

## Durum tablosu

| # | Madde | Durum | Not |
|---|---|---|---|
| 1 | `PackedIndex.score_all` (+`CHUNK_TOKENS`, `/EMBED_DIM`) | ✅ | Çekirdek `retrieval/core.py`'den taşındı; `as_u64` de store.py'ye taşındı (tek kopya) |
| 1 | `Int8Index.score_all` imza standardizasyonu | ✅ | `(q_emb, chunk_tokens=None)`; instance override'ı korundu; matematik değişmedi |
| 1 | `FloatIndex` → `index/float_store.py` + `score_all` | ✅ | `bench/oracle.py` re-export eder; `native_float_scores` ince delege |
| 1 | `ExhaustiveRetriever` + alias, `ScorableIndex` Protocol | ✅ | Protocol `index/loader.py`'de; retriever `self.tokens/offsets`'i bıraktı |
| 1 | `TwoStageRetriever.search` → `raw/(n_q*EMBED_DIM)` | ✅ | `search_embedding` RAW kaldı (docstring'i öyle diyor) |
| 1 | `TwoStageDiagnosticAdapter` aynı bölme | ✅ | Üretimle birebir aynı ifade; eşitlik testi korundu |
| 2 | `index/loader.py::load_scorable_index` | ✅ | manifest → sınıf; legacy `tokens.npy`; bilinmeyen quant → hata |
| 2 | `app/main.py` loader + iki korkuluk + `/healthz` | ✅ | `build_retriever` olarak çıkarıldı (aşağıya bkz.) |
| 2 | `cli.py bench_run` + `scripts/d1_augmentation.py` | ✅ | Aynı loader; two-stage'de `typer.BadParameter` |
| 3 | `config.py` varsayılanları + Türkçe gerekçeler | ✅ | `index-traincompat-int8`, `0.58` |
| 4 | UI: 2 ondalık, negatif-güvenli çubuk, footer, dipnot | ✅ | + `THRESHOLD` yedeği 60.0 → 0.58, `top_k` healthz'den |
| 5 | Prometheus bucket'ları + katalog | ✅ | + `quantization` etiketi (audit #8) |
| 6 | `colpali-engine==0.3.18` + `uv lock` | ✅ | Lock diff'i tek satır (specifier); sürüm 0.3.18'de kaldı |
| 7 | Test güncellemeleri + yeni testler | ✅ | Hiçbiri gevşetilmedi; 5 yeni test dosyası/bloğu |
| 8 | README (quickstart, kuantizasyon, skor/eşik, mermaid) | ✅ | + token sayısı düzeltmesi |

### Ek tur 1 (ölçek denetimi)

| # | Madde | Durum |
|---|---|---|
| 1 | `bg_retrieval_top_score`/`_score_margin` → `quantization` etiketi | ✅ |
| 2 | `EMBED_DIM` sabiti (`index/chunking.py`) + `as_u64` şekil kontrolü | ✅ |
| 3 | `INT8_MAX` sabiti | ✅ |
| 4 | Cırcır temsile göre anahtarlandı (`"quantization": "int8"`) | ✅ |

### Ek tur 2 (final fold-in)

| # | Madde | Durum | Not |
|---|---|---|---|
| 1 | `build_retriever` çıkarımı; retrieval_eval fixture onu çağırır | ✅ | Fixture'ın kopya mantığı silindi (drift sınıfı kapandı) |
| 2 | `index build` kuantizasyon üzerine yazma korkuluğu | ✅ | + CLI testi |
| 3 | `Quantization` enum'u `index/manifest.py`'ye, 3 üye | ✅ | + `derive --quant float16` reddi (yeni sessiz hata yolu kapatıldı) |
| 4 | `/healthz` `top_k` + UI'nin "ilk 5"i healthz'den | ✅ | |
| 5 | `test_config` varsayılan kilitleri | ✅ | |
| 6 | README token sayısı düzeltmesi | ✅ | 3.759.994 (train-compat); 3.776.882 cpe-0.3.18'e ait |

## Atlanan / kapsam dışı

- `data/bench/results/int8-threshold-transfer.json` **commit edilmedi**: görev
  "data/ altına hiçbir şey yazma" diyordu ve dosya denetim/ölçüm kanıtı olarak
  controller'ın `docs/research/findings/` işine ait. Kod yorumları ve README
  ona referans veriyor — controller'ın commit'ine girmeli.
- `docs/research/findings/` ve `docs/research/evidence/agent-reports/`'a
  dokunulmadı (talimat gereği). `docs/research/metrics-catalog.md` findings
  dizininde değil ve görev açıkça istediği için güncellendi.
- `StageRecord.stage` etiketi `"exhaustive-binary"` olarak BIRAKILDI: rapor
  şemasının parçası ve mevcut koşum JSON'larıyla karşılaştırılabilirliği
  bozmamak için değiştirilmedi (yalnız bir etiket dizesi; skor ölçeğiyle
  ilgisi yok).
- README'nin "Example queries" tablosundaki `60.0` referansı bırakıldı:
  o blok açıkça "Stale — v0 pipeline, kept for the record only" başlıklı ve
  o tarihteki eşik gerçekten 60.0'dı.

## İki dikkat noktası (ölçek iddiasının sınırı)

1. **`~[-1,1]` bandı birim-norm token'lara dayanır.** Binary kol
   `(EMBED_DIM − 2·ham)/EMBED_DIM` ile yapı gereği bu bantta; int8/float16
   kolları girdinin L2-normlu olmasına bağlı (gerçek ColPali çıktısı öyledir).
   Bu yüzden bant testi `tests/index/test_loader.py`'de **normlu** fikstürle
   yapılır; `FakeEncoder` normlu üretmediği için app testinde bant
   iddiası SINANMAZ (yorumda belirtildi). Yanlış fikstürle yazılmış bir bant
   testi ilk koşumda yakalandı ve düzeltildi.
2. **f16/int8 saklama 1.0'ı ~1e-5 aşabilir** (kendi sayfasıyla eşleşen sorgu);
   test toleransı saklama hatası mertebesinde (1e-4).

---

# Doğrulama çıktıları (verbatim)

## 1. `uv run pytest -q -m "not slow" && make lint`

```
........................................................................ [ 31%]
........................................................................ [ 62%]
........................................................................ [ 94%]
.............                                                            [100%]
229 passed, 5 deselected in 1.33s
```

```
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
89 files already formatted
0 errors, 0 warnings, 0 informations
```

## 2. `uv run pytest -m slow -v` (gerçek model, MPS, int8 indeks)

```
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /Users/barandincoguz/Desktop/project-delta/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/barandincoguz/Desktop/project-delta
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.14.2
collecting ... collected 234 items / 229 deselected / 5 selected

tests/index/test_encode_mask.py::test_batch_vs_single_sign_determinism PASSED [ 20%]
tests/retrieval/test_semantic_retrieval_eval.py::test_retrieval_eval_gold_pages_covered PASSED [ 40%]
tests/retrieval/test_semantic_retrieval_eval.py::test_short_query_gold_in_top5 PASSED [ 60%]
tests/retrieval/test_semantic_retrieval_eval.py::test_long_query_rank_ratchet PASSED [ 80%]
tests/retrieval/test_semantic_retrieval_eval.py::test_out_of_corpus_retrieval_eval_scores_below_threshold XFAIL [100%]

================ 4 passed, 229 deselected, 1 xfailed in 21.02s =================
```

Beklendiği gibi **4 passed + 1 xfailed**, hiçbir şey atlanmadı. Abstain kilidi
(`xfail(strict=True)`) 0.58 eşiğinde de **KIRMIZI kalmaya devam ediyor** (XPASS
değil): korpus-dışı 5 sorunun 4'ü hâlâ eşiğin üstünde — ölçek taşıması
monotonik olduğu için beklenen sonuç bu.

## 3. Demo sunucusu yeniden başlatıldı

Eski PID `73926` (`lsof -nP -iTCP:7860 -sTCP:LISTEN`) kapatıldı, port boşaldı,
tam olarak istenen komutla yeniden başlatıldı:

```
BG_DEVICE=mps nohup uv run belge-gozu serve --port 7860 > /tmp/bg-serve.log 2>&1 &
```

Yeni PID `88577`, **ÇALIŞIR DURUMDA BIRAKILDI**.

### `curl -s localhost:7860/healthz`

```
{"status":"ok","pages":4222,"threshold":0.58,"top_k":5,"index":{"quantization":"int8","revision":"133444d8c235/train-compat-v1/int8"}}
```

### `/search` POST `{"query":"Yerleşim yeri nedir?"}`

```
1. k4734:53     score=0.7450  Kamu İhale Kanunu
2. k4721:80     score=0.7404  Türk Medeni Kanunu
3. k4721:10     score=0.7319  Türk Medeni Kanunu
4. k4721:4      score=0.7307  Türk Medeni Kanunu
5. k2918:53     score=0.7301  Karayolları Trafik Kanunu
k4721:4 top-5 içinde: True
```

Gold `k4721:4` **rank 4**, top-1 skoru **0.7450** — artefaktla (rank 4, top1
0.7450) birebir aynı; skorlar normalize ölçekte.

### Korkuluklar (model YÜKLENMEDEN, stub encoder + `create_app`)

```
threshold-guard: IndexCompatibilityError: min_score_threshold=60.0 eski binary ölçeği (0-128) kalıntısı görünüyor; skorlar artık normalize [-1,1] — bkz. data/bench/results/int8-threshold-transfer.json
two-stage-guard: IndexCompatibilityError: two-stage ablasyonu yalnız sign-1bit (PackedIndex) indeksle çalışır; yüklü: int8
```

### Ek: canlı `/metrics` yeni bucket + etiket

```
bg_retrieval_top_score_bucket{le="0.3",quantization="int8"} 0.0
bg_retrieval_top_score_bucket{le="0.4",quantization="int8"} 0.0
bg_retrieval_top_score_bucket{le="0.45",quantization="int8"} 0.0
bg_retrieval_top_score_bucket{le="0.5",quantization="int8"} 0.0
bg_retrieval_top_score_bucket{le="0.55",quantization="int8"} 0.0
bg_retrieval_top_score_bucket{le="0.58",quantization="int8"} 0.0
```

### `uv lock` sonucu

```
Resolved 105 packages in 5.02s
```

Lock diff'i tek satır (`{ name = "colpali-engine", marker = "extra == 'ml'", specifier = "==0.3.18" }`);
çözülen sürüm `0.3.18` olarak kaldı. `uv.lock` commit'e dahil edildi.

---

# Fix round 1 — inceleme bulguları (`int8-switch-review.md`)

**Commit:** `fix(review): int8 geçişi inceleme bulguları — eşik taşınabilirlik dürüstlüğü, d1 loader, grafana etiketi`
14 bulgunun 14'ü ele alındı; **hiçbiri reddedilmedi**.

## Important

### I1 — Eşik temsiller arasında taşınabilir değil (dokümantasyon + uyarı)

Reviewer'ın ölçümünü bağımsız olarak **doğruladım**
(`a2-traincompat-1bit-exhaustive.json`, ham top-1'ler ÷128):

```
1-bit normalized top1: n=43 min=0.4676 med=0.4953 max=0.6133
clears 0.58: 1 / 43
42/43 çalışma noktası bandı: (0.4676, 0.4698]  -> yani ~0.47
```

Ruling R19 uyarınca **per-temsil eşik config'i EKLENMEDİ**. Yapılanlar:

- **(a) README** — temsil değiştirme teklif edilen iki yerde açık uyarı:
  `BG_INDEX_DIR` ile 1-bit'e geçmek eşiği BERABERİNDE GETİRMEZ; ölçülen
  1-bit sayıları (min 0.4676 / med 0.4953 / maks 0.6133, 0.58 → 1/43,
  denk nokta ≈0.47) ve two-stage ablasyonunun da aynı banda düştüğü
  yazıldı. "v0 limitations" eşik maddesine de aynı not eklendi.
- **(b) `config.py`** — eşik yorumuna TAŞINABİLİRLİK paragrafı.
- **(c) `create_app`** — `quantization != "int8"` iken **WARNING** loglanır,
  başlatma ENGELLENMEZ (14 satır). Sabit `config.THRESHOLD_CALIBRATED_ON`.
  İki testle kilitlendi: sign-1bit'te uyarı ÇIKAR, int8'te ÇIKMAZ (uyarının
  gürültüye dönüşmemesi için karşı taraf da test edildi).
- Ayrıca I1'in kaynağı olan **yanlış iddialar düzeltildi**: `core.py`
  ("eşik tek ve ortaktır" → "ortak bant = karşılaştırılabilirlik,
  kalibrasyon taşınabilirliği DEĞİL") ve `loader.py` modül docstring'i.

### I2 — `d1_augmentation.py` docstring/kod uyuşmazlığı → kod düzeltildi

`load_scorable_index` + `ExhaustiveRetriever`'a geçirildi, `--index` help
metni düzeltildi. Gerçek int8 indeksle uçtan uca doğrulandı:

```
yüklenen tip: Int8Index | int8 mi: True
sayfa: 4222 | manifest quant: int8
score_all çalıştı, şekil: (4222,) | bant: 0.999 0.999
```

(`--help` de temiz.) **Önceki raporumdaki ✅ yanlıştı** — docstring'i
güncelleyip kodu güncellememiştim; reviewer haklı.

### I3 — "question for question" iddiası → düzeltildi

README ve `config.py`: çalışma noktası **SAYICA** korunur, soru-soruya
değil; `c306` kazanır (1-bit 59.85 → int8 0.5965), `c211` kaybeder
(1-bit 61.78 → int8 0.5767), çünkü int8 ve 1-bit aynı sıralamayı üretmez.

## Minor — 11/11 düzeltildi, 0 itiraz

| # | Düzeltme |
|---|---|
| M1 | `harness.py` `128` → `EMBED_DIM` (üretim-bitişik son literal) |
| M2 | Grafik böleni kesin pozitif (`Math.max(..., 0.01)`) + sonuç `[0,100]`'e kırpıldı; reviewer'ın karşı-örneği %416.67 → %0.00, normal render değişmedi |
| M3 | Guard'ın f16 dalının ulaşılamazlığı yorumda açıklandı; karşılaştırma bilinçli olarak değişken üzerinden bırakıldı (f16 kısıtı gevşerse kural doğru kalsın) |
| M4 | `chunk_bounds` casusu iki teste eklendi; **mutasyon testiyle doğrulandı**: devretme silinince test KIRMIZI oluyor (önce yeşildi) |
| M5 | `build_retriever(s, encoder)` — `model_name`/`model_revision` içeride türetiliyor |
| M6 | Eşik korkuluğu `create_app`'in EN BAŞINA alındı (VLM+indeks yüklemeden önce); var olmayan index_dir ile kanıtlayan test eklendi |
| M7 | Grafana: `sum by (le)` → `sum by (le, quantization)` (JSON geçerliliği doğrulandı) |
| M8 | `chunking.py` docstring'i düzeltildi: "çağrı anında oku" yalnız `None` ULAŞTIĞINDA geçerli; ClassVar'lı sınıflar etkilenmez |
| M9 | `ScorableIndex` → `TYPE_CHECKING` + `from __future__ import annotations` |
| M10 | Loader temsilin imza dosyasını (`tokens/codes/embs.npy`) kontrol ediyor → çıplak `FileNotFoundError` yerine Türkçe hata; testi var |
| M11 | Boyutlar tek ölçüme (`latency-by-representation.json`) hizalandı: 58 / 476 / 918 MB — README'de 5 yer |

## Süreç notu

M4 mutasyon testinden sonra `git checkout src/belge_gozu/retrieval/core.py`
mutasyonu geri alırken **aynı dosyadaki fix-round düzenlemelerimi de** (I1
docstring'leri + M9) sildi. Fark edildi ve üçü de yeniden uygulandı;
`make lint` + 233 test bunu doğruluyor. Bir dahakine mutasyon için yedek
kopya kullanılmalı.

## Doğrulama

```
233 passed, 5 deselected in 1.48s
```

```
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
89 files already formatted
0 errors, 0 warnings, 0 informations
```

Slow (retrieval/index/app modülleri değiştiği için tekrar koşuldu):

```
tests/index/test_encode_mask.py::test_batch_vs_single_sign_determinism PASSED [ 20%]
tests/retrieval/test_semantic_retrieval_eval.py::test_retrieval_eval_gold_pages_covered PASSED [ 40%]
tests/retrieval/test_semantic_retrieval_eval.py::test_short_query_gold_in_top5 PASSED [ 60%]
tests/retrieval/test_semantic_retrieval_eval.py::test_long_query_rank_ratchet PASSED [ 80%]
tests/retrieval/test_semantic_retrieval_eval.py::test_out_of_corpus_retrieval_eval_scores_below_threshold XFAIL [100%]

================ 4 passed, 233 deselected, 1 xfailed in 21.36s =================
```

Sunucu **yeniden başlatıldı** (runtime Python değişti: `app/main.py`,
`config.py`, `retrieval/core.py`, `index/loader.py`, `bench/harness.py`).
Eski PID 88577 → yeni PID **1205**, aynı komut, ÇALIŞIR bırakıldı.

```
{"status":"ok","pages":4222,"threshold":0.58,"top_k":5,"index":{"quantization":"int8","revision":"133444d8c235/train-compat-v1/int8"}}
```

```
1. k4734:53     0.7450
2. k4721:80     0.7404
3. k4721:10     0.7319
4. k4721:4      0.7307
5. k2918:53     0.7301
k4721:4 top-5: True
```

I1(c) uyarısı üretim yolunda **çıkmıyor** (int8 yüklü): `/tmp/bg-serve.log`
içinde "eşik taşınabilirlik" eşleşmesi 0 — beklenen davranış.
