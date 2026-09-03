# Final review fix wave — rapor

**Tarih:** 2026-08-27 · **Dal:** `feat/p0-retrieval-correctness` · **Kapsam:** Belge-Gözü
tam-dal (whole-branch) final review'unun tek konsolide düzeltme turu.

Kısıtlar tutuldu: hiçbir indeks yeniden inşa edilmedi, `data/` altına **hiç yazılmadı**
(`git status --porcelain data/` boş; canlı doğrulamalarda `BG_DATA_DIR` scratch'e
yönlendirildi), model yalnız okuma amaçlı yüklendi, `git add -A` kullanılmadı.

## Durum tablosu

| # | Bulgu | Durum | Not |
|---|---|---|---|
| CRITICAL 1 | CLI build varsayılanları serve config'inden sürüklenmiş (veri kaybı) | **fixed** | Varsayılanlar `Settings()`'ten; ayrıca `--out`suz sapma artık reddediliyor |
| IMPORTANT 2 | Uyumluluk kontrolü enjekte encoder'da ölü | **fixed** | Fallback = config'ten çözülen değerler (query format **ve** doc prompt sha) |
| IMPORTANT 3 | Uyumsuzluk ipucu çıkmaz sokak | **fixed** | Metin yeniden inşayı öneriyor; `mask_policy` + `corpus_checksum` dalları için test eklendi |
| IMPORTANT 4 | `bench oracle` kolları aynı korpusu doğrulamıyor | **fixed** | `page_ids` eşitlik guard'ı (float + int8) |
| IMPORTANT 5 | Üretim paketi bench paketine bağımlı | **partial** | `chunk_bounds`/`CHUNK_TOKENS` + `git_commit` taşındı; `FloatIndex` hâlâ `bench/oracle.py`'de (aşağıya bkz.) |
| IMPORTANT 6 | Abstain yolu yeni pipeline altında kilitli değil | **fixed (bulgu ile)** | Test eklendi; **ölçüm iddiayı çürüttü** → `xfail(strict=True)` ile kilitlendi (aşağıya bkz.) |
| IMPORTANT 7 | README kaldırılmış pipeline'ı anlatıyor | **fixed** | Telemetri paragrafı + `stage1_ms`/`stage2_ms` NULL notu |
| MINOR a | Ölü `FakeTorchLike` sınıfı | **fixed** | Silindi |
| MINOR b | `test_dataset.py` totolojik assert | **fixed** | `sha256("ood1") % 2 == 0` → `"dev"` (somut değer) |
| MINOR c | `batch_size = 1` gerekçesiz | **fixed** | MPS sign uyuşması 0.9990/0.9989 ölçümü yorumda |
| MINOR d | slow encode testinde veri guard'ı yok | **fixed** | Repo köküne göre yol + `pytest.skip` |
| MINOR e | CWD'ye bağlı veri yolları | **fixed** | `parents[N]` ile repo kökü; retrieval_eval dosyaları artık collection anında okunmuyor |
| MINOR f | `tests/app/__init__.py` eksik | **fixed** | Eklendi |
| MINOR g | `d1_augmentation.py` docstring örneği | **fixed** | `--index data/index-traincompat-1bit` |

## Ayrıntılar

### CRITICAL 1 — CLI/serve varsayılan sürüklenmesi

`index build`'in `--query-format`/`--doc-prompt` varsayılanları artık modül düzeyinde
tek bir `Settings()` örneğinden okunur (`DEFAULT_QUERY_FORMAT`/`DEFAULT_DOC_PROMPT`);
bayraklar hâlâ elle geçersiz kılınabilir. Ek olarak, `--out` verilmediğinde hedef
üretim indeksi olduğu için serve config'inden **sapan** bir format/prompt kombinasyonu
artık `typer.BadParameter` ile reddedilir (kullanıcıya ayrı bir `--out` istenir) —
belgelenen `uv run belge-gozu index build` çağrısının üretim indeksini kaybeden
formatla ezmesi bu iki katmanla birlikte imkânsız hale geldi.

`tests/test_cli.py` artık beklenen formatı `Settings()`'ten okur (sabit
`"cpe-0.3.18"` literali kaldırıldı) ve iki yeni test eklendi: varsayılanların
config'i izlediği ve guard'ın üretim dizinini korduğu.

### IMPORTANT 2 — kontrolün canlandırılması

`app/main.py`'deki fallback'ler artık `resolved_query_format` ve
`sha256(resolved_doc_prompt)`. `doc_prompt_sha256` fallback'i de aynı bayat şekle
sahipti (sabit `None` → kontrol tamamen ölüydü); `processor-default` seçildiğinde etkin
prompt yalnız processor'dan bilinebildiği için orada bilinçli olarak `None` kalır.

Fikstürler üretim değerlerine çekildi (`make_manifest` → `TRAIN_COMPAT_V1` +
`sha256(TRAIN_COMPAT_DOC_PROMPT)`), `tiny_corpus` bunu miras alıyor, `test_api.py`
`index_revision`/`detail.retrieval` beklentileri `train-compat-v1`'e güncellendi.
Hiçbir assert gevşetilmedi. Fikstürün config ile birlikte sürüklenmesini engelleyen bir
kilit testi de eklendi (`test_make_manifest_defaults_track_production_config`).

### IMPORTANT 5 — katmanlama (kısmi)

Yeni `src/belge_gozu/index/chunking.py` (`CHUNK_TOKENS`, `chunk_bounds`) üç kopyayı
birleştirdi: `bench/oracle.py`, `index/quantize.py` ve `retrieval/core.py` (kendi
üçüncü kopyası ve kendi sabiti vardı). `ExhaustiveBinaryRetriever.CHUNK_TOKENS`
örnek üstünde override edilebilir kaldı (`test_chunk_boundaries_do_not_change_scores`
buna dayanıyor). `chunk_tokens is None` kontrolü kullanıldı — açık `0` sessizce
varsayılana dönmüyor. `git_commit` yeni `src/belge_gozu/provenance.py`'ye taşındı;
`bench.harness.git_commit` re-export olarak duruyor.

**Kalan (bilinçli, kapsam dışı):** `index/quantize.py` hâlâ `FloatIndex`'i
`bench/oracle.py`'den import ediyor — yani üretim → bench bağımlılığı tamamen
kesilmedi. Review'un verdiği düzeltme reçetesi yalnız `chunk_bounds`/`CHUNK_TOKENS` ve
`git_commit` taşımasını kapsıyordu; `FloatIndex`'i taşımak `cli.py`, `scripts/`,
`tests/` dahil ayrı bir dokunuş (öneri: `src/belge_gozu/index/float_store.py` + geriye
dönük re-export). Sessizce atlanmadı, bilinçli olarak bir sonraki tura bırakıldı.

### IMPORTANT 6 — abstain kilidi: test yazıldı, **iddia ölçümle çürüdü**

İstenen test yazıldı (retrieval_eval'nin `korpus-disi` satırları, üretim yolu, top-1 skor <
eşik). **Ölçüm iddiayı doğrulamadı** — üç korpus-dışı sorunun üçü de eşiği geçiyor:

| soru | dilim | top-1 skor | eşik 60.0 |
|---|---|---|---|
| c003 | korpus-disi | 66.28 | **geçiyor** |
| c004 | korpus-disi | 71.95 | **geçiyor** |
| c005 | korpus-disi | 67.88 | **geçiyor** |

Tüm retrieval_eval üzerinde dağılımlar (2026-08-27, `data/index-traincompat-1bit`, exhaustive):

| küme | n | min | medyan | maks | eşiğin altında |
|---|---|---|---|---|---|
| cevaplanabilir | 43 | 59.85 | 63.40 | 78.50 | 1/43 |
| cevaplanamaz | 5 | 59.65 | 67.88 | 71.95 | 1/5 |

Yani cevaplanamaz medyanı (67.88) cevaplanabilir medyanının (63.40) **üstünde**:
dağılımlar iç içe, hiçbir tek eşik bu ikisini ayırmıyor. Eşiği yükseltmek çözüm değil
(gerçek soruları abstain'e düşürür). Bu, T11 format değişikliğinin bilinmeyen bir yan
etkisi ve tam olarak review'un şüphelendiği şeydi.

Karar: assert **gevşetilmedi**, test **atlanmadı**; aynı katı iddia
`xfail(strict=True)` ile işaretlendi. Böylece (a) suite yeşil kalır, (b) mevcut gerçek
ölçülmüş rakamlarla belgelenir, (c) eşik P2'de kalibre edilip iddia tuttuğunda test
KIRMIZI olur ve xfail'in kaldırılmasını zorlar — abstain sözü ne sessizce bozulabilir
ne de sessizce "düzelmiş" sayılabilir. `config.py` yorumu ve README'nin
"v0 limitations" maddesi ölçülmüş rakamlarla güncellendi.

> Not: README'nin "Example queries" tablosundaki iki "Clean abstain" satırı eski
> indeks/format altında ölçülmüştü ve bugünkü pipeline'da muhtemelen artık geçerli
> değil (aynı sorulardan biri retrieval_eval'de 73.17 ile top-5'e giriyor). Bu tabloyu uçtan
> uca yeniden ölçmek bu turun kapsamı dışındaydı — bir sonraki tur için işaretlendi.

## Doğrulama çıktıları

### 1. `uv run pytest -q -m "not slow"` + `make lint`

```
165 passed, 5 deselected in 1.40s

uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
82 files already formatted
0 errors, 0 warnings, 0 informations
```

### 2. `uv run pytest -m slow -v`

```
tests/index/test_encode_mask.py::test_batch_vs_single_sign_determinism PASSED [ 20%]
tests/retrieval/test_semantic_retrieval_eval.py::test_retrieval_eval_gold_pages_covered PASSED [ 40%]
tests/retrieval/test_semantic_retrieval_eval.py::test_short_query_gold_in_top5 PASSED [ 60%]
tests/retrieval/test_semantic_retrieval_eval.py::test_long_query_rank_ratchet PASSED [ 80%]
tests/retrieval/test_semantic_retrieval_eval.py::test_out_of_corpus_retrieval_eval_scores_below_threshold XFAIL [100%]

================ 4 passed, 165 deselected, 1 xfailed in 25.44s =================
```

Atlanan yok (veri mevcut). Tek `xfailed` yukarıda gerekçelendirildi.

### 3. CRITICAL 1 — inşa etmeden kanıt

```
$ uv run belge-gozu index build --help
│ --query-format        <cpe-0.3.18|train-compat-v  [default: train-compat-v1] │
│ --doc-prompt          <processor-default|train-c  [default: train-compat]    │
```

`out_dir = out or s.index_dir` hâlâ geçerli, ama artık `--out` yokken sapma reddediliyor
(scratch `BG_DATA_DIR`/`BG_INDEX_DIR` ile, üretim dizinine dokunmadan koşuldu):

```
$ BG_DATA_DIR=<scratch> BG_INDEX_DIR=<scratch>/index \
    uv run belge-gozu index build --fake --query-format cpe-0.3.18
Invalid value: --query-format/--doc-prompt serve config'inden sapıyor
(build=cpe-0.3.18/train-compat config=train-compat-v1/train-compat); üretim
indeksini (<scratch>/index) ezmemek için --out ile ayrı bir dizin verin
→ exit != 0, hiçbir dizin oluşturulmadı
```

Yani `out_dir` üretim indeksini yalnız format+prompt config ile eşleştiğinde hedefler.

### 4. IMPORTANT 2 — canlı davranış değişikliği

`BG_DATA_DIR` scratch'e alınarak (requests.sqlite `data/` altına yazılmasın diye),
üretim indeksi read-only okundu:

```
config: index_dir=data/index-traincompat-1bit query_format_id=train-compat-v1 doc_prompt_id=train-compat
uretim manifest: query_format=train-compat-v1 doc_prompt_sha256=3d11cdfb8bca mask_policy=drop-padding
OK  gercek yol (encoder=None): create_app basarili, rota sayisi = 11
OK  stub encoder: create_app basarili (YAPILANDIRILMIS train-compat-v1 ile karsilastirildi)
ESKI fallback (cpe-0.3.18 + None) ile ayni manifest: ['query_format: indeks=train-compat-v1 serve=cpe-0.3.18']
```

Davranış değişikliği: `query_format` taşımayan bir encoder enjekte edildiğinde eski kod
üretim manifest'ini **cpe-0.3.18**'e karşı ölçüyordu (uydurma uyumsuzluk üretiyor,
gerçek uyumsuzluğu ise kaçırabiliyordu) ve `doc_prompt_sha256` fallback'i `None` olduğu
için doc-prompt kontrolü tamamen ölüydü. Yeni kod config'ten çözülen değerlerle
karşılaştırıyor; manifest'in `doc_prompt_sha256=3d11cdfb8bca…` değeri
`sha256(TRAIN_COMPAT_DOC_PROMPT)` ile birebir eşleşiyor, yani kontrol artık gerçekten
çalışıyor. Ters yön (config sapınca uyumsuzluğun RAPORLANMASI)
`tests/app/test_compat.py::test_create_app_checks_injected_encoder_against_configured_format`
ile kilitlendi.

### Ek — IMPORTANT 4 guard'ı üretim kollarında yanlış alarm vermiyor

```
data/index-traincompat-1bit   n=4222  packed ile birebir esit=True
data/index-traincompat-f16    n=4222  packed ile birebir esit=True
data/index-traincompat-int8   n=4222  packed ile birebir esit=True
```

## Dokunulan dosyalar

`src/`: `cli.py`, `config.py`, `provenance.py` (yeni), `app/main.py`, `index/compat.py`,
`index/chunking.py` (yeni), `index/quantize.py`, `bench/oracle.py`, `bench/harness.py`,
`retrieval/core.py` ·
`tests/`: `conftest.py`, `test_cli.py`, `app/__init__.py` (yeni), `app/test_api.py`,
`app/test_compat.py`, `bench/test_dataset.py`, `corpus/test_manifest.py`,
`index/test_manifest.py`, `index/test_encode_mask.py`,
`retrieval/test_semantic_retrieval_eval.py` ·
kök: `README.md`, `scripts/d1_augmentation.py`
