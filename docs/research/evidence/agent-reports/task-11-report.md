# Task 11 — Steps 1-2 raporu (eğitim formatı kilidi + ST çapraz doğrulama)

Branch: `feat/p0-retrieval-correctness` · Commit: `feat(index): lock training-time query/doc format from ST config (T11 step 1-2)`
Model snapshot: `vidore/colSmol-500M` @ `0aaa9726104ce485884c7b8faa8a58a72d5fdbe7` (git_hash.txt: `8b4f75e476bddc6c34a02722761bd3ada6ac0d3d`)
Ortam: colpali-engine 0.3.18, transformers 5.15.1, sentence-transformers 6.0.0, device=mps

---

## Step 1 — Birincil kaynaktan kilit

### Kaynak dosyalar

`config_sentence_transformers.json` **prompt şablonu içermiyor** — `prompts` alanı boş:

```json
"model_type": "MultiVectorEncoder",
"prompts": { "document": "", "query": "" },
"requirements": { "transformers": { "specifier": ">=5.15" } }
```

Şablon `sentence_bert_config.json` üzerinden chat template'e delege ediliyor:

```json
"processing_kwargs": { "chat_template": { "chat_template": "sentence_transformers" } }
```

Yani gerçek birincil kaynak **`additional_chat_templates/sentence_transformers.jinja`**:

```jinja
{%- if task is defined and task == 'query' -%}
    {{- 'Query: ' + content['text'] -}}
    {%- for _ in range(10) -%}{{- '<end_of_utterance>' -}}{%- endfor -%}
    {{- '\n' -}}
{%- else -%}
    {{- content['text'] -}}
{%- endif -%}
...
{%- if content['type'] == 'image' -%}
    {{- '<|im_start|>User: Describe the image.<image><end_of_utterance>' -}}
```

### Kilitlenen diziler (verbatim)

**Sorgu şablonu** — girdi `Q` için:

```
Query: {Q}<end_of_utterance><end_of_utterance><end_of_utterance><end_of_utterance><end_of_utterance><end_of_utterance><end_of_utterance><end_of_utterance><end_of_utterance><end_of_utterance>\n
```

Yani: prefix `"Query: "`, 10 adet `<end_of_utterance>` augmentation token'ı, newline **en sonda**
(augmentation'lardan SONRA — `QueryFormat.render`'ın mevcut sırasıyla aynı).

**Doküman prompt'u** (verbatim):

```
<|im_start|>User: Describe the image.<image><end_of_utterance>
```

### Bağımsız doğrulama (colpali-engine sürüm arkeolojisi)

| sürüm | doküman prompt'u | sorgu |
|---|---|---|
| 0.3.8 (eğitim dönemi) | `apply_chat_template([text "Describe the image.", image]).strip()` | `"Query: " + q + aug*10 + "\n"` |
| 0.3.9 | `<\|im_start\|>user\n<image>Describe the image.<end_of_utterance>` | aynı |
| 0.3.11 / 0.3.13 / **0.3.18** | `<\|im_start\|>User:<image>Describe the image.<end_of_utterance>\nAssistant:` | newline (0.3.11) ve `"Query: "` (0.3.13) düştü |

0.3.8'in `apply_chat_template` yolunu repo'nun `chat_template.jinja`'sıyla canlı render ettim:

```
RAW   = '<|im_start|>User: Describe the image.<image><end_of_utterance>\n'
STRIP = '<|im_start|>User: Describe the image.<image><end_of_utterance>'
```

→ ST jinja'sının image dalıyla **birebir aynı**. Model kartının "the ST configuration reproduces the
original training-time format" iddiası iki bağımsız kaynakta doğrulandı.

### Sabitlerde ne değişti

**Sorgu tarafı: DEĞİŞİKLİK YOK.** `TRAIN_COMPAT_V1` zaten byte-exact doğruydu —
`render("X") == "Query: X" + "<end_of_utterance>"*10 + "\n"`. `QueryFormat`'ı genişletmeye gerek
olmadı (`newline_before_suffix` gibi bir alan gerekmedi); `CPE_0_3_18` rendering'i dokunulmadan
byte-identical kaldı. Sabitin üstündeki "T11'de doğrulanacak" yorumu, doğrulamanın sonucu ve
kaynaklarıyla değiştirildi.

**Doküman tarafı: SAPMA VAR, iki yeni sabit eklendi** (`index/manifest.py`):

```python
TRAIN_COMPAT_DOC_PROMPT = "<|im_start|>User: Describe the image.<image><end_of_utterance>"
CPE_0_3_18_DOC_PROMPT = "<|im_start|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"
```

Üç fark: (1) `<image>` metinden ÖNCE değil SONRA, (2) `User:` sonrası boşluk, (3) `\nAssistant:`
kuyruğu eğitimde yok. Aynı sayfada token maliyeti: **871 (train-compat) vs 875 (cpe-0.3.18)**.

**Encoder** (`index/encode.py`): `ColSmolEncoder.__init__` artık
`visual_prompt_override: str | None = None` alıyor. Uygulama saf bir yardımcıya çıkarıldı
(birim testlenebilsin diye):

```python
def apply_visual_prompt_override(processor, override: str | None) -> str:
    if override is not None:
        processor.visual_prompt_prefix = override
    return processor.visual_prompt_prefix
```

`visual_prompt_prefix` colpali-engine'de ClassVar; örnek üzerine atama onu yalnız o örnek için
gölgeler ve `process_images` `self.visual_prompt_prefix` okuduğu için override doğrudan text'e
geçiyor (canlı doğrulandı: sınıf sabiti bozulmuyor). `doc_prompt` / `doc_prompt_sha256` artık
etkin prompt'tan türetiliyor, yani manifest override'ı otomatik kaydediyor.
`override=None` mevcut davranışı aynen koruyor.

---

## Step 2 — ST çapraz doğrulama

`scripts/ab_st_reference.py` — bizim `ColSmolEncoder` (train-compat sorgu + train-compat doc
prompt) vs `sentence_transformers.MultiVectorEncoder`. Sorgu tarafına ek olarak **doküman
tarafını da** karşılaştırıyor (doc prompt değişikliği asıl risk olduğu için).

Koşum: `uv run --with "sentence-transformers>=5.0" python scripts/ab_st_reference.py` (device=mps)

```
[sorgu 1: Türk Medeni Kanunu'na göre yerleşim yeri nasıl tanımlanır?]
  bizim  : (43, 128)   ST ref : (43, 128)
  sign uyuşması : 0.993459   max |a-b| : 0.007706   ort. kosinüs : 1.000213

[sorgu 2: Yerleşim yeri nedir?]
  bizim  : (24, 128)   ST ref : (24, 128)
  sign uyuşması : 0.992188   max |a-b| : 0.007812   ort. kosinüs : 1.000238

[doküman (train-compat doc prompt), data/images/k6098/0001.webp]
  bizim  : (871, 128)  ST ref : (871, 128)
  sign uyuşması : 0.992537   max |a-b| : 0.041382   ort. kosinüs : 0.999748

SONUÇ: GEÇTİ
```

Üç ölçümde de token sayıları birebir eşit ve sign uyuşması hedefin (≥ 0.99) üstünde.

### Artık farkın kaynağı: format değil, dtype

Kalan ~0.7%'lik sign uyuşmazlığının şablondan değil sayısal hassasiyetten geldiğini ayrıca
kanıtladım (processor-only kontrol, model forward'ı yok):

```
ours ids len 24 | ST ids len 24 | IDS IDENTICAL: True
```

Sorgu `input_ids`'leri **birebir aynı** → şablon ispatlı olarak özdeş. Fark yalnız bizim
encoder'ın mps'te float16, ST'nin float32 koşmasından geliyor (ortalama kosinüs ~1.0000;
`max |a-b| ≈ 0.0077` L2-normalize edilmiş 128-boyutlu vektörlerde tam olarak fp16 kuantumu).

### Negatif kontrol (karşılaştırma ayırt edici mi?)

```
doc[train-compat] tokens = 871
doc[cpe-0.3.18]   tokens = 875
```

ST'nin doküman yolu 871 token üretti → ST gerçekten train-compat doc prompt'unu kullanıyor ve
karşılaştırma ayırt edici (yanlış prompt olsaydı token sayısı bile tutmazdı).

---

## Testler

- `tests/index/test_manifest.py::test_query_format_render` — düzeltilmiş dizileri kilitliyor
  (yorum: newline'ın augmentation'lardan SONRA olduğu ST jinja'sıyla eşleştirildi).
- `tests/index/test_manifest.py::test_doc_prompt_constants_are_verbatim` (yeni) — iki doc
  prompt'unu verbatim kilitliyor ve farklı olduklarını doğruluyor.
- `tests/index/test_encode_mask.py::test_visual_prompt_override_reaches_process_images` (yeni) —
  stub processor ile override'ın `process_images`'a giden text'e ulaştığını ve ClassVar'ın
  bozulmadığını doğruluyor.
- `tests/index/test_encode_mask.py::test_visual_prompt_override_none_keeps_processor_default` (yeni).

Regresyon: `uv run pytest -q -m "not slow"` → **139 passed, 1 deselected**; `make lint` →
ruff + ruff format + pyright **0 errors**. CI model-free kaldı (model dokunan her şey betikte
ya da `-m slow`).

---

## Step 3'e devir — brief dışı tek ekleme (dikkat)

Brief Step 3'te "doc prompt: Step 1 kararına göre" diyor ama `index build`'de doc prompt'u
seçecek bir yol yoktu. Controller Step 3'te bloke olmasın diye `cli.py`'ye **ek ve varsayılanı
davranış-koruyan** bir bayrak ekledim:

```
--doc-prompt processor-default   (varsayılan — bugünkü davranışın aynısı, override=None)
--doc-prompt train-compat        (TRAIN_COMPAT_DOC_PROMPT)
```

Sorgu formatından bilinçli olarak bağımsız tutuldu ki iki eksen ayrı ayrı (veya 2x2) denenebilsin.
Hiçbir mevcut varsayılan değişmedi; `--fake` yolunda yok sayılıyor. Controller bunu istemiyorsa
`cli.py` değişikliği tek başına geri alınabilir (encoder API'si bağımsız çalışır).

Step 3 için önerilen komutlar:

```
BG_DEVICE=mps uv run belge-gozu index build --precision f16 \
  --query-format cpe-0.3.18   --doc-prompt processor-default --out data/index-cpe0318-f16
BG_DEVICE=mps uv run belge-gozu index build --precision f16 \
  --query-format train-compat-v1 --doc-prompt train-compat --out data/index-traincompat-f16
```

Not: doküman prompt'u indeksi değiştirdiği için (871 vs 875 token/sayfa) A/B'nin iki kolu
gerçekten iki ayrı build gerektiriyor — sorgu formatı tek başına indeksi değiştirmezdi.

## Kalan risk

- Sorgu tarafı ispatlı (`input_ids` özdeş). Doküman tarafı embedding düzeyinde ≥0.99 uyuşuyor ve
  token sayısı birebir; kalan fark fp16/fp32.
- Hangi formatın retrieval_eval'de daha iyi retrieval verdiği **hâlâ açık** — bu Step 4'ün işi. Step 1-2
  yalnız "train-compat gerçekten eğitim zamanı formatı mı?" sorusunu kapattı: evet.

---

# Ek: review R1 düzeltmeleri

Commit: `fix(index): self-verifying format check + doc-prompt guard in bench oracle (review R1)`
(cca153d üzerine; arada T12'nin `5b63b52` commit'i `cli.py`/`encode.py`'ye dokundu, çakışma yok.)

## IMPORTANT 1 — kanıt artık artefaktın içinde

`scripts/ab_st_reference.py` iki kademeye ayrıldı ve her kontrol `[PASS]`/`[FAIL]` basıyor;
herhangi bir kontrol düşerse çıkış kodu 1. **A) Token düzeyi** (processor-only, model forward'ı
yok, bedelsiz):

- her sorgu için bizim rendered string'imizi processor ile tokenize edip ST'nin `preprocess(...,
  task="query")` çıktısıyla karşılaştırıyor → `IDS IDENTICAL` + iki token sayısı;
- negatif kontrol: aynı sayfa üzerinde iki doküman prompt'unun token sayıları + ST'nin doküman
  yolunun hangisiyle eşleştiği.

**B) Embedding düzeyi** eskisi gibi. Rapordaki kanıt artık yeniden koşulduğunda kendini doğruluyor.

## IMPORTANT 2 — `bench oracle` doc_prompt guard'ı

`cli.py::bench_oracle` artık `query_format.format_id`'nin yanında `doc_prompt_sha256`'yı da
karşılaştırıyor. T11'den beri doküman prompt'u bağımsız bir indeks-değiştiren eksen olduğu için
(`--doc-prompt`), yalnız doc prompt'u farklı iki indeks aynı `format_id` ile sessizce
karşılaştırılabiliyordu:

```
doc_prompt uyuşmuyor: packed=<sha[:12]> float=<sha[:12]>
```

## MINOR 3 — fp16 hipotezi ÖLÇÜLDÜ ve **ÇÜRÜTÜLDÜ**

Betiğe `--device` eklendi (varsayılan `auto`, `BG_DEVICE`'ı okur). CPU koşumu her iki tarafı da
fp32'ye alıyor:

`BG_DEVICE=cpu uv run --with "sentence-transformers>=5.0" python scripts/ab_st_reference.py --device cpu`

| ölçüm | mps (bizim fp16) vs ST fp32 | **cpu (her iki taraf fp32)** |
|---|---|---|
| sorgu 1 sign uyuşması | 0.993459 | **0.994368** |
| sorgu 1 max abs / ort. kosinüs | 0.007706 / 1.000213 | **0.007416 / 0.999551** |
| sorgu 2 sign uyuşması | 0.992188 | **0.991211** |
| sorgu 2 max abs / ort. kosinüs | 0.007812 / 1.000238 | **0.021930 / 1.000387** |
| doküman sign uyuşması | 0.992537 | **0.992044** |
| doküman max abs / ort. kosinüs | 0.041382 / 0.999748 | **0.031696 / 0.999684** |

Token kontrolleri her iki koşumda da PASS: `bizim=43 ST=43` ve `bizim=24 ST=24`,
`IDS IDENTICAL: True`; doküman negatif kontrolü `train-compat -> 871`, `cpe-0.3.18 -> 875`,
`ST doküman yolu -> 871`.

**Sonuç, ilk raporun iddiasını düzeltiyor.** fp32'ye geçince uyuşma 1.0'a YAKLAŞMADI — üç ölçümde
de ~0.991-0.994 bandında kaldı (sorgu 1 hafif iyileşti, sorgu 2 ve doküman hafif kötüleşti). Yani
artık farkın baskın kaynağı **fp16 değil**; dtype'ın payı en fazla ~0.1 puan. Kalan fark
implementasyon düzeyinde: bizim yolumuz colpali-engine `ColIdefics3`, referans ise ST'nin kendi
modül yığını (Transformer + `1_Dense` + Normalize + MultiVectorMask; ST yüklerken
`linear.weight/linear.bias UNEXPECTED` raporluyor). Büyüklüğü sınırlı: ortalama kosinüs ≥ 0.9995,
yani vektörler pratik olarak aynı yönde; uyuşmayan ~%0.8'lik bit sıfıra çok yakın bileşenler.

**Format sorusu bundan etkilenmiyor** ve kapalı: `input_ids` birebir özdeş olduğu için şablon
ispatlı biçimde doğru — embedding farkı şablonla ilgili değil.

Yeni risk notu (T11 kararını değiştirmez, p0-gate'e taşınmalı): sign-1bit paketlemede referans
implementasyona göre ~%0.8 bit belirsizliği var. A/B'nin iki kolu da aynı encoder'ı kullandığı
için karşılaştırmayı çarpıtmaz, ancak "binary indeks ne kadar referans-sadık" sorusu ayrı bir
konu olarak açık kalıyor.

## MINOR 5 — ClassVar gölgelemesi denetlenebilir

`_VisualPromptProcessor` Protocol'ü eklendi (`visual_prompt_prefix: str`) ve
`apply_visual_prompt_override` bununla tiplendi. pyright ClassVar ilan edilmiş bir üyeyi
yazılabilir protokol üyesine denk saymadığı için çağrı yerine gerekçeli tek satırlık
`# type: ignore[reportArgumentType]` kondu — gölgelemenin kasıtlı olduğu hem protokolün
docstring'inde hem çağrı yerinde yazılı.

## MINOR 6 — yol çözümü repo köküne bağlandı

`_sample_image` artık `REPO_ROOT = Path(__file__).resolve().parent.parent` üzerinden çözüyor;
ayrıca referans sayfa (`data/images/k6098/0001.webp`) açıkça hedefleniyor ki 871/875 beklentisi
anlamlı olsun. Sayfa yoksa meta.parquet'in ilk satırına, o da yoksa sentetik görsele düşüyor ve
her iki durumda da "token sayıları farklı olabilir/olacak" uyarısı basılıyor.

## Doğrulama

`uv run pytest -q -m "not slow"` → **144 passed, 1 deselected** (T12'nin eklediği testler dahil);
`make lint` → **0 errors, 0 warnings**. Hiçbir indeks build'i çalıştırılmadı, `data/` yazılmadı
(yalnız referans sayfa okundu); model koşumu tek bir CPU betik çağrısıyla sınırlı tutuldu.

## Controller'ın ertelediği maddeler (yapılmadı)

- okunabilir `doc_prompt_id` manifest alanı;
- `--doc-prompt` için CLI testi.
