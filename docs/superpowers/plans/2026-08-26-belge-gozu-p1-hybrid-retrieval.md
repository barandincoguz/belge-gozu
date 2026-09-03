# Belge-Gözü P1 — Hybrid ve Structure-Aware Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **DURUM GÜNCELLEMESİ (2026-08-29, Ruling R23 — bu plan yazıldıktan SONRA yapılan
> autoresearch ölçümleri kapsamı değiştirdi; eski metin silinmez, supersession şudur):**
> Ölçüm kaynağı: `docs/research/findings/2026-08-29-autoresearch-text-channel.md`
> (retrieval_eval R@5 görsel-only 0.2326 → BM25+F5+stoplist+pencere-yönlendirme 0.8140).
> - **T6 (BM25) + T8 (füzyon) + T1'in metin-çıkarım çekirdeği ÜRETİMLEŞTİRİLDİ** —
>   ama T8'in eşit-RRF tasarımı ÖLÇÜMLE REDDEDİLDİ (0.674→0.395); yerine BM25-birincil
>   + doküman-adı pencere-içi (top-20) yönlendirme girdi. Yönlendirme hard-FILTER
>   değildir (aday kümesi değişmez, yalnız pencere içi sıra) → soft-boost ilkesiyle uyumlu.
> - **Sapma:** metin artefaktı `data/text/page_texts.parquet` yerine
>   `<index_dir>/page_texts.parquet` (o indeksin page_id sırasına hizalama + fail-fast
>   eşitlik kontrolü; ayrı checksum tesisatı gerekmez).
> - **F1 kanal ölçümü fiilen yapıldı** (autoresearch #0/#1); görsel kanalın F5 sonrası
>   @5 benzersiz katkısı SIFIR — T7 (dense) ve T10 (reranker) ancak YENİ ölçüm
>   gerekçesiyle açılır (R@20 0.907 → reranker havuzu recall-kapısını geçer, aday).
> - **Backlog'da kalanlar:** T2 (OCR fallback — korpus %99.98 metin katmanlı çıktı),
>   T3 (madde segmentasyonu), T4-T5 (alias/facet/varyant — alias'ın kural-tabanlı
>   çekirdeği autoresearch round-2 adayı), T7, T9 (kısmen harness'ta var), T10, T11,
>   T12 (İNSAN kapılı), T13.

**Goal:** Aday havuzunu hibrit kanallarla (BM25 + dense + visual, çok-varyantlı sorgu,
RRF füzyonu) candidate-union Recall@50 ≥ %95'e çıkarmak; kanun→madde→sayfa yapısını
kurmak; recall-kapılı cross-encoder reranker ile top-5 kaliteyi kanıtlamak; visual-only
ve hybrid-production modlarını ayrı ayrı ölçmek.

**Architecture:** `corpus/text.py` (gömülü metin + kalite + OCR fallback),
`corpus/articles.py` (madde segmentasyonu), `retrieval/query.py` (facet + varyant),
`retrieval/text.py` (BM25), `retrieval/dense.py`, `retrieval/fusion.py`
(RRF + HybridRetriever), `retrieval/rerank.py`, `retrieval/evidence.py` (EvidencePack).
Tüm yeni katmanlar flag'li; visual kanal P0 çıktısı `ExhaustiveBinaryRetriever`dir.

**Tech Stack:** Mevcut stack; BM25 elle yazılır (bağımlılık yok); dense/CE için mevcut
`transformers` (ml extra); OCR yalnız `ocr` extra'sında (motor T2'de seçilir).

**Spec:** `docs/superpowers/specs/2026-08-26-belge-gozu-rag-quality-v2-design.md`
**Master:** `docs/superpowers/plans/2026-08-26-belge-gozu-rag-quality-master.md`

## Global Constraints

- **Önkoşul: P0 kapısı (G0) PASS ve raporu commit'li.** P0 geçmeden bu planın hiçbir
  flag'i varsayılan açılamaz (deneysel, flag-kapalı kod erken yazılabilir).
- CI'da ağ/GPU/model yok (`-m "not slow"`); OCR/dense/CE modelleri testlerde stub'lanır.
- Korpus P1 boyunca da donuk (checksum sabit); yalnız TÜRETİLMİŞ artefaktlar üretilir:
  `data/text/page_texts.parquet`, `data/text/articles.parquet`,
  `data/text/dense-<model>/`, `data/text/bm25/`. Görsel indeks P0 kararındaki dizindir.
- Karar deneyleri **dev split**'te; **test split** yalnız T13 kapı koşumunda.
- Kapı: master §5 G1.1-G1.7. Orijinal sorgu her varyant kümesinde korunur (ilke 12-13);
  metadata hard-filter yasak (ilke: soft boost); reranker yalnız recall-kapılı havuzda
  (ilke 16).
- Gemini kotası: T5'in opsiyonel LLM-rewrite deneyi dışında P1 API çağrısı yapmaz;
  o deney önbellekli ve `--yes-burn-quota` bayraklıdır.

## File Structure

```
src/belge_gozu/
  corpus/text.py           # PageText çıkarımı + text_quality + OCR fallback kancası   [T1,T2]
  corpus/ocr.py            # OcrEngine protokolü + Tesseract/Paddle sarmalayıcıları    [T2]
  corpus/articles.py       # Article segmentasyonu + sayfa eşlemesi                    [T3]
  retrieval/query.py       # QueryFacets, parse_facets, QueryVariant, make_variants    [T4,T5]
  retrieval/text.py        # tokenize_tr, BM25Index                                    [T6]
  retrieval/dense.py       # TextEmbedder protokolü, HFTextEmbedder, DenseIndex        [T7]
  retrieval/fusion.py      # rrf_fuse, HybridRetriever (mod anahtarlı)                 [T8]
  retrieval/rerank.py      # CrossEncoderReranker (+ risk analizi dokümantasyonu)      [T10]
  retrieval/evidence.py    # EvidenceUnit, EvidencePack, build_evidence_pack           [T11]
  bench/harness.py         # ChannelDiagnosticAdapter (union/kanal kayıtları)          [T9]
  cli.py                   # text build / articles build / dense build / bm25 build    [T1,T3,T6,T7]
  config.py                # retrieval_mode, rerank_enabled, ocr_fallback_enabled,
                           # metadata_boost, dense_model, rerank_model, rerank_pool    [ilgili task'ler]
  app/main.py              # mod'a göre retriever kurulumu                              [T8]
data/manifest/aliases.csv  # kanun kısaltma/alias tablosu                              [T4]
data/bench/bench_v2.jsonl  # 120+30 tam benchmark (insan kapılı)                       [T12]
data/bench/splits_v1.json  # law-grouped dev/test doldurulur                           [T12]
tests/corpus/test_text.py, tests/corpus/test_articles.py, tests/retrieval/test_query.py,
tests/retrieval/test_bm25.py, tests/retrieval/test_dense.py, tests/retrieval/test_fusion.py,
tests/retrieval/test_rerank.py, tests/retrieval/test_evidence.py
docs/research/findings/2026-XX-XX-p1-gate.md                                          [T13]
```

---

### Task 1: Sayfa metni çıkarımı + kalite dedektörü (`corpus/text.py`)

**Files:**
- Create: `src/belge_gozu/corpus/text.py`, `tests/corpus/test_text.py`
- Modify: `src/belge_gozu/cli.py` (`text build`)

**Interfaces:**
- Produces:

```python
class PageText(BaseModel):
    page_id: str
    text: str
    text_source: Literal["embedded", "ocr", "none"]
    quality: float                    # [0,1]

def text_quality(text: str) -> float
def extract_embedded_text(pdf_path: Path, page_no: int) -> str     # pymupdf get_text()
def extract_all_texts(meta: pd.DataFrame, data_dir: Path,
                      ocr: "OcrEngine | None" = None,
                      quality_threshold: float = 0.5) -> pd.DataFrame
    # kolonlar: page_id, text, text_source, quality; data/text/page_texts.parquet'e yazan
    # CLI sarmalayıcısı vardır. Karar: quality >= eşik -> "embedded";
    # değilse ocr verilmişse OCR dener (OCR metni de kaliteden geçer); o da olmazsa "none".
```

`text_quality` sözleşmesi (deterministik, testle kilitli):
`len(text.strip()) < 40 → 0.0`; aksi halde
`alpha = harf_orani`, `bad = '�'_orani`,
`tr = türkçe_özel_harf_orani (çğıöşü büyük/küçük, harflere oranla)`;
`skor = clamp01(1.25*alpha + min(0.15, 3*tr) - 8*bad)`.
Born-digital mevzuat sayfası (yoğun Türkçe metin) ≥0.8; boş/taranmış sayfa 0.0;
bozuk-kodlanmış metin (yüksek `�`) < 0.3 verir.

- [ ] **Step 1: Başarısız testleri yaz**

```python
# tests/corpus/test_text.py
import pandas as pd

from belge_gozu.corpus.text import PageText, extract_all_texts, text_quality

TR_PARA = (
    "Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir. "
    "Bir kimsenin aynı zamanda birden çok yerleşim yeri olamaz. "
    "Bu kural ticari ve sınai kuruluşlar hakkında uygulanmaz."
)


def test_quality_scores():
    assert text_quality("") == 0.0
    assert text_quality("kısa") == 0.0
    assert text_quality(TR_PARA) >= 0.8
    garbled = ("�" * 30 + "abc def ghi jkl mno prs tuv") * 3
    assert text_quality(garbled) < 0.3


def test_extract_all_texts_sources(tmp_path, monkeypatch):
    meta = pd.DataFrame([
        {"page_id": "a:1", "doc_id": "a", "page_no": 1, "image_path": "images/a/0001.webp"},
        {"page_id": "b:1", "doc_id": "b", "page_no": 1, "image_path": "images/b/0001.webp"},
    ])
    texts = {"a:1": TR_PARA, "b:1": ""}
    monkeypatch.setattr(
        "belge_gozu.corpus.text.extract_embedded_text",
        lambda pdf_path, page_no, _t=texts: _t[f"{pdf_path.stem}:{page_no}"],
    )

    class StubOcr:
        def recognize(self, image_path):
            return TR_PARA  # "taranmış" sayfa OCR ile okunur

    df = extract_all_texts(meta, tmp_path, ocr=None)
    assert dict(zip(df.page_id, df.text_source)) == {"a:1": "embedded", "b:1": "none"}
    df2 = extract_all_texts(meta, tmp_path, ocr=StubOcr())
    assert dict(zip(df2.page_id, df2.text_source)) == {"a:1": "embedded", "b:1": "ocr"}
```

- [ ] **Step 2: RED gör** — Run: `uv run pytest tests/corpus/test_text.py -v` —
  Expected: FAIL `ModuleNotFoundError`
- [ ] **Step 3: text.py yaz** — yukarıdaki sözleşmeyle; `extract_embedded_text`
  `fitz.open(pdf)[page_no-1].get_text("text")`; `extract_all_texts` belge başına
  PDF'i bir kez açar, OCR çağrısını `image_path` ile yapar (OCR motoru görüntüden
  okur — PDF yeniden render edilmez). CLI: `belge-gozu text build
  [--ocr {none,tesseract,paddle}]` → `data/text/page_texts.parquet`.
- [ ] **Step 4: GREEN + full regression** — Run:
  `uv run pytest tests/corpus -v && uv run pytest -q -m "not slow" && make lint`
- [ ] **Step 5: Gerçek çıkarım koşumu (runbook)** — Run: `uv run belge-gozu text build`
  — Expected: 4060 kanun sayfasının ≥ %95'i `embedded`; 162 tarihî RG sayfasının çoğu
  `none` (OCR T2'de). Sayılar p1-gate taslağına.
- [ ] **Step 6: Commit** — `feat(text): born-digital page text extraction with quality gating`

---

### Task 2: OCR fallback + Türkçe OCR motor benchmark'ı (`corpus/ocr.py`)

**Files:**
- Create: `src/belge_gozu/corpus/ocr.py`, `tests/corpus/test_ocr.py`,
  `data/text/ocr_ground_truth.jsonl` (insan-doğrulamalı küçük örnek)
- Modify: `pyproject.toml` (`ocr` extra), `src/belge_gozu/config.py`
  (`ocr_fallback_enabled: bool = False`, `ocr_engine: Literal["tesseract","paddle"] = "paddle"`)

**Interfaces:**
- Produces: `OcrEngine(Protocol)`: `recognize(image_path: Path) -> str`;
  `TesseractOcr(lang: str = "tur")` (pytesseract, lazy import);
  `PaddleOcr(lang: str = "tr")` (paddleocr, lazy import);
  `make_ocr_engine(name: str) -> OcrEngine`.
  `pyproject.toml`: `ocr = ["pytesseract>=0.3", "paddleocr>=2.7"]` (sürümler kurulum
  günü `uv add --optional ocr` ile pinlenir; ikisi de ana bağımlılığa GİRMEZ).
- Consumes: T1 `extract_all_texts(ocr=...)`.

- [ ] **Step 1: Başarısız birim testleri yaz** — `tests/corpus/test_ocr.py`:
  `make_ocr_engine("tesseract")`/`("paddle")` doğru sınıfı döner (import'lar
  monkeypatch'lenmiş stub modüllerle — CI'da gerçek OCR yok); bilinmeyen ad
  `ValueError`. `TesseractOcr.recognize`'ın pytesseract'a `lang="tur"` geçirdiği
  stub'la doğrulanır.
- [ ] **Step 2: RED → ocr.py yaz → GREEN** — Run: `uv run pytest tests/corpus/test_ocr.py -v`
- [ ] **Step 3: Ground-truth örneği (İNSAN kapısı)** — 5 tarihî sayfa seçilir
  (ör. `rg1965a` 2, `rg1935a` 1, `rg1928a` 1, `rg1975a` 1); her birinin ilk ~200
  karakteri elle okunup `ocr_ground_truth.jsonl`'e yazılır
  (`{"page_id":..., "prefix_text":...}`). Kullanıcı doğrulamadan geçilmez.
- [ ] **Step 4: Motor benchmark koşumu (runbook)** — Run:
  `uv run --with pytesseract --with paddleocr python scripts/ocr_bench.py`
  — betik (bu task'te yazılır, `scripts/ocr_bench.py`): 162 tarihî sayfada iki motoru
  koşturur; (a) ground-truth 5 sayfada karakter hata oranı (Levenshtein/len),
  (b) tüm tarihî sayfalarda `text_quality` dağılımı, (c) sayfa başına süre. Karar:
  CER'i düşük VE süresi kabul edilebilir motor `ocr_engine` varsayılanı olur
  (OCRTurk'ün PaddleOCR bulgusu ön-beklenti; yerel sonuç bağlayıcı). Sonuç tablosu
  p1-gate taslağına.
- [ ] **Step 5: Tarihî metinleri üret** — Run: `uv run belge-gozu text build --ocr <winner>`
  — Expected: `page_texts.parquet`'te tarihî sayfaların `text_source="ocr"` oranı basılır.
- [ ] **Step 6: Full regression + Commit** — `uv run pytest -q -m "not slow" && make lint`
  — `feat(ocr): pluggable OCR fallback, engine chosen by local historical benchmark`

---

### Task 3: Madde segmentasyonu (`corpus/articles.py`)

**Files:**
- Create: `src/belge_gozu/corpus/articles.py`, `tests/corpus/test_articles.py`
- Modify: `src/belge_gozu/cli.py` (`articles build`)

**Interfaces:**
- Produces:

```python
class Article(BaseModel):
    article_id: str          # "k4721:m19" | "k4721:gm5" (geçici madde)
    doc_id: str
    article_no: int
    kind: Literal["madde", "gecici"]
    heading: str             # madde başlığı satırı (yoksa "")
    text: str
    page_ids: list[str]      # maddenin yayıldığı sayfalar (sıralı)

ARTICLE_RE = re.compile(r"^\s*(?:(GEÇİCİ|Geçici)\s+)?(?:MADDE|Madde)\s+(\d+)\s*[-–—]", re.M)

def segment_articles(page_texts: pd.DataFrame) -> pd.DataFrame
    # girdi: T1 şeması (page_id,text,text_source,quality); text_source=="none" sayfalar atlanır
    # çıktı kolonları: article_id, doc_id, article_no, kind, heading, text, page_ids(list)
    # data/text/articles.parquet'e yazan CLI sarmalayıcısı vardır
def article_page_map(articles: pd.DataFrame) -> dict[str, list[str]]   # article_id -> page_ids
def page_article_map(articles: pd.DataFrame) -> dict[str, list[str]]   # page_id -> article_ids
```

Algoritma: belge başına sayfalar sırayla birleştirilir (sayfa sınır ofsetleri tutulur);
`ARTICLE_RE` eşleşmeleri madde başlangıçlarıdır; madde metni bir sonraki başlangıca
kadar sürer; kapsanan ofset aralığından `page_ids` çıkarılır; `heading` = madde
başlangıcından önceki en yakın boş-olmayan, `ARTICLE_RE`'ye uymayan ve ≤80 karakterlik
satır (mevzuat.gov.tr formatında madde başlıkları maddeden önce gelir), yoksa `""`.

- [ ] **Step 1: Başarısız testleri yaz**

```python
# tests/corpus/test_articles.py
import pandas as pd

from belge_gozu.corpus.articles import article_page_map, page_article_map, segment_articles

P1 = (
    "V. Yerleşim yeri\n1. Tanım\nMadde 19- Yerleşim yeri bir kimsenin sürekli kalma "
    "niyetiyle oturduğu yerdir.\nBir kimsenin aynı zamanda birden çok yerleşim yeri olamaz.\n"
)
P2 = (
    "devamı satırlar buraya taşar.\n2. Değiştirme\nMadde 20- Bir yerleşim yerinin "
    "değiştirilmesi yenisinin edinilmesine bağlıdır.\nGeçici Madde 1- Bu Kanunun geçici hükmü.\n"
)


def make_pt():
    return pd.DataFrame([
        {"page_id": "k4721:4", "text": P1, "text_source": "embedded", "quality": 0.9},
        {"page_id": "k4721:5", "text": P2, "text_source": "embedded", "quality": 0.9},
        {"page_id": "rg1928a:1", "text": "", "text_source": "none", "quality": 0.0},
    ])


def test_segmentation_and_spanning():
    df = segment_articles(make_pt())
    by_id = {r.article_id: r for r in df.itertuples()}
    assert set(by_id) == {"k4721:m19", "k4721:m20", "k4721:gm1"}
    assert by_id["k4721:m19"].page_ids == ["k4721:4", "k4721:5"]  # sayfa taşması
    assert "sürekli kalma" in by_id["k4721:m19"].text
    assert by_id["k4721:m19"].heading == "1. Tanım"
    assert by_id["k4721:gm1"].kind == "gecici"


def test_maps():
    df = segment_articles(make_pt())
    assert article_page_map(df)["k4721:m20"] == ["k4721:5"]
    assert "k4721:m19" in page_article_map(df)["k4721:4"]


def test_none_pages_skipped():
    df = segment_articles(make_pt())
    assert not any(p.startswith("rg1928a") for ps in df.page_ids for p in ps)
```

- [ ] **Step 2: RED gör** — Run: `uv run pytest tests/corpus/test_articles.py -v`
- [ ] **Step 3: articles.py yaz** — yukarıdaki sözleşme + algoritmayla.
- [ ] **Step 4: GREEN + full regression** — Run:
  `uv run pytest tests/corpus -v && uv run pytest -q -m "not slow" && make lint`
- [ ] **Step 5: Gerçek koşum (runbook)** — Run: `uv run belge-gozu articles build`
  — Expected: TMK için m1..m1030 aralığında ~1000+ madde; `k4721:m19` →
  `page_ids` içinde `k4721:4`. Bu iki sayı p1-gate taslağına; `k4721:m19` kontrolü
  ayrıca `tests/corpus/test_articles.py`'ye `-m slow` gerçek-veri testi olarak eklenir
  (parquet varsa koşar, yoksa skip).
- [ ] **Step 6: Commit** — `feat(articles): statute article segmentation with page mapping`

---

### Task 4: Kanun alias tablosu + facet ayrıştırma (`retrieval/query.py`)

**Files:**
- Create: `src/belge_gozu/retrieval/query.py`, `tests/retrieval/test_query.py`,
  `data/manifest/aliases.csv`

**Interfaces:**
- Produces:

```python
class QueryFacets(BaseModel):
    law_numbers: list[str]       # ["4721"]
    doc_ids: list[str]           # ["k4721"] (alias/numara çözümü)
    article_nos: list[int]       # [19]
    quoted_phrases: list[str]

def load_aliases(path: Path) -> pd.DataFrame          # kolonlar: doc_id, alias
def parse_facets(query: str, aliases: pd.DataFrame) -> QueryFacets
def tr_lower(s: str) -> str                           # Türkçe küçük harf (İ->i, I->ı)
```

Ayrıştırma kuralları: kanun no `(\d{3,4})\s*sayılı`; madde
`(?:madde|md\.?|m\.)\s*(\d+)` ve `(\d+)\s*(?:inci|nci|üncü|uncu|ıncı|. )?\s*madde`;
alias eşleşmesi `tr_lower` üzerinden en-uzun-önce alt dizi araması; tırnaklı ifadeler
`"..."` / `'...'`. `aliases.csv` başlangıç içeriği: her korpus belgesi için resmi ad +
yaygın kısaltmalar (TMK, Medeni Kanun → k4721; TBK, Borçlar Kanunu → k6098; TCK → k5237;
KVKK → k6698; VUK → k213; HMK → k6100; CMK → k5271; TTK → k6102; İYUK → k2577;
Anayasa → k2709; İş Kanunu → k4857; KDV Kanunu → k3065; SGK/GSS → k5510; vb. — 50
belgenin tamamı için en az resmi ad satırı).

- [ ] **Step 1: Başarısız testleri yaz**

```python
# tests/retrieval/test_query.py
import pandas as pd

from belge_gozu.retrieval.query import QueryFacets, parse_facets, tr_lower

ALIASES = pd.DataFrame(
    [("k4721", "Türk Medeni Kanunu"), ("k4721", "TMK"), ("k6098", "Türk Borçlar Kanunu")],
    columns=["doc_id", "alias"],
)


def test_tr_lower():
    assert tr_lower("İMAR") == "imar" and tr_lower("ISPARTA") == "ısparta"


def test_law_number_and_article():
    f = parse_facets("4721 sayılı Kanun madde 19 ne diyor?", ALIASES)
    assert f.law_numbers == ["4721"] and f.doc_ids == ["k4721"] and f.article_nos == [19]


def test_alias_resolution():
    f = parse_facets("TMK'ya göre yerleşim yeri nedir?", ALIASES)
    assert f.doc_ids == ["k4721"]
    f2 = parse_facets("Türk Medeni Kanunu'na göre yerleşim yeri nasıl tanımlanır?", ALIASES)
    assert f2.doc_ids == ["k4721"]


def test_quoted_phrase():
    f = parse_facets('"sürekli kalma niyetiyle" ifadesi hangi maddede?', ALIASES)
    assert f.quoted_phrases == ["sürekli kalma niyetiyle"]


def test_no_facets():
    f = parse_facets("kira artışı en fazla ne olabilir?", ALIASES)
    assert f == QueryFacets(law_numbers=[], doc_ids=[], article_nos=[], quoted_phrases=[])
```

- [ ] **Step 2: RED → query.py + aliases.csv yaz → GREEN** — Run:
  `uv run pytest tests/retrieval/test_query.py -v`
- [ ] **Step 3: Full regression + Commit** — `uv run pytest -q -m "not slow" && make lint`
  — `feat(query): law alias table and metadata facet parsing`

---

### Task 5: Sorgu varyantları (`retrieval/query.py` devamı)

**Files:**
- Modify: `src/belge_gozu/retrieval/query.py`, `tests/retrieval/test_query.py`

**Interfaces:**
- Produces:

```python
class QueryVariant(BaseModel):
    text: str
    kind: Literal["original", "normalized", "legal", "keyword"]

STOPWORDS_TR: frozenset[str]   # ~40 kelimelik minimal liste (ne, nasıl, göre, için, ...)

def make_variants(query: str, facets: QueryFacets,
                  doc_names: dict[str, str]) -> list[QueryVariant]
    # [0] HER ZAMAN original (ilke 12-13). normalized: tr_lower + noktalama temizliği.
    # legal: facets.doc_ids çözüldüyse "<resmi belge adı> [madde N] <normalized>";
    #        çözülmediyse üretilmez. keyword: normalized'dan STOPWORDS_TR çıkarılmış hali
    #        (en az 2 kelime kalıyorsa). Boş/yinelenen metinler elenir, sıra korunur.
```

LLM tabanlı rewrite bu task'te YOK (E1 ablasyonu T13'te, flag'li ve önbellekli
`scripts/e1_llm_rewrite.py` ile; kazanç kanıtlanmadan koda girmez — RAGTurk'ün
"üretken katman yığma Türkçe'de morfolojik ipuçlarını bozabilir" bulgusu ve EACL 2024
expansion-failure kanıtı).

- [ ] **Step 1: Başarısız testleri yaz**

```python
def test_variants_original_first_and_preserved():
    f = parse_facets("TMK'ya göre yerleşim yeri nedir?", ALIASES)
    vs = make_variants("TMK'ya göre yerleşim yeri nedir?", f, {"k4721": "Türk Medeni Kanunu"})
    assert vs[0].kind == "original" and vs[0].text == "TMK'ya göre yerleşim yeri nedir?"
    kinds = [v.kind for v in vs]
    assert "legal" in kinds and kinds.count("original") == 1
    legal = next(v for v in vs if v.kind == "legal")
    assert legal.text.startswith("Türk Medeni Kanunu")


def test_variants_no_facets_no_legal():
    f = parse_facets("kira artışı en fazla ne olabilir?", ALIASES)
    vs = make_variants("kira artışı en fazla ne olabilir?", f, {})
    assert [v.kind for v in vs][0] == "original"
    assert all(v.kind != "legal" for v in vs)


def test_variants_dedup():
    f = parse_facets("yerleşim yeri", ALIASES)
    vs = make_variants("yerleşim yeri", f, {})
    assert len({v.text for v in vs}) == len(vs)
```

- [ ] **Step 2: RED → uygula → GREEN** — Run: `uv run pytest tests/retrieval/test_query.py -v`
- [ ] **Step 3: Full regression + Commit** — `uv run pytest -q -m "not slow" && make lint`
  — `feat(query): deterministic query variants, original always preserved`

---

### Task 6: BM25 metin kanalı (`retrieval/text.py`)

**Files:**
- Create: `src/belge_gozu/retrieval/text.py`, `tests/retrieval/test_bm25.py`
- Modify: `src/belge_gozu/cli.py` (`bm25 build`)

**Interfaces:**
- Produces:

```python
def tokenize_tr(text: str, prefix_stem: int | None = 5) -> list[str]
    # tr_lower + alfasayısal olmayanlardan bölme + len<2 eleme +
    # prefix_stem verilirse her token ilk N karaktere kırpılır (kaba Türkçe kök yaklaşımı;
    # F1 ablasyonunda stem'li/stem'siz ölçülür)

class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    @classmethod
    def build(cls, docs: list[tuple[str, str]], prefix_stem: int | None = 5) -> "BM25Index"
        # docs: (id, text) — id sayfa (page_id) VEYA madde (article_id) olabilir
    def search(self, query: str, k: int = 50) -> list[tuple[str, float]]
    def phrase_hits(self, phrase: str) -> list[str]     # tam alt-dizi geçen id'ler
    def save(self, dir: Path) -> None
    @classmethod
    def load(cls, dir: Path) -> "BM25Index"
```

Uygulama: saf numpy/dict (bağımlılık yok; 4222 sayfa + ~10-15k madde ölçeğinde
yeterli). IDF: `ln(1 + (N - df + 0.5)/(df + 0.5))`. `phrase_hits` ham metinde
`tr_lower` alt-dizi araması (tırnaklı ifade kanalı).

- [ ] **Step 1: Başarısız testleri yaz**

```python
# tests/retrieval/test_bm25.py
from belge_gozu.retrieval.text import BM25Index, tokenize_tr


DOCS = [
    ("k4721:4", "Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir."),
    ("k4721:5", "Yerleşim yerinin değiştirilmesi yenisinin edinilmesine bağlıdır."),
    ("k6098:120", "Kira bedelinin belirlenmesine ilişkin anlaşmalar tüketici fiyat endeksi."),
]


def test_tokenizer():
    assert tokenize_tr("Yerleşim yeri, oturduğu YERDİR!") == ["yerle", "yeri", "oturd", "yerdi"]
    assert tokenize_tr("Yerleşim yeri", prefix_stem=None) == ["yerleşim", "yeri"]


def test_relevance_order():
    idx = BM25Index.build(DOCS)
    top = idx.search("yerleşim yeri nedir", k=3)
    assert top[0][0] in ("k4721:4", "k4721:5")
    assert top[0][1] >= top[-1][1]
    assert idx.search("kira bedeli endeks", k=1)[0][0] == "k6098:120"


def test_phrase_hits():
    idx = BM25Index.build(DOCS)
    assert idx.phrase_hits("sürekli kalma niyetiyle") == ["k4721:4"]
    assert idx.phrase_hits("olmayan ifade") == []


def test_roundtrip(tmp_path):
    idx = BM25Index.build(DOCS)
    idx.save(tmp_path)
    idx2 = BM25Index.load(tmp_path)
    assert idx.search("yerleşim", 2) == idx2.search("yerleşim", 2)
```

- [ ] **Step 2: RED → text.py yaz → GREEN** — Run: `uv run pytest tests/retrieval/test_bm25.py -v`
- [ ] **Step 3: CLI build** — `belge-gozu bm25 build` → `data/text/bm25/pages/` +
  `data/text/bm25/articles/` (page_texts + articles parquet'lerinden).
- [ ] **Step 4: Full regression + Commit** — `uv run pytest -q -m "not slow" && make lint`
  — `feat(bm25): dependency-free Turkish BM25 channel (page + article level)`

---

### Task 7: Dense metin kanalı (`retrieval/dense.py`) + model seçim koşumu

**Files:**
- Create: `src/belge_gozu/retrieval/dense.py`, `tests/retrieval/test_dense.py`,
  `scripts/dense_model_select.py`
- Modify: `src/belge_gozu/cli.py` (`dense build --model NAME`),
  `src/belge_gozu/config.py` (`dense_model: str = ""` — seçim koşumu doldurur)

**Interfaces:**
- Produces:

```python
class TextEmbedder(Protocol):
    def embed_passages(self, texts: list[str]) -> np.ndarray    # (n, d) L2-normalize
    def embed_query(self, text: str) -> np.ndarray              # (d,) L2-normalize

class HFTextEmbedder:
    """transformers + mean-pooling; E5 ailesi için 'query: '/'passage: ' önekleri
    (model adına göre otomatik: adı 'e5' içeriyorsa önek uygulanır)."""
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 16): ...

class FakeTextEmbedder:
    """sha256-tohumlu deterministik embedding (FakeEncoder kalıbı); CI testleri için."""

class DenseIndex:
    ids: list[str]; vecs: np.ndarray                   # (n, d) float32, L2-normalize
    @classmethod
    def build(cls, items: list[tuple[str, str]], embedder: TextEmbedder) -> "DenseIndex"
    def search(self, query: str, embedder: TextEmbedder, k: int = 50) -> list[tuple[str, float]]
    def save(self, dir: Path) -> None
    @classmethod
    def load(cls, dir: Path) -> "DenseIndex"
```

- [ ] **Step 1: Başarısız testleri yaz** — `tests/retrieval/test_dense.py`:
  `FakeTextEmbedder` ile 4 maddelik indekste sorgu embedding'i madde-2 embedding'ine
  eşitken `search` madde-2'yi 1. döndürür; skorlar kosinüs ([-1,1]); roundtrip
  save/load; `HFTextEmbedder`'ın E5 adında `query: ` öneki uyguladığı stub tokenizer
  ile doğrulanır (gerçek model yok).
- [ ] **Step 2: RED → dense.py yaz → GREEN** — Run: `uv run pytest tests/retrieval/test_dense.py -v`
- [ ] **Step 3: Model seçim koşumu (runbook)** — adaylar: `BAAI/bge-m3` (dense modu),
  `intfloat/multilingual-e5-small`, `intfloat/multilingual-e5-base` + TR-TEB retrieval
  liderlerinden CPU-uygun en çok 2 ek aday (koşum günü TR-TEB tablosundan; kaynak spec
  §9.2). Run: `uv run python scripts/dense_model_select.py --bench data/bench/retrieval_eval_v1.jsonl`
  — her aday için madde-düzeyi indeks kurar, dev soruları üzerinde madde→sayfa
  Recall@{10,50} + sorgu encode süresi + bellek basar. Karar: Recall@50 en yüksek VE
  Space bütçesine sığan model `dense_model` olur; sayılar p1-gate taslağına.
- [ ] **Step 4: `dense build` CLI** — seçilen modelle `data/text/dense-<model-kisa-ad>/`.
- [ ] **Step 5: Full regression + Commit** — `uv run pytest -q -m "not slow" && make lint`
  — `feat(dense): multilingual dense article channel, model chosen on local benchmark`

---

### Task 8: RRF füzyonu + HybridRetriever (`retrieval/fusion.py`)

**Files:**
- Create: `src/belge_gozu/retrieval/fusion.py`, `tests/retrieval/test_fusion.py`
- Modify: `src/belge_gozu/config.py`
  (`retrieval_mode: Literal["visual-only","hybrid-production"] = "visual-only"`,
  `metadata_boost: float = 0.1`, `rrf_k: int = 60`, `fusion_channel_k: int = 100`),
  `src/belge_gozu/app/main.py` (mod'a göre kurulum)

**Interfaces:**
- Consumes: T4-T7 kanalları, P0 `ExhaustiveBinaryRetriever`, T3 `article_page_map`.
- Produces:

```python
def rrf_fuse(ranked_lists: list[list[str]], k_rrf: int = 60,
             boost_ids: set[str] | None = None, boost: float = 0.0) -> list[tuple[str, float]]
    # RRFscore(d) = Σ_listeler 1/(k_rrf + rank_d)  (Cormack 2009, k=60)
    # boost_ids'teki d için skor *= (1 + boost)  — SOFT boost; asla filtre değil

class HybridRetriever:
    """Kanal orkestrasyonu. mode='visual-only' -> yalnız visual kanal (P0 davranışı).
    mode='hybrid-production' -> varyantlar × {bm25-page, bm25-article, phrase, dense,
    visual} kanal sıralamaları -> sayfa düzeyine indirger (article hit'leri page_ids'e
    açılır) -> rrf_fuse -> dedup -> top-k."""
    def __init__(self, visual: ExhaustiveBinaryRetriever,
                 bm25_pages: BM25Index | None, bm25_articles: BM25Index | None,
                 dense: DenseIndex | None, embedder: TextEmbedder | None,
                 aliases: pd.DataFrame, article_pages: dict[str, list[str]],
                 doc_names: dict[str, str], settings: Settings): ...
    def channel_rankings(self, query: str) -> dict[str, list[str]]
        # kanal adı -> sayfa sıralaması (teşhis/T9 için public)
    def search(self, query: str, k: int = 5) -> list[PageHit]
        # PageHit.score = RRF skoru (UI etiketi 'birleşik sıra skoru')
```

- [ ] **Step 1: Başarısız testleri yaz**

```python
# tests/retrieval/test_fusion.py
from belge_gozu.retrieval.fusion import rrf_fuse


def test_rrf_formula():
    fused = rrf_fuse([["a", "b"], ["b", "a"]], k_rrf=60)
    # a: 1/61+1/62 ; b: 1/62+1/61 -> eşit; deterministik kırılım (ad sırası)
    assert {d for d, _ in fused[:2]} == {"a", "b"}
    assert abs(fused[0][1] - (1 / 61 + 1 / 62)) < 1e-12


def test_rrf_soft_boost_reorders_but_never_filters():
    fused = rrf_fuse([["a", "b", "c"]], boost_ids={"c"}, boost=5.0)
    assert {d for d, _ in fused} == {"a", "b", "c"}       # kimse elenmedi
    assert fused[0][0] == "c"                              # güçlü boost öne aldı


def test_hybrid_union_recovers_channel_miss(tiny_hybrid):
    """visual kanalın kaçırdığı sayfayı BM25 kanalı union'a sokar."""
    hy = tiny_hybrid  # fixture: 3 sayfalık stub kanallar (aşağıda)
    ranked = [h.page_id for h in hy.search("kira bedeli endeks", k=3)]
    assert "k6098:120" in ranked
```

`tiny_hybrid` fixture'ı (aynı dosyada): stub visual (sabit sıralama döndüren sahte
`ExhaustiveBinaryRetriever` yüzeyi: `encoder.encode_query` + `score_all` yerine
`channel_rankings` girdisi olarak kullanılacak `search_embedding` stub'ı), gerçek
`BM25Index.build` (T6), dense=None. Amaç: füzyon mantığının kanal kaybını telafisi.

- [ ] **Step 2: RED → fusion.py yaz → GREEN** — Run: `uv run pytest tests/retrieval/test_fusion.py -v`
- [ ] **Step 3: app entegrasyonu** — `create_app` `retrieval_mode`'a göre
  `ExhaustiveBinaryRetriever` (visual-only) veya `HybridRetriever` kurar; hybrid
  artefaktları (`page_texts`, `articles`, bm25/dense dizinleri) yoksa açık hata
  mesajıyla fail-fast ("önce `text build`/`articles build`/`bm25 build` koşun").
  Telemetri: `with stage("fusion")` + kanal süreleri `detail["retrieval"]`'e.
- [ ] **Step 4: Full regression + Commit** — `uv run pytest -q -m "not slow" && make lint`
  — `feat(fusion): RRF hybrid retriever with soft metadata boost and dual modes`

---

### Task 9: Kanal-düzeyi teşhis + recall gate ölçümü (`bench/harness.py` genişletmesi)

**Files:**
- Modify: `src/belge_gozu/bench/harness.py`, `tests/bench/test_harness.py`

**Interfaces:**
- Produces: `HybridDiagnosticAdapter(hybrid: HybridRetriever, record_top: int = 200)`
  — `DiagnosticPipeline` uygular; `channel_rankings`'ten kanal başına `StageRecord`
  (stage=`"kanal:bm25-page"` vb.), union için `stage="union"`, füzyon sonrası
  `stage="final"`; `candidate_survival` union'a göre.
- Consumes: P0 T8 tipleri (`StageRecord/QuestionDiagnostic/EvalReport` — birebir aynı).

- [ ] **Step 1: Başarısız test yaz** — sahte `HybridRetriever` (stub `channel_rankings`)
  ile adapter'ın kanal + union + final kayıtları ürettiğini, union survival'ın kanal
  kaçırmalarını doğru yansıttığını doğrula (gold yalnız bm25 kanalında → union'da True).
- [ ] **Step 2: RED → uygula → GREEN** — Run: `uv run pytest tests/bench/test_harness.py -v`
- [ ] **Step 3: Recall gate koşumu (runbook)** — Run:
  `uv run belge-gozu bench run --bench data/bench/bench_v2.jsonl --split dev --pipeline hybrid --out data/bench/results/<run_id>.json`
  (`bench run`'a `--split {dev,test,all}` + `--pipeline hybrid` seçenekleri bu adımda
  eklenir; split ataması T12 `question_split` ile). Union Recall@50 overall + kritik
  dilimler okunur. **Karar: G1.1/G1.2 dev'de sağlanmadan T10 reranker'ın flag'i
  açılamaz** (ilke 2-3, 16).
- [ ] **Step 4: Full regression + Commit** — `uv run pytest -q -m "not slow" && make lint`
  — `feat(bench): per-channel diagnostics and candidate-union recall gate`

---

### Task 10: Text cross-encoder reranker + pointwise-VLM risk analizi (`retrieval/rerank.py`)

**Files:**
- Create: `src/belge_gozu/retrieval/rerank.py`, `tests/retrieval/test_rerank.py`
- Modify: `src/belge_gozu/config.py` (`rerank_enabled: bool = False`,
  `rerank_model: str = ""`, `rerank_pool: int = 50`),
  `src/belge_gozu/retrieval/fusion.py` (`HybridRetriever.__init__`'e
  `reranker: "Reranker | None" = None`; `search` sonunda havuz `rerank_pool` ise uygular)

**Interfaces:**
- Produces:

```python
class Candidate(BaseModel):
    page_id: str
    article_id: str | None
    text: str            # madde metni varsa o, yoksa sayfa metni (yoksa "")
    fused_score: float

class Reranker(Protocol):
    def rerank(self, question: str, candidates: list[Candidate], k: int) -> list[Candidate]

class CrossEncoderReranker:
    """transformers AutoModelForSequenceClassification (lazy); (soru, metin) çiftleri
    batch'lenir; metinsiz aday (text=="") SIRASINI KORUR (görsel-kanıt sayfası metin
    kanalından cezalandırılmaz — fused sıra korunarak CE skorlular arasına yerleşir)."""
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 8,
                 max_length: int = 512): ...
```

**Pointwise VLM "0-10" reranker risk analizi (Plan 2 T6'nın reddi — bu task'in
dokümantasyon bölümü olarak `rerank.py` modül docstring'ine yazılır):**
(a) maliyet: aday başına 1 VLM çağrısı × havuz 50 = sorgu başına 50 çağrı — Gemini
kotasıyla (≈20/gün) tek sorgu bile koşamaz; (b) skor güvenilirliği: tek sayı üretimi
kalibre değildir, bağlar (tie) yoğundur, sıcaklık/istem değişimine hassastır;
(c) alternatifler: text-CE (burada), listwise tek-çağrı VLM (aday görüntüleri tek
istemde — bağlam sınırı ve sıra önyargısı riskiyle, ancak deney olarak), classification-
head görsel model (eğitim verisi gerektirir — P2 fine-tuning kapısına bağlı). Görsel
sinyal kaybolmaz: visual kanal sırası RRF üyesi olarak kalır ve `requires_visual`
diliminde CE'siz/CE'li fark ayrıca raporlanır.

- [ ] **Step 1: Başarısız testleri yaz** — stub CE modeliyle: (a) skorlara göre yeniden
  sıralama + k kesimi; (b) `text==""` adayların göreli sırasının korunduğu;
  (c) model exception'ında orijinal sıranın döndüğü (graceful, log'lu);
  (d) `HybridRetriever`'a stub reranker verilince `search` çıktısının değiştiği ve
  `rerank_enabled=False` iken hiç çağrılmadığı.
- [ ] **Step 2: RED → rerank.py + fusion kancası yaz → GREEN** — Run:
  `uv run pytest tests/retrieval/test_rerank.py -v`
- [ ] **Step 3: CE model seçimi + G1 koşumu (runbook)** — adaylar:
  `BAAI/bge-reranker-v2-m3`, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (+ TR-TEB/
  RAGTurk işaret ettiği 1 aday). Run: `scripts/ce_model_select.py` — dev split'te
  rerank'li Recall@5/MRR/nDCG@5 + sorgu başına CPU süresi. Karar: kazanç CI'sı pozitif
  VE bütçeye sığan model; `rerank_model` + `rerank_enabled=True` ancak T9 recall
  gate'i geçilmişse. Sayılar p1-gate taslağına.
- [ ] **Step 4: Full regression + Commit** — `uv run pytest -q -m "not slow" && make lint`
  — `feat(rerank): recall-gated text cross-encoder rerank stage`

---

### Task 11: Evidence pack + komşu sayfa genişletme (`retrieval/evidence.py`)

**Files:**
- Create: `src/belge_gozu/retrieval/evidence.py`, `tests/retrieval/test_evidence.py`
- Modify: `src/belge_gozu/retrieval/fusion.py` (`HybridRetriever.retrieve_evidence`),
  `src/belge_gozu/answer/base.py` (`AskService`'in pack'i answerer'a taşıması —
  imza geriye uyumlu)

**Interfaces:**
- Produces:

```python
class EvidenceUnit(BaseModel):
    article_id: str | None
    page_ids: list[str]           # madde sayfaları + (varsa) komşu sayfa genişletmesi
    text: str | None              # madde/sayfa metni; visual-only modda None
    image_paths: list[str]
    retrieval_score: float        # fused (veya rerank sonrası) skor
    channels: list[str]           # bu birimi getiren kanallar

class EvidencePack(BaseModel):
    question: str
    facets: QueryFacets
    units: list[EvidenceUnit]     # en çok k birim

def build_evidence_pack(question: str, facets: QueryFacets, hits: list[PageHit],
                        page_articles: dict[str, list[str]],
                        articles_df: pd.DataFrame, meta: pd.DataFrame,
                        channels_of: dict[str, list[str]],
                        expand_adjacent: bool = True) -> EvidencePack
    # sayfa hit'leri madde çatısı altında gruplanır (page_articles); maddesiz sayfa
    # (tarihî tarama) tek-sayfalık birim olur. Komşu genişletme: YALNIZ hit alınmış
    # sayfanın maddesi sayfa sınırında kesiliyorsa maddenin diğer sayfaları eklenir
    # (ilke: genişletme doğru sayfa bulunduktan SONRA; korpus-çapında komşuluk değil).
```

- `HybridRetriever.retrieve_evidence(query: str, k: int = 5) -> EvidencePack` —
  `search` + `build_evidence_pack`.
- `AskService`: retriever'da `retrieve_evidence` varsa pack kurulur ve
  `answerer.answer(question, pages, image_loader)` çağrısındaki `pages` pack
  birimlerinin sayfalarından türetilir (mevcut `Answerer` protokolü DEĞİŞMEZ —
  P2'de pack'i doğrudan tüketen verifier gelir; P1'de pack `detail` telemetrisine ve
  `/ask` yanıtına `evidence` alanı olarak eklenir).

- [ ] **Step 1: Başarısız testleri yaz** — sahte articles/meta çerçeveleriyle:
  (a) aynı maddeye ait iki hit tek birimde toplanır; (b) sayfa sınırında kesilen madde
  `expand_adjacent=True` ile eksik sayfalarını alır, `False` ile almaz; (c) maddesiz
  sayfa tek-sayfa birim; (d) `channels` alanı `channels_of`'tan taşınır.
- [ ] **Step 2: RED → evidence.py yaz → GREEN** — Run:
  `uv run pytest tests/retrieval/test_evidence.py -v`
- [ ] **Step 3: AskService/app entegrasyonu + testler** — `/ask` yanıtında `evidence`
  (pack özeti) alanı; mevcut app testleri kırılmaz (alan ekleme geriye uyumlu).
- [ ] **Step 4: Full regression + Commit** — `uv run pytest -q -m "not slow" && make lint`
  — `feat(evidence): article-level evidence pack with post-hit adjacent expansion`

---

### Task 12: Tam benchmark v2 (120 + 30) + law-grouped split (runbook, İNSAN kapılı)

**Files:**
- Create: `data/bench/bench_v2.jsonl`, doldurulmuş `data/bench/splits_v1.json`

- [ ] **Step 1 (ajan taslağı):** retrieval_eval'yi çekirdek alarak ~140 answerable + ~35
  unanswerable taslak üret (şema P0 T6; `verification_status="draft"`). Dilim hedefleri
  (verified sonrası): `dogrudan-madde` 25, `paraphrase` 25, `madde-numarali` 12,
  `ayni-kanun-hard-negative` 12, `capraz-kanun-terim` 10, `tablo-layout` 12,
  `tarihi-tarama` 12, `belirsiz-coklu-dayanak` 6, `multi-hop` 6 (= ~120 answerable);
  `korpus-disi` 12, `eksik-kanit` 8, `anlamsiz-ood` 10 (= 30 unanswerable).
  Unanswerable üretiminde UAEval4RAG uyarlaması: korpusa YAKIN ama cevabı korpusta
  olmayan sorular (`korpus-disi`), var olmayan madde/mülga hüküm referansı
  (`eksik-kanit`), anlamsız/gibberish (`anlamsiz-ood`).
- [ ] **Step 2 (İNSAN — bloklayıcı):** kullanıcı doğrulaması; hedef ≥120 verified
  answerable + ≥30 verified unanswerable. Gerçek kullanıcı sorguları (telemetri
  `events.query_text`) varsa `source_type="insan"` olarak tercih edilir.
- [ ] **Step 3: Split doldurma:** `splits_v1.json` — kanunlar dev/test'e ~%60/%40
  bölünür; kural: (a) iki hedef sorgunun kanunu `k4721` **dev**'de (geliştirme
  regression'ı oldukları için); (b) her kritik dilimde test tarafında ≥5 soru kalmalı;
  (c) `ayni-kanun-hard-negative` çiftleri aynı tarafta. Doğrulama komutu — Run:
  `uv run python -c "from pathlib import Path; from belge_gozu.bench.dataset import load_bench, load_splits, question_split; qs=load_bench(Path('data/bench/bench_v2.jsonl')); sp=load_splits(Path('data/bench/splits_v1.json')); from collections import Counter; print(Counter((question_split(q,sp), q.slice) for q in qs))"`
- [ ] **Step 4: Coverage kontrolü** — Run: `bench run --bench data/bench/bench_v2.jsonl
  --split all --pipeline hybrid` → `missing_gold_pages == []`.
- [ ] **Step 5: Commit** — `data: belge-gozu-bench v2 (120+30, human-verified, law-grouped splits)`

---

### Task 13: HF Space bütçeleri + P1 ablasyonları + kapı raporu (runbook)

**Files:**
- Create: `docs/research/findings/2026-XX-XX-p1-gate.md`
- Modify: `README.md` (sonuç tabloları), `src/belge_gozu/config.py` (kanıtlı flag'ler)

- [ ] **Step 1: Ablasyon koşumları (dev split):** F1 (bm25-only / dense-only /
  visual-only / bm25+dense / text+visual RRF), E1 (original-only / tek-rewrite /
  original+multi-variant; LLM-rewrite kolu `scripts/e1_llm_rewrite.py` önbellekli +
  `--yes-burn-quota`), F2 (soft-boost aç/kapa), F3 (OCR aç/kapa — tarihî dilim),
  G1 (rerank aç/kapa). Her satır: değişen tek faktör + Recall@k/MRR/nDCG@5 +
  bootstrap CI + latency. Komut deseni: `uv run belge-gozu bench run --bench
  data/bench/bench_v2.jsonl --split dev --pipeline <cfg> --out data/bench/results/<run_id>.json`
  (+ config env'leri).
- [ ] **Step 2: Operasyonel bütçe ölçümü:** indeks/artefakt boyutları (`du -sh`),
  peak RSS (`/metrics` process_resident_memory), p50/p95 (`scripts/loadgen.py`
  `/search`), cold-start süresi (serve başlatma logu). Hedef karşılaştırması master
  §10 ile.
- [ ] **Step 3: Final test koşumu (bir kez):** kilitli konfigürasyonla
  `--split test` → G1.1-G1.4 sayıları. Sorgu A top-5 kontrolü:
  `tests/retrieval/test_semantic_retrieval_eval.py`'ye `-m slow`
  `test_long_query_gold_in_top5_hybrid` eklenir (hybrid mod; G1.4 kilidi) ve koşulur.
- [ ] **Step 4: Kapı raporu + flag kararları:** `p1-gate.md` — master §5 G1.1-G1.7
  satır satır sayılarla; kazanç kanıtlayan flag'ler default açılır (her biri ayrı,
  sayı-referanslı commit), kanıtlamayanlar kapalı kalır ve negatif sonuç README'ye
  yazılır. `retrieval_mode` default'u ancak TÜM G1 satırları PASS ise
  `hybrid-production` olur.
- [ ] **Step 5: Commit** — `docs: P1 gate report, ablation tables, budget measurements`

---

## P1 Tamamlanma Kapısı (go/no-go)

Master §5 G1.1-G1.7. Ek kurallar: T9 recall gate dev'de sağlanmadan reranker default
açılamaz; test split faz boyunca yalnız T13 Step 3'te bir kez kullanılır; G1 FAIL ise
`retrieval_mode` visual-only kalır, eksikler p1-gate raporunda planla listelenir ve
P2 başlamaz.

## Self-Review (yazar kontrolü)

1. **Spec kapsaması:** born-digital extraction (T1), quality detector (T1), OCR fallback
   + motor benchmark (T2), madde/paragraf extraction + page mapping (T3), aliases/law
   number/metadata index (T4), BM25/phrase (T6), multilingual dense (T7; sparse yalnız
   BGE-M3 seçilirse onun sparse modu F1'e ek satır olarak girer — karar T7 Step 3'te),
   visual-only kanal (P0 devralımı, T8 mod), original+multi-variant (T5), RRF (T8),
   dedup (T8), soft boost (T8), union (T8), text CE reranker (T10), visual sinyalin
   korunması + pointwise VLM risk analizi (T10), recall gate (T9), adjacent expansion
   (T11), article context pack (T11), iki modun ayrı benchmark'ı (T13), flag/rollback
   (config satırları), Space bütçeleri (T13).
2. **Placeholder taraması:** model adı boş default'ları (`dense_model`, `rerank_model`)
   TBD değil — seçim koşumları (T7 Step 3, T10 Step 3) karar kuralı + komutla tanımlı.
3. **Tip tutarlılığı:** `QueryFacets/QueryVariant` (T4/T5) → T8/T11; `BM25Index/DenseIndex`
   → T8; `Candidate/Reranker` (T10) → fusion kancası; `EvidencePack/EvidenceUnit` (T11)
   → P2 sözleşmesi (master §3) ile birebir; P0 `StageRecord/EvalReport` değişmeden
   yeniden kullanılır (T9).
4. **Bağımlılık:** T1→(T2,T3)→(T6,T7); T4→T5→T8; T8→(T9,T10,T11); T12 insan-paralel;
   T13 en son. P0 dışı hiçbir tanımsız interface tüketilmiyor.
