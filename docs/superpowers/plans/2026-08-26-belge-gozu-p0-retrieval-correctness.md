# Belge-Gözü P0 — Retrieval Correctness ve Ölçülebilirlik Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrieval'ı ölçülebilir ve doğru yapmak: bozuk mean-sign Stage-1'i üretim
yolundan kaldırıp exhaustive binary MaxSim'e geçmek; padding/mask, processor formatı,
index manifest ve fail-fast'i düzeltmek; canary benchmark + teşhis harness + binary/float
oracle'ları kurmak; kuantizasyon kaybını sayılandırmak; README/UI'ı dürüstleştirmek.

**Architecture:** Yeni `bench/` paketi (dataset/metrics/harness/oracle), `index/manifest.py`,
maskeli+formatlı `ColSmolEncoder`, `ExhaustiveBinaryRetriever`. Mevcut TwoStage kod
ablasyon-only konuma iner (silinmez, flag'le seçilir). Telemetri şeması genişler,
değişmez ilkeleri korunur.

**Tech Stack:** Mevcut stack (numpy, pydantic, typer, colpali-engine 0.3.18 pinli).
Yeni zorunlu bağımlılık YOK; A3 karşılaştırması `uv run --with sentence-transformers`
ile geçici ortamda koşar.

**Spec:** `docs/superpowers/specs/2026-08-26-belge-gozu-rag-quality-v2-design.md`
**Master:** `docs/superpowers/plans/2026-08-26-belge-gozu-rag-quality-master.md`

## Global Constraints

- CI testleri ağ/GPU/gerçek model kullanmaz (`-m "not slow"`); ruff line-length 100 +
  pyright basic temiz; her task kendi commit'iyle biter (bu planlama turunda commit YOK —
  commit'ler uygulama onayından sonra atılır).
- **Korpus donduruldu:** `data/images/`, `data/pdf/`, `data/manifest/`, `meta.parquet`
  bu plan boyunca değişmez (checksum manifest'te). Yeni indeksler YENİ dizinlere yazılır
  (`data/index-<format>-<quant>/`); mevcut `data/index/` üzerine yazılmaz.
- Gemini kotası hiçbir P0 adımında harcanmaz (P0 tamamen retrieval).
- Doğrulanmış referans sayılar (2026-08-26 koşumu): Sorgu A
  ("Türk Medeni Kanunu'na göre yerleşim yeri nasıl tanımlanır?") gold `k4721:4`
  binary skor 54.45, Stage-1 sırası 3127, exhaustive sırası 1576; Sorgu B
  ("Yerleşim yeri nedir?") skor 68.57, Stage-1 1768, exhaustive 2; Stage-1∩exhaustive
  top-200 = %19.0 / %11.5; indekste 3960 all-zero token satırı (15 sayfa); vektörize
  exhaustive ~1.2 s (M4 Pro). Bu sayılar T14 baseline raporunun çekirdeğidir.
- Kapı: master plan §5 G0.1–G0.9. G0 raporu commit'lenmeden P1'in default entegrasyonu
  başlamaz.

## File Structure

```
src/belge_gozu/
  index/manifest.py        # IndexManifest, QueryFormat, RenderConfig, checksum, read/write   [T1]
  index/encode.py          # maskeli _run, query_format'lı encode_query, model_revision       [T2]
  index/store.py           # padding reddi, manifest'li save/load                             [T3]
  index/quantize.py        # f16 master -> sign-1bit / int8 türetme                           [T12]
  index/compat.py          # check_compatibility, IndexCompatibilityError                     [T4]
  retrieval/core.py        # + ExhaustiveBinaryRetriever (TwoStage kalır, ablasyon-only)      [T5]
  bench/__init__.py
  bench/dataset.py         # BenchQuestion, load_bench, load_splits                           [T6]
  bench/metrics.py         # recall_at_k, mrr, ndcg_at_k, bootstrap_ci                        [T7]
  bench/harness.py         # StageRecord, QuestionDiagnostic, MetricBlock, EvalReport,
                           # DiagnosticPipeline adapter'ları, run_retrieval_eval              [T8]
  bench/oracle.py          # FloatIndex, native_float_ranks, oracle_gap                       [T9]
  cli.py                   # bench run/oracle, index build --precision/--query-format,
                           # index write-manifest --legacy                                    [T5,T8,T9]
  app/main.py              # pipeline seçimi + compat fail-fast + telemetri alanları          [T4,T5,T13]
  config.py                # retrieval_pipeline, query_format_id, allow_index_mismatch        [T4,T5]
data/bench/canary_v1.jsonl # 30-50 insan-doğrulamalı soru                                     [T10]
data/bench/splits_v1.json  # law-grouped split iskeleti                                       [T6]
tests/index/test_manifest.py, tests/index/test_encode_mask.py, tests/index/test_quantize.py
tests/retrieval/test_exhaustive.py, tests/retrieval/test_semantic_canary.py (slow)
tests/bench/test_dataset.py, tests/bench/test_metrics.py, tests/bench/test_harness.py,
tests/bench/test_oracle.py, tests/app/test_compat.py
docs/research/findings/2026-XX-XX-p0-baseline.md, 2026-XX-XX-p0-gate.md               [T14]
```

Sorumluluklar: `manifest.py` yalnız künye modeli+IO; `compat.py` yalnız karşılaştırma;
`quantize.py` yalnız f16→derived dönüşümler; `harness.py` yalnız koşum/rapor (metrik
hesabı `metrics.py`'de); `oracle.py` yalnız float indeks + oracle sıralamaları.

---

### Task 1: Index manifest modeli (`index/manifest.py`)

**Files:**
- Create: `src/belge_gozu/index/manifest.py`, `tests/index/test_manifest.py`

**Interfaces:**
- Produces:
  - `QueryFormat(BaseModel)`: `format_id: str`, `prefix: str`, `suffix_token: str`,
    `n_suffix: int`, `trailing_newline: bool`; `render(text: str) -> str` metodu.
  - Sabitler: `CPE_0_3_18 = QueryFormat(format_id="cpe-0.3.18", prefix="",
    suffix_token="<end_of_utterance>", n_suffix=10, trailing_newline=False)`;
    `TRAIN_COMPAT_V1 = QueryFormat(format_id="train-compat-v1", prefix="Query: ",
    suffix_token="<end_of_utterance>", n_suffix=10, trailing_newline=True)`
    (T11 Step 1, ST config'e karşı doğrular ve gerekirse bu sabiti düzeltir).
  - `RenderConfig(BaseModel)`: `dpi: int = 150`, `format: str = "webp"`, `quality: int = 80`
  - `IndexManifest(BaseModel)`: `schema_version: int = 1`, `model_name: str`,
    `model_revision: str`, `engine_versions: dict[str, str]`, `query_format: QueryFormat`,
    `doc_prompt_sha256: str`, `quantization: str`, `mask_policy: str`,
    `render: RenderConfig`, `corpus_checksum: str`, `n_pages: int`, `n_tokens: int`,
    `built_at: str`, `git_commit: str`
  - `corpus_checksum(index_dir: Path) -> str` — sha256(`page_ids.json` bayt +
    `meta.parquet` bayt)
  - `write_manifest(dir: Path, m: IndexManifest) -> None` (`manifest.json`),
    `read_manifest(dir: Path) -> IndexManifest | None` (dosya yoksa None)

- [ ] **Step 1: Başarısız testleri yaz**

```python
# tests/index/test_manifest.py
import json
from pathlib import Path

from belge_gozu.index.manifest import (
    CPE_0_3_18,
    TRAIN_COMPAT_V1,
    IndexManifest,
    QueryFormat,
    RenderConfig,
    corpus_checksum,
    read_manifest,
    write_manifest,
)


def make_manifest(**over) -> IndexManifest:
    base = dict(
        model_name="vidore/colSmol-500M",
        model_revision="abc123",
        engine_versions={"colpali-engine": "0.3.18", "transformers": "5.15.1", "torch": "2.13.0"},
        query_format=CPE_0_3_18,
        doc_prompt_sha256="d" * 64,
        quantization="sign-1bit",
        mask_policy="drop-padding",
        render=RenderConfig(),
        corpus_checksum="c" * 64,
        n_pages=3,
        n_tokens=24,
        built_at="2026-08-26T00:00:00+00:00",
        git_commit="deadbeef",
    )
    base.update(over)
    return IndexManifest(**base)


def test_query_format_render():
    assert CPE_0_3_18.render("soru") == "soru" + "<end_of_utterance>" * 10
    assert TRAIN_COMPAT_V1.render("soru") == "Query: soru" + "<end_of_utterance>" * 10 + "\n"


def test_roundtrip(tmp_path: Path):
    m = make_manifest()
    write_manifest(tmp_path, m)
    m2 = read_manifest(tmp_path)
    assert m2 == m
    assert json.loads((tmp_path / "manifest.json").read_text())["schema_version"] == 1


def test_read_missing_returns_none(tmp_path: Path):
    assert read_manifest(tmp_path) is None


def test_corpus_checksum_changes_with_content(tmp_path: Path):
    (tmp_path / "page_ids.json").write_text('["a:1"]')
    (tmp_path / "meta.parquet").write_bytes(b"x")
    c1 = corpus_checksum(tmp_path)
    (tmp_path / "meta.parquet").write_bytes(b"y")
    assert corpus_checksum(tmp_path) != c1
```

- [ ] **Step 2: RED gör** — Run: `uv run pytest tests/index/test_manifest.py -v` —
  Expected: FAIL `ModuleNotFoundError: belge_gozu.index.manifest`
- [ ] **Step 3: manifest.py yaz**

```python
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel


class QueryFormat(BaseModel):
    format_id: str
    prefix: str
    suffix_token: str
    n_suffix: int
    trailing_newline: bool

    def render(self, text: str) -> str:
        out = self.prefix + text + self.suffix_token * self.n_suffix
        return out + "\n" if self.trailing_newline else out


CPE_0_3_18 = QueryFormat(
    format_id="cpe-0.3.18", prefix="", suffix_token="<end_of_utterance>",
    n_suffix=10, trailing_newline=False,
)
# Model kartı: checkpoint "Query: " prefix + sondaki newline ile eğitildi
# (newline 0.3.11'de, prefix 0.3.13'te düştü). Kesin şablon T11'de
# config_sentence_transformers.json'a karşı doğrulanır; sapma varsa bu sabit
# orada düzeltilir ve test_query_format_render güncellenir.
TRAIN_COMPAT_V1 = QueryFormat(
    format_id="train-compat-v1", prefix="Query: ", suffix_token="<end_of_utterance>",
    n_suffix=10, trailing_newline=True,
)


class RenderConfig(BaseModel):
    dpi: int = 150
    format: str = "webp"
    quality: int = 80


class IndexManifest(BaseModel):
    schema_version: int = 1
    model_name: str
    model_revision: str
    engine_versions: dict[str, str]
    query_format: QueryFormat
    doc_prompt_sha256: str
    quantization: str
    mask_policy: str
    render: RenderConfig
    corpus_checksum: str
    n_pages: int
    n_tokens: int
    built_at: str
    git_commit: str


def corpus_checksum(index_dir: Path) -> str:
    h = hashlib.sha256()
    h.update((index_dir / "page_ids.json").read_bytes())
    h.update((index_dir / "meta.parquet").read_bytes())
    return h.hexdigest()


def write_manifest(dir: Path, m: IndexManifest) -> None:
    (dir / "manifest.json").write_text(m.model_dump_json(indent=1), encoding="utf-8")


def read_manifest(dir: Path) -> IndexManifest | None:
    p = dir / "manifest.json"
    if not p.exists():
        return None
    return IndexManifest.model_validate_json(p.read_text(encoding="utf-8"))
```

- [ ] **Step 4: GREEN gör** — Run: `uv run pytest tests/index/test_manifest.py -v` — Expected: PASS
- [ ] **Step 5: Full regression** — Run: `uv run pytest -q -m "not slow" && make lint` — Expected: temiz
- [ ] **Step 6: Commit** — `feat(index): index manifest model with query-format contract`

---

### Task 2: Maskeli ve formatlı encoder (`index/encode.py`)

**Files:**
- Modify: `src/belge_gozu/index/encode.py`
- Create: `tests/index/test_encode_mask.py`

**Interfaces:**
- Consumes: `QueryFormat`, `CPE_0_3_18` (T1)
- Produces: `ColSmolEncoder.__init__(model_name: str, device: str = "auto",
  query_format: QueryFormat | None = None)`; `encode_query` format'ı uygular;
  `_run` attention mask ile padding'i KIRPAR (dönen her embedding yalnız gerçek
  token satırları içerir); `self.model_revision: str` (HF cache `_commit_hash`,
  yoksa `"unknown"`); `self.doc_prompt: str` (processor `visual_prompt_prefix`) ve
  `self.doc_prompt_sha256: str`. `FakeEncoder` değişmez (maskeleme gerektirmez);
  `Encoder` protokolü değişmez.

- [ ] **Step 1: Başarısız birim testleri yaz** (gerçek model YOK — sahte model/processor stub'ı)

```python
# tests/index/test_encode_mask.py
import numpy as np

from belge_gozu.index.manifest import CPE_0_3_18, TRAIN_COMPAT_V1


class FakeTorchLike:
    """_run'ın maske kırpma sözleşmesini gerçek model olmadan sınamak için
    ColSmolEncoder._trim_by_mask saf fonksiyonu test edilir."""


def test_trim_by_mask_drops_padding_rows():
    from belge_gozu.index.encode import trim_by_mask

    emb = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    mask = np.array([[1, 1, 1, 1], [0, 0, 1, 1]], dtype=np.int64)  # sol-pad
    out = trim_by_mask(emb, mask)
    assert [o.shape for o in out] == [(4, 3), (2, 3)]
    np.testing.assert_array_equal(out[1], emb[1, 2:])


def test_query_format_render_used(monkeypatch):
    """encode_query, processor'a QueryFormat.render çıktısını vermeli."""
    from belge_gozu.index import encode as enc_mod

    captured = {}

    class StubSelf:
        query_format = TRAIN_COMPAT_V1

        class processor:  # noqa: N801
            @staticmethod
            def process_texts(texts):
                captured["texts"] = texts
                return {"batch": True}

        def _run(self, batch):
            return [np.zeros((1, 128), dtype=np.float32)]

    out = enc_mod.ColSmolEncoder.encode_query(StubSelf(), "yerleşim yeri")
    assert captured["texts"] == [TRAIN_COMPAT_V1.render("yerleşim yeri")]
    assert out.shape == (1, 128)
```

- [ ] **Step 2: RED gör** — Run: `uv run pytest tests/index/test_encode_mask.py -v` —
  Expected: FAIL (`trim_by_mask` yok; `encode_query` render kullanmıyor)
- [ ] **Step 3: encode.py'yi güncelle** (mevcut yorumlar/kalıp korunur; değişen kısımlar)

```python
def trim_by_mask(emb: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """(B, L, D) embedding + (B, L) attention mask -> padding'siz [(l_i, D)].

    colpali add_model_family sözleşmesi padding embedding'lerini sıfırlar; bu
    sıfırlar dot-product MaxSim'de zararsızdır ama sign-binarizasyonda geçerli
    bit desenine dönüşür (v0 bug'ı: indekste 3960 all-zero satır). Çözüm:
    binarize ETMEDEN önce padding satırlarını at."""
    return [e[m.astype(bool)] for e, m in zip(emb, mask, strict=True)]


class ColSmolEncoder:
    def __init__(self, model_name: str, device: str = "auto",
                 query_format: QueryFormat | None = None):
        # ... mevcut model/processor kurulumunun tamamı aynen ...
        self.query_format = query_format or CPE_0_3_18
        self.model_revision = getattr(model.config, "_commit_hash", None) or "unknown"
        self.doc_prompt = self.processor.visual_prompt_prefix
        self.doc_prompt_sha256 = hashlib.sha256(self.doc_prompt.encode()).hexdigest()

    def _run(self, batch) -> list[np.ndarray]:
        import torch

        with torch.no_grad():
            out = self.model(**{k: v.to(self.device) for k, v in batch.items()})
        emb = out.cpu().float().numpy()
        mask = batch["attention_mask"].cpu().numpy()
        return trim_by_mask(emb, mask)

    def encode_query(self, text: str) -> np.ndarray:
        rendered = self.query_format.render(text)
        return self._run(self.processor.process_texts([rendered]))[0]
```

Not: `encode_query` artık `process_queries` DEĞİL `process_texts` çağırır — prefix/suffix
tamamen `QueryFormat`'tan gelir; kütüphanenin sessiz format değişimlerine bağımlılık biter.
`CPE_0_3_18.render` mevcut 0.3.18 `process_queries` çıktısıyla birebir aynı metni üretir
(dogrulama: T11 Step 1).

- [ ] **Step 4: GREEN gör** — Run: `uv run pytest tests/index/test_encode_mask.py tests/index/test_encode.py -v` — Expected: PASS
- [ ] **Step 5: Slow determinism testini ekle** — `tests/index/test_encode_mask.py`'ye:

```python
import pytest


@pytest.mark.slow
def test_batch_vs_single_sign_determinism():
    """Batch içinde (padding'li) ve tek başına encode edilen sayfa, maske
    kırpması sonrası SIGN düzeyinde aynı olmalı. Sapma varsa index build
    batch_size=1'e iner (karar kuralı; sonuç p0-baseline raporuna yazılır)."""
    from pathlib import Path

    from PIL import Image

    from belge_gozu.index.encode import ColSmolEncoder

    enc = ColSmolEncoder("vidore/colSmol-500M", "auto")
    root = Path("data")
    # k6098:134 v0 indeksinde padding satırı olan sayfalardan biri (bulgu 20)
    paths = ["images/k6098/0134.webp", "images/k4721/0004.webp", "images/rg1965a/0001.webp"]
    imgs = [Image.open(root / p).convert("RGB") for p in paths]
    batch_out = enc.encode_pages(imgs)                      # tek batch (karışık boyut)
    single_out = [enc.encode_pages([im])[0] for im in imgs]
    for b, s, p in zip(batch_out, single_out, paths, strict=True):
        assert b.shape == s.shape, f"{p}: sekans uzunluğu batch'e bağlı olmamalı"
        agree = float(((b > 0) == (s > 0)).mean())
        assert agree == 1.0, f"{p}: sign uyuşması {agree:.4f} < 1.0 -> build batch=1 kararı"
```

- [ ] **Step 6: Slow testi koş, sonucu kaydet** — Run:
  `uv run pytest tests/index/test_encode_mask.py -m slow -v` — Expected: PASS ise
  batch build kalır; FAIL ise `cli.py index build` batch_size=1 yapılır ve test
  mesajdaki karar uygulanır (her iki sonuç da p0-baseline raporuna işlenir).
- [ ] **Step 7: Full regression** — Run: `uv run pytest -q -m "not slow" && make lint`
- [ ] **Step 8: Commit** — `fix(encode): mask-trimmed embeddings + explicit query format contract`

---

### Task 3: PackedIndex v2 — padding reddi + manifest (`index/store.py`)

**Files:**
- Modify: `src/belge_gozu/index/store.py`, `tests/index/test_store.py`

**Interfaces:**
- Produces: `PackedIndex.build(page_ids, embs, manifest: IndexManifest | None = None)`
  — herhangi bir embedding'de all-zero satır varsa `ValueError` ("padding satırı
  sızmış: <page_id>"); `PackedIndex.manifest: IndexManifest | None` alanı;
  `save` manifest varsa `manifest.json` yazar; `load` manifest'i okur (yoksa None —
  v0 legacy indeks yüklenebilir kalır).

- [ ] **Step 1: Başarısız testleri yaz** — `tests/index/test_store.py`'ye ekle:

```python
def test_build_rejects_zero_rows():
    embs = [np.vstack([np.ones((2, 128), dtype=np.float32),
                       np.zeros((1, 128), dtype=np.float32)])]
    with pytest.raises(ValueError, match="padding satırı sızmış: p:1"):
        PackedIndex.build(["p:1"], embs)


def test_manifest_roundtrip(tmp_path: Path):
    from tests.index.test_manifest import make_manifest

    embs = [np.random.default_rng(0).standard_normal((4, 128)).astype(np.float32)]
    idx = PackedIndex.build(["p:1"], embs, manifest=make_manifest(n_pages=1, n_tokens=4))
    idx.save(tmp_path)
    loaded = PackedIndex.load(tmp_path, mmap=False)
    assert loaded.manifest is not None and loaded.manifest.n_pages == 1


def test_legacy_index_loads_without_manifest(tmp_path: Path):
    embs = [np.random.default_rng(0).standard_normal((4, 128)).astype(np.float32)]
    PackedIndex.build(["p:1"], embs).save(tmp_path)
    assert PackedIndex.load(tmp_path, mmap=False).manifest is None
```

- [ ] **Step 2: RED gör** — Run: `uv run pytest tests/index/test_store.py -v`
- [ ] **Step 3: store.py'yi güncelle**

```python
@dataclass
class PackedIndex:
    tokens: np.ndarray
    offsets: np.ndarray
    page_vecs: np.ndarray
    page_ids: list[str]
    manifest: IndexManifest | None = None

    @classmethod
    def build(cls, page_ids, embs, manifest: IndexManifest | None = None) -> "PackedIndex":
        # ... mevcut doğrulamalar aynen ...
        for pid, e in zip(page_ids, embs, strict=True):
            if e.shape[0] == 0:
                raise ValueError(f"sıfır token'lı sayfa: {pid}")
            if (np.abs(e).sum(axis=1) == 0).any():
                raise ValueError(f"padding satırı sızmış: {pid}")
        # ... mevcut packing aynen ...
        return cls(np.vstack(packed), offsets, page_vecs, list(page_ids), manifest)

    def save(self, dir: Path) -> None:
        # ... mevcut kayıtlar aynen ...
        if self.manifest is not None:
            write_manifest(dir, self.manifest)

    @classmethod
    def load(cls, dir: Path, mmap: bool = True) -> "PackedIndex":
        # ... mevcut yüklemeler aynen + manifest=read_manifest(dir) ...
```

- [ ] **Step 4: GREEN + full regression** — Run:
  `uv run pytest tests/index -v && uv run pytest -q -m "not slow" && make lint`
- [ ] **Step 5: Commit** — `feat(store): reject leaked padding rows, carry index manifest`

---

### Task 4: Serve-time uyumluluk kontrolü (`index/compat.py` + app + CLI)

**Files:**
- Create: `src/belge_gozu/index/compat.py`, `tests/app/test_compat.py`
- Modify: `src/belge_gozu/config.py` (`allow_index_mismatch: bool = False`),
  `src/belge_gozu/app/main.py` (create_app başında kontrol),
  `src/belge_gozu/cli.py` (`index write-manifest --legacy`)

**Interfaces:**
- Produces: `IndexCompatibilityError(RuntimeError)`;
  `check_compatibility(manifest: IndexManifest | None, *, model_name: str,
  model_revision: str | None, query_format_id: str, index_dir: Path) -> list[str]`
  — uyumsuzluk açıklamaları listesi (boş = uyumlu). Kontroller: manifest varlığı,
  `model_name`, `model_revision` (encoder "unknown" veriyorsa atlanır, uyarı listelenir),
  `query_format.format_id`, `mask_policy == "drop-padding"`,
  `corpus_checksum(index_dir)` eşitliği.
  CLI `index write-manifest --legacy`: mevcut v0 indeksine bilinen v0 gerçekleriyle
  (`query_format=CPE_0_3_18`, `mask_policy="none"`, `quantization="sign-1bit"`) manifest
  damgalar — böylece fail-fast, "legacy olduğu bilinen" indeksle `mask_policy` uyumsuzluğu
  olarak DOĞRU sebebi söyler.
- Consumes: T1 manifest, T2 encoder attribute'ları.

- [ ] **Step 1: Başarısız testleri yaz**

```python
# tests/app/test_compat.py
from pathlib import Path

import pytest

from belge_gozu.index.compat import IndexCompatibilityError, check_compatibility
from tests.index.test_manifest import make_manifest


def test_missing_manifest_is_mismatch(tmp_path: Path):
    problems = check_compatibility(
        None, model_name="m", model_revision="r", query_format_id="cpe-0.3.18",
        index_dir=tmp_path,
    )
    assert problems and "manifest" in problems[0]


def test_matching_manifest_ok(tmp_path: Path):
    (tmp_path / "page_ids.json").write_text('["a:1"]')
    (tmp_path / "meta.parquet").write_bytes(b"x")
    from belge_gozu.index.manifest import corpus_checksum

    m = make_manifest(corpus_checksum=corpus_checksum(tmp_path))
    assert check_compatibility(
        m, model_name="vidore/colSmol-500M", model_revision="abc123",
        query_format_id="cpe-0.3.18", index_dir=tmp_path,
    ) == []


def test_format_mismatch_reported(tmp_path: Path):
    (tmp_path / "page_ids.json").write_text('["a:1"]')
    (tmp_path / "meta.parquet").write_bytes(b"x")
    from belge_gozu.index.manifest import corpus_checksum

    m = make_manifest(corpus_checksum=corpus_checksum(tmp_path))
    problems = check_compatibility(
        m, model_name="vidore/colSmol-500M", model_revision="abc123",
        query_format_id="train-compat-v1", index_dir=tmp_path,
    )
    assert any("query_format" in p for p in problems)


def test_create_app_fails_fast_on_mismatch(tiny_corpus):
    """tiny_corpus indeksi manifest'siz -> create_app IndexCompatibilityError."""
    from belge_gozu.app.main import create_app
    from belge_gozu.config import Settings

    data_dir, enc, _ = tiny_corpus
    settings = Settings(data_dir=data_dir, index_dir=data_dir / "index")
    with pytest.raises(IndexCompatibilityError):
        create_app(settings=settings, encoder=enc, answerer=object())


def test_mismatch_override(tiny_corpus):
    from belge_gozu.app.main import create_app
    from belge_gozu.config import Settings

    data_dir, enc, _ = tiny_corpus
    settings = Settings(
        data_dir=data_dir, index_dir=data_dir / "index", allow_index_mismatch=True,
        min_score_threshold=-1e9,
    )
    app = create_app(settings=settings, encoder=enc, answerer=object())
    assert app is not None
```

- [ ] **Step 2: RED gör** — Run: `uv run pytest tests/app/test_compat.py -v`
- [ ] **Step 3: compat.py + entegrasyon yaz**

```python
# src/belge_gozu/index/compat.py
from pathlib import Path

from belge_gozu.index.manifest import IndexManifest, corpus_checksum


class IndexCompatibilityError(RuntimeError):
    pass


def check_compatibility(
    manifest: IndexManifest | None, *, model_name: str, model_revision: str | None,
    query_format_id: str, index_dir: Path,
) -> list[str]:
    if manifest is None:
        return ["indekste manifest yok (v0 legacy?) — `belge-gozu index write-manifest --legacy`"]
    problems: list[str] = []
    if manifest.model_name != model_name:
        problems.append(f"model_name: indeks={manifest.model_name} serve={model_name}")
    if model_revision and model_revision != "unknown" and manifest.model_revision != model_revision:
        problems.append(f"model_revision: indeks={manifest.model_revision} serve={model_revision}")
    if manifest.query_format.format_id != query_format_id:
        problems.append(
            f"query_format: indeks={manifest.query_format.format_id} serve={query_format_id}"
        )
    if manifest.mask_policy != "drop-padding":
        problems.append(f"mask_policy: indeks={manifest.mask_policy} (drop-padding bekleniyor)")
    live = corpus_checksum(index_dir)
    if manifest.corpus_checksum != live:
        problems.append("corpus_checksum: indeks manifest'i ile meta/page_ids uyuşmuyor")
    return problems
```

`app/main.py` — `create_app` içinde indeks + encoder kurulduktan hemen sonra:

```python
from belge_gozu.index.compat import IndexCompatibilityError, check_compatibility

problems = check_compatibility(
    index.manifest,
    model_name=s.retriever_model,
    model_revision=getattr(encoder, "model_revision", None),
    query_format_id=getattr(encoder, "query_format", CPE_0_3_18).format_id,
    index_dir=s.index_dir,
)
if problems:
    msg = "indeks/serve uyumsuzluğu: " + "; ".join(problems)
    if not s.allow_index_mismatch:
        raise IndexCompatibilityError(msg)
    logger.warning("BG_ALLOW_INDEX_MISMATCH=true ile devam ediliyor — %s", msg)
```

`config.py`: `allow_index_mismatch: bool = False`.
`cli.py index write-manifest --legacy`: mevcut `data/index/` için manifest üretir
(model_name config'ten, model_revision="unknown", query_format=CPE_0_3_18,
mask_policy="none", quantization="sign-1bit", render=RenderConfig(),
corpus_checksum hesaplanır, n_pages/n_tokens dosyalardan, git_commit
`git rev-parse --short HEAD` çıktısından, built_at damgası).
Not: mevcut testlerdeki `create_app` çağrıları manifest'siz tiny_corpus kullanır —
`tests/conftest.py tiny_corpus` fixture'ı T3 sonrası `PackedIndex.build(...,
manifest=make_manifest(...))` ile manifest'li kurulur ve `make_manifest`'e
`corpus_checksum`/sayılar doğru geçirilir (fixture güncellemesi bu task'ın parçası;
`test_create_app_fails_fast_on_mismatch` manifest'i kasıtlı silerek sınar:
`(data_dir / "index" / "manifest.json").unlink()`).

- [ ] **Step 4: GREEN + full regression** — Run:
  `uv run pytest tests/app -v && uv run pytest -q -m "not slow" && make lint`
- [ ] **Step 5: Commit** — `feat(serve): fail-fast index/model/format compatibility check`

---

### Task 5: ExhaustiveBinaryRetriever + pipeline seçimi (`retrieval/core.py`)

**Files:**
- Modify: `src/belge_gozu/retrieval/core.py`, `src/belge_gozu/config.py`,
  `src/belge_gozu/app/main.py`
- Create: `tests/retrieval/test_exhaustive.py`

**Interfaces:**
- Produces: `ExhaustiveBinaryRetriever(index: PackedIndex, meta: pd.DataFrame,
  encoder: Encoder | None)`:
  - `score_all(q_emb: np.ndarray) -> np.ndarray` — (n_pages,) per-query-token
    normalize skorlar; sayfa-hizalı chunk + `np.maximum.reduceat` (yerelde ölçülen
    ~1.2 s / 4222 sayfa)
  - `search_embedding(q_emb, k: int) -> list[tuple[int, float]]`
  - `search(query: str, k: int = 5) -> list[PageHit]` — telemetri:
    `with stage("exhaustive_maxsim")`
  - `TwoStageRetriever` AYNEN kalır (ablasyon-only; app'ten yalnız flag ile seçilir)
- `config.py`: `retrieval_pipeline: Literal["exhaustive", "two-stage"] = "exhaustive"`
- `app/main.py`: flag'e göre retriever kurar; `/search`-`/ask` çağrıları `candidates`
  parametresini yalnız two-stage'de geçirir.
- Consumes: `binarize_pack`, `PageHit`, telemetri `stage`.

- [ ] **Step 1: Başarısız testleri yaz**

```python
# tests/retrieval/test_exhaustive.py
import numpy as np

from belge_gozu.retrieval.core import ExhaustiveBinaryRetriever, TwoStageRetriever, binary_maxsim
from belge_gozu.index.store import binarize_pack
from tests.retrieval.test_core import build_fixture


def test_scores_match_per_page_maxsim():
    idx, meta, embs = build_fixture(n_pages=30)
    r = ExhaustiveBinaryRetriever(idx, meta, encoder=None)
    q = embs[17]
    scores = r.score_all(q)
    qp = binarize_pack(q)
    for i in (0, 7, 17, 29):
        expected = binary_maxsim(qp, np.asarray(idx.page_tokens(i))) / q.shape[0]
        assert scores[i] == expected


def test_self_match_is_top1():
    idx, meta, embs = build_fixture(n_pages=30)
    r = ExhaustiveBinaryRetriever(idx, meta, encoder=None)
    hits = r.search_embedding(embs[17], k=5)
    assert hits[0][0] == 17


def test_exhaustive_beats_broken_stage1_counterexample():
    """Stage-1'in kaybettiği sonucu exhaustive bulur: Stage-1'i top-1 aday ile
    kısıtla; exhaustive tüm sayfaları görür."""
    idx, meta, embs = build_fixture(n_pages=30)
    ex = ExhaustiveBinaryRetriever(idx, meta, encoder=None)
    ts = TwoStageRetriever(idx, meta, encoder=None)
    q = embs[23]
    ex_top = ex.search_embedding(q, k=30)
    ts_top = ts.search_embedding(q, k=30, candidates=1)
    assert len(ex_top) == 30 and len(ts_top) == 1
    assert ex_top[0][0] == 23


def test_chunk_boundaries_do_not_change_scores():
    idx, meta, embs = build_fixture(n_pages=30)
    r1 = ExhaustiveBinaryRetriever(idx, meta, encoder=None)
    r2 = ExhaustiveBinaryRetriever(idx, meta, encoder=None)
    r2.CHUNK_TOKENS = 16  # sayfa başına 8 token -> her chunk ~2 sayfa
    np.testing.assert_array_equal(r1.score_all(embs[3]), r2.score_all(embs[3]))
```

- [ ] **Step 2: RED gör** — Run: `uv run pytest tests/retrieval/test_exhaustive.py -v`
- [ ] **Step 3: core.py'ye ekle**

```python
class ExhaustiveBinaryRetriever:
    """Tüm korpus üstünde kesin binary MaxSim. 4222 sayfada ~1.2 s (M4 Pro).

    Mean-sign Stage-1 kaldırıldı: ölçülen top-200 kesişimi %11.5-19 ve rank-2
    sonucu 1768'e atma karşı-örneği (spec §1.1). TwoStageRetriever yalnız
    ablasyon için durur (config: retrieval_pipeline="two-stage")."""

    CHUNK_TOKENS = 500_000

    def __init__(self, index: PackedIndex, meta: pd.DataFrame, encoder: Encoder | None):
        self.index = index
        self.encoder = encoder
        self.meta = meta.set_index("page_id", drop=False)
        self.tokens = np.ascontiguousarray(np.asarray(index.tokens))
        self.offsets = np.asarray(index.offsets)

    def _chunk_bounds(self) -> list[int]:
        bounds = [0]
        for i in range(1, len(self.offsets)):
            last = bounds[-1]
            if self.offsets[i] - self.offsets[last] >= self.CHUNK_TOKENS:
                bounds.append(i)
        if bounds[-1] != len(self.offsets) - 1:
            bounds.append(len(self.offsets) - 1)
        return bounds

    def score_all(self, q_emb: np.ndarray) -> np.ndarray:
        q_packed = binarize_pack(q_emb)
        qa = _as_u64(q_packed)
        ta = _as_u64(self.tokens)
        n_pages = len(self.index.page_ids)
        out = np.empty(n_pages, dtype=np.float64)
        bounds = self._chunk_bounds()
        for b0, b1 in zip(bounds[:-1], bounds[1:], strict=True):
            t0, t1 = int(self.offsets[b0]), int(self.offsets[b1])
            ham = np.bitwise_count(qa[:, None, :] ^ ta[None, t0:t1, :]).sum(axis=2)
            sim = (128 - 2 * ham).astype(np.int32)
            starts = (self.offsets[b0:b1] - t0).astype(np.int64)
            out[b0:b1] = np.maximum.reduceat(sim, starts, axis=1).sum(axis=0)
        return out / max(1, q_emb.shape[0])

    def search_embedding(self, q_emb: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores = self.score_all(q_emb)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order]

    def search(self, query: str, k: int = 5) -> list[PageHit]:
        if self.encoder is None:
            raise RuntimeError("encoder yapılandırılmamış")
        with stage("query_encode"):
            q_emb = self.encoder.encode_query(query)
        with stage("exhaustive_maxsim"):
            hits = self.search_embedding(q_emb, k)
        out: list[PageHit] = []
        for i, score in hits:
            row = self.meta.loc[self.index.page_ids[i]]
            out.append(PageHit(
                page_id=row["page_id"], score=score, doc_name=row["doc_name"],
                page_no=int(row["page_no"]), image_path=row["image_path"],
                source_url=row["source_url"],
            ))
        return out
```

`app/main.py`: retriever kurulumu

```python
if s.retrieval_pipeline == "exhaustive":
    retriever = ExhaustiveBinaryRetriever(index, meta, encoder)
else:
    retriever = TwoStageRetriever(index, meta, encoder)
```

`/search` ve `AskService.ask` çağrılarında `candidates` yalnız two-stage'e geçer:
`AskService.ask(question, k, candidates)` imzası `candidates: int | None = None`
olur; retriever `TwoStageRetriever` ise `search(q, k, candidates)`, değilse
`search(q, k)` çağrılır (AskService retriever imzasını `inspect` ile değil,
`isinstance` ile DEĞİL — `search`'ü `search(query, k)` olarak çağırır; TwoStage'in
`search` imzasına `candidates: int = 200` default'u zaten var, app two-stage modda
`functools.partial` ile candidates bağlar). Telemetri: `stage2_maxsim` alanına ek
olarak `exhaustive_maxsim` süresi `detail`'e düşer (T13'te resmîleşir).

- [ ] **Step 4: GREEN + mevcut retrieval testleri** — Run:
  `uv run pytest tests/retrieval -v && uv run pytest -q -m "not slow" && make lint`
- [ ] **Step 5: Commit** — `feat(retrieval): exhaustive binary MaxSim as production path, stage-1 demoted to ablation`

---

### Task 6: Benchmark veri modeli (`bench/dataset.py`)

**Files:**
- Create: `src/belge_gozu/bench/__init__.py`, `src/belge_gozu/bench/dataset.py`,
  `tests/bench/test_dataset.py`, `data/bench/splits_v1.json` (iskelet)

**Interfaces:**
- Produces:

```python
QueryStyle = Literal["dogal", "hukuki", "madde-referansli", "anahtar-kelime"]
Slice = Literal[
    "dogrudan-madde", "paraphrase", "madde-numarali", "ayni-kanun-hard-negative",
    "capraz-kanun-terim", "tablo-layout", "tarihi-tarama", "belirsiz-coklu-dayanak",
    "multi-hop", "korpus-disi", "eksik-kanit", "anlamsiz-ood",
]
UnansReason = Literal["korpus-disi", "eksik-kanit", "anlamsiz", "belirsiz"]

class BenchQuestion(BaseModel):
    question_id: str
    question: str
    query_style: QueryStyle
    answerable: bool
    gold_doc_ids: list[str]
    gold_page_ids: list[str]          # "dok:sayfa"; answerable=True iken >=1
    gold_article_ids: list[str]       # "k4721:m19" / "k4721:gm2"; boş olabilir
    minimal_evidence_spans: list[str]
    reference_answer: str             # answerable=False iken ""
    slice: Slice
    difficulty: Literal["kolay", "orta", "zor"]
    source_type: Literal["insan", "insan-paraphrase", "ajan-taslak-insan-onayli"]
    requires_visual: bool
    requires_multi_hop: bool
    unanswerable_reason: UnansReason | None
    verified_by: str
    verification_status: Literal["draft", "verified", "rejected"]

def load_bench(path: Path, only_verified: bool = True) -> list[BenchQuestion]  # JSONL
def bench_stats(questions: list[BenchQuestion]) -> dict[str, int]              # dilim sayımı
def load_splits(path: Path) -> dict[str, set[str]]   # {"dev_docs": {...}, "test_docs": {...}}
def question_split(q: BenchQuestion, splits: dict[str, set[str]]) -> Literal["dev", "test"]
```

Doğrulama kuralları (pydantic model_validator): `answerable=True` →
`gold_page_ids` boş olamaz, `reference_answer` boş olamaz, `unanswerable_reason is None`;
`answerable=False` → `gold_page_ids == []`, `unanswerable_reason` zorunlu;
her `gold_page_ids` elemanı `":"` içerir ve doc kısmı `gold_doc_ids`'te yer alır;
`verification_status="verified"` → `verified_by` boş olamaz.
`question_split`: gold_doc_ids'in ilki hangi kümede ise o; cevaplanamaz/dok'suz soru
`sha256(question_id)` çiftliğine göre dev/test'e atanır (deterministik).

- [ ] **Step 1: Başarısız testleri yaz**

```python
# tests/bench/test_dataset.py
import json
from pathlib import Path

import pytest

from belge_gozu.bench.dataset import (
    BenchQuestion, bench_stats, load_bench, load_splits, question_split,
)


def q_dict(**over) -> dict:
    base = dict(
        question_id="q1", question="Yerleşim yeri nedir?", query_style="dogal",
        answerable=True, gold_doc_ids=["k4721"], gold_page_ids=["k4721:4"],
        gold_article_ids=["k4721:m19"],
        minimal_evidence_spans=["Yerleşim yeri bir kimsenin sürekli kalma niyetiyle oturduğu yerdir."],
        reference_answer="Sürekli kalma niyetiyle oturulan yerdir (TMK m.19).",
        slice="paraphrase", difficulty="orta", source_type="insan",
        requires_visual=False, requires_multi_hop=False, unanswerable_reason=None,
        verified_by="baran", verification_status="verified",
    )
    base.update(over)
    return base


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_load_verified_only(tmp_path: Path):
    p = tmp_path / "b.jsonl"
    write_jsonl(p, [q_dict(), q_dict(question_id="q2", verification_status="draft")])
    assert [q.question_id for q in load_bench(p)] == ["q1"]
    assert len(load_bench(p, only_verified=False)) == 2


def test_answerable_requires_gold_pages():
    with pytest.raises(ValueError):
        BenchQuestion(**q_dict(gold_page_ids=[]))


def test_unanswerable_requires_reason_and_no_gold():
    with pytest.raises(ValueError):
        BenchQuestion(**q_dict(answerable=False, gold_page_ids=[], reference_answer="",
                               unanswerable_reason=None))
    ok = BenchQuestion(**q_dict(answerable=False, gold_doc_ids=[], gold_page_ids=[],
                                gold_article_ids=[], minimal_evidence_spans=[],
                                reference_answer="", slice="korpus-disi",
                                unanswerable_reason="korpus-disi"))
    assert ok.answerable is False


def test_gold_page_doc_consistency():
    with pytest.raises(ValueError):
        BenchQuestion(**q_dict(gold_page_ids=["k9999:1"]))


def test_split_assignment(tmp_path: Path):
    sp = tmp_path / "splits.json"
    sp.write_text(json.dumps({"dev_docs": ["k4721"], "test_docs": ["k6098"]}))
    splits = load_splits(sp)
    assert question_split(BenchQuestion(**q_dict()), splits) == "dev"
    ood = BenchQuestion(**q_dict(question_id="ood1", answerable=False, gold_doc_ids=[],
                                 gold_page_ids=[], gold_article_ids=[],
                                 minimal_evidence_spans=[], reference_answer="",
                                 slice="anlamsiz-ood", unanswerable_reason="anlamsiz"))
    assert question_split(ood, splits) in ("dev", "test")  # deterministik hash ataması
```

- [ ] **Step 2: RED gör** — Run: `uv run pytest tests/bench/test_dataset.py -v`
- [ ] **Step 3: dataset.py yaz** — yukarıdaki imza ve kurallarla; `load_bench` JSONL'i
  satır satır parse eder, hatada `ValueError(f"bench satır {i}: ...")`; boş sonuç
  `ValueError`. `data/bench/splits_v1.json` iskeleti:
  `{"dev_docs": [], "test_docs": []}` (T10 canary tamamı dev'dir; P1 T12 doldurur).
- [ ] **Step 4: GREEN + full regression** — Run:
  `uv run pytest tests/bench -v && uv run pytest -q -m "not slow" && make lint`
- [ ] **Step 5: Commit** — `feat(bench): rich benchmark data model with verification and split rules`

---

### Task 7: Metrikler (`bench/metrics.py`)

**Files:**
- Create: `src/belge_gozu/bench/metrics.py`, `tests/bench/test_metrics.py`

**Interfaces:**
- Produces: `recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float`;
  `mrr(relevant: set[str], ranked: list[str]) -> float`;
  `ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float` (binary rel,
  log2 indirim, IDCG normalizasyonu);
  `bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05,
  seed: int = 0) -> tuple[float, float]` (yüzdelik bootstrap, deterministik seed).

- [ ] **Step 1: Başarısız testleri yaz**

```python
# tests/bench/test_metrics.py
import pytest

from belge_gozu.bench.metrics import bootstrap_ci, mrr, ndcg_at_k, recall_at_k


def test_recall():
    assert recall_at_k({"a", "b"}, ["a", "x", "y"], 3) == 0.5
    assert recall_at_k({"a"}, [], 5) == 0.0


def test_mrr():
    assert mrr({"a"}, ["x", "a", "y"]) == 0.5
    assert mrr({"a"}, ["x", "y"]) == 0.0


def test_ndcg():
    assert ndcg_at_k({"a"}, ["a", "b"], 5) == pytest.approx(1.0)
    assert ndcg_at_k({"a"}, ["b", "a"], 5) == pytest.approx(0.6309, abs=1e-3)
    assert ndcg_at_k({"a"}, ["b", "c"], 5) == 0.0


def test_bootstrap_ci_deterministic_and_ordered():
    vals = [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    lo, hi = bootstrap_ci(vals, n_boot=500, seed=7)
    assert (lo, hi) == bootstrap_ci(vals, n_boot=500, seed=7)
    assert 0.0 <= lo <= sum(vals) / len(vals) <= hi <= 1.0
    assert bootstrap_ci([], n_boot=10) == (0.0, 0.0)
```

- [ ] **Step 2: RED gör** — Run: `uv run pytest tests/bench/test_metrics.py -v`
- [ ] **Step 3: metrics.py yaz**

```python
import math

import numpy as np


def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def mrr(relevant: set[str], ranked: list[str]) -> float:
    for i, p in enumerate(ranked, start=1):
        if p in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, p in enumerate(ranked[:k]) if p in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0


def bootstrap_ci(
    values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
```

- [ ] **Step 4: GREEN + full regression** — Run:
  `uv run pytest tests/bench -v && uv run pytest -q -m "not slow" && make lint`
- [ ] **Step 5: Commit** — `feat(bench): retrieval metrics with deterministic bootstrap CI`

---

### Task 8: Teşhis harness'ı + `bench run` (`bench/harness.py`)

**Files:**
- Create: `src/belge_gozu/bench/harness.py`, `tests/bench/test_harness.py`
- Modify: `src/belge_gozu/cli.py` (`bench_app` + `bench run`)

**Interfaces:**
- Consumes: T6 dataset, T7 metrics, T5 retriever'lar, T1 manifest.
- Produces:

```python
class StageRecord(BaseModel):
    stage: str                       # "exhaustive-binary" | "stage1" | "stage2" | (P1: kanal adları) | "final"
    gold_ranks: dict[str, int]       # page_id -> 1-tabanlı sıra; listede yoksa -1
    top_ids: list[str]               # ilk record_top eleman
    top_scores: list[float]
    latency_ms: float

class QuestionDiagnostic(BaseModel):
    question_id: str
    stages: list[StageRecord]
    candidate_survival: dict[str, bool]   # gold page_id -> nihai aday havuzunda mı
    final_ranked: list[str]               # ilk record_top

class MetricBlock(BaseModel):
    recall_at: dict[int, float]
    mrr: float
    ndcg5: float
    n: int
    ci_recall5: tuple[float, float] | None = None

class EvalReport(BaseModel):
    run_id: str
    git_commit: str
    index_manifest: dict | None
    config: dict
    missing_gold_pages: list[str]         # korpus coverage ihlalleri
    overall: MetricBlock
    per_slice: dict[str, MetricBlock]
    per_doc: dict[str, MetricBlock]
    diagnostics: list[QuestionDiagnostic]
    def to_json(self, path: Path) -> None: ...

class DiagnosticPipeline(Protocol):
    name: str
    def run(self, question: str) -> tuple[list[str], list[StageRecord]]:
        """tam sıralı page_id listesi (en az record_top) + aşama kayıtları"""

class ExhaustiveDiagnosticAdapter:   # ExhaustiveBinaryRetriever sarar
    def __init__(self, retriever: ExhaustiveBinaryRetriever, record_top: int = 200): ...
class TwoStageDiagnosticAdapter:     # TwoStageRetriever sarar (B1/B2 ablasyonu)
    def __init__(self, retriever: TwoStageRetriever, candidates: int = 200,
                 record_top: int = 200): ...

def run_retrieval_eval(
    pipeline: DiagnosticPipeline, questions: list[BenchQuestion],
    known_page_ids: set[str], ks: tuple[int, ...] = (1, 5, 10, 20, 50, 200),
    run_id: str = "", index_manifest: IndexManifest | None = None,
    config: dict | None = None,
) -> EvalReport
```

Davranış: yalnız `answerable=True` sorular metriklere girer (cevaplanamazlar P2
answer-eval'inde); her soruda gold sayfaların `known_page_ids` içinde olup olmadığı
kontrol edilir (`missing_gold_pages` — G0.1 coverage); `candidate_survival` son
aşamanın `top_ids` kümesine göre; `gold_ranks` tüm aşamalarda hesaplanır.

- [ ] **Step 1: Başarısız testleri yaz**

```python
# tests/bench/test_harness.py
from pathlib import Path

from belge_gozu.bench.harness import EvalReport, StageRecord, run_retrieval_eval
from tests.bench.test_dataset import q_dict
from belge_gozu.bench.dataset import BenchQuestion


class MapPipeline:
    name = "map"

    def __init__(self, answers: dict[str, list[str]]):
        self.answers = answers

    def run(self, question: str):
        ranked = self.answers[question]
        rec = StageRecord(stage="final",
                          gold_ranks={}, top_ids=ranked, top_scores=[1.0] * len(ranked),
                          latency_ms=1.0)
        return ranked, [rec]


def test_report_metrics_and_survival(tmp_path: Path):
    qs = [
        BenchQuestion(**q_dict()),                                        # gold k4721:4
        BenchQuestion(**q_dict(question_id="q2", question="ikinci",
                               gold_doc_ids=["k6098"], gold_page_ids=["k6098:120"],
                               gold_article_ids=[], slice="dogrudan-madde")),
    ]
    pipe = MapPipeline({
        "Yerleşim yeri nedir?": ["k4721:4", "x:1"],
        "ikinci": ["x:1", "x:2"],
    })
    rep = run_retrieval_eval(pipe, qs, known_page_ids={"k4721:4", "k6098:120", "x:1", "x:2"},
                             ks=(1, 5), run_id="t")
    assert rep.overall.recall_at[1] == 0.5 and rep.overall.recall_at[5] == 0.5
    assert rep.overall.mrr == 0.5
    d = {x.question_id: x for x in rep.diagnostics}
    assert d["q1"].candidate_survival == {"k4721:4": True}
    assert d["q2"].candidate_survival == {"k6098:120": False}
    assert rep.per_slice["paraphrase"].n == 1
    out = tmp_path / "r.json"
    rep.to_json(out)
    assert out.exists()


def test_missing_gold_page_reported():
    qs = [BenchQuestion(**q_dict())]
    pipe = MapPipeline({"Yerleşim yeri nedir?": ["x:1"]})
    rep = run_retrieval_eval(pipe, qs, known_page_ids={"x:1"}, ks=(1,))
    assert rep.missing_gold_pages == ["k4721:4"]


def test_exhaustive_adapter_records_ranks():
    from belge_gozu.bench.harness import ExhaustiveDiagnosticAdapter
    from belge_gozu.retrieval.core import ExhaustiveBinaryRetriever
    from tests.retrieval.test_core import build_fixture

    idx, meta, embs = build_fixture(n_pages=30)

    class SelfEnc:
        def encode_pages(self, images):
            raise NotImplementedError

        def encode_query(self, text):
            return embs[int(text)]

    ad = ExhaustiveDiagnosticAdapter(ExhaustiveBinaryRetriever(idx, meta, SelfEnc()),
                                     record_top=30)
    ranked, stages = ad.run("17")
    assert ranked[0] == "d17:1"
    assert stages[0].stage == "exhaustive-binary" and stages[0].latency_ms >= 0
```

- [ ] **Step 2: RED gör** — Run: `uv run pytest tests/bench/test_harness.py -v`
- [ ] **Step 3: harness.py yaz** — imzalar yukarıdaki gibi; `run_retrieval_eval` gövdesi:

```python
def run_retrieval_eval(pipeline, questions, known_page_ids, ks=(1, 5, 10, 20, 50, 200),
                       run_id="", index_manifest=None, config=None) -> EvalReport:
    missing = sorted({p for q in questions for p in q.gold_page_ids
                      if p not in known_page_ids})
    diags: list[QuestionDiagnostic] = []
    rows: list[tuple[BenchQuestion, list[str]]] = []
    for q in questions:
        if not q.answerable:
            continue
        ranked, stages = pipeline.run(q.question)
        rel = set(q.gold_page_ids)
        for st in stages:
            st.gold_ranks = {
                g: (st.top_ids.index(g) + 1 if g in st.top_ids else -1) for g in rel
            }
        final_ids = stages[-1].top_ids if stages else ranked
        diags.append(QuestionDiagnostic(
            question_id=q.question_id, stages=stages,
            candidate_survival={g: g in set(final_ids) for g in rel},
            final_ranked=ranked[: max(ks)],
        ))
        rows.append((q, ranked))

    def block(pairs: list[tuple[BenchQuestion, list[str]]]) -> MetricBlock:
        if not pairs:
            return MetricBlock(recall_at={k: 0.0 for k in ks}, mrr=0.0, ndcg5=0.0, n=0)
        r5 = [recall_at_k(set(q.gold_page_ids), r, 5) for q, r in pairs]
        return MetricBlock(
            recall_at={k: sum(recall_at_k(set(q.gold_page_ids), r, k) for q, r in pairs)
                       / len(pairs) for k in ks},
            mrr=sum(mrr(set(q.gold_page_ids), r) for q, r in pairs) / len(pairs),
            ndcg5=sum(ndcg_at_k(set(q.gold_page_ids), r, 5) for q, r in pairs) / len(pairs),
            n=len(pairs), ci_recall5=bootstrap_ci(r5),
        )

    per_slice: dict[str, list] = {}
    per_doc: dict[str, list] = {}
    for q, r in rows:
        per_slice.setdefault(q.slice, []).append((q, r))
        for d in q.gold_doc_ids:
            per_doc.setdefault(d, []).append((q, r))
    return EvalReport(
        run_id=run_id, git_commit=_git_commit(), 
        index_manifest=index_manifest.model_dump() if index_manifest else None,
        config=config or {}, missing_gold_pages=missing,
        overall=block(rows),
        per_slice={s: block(p) for s, p in per_slice.items()},
        per_doc={d: block(p) for d, p in per_doc.items()},
        diagnostics=diags,
    )
```

Adapter'lar: `ExhaustiveDiagnosticAdapter.run` — encode bir kez, `score_all` bir kez,
`argsort` ile tam sıralama, ilk `record_top` kaydedilir, `time.perf_counter` ile
latency. `TwoStageDiagnosticAdapter.run` — stage1 sıralamasını (`hamming_matrix`
üstünden) `stage="stage1"` olarak, stage2 aday skorlamasını `stage="stage2"` olarak
kaydeder. `_git_commit()`: `subprocess.run(["git", "rev-parse", "--short", "HEAD"], ...)`,
hata halinde `"unknown"`.
CLI: `belge-gozu bench run --bench data/bench/canary_v1.jsonl
--pipeline exhaustive|two-stage --out data/bench/results/<run_id>.json`
(`run_id` üretimi: `<UTC tarih>-<git sha>-<pipeline>`); gerçek indeks + `ColSmolEncoder`
ile koşar, `known_page_ids` indeksten.

- [ ] **Step 4: GREEN + full regression** — Run:
  `uv run pytest tests/bench -v && uv run pytest -q -m "not slow" && make lint`
- [ ] **Step 5: Commit** — `feat(bench): stage-diagnostic eval harness with run provenance`

---

### Task 9: Float oracle (`bench/oracle.py` + `index build --precision f16`)

**Files:**
- Create: `src/belge_gozu/bench/oracle.py`, `tests/bench/test_oracle.py`
- Modify: `src/belge_gozu/cli.py` (`index build --precision {packed,f16}
  --query-format {cpe-0.3.18,train-compat-v1} --out PATH`, `bench oracle`)

**Interfaces:**
- Produces:

```python
class FloatIndex:
    """f16 master token embedding'leri: embs.npy (toplam_token, 128) float16 +
    offsets.npy + page_ids.json + manifest.json (quantization="float16")."""
    embs: np.ndarray; offsets: np.ndarray; page_ids: list[str]
    manifest: IndexManifest | None
    @classmethod
    def build(cls, page_ids: list[str], embs: list[np.ndarray],
              manifest: IndexManifest | None = None) -> "FloatIndex": ...
    def save(self, dir: Path) -> None: ...
    @classmethod
    def load(cls, dir: Path, mmap: bool = True) -> "FloatIndex": ...
    def page_tokens(self, i: int) -> np.ndarray: ...

def native_float_scores(findex: FloatIndex, q_emb: np.ndarray) -> np.ndarray
    # (n_pages,) — float MaxSim (per-query-token ortalama), sayfa-hizalı chunk'lı
def rank_of(scores: np.ndarray, page_ids: list[str], target: str) -> int   # 1-tabanlı
```

- CLI `index build --precision f16`: meta'daki tüm sayfaları maskeli encoder'la
  encode edip `FloatIndex` olarak `--out` dizinine yazar (manifest'li). `--precision
  packed` (varsayılan) mevcut davranış + manifest.
- CLI `bench oracle --bench PATH --packed-index DIR --float-index DIR --out PATH`:
  her soru için exhaustive-binary sırası vs native-float sırası; oracle-gap JSON'u
  (soru başına `{binary_rank, float_rank}` + özet Recall@k tablosu iki oracle için).
- Consumes: T2 encoder, T5 retriever, T6-T8.

- [ ] **Step 1: Başarısız testleri yaz**

```python
# tests/bench/test_oracle.py
from pathlib import Path

import numpy as np

from belge_gozu.bench.oracle import FloatIndex, native_float_scores, rank_of


def make_findex(n_pages=5, tokens=4, seed=0):
    rng = np.random.default_rng(seed)
    embs = [rng.standard_normal((tokens, 128)).astype(np.float32) for _ in range(n_pages)]
    return FloatIndex.build([f"d{i}:1" for i in range(n_pages)], embs), embs


def test_roundtrip(tmp_path: Path):
    fi, _ = make_findex()
    fi.save(tmp_path)
    fi2 = FloatIndex.load(tmp_path, mmap=False)
    assert fi2.page_ids == fi.page_ids
    np.testing.assert_allclose(np.asarray(fi2.embs, dtype=np.float32),
                               np.asarray(fi.embs, dtype=np.float32), atol=1e-3)


def test_self_match_top1():
    fi, embs = make_findex()
    scores = native_float_scores(fi, embs[3])
    assert scores.argmax() == 3
    assert rank_of(scores, fi.page_ids, "d3:1") == 1


def test_scores_are_true_maxsim():
    fi, embs = make_findex(n_pages=2)
    q = embs[0]
    expected = (q @ np.asarray(fi.page_tokens(1), dtype=np.float32).T).max(axis=1).sum() / q.shape[0]
    assert abs(native_float_scores(fi, q)[1] - expected) < 1e-2  # f16 saklama toleransı
```

- [ ] **Step 2: RED gör** — Run: `uv run pytest tests/bench/test_oracle.py -v`
- [ ] **Step 3: oracle.py + CLI yaz** — `FloatIndex` saklamada float16'ya çevirir,
  skorlamada chunk'ları float32'ye açarak `q @ chunk.T` + `np.maximum.reduceat`
  (T5'teki desenle aynı, Hamming yerine dot). `rank_of`: `1 + (scores > scores[i]).sum()`
  (eşitlikte iyimser değil — `np.argsort(-scores, kind="stable")` üstünden pozisyon).
  CLI build yolu `cli.py index_build`'in encode döngüsünü yeniden kullanır
  (`--precision` dalı yalnız saklama sınıfını seçer); manifest her iki yolda da yazılır
  (`git_commit`, `built_at`, `engine_versions` `importlib.metadata` ile).
- [ ] **Step 4: GREEN + full regression** — Run:
  `uv run pytest tests/bench -v && uv run pytest -q -m "not slow" && make lint`
- [ ] **Step 5: Commit** — `feat(bench): native float16 oracle index and oracle-gap tooling`

---

### Task 10: Canary set v1 + semantic canary regression testleri (runbook + kod)

**Files:**
- Create: `data/bench/canary_v1.jsonl` (30-50 soru; İNSAN doğrulama kapılı),
  `tests/retrieval/test_semantic_canary.py`, `tests/retrieval/canary_expectations.json`

**Interfaces:**
- Consumes: T6 şeması, T5 üretim hattı, T8 harness.
- Produces: insan-doğrulamalı canary seti (tamamı `verification_status="verified"`,
  `verified_by` dolu); iki hedef sorgu dahil; slow regression testleri.

- [ ] **Step 1 (ajan taslağı):** Korpustan örneklenen sayfa görüntülerini DOĞRUDAN
  okuyarak (API kotası gerekmez) 45-60 taslak soru üret → `canary_v1.jsonl`,
  `verification_status="draft"`, `source_type="ajan-taslak-insan-onayli"`. Dağılım
  hedefi: `dogrudan-madde` 10, `paraphrase` 10 (Sorgu A ve B dahil), `madde-numarali` 6,
  `ayni-kanun-hard-negative` 5, `capraz-kanun-terim` 4, `tablo-layout` 4,
  `tarihi-tarama` 4, `korpus-disi` 3, `anlamsiz-ood` 2. Her answerable soruda
  `gold_page_ids` ajanın gerçekten okuduğu sayfa(lar); `minimal_evidence_spans`
  sayfadan birebir alıntı.
- [ ] **Step 2 (İNSAN — bloklayıcı):** Kullanıcı taslağı gözden geçirir; doğru satırlara
  `verification_status="verified"` + `verified_by` yazar, bozukları düzeltir/`rejected`
  yapar. Hedef: ≥30 verified (≥25 answerable + ≥5 unanswerable). Bu adım kullanıcı
  onayı olmadan geçilmez.
- [ ] **Step 3: Doğrulama komutu** — Run:
  `uv run python -c "from pathlib import Path; from belge_gozu.bench.dataset import load_bench, bench_stats; qs=load_bench(Path('data/bench/canary_v1.jsonl')); print(len(qs), bench_stats(qs))"`
  — Expected: ≥30 verified, dilim dağılımı basılır.
- [ ] **Step 4: Semantic canary testlerini yaz**

```python
# tests/retrieval/test_semantic_canary.py  (tamamı -m slow: gerçek model + gerçek indeks)
import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.slow

Q_SHORT = "Yerleşim yeri nedir?"
Q_LONG = "Türk Medeni Kanunu'na göre yerleşim yeri nasıl tanımlanır?"
GOLD = "k4721:4"
EXPECT = json.loads(Path("tests/retrieval/canary_expectations.json").read_text())


@pytest.fixture(scope="module")
def prod_retriever():
    from belge_gozu.config import get_settings
    from belge_gozu.index.encode import ColSmolEncoder
    from belge_gozu.index.store import PackedIndex
    from belge_gozu.retrieval.core import ExhaustiveBinaryRetriever

    s = get_settings()
    idx = PackedIndex.load(s.index_dir)
    meta = pd.read_parquet(s.index_dir / "meta.parquet")
    return ExhaustiveBinaryRetriever(idx, meta, ColSmolEncoder(s.retriever_model, s.device))


def test_canary_gold_pages_covered(prod_retriever):
    from belge_gozu.bench.dataset import load_bench

    known = set(prod_retriever.index.page_ids)
    for q in load_bench(Path("data/bench/canary_v1.jsonl")):
        for g in q.gold_page_ids:
            assert g in known, f"{q.question_id}: {g} korpusta yok"


def test_short_query_gold_in_top5(prod_retriever):
    hits = prod_retriever.search(Q_SHORT, k=5)
    assert GOLD in [h.page_id for h in hits], [h.page_id for h in hits]


def test_long_query_rank_ratchet(prod_retriever):
    q_emb = prod_retriever.encoder.encode_query(Q_LONG)
    scores = prod_retriever.score_all(q_emb)
    import numpy as np

    order = np.argsort(-scores, kind="stable")
    rank = int(np.nonzero(order == prod_retriever.index.page_ids.index(GOLD))[0][0]) + 1
    assert rank <= EXPECT["long_query_gold_rank_max"], (
        f"uzun sorgu gold sırası {rank} > cırcır {EXPECT['long_query_gold_rank_max']}"
    )
```

`canary_expectations.json` ilk değeri: `{"long_query_gold_rank_max": 1576}` (bugünkü
ölçüm; format/kuantizasyon kararlarıyla İYİLEŞTİKÇE bilinçli commit'le düşürülür —
cırcır asla sessizce gevşetilmez).
Not: `test_short_query_gold_in_top5` mevcut v0 indeksinde exhaustive sıra 2 ölçüldüğü
için Stage-1 kaldırılınca YEŞİL olmalıdır — bu, P0'ın ana davranış düzelmesinin
regression kilididir (G0.8).

- [ ] **Step 5: Slow koşum** — Run:
  `uv run pytest tests/retrieval/test_semantic_canary.py -m slow -v` — Expected: 3 PASS
- [ ] **Step 6: Commit** — `test: human-verified canary set v1 + real-model semantic regression locks`

---

### Task 11: Processor format A/B yeniden indeksleme + karar (runbook)

**Files:**
- Modify: `src/belge_gozu/index/manifest.py` (gerekirse `TRAIN_COMPAT_V1` düzeltmesi),
  `src/belge_gozu/config.py` (`query_format_id: str = "cpe-0.3.18"` — karar sonrası
  güncellenir)
- Create: `scripts/ab_st_reference.py`, indeks dizinleri
  `data/index-cpe0318-f16/`, `data/index-traincompat-f16/` (+ türetilen packed'ler T12'de)

**Interfaces:**
- Consumes: T2 encoder (`query_format` parametresi), T9 `index build --precision f16
  --query-format ...`, T8 harness, T10 canary.
- Produces: format kararı (`query_format_id` + doc prompt) — p0-gate raporuna sayılarla.

- [ ] **Step 1: Eğitim formatını birincil kaynaktan kilitle** —
  `huggingface_hub.hf_hub_download("vidore/colSmol-500M", "config_sentence_transformers.json")`
  indir; `prompts` alanındaki query/document şablonlarını `TRAIN_COMPAT_V1.render("X")`
  çıktısıyla karşılaştır. Uyuşmazsa `TRAIN_COMPAT_V1` sabitini ve
  `test_query_format_render`'ı gerçek şablona göre düzelt (kaynak: model kartı "The
  Sentence Transformers configuration in this repository reproduces the original
  training-time format"). Doc tarafı şablonu farklıysa encoder'a
  `visual_prompt_override: str | None = None` parametresi ekle (processor
  `visual_prompt_prefix`'ini geçersiz kılar; birim test: override verilince
  `process_images`'a giden text değişir).
- [ ] **Step 2: ST çapraz doğrulama** — Run:
  `uv run --with "sentence-transformers>=5.0" python scripts/ab_st_reference.py`
  — betik: Sorgu A ve B'yi (1) bizim encoder(train-compat) (2) ST `MultiVectorEncoder`
  ile encode eder; token sayısı ve sign-bit uyuşmasını basar. Expected: uyuşma ≥ %99;
  değilse fark raporlanır ve şablon Step 1'e dönerek düzeltilir.
- [ ] **Step 3: f16 master'ları üret** (uzun; MPS ~1 saat/koşum) — Run:
  `BG_DEVICE=mps uv run belge-gozu index build --precision f16 --query-format cpe-0.3.18 --out data/index-cpe0318-f16`
  ve `... --query-format train-compat-v1 --out data/index-traincompat-f16`
  (doc prompt: Step 1 kararına göre; iki build arasında YALNIZ format değişir).
- [ ] **Step 4: A/B koşumu** — Run: her iki f16 dizini için
  `uv run belge-gozu bench oracle --bench data/bench/canary_v1.jsonl --float-index <dir> --packed-index <T12'de türetilen> --out data/bench/results/<run_id>.json`
  (ilk turda yalnız float karşılaştırması yeterli; packed karşılaştırma T12'de
  tamamlanır). Karar metriği: canary answerable Recall@5 / MRR (float düzeyinde);
  eşitlikte bootstrap CI ve iki hedef sorgunun sıraları.
- [ ] **Step 5: D1 augmentation ablasyonu** — kazanan formatta `n_suffix=0` vs `10`
  sorgu-tarafı koşumu (indeks sabit): `bench oracle`'a `--query-format-override`
  gerekmez; betik `scripts/ab_st_reference.py` deseninde küçük bir koşum bloğu
  `scripts/d1_augmentation.py` ile yapılır, sonuç rapora.
- [ ] **Step 6: Kararı uygula** — `config.py` `query_format_id` güncellenir; karar +
  sayılar + koşum künyeleri p0-gate raporu taslağına işlenir. Commit —
  `feat(index): adopt <winner> query/document format (A/B on canary, run <run_id>)`

---

### Task 12: Kuantizasyon ablasyonu C1/C2 (`index/quantize.py`)

**Files:**
- Create: `src/belge_gozu/index/quantize.py`, `tests/index/test_quantize.py`
- Modify: `src/belge_gozu/cli.py` (`index derive --from data/index-<fmt>-f16
  --quant {sign-1bit,int8} --out DIR`)

**Interfaces:**
- Consumes: T9 `FloatIndex`, T3 `PackedIndex`, T1 manifest.
- Produces:

```python
def derive_packed(findex: FloatIndex) -> PackedIndex
    # f16 master -> sign-1bit; manifest kopyalanır, quantization="sign-1bit"
class Int8Index:
    """per-token simetrik ölçek: scale_t = max|x_t| / 127; q_t = round(x_t/scale_t).
    Skor: float32'ye açılmış chunk'larla MaxSim (saklama küçülür, hesap float)."""
    codes: np.ndarray      # (toplam_token, 128) int8
    scales: np.ndarray     # (toplam_token,) float32
    offsets: np.ndarray; page_ids: list[str]; manifest: IndexManifest | None
    @classmethod
    def derive(cls, findex: FloatIndex) -> "Int8Index": ...
    def save(self, dir: Path) -> None: ...
    @classmethod
    def load(cls, dir: Path, mmap: bool = True) -> "Int8Index": ...
    def score_all(self, q_emb: np.ndarray) -> np.ndarray: ...
```

- [ ] **Step 1: Başarısız testleri yaz**

```python
# tests/index/test_quantize.py
import numpy as np

from belge_gozu.bench.oracle import FloatIndex, native_float_scores
from belge_gozu.index.quantize import Int8Index, derive_packed
from belge_gozu.index.store import binarize_pack


def make_findex(n_pages=6, tokens=5, seed=2):
    rng = np.random.default_rng(seed)
    embs = [rng.standard_normal((tokens, 128)).astype(np.float32) for _ in range(n_pages)]
    return FloatIndex.build([f"d{i}:1" for i in range(n_pages)], embs), embs


def test_derive_packed_matches_direct_binarize():
    fi, embs = make_findex()
    packed = derive_packed(fi)
    np.testing.assert_array_equal(
        np.asarray(packed.page_tokens(2)),
        binarize_pack(np.asarray(fi.page_tokens(2), dtype=np.float32)),
    )


def test_int8_scores_close_to_float():
    fi, embs = make_findex()
    i8 = Int8Index.derive(fi)
    f = native_float_scores(fi, embs[1])
    q = i8.score_all(embs[1])
    assert np.argmax(q) == np.argmax(f) == 1
    np.testing.assert_allclose(q, f, rtol=0.05, atol=0.5)


def test_int8_roundtrip(tmp_path):
    fi, _ = make_findex()
    i8 = Int8Index.derive(fi)
    i8.save(tmp_path)
    i82 = Int8Index.load(tmp_path, mmap=False)
    np.testing.assert_array_equal(np.asarray(i82.codes), np.asarray(i8.codes))
```

- [ ] **Step 2: RED gör** — Run: `uv run pytest tests/index/test_quantize.py -v`
- [ ] **Step 3: quantize.py + CLI yaz** — yukarıdaki sözleşmeyle; `derive_packed`
  f16 satırlarını float32'ye açıp `binarize_pack` uygular (all-zero satır f16
  master'da zaten yok — T2 maskesi), manifest'i `quantization` alanı değiştirip taşır.
- [ ] **Step 4: GREEN + full regression** — Run:
  `uv run pytest tests/index -v && uv run pytest -q -m "not slow" && make lint`
- [ ] **Step 5: C1/C2 koşumları (runbook)** — kazanan format dizininden: Run:
  `uv run belge-gozu index derive --from data/index-<fmt>-f16 --quant sign-1bit --out data/index-<fmt>-1bit`
  ve `--quant int8 --out data/index-<fmt>-int8`; ardından canary üzerinde üç koşum
  (float / int8 / 1-bit) `bench oracle` + `bench run` ile. Karar kuralı: **float16
  oracle'a göre Recall@20 kaybı ≤ 2 puan olan en küçük temsil üretim indeksi olur**;
  1-bit bu eşiği geçemiyorsa int8; int8 de geçemiyorsa ColBERTv2-tarzı centroid+residual
  ayrı task olarak P1 öncesine eklenir (koşul, p0-gate raporunda sayıyla gerekçelenir).
- [ ] **Step 6: Kararı uygula + commit** — `config.py`/`index_dir` seçimi güncellenir;
  `feat(index): quantization decision <winner> (C1/C2 ablation, run <run_id>)`

---

### Task 13: Kalite telemetrisi genişletmesi

**Files:**
- Modify: `src/belge_gozu/telemetry/schema.py` (yeni kolonlar), `src/belge_gozu/telemetry/recorder.py`
  (migrasyon), `src/belge_gozu/app/main.py` (alan doldurma), `src/belge_gozu/telemetry/prom.py`
  (`set_app_info`'ya `index_revision` + `query_format` etiketi), `tests/telemetry/test_schema.py`,
  `tests/app/test_api.py`

**Interfaces:**
- Produces: `RequestEvent`'e nullable alanlar: `pipeline: str | None`,
  `index_revision: str | None` (manifest `corpus_checksum[:12] + "/" +
  query_format.format_id + "/" + quantization`); `detail["retrieval"]` sözlüğü:
  `{"candidates": [ilk 20 {page_id, score}], "stage_latencies": {...}}` (mevcut
  `detail["hits"]` korunur — geriye uyumlu). `EVENTS_DDL`'e kolonlar eklenir;
  `EventRecorder._ensure_schema` mevcut tabloya `ALTER TABLE ... ADD COLUMN` ile
  eksik kolonları ekler (best-effort, hata isteği düşürmez — mevcut ilke).
- Consumes: T1 manifest, T5 retriever, mevcut telemetri sözleşmeleri
  (`2026-08-26-telemetry-design.md` §5-6; bu task o spec'i DEĞİŞTİRMEZ, genişletir).

- [ ] **Step 1: Başarısız testleri yaz** — `tests/telemetry/test_schema.py`'ye:
  `RequestEvent(pipeline="exhaustive", index_revision="abc/cpe/1bit", ...)` kurulabilir;
  `tests/app/test_api.py`'ye: `/search` sonrası events satırında `pipeline` dolu ve
  `detail` JSON'unda `"retrieval"` anahtarı var; mevcut kolonlu ESKİ bir events
  tablosuna yazımın hata fırlatmadığı (migrasyon testi: DDL'in eski halini elle kurup
  recorder'a yeni event yazdır).
- [ ] **Step 2: RED gör** — Run: `uv run pytest tests/telemetry tests/app -v`
- [ ] **Step 3: Uygula** — şema + recorder migrasyonu + `build_event`'e alanlar +
  prom `set_app_info(..., index_revision=..., query_format=...)`.
- [ ] **Step 4: GREEN + full regression** — Run: `uv run pytest -q -m "not slow" && make lint`
- [ ] **Step 5: Commit** — `feat(telemetry): retrieval provenance fields (pipeline, index revision, candidates)`

---

### Task 14: README/UI dürüstlüğü + P0 baseline & gate raporu

**Files:**
- Modify: `README.md`, `src/belge_gozu/app/static/index.html`, `src/belge_gozu/config.py`
  (eşik yorumunu güncelle — DEĞERİ DEĞİL)
- Create: `docs/research/findings/2026-XX-XX-p0-baseline.md`,
  `docs/research/findings/2026-XX-XX-p0-gate.md`

**Interfaces:** yok (içerik + rapor).

- [ ] **Step 1: README düzeltmeleri** — (a) "exact ranking ... real ColPali-style
  scoring, not an approximation" → "exact **within the binary code space**; native
  float ColPali skoruna göre bir yaklaşıklıktır (kayıp C1/C2 ablasyonunda ölçülmüştür)";
  (b) mimari diyagramda Stage-1 kutusu "exhaustive binary MaxSim" ile değişir;
  (c) skor açıklamasına "uncalibrated similarity (128 − 2·Hamming, sorgu tokenı başına
  ortalama) — güven/olasılık değildir" notu; (d) v0 limitations bölümüne P0 bulgu
  özeti + rapor linki.
- [ ] **Step 2: UI etiketi** — `index.html` skor dipnotu:
  "skor: kalibre edilmemiş benzerlik (MaxSim); güven ya da doğruluk yüzdesi DEĞİLDİR".
  Eşik çizgisi tooltip'i aynı ifadeyle güncellenir. `config.py` `min_score_threshold`
  yorumu: "kaba v0 kalıntısı; kalibrasyon P2'de — bu değer güven ölçüsü değildir".
- [ ] **Step 3: Baseline raporu yaz** — `p0-baseline.md`: spec §1.1 tablosu + bu plan
  koşumlarının EvalReport künyeleri; mevcut mimarinin (v0) canary sonuçları
  (`bench run --pipeline two-stage` ile v0 davranışı yeniden ölçülür) vs yeni hat.
- [ ] **Step 4: Gate raporu yaz** — `p0-gate.md`: master §5 G0.1-G0.9 satır satır,
  her satırda sayı + koşum `run_id`. Tümü PASS değilse eksikler ve karar.
- [ ] **Step 5: Full regression + slow canary** — Run:
  `uv run pytest -q -m "not slow" && make lint && uv run pytest -m slow -v`
- [ ] **Step 6: Commit** — `docs: honest scoring/README language + P0 baseline and gate reports`

---

### Task 15: Hijyen borcu üçlüsü (Plan 2 T1 devri — bağımsız regresyon ağı)

**Files:**
- Modify: `tests/corpus/test_manifest.py`, `tests/test_cli.py`, `pyproject.toml`,
  gerekirse `src/belge_gozu/corpus/manifest.py` (`build_ssl_context` refactor'u)

Bu task Plan 2 Task 1'in birebir devridir (içerik orada tam yazılıdır:
shipped-manifest parse + benzersiz id testi, `build_ssl_context()` çıkarımıyla TLS
doğrulama testi, 3 belge × 7 sayfa çok-chunk hizalama testi, SWIG uyarı filtresi).
Sıralamadan bağımsızdır; herhangi bir task'ten önce veya sonra koşulabilir.

- [ ] **Step 1: Testleri Plan 2 T1'deki kod bloklarından aynen ekle, RED gör** —
  Run: `uv run pytest tests/corpus/test_manifest.py tests/test_cli.py -v`
- [ ] **Step 2: `build_ssl_context` refactor'unu yap, GREEN gör** — Run: aynı komut
- [ ] **Step 3: Uyarı filtresi + tam süit** — Run: `uv run pytest -q -m "not slow"`
  (uyarı 0) `&& make lint`
- [ ] **Step 4: Commit** — `test: shipped-manifest, TLS-context and multichunk regression nets`

---

## P0 Tamamlanma Kapısı (go/no-go)

Master plan §5 G0.1-G0.9. Ek açık kurallar:

- **G0.3 yorumu:** üretim yolu exhaustive olduğundan Recall@candidate tanım gereği
  %100'dür; herhangi bir aday-üreteci (Stage-1 varyantı, PLAID-tarzı) üretime ancak
  canary üzerinde gold Recall@candidate ≥ %98 ölçümüyle dönebilir — aksi halde flag
  kapalı kalır.
- **Kuantizasyon kuralı (G0.7):** float16 oracle'a göre Recall@20 kaybı ≤ 2 puan
  olmayan hiçbir temsil "tek üretim gerçeği" ilan edilemez.
- **No-go durumunda:** eksik kalan satırlar p0-gate raporunda "FAIL + neden + plan"
  ile listelenir; P1'in default entegrasyonu başlamaz (deneysel, flag-kapalı P1 kodu
  yazılabilir).

## Self-Review (yazar kontrolü)

1. **Spec kapsaması:** brief'in P0 listesi ↔ task eşlemesi: benchmark veri modeli (T6),
   canary 30-50 (T10), geniş benchmark hedefi (P1 T12'ye işaretle devredildi — master §2),
   gold doc/page/article/span (T6), law-grouped split (T6), corpus coverage (T8
   `missing_gold_pages` + G0.1), stage ranks + candidate survival (T8), Stage-1
   Recall@candidate (T8 TwoStageDiagnosticAdapter + G0.3), exhaustive oracle (T5/T8),
   native float oracle (T9), compression ablasyonu (T12), mean-sign kaldırma (T5),
   processor A/B reindex (T11), ST MultiVectorEncoder karşılaştırması (T11 Step 2),
   prompt/augmentation ablasyonu (T11 Step 5), padding/mask (T2/T3), batch determinizmi
   (T2 Step 5), manifest + serve check (T1/T4), golden semantic regression (T10),
   README "exact" düzeltmesi (T14), UI skor etiketi (T14), kalite telemetri şeması +
   çakışmasız entegrasyon (T13).
2. **Placeholder taraması:** "TBD/TODO/uygun test ekle" yok; T11/T12 runbook adımları
   dahil her adımda komut + beklenen sonuç + karar kuralı var. `TRAIN_COMPAT_V1`
   sabitinin T11 Step 1'de birincil kaynağa karşı doğrulanması bir belirsizlik değil,
   tanımlı bir doğrulama adımıdır.
3. **Tip tutarlılığı:** `QueryFormat/IndexManifest` (T1) → T2/T3/T4/T9/T12/T13 aynı
   adlarla; `StageRecord/QuestionDiagnostic/MetricBlock/EvalReport` (T8) → T9 CLI ve
   T14 raporları; `FloatIndex` (T9) → T12 `derive_*` imzaları birebir.
4. **Bağımlılık:** T2→T3→(T4,T5)→(T8,T9)→(T10,T11,T12); T6/T7 bağımsız erken; T13
   T1+T5 sonrası; T14 en son; T15 serbest. Döngü yok.
