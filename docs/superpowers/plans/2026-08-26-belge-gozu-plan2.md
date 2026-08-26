# Belge-Gözü Plan 2 (Kalite Katmanları + Benchmark) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** v0'ın kanıtlanmış zayıflığını (sayfa-hassasiyeti: 17 canlı soruda 1 doyurucu yanıt) ölçülebilir biçimde kapatmak: `belge-gozu-bench` benchmark'ı + OCR baseline + sorgu yeniden yazımı + VLM rerank + LocalVLM yolu — her katman açık/kapalı ablasyonla sayıya bağlanır.

**Architecture:** Mevcut tak-çıkar arayüzlerin üstüne eklemeler: `Retriever` kademesine config-kapılı 3. basamak (VLM rerank), retrieval öncesine config-kapılı yeniden yazım, `Answerer`'a LocalVLM uygulaması, yeni `bench/` paketi (veri seti + NDCG/Recall harness + OCR-baseline hattı). Benchmark'ın manşet metriği retrieval'dır (NDCG@5/Recall@5 — API kotasından bağımsız); uçtan uca yanıt skoru, yanıtlayıcı stratejisi netleşene dek sınırlı/niteliksel tutulur.

**Tech Stack:** v0 stack'i + pytesseract/pdftotext (baseline OCR), sentence-transformers değil — baseline metin gömme için `model2vec` yerine basit ve savunulabilir seçim: `bge-m3` yerine HF `intfloat/multilingual-e5-small` (CPU-dostu; inşa günü doğrulanır), llama.cpp (LocalVLM, `ml` extra'sına eklenmez — ayrı `local` extra).

**Spec:** `docs/superpowers/specs/2026-08-25-belge-gozu-design.md`

## Global Constraints

- v0 kuralları aynen geçerli: CI'da test ağ/GPU/model kullanmaz (`-m "not slow"`); ruff 100 + pyright basic; her task kendi commit'iyle biter; `.env`/anahtar asla yazdırılmaz/commit'lenmez.
- Ana boru hattında OCR yok — OCR YALNIZ `bench/baseline/` altında yaşar (karşılaştırma rakibi olarak; tez bozulmaz).
- Yeni katmanlar (rewrite, rerank) config'te varsayılan KAPALI doğar; ancak benchmark kazancı kanıtlanınca varsayılan açılır (ayrı commit, sayı referanslı).
- Benchmark dilim hedefleri (spec §7): (a) metin ~40, (b) tablo/düzen ~30, (c) tarihî tarama ~30; her soru kullanıcı doğrulamasından geçmeden yayınlanmaz (ana dilimlerde ≥25 doğrulanmış soru altında sonuç yayınlanmaz). Multi-hop dilimi (d) Plan 3'tedir.
- Gemini kotası (≈20 çağrı/gün) planın hiçbir CI/test adımında harcanmaz; koşum adımları (runbook) kota bütçesini açıkça belirtir.
- Kontrolcü kararı (final inceleme triyajından devralınan borç): hijyen üçlüsü bu planın 1. task'ıdır.
- SCOPE DEĞİŞİKLİĞİ (spec §10'dan sapma, gerekçeli): tam korpus 20-30 bin sayfa genişletmesi bu planda YOK — benchmark bütünlüğü için korpus sabit tutulur (yalnız 1475 sayılı Kanun eklenir: canlı testte iki sorunun gerektirdiği kanıtlanmış eksik). Büyük genişleme Plan 3 sonrasına.

## File Structure

```
src/belge_gozu/
  bench/__init__.py
  bench/dataset.py      # BenchQuestion, load_bench, doğrulama kuralları
  bench/metrics.py      # ndcg_at_k, recall_at_k
  bench/harness.py      # run_retrieval_eval: konfigürasyonlu koşum + JSON rapor
  bench/baseline.py     # OCR+metin-RAG rakibi: extract, chunk, embed, search
  rewrite.py            # QueryRewriter protokolü + LLMRewriter + NoopRewriter (önbellekli)
  rerank.py             # VLMReranker: sayfa görüntüsünü soru ile puanlar
  answer/local.py       # LocalVLMAnswerer (llama.cpp server'a OpenAI-uyumlu istemci)
data/bench/v1_draft.csv     # taslak sorular (ajan-üretimi, kullanıcı doğrulayacak)
data/bench/v1.csv           # DOĞRULANMIŞ benchmark (yayınlanacak olan)
tests/bench/..., tests/test_rewrite.py, tests/test_rerank.py, tests/answer/test_local.py
```

Mevcut dosyalara dokunuşlar: `config.py` (yeni alanlar), `retrieval/core.py` (rerank kancası), `app/main.py` (rewrite/rerank config-kapılı entegrasyonu), `cli.py` (`bench` komutları), `pyproject.toml` (`bench` + `local` extra'ları).

---

### Task 1: Hijyen borcu üçlüsü + uyarı filtresi

**Files:**
- Modify: `tests/corpus/test_manifest.py`, `tests/test_cli.py`, `pyproject.toml`
- Create: yok

**Interfaces:**
- Consumes: `load_manifest`, `build_http_client` (v0'dan), `FakeEncoder`, CLI `index build --fake`
- Produces: regresyon ağı — davranış değişikliği yok

- [ ] **Step 1: Üç başarısız/eksik testi yaz**

`tests/corpus/test_manifest.py`'ye ekle:

```python
def test_shipped_manifest_parses_and_ids_unique():
    rows = load_manifest(Path("data/manifest/v0_manifest.csv"))
    ids = [r.doc_id for r in rows]
    assert len(ids) == len(set(ids))
    assert len(rows) >= 56


def test_http_client_keeps_tls_verification():
    import ssl

    from belge_gozu.corpus.manifest import build_http_client

    client = build_http_client()
    ctx = client._transport._pool._ssl_context  # type: ignore[attr-defined]
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
```

Not: `_transport` erişimi kırılgansa `build_http_client`'a test edilebilirlik için
`ssl_context`'i döndüren küçük bir refactor yap (`build_ssl_context()` ayrı fonksiyon,
client onu kullanır; test doğrudan `build_ssl_context()`'i doğrular — tercih edilen yol).

`tests/test_cli.py`'ye çok-chunk hizalama testi (mevcut `make_pdf` yardımıyla, 8'lik
chunk sınırını aşan 3 belge × 7 sayfa = 21 sayfa):

```python
def test_fake_build_multichunk_alignment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BG_INDEX_DIR", str(tmp_path / "index"))
    (tmp_path / "manifest").mkdir(parents=True)
    csv = "doc_id,doc_name,doc_type,url\n" + "\n".join(
        f"d{i},Belge {i},kanun,https://example.org/d{i}.pdf" for i in range(3)
    )
    (tmp_path / "manifest" / "v0_manifest.csv").write_text(csv, encoding="utf-8")
    (tmp_path / "pdf").mkdir()
    for i in range(3):
        make_pdf(tmp_path / "pdf" / f"d{i}.pdf", pages=7)
    assert runner.invoke(app, ["corpus", "render", "--dpi", "72"]).exit_code == 0
    assert runner.invoke(app, ["index", "build", "--fake"]).exit_code == 0

    import json

    import numpy as np
    import pandas as pd
    from PIL import Image

    from belge_gozu.index.encode import FakeEncoder
    from belge_gozu.index.store import PackedIndex, binarize_pack

    idx = PackedIndex.load(tmp_path / "index", mmap=False)
    meta = pd.read_parquet(tmp_path / "index" / "meta.parquet")
    assert idx.page_ids == meta.page_id.tolist()  # sıra birebir
    enc = FakeEncoder()
    # rastgele 3 sayfanın embedding'i, bağımsız yeniden-encode ile birebir aynı mı?
    for pos in (0, 10, 20):
        img = Image.open(tmp_path / meta.iloc[pos]["image_path"]).convert("RGB")
        expected = binarize_pack(enc.encode_pages([img])[0])
        np.testing.assert_array_equal(idx.page_tokens(pos), expected)
```

`pyproject.toml` `[tool.pytest.ini_options]`'a SWIG gürültü filtresi:

```toml
filterwarnings = [
  "ignore:builtin type Swig:DeprecationWarning",
  "ignore:builtin type swigvarlink:DeprecationWarning",
]
```

- [ ] **Step 2: RED gör** — Run: `uv run pytest tests/corpus/test_manifest.py tests/test_cli.py -v` (yeni testler FAIL/ERROR; TLS testi refactor öncesi attribute hatası verebilir — beklenen)
- [ ] **Step 3: Gerekli minimal refactor** — `manifest.py`'de `build_ssl_context() -> ssl.SSLContext` fonksiyonunu çıkar (mevcut context kurulumunu taşı), `build_http_client` onu çağırsın; TLS testi `build_ssl_context()`'i doğrulasın. Davranış değişmez.
- [ ] **Step 4: GREEN + temiz çıktı gör** — Run: `uv run pytest -q` (uyarı sayısı 0 olmalı) ve `make lint`
- [ ] **Step 5: Commit** — `test: shipped-manifest, TLS-context and multichunk regression nets`

---

### Task 2: Benchmark veri modeli (`bench/dataset.py`)

**Files:**
- Create: `src/belge_gozu/bench/__init__.py`, `src/belge_gozu/bench/dataset.py`, `tests/bench/test_dataset.py`

**Interfaces:**
- Produces: `BenchQuestion` (pydantic: `qid: str`, `soru: str`, `dogru_sayfalar: list[str]` (≥1, `page_id` formatı `dok:sayfa`), `referans_yanit: str`, `dilim: Literal["metin","tablo","tarihi"]`, `zorluk: Literal["kolay","orta","zor"]`, `dogrulandi: bool`); `load_bench(path: Path, only_validated: bool = True) -> list[BenchQuestion]` — CSV okur, `only_validated=True` iken `dogrulandi=False` satırları ATLAR ve sonunda dilim başına doğrulanmış sayıyı döndürmek için `bench_stats(questions) -> dict[str, int]`; bozuk satır/boş dosya `ValueError`.

- [ ] **Step 1: Başarısız testleri yaz** — `tests/bench/test_dataset.py`

```python
from pathlib import Path

import pytest

from belge_gozu.bench.dataset import BenchQuestion, bench_stats, load_bench

CSV = """qid,soru,dogru_sayfalar,referans_yanit,dilim,zorluk,dogrulandi
q1,Kira artışı sınırı nedir?,k6098:120;k6098:121,TÜFE oranıyla sınırlıdır,metin,orta,true
q2,Tablodaki oran kaçtır?,k213:45,Yüzde ondur,tablo,zor,false
"""


def test_load_validated_only(tmp_path: Path):
    p = tmp_path / "b.csv"
    p.write_text(CSV, encoding="utf-8")
    qs = load_bench(p)
    assert [q.qid for q in qs] == ["q1"]
    assert qs[0].dogru_sayfalar == ["k6098:120", "k6098:121"]


def test_load_all_and_stats(tmp_path: Path):
    p = tmp_path / "b.csv"
    p.write_text(CSV, encoding="utf-8")
    qs = load_bench(p, only_validated=False)
    assert len(qs) == 2
    assert bench_stats(qs) == {"metin": 1, "tablo": 1, "tarihi": 0}


def test_bad_slice_rejected(tmp_path: Path):
    p = tmp_path / "b.csv"
    p.write_text(CSV.replace("tablo", "bilinmeyen"), encoding="utf-8")
    with pytest.raises(ValueError):
        load_bench(p, only_validated=False)


def test_empty_pages_rejected():
    with pytest.raises(ValueError):
        BenchQuestion(qid="x", soru="s", dogru_sayfalar=[], referans_yanit="r",
                      dilim="metin", zorluk="kolay", dogrulandi=True)
```

- [ ] **Step 2: RED gör** — Run: `uv run pytest tests/bench -v`
- [ ] **Step 3: dataset.py yaz**

```python
import csv
import io
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError, field_validator

Dilim = Literal["metin", "tablo", "tarihi"]


class BenchQuestion(BaseModel):
    qid: str
    soru: str
    dogru_sayfalar: list[str]
    referans_yanit: str
    dilim: Dilim
    zorluk: Literal["kolay", "orta", "zor"]
    dogrulandi: bool

    @field_validator("dogru_sayfalar")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v or any(":" not in p for p in v):
            raise ValueError("dogru_sayfalar en az bir 'dok:sayfa' içermeli")
        return v


def load_bench(path: Path, only_validated: bool = True) -> list[BenchQuestion]:
    text = path.read_text(encoding="utf-8")
    out: list[BenchQuestion] = []
    for i, rec in enumerate(csv.DictReader(io.StringIO(text)), start=2):
        try:
            rec["dogru_sayfalar"] = [s for s in rec["dogru_sayfalar"].split(";") if s]  # type: ignore[index]
            rec["dogrulandi"] = rec["dogrulandi"].strip().lower() == "true"  # type: ignore[index]
            q = BenchQuestion(**rec)  # type: ignore[arg-type]
        except (ValidationError, KeyError) as e:
            raise ValueError(f"bench satır {i}: {e}") from e
        if q.dogrulandi or not only_validated:
            out.append(q)
    if not out:
        raise ValueError("bench boş (veya hiç doğrulanmış soru yok)")
    return out


def bench_stats(questions: list[BenchQuestion]) -> dict[str, int]:
    stats = {"metin": 0, "tablo": 0, "tarihi": 0}
    for q in questions:
        stats[q.dilim] += 1
    return stats
```

- [ ] **Step 4: GREEN gör** — Run: `uv run pytest tests/bench -v && make lint`
- [ ] **Step 5: Commit** — `feat: benchmark dataset model with validation gating`

---

### Task 3: Metrikler + retrieval eval harness

**Files:**
- Create: `src/belge_gozu/bench/metrics.py`, `src/belge_gozu/bench/harness.py`, `tests/bench/test_metrics.py`, `tests/bench/test_harness.py`
- Modify: `src/belge_gozu/cli.py` (`bench run` komutu)

**Interfaces:**
- Consumes: `TwoStageRetriever.search`, `load_bench`
- Produces: `ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float` (binary relevance, log2 indirimli, ideal-DCG normalizasyonlu), `recall_at_k(relevant, ranked, k) -> float`; `run_retrieval_eval(retriever, questions, k: int = 5, candidates: int = 200, rewriter=None) -> EvalReport` (pydantic: `per_slice: dict[str, SliceScores]`, `overall: SliceScores{ndcg: float, recall: float, n: int}`, `config: dict`) — soru başına `retriever.search` (varsa önce `rewriter.rewrite`) çağırır, `page_id` listesiyle skorlar; `report.to_json(path)` yazar. CLI: `belge-gozu bench run [--bench PATH] [--k 5] [--out PATH]` (FakeEncoder DEĞİL — gerçek indeksle; testler harness'ı sahte retriever ile sınar).

- [ ] **Step 1: Başarısız testleri yaz** — `tests/bench/test_metrics.py`

```python
import pytest

from belge_gozu.bench.metrics import ndcg_at_k, recall_at_k


def test_recall():
    assert recall_at_k({"a", "b"}, ["a", "x", "y"], 3) == 0.5
    assert recall_at_k({"a"}, [], 5) == 0.0


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k({"a"}, ["a", "b"], 5) == pytest.approx(1.0)
    assert ndcg_at_k({"a"}, ["b", "c"], 5) == 0.0


def test_ndcg_position_discount():
    # doğru sayfa 2. sırada: DCG = 1/log2(3), IDCG = 1 → 0.6309...
    assert ndcg_at_k({"a"}, ["b", "a"], 5) == pytest.approx(0.6309, abs=1e-3)
```

`tests/bench/test_harness.py` (sahte retriever ile — gerçek indeks gerekmez):

```python
from pathlib import Path

from belge_gozu.bench.dataset import BenchQuestion
from belge_gozu.bench.harness import run_retrieval_eval
from belge_gozu.retrieval.types import PageHit


def hit(pid: str) -> PageHit:
    return PageHit(page_id=pid, score=50.0, doc_name="B", page_no=1,
                   image_path=f"images/{pid}.webp", source_url="u")


class MapRetriever:
    def __init__(self, answers: dict[str, list[str]]):
        self.answers = answers

    def search(self, query, k=5, candidates=200):
        return [hit(p) for p in self.answers[query][:k]]


def make_q(qid, soru, pages, dilim):
    return BenchQuestion(qid=qid, soru=soru, dogru_sayfalar=pages,
                        referans_yanit="r", dilim=dilim, zorluk="orta", dogrulandi=True)


def test_eval_report(tmp_path: Path):
    qs = [make_q("q1", "s1", ["p:1"], "metin"), make_q("q2", "s2", ["p:2"], "tablo")]
    r = MapRetriever({"s1": ["p:1", "x:9"], "s2": ["x:9", "x:8"]})
    report = run_retrieval_eval(r, qs, k=2)
    assert report.per_slice["metin"].recall == 1.0
    assert report.per_slice["tablo"].recall == 0.0
    assert 0.4 < report.overall.ndcg < 0.6  # (1.0 + 0.0) / 2
    out = tmp_path / "r.json"
    report.to_json(out)
    assert out.exists() and "ndcg" in out.read_text()


def test_rewriter_applied():
    calls = []

    class Rw:
        def rewrite(self, q):
            calls.append(q)
            return "yeniden:" + q

    r = MapRetriever({"yeniden:s1": ["p:1"]})
    report = run_retrieval_eval(r, [make_q("q1", "s1", ["p:1"], "metin")], k=1, rewriter=Rw())
    assert calls == ["s1"] and report.overall.recall == 1.0
```

- [ ] **Step 2: RED gör** — Run: `uv run pytest tests/bench -v`
- [ ] **Step 3: metrics.py + harness.py yaz**

`metrics.py`:

```python
import math


def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, p in enumerate(ranked[:k]) if p in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0
```

`harness.py`:

```python
import json
from pathlib import Path

from pydantic import BaseModel

from belge_gozu.bench.dataset import BenchQuestion
from belge_gozu.bench.metrics import ndcg_at_k, recall_at_k


class SliceScores(BaseModel):
    ndcg: float = 0.0
    recall: float = 0.0
    n: int = 0


class EvalReport(BaseModel):
    per_slice: dict[str, SliceScores]
    overall: SliceScores
    config: dict

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), ensure_ascii=False, indent=1))


def run_retrieval_eval(
    retriever, questions: list[BenchQuestion], k: int = 5,
    candidates: int = 200, rewriter=None,
) -> EvalReport:
    buckets: dict[str, list[tuple[float, float]]] = {}
    for q in questions:
        query = rewriter.rewrite(q.soru) if rewriter else q.soru
        ranked = [h.page_id for h in retriever.search(query, k=k, candidates=candidates)]
        rel = set(q.dogru_sayfalar)
        buckets.setdefault(q.dilim, []).append(
            (ndcg_at_k(rel, ranked, k), recall_at_k(rel, ranked, k))
        )
    def agg(pairs: list[tuple[float, float]]) -> SliceScores:
        if not pairs:
            return SliceScores()
        return SliceScores(
            ndcg=sum(p[0] for p in pairs) / len(pairs),
            recall=sum(p[1] for p in pairs) / len(pairs),
            n=len(pairs),
        )
    all_pairs = [p for pairs in buckets.values() for p in pairs]
    return EvalReport(
        per_slice={s: agg(p) for s, p in buckets.items()},
        overall=agg(all_pairs),
        config={"k": k, "candidates": candidates, "rewriter": bool(rewriter)},
    )
```

CLI (`cli.py`'ye `bench_app = typer.Typer()` + `app.add_typer(bench_app, name="bench")`):

```python
@bench_app.command("run")
def bench_run(
    bench: Path = typer.Option(Path("data/bench/v1.csv")),  # noqa: B008
    k: int = typer.Option(5),
    out: Path = typer.Option(Path("data/bench/results/retrieval.json")),  # noqa: B008
) -> None:
    import pandas as pd

    from belge_gozu.bench.dataset import load_bench
    from belge_gozu.bench.harness import run_retrieval_eval
    from belge_gozu.index.encode import ColSmolEncoder
    from belge_gozu.index.store import PackedIndex
    from belge_gozu.retrieval.core import TwoStageRetriever

    s = _settings()
    idx = PackedIndex.load(s.index_dir)
    meta = pd.read_parquet(s.index_dir / "meta.parquet")
    retriever = TwoStageRetriever(idx, meta, ColSmolEncoder(s.retriever_model, s.device))
    report = run_retrieval_eval(retriever, load_bench(bench), k=k, candidates=s.stage1_candidates)
    report.to_json(out)
    typer.echo(f"NDCG@{k}={report.overall.ndcg:.3f} Recall@{k}={report.overall.recall:.3f} -> {out}")
```

- [ ] **Step 4: GREEN gör** — Run: `uv run pytest tests/bench -v && uv run pytest -q && make lint`
- [ ] **Step 5: Commit** — `feat: retrieval eval harness with NDCG/recall and slice breakdown`

---

### Task 4: OCR + metin-RAG baseline (`bench/baseline.py`)

**Files:**
- Create: `src/belge_gozu/bench/baseline.py`, `tests/bench/test_baseline.py`
- Modify: `pyproject.toml` (`bench` extra: `pytesseract`, `sentence-transformers` YOK — bkz. aşağı), `src/belge_gozu/cli.py` (`bench baseline-build`, `bench baseline-run`)

**Interfaces:**
- Consumes: `data/` sayfa görüntüleri + pdf'ler, meta.parquet, `run_retrieval_eval` (retriever protokolü sayesinde baseline da aynı harness'la ölçülür)
- Produces: `TextBaseline` sınıfı — `build(meta, data_dir, out_dir)` (sayfa başına metin çıkar: önce `pymupdf get_text()` [gömülü metin], boşsa pytesseract OCR `lang="tur"`; metinler `page_id`→text parquet'e), `TextBaselineRetriever(index_dir, embedder)` — `search(query, k, candidates)` PageHit döndürür (harness uyumluluğu). Embedder: `intfloat/multilingual-e5-small` sentence-transformers YERİNE düz `transformers` + mean-pooling ile (bağımlılık şişkinliği yok; inşa günü model adı doğrulanır); embedding'ler build sırasında hesaplanıp `.npy` olarak saklanır, sorgu embedding'i çalışma anında.
- Testler: OCR ve model YOK — `FakeTextEmbedder` (sha256-deterministik, FakeEncoder kalıbı) + sahte metin parquet'iyle retriever mantığı (kosinüs top-k) sınanır; gerçek OCR/embed yolu `-m slow` işaretli tek bir smoke testte (CI'da koşmaz).

- [ ] **Step 1: Başarısız testleri yaz** — `tests/bench/test_baseline.py`: `FakeTextEmbedder` ile 4 sayfalık sahte korpusta, sorgu embedding'i sayfa-2 embedding'ine eşitken `search`'ün sayfa-2'yi 1. sırada PageHit olarak döndürdüğünü ve `k` sınırını doğrula; boş metinli sayfanın indekste yer almadığını doğrula. (Kod: Task 2-3 kalıplarıyla aynı yapıda, `np.float32` kosinüs.)
- [ ] **Step 2: RED gör** — `uv run pytest tests/bench/test_baseline.py -v`
- [ ] **Step 3: baseline.py yaz** — `extract_page_texts(meta, data_dir) -> pd.DataFrame(page_id,text)` (pymupdf → boşsa pytesseract; pytesseract importu lazy, `bench` extra), `TextEmbedder` protokolü + `HFTextEmbedder` (lazy transformers) + normalize edilmiş mean-pooling, `TextBaselineRetriever` (np.dot kosinüs, PageHit üretimi meta'dan). CLI komutları: `bench baseline-build` (metin çıkar + embed + kaydet; uzun — ilerleme yazdırır), `bench baseline-run` (harness'ı TextBaselineRetriever ile koşturur, JSON'a `baseline_` önekiyle yazar).
- [ ] **Step 4: GREEN gör** — `uv run pytest -q && make lint` (slow işaretliler hariç hepsi)
- [ ] **Step 5: Commit** — `feat: OCR+text-RAG baseline measurable under the same harness`

---

### Task 5: Sorgu yeniden yazımı (`rewrite.py`)

**Files:**
- Create: `src/belge_gozu/rewrite.py`, `tests/test_rewrite.py`
- Modify: `src/belge_gozu/config.py` (`rewrite_enabled: bool = False`, `rewrite_cache_path: Path = Path("data/rewrite_cache.json")`), `src/belge_gozu/app/main.py` (config açıksa /search ve /ask sorgusunu rewriter'dan geçir)

**Interfaces:**
- Produces: `QueryRewriter` Protocol: `rewrite(query: str) -> str`; `NoopRewriter`; `LLMRewriter(client, cache_path)` — `client` = `generate(prompt: str, images: list[bytes]) -> str` imzalı herhangi bir nesne (GeminiClient uyar); prompt: günlük Türkçe soruyu mevzuat terminolojisine çevirir, TEK satır döndürür; sonuç `{query: rewritten}` JSON önbelleğine atomik yazılır, önbellek isabetinde istemci HİÇ çağrılmaz; istemci hatasında orijinal sorgu döner (graceful).
- Testler: MagicMock istemci — (a) çeviri döner ve önbelleğe yazılır, (b) ikinci çağrıda istemci çağrılmaz, (c) istemci raise ederse orijinal döner, (d) NoopRewriter kimliktir. app/main entegrasyon testi: `BG_REWRITE_ENABLED=true` + sahte rewriter enjeksiyonu (create_app'e `rewriter=None` parametresi eklenir; None + enabled → LLMRewriter(GeminiClient) kurulur, testlerde stub verilir).

- [ ] **Step 1: Testleri yaz (RED)** — yukarıdaki 4 birim test + app testi (`/search` gövdesindeki sorgunun stub rewriter'dan geçtiğini, stub'ın "X"→"Y" çevirisiyle arama sonucunun değiştiğini FakeEncoder determinizmiyle doğrula).
- [ ] **Step 2: rewrite.py + config + main.py entegrasyonu yaz** — LLMRewriter promptu:

```python
REWRITE_PROMPT = (
    "Aşağıdaki günlük dildeki Türkçe hukuk sorusunu, Türk mevzuatında geçen resmi "
    "terminolojiyle tek satırlık bir arama sorgusuna çevir. Sadece sorguyu döndür.\n"
    "Soru: {q}"
)
```

- [ ] **Step 3: GREEN + lint** — `uv run pytest -q && make lint`
- [ ] **Step 4: Commit** — `feat: cached query rewriting layer, disabled by default`

---

### Task 6: VLM reranker (`rerank.py`) — kademenin 3. basamağı

**Files:**
- Create: `src/belge_gozu/rerank.py`, `tests/test_rerank.py`
- Modify: `src/belge_gozu/config.py` (`rerank_enabled: bool = False`, `rerank_pool: int = 20`), `src/belge_gozu/retrieval/core.py` (`TwoStageRetriever.__init__`'e opsiyonel `reranker=None`; `search` sonunda reranker varsa top-`rerank_pool` PageHit'i ona verir), `src/belge_gozu/app/main.py` (config-kapılı kurulum)

**Interfaces:**
- Produces: `Reranker` Protocol: `rerank(question: str, hits: list[PageHit], image_loader, k: int) -> list[PageHit]`; `VLMReranker(client)` — her aday için client'a sayfa görüntüsü + `RERANK_PROMPT` ("Bu sayfa şu soruya cevap içeriyor mu? 0-10 arası tek sayı döndür: {q}") gönderir, sayıyı parse eder (parse edilemeyen → 0), skora göre yeniden sıralar, orijinal `score` alanını `rerank_score`la DEĞİŞTİRMEZ (PageHit'e `rerank_score: float | None = None` alanı eklenir — geriye uyumlu default).
- Testler: MagicMock client sabit puanlarla — sıralamanın puanlara göre değiştiği, k kesimi, parse hatasında 0, client exception'ında orijinal sıranın korunduğu (graceful). Retriever entegrasyonu: stub reranker'ın çağrıldığı ve çıktısının döndüğü.

- [ ] **Step 1: Testleri yaz (RED)** — yukarıdaki davranışlar + `PageHit.rerank_score` default None geriye-uyumluluk testi (mevcut testler kırılmamalı).
- [ ] **Step 2: rerank.py + core.py kancası + config + main.py yaz** (client çağrıları aday başına — 20 çağrı/sorgu maliyeti config'te belgelenir; kota notu: canlı kullanımda rerank yalnız ücretli/lokal istemciyle açılır).
- [ ] **Step 3: GREEN + lint** — tüm süit + `make lint`
- [ ] **Step 4: Commit** — `feat: VLM reranker as optional cascade stage 3`

---

### Task 7: LocalVLM yanıtlayıcı (`answer/local.py`)

**Files:**
- Create: `src/belge_gozu/answer/local.py`, `tests/answer/test_local.py`
- Modify: `pyproject.toml` (`local` extra: `openai>=1.40` — llama.cpp server'ın OpenAI-uyumlu ucu için istemci), `src/belge_gozu/config.py` (`answerer: Literal["gemini","local"] = "gemini"`, `local_vlm_url: str = "http://127.0.0.1:8080/v1"`, `local_vlm_model: str = "qwen2.5-vl-3b"`), `src/belge_gozu/app/main.py` (config'e göre answerer seçimi)

**Interfaces:**
- Produces: `LocalVLMAnswerer(base_url, model, client=None)` — `Answerer` protokolünü uygular; sayfa görüntülerini base64 data-URI olarak OpenAI-uyumlu `chat.completions` çağrısına koyar; prompt ve atıf parse'ı `gemini.py`'dekiyle AYNI (`build_prompt` ve regex oradan import edilir — kopyalanmaz); istemci hatası AskService guard'ına düşer (yakalamaz).
- Runbook notu (koşum, kod değil): lokal sunucu `llama-server -m <qwen2.5-vl-3b-gguf> --port 8080` ile açılır; model GGUF'u inşa günü HF'ten seçilir (Task 10'da canlı doğrulanır).
- Testler: MagicMock OpenAI istemcisi — mesaj yapısında görüntülerin data-URI olarak yer aldığı, yanıt metninden atıf parse'ının `[S2]`→page_id eşlemesi, fallback top-1.

- [ ] **Step 1: Testleri yaz (RED)** → **Step 2: local.py yaz** → **Step 3: GREEN + lint** → **Step 4: Commit** — `feat: local VLM answerer via OpenAI-compatible endpoint`

---

### Task 8: Korpus eki — 1475 sayılı Kanun + indeks deltası (runbook)

Kod yok; operasyon. Kanıt: canlı testte iki kıdem tazminatı sorusunun dayanağı (1475 m.14) indekste yoktu.

- [ ] **Step 1:** `data/manifest/v0_manifest.csv`'ye `k1475,İş Kanunu (1475 - yürürlükteki 14. madde),kanun,https://www.mevzuat.gov.tr/mevzuatmetin/1.3.1475.pdf` satırı ekle (tertip segmentini `corpus probe` ile doğrula; 1475 eski kanun — 1.5 değil 1.3/1.4 olabilir, probe hangisi 200 veriyorsa o). Commit: `data: add 1475 sayılı Kanun (kıdem tazminatı basis)`
- [ ] **Step 2:** `uv run belge-gozu corpus download && uv run belge-gozu corpus render` (idempotent — yalnız yeni belge iner/render olur; sayfa sayısını raporla)
- [ ] **Step 3:** `BG_DEVICE=mps uv run belge-gozu index build` (fp16+batch; ~52 dk/4222 sayfa hızında tam yeniden kurulum — kabul: build artımlı değil, bilinen v0 sınırı) ve `uv run belge-gozu index push` (indeks + yeni görüntüler HF'e)
- [ ] **Step 4:** Doğrula: yeni sayfa sayısı = eski + 1475'in sayfaları; `data/index/meta.parquet`'te `k1475:*` var; HF reposunda güncel.

---

### Task 9: Benchmark taslakları (ajan üretimi) + kullanıcı doğrulama turu (runbook)

Kod yok; içerik üretimi + İNSAN kapısı. Spec §7: taslakları Claude üretir, HER çifti kullanıcı doğrular.

- [ ] **Step 1 (ajanlar):** Korpusdan örneklenmiş sayfa görüntülerini OKUYARAK (görüntü dosyaları lokalde; API kotası gerekmez — çok-modelli ajanlar sayfayı doğrudan okur) `data/bench/v1_draft.csv` üret: hedef ~120 taslak (metin ~48, tablo ~36, tarihi ~36), şema Task 2'deki CSV başlığı, `dogrulandi=false`. Her soru için `dogru_sayfalar` ajanın okuduğu somut sayfa(lar); `referans_yanit` sayfadan birebir dayanaklı. Çeşitlilik: kolay/orta/zor karışımı, farklı kanunlar, tarihî dilim RG taramalarından.
- [ ] **Step 2 (İNSAN — kullanıcı):** Kullanıcı `v1_draft.csv`'yi gözden geçirir: doğru bulduğu satırlarda `dogrulandi=true` yapar, bozukları düzeltir/siler. Hedef: dilim başına ≥25 (tarihi ≥25, toplam ≥85) doğrulanmış soru. Bu adım kullanıcı tamamlandı diyene kadar bloktur — sonuç `data/bench/v1.csv`'ye kopyalanıp commit edilir: `data: belge-gozu-bench v1 (user-validated)`
- [ ] **Step 3:** `uv run pytest tests/bench -q` (shipped bench parse testi Task 2'dekiyle; gerekirse `test_shipped_bench_parses` eklenir) + `belge-gozu bench run` İLK resmi koşum → `data/bench/results/retrieval.json` commit.

---

### Task 10: Kalibrasyon + ablasyon koşuları + sonuç tabloları (runbook)

- [ ] **Step 1:** Ablasyon koşuları (hepsi retrieval-metrik — API kotası gerektirmez, rewrite hariç):
  - k ∈ {5, 10}; rewrite {kapalı, açık — LLMRewriter, kota bütçesi: ~85 soru × 1 çağrı, önbellekli; Gemini günlük kota × 5 güne bölünebilir VEYA LocalVLM istemciyle}; rerank {kapalı, açık — LocalVLM istemciyle (kota yakmaz), pool=20}; baseline (Task 4) aynı sorularla.
- [ ] **Step 2:** Eşik kalibrasyonu: doğrulanmış sorularda skor dağılımından (doğru-sayfa-top1 skorları vs alakasız-soru skorları) `min_score_threshold` seçilir; config güncellenir, gerekçe yorum satırında. Commit: `chore: calibrate abstain threshold from bench distributions`
- [ ] **Step 3:** Kazanç kanıtlanan katmanların varsayılanı açılır (`rewrite_enabled`/`rerank_enabled`), sayı referanslı commit: `feat: enable <layer> by default (+X NDCG@5 on bench)` — kanıt yoksa kapalı kalır ve negatif sonuç README'ye yazılır.
- [ ] **Step 4:** README güncellemesi: Results bölümü — görsel-RAG vs OCR-baseline tablosu (dilim kırılımlı), katman katkı tablosu (spec §7), dürüst analiz (nerede kaybediyoruz). Benchmark'ın HF Dataset olarak yayını: `barandincoguz/belge-gozu-bench` (README'li, lisans+metodoloji notlu). Commit + (kullanıcı onayıyla) `v0.2` etiketi.

---

## Self-Review (yazar kontrolü — tamamlandı)

1. **Spec kapsaması (Hafta 2):** rewrite (T5), rerank (T6), benchmark+kullanıcı doğrulaması (T2,T3,T9), OCR baseline (T4), LocalVLM (T7), eşik kalibrasyonu (T10), eval koşuları+tablolar (T10). Bilinçli sapmalar Global Constraints'te: tam-korpus genişletmesi ertelendi (yalnız 1475 eki — T8), multi-hop Plan 3'te. Hijyen borcu (T1) final incelemenin devri.
2. **Yer tutucu taraması:** T4/T5/T7'de bazı adımlar davranış-sözleşmesi düzeyinde tanımlı (tam kod bloğu yerine); her birinde imzalar, test etme yolu (Fake/Mock kalıbı) ve kabul ölçütleri açık — uygulayıcı v0'daki birebir kalıpları (FakeEncoder, MagicMock istemci, lazy import) izler. "TBD" yok.
3. **Tip tutarlılığı:** `PageHit.rerank_score: float | None = None` (T6) geriye uyumlu; harness `retriever.search` protokolüyle çalıştığından baseline (T4) ve gerçek retriever aynı kapıdan ölçülür; `LLMRewriter(client)` ve `VLMReranker(client)` aynı `generate(prompt, images)` imzasını kullanır (GeminiClient ve T7 istemcisi uyar).
4. **Kota disiplini:** CI'da sıfır API çağrısı; koşum adımlarında kota bütçeleri açık; rerank canlıda yalnız lokal/ücretli istemciyle önerilir.
