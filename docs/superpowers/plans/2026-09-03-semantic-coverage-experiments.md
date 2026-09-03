# Semantic Coverage Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a local dense page channel and then deterministic local query expansion increase paraphrase candidate-pool coverage, and render the evidence as a standalone HTML report.

**Architecture:** Keep all new behavior offline. `retrieval/dense.py` owns the official Qwen3 embedding protocol, a one-model-at-a-time memory preflight, and a validated page index; `retrieval/expand.py` owns pinned Qwen3 generation and a provenance-keyed cache. `bench/semantic_coverage.py` composes scoreless candidate lists and calculates evidence; the CLI runner loads real artefacts and writes JSON atomically. A separate pure renderer turns that JSON into one self-contained HTML report.

**Tech Stack:** Python 3.12, NumPy, pandas/Parquet, PyTorch, Transformers, Hugging Face Hub, pytest, Ruff, Pyright, inline HTML/CSS/SVG.

## Global Constraints

- Do not modify `retrieval/text.py`, `recipe_fingerprint()`, BM25 page retrieval, `min_score_threshold=10.6`, `HybridRetriever.search()`, `/search`, or `/ask`.
- All new channels are CLI/bench only; no production setting or default is added.
- Candidate lists are combined only through first-seen deduplication. No RRF or numeric score fusion.
- Dense candidates are exactly the first 50 pages. The baseline is BM25-50 + Mogan-50 + Colmm-50.
- Dense model revisions are `Qwen/Qwen3-Embedding-8B@1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` and `Qwen/Qwen3-Embedding-4B@5cf2132abc99cad020ac570b19d031efec650f2b`. Query embeddings use one fixed English legal-retrieval instruction; pages do not carry an instruction.
- Dense models use Qwen's official left-padding, last-token pooling and L2 normalization protocol. CLS/mean pooling is not an option.
- Query expansion uses `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`, `enable_thinking=False`, `do_sample=False`, and exactly one generated Turkish search variant.
- The host has 24 GiB unified memory. Before every real model arm, unload prior Transformer objects, call MPS cache release where available, then perform a one-item preflight. An OOM produces a recorded `skipped_oom` arm; it never silently selects a smaller model.
- `canary_v2` is development-only. No result may select a production configuration without a new human-verified law-group-disjoint holdout.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/belge_gozu/retrieval/dense.py` | Pinned Qwen3 specs, official last-token encoding, memory preflight, normalized page vectors, ID-aligned top-k index. |
| `src/belge_gozu/retrieval/expand.py` | Pinned Qwen query variant generator and provenance-keyed JSONL cache. |
| `src/belge_gozu/bench/semantic_coverage.py` | Pure candidate-arm composition, metrics, selection, and report-shaped dicts. |
| `scripts/eval_semantic_coverage.py` | Real artefact/model loader and atomic JSON experiment runner. |
| `scripts/render_semantic_coverage_html.py` | JSON-to-standalone HTML renderer; no model or index import. |
| `tests/retrieval/test_dense.py` | Dense index alignment/ranking contracts. |
| `tests/retrieval/test_expand.py` | Expansion validation/cache provenance contracts. |
| `tests/bench/test_semantic_coverage.py` | Arm composition, model selection, metrics and source attribution. |
| `tests/test_render_semantic_coverage_html.py` | HTML report content/escaping/schema contracts. |

### Task 1: Add a validated, scoreless dense page index

**Files:**
- Create: `src/belge_gozu/retrieval/dense.py`
- Create: `tests/retrieval/test_dense.py`

**Interfaces:**
- Produces `DenseModelSpec(repo: str, revision: str, instruction: str, max_length: int)`.
- Produces `DENSE_MODELS: dict[str, DenseModelSpec]` with keys `qwen3-embedding-8b` and `qwen3-embedding-4b`.
- Produces `DensePageIndex(page_ids: Sequence[str], embeddings: np.ndarray)` and `candidate_pages(query_embedding: np.ndarray, limit: int = 50) -> list[str]`.
- Produces `TransformerDenseEncoder(spec: DenseModelSpec, *, device: str | None = None, batch_size: int = 8)` with `encode_queries(texts: Sequence[str]) -> np.ndarray` and `encode_passages(texts: Sequence[str]) -> np.ndarray`.

- [ ] **Step 1: Write failing dense-index tests**

```python
import numpy as np
import pytest

from belge_gozu.retrieval.dense import DensePageIndex


def test_dense_index_returns_stable_descending_page_ids():
    index = DensePageIndex(["p1", "p2", "p3"], np.eye(3, dtype=np.float32))
    assert index.candidate_pages(np.array([0.1, 0.9, 0.9], dtype=np.float32), limit=2) == ["p2", "p3"]


def test_dense_index_rejects_duplicate_ids_and_bad_query_shape():
    with pytest.raises(ValueError, match="benzersiz"):
        DensePageIndex(["p1", "p1"], np.eye(2, dtype=np.float32))
    index = DensePageIndex(["p1", "p2"], np.eye(2, dtype=np.float32))
    with pytest.raises(ValueError, match="boyutu"):
        index.candidate_pages(np.ones(3, dtype=np.float32))
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/retrieval/test_dense.py -q`  
Expected: import failure for `belge_gozu.retrieval.dense`.

- [ ] **Step 3: Implement the pure index and model specs**

```python
@dataclass(frozen=True)
class DenseModelSpec:
    repo: str
    revision: str
    instruction: str
    max_length: int


DENSE_MODELS = {
    "qwen3-embedding-8b": DenseModelSpec(
        "Qwen/Qwen3-Embedding-8B", "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
        "Given a Turkish legal search query, retrieve relevant passages that answer the query.", 8192,
    ),
    "qwen3-embedding-4b": DenseModelSpec(
        "Qwen/Qwen3-Embedding-4B", "5cf2132abc99cad020ac570b19d031efec650f2b",
        "Given a Turkish legal search query, retrieve relevant passages that answer the query.", 8192,
    ),
}


class DensePageIndex:
    def candidate_pages(self, query_embedding: np.ndarray, limit: int = 50) -> list[str]:
        query = _normalized_vector(query_embedding, self.embeddings.shape[1])
        order = np.argsort(-(self.embeddings @ query), kind="stable")
        return [self.page_ids[int(i)] for i in order[:limit]]
```

Validate two-dimensional finite embeddings, one row per unique page id, positive limits, query dimension, and L2-normalize both stored vectors and query. Implement Transformer loading lazily in `__init__`; use the exact-revision `AutoTokenizer(..., padding_side="left")` and `AutoModel`, `.eval()`, inference mode and MPS when available. Query text is exactly `Instruct: {instruction}\nQuery:{question}`; page text is unchanged. Pool the last non-padding hidden state according to the official Qwen implementation, then normalize. `preflight()` must encode one fixed non-empty string and raise a typed `DenseModelOutOfMemory` on device OOM after cleanup; it may not choose another checkpoint.

- [ ] **Step 4: Verify GREEN and static checks**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/retrieval/test_dense.py -q && uv run ruff check src/belge_gozu/retrieval/dense.py tests/retrieval/test_dense.py && uv run pyright src/belge_gozu/retrieval/dense.py tests/retrieval/test_dense.py`  
Expected: all tests pass; Ruff and Pyright report no errors.

- [ ] **Step 5: Commit**

```bash
git add src/belge_gozu/retrieval/dense.py tests/retrieval/test_dense.py
git commit -m "feat(retrieval): add offline dense page index"
```

### Task 2: Add deterministic local query expansion with an auditable cache

**Files:**
- Create: `src/belge_gozu/retrieval/expand.py`
- Create: `tests/retrieval/test_expand.py`

**Interfaces:**
- Produces `EXPANDER_REPO`, `EXPANDER_REVISION`, `EXPANSION_PROMPT`, `prompt_fingerprint() -> str`.
- Produces `ExpansionRecord(question_id: str, question_sha256: str, prompt_fingerprint: str, model_revision: str, expansion: str)`.
- Produces `validate_expansion(question: str, expansion: str) -> str`, `load_expansion_cache(path: Path) -> dict[str, ExpansionRecord]`, and `write_expansion_cache(path: Path, records: Sequence[ExpansionRecord]) -> None`.
- Produces `LocalQueryExpander.expand(question: str) -> str` using the pinned Qwen model.

- [ ] **Step 1: Write failing cache and validation tests**

```python
import pytest

from belge_gozu.retrieval.expand import prompt_fingerprint, validate_expansion


def test_expansion_rejects_empty_and_identity_text():
    with pytest.raises(ValueError, match="boş"):
        validate_expansion("İzin süresi nedir?", "  ")
    with pytest.raises(ValueError, match="özgün"):
        validate_expansion("İzin süresi nedir?", "İzin süresi nedir?")


def test_prompt_fingerprint_changes_when_prompt_changes(monkeypatch):
    import belge_gozu.retrieval.expand as expand

    before = prompt_fingerprint()
    monkeypatch.setattr(expand, "EXPANSION_PROMPT", "başka istem")
    assert prompt_fingerprint() != before
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/retrieval/test_expand.py -q`  
Expected: import failure because `retrieval.expand` does not exist.

- [ ] **Step 3: Implement prompt, strict parsing, and atomic JSONL cache**

```python
EXPANDER_REPO = "Qwen/Qwen3-8B"
EXPANDER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPANSION_PROMPT = (
    "Tek satır Türkçe hukukî arama varyantı yaz. Anlamı koru; cevap, delil, "
    "madde numarası veya olmayan kanun adı uydurma. Yalnız varyantı yaz."
)


def validate_expansion(question: str, expansion: str) -> str:
    value = " ".join(expansion.split())
    if not value:
        raise ValueError("genişletme boş")
    if value.casefold() == " ".join(question.split()).casefold():
        raise ValueError("genişletme özgün sorguyla aynı")
    return value
```

`LocalQueryExpander` only loads `AutoModelForCausalLM` and `AutoTokenizer` after construction; build messages with `apply_chat_template(..., enable_thinking=False)`, call `generate(do_sample=False, max_new_tokens=64)`, decode only newly generated tokens, and validate. Before load it calls the shared Transformer cleanup; it preflights one fixed question. A device OOM is exposed as `ExpansionModelOutOfMemory` and the runner records `skipped_oom`; no smaller model is selected. Cache loading must reject duplicate question ids and any record whose question hash, model revision, or prompt fingerprint differs from the active run. Cache writing uses `NamedTemporaryFile` plus `os.replace`.

- [ ] **Step 4: Verify GREEN and commit**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/retrieval/test_expand.py -q && uv run ruff check src/belge_gozu/retrieval/expand.py tests/retrieval/test_expand.py && uv run pyright src/belge_gozu/retrieval/expand.py tests/retrieval/test_expand.py`  
Expected: all tests pass and static checks are clean.

```bash
git add src/belge_gozu/retrieval/expand.py tests/retrieval/test_expand.py
git commit -m "feat(retrieval): add deterministic local query expansion"
```

### Task 3: Build the pure semantic-coverage evaluator

**Files:**
- Create: `src/belge_gozu/bench/semantic_coverage.py`
- Create: `tests/bench/test_semantic_coverage.py`

**Interfaces:**
- Produces `CandidateChannel` protocol: `candidate_pages(query: str, limit: int) -> list[str]`.
- Produces `evaluate_coverage(questions, bm25_pages, channels, *, limit=50) -> dict[str, object]`.
- Produces `select_dense_arm(arms: Mapping[str, Mapping[str, object]]) -> str`.
- Produces diagnostics with `question_id`, `slice`, `gold_page_ids`, `candidate_pool`, and `gold_sources` mapping each retrieved gold page to its source names.

- [ ] **Step 1: Write failing composition/selection tests**

```python
def test_evaluator_records_unique_gold_source_and_full_pool_coverage():
    report = evaluate_coverage(
        questions=[Question("q1", "soru", ["gold"], "paraphrase")],
        bm25_pages=lambda _: ["b1"],
        channels={"dense": FixedChannel(["gold"]), "mogan": FixedChannel(["b1"])},
    )
    assert report["overall"]["coverage"] == 1.0
    assert report["diagnostics"][0]["gold_sources"] == {"gold": ["dense"]}


def test_dense_selection_prioritizes_paraphrase_then_overall_then_disk_then_latency():
    assert select_dense_arm({"a": arm(para=1, overall=1, disk=9, p50=9), "b": arm(para=1, overall=1, disk=8, p50=1)}) == "b"
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/bench/test_semantic_coverage.py -q`  
Expected: import failure for `bench.semantic_coverage`.

- [ ] **Step 3: Implement scoreless arms and metrics**

```python
def evaluate_coverage(questions, bm25_pages, channels, *, limit=50):
    for question in answerable_questions(questions):
        sources = {"bm25": bm25_pages(question.question)}
        sources.update({name: channel.candidate_pages(question.question, limit) for name, channel in channels.items()})
        pool = build_candidate_pool(sources["bm25"], [pages for name, pages in sources.items() if name != "bm25"], limit=limit)
        # coverage uses len(pool); R@5/R@20/R@50 retain the pool's declared order as diagnosis only.
```

Use existing `recall_at_k`, `mrr`, `ndcg_at_k`, and `bootstrap_ci`. Verify every channel page id occurs in the known page-id set before scoring metrics. `select_dense_arm` must sort exact tuple `(-paraphrase_coverage, -overall_coverage, disk_bytes, latency_p50_ms, arm_name)` and return the first key; no metric tuning after the result.

- [ ] **Step 4: Verify GREEN and commit**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/bench/test_semantic_coverage.py -q && uv run ruff check src/belge_gozu/bench/semantic_coverage.py tests/bench/test_semantic_coverage.py && uv run pyright src/belge_gozu/bench/semantic_coverage.py tests/bench/test_semantic_coverage.py`  
Expected: focused tests pass and static checks are clean.

```bash
git add src/belge_gozu/bench/semantic_coverage.py tests/bench/test_semantic_coverage.py
git commit -m "feat(bench): evaluate semantic candidate coverage"
```

### Task 4: Add the real, provenance-bearing sequential experiment runner

**Files:**
- Create: `scripts/eval_semantic_coverage.py`
- Modify: `tests/bench/test_semantic_coverage.py`

**Interfaces:**
- Command: `uv run python scripts/eval_semantic_coverage.py --bench data/bench/canary_v2.jsonl --min-verification human --out data/bench/results/semantic-coverage-dev-v1.json --cache data/bench/cache/semantic-expansions-v1.jsonl`.
- Produces baseline, two dense arms, one selected-dense-plus-expansion arm, model/artefact provenance, disk bytes and p50/p95 latency in atomic JSON.

- [ ] **Step 1: Write a failing loader test with fake encoders**

```python
def test_runner_rejects_dense_page_ids_that_do_not_match_page_texts(tmp_path):
    with pytest.raises(ValueError, match="page_texts"):
        build_dense_index(["p1"], {"p2": "metin"}, FakeDenseEncoder())
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/bench/test_semantic_coverage.py -q`  
Expected: import failure for `build_dense_index` from the runner module.

- [ ] **Step 3: Implement loader, sequential model selection, and provenance**

```python
def build_dense_index(page_ids: list[str], page_texts: Mapping[str, str], encoder: TransformerDenseEncoder) -> DensePageIndex:
    if list(page_texts) != page_ids:
        raise ValueError("page_texts page_ids ile birebir hizalı olmalı")
    return DensePageIndex(page_ids, encoder.encode_passages([page_texts[pid] for pid in page_ids]))


def run_dense_arms(
    questions: Sequence[Question],
    bm25_pages: Callable[[str], list[str]],
    mogan: CandidateChannel,
    colmm: CandidateChannel,
    dense_channels: Mapping[str, CandidateChannel],
) -> tuple[dict[str, dict[str, object]], str | None]:
    arms = {
        "baseline": evaluate_coverage(
            questions, bm25_pages, {"mogan": mogan, "colmm": colmm}
        )
    }
    for name, dense in dense_channels.items():
        arms[f"dense:{name}"] = evaluate_coverage(
            questions, bm25_pages, {"mogan": mogan, "colmm": colmm, "dense": dense}
        )
    selectable = {name: arms[f"dense:{name}"] for name in dense_channels}
    return arms, select_dense_arm(selectable) if selectable else None
```

Load `page_texts.parquet` through `load_page_texts`, BM25 through `load_text_channel`, chunks via the same duplicate/unknown-page validation as `app.main.load_configured_late_channels`, and late channels through `load_late_channel`. Build Qwen dense indexes one at a time, writing each artifact (`embeddings.npy` + `dense.json`) atomically under `data/index-dense-<name>/`, and hash every input/output. An OOM arm stays in the report as `skipped_oom` and cannot be selected. Only after a successful `winner` is selected, release its dense encoder, load the Qwen3 expander, read/refresh the provenance-valid cache, run original plus expanded BM25/dense/Mogan/Colmm channels, and write `dense:<winner>+expand`.

Include exact model revisions, benchmark hashes, primary/late/dense index hashes, cache hash, prompt fingerprint, question selection, `development` mode, and Git commit. Reject `--final` without `--yes-final-gate`. Do not import FastAPI.

- [ ] **Step 4: Verify GREEN, run both real stages, and commit code**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/bench/test_semantic_coverage.py -q && uv run ruff check scripts/eval_semantic_coverage.py src/belge_gozu/retrieval/dense.py src/belge_gozu/retrieval/expand.py src/belge_gozu/bench/semantic_coverage.py && uv run pyright scripts/eval_semantic_coverage.py src/belge_gozu/retrieval/dense.py src/belge_gozu/retrieval/expand.py src/belge_gozu/bench/semantic_coverage.py`  
Expected: focused tests, lint, and Pyright pass before any real model download.

```bash
git add scripts/eval_semantic_coverage.py tests/bench/test_semantic_coverage.py
git commit -m "feat(bench): run semantic coverage experiments"
```

### Task 5: Render a self-contained HTML decision report

**Files:**
- Create: `scripts/render_semantic_coverage_html.py`
- Create: `tests/test_render_semantic_coverage_html.py`

**Interfaces:**
- Produces `render_report(report: Mapping[str, object]) -> str`.
- Command: `uv run python scripts/render_semantic_coverage_html.py --input data/bench/results/semantic-coverage-dev-v1.json --out data/bench/results/semantic-coverage-dev-v1.html`.
- Output includes no external script, font, image, or network dependency.

- [ ] **Step 1: Write failing renderer tests**

```python
from render_semantic_coverage_html import render_report


def test_renderer_includes_development_banner_metrics_and_escaped_question():
    html = render_report(sample_report(question="<script>alert(1)</script>"))
    assert "DEVELOPMENT ONLY" in html
    assert "Paraphrase coverage" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_render_semantic_coverage_html.py -q`  
Expected: import failure for `render_semantic_coverage_html`.

- [ ] **Step 3: Implement the local HTML renderer**

```python
def render_report(report: Mapping[str, object]) -> str:
    require_schema(report, version=1, mode="development")
    rows = "".join(render_arm_row(name, arm) for name, arm in report["arms"].items())
    diagnostics = "".join(render_diagnostic(row) for row in report["diagnostics"])
    return PAGE_TEMPLATE.format(rows=rows, diagnostics=diagnostics, provenance=render_provenance(report))
```

Use `html.escape` for every text field, inline CSS for an accessible contrast-safe table, and compact inline SVG bars for overall/paraphrase coverage and p50/p95 latency. Include an always-visible `DEVELOPMENT ONLY — HOLDOUT REQUIRED` banner, baseline/dense/expansion rows, selected dense model evidence, `c206`/`c404` rows, provenance, and a collapsible question diagnostics table. Write HTML atomically with `NamedTemporaryFile` plus `os.replace`.

- [ ] **Step 4: Verify GREEN and commit**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_render_semantic_coverage_html.py -q && uv run ruff check scripts/render_semantic_coverage_html.py tests/test_render_semantic_coverage_html.py && uv run pyright scripts/render_semantic_coverage_html.py tests/test_render_semantic_coverage_html.py`  
Expected: renderer test passes; static checks are clean.

```bash
git add scripts/render_semantic_coverage_html.py tests/test_render_semantic_coverage_html.py
git commit -m "feat(bench): render semantic coverage HTML report"
```

### Task 6: Execute, visually inspect, and record the development diagnosis

**Files:**
- Create: `data/bench/results/semantic-coverage-dev-v1.json`
- Create: `data/bench/results/semantic-coverage-dev-v1.html`
- Create: `docs/research/findings/2026-09-03-semantic-coverage-experiments.md`

- [ ] **Step 1: Run the full sequential experiment once**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/eval_semantic_coverage.py \
  --bench data/bench/canary_v2.jsonl \
  --min-verification human \
  --cache data/bench/cache/semantic-expansions-v1.jsonl \
  --out data/bench/results/semantic-coverage-dev-v1.json
```

Expected: JSON contains baseline, one row for every attempted dense arm (successful
or `skipped_oom`), at most one `+expand` arm, every successful dense-artifact hash,
Qwen prompt/cache provenance, and `mode="development"`.

- [ ] **Step 2: Render and visually inspect the HTML**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/render_semantic_coverage_html.py \
  --input data/bench/results/semantic-coverage-dev-v1.json \
  --out data/bench/results/semantic-coverage-dev-v1.html
```

Open the local HTML with the available visual inspection tool. Verify the development-only banner, all four arms, paraphrase coverage, c206/c404 status, latency/disk bars, and escaped question text are legible.

- [ ] **Step 3: Record only allowed conclusions**

Write the exact measured numbers, selected dense arm under the documented development tie-break, expansion delta, resource cost, and whether c206/c404 entered. State explicitly that observed canary cannot authorize production and list the required new human-verified law-group-disjoint holdout.

- [ ] **Step 4: Full verification and commit artefacts**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest -q && uv run ruff check . && uv run pyright --pythonpath /Users/barandincoguz/Desktop/project-delta/.venv/bin/python && jq '.mode, .arms, .provenance' data/bench/results/semantic-coverage-dev-v1.json`  
Expected: full suite passes, Ruff and Pyright have no errors, and report carries all required provenance.

```bash
git add data/bench/results/semantic-coverage-dev-v1.json \
  data/bench/results/semantic-coverage-dev-v1.html \
  docs/research/findings/2026-09-03-semantic-coverage-experiments.md
git commit -m "docs(research): record semantic coverage experiments"
```

## Plan Self-Review

- Spec coverage: Tasks 1–4 provide both offline channels, fixed revisions, scoreless union, cache/provenance, errors, and development-only boundary; Task 5 creates the requested self-contained HTML; Task 6 executes and records without an unsupported production claim.
- Placeholder scan: every task lists files, exact public interfaces, a red test, a green verification, and a commit. No deferred behavior remains.
- Type consistency: dense and late channels share `candidate_pages(query, limit) -> list[str]`; the evaluator owns list union and metrics; the runner owns real artefact loading; the renderer consumes only serialized report dictionaries.
