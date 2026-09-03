# Candidate-Pool Reranker Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare a BM25-top-1-pinned reranker with an unpinned reranker offline, without altering the production retrieval or answer path.

**Architecture:** A pure candidate-pool helper unions the existing routed BM25 and two ColBERT page lists without numeric score fusion. A local pinned-revision Transformers cross-encoder scores the resulting `query, page_text` pairs. The experiment emits both a pinned ranking and an unpinned ranking, preserving each page's original BM25 score for abstention diagnostics.

**Tech Stack:** Python 3.12, NumPy, pandas/Parquet, PyTorch, Transformers, Hugging Face Hub, pytest, Ruff, Pyright.

## Global Constraints

- Do not change `retrieval/text.py`, `recipe_fingerprint()`, BM25 page retrieval, `min_score_threshold=10.6`, `HybridRetriever.search()`, `/search`, or `/ask`.
- Candidate pool depth is exactly 50 per source: routed BM25 pages plus page lists from Mogan and Colmm ColBERT.
- No BM25, Mogan, Colmm, or reranker scores are fused numerically.
- Reranker checkpoint is `BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`; use the existing `ml` extra only.
- The observed `retrieval_eval_v2` is development diagnosis only. A final holdout is a separate, human-data gate and cannot be fabricated or selected from observed results.
- Unpinned output is offline-only until a new end-to-end abstention calibration and locked test pass.

---

### Task 1: Add a pure, scoreless candidate-pool boundary

**Files:**
- Create: `src/belge_gozu/retrieval/candidates.py`
- Create: `tests/retrieval/test_candidates.py`

**Interfaces:**
- Produces: `build_candidate_pool(bm25_pages: Sequence[str], late_page_lists: Sequence[Sequence[str]], *, limit: int = 50) -> list[str]`.
- Contract: take at most `limit` items from each source, retain the first occurrence, and preserve BM25's first page as pool item zero.

- [ ] **Step 1: Write failing tests**

```python
from belge_gozu.retrieval.candidates import build_candidate_pool


def test_pool_uses_each_source_at_its_own_depth_and_deduplicates():
    assert build_candidate_pool(
        ["b1", "b2", "b3"],
        [["a1", "b2", "a2"], ["c1", "a1", "c2"]],
        limit=2,
    ) == ["b1", "b2", "a1", "c1"]


def test_pool_keeps_bm25_first_even_when_late_repeats_it():
    assert build_candidate_pool(["b1", "b2"], [["b1", "a1"]], limit=50)[0] == "b1"


def test_pool_rejects_non_positive_depth():
    with pytest.raises(ValueError, match="limit"):
        build_candidate_pool(["b1"], [["a1"]], limit=0)
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/retrieval/test_candidates.py -q`

Expected: import failure because `retrieval.candidates` does not yet exist.

- [ ] **Step 3: Implement the narrow helper**

```python
def build_candidate_pool(bm25_pages, late_page_lists, *, limit=50):
    if limit < 1:
        raise ValueError(f"candidate limit pozitif olmalı: {limit}")
    seen, out = set(), []
    for source in (bm25_pages, *late_page_lists):
        for page_id in source[:limit]:
            if page_id not in seen:
                seen.add(page_id)
                out.append(page_id)
    return out
```

- [ ] **Step 4: Verify GREEN and commit**

Run: `uv run pytest tests/retrieval/test_candidates.py -q && uv run ruff check src/belge_gozu/retrieval/candidates.py tests/retrieval/test_candidates.py`

Expected: all tests pass and Ruff is clean.

```bash
git add src/belge_gozu/retrieval/candidates.py tests/retrieval/test_candidates.py
git commit -m "feat(retrieval): add scoreless candidate pool"
```

### Task 2: Add deterministic pinned and unpinned ranking composition

**Files:**
- Create: `src/belge_gozu/retrieval/rerank.py`
- Create: `tests/retrieval/test_rerank.py`

**Interfaces:**
- Produces: `PageReranker` protocol with `score(query: str, documents: Sequence[str]) -> np.ndarray`.
- Produces: `RerankComparison(pinned_pages: tuple[str, ...], unpinned_pages: tuple[str, ...], bm25_top1_rank_unpinned: int, selected_top1_bm25_score: float, would_abstain: bool)`.
- Produces: `compare_rerankings(query, pool, page_texts, bm25_scores, reranker, threshold) -> RerankComparison`.

- [ ] **Step 1: Write failing tests with a fixed reranker**

```python
class FixedReranker:
    def score(self, query, documents):
        return np.array([0.1, 0.9, 0.5])


def test_comparison_pins_bm25_first_but_unpinned_exposes_its_rank():
    result = compare_rerankings(
        "soru", ["b1", "a1", "a2"],
        {"b1": "bm25", "a1": "en iyi", "a2": "orta"},
        {"b1": 11.0, "a1": 7.0, "a2": 12.0}, FixedReranker(), threshold=10.6,
    )
    assert result.pinned_pages == ("b1", "a1", "a2")
    assert result.unpinned_pages == ("a1", "a2", "b1")
    assert result.bm25_top1_rank_unpinned == 3
    assert result.selected_top1_bm25_score == 7.0
    assert result.would_abstain is True
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/retrieval/test_rerank.py -q`

Expected: import failure for `compare_rerankings`.

- [ ] **Step 3: Implement ordering and boundary validation**

```python
scores = np.asarray(reranker.score(query, documents), dtype=np.float64)
if scores.shape != (len(pool),) or not np.isfinite(scores).all():
    raise ValueError("reranker skorları havuzla hizalı ve sonlu olmalı")
order = np.argsort(-scores, kind="stable")
unpinned = tuple(pool[int(i)] for i in order)
pinned = (pool[0], *[page_id for page_id in unpinned if page_id != pool[0]])
```

Look up every `page_id` in `page_texts` and `bm25_scores` before calling the reranker; raise a `ValueError` containing the missing id if either mapping is incomplete.

- [ ] **Step 4: Verify GREEN and commit**

Run: `uv run pytest tests/retrieval/test_rerank.py -q && uv run ruff check src/belge_gozu/retrieval/rerank.py tests/retrieval/test_rerank.py`

Expected: all tests pass and Ruff is clean.

```bash
git add src/belge_gozu/retrieval/rerank.py tests/retrieval/test_rerank.py
git commit -m "feat(retrieval): compare pinned reranker rankings"
```

### Task 3: Implement the pinned local Transformers reranker

**Files:**
- Modify: `src/belge_gozu/retrieval/rerank.py`
- Modify: `tests/retrieval/test_rerank.py`

**Interfaces:**
- Produces: `BGE_RERANKER_REPO`, `BGE_RERANKER_REVISION`, and `TransformerPageReranker(device: str | None = None, max_length: int = 512, batch_size: int = 8)`.
- Contract: loads `AutoTokenizer` and `AutoModelForSequenceClassification` with the pinned revision, runs `eval()` and inference mode, returns one raw float logit per input pair in input order.

- [ ] **Step 1: Write failing isolated tests**

```python
def test_transformer_reranker_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        TransformerPageReranker(batch_size=0)


def test_reranker_output_validator_rejects_nan_and_wrong_length():
    with pytest.raises(ValueError, match="sonlu"):
        validate_reranker_scores(np.array([np.nan]), expected=1)
    with pytest.raises(ValueError, match="hizalı"):
        validate_reranker_scores(np.array([0.1]), expected=2)
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/retrieval/test_rerank.py -q`

Expected: missing class/helper failures.

- [ ] **Step 3: Implement only the Transformers adapter**

Use the already-installed `torch`, `transformers`, and `huggingface_hub` dependencies; do not add FlagEmbedding or sentence-transformers. Resolve `device=None` as MPS when available, otherwise CPU, following `ColBERTEncoder`. Tokenize `(query, document)` pairs with `padding=True`, `truncation=True`, `max_length=self.max_length`, move tensors to the selected device, and collect `model(**inputs).logits.view(-1).float().cpu().numpy()` batch by batch.

- [ ] **Step 4: Verify GREEN and commit**

Run: `uv run pytest tests/retrieval/test_rerank.py -q && uv run ruff check src/belge_gozu/retrieval/rerank.py tests/retrieval/test_rerank.py`

Expected: all tests pass and no model/network test is required.

```bash
git add src/belge_gozu/retrieval/rerank.py tests/retrieval/test_rerank.py
git commit -m "feat(retrieval): add pinned transformers reranker"
```

### Task 4: Build an offline, provenance-bearing comparison runner

**Files:**
- Create: `scripts/eval_candidate_reranker.py`
- Create: `tests/test_eval_candidate_reranker.py`

**Interfaces:**
- Consumes: `Settings`, `load_text_channel`, `load_late_channel`, `build_candidate_pool`, `compare_rerankings`, and `BenchQuestion`.
- Produces: atomic JSON report with `pinned`, `unpinned`, `candidate_pool`, per-question diagnostics, model/index/bench SHA-256 provenance, and p50/p95 rerank latency.
- Command: `uv run python scripts/eval_candidate_reranker.py --bench data/bench/retrieval_eval_v2.jsonl --min-verification human --out data/bench/results/candidate-reranker-dev-v1.json`.

- [ ] **Step 1: Write failing runner tests using fake late channels and a fixed reranker**

```python
def test_run_comparison_reports_both_arms_and_bm25_top1_rank(tmp_path):
    report = run_comparison(fixture_inputs, FixedReranker())
    assert report["pinned"]["overall"]["recall_at"][5] >= 0
    assert report["unpinned"]["diagnostics"][0]["bm25_top1_rank_unpinned"] == 3
    assert report["candidate_pool"]["overall"]["recall_at"][50] >= report["pinned"]["overall"]["recall_at"][50]
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_eval_candidate_reranker.py -q`

Expected: import failure for the runner's `run_comparison` API.

- [ ] **Step 3: Implement the runner with existing provenance conventions**

Load `page_texts.parquet` through `load_page_texts`; construct the checked `chunk_id -> page_ids` map from `chunks.parquet` and call the existing `load_late_channel` twice rather than parsing sidecars in the script. For each answerable selected question: compute routed BM25 ranking and scores once, request 50 candidates from each late channel, construct the pool, rerank it once, then derive P and U. Use the existing `recall_at_k`, `mrr`, `ndcg_at_k`, and `bootstrap_ci` functions; record candidate-pool coverage separately from final rankings. Hash bench and index artefacts with existing `sha256_file`, use atomic output, and reject `--final` unless `--yes-final-gate` is present.

- [ ] **Step 4: Verify GREEN and commit**

Run: `uv run pytest tests/test_eval_candidate_reranker.py -q && uv run ruff check scripts/eval_candidate_reranker.py tests/test_eval_candidate_reranker.py`

Expected: all tests pass and runner imports without FastAPI.

```bash
git add scripts/eval_candidate_reranker.py tests/test_eval_candidate_reranker.py
git commit -m "feat(bench): add reranker comparison runner"
```

### Task 5: Run the observed development diagnosis and enforce the holdout boundary

**Files:**
- Modify: `docs/research/findings/2026-09-03-candidate-pool-reranker-experiment.md`
- Create: `data/bench/results/candidate-reranker-dev-v1.json`

**Interfaces:**
- Consumes: the Task 4 runner and observed `retrieval_eval_v2`.
- Produces: a diagnosis that is explicitly non-final, plus the exact missing input for final selection: a new human-verified, law-group-disjoint holdout.

- [ ] **Step 1: Run the model-backed development comparison once**

Run: `uv run python scripts/eval_candidate_reranker.py --bench data/bench/retrieval_eval_v2.jsonl --min-verification human --out data/bench/results/candidate-reranker-dev-v1.json`

Expected: report contains P/U metrics, BM25-top1 rank distribution, selected-top1 `would_abstain` counts, and latency percentiles.

- [ ] **Step 2: Record only the permitted conclusion**

Write the measured P/U metrics and explicitly state that this observed set cannot select a production winner. State whether a newly created, human-verified holdout is available. If it is absent, do not create questions, labels, or a claimed final result automatically.

- [ ] **Step 3: Verify the code and report provenance**

Run: `uv run pytest -q && uv run ruff check . && uv run pyright --pythonpath /Users/barandincoguz/Desktop/project-delta/.venv/bin/python && jq '.model_revision, .candidate_pool, .pinned, .unpinned' data/bench/results/candidate-reranker-dev-v1.json`

Expected: test suite passes; static checks are clean; report has the pinned reranker revision and both arms.

- [ ] **Step 4: Commit code, report, and finding**

```bash
git add docs/research/findings/2026-09-03-candidate-pool-reranker-experiment.md \
  data/bench/results/candidate-reranker-dev-v1.json
git commit -m "docs(research): record reranker development comparison"
```

## Plan Self-Review

- Spec coverage: Tasks 1--4 implement isolated pool, reranker, P/U comparison, provenance, error handling, and offline-only scope; Task 5 enforces the observed-data and human-holdout boundary.
- Placeholder scan: no deferred implementation markers; the only non-automatic action is human verification, which is intentionally an external authority boundary.
- Type consistency: pool outputs page ids; reranker consumes matching page text in the same order; comparison consumes aligned BM25 scores; runner serializes only Python/JSON numeric values.
