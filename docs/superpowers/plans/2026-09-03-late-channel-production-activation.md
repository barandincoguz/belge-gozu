# Late Channel Production Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the measured two-channel ColBERT candidate union the default hybrid production retrieval path without changing the BM25 abstention scale.

**Architecture:** `HybridRetriever` keeps page-BM25 ranking as primary and interleaves page candidates from two chunk ColBERT channels. Late-channel loading validates index sidecars against the corpus chunk map at startup. The first result and `PageHit.score` remain BM25-derived.

**Tech Stack:** Python 3.12, NumPy memmaps, pandas/Parquet, Pydantic settings, pytest, Ruff, Pyright.

## Global Constraints

- The BM25 recipe, fingerprint, `min_score_threshold=10.6`, and primary top-1 remain unchanged.
- Use Mogan `ad90b4f64135e4db75a6453feee85fd7b44b33a1` and Colmm `3b5dd416a29c8f3abff1c9274f0b08ba69de5232` only.
- Candidate union only; never combine BM25 and ColBERT numeric scores.
- A channel whose score could determine top-1 requires calibration; this activation pins BM25 first.
- Missing or mismatched late artefacts fail at application startup.

---

### Task 1: Configure and validate late indices

**Files:**
- Modify: `src/belge_gozu/config.py`
- Modify: `src/belge_gozu/retrieval/late.py`
- Test: `tests/test_config.py`
- Test: `tests/retrieval/test_late.py`

**Interfaces:**
- Produces `Settings.late_channel_enabled`, `late_mogan_index_dir`, `late_colmm_index_dir`, and `late_candidate_limit`.
- Produces `load_late_channel(index_dir: Path, chunk_pages: Mapping[str, tuple[str, ...]], *, device: str | None) -> LateInteractionChannel`.

- [x] **Step 1: Write the failing tests**

```python
def test_late_channel_is_enabled_for_default_hybrid_production():
    s = Settings()
    assert s.late_channel_enabled is True
    assert s.late_candidate_limit == 200

def test_late_loader_rejects_chunk_ids_missing_from_corpus_map(tmp_path):
    with pytest.raises(ValueError, match="chunk eşlemesi"):
        load_late_channel(tmp_path, {"known": ("p:1",)}, device="cpu")
```

- [x] **Step 2: Verify RED**

Run: `uv run pytest tests/test_config.py tests/retrieval/test_late.py -q`

Expected: FAIL because the settings and loader do not exist.

- [x] **Step 3: Implement the minimal loader**

```python
def load_late_channel(index_dir, chunk_pages, *, device=None):
    side = json.loads((index_dir / "colbert.json").read_text())
    ids = json.loads((index_dir / "chunk_ids.json").read_text())
    if set(ids) != set(chunk_pages):
        raise ValueError("geç indeks chunk eşlemesiyle uyuşmuyor")
    return LateInteractionChannel(
        embeddings=np.load(index_dir / "embs.npy", mmap_mode="r"),
        offsets=np.load(index_dir / "offsets.npy", mmap_mode="r"),
        chunk_ids=ids, chunk_pages=chunk_pages,
        encoder=ColBERTEncoder(side["model_repo"], side["revision"],
                               device=device, document_length=side["document_length"]),
    )
```

- [x] **Step 4: Verify GREEN**

Run: `uv run pytest tests/test_config.py tests/retrieval/test_late.py -q`

Expected: PASS.

### Task 2: Interleave late candidates without changing the score gate

**Files:**
- Modify: `src/belge_gozu/retrieval/hybrid.py`
- Test: `tests/retrieval/test_hybrid.py`

**Interfaces:**
- Consumes `LateInteractionChannel.search_with_scores(query, limit)`.
- Produces `HybridRetriever(..., late_channels=(), late_candidate_limit=200)` whose `search()` keeps BM25 first.

- [x] **Step 1: Write the failing integration test**

```python
def test_late_candidates_are_interleaved_but_bm25_top1_and_score_are_preserved():
    retriever = _fixture_with_late_pages(["k2:2", "k3:1"])
    hits = retriever.search("yerleşim yeri", k=3)
    assert hits[0].page_id == _bm25_order(retriever.text, "yerleşim yeri")[0]
    assert hits[0].score == pytest.approx(retriever.text.scores("yerleşim yeri").max())
    assert "k2:2" in [hit.page_id for hit in hits]
```

- [x] **Step 2: Verify RED**

Run: `uv run pytest tests/retrieval/test_hybrid.py -q`

Expected: FAIL because `HybridRetriever` has no late-channel constructor argument.

- [x] **Step 3: Add candidate-only interleaving**

```python
for channel in self.late_channels:
    result = channel.search_with_scores(query, limit=self.late_candidate_limit)
    ranking = union_candidates(ranking, list(result.pages))
    late_meta.append({"query_tokens": result.query_tokens, "mean_top1": result.mean_top1})
```

Keep `by_id[pid]` as the only source of `PageHit.score` and assert the first ranked page remains the original BM25 first page.

- [x] **Step 4: Verify GREEN**

Run: `uv run pytest tests/retrieval/test_hybrid.py -q`

Expected: PASS.

### Task 3: Wire production startup, disclose activation, and clean generated output

**Files:**
- Modify: `src/belge_gozu/app/main.py`
- Test: `tests/app/test_api.py`
- Modify: `docs/research/findings/2026-09-03-late-channel-calibration.md`

**Interfaces:**
- `build_retriever()` loads both channels only for `hybrid` plus `late_channel_enabled`.
- `/healthz` returns `retrieval.late_channel="enabled"` when active.

- [x] **Step 1: Write the failing health test**

```python
def test_healthz_reports_late_candidate_channel(tiny_corpus):
    client = make_client(tiny_corpus, late_channel_enabled=False)
    assert client.get("/healthz").json()["retrieval"]["late_channel"] == "disabled"
```

- [x] **Step 2: Verify RED**

Run: `uv run pytest tests/app/test_api.py -q`

Expected: FAIL because the health payload has no late-channel state.

- [x] **Step 3: Wire only the approved hybrid path**

```python
if s.late_channel_enabled and s.retrieval_pipeline != "hybrid":
    raise IndexCompatibilityError("geç kanal yalnız hybrid pipeline ile kullanılabilir")
if s.late_channel_enabled:
    channels = load_configured_late_channels(s, chunks)
    retriever = HybridRetriever(index, meta, encoder, bm25, doc_names,
                                late_channels=channels,
                                late_candidate_limit=s.late_candidate_limit)
```

Record that calibrated confidence remains blocked, but BM25-pinned candidate union is now the production route.

- [x] **Step 4: Verify API and parity**

Run: `uv run pytest tests/app/test_api.py -q && uv run python scripts/eval_late_channel.py`

Expected: API tests PASS and parity prints `ÜRETİM YOLU ÖLÇÜMLE BİREBİR`.

- [x] **Step 5: Remove only generated artefacts**

Run: `rm -rf /Users/barandincoguz/Desktop/project-delta/graphify-out /Users/barandincoguz/Desktop/project-delta/.pytest_cache /Users/barandincoguz/Desktop/project-delta/.ruff_cache`

Then remove only `__pycache__` directories under `src/`, `tests/`, `scripts/`, and `research/`; leave indexes, bench JSONs, `.agents/`, `skills-lock.json`, and `st.bin` intact.

### Task 4: Verify the merged production state

- [x] **Step 1: Run full verification**

Run: `uv run pytest -q && uv run ruff check . && uv run pyright --pythonpath /Users/barandincoguz/Desktop/project-delta/.venv/bin/python && uv run python scripts/eval_late_channel.py`

Expected: all tests pass, Ruff is clean, Pyright reports zero errors, and parity reports the four measured retrieval values.

- [x] **Step 2: Inspect final state**

Run: `git status --short --branch && git log --oneline -4`

Expected: only explicitly preserved user-owned untracked files may remain.
