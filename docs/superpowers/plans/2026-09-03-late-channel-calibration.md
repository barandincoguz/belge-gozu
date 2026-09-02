# Late Channel Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a leakage-resistant calibration bench that decides whether the two-ColBERT late channel can safely bypass the legacy BM25 abstention failure mode.

**Architecture:** The production `LateInteractionChannel` will expose one runtime-aligned scored search result, including page-level raw and query-token-normalized confidence statistics. A separate late-calibration module will reuse the repository's deterministic NumPy logistic calibrator while owning its own four-feature schema, artifact fingerprint, grouped fit/calibration split, evaluation metrics, and enablement verdict. A standalone script will load the real indices and frozen benchmarks, fit only on development data, and require an explicit flag before touching the locked test split.

**Tech Stack:** Python 3.12, NumPy, pandas/Parquet, Pydantic-compatible JSON, pytest, Typer-independent argparse CLI, existing `belge_gozu.answer.calibrate` metrics and model.

## Global Constraints

- Keep BM25 on pages and do not modify `retrieval/text.py`, `RECIPE_VERSION`, or `recipe_fingerprint()`.
- Keep query expansion in encoding but exclude expansion vectors from the MaxSim sum through `encode_query_vectors()`.
- Use only human-verified answerable rows from `canary_v2` and verified rows from `unans_v1`.
- Select the feature schema and threshold without reading the locked test split.
- Use `risk <= 0.05` for threshold calibration and all five enablement checks from the design spec.
- Preserve the existing default-closed/fail-fast late-channel behavior unless the final gate says `eligible_to_enable=true`.
- Do not touch the unrelated untracked `graphify-out/` directory.

---

### Task 1: Bring the measured late-channel baseline into the working branch

**Files:**
- Merge from local branch: `main` commits `52f1625..5d5f80a`
- Verify: `src/belge_gozu/retrieval/late.py`
- Verify: `src/belge_gozu/index/colbert_encode.py`
- Verify: `scripts/eval_late_channel.py`

**Interfaces:**
- Consumes: the current branch at `a3b158d` plus the committed design document.
- Produces: `LateInteractionChannel`, `ColBERTEncoder.encode_query_vectors()`, `union_candidates()`, `canary_v2`, and the two-channel production parity script.

- [x] **Step 1: Merge the local measured implementation**

```bash
git merge --no-edit main
```

Expected: a merge commit containing the 15 late-channel research/implementation commits and no conflict with the new spec.

- [x] **Step 2: Verify the imported focused tests**

```bash
uv run pytest tests/retrieval/test_late.py tests/retrieval/test_union.py tests/index/test_colbert_encode.py -q
```

Expected: all focused tests pass.

- [x] **Step 3: Verify measured/production parity on the data-bearing checkout**

```bash
cd /Users/barandincoguz/Desktop/project-delta
uv run python scripts/eval_late_channel.py
```

Expected: `R@5=0.7766`, `R@20=0.9149`, `R@50=0.9362`, and `paraphrase R@50=0.8571`, followed by `ÜRETİM YOLU ÖLÇÜMLE BİREBİR`.

### Task 2: Add runtime-aligned late score summaries

**Files:**
- Modify: `src/belge_gozu/retrieval/late.py`
- Modify: `tests/retrieval/test_late.py`

**Interfaces:**
- Consumes: `QueryEncoder.encode_query_vectors(text) -> np.ndarray`, flattened chunk embeddings/offsets, and `chunk_pages`.
- Produces: `LateSearchResult` and `LateInteractionChannel.search_with_scores(query, limit=200) -> LateSearchResult`.

- [x] **Step 1: Write failing score-summary tests**

Add tests proving that one query encoding yields a page ranking and these exact values for two query vectors over synthetic chunks:

```python
result = channel.search_with_scores("soru", limit=2)
assert result.pages == ["p1", "p2"]
assert result.query_tokens == 2
assert result.raw_top1 == pytest.approx(2.0)
assert result.raw_margin == pytest.approx(1.0)
assert result.mean_top1 == pytest.approx(1.0)
assert result.mean_margin == pytest.approx(0.5)
assert encoder.calls == 1
```

Also add cases for duplicate pages across chunks and an encoder returning zero query vectors.

- [x] **Step 2: Run the tests and observe the expected failure**

```bash
uv run pytest tests/retrieval/test_late.py -q
```

Expected: failure because `search_with_scores` and `LateSearchResult` do not exist.

- [x] **Step 3: Implement the minimal score result**

Add an immutable result type with:

```python
@dataclass(frozen=True)
class LateSearchResult:
    pages: tuple[str, ...]
    query_tokens: int
    raw_top1: float
    raw_margin: float
    mean_top1: float
    mean_margin: float
```

Implement `search_with_scores()` by encoding once, computing the existing chunk MaxSim scores, walking ranked chunks until page-level top-1/top-2 distinct pages are known, and dividing both top-1 and margin by `query_tokens`. Raise `ValueError` for an empty query-vector matrix. Make `scores()` and `candidate_pages()` delegate to shared private helpers so the scoring formula has one implementation.

- [x] **Step 4: Run focused tests**

```bash
uv run pytest tests/retrieval/test_late.py tests/retrieval/test_union.py -q
```

Expected: all tests pass and the fake encoder call count is one.

- [x] **Step 5: Commit the runtime-aligned feature surface**

```bash
git add src/belge_gozu/retrieval/late.py tests/retrieval/test_late.py
git commit -m "feat(retrieval): expose calibrated late score summary"
```

### Task 3: Implement pure late calibration and artifact contracts

**Files:**
- Create: `src/belge_gozu/answer/late_calibrate.py`
- Create: `tests/answer/test_late_calibrate.py`

**Interfaces:**
- Consumes: `Calibrator`, `choose_threshold`, `evaluate`, `univariate_auc`, `sha256_file`, `git_blob_sha`, and `LateSearchResult`.
- Produces: `LATE_FEATURE_ORDER`, `LateCalibrationRow`, `LateCalibrationArtifact`, `group_key()`, `assign_inner_split()`, `fit_late_calibration()`, `evaluate_late_calibration()`, and `enablement_verdict()`.

- [x] **Step 1: Write failing tests for feature and split contracts**

Cover:

```python
assert LATE_FEATURE_ORDER == (
    "mogan_top1_mean", "mogan_margin_mean",
    "colmm_top1_mean", "colmm_margin_mean",
)
assert group_key(answerable_row) == "doc:k4721"
assert group_key(korpus_disi_row) == "anchor:5901"
assert group_key(eksik_kanit_row) == "doc:k4857"
assert group_key(anlamsiz_row) == "qid:u201"
```

Verify deterministic inner assignment, same-group co-location, class-presence validation, and finite four-feature vectors.

- [x] **Step 2: Run the new unit test and observe failure**

```bash
uv run pytest tests/answer/test_late_calibrate.py -q
```

Expected: collection failure because `belge_gozu.answer.late_calibrate` is absent.

- [x] **Step 3: Implement row and grouped-split primitives**

Use the fixed salt `late-calibration-v1`. Map the first digest byte into `fit` for values `< 170` and `calibration` otherwise. Validate that each inner split contains both labels before fitting or threshold selection.

- [x] **Step 4: Run split tests green**

```bash
uv run pytest tests/answer/test_late_calibrate.py -q
```

Expected: split and feature tests pass.

- [x] **Step 5: Write failing artifact and metric tests**

Create synthetic rows where the fit model separates labels and the calibration threshold is known. Assert:

```python
artifact = fit_late_calibration(fit_rows, calibration_rows, identity=identity)
loaded = LateCalibrationArtifact.load(path, expected_key=artifact.key)
assert loaded.tau == artifact.tau
assert evaluate_late_calibration(loaded, calibration_rows)["counts"]["total"] == 4
```

Also assert wrong fingerprint failure, atomic JSON round-trip, `safe_answerable_accept_rate`, unanswerable false accepts with Clopper-Pearson upper bound, and all five enablement checks.

- [x] **Step 6: Run artifact tests and observe failure**

```bash
uv run pytest tests/answer/test_late_calibrate.py -q
```

Expected: failures for missing artifact/metric behavior.

- [x] **Step 7: Implement calibration, evaluation, and verdict**

Fit the existing deterministic `Calibrator` on `fit_rows`, predict on the independent `calibration_rows`, and call `choose_threshold(..., max_risk=0.05)` only there. Store the fitted model, chosen threshold, feature order, late recipe ID, identity fields, fit/calibration metrics, counts, and provenance. Evaluation must never refit or change `tau`.

Compute enablement checks exactly as:

```python
checks = {
    "selective_risk_point_lte_0_05": selective_risk <= 0.05,
    "unanswerable_false_accept_point_lte_0_02": unans_rate <= 0.02,
    "unanswerable_false_accept_cp95_lte_0_05": unans_upper <= 0.05,
    "safe_answerable_accept_rate_gte_0_80": safe_accept >= 0.80,
    "identity_matches": identity_matches,
}
eligible = all(checks.values())
```

- [x] **Step 8: Run pure calibration tests green**

```bash
uv run pytest tests/answer/test_late_calibrate.py tests/answer/test_calibrate.py -q
```

Expected: all tests pass; the existing BM25 calibration suite is unchanged.

- [x] **Step 9: Commit the pure calibration layer**

```bash
git add src/belge_gozu/answer/late_calibrate.py tests/answer/test_late_calibrate.py
git commit -m "feat(answer): add leakage-resistant late calibration artifact"
```

### Task 4: Build the production-class calibration runner

**Files:**
- Create: `scripts/calibrate_late_channel.py`
- Create: `tests/test_calibrate_late_channel.py`

**Interfaces:**
- Consumes: primary `chunks.parquet`/`page_texts.parquet`, two late-index directories, `canary_v2.jsonl`, `unans_v1.jsonl`, `splits_v1.json`, `LateInteractionChannel.search_with_scores()`, and `union_candidates()`.
- Produces: `fit` and `eval` subcommands, `calibrator.json`, and self-contained JSON reports.

- [x] **Step 1: Write failing CLI tests with injected synthetic channels**

Test the parser and orchestration through a `run_fit(inputs, scorer)` / `run_eval(inputs, scorer)` boundary so no model or network is needed. Assert that fit reads only outer-dev rows, eval refuses without `--yes-final-gate`, eval reads only outer-test rows, and output JSON includes per-question features plus raw-vs-normalized diagnostics.

- [x] **Step 2: Run the CLI test and observe failure**

```bash
uv run pytest tests/test_calibrate_late_channel.py -q
```

Expected: failure because the script module does not exist.

- [x] **Step 3: Implement input loading and runtime-aligned scoring**

The scorer must:

1. load the frozen page BM25 channel and chunk-to-page mapping,
2. call `search_with_scores()` once per late model and query,
3. reproduce the measured sequential union order,
4. label a row from union top-5 membership,
5. keep raw top-1/query-token counts for diagnostics but feed only normalized top-1/margins to the calibrator,
6. derive all content hashes and late-index identities before fitting.

- [x] **Step 4: Implement guarded fit/eval commands and atomic reports**

Use these defaults:

```text
--canary data/bench/canary_v2.jsonl
--unans data/bench/unans_v1.jsonl
--splits data/bench/splits_v1.json
--index-dir data/index-traincompat-int8
--late-index data/index-colbert-mogan-f16
--late-index data/index-colbert-colmm-f16
--artifact-dir data/calibration/late-channel-v1
```

`eval` must require `--yes-final-gate`, load the existing artifact by expected identity, and never call a fit function.

- [x] **Step 5: Run CLI and related tests green**

```bash
uv run pytest tests/test_calibrate_late_channel.py tests/answer/test_late_calibrate.py tests/retrieval/test_late.py -q
```

Expected: all tests pass without model downloads.

- [x] **Step 6: Commit the runner**

```bash
git add scripts/calibrate_late_channel.py tests/test_calibrate_late_channel.py
git commit -m "feat(bench): add locked late-channel calibration runner"
```

### Task 5: Run development calibration and the one-shot final gate

**Files:**
- Create: `data/bench/results/late-channel-calibration-dev-v1.json`
- Create: `data/bench/results/late-channel-calibration-test-v1.json`
- Generated, ignored: `data/calibration/late-channel-v1/calibrator.json`

**Interfaces:**
- Consumes: real data and late indices from `/Users/barandincoguz/Desktop/project-delta/data` through explicit absolute CLI paths.
- Produces: locked threshold artifact, dev report, test report, and `eligible_to_enable` verdict.

- [x] **Step 1: Fit without reading test rows**

```bash
uv run python scripts/calibrate_late_channel.py fit \
  --index-dir /Users/barandincoguz/Desktop/project-delta/data/index-traincompat-int8 \
  --late-index /Users/barandincoguz/Desktop/project-delta/data/index-colbert-mogan-f16 \
  --late-index /Users/barandincoguz/Desktop/project-delta/data/index-colbert-colmm-f16 \
  --artifact-dir data/calibration/late-channel-v1 \
  --out data/bench/results/late-channel-calibration-dev-v1.json
```

Expected: report states `outer_split=dev`, distinct non-empty `fit` and `calibration` groups, the raw Mogan length correlation remains high, and the selected threshold comes only from inner calibration rows.

- [x] **Step 2: Verify deterministic replay of fit**

Run the same command to a temporary output path and compare calibrator model fields, threshold, row assignments, and per-question probabilities while excluding timestamps/output paths.

Expected: values and assignments are identical.

- [x] **Step 3: Run the locked test exactly once**

```bash
uv run python scripts/calibrate_late_channel.py eval \
  --yes-final-gate \
  --index-dir /Users/barandincoguz/Desktop/project-delta/data/index-traincompat-int8 \
  --late-index /Users/barandincoguz/Desktop/project-delta/data/index-colbert-mogan-f16 \
  --late-index /Users/barandincoguz/Desktop/project-delta/data/index-colbert-colmm-f16 \
  --artifact-dir data/calibration/late-channel-v1 \
  --out data/bench/results/late-channel-calibration-test-v1.json
```

Expected: report states `outer_split=test`, records every enablement check, and prints either `ELIGIBLE` or `BLOCKED` without changing the artifact.

- [x] **Step 4: Commit only reproducible reports**

```bash
git add data/bench/results/late-channel-calibration-dev-v1.json \
        data/bench/results/late-channel-calibration-test-v1.json
git commit -m "exp(calibration): measure late-channel abstention gate"
```

### Task 6: Document the decision and run final verification

**Files:**
- Create: `docs/research/findings/2026-09-03-late-channel-calibration.md`
- Modify only if eligible: no production runtime file; opening the actual serving flag remains a separately reviewed integration because this experiment's decision must exist first.

**Interfaces:**
- Consumes: both committed reports and the immutable artifact identity.
- Produces: a concise decision record that says which checks passed, which failed, and why the default-closed guard remains or may proceed to a separate integration.

- [x] **Step 1: Write the finding from report values**

Include dataset counts and verification caveats, fit/calibration/test metrics, raw-vs-normalized evidence, the exact `tau`, all enablement checks, and the decision. Do not describe model-cross-checked `unans_v1` rows as human verified.

- [x] **Step 2: Verify documentation and repository diff**

```bash
git diff --check
rg -n "TBD|TODO|implement later|fill in details" \
  docs/research/findings/2026-09-03-late-channel-calibration.md \
  docs/superpowers/specs/2026-09-03-late-channel-calibration-design.md \
  docs/superpowers/plans/2026-09-03-late-channel-calibration.md
```

Expected: no whitespace errors and no placeholders.

- [x] **Step 3: Run the full verification suite**

```bash
uv run pytest -q
uv run ruff check .
uv run pyright
```

Expected: all tests pass, Ruff reports no errors, and Pyright reports zero errors.

- [x] **Step 4: Commit the finding**

```bash
git add docs/research/findings/2026-09-03-late-channel-calibration.md \
        docs/superpowers/plans/2026-09-03-late-channel-calibration.md
git commit -m "docs(research): record late-channel calibration verdict"
```

- [x] **Step 5: Re-run final evidence commands after the last commit**

```bash
uv run pytest -q
uv run ruff check .
uv run pyright
git status --short
```

Expected: verification commands succeed and only the pre-existing `graphify-out/` remains untracked.

