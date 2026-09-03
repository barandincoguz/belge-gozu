# Task 8 Report: Teşhis harness'ı + `bench run` (`bench/harness.py`)

## Status: DONE

## Files changed
- Created `src/belge_gozu/bench/harness.py`
- Created `tests/bench/test_harness.py`
- Modified `src/belge_gozu/cli.py` (`bench_app` sub-app + `bench run` command)

## What was implemented

### `src/belge_gozu/bench/harness.py`
- `StageRecord`, `QuestionDiagnostic`, `MetricBlock`, `EvalReport` pydantic models exactly per brief's schema.
  - `EvalReport.to_json(path)`: `path.parent.mkdir(parents=True, exist_ok=True)` then writes `model_dump_json(indent=1)`.
- `DiagnosticPipeline` Protocol (`name: str`, `run(question) -> (ranked, stages)`).
- `ExhaustiveDiagnosticAdapter`:
  - Raises `RuntimeError("encoder yapılandırılmamış")` if `retriever.encoder is None`.
  - Encodes query once, calls `score_all` once, full `argsort(-scores)` (stable), records top `record_top` ids+scores as a single `StageRecord(stage="exhaustive-binary")`, `time.perf_counter()`-measured latency covering encode+score+argsort. `final_ranked`/`ranked` = the full argsort order.
- `TwoStageDiagnosticAdapter`:
  - Same encoder guard.
  - Encodes query once.
  - Stage 1: full mean-sign Hamming ordering via `hamming_matrix(q_vec, index.page_vecs)[0]`, ascending argsort (stable); records top `record_top` ids with **negated** distances as scores (comment explains: small=good, sign flipped so "bigger score = better" stays consistent across stages) — `stage="stage1"`.
  - Stage 2: calls `retriever.search_embedding(q_emb, k=candidates, candidates=candidates)` (raw MaxSim sums), divides each by `n_q = max(1, q_emb.shape[0])`, records as `stage="stage2"`. `ranked` = full stage2 ordering (length = `candidates`).
  - Per-stage latency measured with `time.perf_counter()` checkpoints (stage1 checkpoint includes the shared query-encode time since it's the first stage; stage2 timing is stage2-only).
- `_git_commit()`: `subprocess.run(["git","rev-parse","--short","HEAD"], capture_output=True, text=True, check=True)`, any of `CalledProcessError/FileNotFoundError/OSError` → `"unknown"`.
- `run_retrieval_eval(...)`: implemented verbatim per the brief's Step-3 reference body — filters to `answerable=True`, computes `missing_gold_pages` (G0.1 coverage) against `known_page_ids`, computes `gold_ranks` per `StageRecord` from that record's own `top_ids`, `candidate_survival` from the last stage's `top_ids`, aggregates `overall`/`per_slice`/`per_doc` `MetricBlock`s (recall@k, mrr, ndcg@5, bootstrap CI on recall@5) via `belge_gozu.bench.metrics`.

### `src/belge_gozu/cli.py`
- Added `bench_app = typer.Typer()`, `app.add_typer(bench_app, name="bench")`.
- Added `bench run` command:
  - Options: `--bench` (default `data/bench/retrieval_eval_v1.jsonl`), `--pipeline` (`exhaustive`|`two-stage`, default `exhaustive`), `--out` (optional; defaults to `data/bench/results/<run_id>.json`).
  - `run_id = f"{datetime.now(UTC):%Y%m%d-%H%M}-{_git_commit()}-{pipeline}"` (imports `_git_commit` from the harness module).
  - Loads real index via `PackedIndex.load(s.index_dir)` + `pd.read_parquet(s.index_dir / "meta.parquet")` (same pattern as `app/main.py`'s `create_app`).
  - Lazily imports `ColSmolEncoder` inside the command body (never imported at module import time, so CI/lint/tests never trigger the `ml` extra).
  - Builds `ExhaustiveDiagnosticAdapter`/`TwoStageDiagnosticAdapter` (two-stage uses `s.stage1_candidates` for `candidates`) wrapping `ExhaustiveBinaryRetriever`/`TwoStageRetriever`.
  - Calls `run_retrieval_eval(adapter, questions, known_page_ids=set(idx.page_ids), run_id=run_id, index_manifest=idx.manifest, config={"pipeline": pipeline, "bench": str(bench)})`.
  - Writes the report via `report.to_json(out_path)`, echoes `recall@5`/`mrr` and the output path, and echoes `missing_gold_pages=<n>` when non-empty.
- Verified `belge-gozu bench run --help` and `belge-gozu --help` show correct wiring (`bench` sub-app with `--bench/--pipeline/--out` options).

### `tests/bench/test_harness.py`
Written exactly per brief's Step 1 (3 tests: `test_report_metrics_and_survival`, `test_missing_gold_page_reported`, `test_exhaustive_adapter_records_ranks`). Ran `ruff check --fix` + `ruff format` on it afterward purely for import ordering / removing the unused `EvalReport` import that the brief's literal snippet left in — no behavioral change, all three tests still assert exactly what the brief specified.

## Verification

1. RED confirmed first: `uv run pytest tests/bench/test_harness.py -v` → `ModuleNotFoundError: No module named 'belge_gozu.bench.harness'` (1 collection error) before implementation existed.
2. GREEN after implementation: `uv run pytest tests/bench/test_harness.py -v` → 3 passed.
3. Full regression:
   - `uv run pytest tests/bench -v` → 19 passed.
   - `uv run pytest -q -m "not slow"` → 123 passed, 1 deselected.
   - `make lint` (`ruff check .` + `ruff format --check .` + `pyright`) → all clean, `0 errors, 0 warnings, 0 informations`.
4. Manual functional smoke test (ad hoc script in scratchpad, not committed) using `tests/retrieval/test_core.build_fixture` + a self-embedding fake encoder, run through both `ExhaustiveDiagnosticAdapter` and `TwoStageDiagnosticAdapter` end-to-end via `run_retrieval_eval` + `to_json`:
   - Both pipelines correctly rank the planted needle (`d17:1`) first for query `"17"`.
   - `EvalReport.to_json` produces a JSON file with all expected top-level keys (`run_id, git_commit, index_manifest, config, missing_gold_pages, overall, per_slice, per_doc, diagnostics`).
   - `TwoStageDiagnosticAdapter` diagnostics show `stages == ["stage1", "stage2"]` as expected.
   - Scratch files removed after the check; nothing added to the repo beyond the three permitted paths.

## Commit
`cb676b5` — `feat(bench): stage-diagnostic eval harness with run provenance` (3 files changed, 354 insertions), staged by explicit path only (`src/belge_gozu/bench/harness.py`, `src/belge_gozu/cli.py`, `tests/bench/test_harness.py`) per R5. Pre-existing untracked `.agents/` and `skills-lock.json` were left untouched.

## Concerns / judgment calls
- `TwoStageDiagnosticAdapter`'s stage1/stage2 latency split is a judgment call: stage1's `latency_ms` includes the shared query-encode time (since it's the first stage and encode is its precursor). This wasn't pinned down by the brief/clarifications beyond "encode once". See Fix Report below for the corrected stage2 timing claim.

---

## Fix Report (review R1 — 2 Important + 5 minor findings)

Commit `440d8c6` — `fix(bench): validated pipeline option, two-stage adapter test, honest latency notes (review R1)`.

### IMPORTANT 1 — TwoStageDiagnosticAdapter unit test
Added `test_two_stage_adapter_matches_production_score` to `tests/bench/test_harness.py`: builds the same `build_fixture`/`SelfEnc` pattern used by the exhaustive-adapter test, runs `TwoStageDiagnosticAdapter(TwoStageRetriever(idx, meta, SelfEnc()), candidates=30, record_top=30).run("17")`, and asserts:
- `[s.stage for s in stages] == ["stage1", "stage2"]`
- `ranked[0] == "d17:1"` (the self-embedding needle)
- `stages[1].top_scores[0] == retriever.search("17", k=1, candidates=30)[0].score` — proves the harness's `/n_q` normalization for stage2 scores matches production `TwoStageRetriever.search()`'s `PageHit.score` exactly (same encoder instance/call, same raw MaxSim sum, same normalization divisor).

### IMPORTANT 2 — `--pipeline` validated as a real choice type
Added `class Pipeline(StrEnum): exhaustive = "exhaustive"; two_stage = "two-stage"` in `cli.py` (used `enum.StrEnum` rather than the literal `class Pipeline(str, Enum)` the reviewer suggested, because ruff's `UP042` rule — enabled in this repo's `[tool.ruff.lint] select = [... "UP" ...]` — flags `(str, Enum)` and requires `StrEnum` on Python ≥3.11; functionally identical: members are still real `str` instances, `.value` still works, click/typer still render `<exhaustive|two-stage>` choices and reject anything else). `bench_run`'s `pipeline` parameter is now typed `Pipeline = typer.Option(Pipeline.exhaustive, "--pipeline")`; `run_id` and `config["pipeline"]` now use `pipeline.value`. Verified: `belge-gozu bench run --pipeline bogus` exits 2 with `Invalid value for '--pipeline': 'bogus' is not one of 'exhaustive', 'two-stage'.`; `--help` shows `--pipeline <exhaustive|two-stage> [default: exhaustive]`.

### MINOR 3 — Latency-honesty comment correction
Corrected the code comment above the stage2 `search_embedding` call in `harness.py`: it now states that `search_embedding`'s production path internally re-runs its own stage-1 hamming/argpartition candidate selection, so the measured stage2 `latency_ms` is **not** stage2-only — it includes that internal (sub-ms) re-computation. (My original task report's claim that "stage2 timing is stage2-only" was inaccurate; corrected above in this file's Concerns section and in the code comment.)

### MINOR 4 — record_top/candidates coupling guard
- `TwoStageDiagnosticAdapter.__init__`: `self.record_top = max(record_top, candidates)` (with an explanatory comment: a smaller `record_top` would silently truncate scored candidates out of the recorded `top_ids`).
- `cli.py`'s `bench run`, two-stage branch: `TwoStageDiagnosticAdapter(..., candidates=s.stage1_candidates, record_top=max(200, s.stage1_candidates))`.

### MINOR 5 — TwoStageDiagnosticAdapter docstring
Added a class docstring note: stage-1 recording uses full-corpus `argsort` while the production path uses `argpartition` (candidate sets may diverge slightly at tie boundaries); `gold_ranks` is computed only against the `record_top`-bounded `top_ids` (`-1` means "not in the recorded top N", not "not in the full corpus"); true full-corpus rank diagnostics are oracle-run territory per controller ruling R12.

### MINOR 6 — `_git_commit` → `git_commit` (public)
Renamed the module-level function in `harness.py`; `EvalReport` construction now calls `git_commit=git_commit()` (no collision — it's a keyword argument label vs. a module-global function call, resolved independently). `cli.py`'s `bench_run` now imports and calls `git_commit` (not `_git_commit`) from `belge_gozu.bench.harness`. `cli.py`'s pre-existing *local* variable named `git_commit` inside `index_write_manifest` (a different, unrelated function scope) was left untouched — no naming conflict since it's a separate function's local variable, never imported.

### MINOR 7 — `bench run` output now prints ndcg5/n/ci_recall5
Changed the echoed summary line to:
`recall@5={o.recall_at.get(5, 0.0):.3f} mrr={o.mrr:.3f} ndcg5={o.ndcg5:.3f} n={o.n} ci_recall5={o.ci_recall5}`

### Verification after fixes
- `uv run pytest tests/bench -v` → 20 passed (was 19; +1 new two-stage test).
- `uv run pytest -q -m "not slow"` → 124 passed, 1 deselected.
- `make lint` (`ruff check .` + `ruff format --check .` + `pyright`) → all clean, `0 errors, 0 warnings, 0 informations`.
- Manually confirmed `belge-gozu bench run --pipeline bogus` is rejected by typer/click (exit code 2) and valid values (`exhaustive`, `two-stage`) are accepted per `--help`.
- `git status`/`git diff --stat` confirmed only the three permitted paths changed before staging; committed by explicit path (no `git add -A`/`.`).
