# Task 5 Report: ExhaustiveBinaryRetriever + pipeline seçimi

## Status: DONE

## Summary

Implemented `ExhaustiveBinaryRetriever` in `src/belge_gozu/retrieval/core.py`
exactly per the brief's transcription (chunked binary MaxSim over the full
corpus, `np.maximum.reduceat` per page, normalized by `max(1, q_emb.shape[0])`).
Wired pipeline selection through `config.py` and `app/main.py` per controller
ruling R2 (replacing the brief's `functools.partial` sketch). Updated
`AskService.ask` to make `candidates` optional. `TwoStageRetriever` left
byte-identical (only new code appended after it in `core.py`).

## Files changed

- `src/belge_gozu/retrieval/core.py` — added `ExhaustiveBinaryRetriever` class
  (verbatim from brief). `TwoStageRetriever` untouched (diff shows pure
  addition, confirmed via `git diff`).
- `src/belge_gozu/config.py` — added `from typing import Literal` and
  `retrieval_pipeline: Literal["exhaustive", "two-stage"] = "exhaustive"`
  field with a short Turkish comment explaining the two modes.
- `src/belge_gozu/app/main.py`:
  - Import `ExhaustiveBinaryRetriever` alongside `TwoStageRetriever`.
  - Retriever construction: `if s.retrieval_pipeline == "exhaustive":
    ExhaustiveBinaryRetriever(...) else: TwoStageRetriever(...)`.
  - `/search`: `cand = s.stage1_candidates if s.retrieval_pipeline ==
    "two-stage" else None`; calls `retriever.search(body.query, k=k)` when
    `cand is None`, else `retriever.search(body.query, k=k, candidates=cand)`.
    `record_event(..., k=k, candidates=cand)` passes the same values used for
    the actual call (previously always `body.k or s.top_k` /
    `s.stage1_candidates`, recomputed here as `k`/`cand`).
  - `/ask`: same `cand` rule; `service.ask(body.question, k=s.top_k,
    candidates=cand)`; `record_event(..., candidates=cand)`.
  - One pyright suppression added: the `retriever.search(..., candidates=cand)`
    call in the two-stage branch of `/search` triggers
    `reportCallIssue` because pyright infers `retriever`'s static type as the
    union `ExhaustiveBinaryRetriever | TwoStageRetriever` and can't correlate
    that with `cand`'s nullability (a real per-branch invariant, not a bug).
    Added `# type: ignore[reportCallIssue]` with an explanatory comment on
    that line. This is the repo's existing convention for such pyright
    limitations (see `src/belge_gozu/index/encode.py` for prior examples).
- `src/belge_gozu/answer/base.py` — `AskService.ask` signature changed to
  `ask(self, question: str, k: int, candidates: int | None = None)`. Body:
  `if candidates is None: hits = self.retriever.search(question, k=k) else:
  hits = self.retriever.search(question, k=k, candidates=candidates)`.
  Everything else in the method unchanged.
- `tests/retrieval/test_exhaustive.py` (new) — the four tests from the brief,
  transcribed verbatim. The only change from the brief's literal text is
  import ordering (ruff/isort reordered the three import lines into
  stdlib/np → belge_gozu.* alphabetical → tests.* groups); no test logic
  changed.

## No test adaptation needed (verified, not assumed)

The brief flagged a risk that some app/telemetry test might assert
`stage1_ms`/`stage2_ms` non-null through `create_app`, which would break once
the default pipeline becomes "exhaustive". I checked all three candidate
locations before concluding no adaptation was required:

- `tests/telemetry/test_stages_integration.py` — builds `TwoStageRetriever`
  directly (`_retriever()` helper), never goes through `create_app`. Its
  stage-name assertions (`stage1_hamming`, `stage2_maxsim`) are exercised
  against the retriever object directly and are completely unaffected by the
  app's default pipeline.
- `tests/app/test_api.py` — no assertion anywhere on `stage1_ms`/`stage2_ms`
  or retriever type. `test_metrics_endpoint_exposes_series` only checks that
  `bg_stage_duration_seconds` (the metric family, not a specific stage label)
  appears in `/metrics` output — true for both pipelines.
- `tests/test_config.py` — asserts specific defaults (`retriever_model`,
  `stage1_candidates`, `top_k`, `request_delay_s`) but not
  `retrieval_pipeline`, so the new field's default is unconstrained by
  existing tests.

Decision: left all three files untouched.

## Testing (pristine output)

1. RED (brief step 2):
   `uv run pytest tests/retrieval/test_exhaustive.py -v` →
   `ImportError: cannot import name 'ExhaustiveBinaryRetriever'` (1 error,
   collection failure) — confirmed before implementation.

2. GREEN after implementation:
   `uv run pytest tests/retrieval/test_exhaustive.py -v` → 4 passed.

3. Full regression (brief step 4 + task instructions):
   - `uv run pytest tests/retrieval tests/app tests/answer -v` →
     **38 passed**, 6 warnings (pre-existing `StarletteDeprecationWarning` /
     SWIG `DeprecationWarning`s unrelated to this change).
   - `uv run pytest -q -m "not slow"` → **115 passed, 1 deselected**, same
     pre-existing warnings only.
   - `make lint` (`ruff check .` + `ruff format --check .` + `pyright`) →
     **all green** (`All checks passed!`, `67 files already formatted`,
     `0 errors, 0 warnings, 0 informations`).

   Lint required two mechanical fixes, applied via `ruff --fix` /
   `ruff format` (no logic changes):
   - `tests/retrieval/test_exhaustive.py`: import block reordered by
     isort/ruff (I001).
   - `src/belge_gozu/retrieval/core.py`: `ExhaustiveBinaryRetriever.search`'s
     final `PageHit(...)` construction reformatted from the brief's compact
     multi-arg-per-line style to ruff's one-arg-per-line style (matches the
     existing `TwoStageRetriever.search` formatting in the same file).

## Discipline check

- `TwoStageRetriever` byte-identical: confirmed via `git diff` — the diff for
  `core.py` is a pure append after the existing `TwoStageRetriever.search`
  method; no lines inside `TwoStageRetriever` were touched.
- Turkish comments preserved/added consistent with existing style (config.py
  comment, core.py docstring is the brief's original Turkish text, main.py's
  new inline comment explaining the pyright suppression is in English to
  match the surrounding code-quality-tooling context, not domain logic —
  can switch to Turkish if reviewer prefers).
- No `git add -A` / `git add .` used; staged exactly:
  `src/belge_gozu/retrieval/core.py`, `src/belge_gozu/config.py`,
  `src/belge_gozu/app/main.py`, `src/belge_gozu/answer/base.py`,
  `tests/retrieval/test_exhaustive.py`. Left `.agents/`, `skills-lock.json`,
  and `docs/research/observability-architecture.md` untouched (untracked,
  not part of this task; the first two were already untracked at session
  start, the third appeared independently and was not created by this task).
- Manifest/compat check block in `create_app` left intact — only the
  retriever-construction line right after it was changed, per instructions.

## Commit

`285beae feat(retrieval): exhaustive binary MaxSim as production path,
stage-1 demoted to ablation` on branch `feat/p0-retrieval-correctness`.

---

## Fix report (review R1)

Addressed all 6 review items (2 Important + 4 Minor; the 5th minor,
ledger-only, was not mine).

- **IMPORTANT 1** — `src/belge_gozu/app/main.py` `build_event`: added
  `"stages": dict(col.stages)` to the `detail` dict, after `"app_version"`,
  leaving all existing keys untouched. `telemetry/prom.py` was not touched.
  Added `tests/app/test_api.py::test_search_detail_records_exhaustive_stage_timing`:
  posts to `/search` on the default-pipeline client (`make_client`, which
  uses default `Settings()` → `retrieval_pipeline="exhaustive"`), reads the
  `/search` events row's `detail` JSON directly from sqlite, and asserts
  `"stages" in detail` and `"exhaustive_maxsim" in detail["stages"]`.

- **IMPORTANT 2** — replaced `test_exhaustive_beats_broken_stage1_counterexample`
  in `tests/retrieval/test_exhaustive.py` with
  `test_exhaustive_recovers_what_stage1_loses`, exactly as specified (mixed
  two-page query, brute-force pair search, `return` on first Stage-1/exhaustive
  divergence, `AssertionError` if none found). Ran it standalone first to
  confirm the seeded fixture (`build_fixture(n_pages=30)`, seed 11) actually
  diverges: **it exits on pair (a=0, b=2)** — exhaustive argmax picks page 0
  (normalized score 72.75), while Stage-1 (mean-sign Hamming, candidates=1)
  wrongly picks page 2 (raw score 560.0 → normalized 70.0). The assertion
  `ex_best_s >= ts_best_raw / q.shape[0]` holds (72.75 ≥ 70.0). Confirmed via
  `pytest -v` that the test passes.

- **MINOR 3** — `core.py` `ExhaustiveBinaryRetriever.score_all`: changed
  `np.bitwise_count(...).sum(axis=2)` to `.sum(axis=2, dtype=np.int32)` and
  dropped the now-redundant `.astype(np.int32)` on `sim` (numpy 2.5.2 / NEP 50
  keeps `128 - 2 * ham` at `int32` when `ham` is `int32`). Verified
  `test_scores_match_per_page_maxsim` still passes exactly (no float/int
  drift) — ran `tests/retrieval/test_exhaustive.py -v`, all 4 green.

- **MINOR 5** — added one-line docstrings: `TwoStageRetriever.search_embedding`
  → "RAW MaxSim toplamları döner (normalize edilmemiş); normalize search()'te
  yapılır."; `ExhaustiveBinaryRetriever.search_embedding` → "Per-query-token
  NORMALIZE edilmiş skorlar döner (score_all zaten böler)." No behavior change
  to `TwoStageRetriever`.

- **MINOR 6** — changed `# type: ignore[reportCallIssue]` to
  `# pyright: ignore[reportCallIssue]` in `app/main.py`'s `/search` two-stage
  branch. Verified `uv run pyright` still reports `0 errors, 0 warnings, 0
  informations` (the comment must sit on the exact line pyright reports the
  diagnostic on — confirmed by re-running pyright after the change).

- **MINOR 7** — added a one-line comment directly above the `reduceat` call
  in `score_all`: "offsets kesin artan (PackedIndex.build sıfır-token
  sayfayı reddeder) -> reduceat boş segment göremez."

### Testing (pristine output, post-fix)

- `uv run pytest tests/retrieval/test_exhaustive.py tests/app -v` →
  **23 passed**, only pre-existing unrelated deprecation warnings.
- `uv run pytest -q -m "not slow"` → **116 passed, 1 deselected**, same
  pre-existing warnings only.
- `make lint` (`ruff check .` + `ruff format --check .` + `pyright`) →
  **all green**: `All checks passed!`, `67 files already formatted`,
  `0 errors, 0 warnings, 0 informations`.

### Discipline

Staged and committed only the 4 touched files:
`src/belge_gozu/app/main.py`, `src/belge_gozu/retrieval/core.py`,
`tests/app/test_api.py`, `tests/retrieval/test_exhaustive.py`. Did not touch
`telemetry/prom.py`. Left `.agents/` and `skills-lock.json` untracked (not
mine); `docs/research/observability-architecture.md` was independently
committed elsewhere in this session (`cc21f19`, not by this task) and is no
longer untracked.

### Commit

`de1b68b fix(retrieval): stage telemetry in detail, discriminating stage-1
counterexample, int32 sim (review R1)` on branch `feat/p0-retrieval-correctness`.
