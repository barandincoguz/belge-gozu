# Task 2 Report: Maskeli ve formatlı encoder (`index/encode.py`)

## Status: DONE_WITH_CONCERNS

(Concern is not a blocker — the slow test's determinism FAIL and the
resulting `batch_size=1` decision were an expected, brief-anticipated
outcome, but it's flagged since it affects index build performance and
feeds the P0 baseline report.)

## What was implemented

`src/belge_gozu/index/encode.py`:
- New pure function `trim_by_mask(emb, mask) -> list[np.ndarray]`: drops
  padded rows from a `(B, L, D)` embedding batch using the `(B, L)`
  attention mask, before any downstream sign-binarization. Docstring
  reproduces the brief's rationale verbatim (v0 bug: all-zero padding rows
  flipping to a valid sign pattern after binarization).
- `ColSmolEncoder.__init__` now accepts `query_format: QueryFormat | None`
  (defaults to `CPE_0_3_18`), and additionally computes:
  - `self.model_revision` from `model.config._commit_hash` (falls back to
    `"unknown"`)
  - `self.doc_prompt = self.processor.visual_prompt_prefix`
  - `self.doc_prompt_sha256` (sha256 hex digest of `doc_prompt`)
  All existing model/processor setup, comments, and device/dtype logic
  left untouched.
- `_run` now converts the whole batch tensor at once
  (`emb = out.cpu().float().numpy()`, `mask = batch["attention_mask"].cpu().numpy()`)
  and returns `trim_by_mask(emb, mask)`, replacing the old
  `[e.cpu().float().numpy() for e in out]` per-item loop (which kept
  padding rows).
- `encode_query` now renders the prompt via
  `self.query_format.render(text)` and calls
  `self.processor.process_texts([rendered])` instead of
  `process_queries([text])` — prefix/suffix is now fully owned by
  `QueryFormat`, decoupling from colpali-engine's internal (and
  version-dependent) query formatting.
- `FakeEncoder` unchanged, `Encoder` protocol unchanged.

`tests/index/test_encode_mask.py` (new): stub-based unit tests for
`trim_by_mask` and `encode_query`'s use of `QueryFormat.render`, plus the
slow batch-vs-single sign-determinism test.

`src/belge_gozu/cli.py`: `index_build`'s `batch_size = 8` changed to
`batch_size = 1` per the brief's Step 6 decision rule (slow test FAILED —
see below).

## Deviation from brief (noted, justified)

The brief's Step 1 code imports `CPE_0_3_18` in the test file but never
uses it (only `TRAIN_COMPAT_V1` is used in the two unit tests). This trips
ruff's F401 (`select = ["E", "F", "I", "UP", "B"]` in `pyproject.toml`),
which `make lint` runs. Dropped the unused `CPE_0_3_18` import from the
test file; all test logic/assertions are otherwise verbatim from the
brief. Also reflowed one over-spaced inline comment
(`batch_out = enc.encode_pages(imgs)  # tek batch (karışık boyut)`) to
satisfy `ruff format --check` — no semantic change.

## TDD evidence

**Step 2 — RED** (`uv run pytest tests/index/test_encode_mask.py -v`):
```
FAILED tests/index/test_encode_mask.py::test_trim_by_mask_drops_padding_rows
FAILED tests/index/test_encode_mask.py::test_query_format_render_used - AttributeError: type object 'processor' has no attribute 'process_queries'. Did you mean: 'process_texts'?
2 failed in 0.06s
```
(First failure: `trim_by_mask` doesn't exist yet. Second: `encode_query`
still called `process_queries`, not `process_texts`.)

**Step 4 — GREEN** (`uv run pytest tests/index/test_encode_mask.py tests/index/test_encode.py -v`):
```
tests/index/test_encode_mask.py::test_trim_by_mask_drops_padding_rows PASSED
tests/index/test_encode_mask.py::test_query_format_render_used PASSED
tests/index/test_encode.py::test_fake_encoder_shapes_and_determinism PASSED
3 passed in 0.01s
```

## Slow determinism test (Step 5/6)

Command: `uv run pytest tests/index/test_encode_mask.py -m slow -v`

**Result: FAILED** on the first assertion (page `images/k6098/0134.webp`,
sign agreement 0.9990 < 1.0). Ran a supplementary non-asserting diagnostic
(same encoder calls, no early stop) to capture all three pages' values for
this report:

| page                        | batch shape | single shape | sign agreement |
|------------------------------|-------------|--------------|-----------------|
| images/k6098/0134.webp        | (875, 128)  | (875, 128)   | 0.9990089285714285 |
| images/k4721/0004.webp        | (1139, 128) | (1139, 128)  | 1.0 |
| images/rg1965a/0001.webp      | (875, 128)  | (875, 128)   | 0.9989196428571429 |

Shapes match in all cases (mask-trimming works correctly — sequence
length is not batch-dependent). But 2 of 3 pages show ~0.999 sign
agreement, i.e. small floating-point drift between batched (mixed-size,
left-padded) and single-image forward passes on MPS produces occasional
sign flips near zero after mask-trimming. This is a genuine numerical
non-determinism, not an environmental error (model loaded fine, inference
completed normally on MPS, no OOM/crash).

**Decision applied per brief Step 6:** since the test FAILED (sign
agreement < 1.0 on 2/3 pages), changed `batch_size = 8` to `batch_size = 1`
in `src/belge_gozu/cli.py` `index_build` (single-line change,
included in the same commit). This forces index build to encode one page
at a time, matching the "single" path of the test and eliminating the
batch-induced drift for the persisted index.

This result and the batch_size=1 decision should be carried into the P0
baseline report as instructed.

## Full regression (Step 7)

`uv run pytest -q -m "not slow"`:
```
89 passed, 1 deselected, 6 warnings in 1.07s
```
(warnings are pre-existing, unrelated: fastapi/httpx deprecation, SWIG
typelib deprecation from an unrelated dependency — not touched by this
task.)

`make lint`:
```
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
60 files already formatted
0 errors, 0 warnings, 0 informations
```

## Files changed

- Modified: `/Users/barandincoguz/Desktop/project-delta/src/belge_gozu/index/encode.py`
- Modified: `/Users/barandincoguz/Desktop/project-delta/src/belge_gozu/cli.py` (batch_size 8→1, decision-rule outcome)
- Created: `/Users/barandincoguz/Desktop/project-delta/tests/index/test_encode_mask.py`

Commit: `fda4af7` — `fix(encode): mask-trimmed embeddings + explicit query format contract`
(exactly these 3 files staged by explicit path, per controller ruling R5;
pre-existing unrelated working-tree state — `.agents/`, `skills-lock.json`
— left untouched and unstaged).

## Self-review

- **Completeness:** all brief requirements implemented — `trim_by_mask`,
  `QueryFormat`-driven `__init__`/`encode_query`, `model_revision`,
  `doc_prompt`/`doc_prompt_sha256`, `FakeEncoder`/`Encoder` untouched,
  slow test added exactly as specified, decision rule applied and
  recorded.
- **Quality:** existing comments, docstrings, and device/dtype logic in
  `ColSmolEncoder.__init__` and `_run` preserved verbatim; only the
  specified lines changed. No unrelated refactors.
- **Discipline (YAGNI):** no scope creep — did not touch
  `encode_pages`'s per-4 chunking logic, did not add extra validation or
  logging beyond the brief.
- **Testing:** non-slow suite is pristine (89 passed, 0 failed, 0
  skipped besides the 1 deliberately-deselected slow test). `make lint`
  clean (ruff check, ruff format --check, pyright all pass).

## Concerns

1. The slow determinism test FAILED (as anticipated by the brief's own
   framing) — genuine ~0.999 sign-agreement drift between batched and
   single-image MPS inference on 2 of 3 sampled pages. Applied the
   brief's decision rule (`batch_size=1`), which trades index-build speed
   for determinism. This should be flagged in the P0 baseline report as a
   known MPS/batching numerical-stability finding, and future work might
   investigate whether it's MPS-specific (vs. CPU/CUDA) or reproducible
   with a fixed seed/dtype.
2. Minor deviation from the brief's literal Step 1 test code: dropped the
   unused `CPE_0_3_18` import (F401) and reflowed one inline-comment
   spacing to satisfy `ruff format --check` — both required for `make
   lint` (part of Step 7) to pass; no test assertions or logic changed.

## Fix report: R7 pre-review — reshape slow test to lock post-decision invariant

**Issue:** `test_batch_vs_single_sign_determinism`'s original `agree ==
1.0` assertion permanently fails on this hardware (measured 0.9990/0.9989
above), leaving the slow suite red forever even though the decision it
exists to make (`batch_size=1`) was already applied. Reshaped per
controller ruling R7 to assert the post-decision invariant instead of the
pre-decision question.

**Change** (`tests/index/test_encode_mask.py`,
`test_batch_vs_single_sign_determinism`):
- Threshold: `assert agree == 1.0` → `assert agree >= 0.995` (shape-equality
  assertion left untouched).
- Docstring/comment now records the measured drift and rationale exactly
  as specified:
  > "==1.0 MPS'te tutmadı (ölçüm: 0.9990/0.9989, 2026-08-26); karar: index
  > build batch_size=1 (cli.py). Bu eşik yalnız kaba gerilemeleri (ör.
  > maskeleme bozulması) yakalar; bit-exact kilit T10 canary testlerinde."

**Commands + output:**

`uv run pytest tests/index/test_encode_mask.py -m slow -v`:
```
tests/index/test_encode_mask.py::test_batch_vs_single_sign_determinism PASSED [100%]
1 passed, 2 deselected in 15.05s
```

`uv run pytest -q -m "not slow"`:
```
89 passed, 1 deselected, 6 warnings in 0.86s
```
(same pre-existing unrelated deprecation warnings as before.)

`make lint`:
```
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
60 files already formatted
0 errors, 0 warnings, 0 informations
```

**Commit:** new commit (no amend) `2cd099e` —
`test(encode): batch-determinism check locks post-decision threshold
(R7)`, staged only `tests/index/test_encode_mask.py` by explicit path
(controller ruling R5 still honored — `.agents/`, `skills-lock.json`
left untouched/unstaged).

**Outcome:** slow suite is green again while still catching gross
regressions (e.g. a masking bug that would push agreement well below
0.995); the original bit-exact question is explicitly deferred to T10
canary tests per the docstring.
