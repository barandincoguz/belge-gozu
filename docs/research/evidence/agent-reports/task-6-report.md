# Task 6 Report: Benchmark veri modeli (`bench/dataset.py`)

## TDD Evidence

**RED** (before implementation):
```
uv run pytest tests/bench/test_dataset.py -v
ERROR collecting tests/bench/test_dataset.py
ModuleNotFoundError: No module named 'belge_gozu.bench.dataset'
```

**GREEN** (after implementation, brief's exact test code):
```
tests/bench/test_dataset.py::test_load_verified_only PASSED
tests/bench/test_dataset.py::test_answerable_requires_gold_pages PASSED
tests/bench/test_dataset.py::test_unanswerable_requires_reason_and_no_gold PASSED
tests/bench/test_dataset.py::test_gold_page_doc_consistency PASSED
tests/bench/test_dataset.py::test_split_assignment PASSED
5 passed in 0.01s
```

**Post-lint-fix addition**: added `test_bench_stats_counts_all_slices_with_zero_default`
to exercise `bench_stats` (imported in the brief's test but never called there, which
ruff F401-flagged as unused). This test also locks in the brief's "tüm dilimler 0
varsayılanıyla dahil" (all 12 slices default to 0) requirement, which otherwise had
no test coverage.

**Final regression** (after all fixes):
```
uv run pytest tests/bench -v         -> 6 passed
uv run pytest -q -m "not slow"       -> 98 passed, 1 deselected
make lint (ruff check + ruff format --check + pyright)
  -> All checks passed! / 64 files already formatted / 0 errors, 0 warnings, 0 informations
```

## Files Changed

- `src/belge_gozu/bench/__init__.py` — new, empty (package marker)
- `src/belge_gozu/bench/dataset.py` — new: `QueryStyle`, `Slice`, `UnanswerableReason` literals;
  `BenchQuestion` pydantic model with three `model_validator(mode="after")` methods
  (`_check_answerability`, `_check_gold_page_doc_consistency`, `_check_verification`);
  `load_bench`, `bench_stats`, `load_splits`, `question_split`.
- `tests/bench/__init__.py` — new, empty (package marker, per instructions so later
  tasks can `from tests.bench.test_dataset import q_dict`)
- `tests/bench/test_dataset.py` — brief's test code, reformatted for ruff (import
  sort, one wrapped long line, `ruff format` multi-line call reformatting) plus the
  added `bench_stats` coverage test. No test semantics changed.
- `data/bench/splits_v1.json` — new skeleton, exact content
  `{"dev_docs": [], "test_docs": []}`. This path matches the repo's `data/*`
  gitignore rule (only `data/manifest/` is excepted), so it required
  `git add -f` to stage — did not touch `.gitignore` (out of task file scope,
  and shared config a concurrent session might also rely on).

## Implementation Notes

- `load_bench`: reads with `path.read_text(...).splitlines()`, enumerates from 1,
  skips blank lines (defensive, doesn't affect line numbering of real content since
  none of the brief's fixtures produce blank lines). Catches
  `(json.JSONDecodeError, ValidationError, KeyError, TypeError)` and re-raises as
  `ValueError(f"bench satır {i}: {e}") from e`, matching the existing
  `load_manifest_from_text` pattern in `src/belge_gozu/corpus/manifest.py`.
  `TypeError` added defensively (e.g. non-dict JSON line) — not required by the
  brief's exact exception list but consistent with "wrap validation-shaped errors."
  Empty result (no verified rows, or empty file) raises
  `ValueError("bench boş: yüklenecek soru yok")`.
- `BenchQuestion` cross-field rules implemented exactly per brief:
  - `answerable=True` → `gold_page_ids` non-empty, `reference_answer` non-empty,
    `unanswerable_reason is None`.
  - `answerable=False` → `gold_page_ids == []`, `unanswerable_reason` required.
  - Every `gold_page_ids` element must contain `":"`; the text before the first
    `":"` must be in `gold_doc_ids`.
  - `verification_status == "verified"` → `verified_by` non-empty.
- `question_split`: primary doc = `gold_doc_ids[0]` when non-empty → `"dev"` if in
  `splits["dev_docs"]`, `"test"` if in `splits["test_docs"]`, else `"dev"` (safe
  default, doc in neither set). Empty `gold_doc_ids` → deterministic
  `sha256(question_id)` parity split, exactly as specified.
- `bench_stats`: builds the 0-default dict from `get_args(Slice)` rather than a
  hand-duplicated tuple of the 12 slice strings, so the literal type and the stats
  keys can't drift apart.
- `load_splits`: `{"dev_docs": set(...), "test_docs": set(...)}`, tolerant of
  missing keys via `.get(..., [])`.

## Self-Review

- **Completeness**: all four cross-field validators from the brief are present and
  individually exercised by the brief's own tests (`test_answerable_requires_gold_pages`,
  `test_unanswerable_requires_reason_and_no_gold`, `test_gold_page_doc_consistency`;
  `verified_by` rule is exercised implicitly by every fixture using
  `verification_status="verified"` + non-empty `verified_by`, and by the "draft"
  row in `test_load_verified_only` which has no `verified_by` override needed since
  the rule only fires for `verified`). All four documented interface functions
  (`load_bench`, `bench_stats`, `load_splits`, `question_split`) are implemented and
  tested. `data/bench/splits_v1.json` skeleton created with exact specified content.
- **Quality**: reused the existing repo convention for JSONL-with-line-numbers error
  wrapping (`corpus/manifest.py`) rather than inventing a new one. Derived the 12
  slice keys from the `Slice` Literal via `get_args` instead of retyping them,
  removing a duplication/drift risk.
- **Discipline (YAGNI)**: no extra fields, no extra public functions, no `Enum`
  wrapper class, no CLI hookup — this task is dataset-model only, metrics/harness
  are later tasks per the brief.
- **Testing (pristine output)**: `uv run pytest tests/bench -v` → 6 passed, no
  warnings. Full suite `uv run pytest -q -m "not slow"` → 98 passed, 1 deselected;
  the only warnings present (`StarletteDeprecationWarning`, SWIG
  `DeprecationWarning`s) are pre-existing and unrelated to this task's files.
  `make lint` → 0 errors/warnings/informations.

## Concerns

- `data/bench/splits_v1.json` needed `git add -f` because `data/*` is gitignored
  (only `data/manifest/` is excepted in `.gitignore`). The file is now tracked, but
  the `.gitignore` rule itself is unchanged, so P1/T12 (which fills this file in)
  or any future new files under `data/bench/` will need the same `-f` treatment (or
  a `.gitignore` update) unless a later task adds a `!data/bench/` exception. Flagging
  this for the controller/T12 rather than editing `.gitignore` myself, since it's
  outside this task's explicit file list and touches shared config a concurrent
  session may also be relying on.
- Concurrent session activity was visible throughout (unstaged modifications to
  `docs/research/findings/2026-08-26-baseline.md`, `docs/research/metrics-catalog.md`,
  `scripts/loadgen.py`, `src/belge_gozu/app/main.py`, `src/belge_gozu/cli.py`,
  `tests/app/test_api.py`, `tests/test_cli.py`, plus untracked `.agents/` and
  `skills-lock.json`). None of these were staged or touched; verified via
  `git status --short` immediately before and after the commit that only the 5
  task-scoped files were included (`git show --stat HEAD`).
- No transient `index.lock` was encountered; the commit succeeded on the first
  attempt.

## Follow-up (Controller Ruling R10)

The coordinator confirmed the `git add -f` concern above and directed a proper
fix: add explicit `.gitignore` exceptions for `data/bench/` (same pattern as the
existing `data/manifest/` exception) rather than relying on force-add.

**Change**: `.gitignore` — after the existing `!data/manifest/**` line, added:
```
!data/bench/
!data/bench/**
```

**Verification**:
```
$ git check-ignore data/bench/splits_v1.json || echo "not ignored"
not ignored

$ git check-ignore data/meta.parquet data/index/tokens.npy
data/meta.parquet
data/index/tokens.npy
```
Both still correctly ignored; `data/bench/` is now correctly excepted.

`git status --short` before committing showed no unexpected `data/` paths
surfacing (`data/images`, `data/index`, `data/pdf`, `data/meta.parquet` all
remained untracked/ignored as before) — only pre-existing concurrent-session
modifications (`docs/research/...`, `scripts/loadgen.py`,
`src/belge_gozu/app/main.py`, `src/belge_gozu/cli.py`, `tests/app/test_api.py`,
`tests/test_cli.py`) plus untracked `.agents/`/`skills-lock.json`, none of which
were touched or staged.

**Commit**: staged only `.gitignore` (`git add .gitignore`, confirmed via
`git diff --cached` showing exactly the 2-line addition) and committed as
`chore: track data/bench benchmark artifacts (R10)` (commit `63a98ab`). No
`index.lock` contention encountered.

This resolves the earlier concern: future files added under `data/bench/`
(e.g. P1/T12's filled-in `splits_v1.json`, or new bench artifacts) will no
longer need `git add -f`.

## Fix Report (Review Verdict: Needs Fixes)

Review found one Important (coverage gap, not correctness) and one Minor issue.
Both addressed:

**IMPORTANT — 5 untested validator failure branches.** Added targeted negative
tests to `tests/bench/test_dataset.py`, each `pytest.raises(ValueError)` built
off the existing `q_dict` helper:
1. `test_answerable_requires_reference_answer` — `answerable=True`,
   `reference_answer=""` → raises (dataset.py `_check_answerability`,
   the `reference_answer` branch).
2. `test_answerable_forbids_unanswerable_reason` — `answerable=True`,
   `unanswerable_reason="korpus-disi"` → raises (the
   `unanswerable_reason is not None` branch).
3. `test_unanswerable_forbids_nonempty_gold_pages` — `answerable=False`,
   `gold_page_ids=["k4721:4"]` (default `gold_doc_ids=["k4721"]` left
   unchanged so the doc-consistency validator would otherwise pass; this
   isolates the `gold_page_ids != []` branch in `_check_answerability`,
   confirmed it fires before `_check_gold_page_doc_consistency` since
   pydantic runs `mode="after"` validators in declaration order).
4. `test_gold_page_id_requires_colon` — `gold_page_ids=["k4721"]` (no `":"`)
   → raises the no-colon branch of `_check_gold_page_doc_consistency`.
5. `test_verified_requires_verified_by` — `verification_status="verified"`
   (default), `verified_by=""` → raises `_check_verification`.

**MINOR — `question_split`'s "dev" fallback lacked an intent comment and a
test.** In `src/belge_gozu/bench/dataset.py`, the fallback line now reads:
```python
# T12 öncesi doldurulmamış split → güvenli varsayılan dev (hiçbir kümede yok)
return "dev"
```
Added `test_split_assignment_unknown_doc_defaults_to_dev`: builds a
`BenchQuestion` with `gold_doc_ids=["k9999"]` / `gold_page_ids=["k9999:4"]`
(a doc absent from both `dev_docs={"k4721"}` and `test_docs={"k6098"}`) and
asserts `question_split(...) == "dev"`.

**Covering test run:**
```
uv run pytest tests/bench/test_dataset.py -v
12 passed in 0.04s
```
(6 pre-existing + 6 new: the 5 required negative-path tests plus the
split-fallback test.)

**Full regression:**
```
uv run pytest -q -m "not slow"
105 passed, 1 deselected, 6 warnings (pre-existing, unrelated warnings only)

make lint
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed! / 64 files already formatted / 0 errors, 0 warnings, 0 informations
```

**Commit:** staged only `src/belge_gozu/bench/dataset.py` and
`tests/bench/test_dataset.py` (verified via `git status --short` /
`git diff --cached --stat` — 2 files changed, 42 insertions(+), 1 deletion(-))
and committed as `test(bench): negative-path coverage for all validator
branches` (commit `b4cfe1e`). No `index.lock` contention encountered.

Note: by the time this fix round started, the concurrent session's earlier
in-progress modifications (`docs/research/...`, `scripts/loadgen.py`,
`src/belge_gozu/app/main.py`, `src/belge_gozu/cli.py`, `tests/app/test_api.py`,
`tests/test_cli.py`) no longer appeared in `git status` — that session must
have committed or reverted them independently; not touched by this task.
