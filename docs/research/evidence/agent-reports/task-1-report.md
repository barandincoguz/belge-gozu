# Task 1 Report: Index manifest modeli (`index/manifest.py`)

## What was implemented

Created `src/belge_gozu/index/manifest.py` per the brief, providing the
`IndexManifest`/`QueryFormat` contract that later P0 tasks (encoder, store,
serve-time compat check) will consume:

- `QueryFormat(BaseModel)` — `format_id`, `prefix`, `suffix_token`, `n_suffix`,
  `trailing_newline`, with a `render(text: str) -> str` method that applies
  prefix + text + repeated suffix token, plus optional trailing newline.
- `CPE_0_3_18` and `TRAIN_COMPAT_V1` constants (values verbatim from the brief),
  with the brief's Turkish comment explaining the training-format uncertainty
  to be resolved in T11.
- `RenderConfig(BaseModel)` — `dpi=150`, `format="webp"`, `quality=80`.
- `IndexManifest(BaseModel)` — all 14 fields verbatim from the brief
  (`schema_version`, `model_name`, `model_revision`, `engine_versions`,
  `query_format`, `doc_prompt_sha256`, `quantization`, `mask_policy`, `render`,
  `corpus_checksum`, `n_pages`, `n_tokens`, `built_at`, `git_commit`).
- `corpus_checksum(index_dir: Path) -> str` — sha256 over the concatenated
  bytes of `page_ids.json` and `meta.parquet`.
- `write_manifest(dir, m)` / `read_manifest(dir) -> IndexManifest | None` —
  JSON round-trip via `manifest.json`, returning `None` when the file is
  absent.

Also created `tests/index/test_manifest.py` (brief's Step 1 test code) and the
missing root `tests/__init__.py` (empty file, per controller ruling R1 — later
tasks import test helpers like `make_manifest` across modules via
`from tests.index.test_manifest import make_manifest`).

## Deviations from the brief's literal code (both needed for clean lint)

1. **`src/belge_gozu/index/manifest.py`**: brief's Step 3 code includes
   `import json` at the top, but `json` is never referenced (both
   `write_manifest`/`read_manifest` use pydantic's own
   `model_dump_json`/`model_validate_json`). `ruff check` flags this as F401
   (unused import, rule F is in the selected set). Removed the import; no
   behavior change.
2. **`tests/index/test_manifest.py`**: brief's Step 1 code imports
   `QueryFormat` from `belge_gozu.index.manifest` but never references it by
   name in the test body (only the `CPE_0_3_18`/`TRAIN_COMPAT_V1` instances
   and `RenderConfig()` are used). Same F401 violation. Removed the unused
   import; test semantics/assertions are unchanged.
3. `ruff format` reflowed the two multi-line `QueryFormat(...)` constant
   constructions in `manifest.py` from 2-line to one-arg-per-line — applied
   via `uv run ruff format` (auto-formatter, no manual code change).

No other deviations. Field names, types, defaults, docstring/comment content,
and function signatures all match the brief verbatim.

## TDD evidence

### RED

Command: `uv run pytest tests/index/test_manifest.py -v`

```
ERROR collecting tests/index/test_manifest.py
ImportError while importing test module '.../tests/index/test_manifest.py'.
tests/index/test_manifest.py:4: in <module>
    from belge_gozu.index.manifest import (
E   ModuleNotFoundError: No module named 'belge_gozu.index.manifest'
=========================== short test summary info ============================
ERROR tests/index/test_manifest.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.06s ===============================
```

Matches the brief's expected failure exactly.

### GREEN

Command: `uv run pytest tests/index/test_manifest.py -v`

```
tests/index/test_manifest.py::test_query_format_render PASSED            [ 25%]
tests/index/test_manifest.py::test_roundtrip PASSED                      [ 50%]
tests/index/test_manifest.py::test_read_missing_returns_none PASSED      [ 75%]
tests/index/test_manifest.py::test_corpus_checksum_changes_with_content PASSED [100%]

============================== 4 passed in 0.07s ===============================
```

### Full regression

Command: `uv run pytest -q -m "not slow"`

```
........................................................................ [ 82%]
...............                                                          [100%]
87 passed, 6 warnings in 1.15s
```

(6 warnings are pre-existing: fastapi/httpx `StarletteDeprecationWarning` and
pymupdf/swig `DeprecationWarning`s — unrelated to this change, present before
this task's files were added.)

Command: `make lint` (`ruff check . && ruff format --check . && pyright`)

```
All checks passed!
59 files already formatted
0 errors, 0 warnings, 0 informations
```

## Files changed

- `src/belge_gozu/index/manifest.py` (new, 77 lines)
- `tests/index/test_manifest.py` (new, 57 lines)
- `tests/__init__.py` (new, empty — R1)

Commit: `bb09374` — `feat(index): index manifest model with query-format contract`
(3 files changed, 134 insertions(+), staged by explicit path per R5 — did not
touch the pre-existing unstaged `scripts/loadgen.py` modification or the
untracked `.agents/`/`skills-lock.json`).

## Self-review

- **Completeness**: every symbol/field/method/constant from the brief's
  "Produces" list is implemented with matching names and types. Test file
  matches brief's Step 1 code (minus the unused-import fix). `tests/__init__.py`
  created per R1.
- **Quality**: names and structure match the brief and the existing
  `index/store.py` style (minimal, dataclass/pydantic + pathlib, Turkish
  comments only where they add domain context). No superfluous abstractions.
- **Discipline (YAGNI)**: implementation is exactly the brief's contract —
  no extra fields, no extra methods, no speculative validation logic beyond
  what pydantic gives for free.
- **Testing**: the 4 tests exercise real behavior (render formatting for both
  formats, JSON round-trip including schema_version presence, missing-file
  None-return, checksum sensitivity to content changes) — no mocks, real
  tmp_path filesystem I/O. Test output is pristine: no warnings, no skips,
  deterministic.
- Verified via `git status`/`git diff --stat` before staging that only the
  three intended files were added, and that the pre-existing unrelated
  `scripts/loadgen.py` diff plus untracked `.agents/`/`skills-lock.json` were
  left out of the commit.

## Concerns

None. The two unused-import removals are mechanical lint fixes with zero
behavioral impact and are called out above for transparency; everything else
matches the brief verbatim.
