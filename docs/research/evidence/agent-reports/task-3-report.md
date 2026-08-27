# Task 3 Report: PackedIndex v2 — padding reddi + manifest (`index/store.py`)

## Summary

Implemented per brief, verbatim: `PackedIndex.build` now rejects any embedding
row that is all-zero (leaked padding) with a Turkish `ValueError`, added an
optional `manifest: IndexManifest | None = None` field (last, default `None`,
preserving existing 4-positional-arg call sites), and wired `save`/`load` to
persist/restore `manifest.json` via the existing `belge_gozu.index.manifest`
helpers. Legacy indexes (no `manifest.json`) still load fine with
`manifest=None`.

## Files changed

- `src/belge_gozu/index/store.py`
- `tests/index/test_store.py`

## TDD evidence

### RED — `uv run pytest tests/index/test_store.py -v` (before implementation)

Added the brief's three tests to `tests/index/test_store.py` first (imports
`pytest`/`Path` were already present in the file, so no import changes were
needed there).

```
tests/index/test_store.py::test_binarize_pack_bits PASSED                [ 12%]
tests/index/test_store.py::test_roundtrip PASSED                         [ 25%]
tests/index/test_store.py::test_build_rejects_length_mismatch PASSED     [ 37%]
tests/index/test_store.py::test_build_rejects_empty PASSED               [ 50%]
tests/index/test_store.py::test_build_rejects_zero_token_page PASSED     [ 62%]
tests/index/test_store.py::test_build_rejects_zero_rows FAILED           [ 75%]
tests/index/test_store.py::test_manifest_roundtrip FAILED                [ 87%]
tests/index/test_store.py::test_legacy_index_loads_without_manifest FAILED [100%]

test_build_rejects_zero_rows: Failed: DID NOT RAISE ValueError
test_manifest_roundtrip: TypeError: PackedIndex.build() got an unexpected keyword argument 'manifest'
test_legacy_index_loads_without_manifest: AttributeError: 'PackedIndex' object has no attribute 'manifest'

3 failed, 5 passed in 0.07s
```

All 3 new tests failed for the expected reasons (feature not yet implemented);
the 5 pre-existing tests were untouched and still passed.

### Implementation

`src/belge_gozu/index/store.py`:
- Added `from belge_gozu.index.manifest import IndexManifest, read_manifest, write_manifest`.
- `PackedIndex` dataclass: added `manifest: IndexManifest | None = None` as the
  last field.
- `build(cls, page_ids, embs, manifest: IndexManifest | None = None)`: kept the
  existing length/empty/zero-token checks unchanged; added, immediately after
  the zero-token check, per the brief exactly:
  ```python
  if (np.abs(e).sum(axis=1) == 0).any():
      raise ValueError(f"padding satırı sızmış: {pid}")
  ```
  Packing, offsets, and page_vecs (mean-sign) logic is byte-identical to
  before; only the final `return cls(...)` now also passes `manifest`.
- `save`: after the existing four writes, added
  `if self.manifest is not None: write_manifest(dir, self.manifest)`.
- `load`: added `manifest=read_manifest(dir)` to the returned `cls(...)` call;
  all other loads unchanged.

`tests/index/test_store.py`: appended the brief's three tests verbatim
(`test_build_rejects_zero_rows`, `test_manifest_roundtrip`,
`test_legacy_index_loads_without_manifest`) after the existing
`test_build_rejects_zero_token_page`. No existing test was altered or removed.

### GREEN — `uv run pytest tests/index/test_store.py -v`

```
tests/index/test_store.py::test_binarize_pack_bits PASSED                [ 12%]
tests/index/test_store.py::test_roundtrip PASSED                         [ 25%]
tests/index/test_store.py::test_build_rejects_length_mismatch PASSED     [ 37%]
tests/index/test_store.py::test_build_rejects_empty PASSED               [ 50%]
tests/index/test_store.py::test_build_rejects_zero_token_page PASSED     [ 62%]
tests/index/test_store.py::test_build_rejects_zero_rows PASSED           [ 75%]
tests/index/test_store.py::test_manifest_roundtrip PASSED                [ 87%]
tests/index/test_store.py::test_legacy_index_loads_without_manifest PASSED [100%]

8 passed in 0.06s
```

## Full regression

`uv run pytest tests/index -v`:
```
22 passed in 16.15s
```
(includes test_encode.py, test_encode_mask.py, test_hub.py, test_manifest.py,
test_store.py — all green.)

`uv run pytest -q -m "not slow"`:
```
92 passed, 1 deselected, 6 warnings in 1.01s
```
Warnings are pre-existing/unrelated (starlette/httpx deprecation, SWIG
`__module__` deprecations) — not introduced by this change.

`make lint`:
```
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
60 files already formatted
0 errors, 0 warnings, 0 informations
```

## Call-site compatibility check

Searched all repo call sites of `PackedIndex`:
- `src/belge_gozu/cli.py`: `PackedIndex.build(ids, embs).save(...)` — unaffected (new param is optional/keyword, default None).
- `src/belge_gozu/app/main.py`: `PackedIndex.load(s.index_dir)` — unaffected.
- `tests/conftest.py`, `tests/retrieval/test_core.py`: `PackedIndex.build(ids, embs)` — unaffected.
- `tests/telemetry/test_stages_integration.py`: `PackedIndex.load(...)` — unaffected.
- No site anywhere constructs `PackedIndex(...)` positionally outside the
  `cls(...)` call inside `build` itself, so adding `manifest` as the last
  field with a default did not break any existing construction.

## Diff (only the two files in scope)

```
git diff -- src/belge_gozu/index/store.py tests/index/test_store.py
```
- `store.py`: +14/-2 lines — import, `manifest` field, `build` signature +
  zero-row check + passing manifest through, `save` conditional write,
  `load` conditional read.
- `test_store.py`: +29 lines — three new tests appended, nothing removed or
  altered.

## Self-review

- **Completeness**: All 5 steps from the brief done — RED tests added and
  verified failing for the right reasons, implementation matches the brief's
  code block exactly (including the exact Turkish error message and match
  string `"padding satırı sızmış: p:1"`), GREEN achieved, full regression run,
  committed.
- **Quality**: Zero-row check placed immediately after the existing zero-token
  check, inside the same per-page loop, as specified. Packing/offsets/page_vecs
  logic untouched (verified via diff — those lines are unchanged aside from
  the final `return cls(...)` argument list). `save`/`load` changes are pure
  additions, no reordering of existing writes/reads.
- **Discipline**: Changed only `src/belge_gozu/index/store.py` and
  `tests/index/test_store.py`, exactly as scoped. Did not touch
  `pyproject.toml`, other index modules, or unrelated tests. Staged files by
  explicit path only (`git add src/belge_gozu/index/store.py
  tests/index/test_store.py`); never used `git add -A`/`git add .`. Untracked
  `.agents/` and `skills-lock.json` (pre-existing in working tree, unrelated to
  this task) were left untouched and unstaged.
- **Testing**: `tests/index` full suite green (22/22), whole non-slow suite
  green (92 passed, 1 deselected slow test), lint/format/pyright all clean.
  Confirmed via `grep` that no other code constructs `PackedIndex` positionally
  in a way the new trailing field could break.

## Concerns

None. The change is a narrow, additive extension exactly matching the brief;
no ambiguity encountered and no scope creep was needed.
