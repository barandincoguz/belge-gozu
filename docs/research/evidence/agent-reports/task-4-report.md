# Task 4 Report: Serve-time uyumluluk kontrolü

## Status: DONE

## Summary

Implemented `belge_gozu.index.compat` (fail-fast index/model/format
compatibility check) and wired it into `create_app` and the CLI, following
the brief's Step 1/3 code verbatim plus the controller rulings in the
dispatch message.

## Files changed

- **Created** `src/belge_gozu/index/compat.py` — `IndexCompatibilityError`
  and `check_compatibility(manifest, *, model_name, model_revision,
  query_format_id, index_dir) -> list[str]`, exactly per brief Step 3.
- **Created** `tests/app/test_compat.py` — the 5 tests from the brief
  (`test_missing_manifest_is_mismatch`, `test_matching_manifest_ok`,
  `test_format_mismatch_reported`, `test_create_app_fails_fast_on_mismatch`,
  `test_mismatch_override`), adapted per controller ruling:
  - `test_create_app_fails_fast_on_mismatch` and `test_mismatch_override`
    now unlink `(data_dir / "index" / "manifest.json")` before calling
    `create_app`, since the fixture now writes a manifest by default.
  - Added one extra light unit test, `test_write_manifest_legacy_cli`,
    invoking `belge-gozu index write-manifest --legacy` via
    `typer.testing.CliRunner` against the tiny_corpus index dir (env vars
    `BG_DATA_DIR`/`BG_INDEX_DIR`, following the pattern in
    `tests/test_cli.py`), asserting a manifest appears with
    `mask_policy == "none"`.
- **Modified** `src/belge_gozu/app/main.py` — imports
  `IndexCompatibilityError`, `check_compatibility` from
  `belge_gozu.index.compat` and `CPE_0_3_18` from
  `belge_gozu.index.manifest`. In `create_app`, the compat check is placed
  immediately after the encoder/answerer default-resolution block and
  before `retriever = TwoStageRetriever(...)`, exactly per brief Step 3
  (raises `IndexCompatibilityError` unless `s.allow_index_mismatch`, in
  which case it logs a warning and continues).
- **Modified** `src/belge_gozu/config.py` — added
  `allow_index_mismatch: bool = False`.
- **Modified** `src/belge_gozu/cli.py` — added
  `index write-manifest --legacy` command:
  - `model_name` from `Settings().retriever_model`, `model_revision="unknown"`,
    `query_format=CPE_0_3_18`, `doc_prompt_sha256="unknown"`,
    `quantization="sign-1bit"`, `mask_policy="none"`, `render=RenderConfig()`.
  - `corpus_checksum` computed via `belge_gozu.index.manifest.corpus_checksum`.
  - `n_pages`/`n_tokens` from the loaded index (`PackedIndex.load(s.index_dir)`:
    `len(page_ids)` and `int(offsets[-1])`, i.e. total token count).
  - `built_at` via `datetime.now(UTC).isoformat()`.
  - `git_commit` via `subprocess.run(["git", "rev-parse", "--short", "HEAD"], ...)`,
    falling back to `"unknown"` on `CalledProcessError`/`FileNotFoundError`/`OSError`.
  - `engine_versions` for `colpali-engine`/`transformers`/`torch` via
    `importlib.metadata.version`, each falling back to `"unknown"` on
    `PackageNotFoundError`.
  - If `--legacy` is not passed, raises `typer.BadParameter` (only mode
    implemented currently, per brief scope).
- **Modified** `tests/conftest.py` — `tiny_corpus` fixture now writes a
  manifest after `idx.save(idx_dir)`/`meta.to_parquet(...)`:
  ```python
  write_manifest(
      idx_dir, make_manifest(corpus_checksum=corpus_checksum(idx_dir), n_pages=3, n_tokens=24)
  )
  ```
  with `from belge_gozu.index.manifest import corpus_checksum, write_manifest`
  and `from tests.index.test_manifest import make_manifest` added at the top.
  The checksum is computed from the files actually on disk (not passed into
  `PackedIndex.build`), per the controller ruling.

## No other test files needed adjustment

Ran the full app + full "not slow" suite after the fixture change: all
existing `create_app` callers (`tests/app/test_api.py`, all 12 tests) passed
unmodified — `tiny_corpus`'s `FakeEncoder` lacks `model_revision`/
`query_format` attrs, so those checks are skipped via `getattr(..., None)`/
`getattr(..., CPE_0_3_18)`, and `s.retriever_model` defaults to
`"vidore/colSmol-500M"` matching `make_manifest`'s default `model_name`.
`tests/telemetry/test_stages_integration.py` builds `TwoStageRetriever`
directly from `PackedIndex.load`/`meta.parquet` without going through
`create_app`, so it is unaffected by the compat check entirely. No other
index-building test fixtures call `create_app`, so no further adjustments
were required.

## TDD trail

1. RED: wrote `tests/app/test_compat.py` (brief tests + fixture unlink
   adaptation + 1 extra CLI test) and `src/belge_gozu/index/compat.py`
   standalone first. `uv run pytest tests/app/test_compat.py -v` showed the
   3 pure `check_compatibility` tests passing immediately (module-level
   logic has no other dependency) and the 2 `create_app`/CLI-integration
   tests failing as expected (`DID NOT RAISE IndexCompatibilityError`; `No
   such command 'write-manifest'`).
2. Implemented the `config.py`, `app/main.py`, `cli.py` wiring.
3. GREEN: `uv run pytest tests/app/test_compat.py -v` → 6/6 passed.

## Verification (final, post-commit)

```
uv run pytest tests/app -v            -> 18 passed
uv run pytest -q -m "not slow"        -> 111 passed, 1 deselected
make lint                             -> ruff check/format clean, pyright 0 errors
```

`git status` after commit: only pre-existing untracked `.agents/` and
`skills-lock.json` remain (not part of this task, not staged/committed).

## Commit

`a09ee75` — `feat(serve): fail-fast index/model/format compatibility check`
Staged explicitly (no `git add -A`): `src/belge_gozu/index/compat.py`,
`src/belge_gozu/app/main.py`, `src/belge_gozu/config.py`,
`src/belge_gozu/cli.py`, `tests/app/test_compat.py`, `tests/conftest.py`.

## Concerns / notes for reviewer

- The brief's Interfaces prose says model_revision mismatch "encoder
  'unknown' veriyorsa atlanır, uyarı listelenir" (skipped, warning listed),
  but the brief's own Step 3 code sample only skips silently (no entry
  appended to `problems`) when `model_revision` is `None`/`"unknown"` —
  no separate warning list exists in the function's return type (`list[str]`
  is the uyumsuzluk list itself). I implemented Step 3's code verbatim,
  which matches the controller ruling ("revision check skipped (by
  design)"). No behavior change needed; flagging only because the prose and
  code sample differ slightly and I deferred to the code + controller
  ruling as authoritative.
- `index write-manifest` currently only supports `--legacy` (raises
  `typer.BadParameter` otherwise) since the brief only specifies the
  `--legacy` path; no non-legacy manifest-stamping mode was requested.
