# R17 — doc_prompt_sha256 gate gap closed

## Summary

`check_compatibility` in `src/belge_gozu/index/compat.py` now compares
`doc_prompt_sha256` in addition to model name/revision, `query_format.format_id`,
`mask_policy`, and `corpus_checksum`. `create_app()` in
`src/belge_gozu/app/main.py` now passes
`doc_prompt_sha256=getattr(encoder, "doc_prompt_sha256", None)` at its
`check_compatibility(...)` call site. Tests added in `tests/app/test_compat.py`.

## Changes

### `src/belge_gozu/index/compat.py`

Added a new keyword-only parameter `doc_prompt_sha256: str | None = None` to
`check_compatibility`, placed after `query_format_id` (before `index_dir`).
Mirrors the existing `model_revision` skip logic exactly: the check is
skipped when the caller passes `None` or the literal string `"unknown"`
(covers test doubles and legacy manifests that don't expose a doc-prompt
hash). When both sides are known and differ, a problem string is appended
with truncated (12-char) hashes from each side so a human can eyeball the
difference:

```python
if (
    doc_prompt_sha256
    and doc_prompt_sha256 != "unknown"
    and manifest.doc_prompt_sha256 != doc_prompt_sha256
):
    problems.append(
        "doc_prompt_sha256: "
        f"indeks={manifest.doc_prompt_sha256[:12]} serve={doc_prompt_sha256[:12]}"
    )
```

### `src/belge_gozu/app/main.py`

Added one line at the existing `check_compatibility(...)` call site:

```python
doc_prompt_sha256=getattr(encoder, "doc_prompt_sha256", None),
```

`ColSmolEncoder` already exposes `self.doc_prompt_sha256` (set in
`src/belge_gozu/index/encode.py`, sha256 of the resolved doc prompt). Test
doubles (`FakeEncoder`, ad-hoc stub encoders) don't have this attribute, so
`getattr(..., None)` naturally skips the new check for them — same pattern
as the existing `model_revision` line.

### `tests/app/test_compat.py`

Added three unit tests, following the file's existing `make_manifest(...)`
inline-helper style (writing `page_ids.json`/`meta.parquet` into `tmp_path`
and computing `corpus_checksum` for a passing baseline):

- `test_matching_doc_prompt_ok` — manifest and serve both `"d" * 64` →
  `problems == []`.
- `test_doc_prompt_mismatch_reported` — manifest `"d" * 64` vs serve
  `"e" * 64` → a problem string containing `"doc_prompt"`.
- `test_doc_prompt_none_or_unknown_skips_check` — serve value `None` and
  `"unknown"`, both against a manifest with a real hash → no `doc_prompt`
  problem in either case.

## The `query_format_id` fallback question

The call site derives the format-id fallback as
`getattr(encoder, "query_format", CPE_0_3_18).format_id`, which is a stale
default: current `Settings.query_format_id` defaults to `"train-compat-v1"`,
not `"cpe-0.3.18"` (`CPE_0_3_18`'s id). I tried changing the fallback to read
the configured value instead:

```python
query_format_id=getattr(encoder, "query_format", resolved_query_format).format_id,
```

Ran `uv run pytest -q -m "not slow"` with this change in place: **15 tests
failed**, all `IndexCompatibilityError: ... query_format: indeks=cpe-0.3.18
serve=train-compat-v1` — `tests/app/test_api.py` (14 tests, e.g.
`test_healthz`, `test_search_returns_hits`, `test_ask_returns_answer_and_logs`,
...) and `tests/telemetry/test_prom.py::test_metrics_endpoint_exposes_exhaustive_stage_and_index_revision`.

Root cause: these tests use `tiny_corpus`'s `FakeEncoder` (no `query_format`
attribute) together with a `manifest` built via `make_manifest()`, whose
default `query_format=CPE_0_3_18` (`"cpe-0.3.18"`). They construct `Settings`
without overriding `query_format_id`, so the resolved default
(`"train-compat-v1"`) would now flow into `check_compatibility` and
legitimately conflict with the fixture's `"cpe-0.3.18"` manifest — a real
mismatch introduced by the fixture/default drift, not a bug in the fallback
logic itself, but exactly the "weaken a test to force it" trap the task
warned against.

Per instructions, **left the fallback as `CPE_0_3_18` (unchanged)** and did
not touch the tests to accommodate the alternative. Confirmed the revert
restores a fully green `-m "not slow"` run (157 passed) before committing.

## Verification

### `uv run pytest -q -m "not slow"`

```
157 passed, 4 deselected in 1.17s (final run just before commit)
```

Also ran `tests/app/test_compat.py` alone: `9 passed`.

### `make lint`

```
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
79 files already formatted
0 errors, 0 warnings, 0 informations
```

### End-to-end proof (real model, `data/index-traincompat-1bit`)

Script built `create_app()` twice via `uv run python`, once with default
settings, once with `BG_DOC_PROMPT_ID=processor-default` set beforehand
(fresh `Settings()` instance so the env var is picked up).

Run 1 (default settings) — verbatim relevant output:

```
=== Run 1: default settings (expect success) ===
Loading weights: 100%|██████████| 490/490 [00:00<00:00, 5585.64it/s]
Loading weights: 100%|██████████| 448/448 [00:00<00:00, 17302.79it/s]
OK: create_app() succeeded with default settings.
```

Run 2 (`BG_DOC_PROMPT_ID=processor-default`) — verbatim relevant output:

```
=== Run 2: BG_DOC_PROMPT_ID=processor-default (expect IndexCompatibilityError) ===
Loading weights: 100%|██████████| 490/490 [00:00<00:00, 8905.84it/s]
Loading weights: 100%|██████████| 448/448 [00:00<00:00, 16963.05it/s]
OK: IndexCompatibilityError raised as expected:
indeks/serve uyumsuzluğu: doc_prompt_sha256: indeks=3d11cdfb8bca serve=bb16e19ccb55
```

The second run raised `IndexCompatibilityError` with a message explicitly
naming `doc_prompt_sha256` and showing the two differing truncated hashes,
exactly as the gate is meant to do. No index rebuild was performed; the
verification script was deleted from the scratchpad after use.

## Commit

```
8b77b54 fix(serve): fail fast on document-prompt mismatch too (R17, G0.5)

 src/belge_gozu/app/main.py     |  1 +
 src/belge_gozu/index/compat.py | 10 ++++++++
 tests/app/test_compat.py       | 52 ++++++++++++++++++++++++++++++++++++++++++
 3 files changed, 63 insertions(+)
```

Staged explicitly by path (`git add src/belge_gozu/app/main.py
src/belge_gozu/index/compat.py tests/app/test_compat.py`) — no `git add -A`
or `git add .` used. Untracked `.agents/`, `data/bench/results/`,
`skills-lock.json` were left untouched, as they're unrelated to this task.

## Concerns

None blocking. One note for the record: the `Settings.query_format_id`
default (`"train-compat-v1"`) has drifted from the `tiny_corpus`/
`make_manifest` test fixtures' default (`CPE_0_3_18`, `"cpe-0.3.18"`) — they
only agree today because the stale `CPE_0_3_18` fallback in `main.py`
happens to match the fixture default too. This is exactly the trap the task
description flagged; fixing it properly would mean updating the `tiny_corpus`
/`make_manifest` fixtures to use `TRAIN_COMPAT_V1` (or whatever the current
intended default is) in the same change as fixing the fallback, which is out
of scope for R17.
