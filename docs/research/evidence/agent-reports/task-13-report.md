# Task 13 Report: Kalite telemetrisi genişletmesi

## Status
DONE.

## Commit
`0e898c4` — `feat(telemetry): retrieval provenance fields (pipeline, index revision, stage series)`
(branch `feat/p0-retrieval-correctness`, 8 files changed, 151 insertions, 0 deletions)

## Deviation from brief — controller rulings applied instead
The task brief (`task-13-brief.md`) describes `detail["retrieval"]` as
`{"candidates": [first 20 {page_id, score}], "stage_latencies": {...}}`. Per the
controller rulings supplied in my instructions (which explicitly refine the brief),
this was **not** implemented. Instead, R13 was followed exactly:
`detail["retrieval"] = {"query_format": <format_id or None>, "quantization": <or None>}`
only — no top-20 candidate list, no stage_latencies duplication (stage timings already
live in `detail["stages"]` from Task 5 and are now also fed into the Prometheus
`bg_stage_duration_seconds` histogram for stage names not covered by `_STAGE_COLS`).

## Changes

### `src/belge_gozu/telemetry/schema.py`
- `EVENTS_DDL`: added `pipeline TEXT` and `index_revision TEXT` columns (placed after
  `error_type`, before `detail`).
- `RequestEvent`: added `pipeline: str | None = None` and
  `index_revision: str | None = None`.

### `src/belge_gozu/telemetry/recorder.py`
- `_COLUMNS`: added `"pipeline"`, `"index_revision"` (before `"detail"`) so `INSERT`
  covers the new fields.
- `EventRecorder.__init__`: after `CREATE TABLE IF NOT EXISTS`, runs a best-effort
  `ALTER TABLE events ADD COLUMN <col> TEXT` for each of the two new columns, wrapped
  in `try/except sqlite3.OperationalError: pass` — migrates pre-existing DBs (old
  table shape) without ever raising. Preserves the existing "telemetry never drops a
  request" principle.

### `src/belge_gozu/telemetry/prom.py`
- `set_app_info`: gained required `index_revision: str` and `query_format: str`
  keyword params, both folded into the `bg_app_info` Info metric labels.
- `observe`: after the existing `_STAGE_COLS` loop, added a second loop over
  `ev.detail.get("stages", {})` that observes any stage name **not** already in
  `_STAGE_COLS` into the same `bg_stage_duration_seconds` histogram (ms→s conversion),
  avoiding double-counting for the four already-covered stages. This makes
  `exhaustive_maxsim` (and any future ad hoc stage) visible in `/metrics`.

### `src/belge_gozu/app/main.py`
- In `create_app`, after building `retriever`: derives `index_revision`,
  `query_format_id`, `quantization` from `index.manifest` (or `None`/`None`/`None`
  when no manifest) using
  `f"{manifest.corpus_checksum[:12]}/{manifest.query_format.format_id}/{manifest.quantization}"`.
- `prom.set_app_info(...)` call now passes `index_revision=index_revision or "unknown"`
  and `query_format=query_format_id or "unknown"`.
- `build_event`: fills `pipeline=s.retrieval_pipeline`, `index_revision=index_revision`,
  and adds `"retrieval": {"query_format": query_format_id, "quantization": quantization}`
  to `detail`, alongside the pre-existing `hits`/`threshold`/`stages`/fingerprint keys
  (all preserved unchanged).

## Tests added (TDD: RED confirmed before implementation, then GREEN)
- `tests/telemetry/test_schema.py`: `test_request_event_accepts_pipeline_and_index_revision`;
  extended `test_ddl_creates_table_and_indexes` to assert `pipeline`/`index_revision` columns.
- `tests/telemetry/test_recorder.py`: `test_migration_adds_new_columns_to_old_table` — builds
  a DB with the OLD `EVENTS_DDL` (inlined, without the two new columns), constructs
  `EventRecorder` against it (migration runs, no exception), then records an event with
  `pipeline`/`index_revision` set and verifies both the column existence (`PRAGMA table_info`)
  and the persisted values.
- `tests/telemetry/test_prom.py`: updated the existing `set_app_info` call to pass the two
  new required kwargs; added `test_observe_records_uncovered_stage_names` (unit-level,
  `exhaustive_maxsim` observed into `bg_stage_duration_seconds`) and
  `test_metrics_endpoint_exposes_exhaustive_stage_and_index_revision` (full app + `tiny_corpus`
  fixture, `/search` then `/metrics`, asserts the stage bucket line and `index_revision=` in
  the `bg_app_info` line).
- `tests/app/test_api.py`: `test_search_records_pipeline_and_index_revision` — after `/search`
  on the default `exhaustive` pipeline, asserts `pipeline == "exhaustive"`, `index_revision`
  is non-null and contains `"cpe-0.3.18"`, and `detail["retrieval"] == {"query_format":
  "cpe-0.3.18", "quantization": "sign-1bit"}` with no `"candidates"` key inside `retrieval`
  (confirms R13's "identity fields only" ruling).

## Test summary
- `uv run pytest tests/telemetry tests/app -v` → 42 passed (RED confirmed first with 7
  failures matching the new/changed assertions, then GREEN after implementation).
- `uv run pytest -q -m "not slow"` → 133 passed, 1 deselected.
- `make lint` (ruff check + ruff format --check + pyright) → all clean (one test file
  needed `ruff format` for a line-wrap, applied and re-verified).

## Concerns / notes for reviewer
- `set_app_info`'s two new params are required kwargs (no default) rather than optional —
  there was exactly one call site (`app/main.py`) and one test file (`test_prom.py`), both
  updated, so this was safe and keeps the API honest (no silent "unknown" default inside
  `PromMetrics` itself; the "unknown" fallback lives at the call site in `main.py` where the
  manifest-absent case is actually known).
- `index_revision`/`query_format_id`/`quantization` are computed once in `create_app` (not
  per-request) since the manifest is static for the process lifetime — matches existing
  patterns like `app_version`.
- No changes were made to `EVENTS_INDEXES`, `_STAGE_COLS`, or any other existing detail keys;
  backward compatibility with pre-Task-13 `detail` consumers (e.g. `/stats`) is preserved.
