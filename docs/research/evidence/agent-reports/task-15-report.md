# Task 15 Report: Hijyen borcu üçlüsü (Plan 2 T1 devri)

## Status: DONE

## Commit
`67b01be` — `test: shipped-manifest, TLS-context and multichunk regression nets`
(branch `feat/p0-retrieval-correctness`, staged files only: `tests/corpus/test_manifest.py`,
`tests/test_cli.py`, `pyproject.toml`, `src/belge_gozu/corpus/manifest.py`)

## What was done

1. **`tests/corpus/test_manifest.py`**: added `test_shipped_manifest_parses_and_ids_unique`
   verbatim from the old plan (loads `data/manifest/v0_manifest.csv`, asserts unique
   `doc_id`s and `len(rows) >= 56`). Confirmed the shipped CSV has 57 lines (56 data
   rows + header) — assertion holds with margin of exactly 0 (`>= 56` passes at 56).

2. **TLS test — adapted per the brief's preferred path**: instead of the old plan's
   `test_http_client_keeps_tls_verification` which poked
   `client._transport._pool._ssl_context`, I wrote it against `build_ssl_context()`
   directly (the brief explicitly said this is the preferred path; "do NOT poke
   `_transport` internals"):
   ```python
   def test_http_client_keeps_tls_verification():
       import ssl
       from belge_gozu.corpus.manifest import build_ssl_context
       ctx = build_ssl_context()
       assert isinstance(ctx, ssl.SSLContext)
       assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
   ```

3. **Refactor of `src/belge_gozu/corpus/manifest.py`**: extracted the existing
   `ssl.create_default_context(cafile=certifi.where())` +
   `load_verify_locations(cadata=_GEOTRUST_TLS_RSA_CA_G1)` block out of
   `build_http_client` into a new `build_ssl_context() -> ssl.SSLContext` function.
   `build_http_client` now calls `httpx.Client(verify=build_ssl_context(), **kwargs)`.
   Behavior identical — TLS verification stays ON, same certifi CA bundle + embedded
   GeoTrust intermediate cert, same defaults (`verify_mode=CERT_REQUIRED`,
   `check_hostname=True` from `ssl.create_default_context`).

4. **`tests/test_cli.py`**: added `test_fake_build_multichunk_alignment` verbatim from
   the old plan (3 docs × 7 pages = 21 pages via existing `make_pdf` helper; renders,
   fake-builds with `BG_DATA_DIR`/`BG_INDEX_DIR` env overrides, asserts
   `idx.page_ids == meta.page_id.tolist()` and that `idx.page_tokens(pos)` for
   `pos in (0, 10, 20)` bit-for-bit matches an independent `FakeEncoder` re-encode of
   the same page image via `np.testing.assert_array_equal`). One deliberate deviation:
   dropped the old plan's unused `import json` line inside the test body — nothing in
   the test uses `json`, and `ruff` (select `F`) would flag it as F401.

5. **`pyproject.toml`**: added the SWIG `filterwarnings` block to
   `[tool.pytest.ini_options]` exactly as specified:
   ```toml
   filterwarnings = [
     "ignore:builtin type Swig:DeprecationWarning",
     "ignore:builtin type swigvarlink:DeprecationWarning",
   ]
   ```
   Confirmed (via `git stash` A/B before/after) that these SWIG `DeprecationWarning`s
   are real and present in the current suite (pymupdf-emitted, 6 occurrences pre-fix),
   so this filter is not merely defensive here — it actively cleans output.

## Adaptation beyond the verbatim plan text (flagging per instructions)

Adding the `filterwarnings` ini key had a side effect unrelated to SWIG: it appears to
reset/invalidate Python's warnings-registry dedup cache process-wide (empirically
verified via `git stash` A/B, reproduced twice each side), which unmasked a
pre-existing `StarletteDeprecationWarning` ("Using `httpx` with `starlette.testclient`
is deprecated; install `httpx2` instead.", raised from
`fastapi/testclient.py` when `tests/app/test_api.py` / `tests/telemetry/test_prom.py`
import `TestClient`). This warning is a subclass of `UserWarning`, not
`DeprecationWarning`, so it was never covered by the plan's SWIG filter and was not
mentioned in the brief. It is pre-existing library-version tech debt, unrelated to
Task 15's scope, and not something introduced by these changes — before this task's
`filterwarnings` block existed, it was silently swallowed (never shown) due to
pytest's warning-capture timing; adding any `filterwarnings` ini entry exposes it.

Since the task's explicit gate is "warning count 0" after `uv run pytest -q -m "not
slow"`, and `pyproject.toml` is within the allowed file set, I added one more targeted
ignore entry (with an inline Turkish comment explaining why) rather than leaving a
residual unrelated warning in the gate output:
```toml
"ignore:Using `httpx` with `starlette.testclient` is deprecated:UserWarning",
```
This does not change any runtime behavior — it only suppresses a display-time warning
from a third-party test-only shim. A real fix (bumping `fastapi`/`starlette`/`httpx`
versions or migrating off the deprecated TestClient path) is out of scope for this
task and should be tracked separately if desired.

No other adaptations were needed — `index build --fake`'s manifest.json side effect
and the CLI's `batch_size=1` did not affect the alignment assertions, as anticipated
in the brief's adaptation notes.

## Test results

- Targeted RED (before refactor): `uv run pytest tests/corpus/test_manifest.py
  tests/test_cli.py -v` → 14 passed, 1 failed
  (`test_http_client_keeps_tls_verification` — `ImportError: cannot import name
  'build_ssl_context'`, as expected pre-refactor).
- Targeted GREEN (after refactor): same command → **15 passed**, 0 warnings.
- Full regression: `uv run pytest -q -m "not slow"` → **136 passed, 1 deselected, 0
  warnings**.
- Lint: `make lint` (`ruff check` + `ruff format --check` + `pyright`) → **all clean**,
  "0 errors, 0 warnings, 0 informations".

## Concerns

None blocking. Only note is the Starlette-warning adaptation documented above —
flagging for awareness in case a reviewer wants the real dependency fix tracked as a
separate follow-up item instead of (or in addition to) the filter suppression.
