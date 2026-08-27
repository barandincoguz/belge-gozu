# Task 9 report: Float oracle (`bench/oracle.py` + `index build --precision f16`)

## Status: DONE

Commit: `93aedc8` on `feat/p0-retrieval-correctness` (HEAD unchanged since, still on this branch).

## Files changed

- `src/belge_gozu/bench/oracle.py` (new) — `FloatIndex`, `native_float_scores`, `rank_of`.
- `src/belge_gozu/cli.py` (modified) — `index build --precision {packed,f16} --query-format {cpe-0.3.18,train-compat-v1} --out PATH`, new `bench oracle` command, plus a small internal refactor of `index write-manifest` (see "Incidental refactor" below).
- `tests/bench/test_oracle.py` (new) — brief's 3 tests verbatim (with one cosmetic line-length fix for lint).
- `tests/test_cli.py` (modified) — added `test_index_build_manifest_passes_compat_check`.

## TDD evidence

1. **RED**: Wrote `tests/bench/test_oracle.py` with the brief's exact 3 tests before `oracle.py` existed.
   ```
   uv run pytest tests/bench/test_oracle.py -v
   ERROR collecting tests/bench/test_oracle.py — ModuleNotFoundError: No module named 'belge_gozu.bench.oracle'
   ```
2. **GREEN**: Implemented `src/belge_gozu/bench/oracle.py`, re-ran:
   ```
   uv run pytest tests/bench/test_oracle.py -v
   test_roundtrip PASSED
   test_self_match_top1 PASSED
   test_scores_are_true_maxsim PASSED
   3 passed
   ```
   R4 note: the `test_roundtrip` `atol=1e-3` passed as-is on the seed-0 fixture; no widening to 2e-3 was needed (pre-authorization not exercised).
3. Implemented CLI changes (`index build --precision`, `bench oracle`) and the compat-check CLI test, then ran full regression (below) — all green, no widening/deviation needed beyond one cosmetic line-length break in the test file (semantics unchanged, verified equal to brief's literal expression via intermediate `doc = ...` variable).

## Design decisions

- `FloatIndex` mirrors `PackedIndex`'s shape (`embs.npy`/`offsets.npy`/`page_ids.json` + optional manifest), stores tokens as float16, and rejects zero-token pages (same invariant `PackedIndex.build` enforces) so `offsets` stays strictly increasing for `np.maximum.reduceat`.
- `native_float_scores` reuses the exact page-aligned chunking pattern from `ExhaustiveBinaryRetriever.score_all` (chunk bound at 500k tokens, `np.maximum.reduceat` over per-page column segments, sum over query tokens, divide by `n_q`), swapping Hamming similarity for `q @ chunk.T` on float32-upcast chunks.
- `rank_of` uses `np.argsort(-scores, kind="stable")` position (not the raw `1 + (scores > scores[i]).sum()` tie-optimistic formula the brief describes as the naive alternative), per the brief's explicit correction.
- `index build`: added `Precision` and `QueryFormatChoice` `StrEnum`s (matching the existing `Pipeline` pattern), a `_QUERY_FORMATS` mapping to `CPE_0_3_18`/`TRAIN_COMPAT_V1`, and `--out` (`None` default preserves current behavior for `packed`; required — `typer.BadParameter` — for `f16`).
- R3 manifest ordering implemented literally: build index in memory → `index.save(out_dir)` → copy `meta.parquet` → `corpus_checksum(out_dir)` → construct `IndexManifest` → `write_manifest(out_dir, manifest)`. `PackedIndex.build`/`FloatIndex.build` are never given a `manifest=` kwarg from the CLI.
- Manifest fields set exactly per the ruling: `model_revision`/`doc_prompt_sha256` via `getattr(encoder, ..., "unknown")` (so `FakeEncoder` yields `"unknown"` for both); `engine_versions` via `importlib.metadata.version` with `PackageNotFoundError` → `"unknown"` fallback (factored into `_engine_versions()`); `query_format` = the enum-selected `QueryFormat` object; `quantization` = `"sign-1bit"` (packed) / `"float16"` (f16); `mask_policy="drop-padding"` (both paths); `git_commit()` reused from `belge_gozu.bench.harness` (public helper, as instructed) instead of re-shelling out.
- `ColSmolEncoder` is now constructed with `query_format=qf` in `index build` so `encode_query` renders consistently with what gets recorded in the manifest (documented as out of scope: no `visual_prompt_override` — that's Task 11, per the brief).
- `bench oracle`: loads both indexes, hard-fails with `typer.BadParameter` if either is missing a manifest or if `query_format.format_id` disagrees between the two, encodes each answerable question once with a `ColSmolEncoder` built from the packed index's manifest `query_format`, computes exhaustive-binary (`ExhaustiveBinaryRetriever.score_all`, constructed with `encoder=None` since only `score_all` is used) and native-float (`native_float_scores`) full rankings, records per-question `{question_id, binary_rank, float_rank}` (gold pages only, via `rank_of`), and Recall@{1,5,20,50,200} summary for both oracles, plus run künyesi (`git_commit()`, both manifests dumped via `model_dump()`). All heavy imports (`ColSmolEncoder`, retriever, dataset/metrics loaders) are lazy inside the command, matching the `bench run` pattern.

## Incidental refactor (scope note)

Removing the top-level `import subprocess` (no longer needed once `index_build` uses `belge_gozu.bench.harness.git_commit()`) would have broken `index write-manifest`'s own inline subprocess-based git-commit lookup and its inline `_pkg_version` closure. Rather than leave a second `import subprocess` around for one caller, I refactored `index_write_manifest` to reuse the new `git_commit()` and `_pkg_version()`/`_engine_versions()` helpers — same behavior (git short-hash or `"unknown"` fallback; same package-version lookups), confirmed by the existing `test_write_manifest_legacy_cli` still passing unchanged. This stayed inside `cli.py`, one of the four files this task is authorized to touch.

## Full regression

```
uv run pytest tests/bench tests/test_cli.py -v      # 31 passed
uv run pytest -q -m "not slow"                       # 128 passed, 1 deselected
make lint                                            # ruff check: all checks passed
                                                      # ruff format --check: 73 files already formatted
                                                      # pyright: 0 errors, 0 warnings, 0 informations
```

Manual CLI sanity checks:
- `uv run belge-gozu index build --help` shows `--precision <packed|f16>`, `--query-format <cpe-0.3.18|train-compat-v1>`, `--out <path>`.
- `uv run belge-gozu bench oracle --help` shows all four required options (`--bench`, `--packed-index`, `--float-index`, `--out`).
- `tests/test_cli.py::test_index_build_manifest_passes_compat_check` builds a tiny fake corpus, runs `index build --fake`, and asserts `read_manifest(index_dir) is not None` and `check_compatibility(manifest, model_name=Settings().retriever_model, model_revision=None, query_format_id="cpe-0.3.18", index_dir=index_dir) == []`.

## Self-review

- Confirmed `git status --short` before commit staged exactly the 4 authorized files (`src/belge_gozu/bench/oracle.py`, `src/belge_gozu/cli.py`, `tests/bench/test_oracle.py`, `tests/test_cli.py`); pre-existing untracked `.agents/` and `skills-lock.json` were left alone (not `git add -A`/`.`).
- Verified `PackedIndex.build`/`FloatIndex.build` are called with no `manifest=` argument anywhere in `cli.py`.
- Verified `bench_run`'s existing local `from belge_gozu.bench.harness import ... git_commit ...` (function-scoped) safely shadows the new module-level `git_commit` import without any lint/runtime issue (confirmed by ruff + full test pass).

## Concerns / follow-ups (none blocking)

- `bench oracle` was only exercised via `--help` and unit-level building blocks (`FloatIndex`, `native_float_scores`, `rank_of` all have direct tests); there is no end-to-end CLI test invoking `bench oracle` against real encoded data, since doing so would require a real or fake `ColSmolEncoder`-driven build of both a packed and float index plus a bench JSONL — out of scope for this task's TDD checklist (brief's Step 1 tests only cover `oracle.py`'s pure functions/classes) and consistent with `bench run`'s own CLI command, which similarly has no CLI-level test in this repo.
- The one deviation from the brief's literal test text: `test_scores_are_true_maxsim`'s `expected = (q @ ...).T).max(axis=1).sum() / q.shape[0]` one-liner was split into two lines (`doc = ...`; `expected = (q @ doc.T)...`) purely to satisfy `ruff`'s 100-char line limit — the computed value and behavior are identical.

## Review R1 fixes (commit `6e9864e`)

Files touched: `src/belge_gozu/cli.py`, `tests/bench/test_oracle.py` (only).

- **IMPORTANT 1 — oracle missing-gold-page crash**: `bench_oracle` previously called `rank_of(scores, page_ids, g)` unconditionally for every `q.gold_page_ids` entry; `rank_of`'s `page_ids.index(target)` raises a bare `ValueError` if the gold page isn't in that index, which would abort a long bench run on one bad row. Fixed by precomputing `known_binary_ids = set(idx.page_ids)` / `known_float_ids = set(findex.page_ids)` before the loop, and building `binary_rank`/`float_rank` per-gold-page with a membership check: pages present get a `rank_of` entry, pages absent get added to a `missing_gold_pages: set[str]` instead (checked independently against each index, since the two indexes could in principle diverge). The report JSON now includes a sorted `"missing_gold_pages"` list (mirroring `bench run`'s `EvalReport.missing_gold_pages` field/pattern), and the CLI echoes `missing_gold_pages=<n>` after the run summary when non-empty, matching `bench run`'s existing `if report.missing_gold_pages: typer.echo(...)` echo.
- **IMPORTANT 2 — `bench run` ignored the index manifest's query_format**: `ColSmolEncoder(s.retriever_model, s.device)` was constructed with no `query_format`, silently defaulting to `CPE_0_3_18` regardless of what the loaded index's manifest actually recorded — meaning a `train-compat-v1` index would get queries rendered in the wrong format. Fixed: `query_format = idx.manifest.query_format if idx.manifest else CPE_0_3_18` (computed after `idx = PackedIndex.load(...)`, before encoder construction), passed into `ColSmolEncoder(..., query_format=query_format)`. `CPE_0_3_18` was already imported at module scope in `cli.py`, so no new import was needed. This makes `bench run` consistent with `bench oracle`'s existing manifest-driven encoder construction.
- **MINOR — redundant local `git_commit` import in `bench_run`**: removed `git_commit` from the function-scoped `from belge_gozu.bench.harness import (...)` block; the module-level `from belge_gozu.bench.harness import git_commit` (added in the original Task 9 commit for `index_build`) already covers `bench_run`'s two `git_commit()` call sites.
- **MINOR — `test_roundtrip` atol justification comment**: added `# seed-0 fikstürde ölçülen max f16 hata 9.7e-4; R4 gereği gerekirse 2e-3'e genişletilebilir` directly above the `np.testing.assert_allclose(..., atol=1e-3)` call. Verified the 9.7e-4 figure empirically before adding it (`np.abs(stacked.astype(np.float16).astype(np.float32) - stacked).max()` on the exact seed-0, 5-page, 4-token fixture → `0.0009703636`), so the comment states a measured fact rather than an unverified guess.

### Verification after fixes

```
uv run pytest tests/bench tests/test_cli.py -v      # 31 passed
uv run pytest -q -m "not slow"                       # 128 passed, 1 deselected
make lint                                            # ruff check: all checks passed
                                                      # ruff format --check: 73 files already formatted
                                                      # pyright: 0 errors, 0 warnings, 0 informations
```

Staged and committed only `src/belge_gozu/cli.py` and `tests/bench/test_oracle.py` (confirmed via `git status --short` before commit; `.agents/` and `skills-lock.json` untouched) as `6e9864e fix(bench): oracle missing-gold guard + manifest-driven query format in bench run (review R1)`.

### Follow-up note

`missing_gold_pages` is tracked as a single set shared across both the binary and float rank checks (a page missing from either index adds to the same set) rather than two separate lists — this matches the coordinator's request for "a `missing_gold_pages` list" (singular) and `bench run`'s single-list precedent, at the minor cost of not distinguishing *which* index a given page was missing from in the top-level list (the per-question `binary_rank`/`float_rank` dicts still make this recoverable, since a missing page simply has no key in whichever oracle's dict it was absent from).
