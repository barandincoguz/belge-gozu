# Re-review: eec067e — fix round for int8-switch-review.md (b790f6c..eec067e)

**Scope:** commit `eec067e` only (16 files, +286/−70). `1a92610` (docs/evidence commit) excluded
per instructions — it touches `docs/research/**` and the threshold-transfer artifact, none of
which are under re-review.
**Method:** `git show eec067e -- <file>` per file (not the pre-rendered diff, which the reader
paginated at 25k tokens) + full read of `src/belge_gozu/index/loader.py` and `src/belge_gozu/app/main.py:170-214`.
**Verification run:** `uv run pytest -q -m "not slow"` → 233 passed (matches report, = 229 + 4 new
tests: `test_non_int8_index_warns_about_threshold_portability`,
`test_int8_index_does_not_warn_about_threshold_portability`, `test_threshold_guard_runs_before_index_load`,
`test_manifest_data_disagreement_raises_readable_error`). `make lint` → ruff + ruff format + pyright
all clean. Live `curl localhost:7860/healthz` → `{"status":"ok","pages":4222,"threshold":0.58,"top_k":5,"index":{"quantization":"int8",...}}`.

**Verdict: ALL 14 RESOLVED.** No new defects found; no scope creep (all 16 changed files map
1:1 to a finding; test count delta is exactly accounted for).

## Per-finding table

| # | Status | Evidence |
|---|---|---|
| I1 | RESOLVED | (a) README:79-87 adds explicit non-portability warning + measured 1-bit numbers (min 0.4676/med 0.4953/max 0.6133, 1/43, ≈0.47 equiv. point) and repeats it in v0-limitations (README:215-216); (b) `config.py` gets a TAŞINABİLİRLİK paragraph with the same numbers; (c) `create_app` (`main.py:186-202`) logs `logger.warning(...)` when `quantization != THRESHOLD_CALIBRATED_ON` (constant `="int8"` in `config.py`), does not block startup; `tests/app/test_compat.py` has both directions — `test_non_int8_index_warns_about_threshold_portability` (asserts "eşik taşınabilirlik" + "sign-1bit" in caplog) and `test_int8_index_does_not_warn_about_threshold_portability` (asserts no such message). Confirmed live: report's server log grep shows 0 matches while serving int8. |
| I2 | RESOLVED | `scripts/d1_augmentation.py`: `from belge_gozu.index.loader import load_scorable_index` (line 47) replaces `PackedIndex.load` import; `idx = load_scorable_index(args.index)` at line 175 (was `PackedIndex.load`); `--index` help text no longer claims "packed"; retriever swapped to generic `ExhaustiveRetriever`. `grep -n "load_scorable_index\|PackedIndex.load" scripts/d1_augmentation.py` → only the loader is referenced now. Report's end-to-end repro (Int8Index, 4222 pages, band 0.999) is consistent with the code. |
| I3 | RESOLVED | README:111-114 and `config.py` comment rewritten from "question for question" to "by count... though not question-for-question: two rows swap sides (`c306` now clears it, `c211` no longer does)" — matches the reviewer's own cross-join numbers exactly (c306 1-bit 59.85→int8 0.5965; c211 1-bit 61.78→int8 0.5767). |
| M1 | RESOLVED | `bench/harness.py` imports `EMBED_DIM` from `index.chunking` and uses `score / (n_q * EMBED_DIM)` (was `/128`) at the one production-adjacent site the review flagged. |
| M2 | RESOLVED | `index.html`: `maxV = Math.max(...hits.map(h=>h.score), THRESHOLD, 0.01) * 1.08` — denominator now strictly positive (0.01 floor); `pct = (v) => Math.min(100, Math.max(0, v/maxV*100))` — result clamped to [0,100]. Traced the reviewer's exact counter-example (scores [-0.3,-0.9], threshold -0.2): maxV=0.01×1.08=0.0108, all pct→0.00%, no 416% overflow. |
| M3 | RESOLVED | Not a functional defect per original review ("dead breadth", not a bug) — `cli.py:169-174` adds a comment explaining the guard is intentionally written over the variable (not hardcoded to sign-1bit) so it stays correct if the f16-without-`--out` restriction is ever relaxed. Appropriately conservative, matches the review's own framing. |
| M4 | RESOLVED | Both `tests/index/test_store.py::test_score_all_respects_chunk_tokens` and `tests/retrieval/test_exhaustive.py::test_chunk_boundaries_do_not_change_scores` now monkeypatch `store_mod.chunk_bounds` with a spy that records the received `chunk_tokens` and assert on the recorded sequence (`[10, 1, 10]` and `[ExhaustiveBinaryRetriever.CHUNK_TOKENS, 16]` respectively) instead of only asserting score equality. Inspected without mutating: if `core.py:112`'s `chunk_tokens=self.CHUNK_TOKENS` forwarding were deleted, the store-level default path would feed `chunk_bounds` a different value than the recorded instance override (16), so `seen` would no longer match the asserted list and the test would fail — the invariant-that-holds-anyway problem the finding described is closed. Report additionally claims this was mutation-tested (forwarding actually removed, test went red); diff-level inspection supports that this would be the outcome. |
| M5 | RESOLVED | `build_retriever(s, encoder)` — `model_name`/`model_revision` params removed, derived internally via `s.retriever_model` / `getattr(encoder, "model_revision", None)`. `grep -rn "build_retriever("` shows exactly 3 hits: the def, `main.py:168`, and `tests/retrieval/test_semantic_retrieval_eval.py:68` — both call sites updated, no orphaned old-signature caller anywhere in the repo. |
| M6 | RESOLVED | The `if s.min_score_threshold > 1.5: raise IndexCompatibilityError(...)` block moved from after `build_retriever` to immediately after `s = settings or get_settings()`, before encoder creation and before `build_retriever`. New test `test_threshold_guard_runs_before_index_load` passes a nonexistent `index_dir` + `min_score_threshold=60.0` and asserts the *threshold* error (not a file/loader error) fires — proves guard precedence. |
| M7 | RESOLVED | `observability/grafana/provisioning/dashboards/belge-gozu.json:307`: `sum by (le)` → `sum by (le, quantization)`. Confirmed via grep; JSON parses (file otherwise well-formed, no syntax breakage from the edit). |
| M8 | RESOLVED | `index/chunking.py` docstring corrected: no longer claims monkeypatching the module global works universally; now explicitly states `PackedIndex`/`Int8Index` resolve `CHUNK_TOKENS` from an import-time `ClassVar` (unaffected by monkeypatching the global) and that test overrides go through the instance (`idx.CHUNK_TOKENS = ...`). Documentation-only, matches the pre-existing (unchanged) behavior — accurately describes it now instead of overclaiming. |
| M9 | RESOLVED | `retrieval/core.py` adds `from __future__ import annotations` and moves `from belge_gozu.index.loader import ScorableIndex` under `if TYPE_CHECKING:`. Verified `ScorableIndex` has no runtime use in the file — its only two occurrences are a parameter annotation (`index: ScorableIndex`) and a docstring, both safe under deferred-annotation evaluation. Import-chain edge is gone. |
| M10 | RESOLVED | `index/loader.py` adds `_SIGNATURE_FILE` map (sign-1bit→tokens.npy, int8→codes.npy, float16→embs.npy) and checks `(dir / signature).exists()` before dispatch, raising a Turkish `IndexCompatibilityError` naming the missing file instead of a bare `FileNotFoundError`. New test `test_manifest_data_disagreement_raises_readable_error` builds exactly the scenario the finding described (manifest says int8, only tokens.npy on disk) and asserts the error matches "codes.npy". |
| M11 | RESOLVED | `grep -n "474\|476\|919\|918" README.md` → all 5 occurrences now read 476 (int8) / 918 (float16), matching `data/bench/results/latency-by-representation.json`'s `index_mb: 476` / `918` exactly (`du`'s 473/919 MiB vs the JSON's MB figures is a unit artifact, not part of the finding — the finding was about the two *docs* disagreeing with each other, which they no longer do). |

## New findings

None that rise above trivial. One cosmetic observation, not filed as a defect:

- **Trivial — M2's `0.01` floor slightly under-scales the near-zero-positive-score case.** `index.html`'s new `Math.max(...scores, THRESHOLD, 0.01)` uses a fixed `0.01` floor. If the real max score (or threshold) were a small positive number below `0.01/1.08 ≈ 0.0093` (not the all-negative case M2 was about), the top bar would render at a smaller percentage than the pre-existing `*1.08` headroom convention implies (e.g. true max 0.005 → ~46% instead of ~93%). Unreachable with real ColPali output — observed top-1 scores run 0.57-0.86 (per `int8-threshold-transfer.json`) — and it doesn't violate either invariant the review demanded (denominator positive, output clamped to [0,100]), so this is a documentation-worthy corner case at most, not a regression.

## Scope-creep check

All 16 files touched by `eec067e` map to exactly one of the 14 findings (README.md → I1a/I3/M11;
`config.py` → I1b/I3; `app/main.py` → I1c/M5/M6; `scripts/d1_augmentation.py` → I2;
`index.html` → M2; `cli.py` → M3; `bench/harness.py` → M1; `index/chunking.py` → M8;
`index/loader.py` → I1-docstring/M10; `retrieval/core.py` → I1-docstring/M9; the 5 test files →
their corresponding finding's new/updated test; grafana json → M7). No unrelated refactors, no
drive-by changes. Test count delta (229→233 = +4) is fully accounted for by the four new tests
listed above.
