# Task 12 Report: Kuantizasyon ablasyonu C1/C2 (`index/quantize.py`)

Scope executed: Steps 1-4 only (code + tests + full regression + commit). Steps
5-6 (real ablation runs, decision, config change) are explicitly NOT run — no
index builds performed, no real indexes derived.

## Status: DONE (code-only scope)

## Commit
`5b63b52` — `feat(index): int8 and sign-1bit derivation from float16 master`
(+ Co-Authored-By / Claude-Session trailer per instructions).

Branch: `feat/p0-retrieval-correctness` (already checked out, not switched).

## Files changed (staged by explicit path, no `git add -A`/`.`)
- `src/belge_gozu/index/quantize.py` — new: `derive_packed`, `Int8Index`
  (`derive`, `page_tokens`, `score_all`, `save`, `load`).
- `tests/index/test_quantize.py` — new: 5 tests (3 from brief + 2 controller-
  specified extras).
- `src/belge_gozu/cli.py` — new `Quantization` StrEnum + `index derive`
  command.
- `src/belge_gozu/bench/oracle.py` — **shared helper extracted**: `_chunk_bounds`
  gained an optional `chunk_tokens: int = CHUNK_TOKENS` parameter so
  `Int8Index.score_all` can import and reuse it with its own (instance-
  overridable) `CHUNK_TOKENS` instead of duplicating the chunking loop.
  Default value is unchanged, so `native_float_scores`'s public behavior
  (the only caller, called with no second argument) is untouched. This file
  is listed here per controller ruling R5 ("... ONLY if you extract a shared
  chunk-bounds helper — list it in your report").

Untracked pre-existing files `.agents/` and `skills-lock.json` were left
untouched/unstaged — unrelated to this task.

## TDD evidence

### RED
```
$ uv run pytest tests/index/test_quantize.py -v
...
E   ModuleNotFoundError: No module named 'belge_gozu.index.quantize'
=========================== short test summary info ============================
ERROR tests/index/test_quantize.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.06s ===============================
```

Wrote the brief's three tests verbatim, plus two controller-specified extras
before writing any implementation:
- `test_int8_score_all_multichunk_matches_single_chunk` — sets
  `i8_multi.CHUNK_TOKENS = 10` on the instance (mirrors the existing
  `retrieval/core.py` / `tests/retrieval/test_exhaustive.py` override pattern)
  and asserts the multi-chunk run matches the single-chunk run.
- `test_derive_packed_carries_manifest_with_sign_1bit_quantization` — attaches
  a manifest (`make_manifest` from `tests/index/test_manifest.py`,
  `quantization="float16"`) to a `FloatIndex`, asserts `derive_packed(...)`
  returns `manifest.quantization == "sign-1bit"` and all other fields
  untouched.

First run of the multi-chunk test against the finished implementation failed
with an exact-equality assertion:
```
E   Mismatched elements: 2 / 6 (33.3%)
E   Max absolute difference among violations: 1.90734863e-06
E   Max relative difference among violations: 1.18452761e-07
```
This is expected float32 matmul reassociation noise from different chunk
boundaries (same property `native_float_scores` would have under a changed
`CHUNK_TOKENS`), not a bug — relaxed the test to
`np.testing.assert_allclose(multi, single, rtol=1e-5, atol=1e-5)`.

### GREEN
```
$ uv run pytest tests/index/test_quantize.py -v
tests/index/test_quantize.py::test_derive_packed_matches_direct_binarize PASSED
tests/index/test_quantize.py::test_int8_scores_close_to_float PASSED
tests/index/test_quantize.py::test_int8_roundtrip PASSED
tests/index/test_quantize.py::test_int8_score_all_multichunk_matches_single_chunk PASSED
tests/index/test_quantize.py::test_derive_packed_carries_manifest_with_sign_1bit_quantization PASSED
============================== 5 passed in 0.02s ===============================
```

### Full regression (as specified)
```
$ uv run pytest tests/index tests/bench -v
... 53 passed in ~20-22s

$ uv run pytest -q -m "not slow"
144 passed, 1 deselected in 1.27-1.43s

$ make lint
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
76 files already formatted
0 errors, 0 warnings, 0 informations
```
(One `ruff format` fixup was needed on `cli.py` — an f-string join for a
`BadParameter` message the formatter preferred as one line; applied via
`uv run ruff format src/belge_gozu/cli.py`, then full regression re-run clean.)

## Implementation summary

- `derive_packed(findex) -> PackedIndex`: reads every page's f16 tokens via
  `findex.page_tokens(i)`, upcasts to float32, and calls `PackedIndex.build`
  (which internally applies `binarize_pack` per page and rebuilds
  `page_vecs`). Manifest carried via
  `findex.manifest.model_copy(update={"quantization": "sign-1bit"})` when a
  manifest exists, else `None`. Relies on `PackedIndex.build`'s existing
  all-zero-row rejection (T2 mask policy guarantees f16 master has no padding
  rows already, per brief).
- `Int8Index` (plain `@dataclass`, mutable — needed by the CLI's manifest-
  ordering trick, see below):
  - `codes: np.ndarray` int8 `(total_tokens, 128)`, `scales: np.ndarray`
    float32 `(total_tokens,)`, `offsets`, `page_ids`, `manifest`.
  - `CHUNK_TOKENS: ClassVar[int] = CHUNK_TOKENS` (imported from
    `belge_gozu.bench.oracle`) — a class attribute overridable per instance,
    mirroring the existing `ExhaustiveBinaryRetriever.CHUNK_TOKENS` /
    `_chunk_bounds(self)` pattern in `retrieval/core.py`.
  - `derive(findex)`: `scale = max(|x_t|, axis=-1) / 127`, floored at `1e-8`
    to avoid a zero divisor on a (hypothetical) all-zero token row;
    `codes = clip(round(x / scale), -127, 127).astype(int8)` (clip is a
    defensive no-op in the normal case since `scale` is derived from the same
    row's max — kept for float rounding safety).
  - `score_all(q_emb)`: same page-aligned chunked-`_chunk_bounds` +
    `q @ chunk.T` + `np.maximum.reduceat` + sum-over-query-tokens + `/n_q`
    pattern as `oracle.native_float_scores`, with the chunk itself
    dequantized to float32 first
    (`codes[chunk].astype(float32) * scales[chunk, None]`). Uses
    `_chunk_bounds` imported from `oracle.py` (not copied) with
    `self.CHUNK_TOKENS`.
  - `save`/`load`: `codes.npy`, `scales.npy`, `offsets.npy`, `page_ids.json`,
    optional `manifest.json` (mmap-able load, same convention as
    `FloatIndex`/`PackedIndex`).
- CLI `index derive --from DIR --quant {sign-1bit,int8} --out DIR`:
  - Errors clearly (`typer.BadParameter`) if `--from` isn't a float index
    (missing `embs.npy`), has no `manifest.json`, or its manifest's
    `quantization != "float16"`.
  - Derives via `derive_packed` or `Int8Index.derive`.
  - R3 ordering: temporarily clears `derived.manifest` before `derived.save(out)`
    (so `save()`'s internal `if self.manifest is not None: write_manifest(...)`
    doesn't fire with a stale checksum), copies `meta.parquet` from the source
    dir, then constructs the final manifest via
    `source_manifest.model_copy(update={quantization, corpus_checksum(out)
    [recomputed], n_pages, n_tokens, built_at, git_commit})` and writes it
    last with `write_manifest`.
  - Echoes page/token counts on success.
  - `Quantization` StrEnum (`sign-1bit`, `int8`) added next to `Precision` /
    `QueryFormatChoice`.
  - Manually smoke-tested end-to-end via `CliRunner` (not part of the
    committed test suite — no CLI test for `derive` was in the ruled test
    list): built a synthetic 4-page `FloatIndex` with manifest, ran
    `index derive --quant sign-1bit` and `--quant int8`, confirmed correct
    output files (`tokens.npy`/`page_vecs.npy` vs `codes.npy`/`scales.npy`),
    correct `manifest.json` `quantization` field, and a freshly recomputed
    `corpus_checksum` (identical across the two output dirs since built from
    the same source `page_ids.json`+`meta.parquet`, as expected). Also
    confirmed the "not a float index" error path fires with exit code 2 and a
    clear message.

## Measured int8-vs-float deviation (from the test's synthetic data)

Using `test_int8_scores_close_to_float`'s fixture (6 pages, 5 tokens/page,
128-dim `standard_normal`, seed 2), scoring query `embs[1]` (`native_float_scores`
vs `Int8Index.score_all`):

```
float: [  8.6030304  129.40775146  16.85588226  12.650634    12.58597565  11.72861633]
int8 : [  8.6028595  129.39567871  16.80743713  12.58701935  12.52952271  11.73875427]
max abs diff: 0.0636   (test tolerance atol=0.5)
max rel diff: 0.00503  (test tolerance rtol=0.05)
```

So on this synthetic Gaussian fixture, int8 deviates from true float MaxSim by
well under 1% relative / ~0.06 absolute — an order of magnitude inside the
brief's asserted tolerances. This is a sanity signal only; it is **not** a
substitute for the real C1/C2 ablation (Step 5), which needs the actual
production embedding distribution and the retrieval_eval bench's Recall@20 metric to
make the P0-gate decision.

## Self-review

- `derive_packed`/`Int8Index.derive` both use the "carry findex.manifest via
  `model_copy`, else `None`" rule exactly as ruled; verified by the two new
  tests.
- `_chunk_bounds` extraction is minimal and behavior-preserving: added one
  optional parameter with the original global as its default; the only
  existing call site (`native_float_scores`) still calls it with zero extra
  args, so `oracle.py`'s public behavior (`FloatIndex`, `native_float_scores`,
  `rank_of`) is unchanged. Confirmed via `tests/bench/test_oracle.py` (all 3
  pass) and the full `tests/bench` suite.
- Considered whether `Int8Index` needed to be frozen/immutable; kept it a
  plain mutable dataclass because the CLI's R3-ordering trick
  (`derived.manifest = None` before `save()`, restore-equivalent via a local
  variable afterward) needs mutation — matches `PackedIndex`/`FloatIndex`,
  which are also plain mutable dataclasses.
- Did not add a dedicated CLI test for `index derive` — the controller's test
  list for this task enumerates exactly 5 tests, all in
  `tests/index/test_quantize.py`, and none targets the CLI. Compensated with
  a manual `CliRunner` smoke test (see above) as part of self-review, not
  committed.
- Ran `uv run pyright src/belge_gozu/index/quantize.py src/belge_gozu/cli.py
  src/belge_gozu/bench/oracle.py` explicitly in addition to `make lint`'s
  project-wide pyright pass — 0 errors/warnings on all three files.

## Concerns

1. **No CLI-level automated test for `index derive`.** Deliberate per the
   controller's enumerated test list, but flagging in case Step 5/6 (owned by
   the controller) wants one added before or during the real ablation runs —
   e.g. a small fixture-index round-trip through the CLI would catch
   regressions in the R3 manifest-ordering logic that the direct
   `derive_packed`/`Int8Index.derive` unit tests can't see (they don't
   exercise `corpus_checksum` recomputation or the `meta.parquet` copy step).
2. **int8 deviation numbers above are synthetic**, not representative of the
   real ColSmol-500M embedding distribution/scale. The actual P0-gate
   decision in Step 5 needs a real retrieval_eval run with Recall@20 — this report's
   numbers are only a sanity check that the implementation is numerically
   correct, not evidence for the C1/C2 decision itself.
3. `np.clip(..., -127, 127)` in `Int8Index.derive` is defensive (guards
   float-rounding edge cases at exactly `±127`); in the normal case
   `scale = max|x|/127` makes clipping a no-op, so it should never actually
   truncate a value on real data — flagging only because it's unverified
   against production-scale embeddings.

---

## Addendum: review R1 fix-up (commit `e5b1283`)

Coordinator review verdict: Approved with 3 Important + 3 Minor follow-ups,
to land before the C1/C2 ablation runs. Operating constraint for this pass:
another agent's MPS index build was running concurrently — no index build,
no writes under `data/`, modest memory. All verification below used tiny
synthetic fixtures in the scratchpad temp dir (5-6 pages, 6-8 tokens/page),
never `data/`, and was cleaned up (`shutil.rmtree`) after each run. Also
rebased implicitly onto HEAD `2710779` (another agent's `fix(index):
self-verifying format check + doc-prompt guard in bench oracle` commit,
already an ancestor of my working tree — no merge conflicts, no git surgery
needed since there was no local divergence).

### Commit
`e5b1283` — `fix(index): per-token-scale test, chunked int8 derive, int8
oracle arm (review R1)`. Files touched (staged by explicit path): `src/
belge_gozu/bench/oracle.py`, `src/belge_gozu/cli.py`, `src/belge_gozu/index/
quantize.py`, `tests/index/test_quantize.py` — same four files as Task 12's
original commit, no scope creep.

### IMPORTANT 1 — per-token-scale contract was untested (FIXED)
`test_int8_scores_close_to_float` gained two assertions (kept the original
argmax/closeness ones):
- `assert (np.abs(i8.codes).max(axis=1) == 127).all()` — every token row
  saturates to ±127 against its *own* scale; a global per-matrix scale or a
  coarser (e.g. 3-bit) quantizer would not satisfy this for every row.
- `np.testing.assert_allclose` — actually a plain `assert np.all(...)` bound:
  `|dequant - src| <= scale/2 + 1e-4` element-wise, the standard round-to-
  nearest quantization error bound; a truncating quantizer would violate it
  for roughly half the elements.

Ran alone first to confirm both assertions hold against the existing
(pre-chunking) implementation, then again after the IMPORTANT-2 rewrite —
passed both times (chunking doesn't change the per-row math, see below).

### IMPORTANT 2 — `Int8Index.derive` peak memory (FIXED)
Old code: `np.asarray(findex.embs, dtype=np.float32)` (full-corpus copy),
`np.abs(embs)` (another full copy), `embs / scale[:, None]` (another), then
`round`/`clip` (more temporaries) — reviewer's estimate: 5.5-6.5 GB peak at
4222 pages x ~871 tokens.

Rewrote `derive` to loop over `_chunk_bounds(offsets, chunk_tokens or
cls.CHUNK_TOKENS)` (page-aligned, same helper `score_all` uses), pre-
allocating the *output* `codes`/`scales` arrays once, and doing `divide`/
`round`/`clip` **in-place** (`out=chunk`) on a single per-chunk float32
buffer that's reused across chunks (each of size `chunk_tokens*128*4` bytes
— 256 MB at the default `CHUNK_TOKENS=500_000`, vs. ~1.88 GB per full-corpus
float32 copy before). `derive(findex, chunk_tokens: int | None = None)` now
accepts an explicit override (defaults to `cls.CHUNK_TOKENS`) so tests can
force multiple chunks on a tiny fixture.

Added `test_int8_derive_chunked_matches_single_shot`: derives the same
6-page/8-token-per-page fixture once with default chunking (single chunk,
tiny corpus) and once with `chunk_tokens=10` (forces ~1-2 pages/chunk),
asserts `codes`/`scales` are **bit-identical** (`assert_array_equal`, not
`allclose`) — justified because each token row's quantization is computed
independently of every other row (unlike `score_all`'s cross-row
`reduceat`/`sum`, which does reassociate floats across chunk boundaries and
needed a tolerance in the existing multichunk test). Passed on first run.

### IMPORTANT 3 — no C2 int8 scoring path in `bench oracle` (FIXED)
Added `--int8-index DIR` (optional, default `None`) to `bench_oracle`. When
given: loads `Int8Index.load(int8_index)`, requires its `manifest.json` to
exist, and applies the *same* cross-check already used for packed-vs-float
(`query_format.format_id` + `doc_prompt_sha256`, compared against the packed
index's manifest) — raises `typer.BadParameter` on any mismatch. Per
question, computes `int8_scores = i8.score_all(q_emb)` with the **same**
`q_emb` used for the other two arms, adds `int8_rank` (mirroring
`binary_rank`/`float_rank`, including the missing-gold-page skip pattern),
and accumulates `int8_recalls` at the same `ks = (1, 5, 20, 50, 200)`. Output
JSON gains `summary.int8` (same shape as `summary.binary`/`summary.float`),
`int8_index`, and `int8_manifest`, and the echo line appends ` int8=...`
when the flag is given.

When `--int8-index` is omitted, `i8` stays `None` and every `int8_*` branch
is skipped — verified this produces **byte-for-byte the same key set** as
before the change (see manual verification below): `report.keys()` and
`per_question[0].keys()` and `summary.keys()` are identical to a pre-change
run, so existing callers/consumers of this JSON (if any) are unaffected.

No automated test was added for this command (there was none before either
— `bench oracle`'s only production encoder is `ColSmolEncoder`, which needs
real model weights; that's presumably why the whole command has zero CLI
test coverage today, flag added or not). Compensated with a manual
end-to-end verification instead: built a synthetic 5-page float/packed/int8
index triad + a 1-question bench JSONL in the scratchpad temp dir,
monkeypatched `belge_gozu.index.encode.ColSmolEncoder` to a tiny
`FakeEncoder` subclass (deterministic, no model download), and ran the CLI
via `CliRunner` four ways:
1. No `--int8-index`: `exit=0`, output keys `['bench', 'float_index',
   'float_manifest', 'git_commit', 'missing_gold_pages', 'packed_index',
   'packed_manifest', 'per_question', 'run_id', 'summary']`,
   `per_question[0]` keys `['binary_rank', 'float_rank', 'question_id']`,
   `summary` keys `['binary', 'float', 'n']` — the original schema, untouched.
2. With `--int8-index`: `exit=0`, `n=1 recall@5 binary=1.000 float=1.000
   int8=1.000`; report gained exactly `int8_index`, `int8_manifest` at the
   top level, `int8_rank` per question, and `summary.int8 =
   {'1': 1.0, '5': 1.0, '20': 1.0, '50': 1.0, '200': 1.0}`.
3. Int8 index with a deliberately mismatched `doc_prompt_sha256` in its
   manifest: `exit=2`, clear error `"int8 indeksin query_format/doc_prompt'u
   packed/float ile uyuşmuyor: int8=cpe-0.3.18/zzzzzzzzzzzz
   packed=cpe-0.3.18/dddddddddddd"`.
4. (Minor 6, same session) `index derive --from <dir-without-meta.parquet>`:
   `exit=2`, `"--from dizininde meta.parquet yok: ..."`, confirmed no files
   were written to `--out` first. `index derive --from X --out X` (same
   dir): `exit=2`, `"--out --from ile aynı olamaz (f16 master'ın üstüne
   yazılır)"`.

### MINOR 4 — `_chunk_bounds` late binding (FIXED)
Signature changed from `chunk_tokens: int = CHUNK_TOKENS` (bound at
function-definition/import time) to `chunk_tokens: int | None = None`, with
`chunk_tokens = chunk_tokens or CHUNK_TOKENS` evaluated on every call. The
only zero-arg caller (`native_float_scores`) is unaffected; a test that
monkeypatches `belge_gozu.bench.oracle.CHUNK_TOKENS` and then calls
`_chunk_bounds(offsets)` (no second arg) will now see the patched value —
previously it would have kept using whatever `CHUNK_TOKENS` was at import
time.

### MINOR 6 — `index derive` guards (FIXED)
Added, before any array is written:
- `if not (from_dir / "meta.parquet").exists(): raise typer.BadParameter(...)`
- `if out.resolve() == from_dir.resolve(): raise typer.BadParameter(...)`
  (guards against `--out` silently overwriting the f16 master; `.resolve()`
  works even when `--out` doesn't exist yet, `strict=False` by default).

Both verified manually above (case 4).

### MINOR 7 — `test_int8_roundtrip` under-asserted (FIXED)
Added `assert_array_equal` for `scales` and `offsets`, and `assert
i82.page_ids == i8.page_ids`, alongside the existing `codes` check. A silent
`scales.npy` save/load bug (e.g. wrong dtype cast, wrong shape) would now
fail this test instead of only manifesting as corrupted downstream scores.

### Deferred (per controller instruction, NOT done)
Relocating `_chunk_bounds` out of the `bench` package — left in
`belge_gozu.bench.oracle`, still imported by `quantize.py` as before.

### Verification (as specified)
```
$ uv run pytest tests/index/test_quantize.py -v
6 passed in 0.05s   (was 5; added test_int8_derive_chunked_matches_single_shot)

$ uv run pytest tests/index tests/bench -v
54 passed in 21.55s

$ uv run pytest -q -m "not slow"
145 passed, 1 deselected in 1.22s

$ make lint
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
76 files already formatted
0 errors, 0 warnings, 0 informations
```
`uv run pyright src/belge_gozu/index/quantize.py src/belge_gozu/cli.py
src/belge_gozu/bench/oracle.py` also run explicitly (0 errors) to double
check the new `Int8Index | None` / `IndexManifest | None` narrowing in
`bench_oracle` type-checks cleanly (it does — used a dedicated
`i8_manifest: IndexManifest | None` local, re-checked `is not None` at each
use site rather than relying on cross-statement narrowing from the earlier
guard).

### New concerns from this pass
1. `bench oracle`'s `--int8-index` arm still has zero automated test
   coverage (consistent with the rest of that command, but worth a note for
   whoever eventually adds a `--fake`-style encoder seam to `bench oracle`
   the way `index build` already has one).
2. `Int8Index.derive`'s chunked rewrite assumes `findex.embs[t0:t1]` slicing
   on a memmapped array only pages in the requested chunk (standard numpy
   memmap behavior) — this was not verified against a real memmapped file
   under memory pressure in this pass (only against small in-memory arrays
   from `FloatIndex.build`); worth confirming during the actual C1/C2 run
   with a memory profiler if the concurrent MPS build makes memory tight.
