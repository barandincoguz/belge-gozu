# Review: b790f6c — int8 üretim geçişi, tek skor ölçeği

**Reviewed:** `dd4f251..b790f6c` on `feat/p0-retrieval-correctness` (30 files, +1111/−302)
**Reviewer state:** read-only. Ran `uv run pytest -q -m "not slow"` (229 passed),
`make lint` (clean), `uv run pytest tests/index/test_loader.py tests/retrieval/test_exhaustive.py
tests/app/test_compat.py -q` (28 passed), and `BG_DEVICE=mps uv run pytest -m slow -q`
(**4 passed, 1 xfailed** — independently reproduced).
**Verdict:** **FIX REQUIRED** — 0 Critical, 3 Important, 10 Minor.
The shipped path (int8 + threshold 0.58) is **mathematically correct and verified**;
all 19 spec items are present. The Important findings are one silent-failure hole in a
*documented* alternative configuration, one script whose new docstring contradicts its
code, and one falsifiable README claim. None block the default production path.

---

## 1. Scale-math audit (review priority #1) — CLEAN

Every score-producing/consuming path divides by `n_q` exactly once and by `EMBED_DIM`
exactly once **only where the raw kernel is Hamming-based**. Traced end to end:

| Path | Divisions | Verdict |
|---|---|---|
| `PackedIndex.score_all` (`index/store.py:73-105`) | `/ n_q` then `/ EMBED_DIM` — once each | ✅ |
| `Int8Index.score_all` (`index/quantize.py:86-108`) | `/ n_q` only (dot-product already ~[-1,1]) | ✅ |
| `FloatIndex.score_all` (`index/float_store.py:58-83`) | `/ n_q` only | ✅ |
| `ExhaustiveRetriever.score_all` → `search_embedding` → `search` (`retrieval/core.py:107-140`) | pure pass-through, **no** extra division | ✅ |
| `TwoStageRetriever.search_embedding` | RAW (documented, `core.py:38`) | ✅ |
| `TwoStageRetriever.search` (`core.py:71`) | `score / (n_q * EMBED_DIM)` — once | ✅ |
| `ExhaustiveDiagnosticAdapter` (`bench/harness.py:83,93`) | none (consumes normalized) | ✅ |
| `TwoStageDiagnosticAdapter` (`bench/harness.py:159`) | `score / (n_q * 128)` — identical expression to production | ✅ |
| `binary_maxsim` (`core.py:20-27`) | RAW by contract; only caller is `TwoStageRetriever.search_embedding` | ✅ |
| `native_float_scores` → `FloatIndex.score_all` | thin delegation, same math | ✅ |
| `bench oracle` CLI (`cli.py:587-636`) | scores used only through `argsort`/`rank_of`/`recall_at_k` — the new `/128` is a monotone transform, recalls unchanged | ✅ |
| `scripts/d1_augmentation.py:209` | ranks only — monotone-safe | ✅ (but see I2) |
| `/search`, `/ask`, `AskService` (`answer/base.py:47`), events `top_score`/`margin`, `/metrics` | consume `PageHit.score` (normalized) | ✅ |
| UI (`index.html:334,348,351,386`) | normalized, 2 decimals | ✅ |

**No off-by-one division anywhere.** The equality lock
`tests/bench/test_harness.py::test_two_stage_adapter_matches_production_score`
(pre-existing, untouched) would have gone red had the harness and production diverged —
it passes, so the two-stage diagnostic scale is genuinely pinned to production.

Two supporting facts I verified rather than assumed:
- `as_u64` moved into the per-call hot path does **not** copy the mmap: measured
  `np.shares_memory(memmap, np.ascontiguousarray(memmap)) == True`. No per-query
  58 MB copy, no perf regression from the kernel move.
- `chunk_bounds` is still read at call time for the `chunk_tokens=None` path, so
  `FloatIndex` keeps the monkeypatchable behaviour its docstring advertises.

---

## 2. Findings

### CRITICAL — none

### IMPORTANT

---

**I1 — The unified scale does NOT make `min_score_threshold` representation-neutral, but four
places claim it does; the README hands users the exact config that silently breaks the service.**

*Where:* `README.md:80-82`; `src/belge_gozu/config.py:56-78` (rationale comment);
`src/belge_gozu/retrieval/core.py:70, 87` ("eşik … tek ve ortaktır" / "tek ve ortak kalır");
`src/belge_gozu/index/loader.py:8-11`; guard at `src/belge_gozu/app/main.py:162-167`.

Normalizing all three representations into ~[-1,1] makes them *numerically comparable*,
not *distributionally identical*. Measured, on the same canary set and the same
train-compat format:

| representation | top-1 min | median | max | clears 0.58 |
|---|---|---|---|---|
| int8 (`data/bench/results/int8-threshold-transfer.json`) | 0.5767 | 0.6250 | 0.7450 | **42 / 43** |
| 1-bit (`data/bench/results/a2-traincompat-1bit-exhaustive.json`, raw ÷128) | 0.4676 | 0.4953 | 0.6133 | **1 / 43** |

(1-bit figures computed from the pre-T14 artifact's `stages[0].top_scores[0]`, which are
exactly the raw values this commit now divides by 128.)

*Failure scenario:* an operator follows README:80-82 — "1-bit remains available as the
ablation / disk-budget option (`data/index-traincompat-1bit`, 58 MB) via `BG_INDEX_DIR`" —
and runs `BG_INDEX_DIR=data/index-traincompat-1bit belge-gozu serve`. The loader happily
returns a `PackedIndex`, `check_compatibility` passes (it never compares quantization),
`/healthz` reports `"status":"ok"` with `quantization:"sign-1bit"`, the `>1.5` guard does
not fire — and **42 of 43 answerable canary questions now fall below the threshold**. The
product answers essentially nothing, with no error anywhere. The same applies to the
`BG_RETRIEVAL_PIPELINE=two-stage` ablation, whose scores now land in the same 1-bit band.

This is the *same* silent-abstain failure mode the commit's own `>1.5` guard was written
to prevent — the guard only covers the "threshold left far too high" direction, not
"threshold measured on a different representation". The canary ratchet already solves
exactly this problem for ranks (`canary_expectations.json` is keyed by `"quantization"`
and `test_long_query_rank_ratchet` asserts the loaded manifest matches); the threshold
deserves the same treatment.

*Suggested fix (pick one):* (a) a per-quantization threshold default/table, (b) a startup
warning or `IndexCompatibilityError` when the loaded `manifest.quantization` differs from
the representation the threshold was transferred on, or at minimum (c) a one-sentence
warning at README:80-82 and in the `config.py` comment that `BG_INDEX_DIR` to 1-bit
requires re-deriving the threshold (~0.47 for the same operating point).

---

**I2 — `scripts/d1_augmentation.py` docstring was rewritten to claim the new loader; the code
was not changed. The command the docstring prints now crashes.**

*Where:* `scripts/d1_augmentation.py:13-17` and `:23` (new text) vs `:172`, `:190`, `:197`,
`:113` (unchanged code/help).

The docstring now says the index is loaded "`belge_gozu.index.loader.load_scorable_index`
ile … manifest'teki `quantization`'a göre packed/int8/float16 — yani üretimin skorladığı
temsil neyse D1 de onu ölçer", and the usage line reads
`uv run python scripts/d1_augmentation.py --index data/index-traincompat-int8`.
Line 172 still reads `idx = PackedIndex.load(args.index)`, and `--index`'s help still says
"skorlanacak paketli (PackedIndex) indeks dizini".

*Failure scenario:* running the documented command dies before the model is even imported:
`FileNotFoundError: [Errno 2] No such file or directory:
'data/index-traincompat-int8/tokens.npy'` (reproduced). D1 can therefore only measure the
representation production no longer serves — the precise drift the rest of the commit closes.
The implementer's report marks this row ✅ ("`cli.py bench_run` + `scripts/d1_augmentation.py`
| ✅ | Aynı loader"), so the report is inaccurate here.

*Fix:* three lines — import `load_scorable_index`, call it at :172, retitle the `--index` help.

---

**I3 — README's "question for question" equivalence claim is measurably false.**

*Where:* `README.md:107-110` ("it reproduces the same operating point, question for
question"); echoed in `src/belge_gozu/config.py:64-66` ("int8 ölçeğinde 0.58 tam olarak
aynı bölmeyi verir").

The *count* is preserved (42/43 answerable, 4/5 unanswerable under both), but the
*partition* is not. Cross-joining the two artifacts on `question_id`:

- `c306`: 1-bit raw **59.85** → abstained at 60.0; int8 **0.5965** → now passes.
- `c211`: 1-bit raw **61.78** → passed at 60.0; int8 **0.5767** → now abstains.

Two questions swap sides. Every other claim in the same paragraph checks out
(42/43 + 4/5, min/median/max, 0.5767 stays out / 0.5860+ passes, artifact path), so this is
a single overstated clause, not a pattern — but it is exactly the kind of claim this project
sells itself on. *Fix:* "reproduces the same operating point — the same number of questions
on each side, though not question-for-question (two rows swap)".

---

### MINOR

**M1 — `bench/harness.py:159` hardcodes `128` on the normalization path.**
Spec item 16 single-sources `EMBED_DIM`; this is the one production-adjacent site that
still carries the literal, and it is precisely the coupled constant (`score / (n_q * 128)`).
If `EMBED_DIM` ever changes, the harness silently reports a different scale than production
and the equality test would be the only thing that catches it. Tests using the literal
(`test_exhaustive.py:19`, `test_store.py:110`, `test_core.py:82,98`) are defensible as
independent constants; the harness is not a test.

**M2 — Chart clamp is incomplete for the all-negative case.**
`src/belge_gozu/app/static/index.html:330-335`: `maxV = Math.max(...scores, THRESHOLD) * 1.08`.
If every hit score *and* the threshold are negative, `maxV < 0` and `v / maxV` flips sign,
so `Math.max(0, …)` never engages. Example: scores `[-0.3, -0.9]`, `THRESHOLD = -0.2` →
`maxV = -0.216` → bar width `416.67%`, threshold line at `92.59%`. Low probability with real
ColPali output (needs all five pages' MaxSim < 0), but the whole point of the clamp was
"scores can now be negative". *Fix:* `const denom = maxV > 0 ? maxV : 1e-9;` or clamp the
result into `[0, 100]`.

**M3 — Half the `index build` quantization guard is unreachable.**
`cli.py:149-150` already rejects `--precision f16` without `--out`, so the guard at
`cli.py:163-179` can only ever run with `quantization == sign-1bit`. Not a defect (the case
it must catch — packed build over an int8 prod dir — is covered, tested, and correctly
allows fresh dirs since `read_manifest` returns `None` for a missing file and `--out` skips
the guard entirely), just dead breadth.

**M4 — Two chunk tests assert an invariance that would hold even if the plumbing were removed.**
`tests/index/test_store.py:116-127` and `tests/retrieval/test_exhaustive.py:52-56` assert
that different `chunk_tokens` give identical scores. Since chunking genuinely cannot change
the result, these pass unchanged if `ExhaustiveRetriever.score_all` stopped forwarding
`chunk_tokens=self.CHUNK_TOKENS` altogether. The forwarding *is* correct (`core.py:112`), but
nothing locks it. A spy on `chunk_bounds` (assert the resolved value) would make the
override lock real.

**M5 — `build_retriever` still requires `model_name`/`model_revision` from every caller.**
`app/main.py:60-68`: both are derivable inside (`s.retriever_model`, `getattr(encoder,
"model_revision", None)`), and both call sites (`main.py:147-152`,
`tests/retrieval/test_semantic_canary.py:60-65`) pass the identical two lines — i.e. the
duplication the extraction was meant to kill is partly still there. Worse, a caller can pass
a `model_name` inconsistent with `s`, silently weakening the compat check the function exists
to run.

**M6 — Guard ordering: the cheapest check runs last.**
`create_app` builds the encoder (possibly loading the VLM) and the index, then checks
`min_score_threshold > 1.5` (`main.py:162`). A misconfigured threshold costs a full model +
index load before failing. Moving the check above `build_retriever` is free.

**M7 — The Grafana panel doesn't realize the catalog's remedy.**
`docs/research/metrics-catalog.md` says pre/post-transition samples are distinguished by the
new `quantization` label, but
`observability/grafana/provisioning/dashboards/belge-gozu.json:307` queries
`sum by (le) (rate(bg_retrieval_top_score_bucket[5m]))` — the label is aggregated away, so
the heatmap still mixes old 0–128 samples with new [-1,1] ones. Nothing breaks (the panel has
no hardcoded axis), but the documented mitigation isn't in effect. Also note adding a label
means the two histograms now emit no series at all until the first scored request.

**M8 — Monkeypatch asymmetry across the three `score_all`s.**
`index/chunking.py`'s docstring advertises that tests can monkeypatch the module global, but
`PackedIndex.CHUNK_TOKENS` / `Int8Index.CHUNK_TOKENS` are `ClassVar`s bound at import, so only
`FloatIndex` (which passes `None` through) honours a patched global. Pre-existing for
`Int8Index`; the kernel move extends it to `PackedIndex`.

**M9 — `retrieval/core.py:8` imports `ScorableIndex` at runtime for an annotation only.**
This makes `retrieval` import `index.loader` → `index.quantize` → `index.float_store` on every
import. `if TYPE_CHECKING:` would keep the layering claim without the runtime edge.

**M10 — Manifest/data disagreement escapes the loader's friendly error.**
`load_scorable_index` dispatches on the manifest but never checks that the named
representation's files exist: a dir whose manifest says `int8` but holds only `tokens.npy`
raises a bare `FileNotFoundError: codes.npy` instead of the loader's Turkish
`IndexCompatibilityError`. Cheap to add alongside the existing `tokens.npy` legacy probe.

**M11 — Cosmetic size drift between two docs quoting one measurement.**
README:77 says int8 is 474 MB; `data/bench/results/latency-by-representation.json` says 476;
`du` says 473 MiB. Same for float16 (919 vs 918). Harmless, but they're quoting the same run.

---

## 3. Spec compliance checklist (19 items)

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | All scorers return per-query-token avg in ≈[-1,1]; packed kernel moved + `/EMBED_DIM`; `Int8Index` signature; `FloatIndex` moved to `index/float_store.py`, oracle re-exports, `native_float_scores` delegates | ✅ | `store.py:73-105`, `quantize.py:86-108`, `float_store.py:58-83`, `oracle.py:11-31`; scale table §1 |
| 2 | `ExhaustiveRetriever` + `ExhaustiveBinaryRetriever` alias, generic over `.page_ids`/`.score_all`, retriever `CHUNK_TOKENS` override preserved | ✅ | `core.py:81-145`; override forwarded at `:112`, exercised by `test_exhaustive.py:52` |
| 3 | `TwoStageRetriever.search` → `raw/(n_q*128)`; `search_embedding` stays RAW | ✅ | `core.py:38, 71`; `test_core.py:82,98` |
| 4 | Harness `TwoStageDiagnosticAdapter` matches production | ✅ | `harness.py:159`; locked by pre-existing `test_two_stage_adapter_matches_production_score` |
| 5 | New `index/loader.py` with manifest dispatch, legacy `tokens.npy` fallback, clear Turkish error | ✅ | `loader.py:28-73`; 5 tests in `tests/index/test_loader.py` (all three representations loaded from real files) |
| 6 | `app/main.py` uses loader; two-stage requires `PackedIndex`; threshold guard mentioning "binary ölçeği" | ✅ | `main.py:79, 115-120, 162-167`; `test_compat.py:240,255,273` |
| 7 | Defaults `index-traincompat-int8` / `0.58`; mechanical-transfer wording; artifact path | ✅ | `config.py:31, 78, 56-77`; artifact verified: 42/43 + 4/5, min/med/max exact. **Caveat I3** on the "aynı bölme" phrasing |
| 8 | `/healthz` + `top_k` + `index{quantization, revision}` | ✅ | `main.py:294-307`; exact-payload lock `test_api.py:48-63` |
| 9 | UI: 2-decimal scores, negative clamps, footer, footnote, `THRESHOLD` 0.58, "ilk 5" from `top_k` | ✅ (partial, M2) | `index.html:285-296, 330-357, 386`; no stale binary constants (only CSS/comment `60`s remain) |
| 10 | `SCORE_BUCKETS`/`MARGIN_BUCKETS` + `quantization` label from `detail.retrieval.quantization`, default `unknown` | ✅ | `prom.py:17-24, 58-85, 132-137`; label source pre-exists in `build_event` (`main.py:280`); fallback path tested; `exhaustive_maxsim` stage fallback tests still green |
| 11 | `colpali-engine==0.3.18` pin | ✅ | `pyproject.toml:24`, `uv.lock` single-line specifier change |
| 12 | `bench run` uses loader + generic retriever, two-stage branch guarded | ✅ | `cli.py:431-458` |
| 13 | `index build` no-`--out` guard extended to quantization | ✅ | `cli.py:163-179`; fresh dir OK (`read_manifest→None`), `--out` still allowed; `test_cli.py:138` |
| 14 | `Quantization` StrEnum in `index/manifest.py` with `float16`; loader dispatches on it | ✅ | `manifest.py:79-91`, `loader.py:28-33`; `derive --quant float16` rejected (`cli.py:256-261`, `test_cli.py:162`) |
| 15 | `build_retriever` extracted; canary fixture uses it | ✅ | `main.py:60-122`; `test_semantic_canary.py:44-66` (copy deleted) — M5 is a shape nit only |
| 16 | `EMBED_DIM`/`INT8_MAX` single-sourced; `_as_u64` shape guard | ✅ (one leak, M1) | `chunking.py:16-25`, `store.py:24-35` + `test_store.py:130` |
| 17 | `canary_expectations.json` carries `"quantization":"int8"` + ratchet 664; slow test asserts manifest match | ✅ | `canary_expectations.json`, `test_semantic_canary.py:131-139`; 664 equals `c001` gold rank in the artifact; slow suite re-run green |
| 18 | Test updates (exhaustive /128, test_core /(n_q*128), harness equality, loader tests, threshold guard, two-stage-on-int8, healthz exact, config defaults, abstain xfail rewritten and still FAILING) | ✅ | 229 passed / 4 slow passed + **1 xfailed** (no XPASS) reproduced locally; nothing loosened |
| 19 | README: quickstart int8, quantization paragraph, score scale + 0.58, mermaid neutral, n_tokens 3,759,994, stale-v0 keeps 60.0 | ✅ (with I3) | README:44-47, 61-89, 101-110, 144-152, 199-215, 236-239; n_tokens matches all three manifests; R@5 0.233, R@20 0.233 vs 0.302, 1.08/0.24/0.08 s all match artifacts. Mermaid S2 no longer names a binary algorithm (it does name "the int8 index", which is accurate and matches the IDX node) |

**19 / 19 present.** Items 7 and 19 carry the I3 wording caveat; item 9 carries M2.

---

## 4. Behavioural-regression sweep (review priority #2)

- **`CHUNK_TOKENS` override** — forwarded per call (`core.py:112`); `test_chunk_boundaries_do_not_change_scores` still green. Side effect worth knowing: because the retriever *always* passes a value, an index-level `idx.CHUNK_TOKENS` override is now ignored when scoring *through* the retriever. Harmless today (chunking never changes results); noted for whoever tunes memory later.
- **Legacy fallback** — `loader.py:56-60` returns `PackedIndex` for a manifest-less dir with `tokens.npy`; `create_app` still rejects it via `check_compatibility(None, …)` unless `allow_index_mismatch`. Tested at loader level; `test_mismatch_override` covers the override path. `manifest is None` degrades cleanly everywhere downstream (`healthz` nulls, prom → `"unknown"`).
- **Injected-encoder compat check** — byte-for-byte the same arguments as before the extraction (`getattr(encoder, "query_format", resolved_query_format).format_id`, `doc_prompt_sha256` fallback to the config-resolved value). `test_create_app_checks_injected_encoder_against_configured_format` green.
- **Threshold guard boundaries** — `-1e9` fixtures unaffected; `> 1.5` leaves the entire legitimate band (and 0.5× headroom above the [-1,1] max) open; env overrides below 1.5 unimpeded. **Except** the I1 direction, which the guard cannot see.
- **`bench oracle` arms** — the binary arm's new `/128` is monotone; all outputs are ranks/recalls, so historical comparability of that report is preserved.
- **`StageRecord.stage == "exhaustive-binary"`** — deliberately left as a schema label (implementer documented this). Agreed; it is a string key, not a scale.

## 5. Quality verdict

**Good.** The refactor moves the kernel to the right layer (`index` owns scoring, `retrieval`
owns orchestration), kills the `index → bench` layering inversion, replaces three duplicated
literals with named constants, and turns two previously-silent failure modes (wrong-scale
threshold, two-stage on a non-packed index) into fail-fast errors with actionable Turkish
messages. Comment density is high but earns its keep: nearly every comment states *why*, with
a measurement or a ruling id behind it. New tests are real assertions on real files (three
representations written to `tmp_path` and re-loaded), and the strict-xfail was rewritten with
new numbers without being weakened — verified by reproducing the red.

The gap between this commit's engineering and its prose is where the fixes are needed: three
of four claims about the unified scale overreach what the data supports (I1, I3), and one
docstring was updated as though the code under it had been (I2). Dead code from the kernel
move: none found (`retriever.tokens/offsets` have no remaining users; `oracle.py`'s re-exports
are intentional and documented). Duplication between loader and `build_retriever`: none —
`build_retriever` reads the manifest off the loaded index rather than re-parsing, and says so.

**Recommended before merge:** I2 (3-line code fix), I3 (one clause), and at minimum the
documentation half of I1. M1, M2 and M6 are each a one-liner and worth folding in.
