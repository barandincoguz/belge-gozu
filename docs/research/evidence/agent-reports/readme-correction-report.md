# README correction report — padding-impact and quantization claims

Commit: `4044a08` on `feat/p0-retrieval-correctness`
Tests/lint: `uv run pytest -q -m "not slow"` → 157 passed, 4 deselected. `make lint` → ruff check/format + pyright all clean.

## Correction 1 — padding-row claim (v0 limitations bullet, ~line 176)

**Before:**

> different set of pages, not a faster version of the same ranking. Separately, the
> index was found to contain 3,960 all-zero padding-token rows across 15 pages (now
> rejected at build time), and the encoder's retrieval training data is English-only,
> which is the likely reason Turkish paraphrase queries score weaker than queries that
> name the statute explicitly. A hybrid text+visual retrieval path is the planned fix
> (P1); a full retrieval benchmark is in progress to quantify where things stand today.

**After:**

> different set of pages, not a faster version of the same ranking. Separately, the
> index was found to contain 3,960 all-zero padding-token rows across 15 pages — a
> real correctness defect (padding embeddings collapsing to an all-zero bit vector and
> scoring as if it were a genuine token). This is now fixed and locked:
> `PackedIndex.build` rejects all-zero rows at build time, and the rebuilt index has
> 0 such rows and 3,776,882 tokens — exactly 3,960 fewer than the old index's
> 3,780,842. It was **not**, however, one of the causes of today's poor retrieval
> numbers: measured on the retrieval_eval benchmark (draft, pending human verification), an
> index rebuilt in the same format without the padding rows produced byte-identical
> Recall at every k and an identical top-20 list for 42 of the 43 questions versus the
> old, padded index. Independently, the encoder's retrieval training data is
> English-only, which is the likely reason Turkish paraphrase queries score weaker
> than queries that name the statute explicitly. A hybrid text+visual retrieval path
> is the planned fix (P1); a full retrieval benchmark is in progress to quantify where
> things stand today.

Rationale: keeps the defect and the fix (now correctness-locked at build time via
`PackedIndex.build`), but explicitly states the measured retrieval impact was zero
(byte-identical Recall at every k; 42/43 identical top-20), so it no longer reads as
a cause of poor retrieval quality. Numbers match
`docs/research/findings/2026-08-27-p0-baseline.md` §2(d): 3,780,842 → 3,776,882
(Δ = 3,960), 0 all-zero rows post-fix, byte-identical Recall table, 42/43 top-20 match.

## Correction 2 — stale "C1/C2 not run yet" claim (quantization/approximation passage, ~line 70)

**Before:**

> that binary code space*, but relative to native float ColPali scoring it is an
> approximation; the size of that loss is meant to be measured by the P0 plan's
> quantization ablation (C1/C2: float16 oracle vs. int8 vs. 1-bit), which has not been
> run yet — no results exist for it as of this writing. The resulting score is itself

**After:**

> that binary code space*, but relative to native float ColPali scoring it is an
> approximation, and the P0 plan's quantization ablation (C1/C2: float16 oracle vs.
> int8 vs. 1-bit) has now been run on the 48-question retrieval_eval benchmark (43 answerable;
> the retrieval_eval set is **draft, pending human verification**, so treat these numbers as
> provisional), in the production query/document format: **int8 matches float16 exactly
> at every k** (Recall@1/5/20/50/200 all identical); **1-bit loses 7.0 points of
> Recall@20** relative to float16 (0.233 vs. 0.302). 1-bit is also **slower, not
> faster**: scoring all 4,222 pages against a 40-token query takes 1.08 s at 1-bit vs.
> 0.24 s at int8 vs. 0.08 s at float16 (CPU, idle machine), because int8/float16 hit a
> BLAS matmul path while the 1-bit path builds large temporaries for the popcount
> reduction. Index size is the one axis where 1-bit still wins (58 MB vs. 474 MB for
> int8 vs. 919 MB for float16). The production index is still 1-bit purely because the
> retriever currently only accepts the packed 1-bit index — wiring int8 into serving is
> deferred to P1, so 1-bit is what ships, not what won the ablation. Full tables:
> [`docs/research/findings/2026-08-27-p0-baseline.md`](docs/research/findings/2026-08-27-p0-baseline.md)
> and
> [`docs/research/findings/2026-08-27-p0-gate.md`](docs/research/findings/2026-08-27-p0-gate.md).
> Separately, the single biggest P0 result to date: switching the document encoder to
> the checkpoint's training-time prompt (instead of the format `colpali-engine==0.3.18`
> emits by default) raised float16 Recall@5 from 0.093 to 0.233 on that same draft
> retrieval_eval set. The resulting score is itself

Rationale: replaces the stale "not run yet" sentence with the measured C1/C2 outcome
(int8 == float16 exactly; 1-bit loses 7.0 pts Recall@20; 1-bit is slower not faster;
disk sizes), states plainly that 1-bit ships only because serving can't yet consume
int8 (P1 work), links both findings docs, and adds the single biggest P0 result
(training-time document prompt: Recall@5 0.093 → 0.233) with the draft/pending-
verification caveat attached to the retrieval_eval set. Numbers match
`docs/research/findings/2026-08-27-p0-baseline.md` §2(b)/§2(c)/§3.2/§3.3 (Seri B
latency: 1.0827 s / 0.2435 s / 0.0789 s; disk: 58 MB / 474 MB / 919 MB, A2/production
family; Recall@20 0.232558 vs 0.302326; Recall@5 0.093023 → 0.232558).

## Concerns

None blocking. Noted for awareness: `git status` shows `docs/research/evidence/p0-decision-log.md`
modified and several `data/bench/results/*.json` files untracked — these are from the
concurrent benchmark sweep / other agents, not touched by this task. Only `README.md`
was staged and committed.
