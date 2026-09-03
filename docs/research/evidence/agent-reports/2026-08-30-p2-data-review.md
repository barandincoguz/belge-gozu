# P2 data review — commit 5d3bd83 (`abstention_eval_v1` cevaplanamaz benchmark)

Scope: code + schema + split correctness only. Question/label quality is out of
scope (separate checker agent). Read-only review; ran pytest/lint/python for
verification, cleaned up all scratch artifacts afterward (`git status` confirmed
clean at end — only pre-existing untracked `.agents/`, `skills-lock.json`).

## Verdict: **APPROVE**

No CRITICAL or HIGH findings. The determinism risk named in the review brief as
a potential CRITICAL ("python hash() is per-process-salted for str") does **not**
apply here — the implementation uses `hashlib.sha256` exclusively, confirmed
deterministic across different `PYTHONHASHSEED` values in fresh processes. Every
quantitative claim in the implementer's report and README was independently
recomputed (not re-run/trusted) and matched exactly. 3 LOW findings + 2
informational notes below; none block merge.

---

## Findings (severity-ranked)

### LOW

**L1 — `scripts/validate_abstention_eval.py:325`: unhandled crash on out-of-repo `--bench` path.**
```python
print(f"abstention_eval doğrulama — {bench_path.relative_to(REPO)}")
```
`Path.relative_to` raises `ValueError` if `bench_path` isn't under `REPO`. Repro:
copied the jsonl to a scratch dir outside the repo and ran
`--bench <scratch>/broken_abstention_eval.jsonl` → unhandled traceback instead of a clean
error. Confirmed this still exits `1` (Python's default uncaught-exception
behavior), so the exit-code *contract* holds — it's an ugly failure mode, not a
false pass. Default invocation and every documented usage (README §5, the
implementer's report) always pass a repo-relative path, so this never fires in
practice. One-line fix if desired: use `os.path.relpath` or wrap in try/except.

**L2 — no isolated unit test exercises `load_bench`/`load_splits` with a plain `str`.**
`tests/bench/test_dataset.py` always calls these with `tmp_path / "x.jsonl"`
(a `Path`), never a bare string, so the `Path | str` fix itself has no dedicated
unit test. The fix is real and works — I verified manually:
`load_bench('data/bench/abstention_eval_v1.jsonl', only_verified=False)` → 300,
default (`only_verified=True`) → 200 — and it's exercised by the exact command
documented in `abstention_eval_v1.README.md` §5. Not blocking, just a coverage gap.

**L3 — `assign_split` has no production consumer yet (forward-looking note, not a defect).**
Grepped `src/` and `scripts/`: only `scripts/validate_abstention_eval.py` calls
`assign_split`. `src/belge_gozu/bench/harness.py` and `cli.py` don't reference it
or `abstention_eval` at all. Expected for a data-only commit. Flagging so whoever wires
`abstention_eval_v1` into the real dev/test metrics harness uses `assign_split`
(law-grouped) rather than the older `question_split` — the latter falls back to
plain qid-hashing for any row without `gold_doc_ids`, which would silently lose
the law-grouping guarantee against corpus leakage for `korpus-disi`/`eksik-kanit`.

### Informational (not defects)

**I1 — test count is 489/6-deselected (495 total), not the brief's "494".**
Fully reconciled, not a regression: checked out parent commit `d89cee7` into an
isolated `git worktree` and ran `-m "not slow" -q` there → **443 passed**
(baseline). This commit adds exactly **46** new tests: 31 in the new
`tests/test_validate_abstention_eval.py` + 15 new in `tests/bench/test_dataset.py`
(collect-only: 33 now vs. 18 before). 443 + 46 = 489 exactly. No test was
removed, skipped, or xfailed. The brief's "494" was off by one against reality;
worktree removed after the check, repo left clean.

**I2 — `eksik-kanit` verification_note grep-counts are self-reported, not machine-checked.**
`validate_abstention_eval.py` never opens `data/research/page_texts.parquet` to verify the
grep counts quoted in `verification_note` (e.g. `'kasko sigortası değeri'=0`) —
it only checks that the note is non-empty and that `_subject_doc` is a real
corpus id. This is a real gap in mechanical coverage, but it is **honestly
disclosed**: the README marks this slice `draft`/`model-cross-check` precisely
because of it ("Grep yokluğu, 'sayfa görüntüsünde de yok' demek DEĞİLDİR").
Disclosed limitation, not a hidden one — no action needed.

---

## What was independently verified (not just re-reading the implementer's report)

1. **Determinism (the named CRITICAL risk).** `_hash50`, `assign_split`,
   `derive_test_docs` (`src/belge_gozu/bench/dataset.py:164-166, 175-219`;
   `scripts/validate_abstention_eval.py:271-298`) use `hashlib.sha256(key.encode()).hexdigest()`
   exclusively. Grepped for bare `hash(` — zero hits outside `hashlib.`. Ran
   `scripts/validate_abstention_eval.py` twice in fresh processes with
   `PYTHONHASHSEED=1` and `PYTHONHASHSEED=99999`: **byte-identical output**,
   both exit 0. The feared "process-salted str hash → non-reproducible split"
   failure mode does not occur — confirmed, not just inspected.

2. **`assign_split` vs. `splits_v1.json`'s `unanswerable_rule`.** Line-by-line
   code match confirmed (korpus-disi → `sha256("anchor:<law>")`; eksik-kanit →
   subject-doc lookup in `test_docs`; answerable → `gold_doc_ids[0]` lookup;
   everything else → `sha256("qid:<id>")`). Independently recomputed by loading
   `abstention_eval_v1.jsonl` + `retrieval_eval_v1.jsonl` raw and calling `assign_split` directly
   (bypassing `validate_abstention_eval.py`'s own `derive_test_docs`): got
   **dev=154 unanswerable/26 answerable, test=151/17** — exact match to the
   report and to `splits_v1.json`'s `retrieval_eval_answerable_split`.

3. **`Path | str` fix** (`dataset.py:114-116, 143-144`). Ran the exact
   README-documented command; got 300 (`only_verified=False`) and 200
   (default `only_verified=True`, korpus-disi only).

4. **Corpus derivation.** Loaded `data/state.json` + `data/manifest/v0_manifest.csv`
   directly: 56 doc ids, sets match exactly, 50 `k<number>` law docs + 6 `rg*`
   docs. Matches the validator's reported "56 belge, 50 kanun numarası."

5. **Validator run.** `uv run python scripts/validate_abstention_eval.py` → output
   byte-for-byte identical to the implementer's report §1, `TEMİZ`, exit 0.

6. **Exit-code=1 path, for real.** Injected a genuine violation (anchored a
   question at `k4857`/İş Kanunu, which *is* in the corpus) into a scratch copy
   placed temporarily under `data/bench/` (removed immediately after; `git
   status` confirmed clean). Reran → got both expected error lines
   (`ÇAPA KORPUSTA` + the Jaccard name-overlap hit) and real exit code `1`.
   This is also how L1 was found (a second repro with an out-of-repo path).

7. **Split file structure.** 22 `test_docs`, 34 `dev_docs`, disjoint, union
   covers all 56 corpus docs (recomputed independently, not just trusting the
   validator's internal check). RG docs in `test_docs`: exactly
   `{rg1935a, rg1945a}` = 2. `seed` field present, `"belge-gozu-splits-v1"`.

8. **Pinned-doc math.** Independently tallied `retrieval_eval_v1.jsonl`'s answerable
   rows by primary gold doc: 13 distinct docs; the 4 pinned
   (`k6098, k5237, k6698, rg1935a`) sum to **exactly 17**; the other 9 sum to
   **exactly 26**. Matches `pinned_rationale` and the README's 26/17 target
   precisely.

9. **15 jsonl rows sampled across all 3 slices** (u001/u050/u100/u150/u200,
   u210/u225/u240/u250/u260, u265/u275/u285/u295/u300): structurally sound —
   correct empty `gold_*`/`reference_answer`, correct per-slice
   `unanswerable_reason`, correct verification triple per slice, correct extra
   underscore fields only where expected (`_anchor_law`+`_anchor_name` for
   korpus-disi, `_subject_doc` for eksik-kanit, none for anlamsiz-ood). u250
   ("ignore prior instructions...") and u210 (lorem-ipsum mixed with Turkish
   legal words) are intentional injection/gibberish probes per the README, not
   defects.

10. **Anchor spot-checks (5, exceeding the requested 3).** `5901, 6769, 6197,
    5718, 4250` confirmed absent as `k<number>` from `data/state.json`. Also
    checked 5 `eksik-kanit` `_subject_doc` values (`k2918, k6102, k5510, k5846,
    k5651`) are present. Full-population (not sampled) checks: 0 exact-duplicate
    questions, 0 empty `verification_note`, 0 anchor-law/name inconsistencies
    across rows sharing an `_anchor_law`, id sequence exactly `u001..u300` in
    order.

11. **README numeric claims**, recomputed from raw jsonl independent of the
    validator's print statements: slice counts 200/60/40 ✓; korpus-disi
    `query_style` 79/86/35 ✓; 117 distinct anchor laws ✓; overall difficulty
    49/161/90 ✓; source_type 300/300 `ajan-taslak` ✓; eksik-kanit 40 distinct
    subject docs split exactly 20/20 ✓. Confirmed the historically-caught
    near-dup (u108/u109) is now genuinely distinct in committed data (three
    different questions about 6136 sayılı Kanun: possession vs. carrying permit
    vs. unlicensed-carrying penalty). Residual-risk disclosures in the README
    are accurate, not oversold, and match exactly what the code does/doesn't
    check.

12. **Tests + lint.** `tests/bench -q` → 82 passed. `tests/test_validate_abstention_eval.py -q`
    → 31 passed (every one of the validator's 9 check categories has at least
    one deliberately-broken-input test — "TEMİZ" is a meaningful signal).
    `-m "not slow" -q` → 489 passed, 6 deselected (see I1). `make lint` → ruff
    check clean, ruff format clean (105 files), pyright 0 errors/warnings.

---

## Checklist

- [x] `source_type` += `ajan-taslak` — done, tested, distinct from `ajan-taslak-insan-onayli`
- [x] `verification_kind` += `mechanical:manifest-absence` — done, tested as a third distinct kind
- [x] `assign_split` determinism — sha256-based, not the salted builtin; cross-`PYTHONHASHSEED` confirmed
- [x] `assign_split` rule matches `splits_v1.json`'s documented rule — code + independent recompute
- [x] `Path()` fix — verified working end-to-end
- [x] corpus-set derivation (56 docs / 50 laws) — independently reproduced
- [x] anchor-absence check (number AND name-token) — tested both directions, crash-test confirms it fires
- [x] mention check — tested both directions
- [x] dup detection — tested, historical catch (u108/u109) confirmed resolved in data
- [x] exit codes — 0 clean (confirmed), 1 on violation (confirmed via injected repro)
- [x] 22 test docs, ≥2 RG — confirmed (rg1935a, rg1945a)
- [x] seed recorded — `belge-gozu-splits-v1`
- [x] composition claims (dev 154+26 / test 151+17) — independently recomputed, exact match
- [x] 15 jsonl rows sampled — structurally clean
- [x] ≥3 korpus-dışı anchors spot-checked vs. `data/state.json` — did 5, all confirmed absent
- [x] `tests/bench -q` — 82 passed
- [x] full `-m "not slow" -q` — 489 passed / 6 deselected (495 total; brief said 494 — reconciled, see I1)
- [x] `make lint` — clean
- [x] README honesty — claims match reality; residual-risk disclosure accurate
