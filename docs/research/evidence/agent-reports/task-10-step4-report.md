# Task 10 / Step 4 — Semantic retrieval_eval regression tests (report)

## Files created

- `tests/retrieval/test_semantic_retrieval_eval.py` — 3 slow tests, module-scoped
  `prod_retriever` fixture.
- `tests/retrieval/retrieval_regression_expectations.json` — seeded with
  `{"long_query_gold_rank_max": 1221, "_measured": "2026-08-27, data/index-traincompat-1bit, exhaustive, retrieval_eval c001"}`.

## Design decisions / deviations from the brief's literal code sample

1. **Fixture mirrors `app/main.py::create_app` exactly**: resolves
   `query_format`/`doc_prompt` via the shared `QUERY_FORMATS`/`DOC_PROMPTS`
   dicts and `QueryFormatChoice`/`DocPromptChoice` enums from
   `belge_gozu.index.manifest` (no re-declared literals), and branches on
   `s.retrieval_pipeline` the same way `create_app` does
   (`ExhaustiveBinaryRetriever` vs `TwoStageRetriever`). This was requested
   explicitly ("mirror exactly... import the shared lookup maps") and goes
   beyond the brief's hard-coded sample, which only ever built
   `ExhaustiveBinaryRetriever` with an unqualified `ColSmolEncoder(...)`.
2. **Numbers corrected per the task's "current measured reality"**:
   `long_query_gold_rank_max` seeded at **1221** (not the brief's stale
   1576), and docstrings/comments cite rank 4 / score 73.17 for the short
   query and rank 1221/4222 for the long query, dated 2026-08-26.
3. **`load_bench(..., only_verified=False)`** in
   `test_retrieval_eval_gold_pages_covered`, since human verification (Step 2) is
   still pending — matches the task instructions, not the brief's
   unqualified `load_bench(path)` call (which defaults to
   `only_verified=True` and would silently skip all 48 draft rows).
4. **Reused `belge_gozu.bench.oracle.rank_of`** instead of re-deriving the
   1-based stable-argsort rank computation inline (the brief's sample
   hand-rolled `np.argsort` + `np.nonzero` logic already exists as a tested
   utility — DRY, and it's what T8's oracle harness already relies on).
5. **Skip guards**: `pytest.skip` in the fixture if `settings.index_dir`
   doesn't exist; `pytest.skip` in `test_retrieval_eval_gold_pages_covered` if the
   retrieval_eval file doesn't exist; `pytest.skip` in
   `test_long_query_rank_ratchet` if the retriever isn't the exhaustive
   pipeline (i.e. doesn't expose `score_all`), since the rank ratchet
   requires full-corpus scoring. None of these fired in this environment —
   all three tests ran for real.
6. **Failure messages** in all three assertions include the measured
   values (mismatched gold pages, the actual top-5 `(page_id, score)`
   list, or the actual rank vs. the ratchet ceiling) plus, for the ratchet,
   an explicit statement that it may only be *tightened* by a deliberate
   commit, never loosened silently.
7. Discovered and worked around a footgun: running `ruff format` directly
   on the `.json` file (explicit path) makes ruff mis-treat it as a Python
   dict literal and inject an invalid trailing comma (breaks `json.loads`).
   Confirmed via `--verbose` that the repo-wide `ruff format --check .`
   (what `make lint` actually runs) never selects `.json` files for
   formatting at all — only `.py`/`.pyi`/markdown — so this had no bearing
   on `make lint`'s outcome. The committed file is plain valid JSON,
   2-space indent, no trailing comma.

## Verification

`uv run pytest -q -m "not slow" && make lint`:

```
........................................................................ [ 46%]
........................................................................ [ 93%]
..........                                                               [100%]
154 passed, 4 deselected in 1.10s
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
79 files already formatted
0 errors, 0 warnings, 0 informations
```

Baseline before this change was `154 passed, 1 deselected` — the
non-slow count is unchanged (154); deselected grew from 1 to 4 because
3 new slow tests were added and none run under `-m "not slow"`.

## Slow-test run (real model, real index)

`uv run pytest tests/retrieval/test_semantic_retrieval_eval.py -m slow -v`:

```
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /Users/barandincoguz/Desktop/project-delta/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/barandincoguz/Desktop/project-delta
configfile: pyproject.toml
plugins: anyio-4.14.2
collecting ... collected 3 items

tests/retrieval/test_semantic_retrieval_eval.py::test_retrieval_eval_gold_pages_covered PASSED [ 33%]
tests/retrieval/test_semantic_retrieval_eval.py::test_short_query_gold_in_top5 PASSED [ 66%]
tests/retrieval/test_semantic_retrieval_eval.py::test_long_query_rank_ratchet PASSED [100%]

============================== 3 passed in 11.55s ==============================
```

## Commit

```
a21a773 test: real-model semantic retrieval_eval regression locks (G0.1, G0.8, rank ratchet)
 2 files changed, 134 insertions(+)
 create mode 100644 tests/retrieval/retrieval_regression_expectations.json
 create mode 100644 tests/retrieval/test_semantic_retrieval_eval.py
```

Staged and committed only these two files (`git add
tests/retrieval/test_semantic_retrieval_eval.py
tests/retrieval/retrieval_regression_expectations.json`); no `git add -A`/`.` used.
`.agents/`, `data/bench/results/`, and `skills-lock.json` remain untouched
and untracked, as they were before this task started.

## Concerns / notes for reviewers

- The rank ratchet test currently depends on `retrieval_pipeline ==
  "exhaustive"` to expose `score_all`; if production ever defaults to
  `two-stage`, this test will skip rather than fail. That's intentional
  (two-stage doesn't support a meaningful full-corpus rank the same way)
  but worth knowing if pipeline defaults change later.
- No index builds were run and no existing index/data files were
  modified — only the two new test-related files were touched.
