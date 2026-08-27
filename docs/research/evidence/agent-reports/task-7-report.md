# Task 7 Report: Metrikler (`bench/metrics.py`)

## Summary

Implemented pure retrieval metric functions (`recall_at_k`, `mrr`, `ndcg_at_k`, `bootstrap_ci`)
in `src/belge_gozu/bench/metrics.py`, transcribed verbatim from the task brief, with the
matching test file `tests/bench/test_metrics.py`.

## TDD Evidence

**RED** — before implementation existed:
```
$ uv run pytest tests/bench/test_metrics.py -v
ERROR collecting tests/bench/test_metrics.py
ModuleNotFoundError: No module named 'belge_gozu.bench.metrics'
Interrupted: 1 error during collection
```

**GREEN** — after implementation:
```
$ uv run pytest tests/bench/test_metrics.py -v
tests/bench/test_metrics.py::test_recall PASSED                          [ 25%]
tests/bench/test_metrics.py::test_mrr PASSED                             [ 50%]
tests/bench/test_metrics.py::test_ndcg PASSED                            [ 75%]
tests/bench/test_metrics.py::test_bootstrap_ci_deterministic_and_ordered PASSED [100%]
4 passed in 0.01s
```

**Full bench suite:**
```
$ uv run pytest tests/bench -v
... 16 passed in 0.05s
```
(12 pre-existing `test_dataset.py` tests + 4 new `test_metrics.py` tests, all pass.)

**Full regression (non-slow):**
```
$ uv run pytest -q -m "not slow"
120 passed, 1 deselected, 6 warnings in 1.02s
```
Warnings are pre-existing (httpx/starlette deprecation, SWIG type deprecations) and unrelated
to this change.

**Lint:**
```
$ make lint
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
69 files already formatted
0 errors, 0 warnings, 0 informations
```

## Files Changed

- `src/belge_gozu/bench/metrics.py` (new, 37 lines) — `recall_at_k`, `mrr`, `ndcg_at_k`,
  `bootstrap_ci`, transcribed exactly as specified in the brief. No lint issues found in the
  brief's code (no unused imports, formatting already ruff-compliant) — no fixes were needed.
- `tests/bench/test_metrics.py` (new, 27 lines) — the four test functions from the brief,
  transcribed verbatim.

## Git

Staged only the two files above by explicit path (per controller ruling R5 — no `git add -A`/
`git add .`). Pre-existing untracked `.agents/` and `skills-lock.json` in the working tree were
left untouched and are not part of this commit.

Commit: `223c40dc9c719d31032ce2932e7d6837d7dce333`
```
feat(bench): retrieval metrics with deterministic bootstrap CI

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FLYyqhr8TzmrziP3vsKgjY
```
`git show --stat HEAD` confirms exactly the two intended files in the commit.

## Self-Review

- All four functions match the brief's interface signatures exactly (types, defaults, param
  names).
- `recall_at_k`: guards empty `relevant` set (returns 0.0), matches test cases.
- `mrr`: 1-indexed rank, returns 0.0 if no relevant doc found in ranked list.
- `ndcg_at_k`: binary relevance, log2 discount (`i+2` since enumerate starts at 0 → rank 1 gets
  `log2(2)=1`), IDCG computed over `min(len(relevant), k)` ideal positions, guards division by
  zero via `if idcg else 0.0`. Manually verified the brief's second assertion
  (`ndcg_at_k({"a"}, ["b", "a"], 5) ≈ 0.6309`): DCG = 1/log2(3) ≈ 0.6309, IDCG = 1/log2(2) = 1.0,
  ratio ≈ 0.6309 — matches.
- `bootstrap_ci`: percentile bootstrap using `np.random.default_rng(seed)` for determinism;
  empty input short-circuits to `(0.0, 0.0)` before any RNG use, matching the test. Verified
  ordering invariant `0.0 <= lo <= mean <= hi <= 1.0` holds for binary-valued inputs used in the
  test.
- No behavioral deviation from the brief was needed or made.

## Concerns

None. Implementation is pure, dependency-free beyond `numpy` (already a project dependency used
elsewhere), fully covered by the brief's tests, and lint/regression is clean.
