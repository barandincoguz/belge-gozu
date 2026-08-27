# R15 report: --all/--only-verified for `bench run` and `bench oracle`

## Problem
`bench run` and `bench oracle` in `src/belge_gozu/cli.py` called `load_bench(bench)`,
which defaults to `only_verified=True`. `data/bench/canary_v1.jsonl` (48 rows) is
currently all `verification_status="draft"` (human verification pending), so both
commands raised `ValueError` and no measurement could run at all.

## Change

### `src/belge_gozu/cli.py`
- Added a new pure helper `_load_bench_mode(bench: Path, only_verified: bool) -> tuple[list, bool]`,
  placed immediately above `bench_run`. It calls `belge_gozu.bench.dataset.load_bench`
  (lazy import, no torch/model touch), prints the active mode, and returns
  `(questions, only_verified)`:
  - `only_verified=True` -> `bench modu: yalnız doğrulanmış (n=<count>)`
  - `only_verified=False` -> `bench modu: TÜMÜ (taslak dahil, n=<count>)`
- `bench_run`: added `only_verified: bool = typer.Option(False, "--only-verified/--all")`
  (single paired flag, mirroring the requested Typer idiom — `--all` is a real flag,
  not a silent no-op like in `scripts/d1_augmentation.py`). Default is `--all`
  (`only_verified=False`). Replaced the direct `load_bench(bench)` call with
  `_load_bench_mode(bench, only_verified)`. Added `"only_verified": only_verified`
  to the existing `config` dict passed into `run_retrieval_eval` (so it lands in
  `EvalReport.config` in the output JSON).
- `bench_oracle`: same `only_verified` option added (placed after `--int8-index`,
  before `--out`). Replaced `load_bench(bench)` with `_load_bench_mode(bench, only_verified)`.
  Added `"only_verified": only_verified` as a top-level key in the output report dict
  (next to `"bench"`).
- Removed the now-redundant per-command `from belge_gozu.bench.dataset import load_bench`
  imports (the helper does its own lazy import).

### `tests/test_cli.py`
- Added `_bench_q(**over)` / `_write_bench_jsonl(...)` helpers (mirroring the existing
  pattern in `tests/bench/test_dataset.py`) to build a minimal valid `BenchQuestion` row
  and write a temp JSONL.
- `test_load_bench_mode_only_verified`: temp JSONL with one `verified` + one `draft` row;
  asserts `_load_bench_mode(p, only_verified=True)` returns only the verified question and
  prints `bench modu: yalnız doğrulanmış (n=1)` (checked via `capsys`).
- `test_load_bench_mode_all`: same fixture; asserts `only_verified=False` returns both
  questions and prints `bench modu: TÜMÜ (taslak dahil, n=2)`.
- `test_bench_run_help_lists_only_verified_and_all` / `test_bench_oracle_help_lists_only_verified_and_all`:
  `CliRunner().invoke(app, ["bench", "run"/"oracle", "--help"])`, assert exit code 0 and
  that `--only-verified` and `--all` both appear in the help text. These do not construct
  an encoder or touch the model/index (typer/click do not execute the command callback for
  `--help`).

## Verification
- `uv run pytest -q -m "not slow"` -> `154 passed, 1 deselected in 1.82s`.
- `make lint` -> ruff check, ruff format --check, and pyright all green
  (`All checks passed!`, `78 files already formatted`, `0 errors, 0 warnings, 0 informations`).
- `uv run belge-gozu bench run --help` and `uv run belge-gozu bench oracle --help`
  both exit 0 instantly and show `--only-verified --all [default: all]` in the options
  table — confirmed no model/index load is triggered by `--help` (fast, no MPS activity).
- `git status --porcelain data/` was empty before and after all commands — nothing was
  written under `data/`; no index build was run; no encoder/model was instantiated.

## Commit
`eb66be5` on `feat/p0-retrieval-correctness`:
`feat(bench): --all/--only-verified selection for bench run and oracle (R15)`
Only `src/belge_gozu/cli.py` and `tests/test_cli.py` were staged (explicit paths,
no `git add -A`/`.`). Untracked `.agents/` and `skills-lock.json` in the working tree
were left untouched.

## Concerns / notes
- `_load_bench_mode`'s return type is annotated as `tuple[list, bool]` (unparameterized
  `list`) rather than `tuple[list[BenchQuestion], bool]`, to avoid adding a module-level
  import of `belge_gozu.bench.dataset.BenchQuestion` purely for a type hint, keeping with
  this file's existing pattern of lazy-importing bench/model-adjacent modules inside command
  bodies. pyright is satisfied as-is; if stricter typing is wanted later, a
  `TYPE_CHECKING`-guarded import would tighten it.
- `only_verified` is round-tripped through `_load_bench_mode` (passed in, returned back)
  per the task's specified preferred shape (`return (questions, only_verified)`), even
  though the value is already known at the call site — kept intentionally for parity with
  the spec and to make the call-site destructuring self-documenting.
- Did not touch `scripts/d1_augmentation.py` — it was read only as the pattern reference,
  per instructions its own `--all` no-op wart was called out but explicitly not to be
  copied, and no fix was requested there.
