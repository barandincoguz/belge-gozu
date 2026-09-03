# P0 Reliability and Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the six P0 issues by making benchmark claims measurable, presentation truthful, and the Docker/Hugging Face deployment chain pinned, validated, and recoverable.

**Architecture:** Keep measurement, serving, and artifact transport as separate units. `bench.dataset` owns verification selection, the new `bench.answer_eval` module owns answer-level report models and metrics, FastAPI owns the public response contract, and `index.hub` owns authenticated/pinned/crash-safe transfers. Existing verifier, cache, and `AskService` behavior are reused instead of duplicated.

**Tech Stack:** Python 3.12, Pydantic, Typer, FastAPI, pytest, NumPy/Pandas, Hugging Face Hub, uv, Docker, GitHub Actions.

## Global Constraints

- Work only on branch `codex/p0-implementation` in `/Users/barandincoguz/Desktop/project-delta-p0-worktree`; do not modify the user's untracked `.agents/`, `skills-lock.json`, or personal `.env` in the main checkout.
- Use test-first RED → GREEN cycles for every production behavior change; documentation and generated lockfile changes are paired with validation commands.
- Never run the `test` benchmark split without `--yes-final-gate`; all automated answer-evaluation tests use stubs and no network.
- Keep `gate_calibrated=False` and `gate_verifier=False` as library defaults; the answer-evaluation command explicitly enables both gates for its own run only.
- Never print or commit Hugging Face or Gemini token values. The supported Hugging Face environment name is `HF_TOKEN`.
- A pull used for serving must name a non-empty immutable Hugging Face revision; uploads target a named branch and return the resulting commit SHA for later pinning.
- Publish the 479 MB int8 index only after local unit, lint, type, and transfer-failure tests pass. Existing remote images are reused only after their 4,222-file inventory is checked.
- Produce one independently revertible commit per task, followed by a provenance-only pin commit after the remote upload.

---

### Task 1: Make benchmark verification selection explicit (#3)

**Files:**
- Modify: `src/belge_gozu/bench/dataset.py`
- Modify: `src/belge_gozu/bench/harness.py`
- Modify: `src/belge_gozu/cli.py`
- Modify: `tests/bench/test_dataset.py`
- Modify: `tests/bench/test_harness.py`
- Modify: `tests/test_cli.py`
- Modify: `docs/research/findings/2026-08-27-p0-gate.md`

**Interfaces:**
- Produces: `VerificationLevel`, `BenchSelection`, and `select_bench(path, only_verified, min_verification)`.
- Preserves: `load_bench(...) -> list[BenchQuestion]` as a compatibility wrapper.
- Records: `config.verification = {only_verified, min_verification, total, selected, filtered_out}` in retrieval reports.

- [ ] **Step 1: Write failing selection tests**

```python
def test_min_verification_human_filters_model_and_mechanical_rows(tmp_path):
    path = write_rows(tmp_path, kinds=["human", "model-cross-check", "mechanical:manifest-absence"])
    selected = select_bench(path, min_verification=VerificationLevel.human)
    assert [q.verification_kind for q in selected.questions] == ["human"]
    assert selected.total == 3
    assert selected.filtered_out == 2

def test_min_verification_model_includes_human_and_model(tmp_path):
    path = write_rows(tmp_path, kinds=["human", "model-cross-check", "mechanical:manifest-absence"])
    selected = select_bench(path, min_verification=VerificationLevel.model_cross_check)
    assert {q.verification_kind for q in selected.questions} == {"human", "model-cross-check"}
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/bench/test_dataset.py -q`

Expected: import or assertion failure because `VerificationLevel`, `BenchSelection`, and `select_bench` do not exist.

- [ ] **Step 3: Implement the single verification ranking and selection result**

```python
class VerificationLevel(StrEnum):
    mechanical = "mechanical"
    model_cross_check = "model-cross-check"
    human = "human"

VERIFICATION_RANK = {
    VerificationLevel.mechanical: 0,
    VerificationLevel.model_cross_check: 1,
    VerificationLevel.human: 2,
}

class BenchSelection(BaseModel):
    questions: list[BenchQuestion]
    total: int
    filtered_out: int
    only_verified: bool
    min_verification: VerificationLevel | None
```

Map `mechanical:manifest-absence` to `VerificationLevel.mechanical`; a minimum verification level also requires `verification_status == "verified"`. Raise the existing empty-benchmark error after filtering.

- [ ] **Step 4: Add CLI/report provenance tests and verify RED**

Add a mixed-kind fixture and assert `bench run --min-verification human` selects 3 of 48 rows without loading the real encoder by testing `_load_bench_mode` directly. Add a harness assertion that the passed verification dictionary survives JSON serialization.

Run: `uv run pytest tests/test_cli.py tests/bench/test_harness.py -q`

Expected: failure because `--min-verification` and the report provenance block are absent.

- [ ] **Step 5: Wire the option through `bench run`, `bench oracle`, and `verify run`**

Keep `--only-verified/--all` for compatibility, add `--min-verification {mechanical,model-cross-check,human}`, and print all three counts:

```text
bench seçimi: toplam=48 seçilen=3 elenen=45; verification_status=verified; min=human
```

Store the same values in each report. Update the old P0 gate report with a supersession note that its “unchanged under `--only-verified`” observation was tautological because all rows had `verification_status=verified`; do not delete the old statement.

- [ ] **Step 6: Verify GREEN and commit**

Run: `uv run pytest tests/bench/test_dataset.py tests/bench/test_harness.py tests/test_cli.py -q`

Run: `uv run belge-gozu bench run --help`

Expected: tests pass and help lists `--min-verification` with the three levels.

Commit: `fix(bench): make verification level explicit`

---

### Task 2: Adjudicate the P1/G1 gate with pinned evidence (#1)

**Files:**
- Create: `docs/research/findings/2026-08-31-p1-gate.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1's corrected verification semantics.
- Produces: one seven-row G1 decision table and one explicit ASCII-folding decision.

- [ ] **Step 1: Recompute cited values from committed artifacts**

Run `jq` against:

```text
data/bench/results/20260830-1611-6d5b345-hybrid.json
data/bench/results/20260829-2115-3a031ca-hybrid.json
data/bench/results/verified-production-exhaustive.json
data/bench/results/latency-by-representation.json
```

Expected values include fractional R@5 `0.848837`, R@20/R@50 `0.930233`, pre-fold R@5 `0.825581`, pre-fold R@50 `0.953488`, and paraphrase R@50 `0.571429`.

- [ ] **Step 2: Write the seven-row gate report**

Use these verdicts:

```text
G1.1 FAIL — measured value is single-channel BM25 R@50 0.9302, not candidate-union recall.
G1.2 FAIL — paraphrase R@50 0.5714 while three named slices pass.
G1.3 NOT MEASURED — no reranker implementation; R@20 0.9302 makes a future trial eligible.
G1.4 PASS — long-query gold rank <= 2 from the committed expectation/diagnostic.
G1.5 PARTIAL — visual-only and hybrid quality plus scoring latency measured; live RAM/cold-start absent.
G1.6 PASS — three fusion attempts measured and rejected.
G1.7 NOT MEASURED — no live deployment.
```

The report must literally state `kapı kuralı çiğnendi`. Preserve ASCII folding because it makes accented/unaccented input invariant and improves served top-5 R@5 by `+0.0232`; record the `-0.0233` R@50 cost and bind its repair to dense-channel issue #11 rather than retroactively calling G1 passed.

- [ ] **Step 3: Validate references and commit**

Run: `rg -n "G1\.[1-7]|kapı kuralı çiğnendi|20260830-1611-6d5b345-hybrid.json|20260829-2115-3a031ca-hybrid.json" docs/research/findings/2026-08-31-p1-gate.md`

Expected: all seven gate identifiers, the explicit violation sentence, and both artifact names are present.

Commit: `docs(gates): adjudicate P1 quality gate`

---

### Task 3: Add answer-evaluation report models and metrics (#2 core)

**Files:**
- Create: `src/belge_gozu/bench/answer_eval.py`
- Create: `tests/bench/test_answer_eval.py`
- Modify: `src/belge_gozu/bench/__init__.py`

**Interfaces:**
- Produces: `ClaimRecord`, `AnswerRecord`, `RateEstimate`, `AnswerMetrics`, `AnswerEvalReport`, and `run_answer_eval(...)`.
- Consumes: `clopper_pearson_upper_bound` from `bench.calibration_metrics` and gate claim rows from `answer.verify.EvidenceGate`.

The concrete function signature is `run_answer_eval(records: Sequence[AnswerRecord], *,
run_id: str, git_commit: str, created_at: datetime, split: Literal["dev", "test"],
index_manifest: dict | None, index_revision: str | None, calibrator_key: str | None,
config: dict, dataset: dict, budget: dict) -> AnswerEvalReport`.

- [ ] **Step 1: Write failing metric tests**

```python
def test_answer_metrics_count_claim_support_completeness_and_false_support():
    records = [
        answer_record(answerable=True, verdicts=[("supported", [1]), ("unsupported", [1])]),
        answer_record(answerable=False, verdicts=[("supported", [1])]),
        answer_record(answerable=False, status="answered", honest_miss=True, verdicts=[]),
    ]
    report = run_answer_eval(records, provenance())
    assert report.metrics.citation_precision.rate == pytest.approx(2 / 3)
    assert report.metrics.citation_completeness.rate == 1.0
    assert report.metrics.false_supported_answer_rate.numerator == 1
    assert report.metrics.false_supported_answer_rate.denominator == 2

def test_empty_metric_denominator_is_explicitly_null():
    report = run_answer_eval([], provenance())
    assert report.metrics.citation_precision.rate is None
    assert report.metrics.citation_precision.upper_bound_95 is None
```

- [ ] **Step 2: Run the new test and verify RED**

Run: `uv run pytest tests/bench/test_answer_eval.py -q`

Expected: module import failure because `bench.answer_eval` is absent.

- [ ] **Step 3: Implement immutable report models and metric definitions**

Use these exact definitions:

- Citation precision: supported verified claims divided by all verified claims.
- Citation completeness: claims with one or more `cited_sources` divided by all segmented claims.
- False supported-answer rate: unanswerable questions presented as `answered`, not an honest miss, with at least one verified claim and every verified claim `supported`.
- For each metric, `upper_bound_95` is the Clopper–Pearson upper bound on its error/event numerator. Citation precision additionally exposes `error_upper_bound_95` and `lower_bound_95 = 1 - error_upper_bound_95`.
- Empty denominators remain `None`; they are never reported as perfect zero or one.

- [ ] **Step 4: Verify GREEN and commit**

Run: `uv run pytest tests/bench/test_answer_eval.py tests/bench/test_calibration_metrics.py -q`

Expected: all tests pass without network access.

Commit: `feat(bench): add answer evaluation metrics`

---

### Task 4: Add `bench answers` while reusing verifier/cache/budget behavior (#2 CLI)

**Files:**
- Modify: `src/belge_gozu/cli.py`
- Modify: `src/belge_gozu/bench/answer_eval.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `belge-gozu bench answers --split dev --max-llm-attempts N`.
- Preserves: `belge-gozu verify run` as a compatibility alias calling the same implementation.
- Consumes: `_verify_service`, `VerifierBudget`, `collecting()`, and Task 3 report models.
- Changes `_verify_service` to return `(service, gates, manifest, revision)` so the shared
  evaluator records the exact local artifact identity; update existing stub factories and
  compatibility tests in the same RED/GREEN cycle.

- [ ] **Step 1: Extend the existing stub harness tests and verify RED**

Invoke `bench answers` using `_verify_fixture` and assert:

```python
assert report["metrics"]["citation_precision"]["rate"] == 1.0
assert report["metrics"]["false_supported_answer_rate"]["rate"] == 0.0
assert report["metrics"]["false_supported_answer_rate"]["upper_bound_95"] is not None
assert report["index_revision"] == "rev/x/int8"
assert report["calibrator_key"] is None or isinstance(report["calibrator_key"], str)
```

Also retain tests for required budget, test-split lock, budget stop, and a second run producing zero verifier calls through the existing SHA-256 cache.

Run: `uv run pytest tests/test_cli.py -k 'answers or verify_run' -q`

Expected: `bench answers` is not registered.

- [ ] **Step 2: Refactor one shared command implementation**

Create `_answer_eval_command(...)` and have both Typer commands call it. For each question, capture final answer text, citations, `is_honest_miss`, status, top score, `gate1`, and every `gate2.claims` row into `AnswerRecord`. Include dataset SHA-256/git-blob, split SHA-256, index manifest/revision, recipe fingerprint, gate/calibrator key, model/config, budget use, and stop reason in `AnswerEvalReport`.

- [ ] **Step 3: Verify GREEN and commit**

Run: `uv run pytest tests/test_cli.py tests/bench/test_answer_eval.py -q`

Run: `uv run belge-gozu bench answers --help`

Expected: tests pass; help requires an explicit LLM-attempt budget and documents the `--yes-final-gate` barrier.

Commit: `feat(cli): add bench answers evaluation harness`

---

### Task 5: Make API and UI presentation reflect actual retrieval (#4)

**Files:**
- Modify: `src/belge_gozu/app/main.py`
- Modify: `src/belge_gozu/app/static/index.html`
- Modify: `tests/app/test_api.py`
- Modify: `tests/app/test_gates_api.py`
- Modify: `README.md`

**Interfaces:**
- Extends `/healthz` with server-owned `retrieval` labels.
- Extends `/search` with `status`, `no_match`, `threshold`, and `pipeline`.
- Extends `/ask` with additive `no_match`; existing status vocabulary remains unchanged.

- [ ] **Step 1: Write failing API contract tests**

```python
def test_search_marks_an_oov_result_as_no_match(tiny_corpus):
    c = make_client(tiny_corpus, min_score_threshold=10.6)
    body = c.post("/search", json={"query": "asdfgh qwerty"}).json()
    assert body["status"] == "no_match"
    assert body["no_match"] is True

def test_healthz_owns_retrieval_labels(tiny_corpus):
    retrieval = make_client(tiny_corpus).get("/healthz").json()["retrieval"]
    assert retrieval["ranking_channel"] == "BM25"
    assert "görsel" in retrieval["visual_role"].lower()
```

Run: `uv run pytest tests/app/test_api.py -k 'no_match or retrieval_labels' -q`

Expected: response fields are absent.

- [ ] **Step 2: Implement the additive server contract**

Compute `no_match = not hits or hits[0].score < s.min_score_threshold`, matching `AskService`'s actual threshold decision. Keep returned hits for API diagnostics but let the UI render a dedicated no-match card instead of presenting them as valid results.

- [ ] **Step 3: Write failing UI source-contract tests**

Assert that the HTML:

- says four example chips come from retrieval_eval and two are showcase queries;
- contains no `let THRESHOLD = 10.6`, no `const pacing`, and no client-side pipeline label table;
- reads threshold/ranking labels from `/healthz`;
- branches on `data.no_match` and renders `Eşleşme bulunamadı`;
- labels the bar as a relative score axis and explicitly says it is not a percentage.

Run: `uv run pytest tests/app/test_api.py -k 'ui' -q`

Expected: the old source sentence, fixed fallback, and pacing remain.

- [ ] **Step 4: Replace fabricated timing and fixed labels**

At request start, mark only the first stage active. Do not advance stages on timers. On response, mark completion and show only measured end-to-end elapsed time. Use `/healthz.retrieval` for ranking/score labels and use `null` until health is loaded; do not invent a fallback number. Change the chart explanation to `bağıl eksen; yüzde değildir` and keep the threshold line only when the server value exists.

- [ ] **Step 5: Verify GREEN and commit**

Run: `uv run pytest tests/app/test_api.py tests/app/test_gates_api.py -q`

Expected: API and source-contract tests pass.

Commit: `fix(app): make retrieval presentation truthful`

---

### Task 6: Authenticate, pin, validate, and atomically replace Hub artifacts (#6 code)

**Files:**
- Modify: `src/belge_gozu/config.py`
- Modify: `src/belge_gozu/index/hub.py`
- Modify: `src/belge_gozu/cli.py`
- Modify: `tests/test_config.py`
- Modify: `tests/index/test_hub.py`
- Modify: `tests/test_cli.py`
- Create: `.env.example`

**Interfaces:**
- Produces: `Settings.hf_token`, `Settings.hf_revision`.
- Changes: `push_index(..., token, revision) -> str` returns the resulting commit SHA.
- Changes: `pull_index(..., token, revision, expected_*) -> str` requires a pinned revision and returns the resolved commit SHA.

- [ ] **Step 1: Write failing token/revision/upload tests**

Assert `HfApi(token=...)` construction through an injected factory, `snapshot_download(revision=...)`, `upload_folder(revision=..., delete_patterns=["*"])`, and the returned `repo_info.sha`.

Run: `uv run pytest tests/index/test_hub.py tests/test_config.py -q`

Expected: signatures and settings fields are missing.

- [ ] **Step 2: Implement credentials and immutable revision flow**

`HF_TOKEN` is accepted through a validation alias without a `BG_` prefix; `BG_HF_REVISION` is the production pin. Uploads default to branch `main`; pulls reject an empty revision. Do not log token values.

- [ ] **Step 3: Write failing crash-safety tests**

Cover these cases with a valid tiny manifest/index/text fixture:

1. Interrupted `snapshot_download` leaves an existing target byte-for-byte unchanged.
2. Invalid/missing manifest leaves the target unchanged.
3. Incompatible corpus checksum leaves the target unchanged.
4. Successful pull removes stale representation files from the old target.
5. Failure during second `os.replace` restores the backup.
6. `page_texts.parquet` page IDs must exactly match `page_ids.json`.

Run: `uv run pytest tests/index/test_hub.py -q`

Expected: current in-place copy corrupts or preserves stale target files.

- [ ] **Step 4: Implement staged validation and rollback-safe tree swap**

Download to a temporary directory located beside the destination, validate `manifest.json`, representation signature, `check_compatibility`, and text/page alignment, then rename current target to a unique backup and staged target into place with `os.replace`. If the second rename fails, restore the backup. Apply the same staged swap to `data/images` when images are requested.

- [ ] **Step 5: Wire CLI fail-fast behavior and verify GREEN**

`serve --pull` and `index pull` must reject a missing repo or revision before starting Uvicorn. `index push --revision main` prints the returned immutable SHA.

Run: `uv run pytest tests/index/test_hub.py tests/test_config.py tests/test_cli.py -q`

Expected: all Hub and CLI tests pass with mocked network.

Commit: `fix(hub): pin and atomically validate index transfers`

---

### Task 7: Make telemetry initialization fail-soft (#5 recorder)

**Files:**
- Modify: `src/belge_gozu/telemetry/recorder.py`
- Modify: `tests/telemetry/test_recorder.py`
- Modify: `tests/app/test_api.py`

**Interfaces:**
- Preserves: `EventRecorder.record()` and `.close()`.
- Adds: an internal in-memory SQLite fallback when directory creation or persistent DB initialization fails.

- [ ] **Step 1: Write failing permission tests**

Monkeypatch `Path.mkdir` to raise `PermissionError` for the requested DB parent and assert `EventRecorder(...)` does not raise, logs one warning, and can still accept `record()`. Add an application test proving `create_app` and `/healthz` survive the same condition.

Run: `uv run pytest tests/telemetry/test_recorder.py tests/app/test_api.py -k 'unwritable or permission' -q`

Expected: constructor raises before creating SQLite.

- [ ] **Step 2: Implement one initialization path with memory fallback**

Factor schema/migration setup into `_initialize_connection(connection)`. First try the requested parent/file; on any `OSError` or `sqlite3.Error`, log `kalıcı telemetri açılamadı; bellek içi kayda düşülüyor` with exception context and initialize `sqlite3.connect(":memory:")`. Keep `db_path` pointing at the requested path so `/stats` continues its existing zeroed fallback instead of pretending persistence exists.

- [ ] **Step 3: Verify GREEN and commit**

Run: `uv run pytest tests/telemetry/test_recorder.py tests/app/test_api.py -q`

Expected: all recorder and application tests pass.

Commit: `fix(telemetry): survive unwritable recorder storage`

---

### Task 8: Harden the non-root CPU Docker runtime and CI smoke path (#5 packaging)

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `Dockerfile`
- Create: `.dockerignore`
- Create: `scripts/docker_smoke.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`

**Interfaces:**
- Docker runtime uses UID/GID 1000, `/data` for writable index/images/telemetry, and `/data/hf` for model cache.
- Linux resolves `torch` from the explicit PyTorch CPU index; macOS continues using PyPI.
- CI has one expected fail-fast check and one `/healthz` 200 check using a generated tiny index and stub answerer inside the built image.

- [ ] **Step 1: Add packaging source and refresh the lock**

Use the official uv layout:

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cpu", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

Run: `uv lock`

Run: `uv lock --check`

Expected: lock is current and Linux torch wheels carry the CPU index source.

- [ ] **Step 2: Write the runtime Docker contract**

Create `/data/index`, `/data/images`, and `/data/hf`; chown them to `app:app`; set `BG_DATA_DIR=/data`, `BG_INDEX_DIR=/data/index`, `HF_HOME=/data/hf`, public rate limits, query logging off, and the repo ID; switch to `USER 1000:1000`. Introduce `ARG BG_HF_REVISION=""` and copy it to `ENV`, deliberately making an unpinned default fail fast during this task. Task 9 replaces that empty build-argument default with the published immutable SHA. Add `.dockerignore` entries for `data`, `.venv`, caches, docs, `.git`, and local databases.

- [ ] **Step 3: Add two CI smoke checks**

The first runs the default command with an empty revision override and asserts non-zero exit plus the explicit revision error. The second launches `scripts/docker_smoke.py`, polls `http://127.0.0.1:7860/healthz`, asserts HTTP 200 and `status=ok`, then stops the container. The smoke script builds a three-page temporary fake index through production package APIs and injects `FakeEncoder` plus a stub answerer; it performs no network calls.

- [ ] **Step 4: Verify image behavior locally**

Run: `docker build -t belge-gozu:p0 .`

Run: `docker run --rm --entrypoint sh belge-gozu:p0 -c 'test "$(id -u)" = 1000 && test -w /data && test -w /data/hf'`

Run the same fail-fast and health smoke commands from the workflow.

Expected: build exits 0, user is 1000, writable paths pass, missing revision fails clearly, and smoke health returns 200.

Commit: `build(docker): harden runtime and add smoke checks`

---

### Task 9: Publish and pin the refreshed int8+text artifact (#6 provenance)

**Files:**
- Modify: `Dockerfile`
- Modify: `README.md`
- Create: `docs/research/evidence/2026-08-31-hf-index-publication.md`

**Interfaces:**
- Consumes: Tasks 6 and 8.
- Produces: one immutable 40-character HF revision used by Docker and documented with manifest identity.

- [ ] **Step 1: Validate the local artifact before any upload**

Check that the source directory has `codes.npy`, `scales.npy`, `offsets.npy`, `page_ids.json`, `meta.parquet`, `manifest.json`, and `page_texts.parquet`; validate the manifest checksum and assert 4,222 text rows aligned to 4,222 page IDs. Count local and remote images and require both to equal 4,222.

- [ ] **Step 2: Upload only the refreshed index tree**

Use the personal token without printing it. The legacy private file is sourced only in
the current shell; its value is copied to the supported variable and never echoed:

```text
BG_INDEX_DIR=/Users/barandincoguz/Desktop/project-delta/data/index-traincompat-int8
BG_DATA_DIR=/Users/barandincoguz/Desktop/project-delta/data
BG_HF_DATASET_REPO=barandincoguz/belge-gozu-index
source /Users/barandincoguz/Desktop/project-delta/.env
HF_TOKEN="$HF_KEY"
uv run belge-gozu index push --revision main --no-images
unset HF_TOKEN HF_KEY
```

Capture the command's returned immutable SHA. Verify through the public Hub API that the new tree contains `index/page_texts.parquet`, `index/codes.npy`, and no stale `index/tokens.npy` or `index/page_vecs.npy`.

- [ ] **Step 3: Test a fresh pinned pull**

Pull the returned SHA into a new temporary directory, compare `corpus_checksum`, `quantization=int8`, page count, and text/page alignment with the local manifest, then delete only that explicit temporary directory.

- [ ] **Step 4: Pin provenance and commit**

Write the exact SHA and publication date to Docker, README, and the evidence note. Include local/remote manifest fields and the successful fresh-pull command, without credentials.

Commit: `chore(release): pin refreshed HF index revision`

---

### Task 10: Self-review the commit chain and run full verification

**Files:**
- Modify only files required to fix review findings.

**Interfaces:**
- Consumes: all prior task commits.
- Produces: reviewed, verified branch with a linear rollback chain.

- [ ] **Step 1: Review requirements against the diff**

Compare `git diff e1ea02d..HEAD` line by line with GitHub issues #1–#6 and this plan. Confirm every acceptance criterion has a code, test, report, or explicit external evidence owner. Check that no secret, personal `.env`, test-split result, model cache, index binary, or generated data file is tracked.

- [ ] **Step 2: Review each commit independently**

Run: `git log --reverse --oneline e1ea02d..HEAD`

Run: `git diff --check e1ea02d..HEAD`

Inspect every commit with `git log --reverse --format=%H e1ea02d..HEAD | while read -r commit_sha; do git show --stat --oneline "$commit_sha"; done` and verify each is independently understandable and revertible. Fix Critical and Important findings in a separate `fix(review): address P0 self-review findings` commit.

- [ ] **Step 3: Run complete fresh verification**

Run: `uv sync --locked --extra dev`

Run: `uv run pytest -m "not slow" -q`

Run: `uv run ruff check .`

Run: `uv run ruff format --check .`

Run: `uv run pyright`

Run: `uv run python scripts/validate_abstention_eval.py`

Run: `docker build -t belge-gozu:p0 .`

Run the default fail-fast and `/healthz` smoke checks from Task 8.

Expected: all commands exit 0 except the intentionally missing-revision container check, which exits non-zero with the asserted message.

- [ ] **Step 4: Record final rollback chain**

Report the base SHA, ordered commit SHAs/messages, full verification counts, Docker image result, HF revision, and any acceptance criterion that could not be demonstrated. Do not merge, push the Git branch, close issues, or create a PR without a separate user request.
