# Colab Dense Artefacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible Qwen dense page artefacts on Colab, publish completed artefacts to a separate Hugging Face Dataset, and safely pull them into offline semantic-coverage evaluation.

**Architecture:** A small `bench.dense_artifacts` module owns the portable on-disk manifest and verification contract. A separate Hub adapter resolves immutable revisions, stages downloads, validates them, and atomically replaces only a named model directory. The existing evaluator consumes only verified artefacts; a dedicated builder and Colab notebook reuse the project encoder rather than implementing model logic independently.

**Tech Stack:** Python 3.12, NumPy, pandas, huggingface-hub, PyTorch/Transformers, pytest, Ruff, Pyright, Jupyter/Colab.

## Global Constraints

- The Hugging Face Dataset is `barandincoguz/belge-gozu-semantic-artifacts`; it is not the production index Dataset.
- A pull revision is exactly a 40-character Hub commit SHA; branches and tags are rejected.
- The page order, page-text hash, model repo/revision, encoding fingerprint, dtype, shape and file SHA-256 must validate before use.
- Artefact transfer must not touch the production index directory or production answer/calibration paths.
- Tests use mocked Hub APIs and synthetic NumPy files; no test downloads a model or calls the network.
- Changes use TDD and `apply_patch`; existing untracked files remain untouched.

---

## File structure

| Path | Responsibility |
| --- | --- |
| `src/belge_gozu/bench/dense_artifacts.py` | Manifest schema, deterministic fingerprints/hashes, and local artefact validation. |
| `src/belge_gozu/bench/dense_artifact_hub.py` | Immutable Hub upload/download plus atomic model-directory replacement. |
| `scripts/build_dense_artifacts.py` | Model-at-a-time, resumable dense artefact producer for a prepared source index. |
| `scripts/pull_dense_artifacts.py` | Explicit verified local downloader for a named model and immutable Hub commit. |
| `scripts/eval_semantic_coverage.py` | Rejects invalid artefacts before creating `DensePageIndex`. |
| `tests/bench/test_dense_artifacts.py` | Pure schema and integrity-contract tests. |
| `tests/bench/test_dense_artifact_hub.py` | Mocked Hub transfer and atomicity tests. |
| `tests/test_build_dense_artifacts.py` | Builder CLI/provenance tests without a transformer. |
| `tests/test_pull_dense_artifacts.py` | Pull CLI argument and validation wiring tests. |
| `notebooks/build_dense_artifacts_colab.ipynb` | Colab GPU runbook using the pinned repository code and `HF_TOKEN` secret. |
| `README.md` | Short operator instructions for Colab build, Hub commit recording, and verified pull. |

### Task 1: Local dense artefact contract

**Files:**
- Create: `src/belge_gozu/bench/dense_artifacts.py`
- Create: `tests/bench/test_dense_artifacts.py`

**Interfaces:**
- Produces `DenseArtifactExpectation`, `page_ids_sha256`, `sha256_file`, `encoding_fingerprint`, `write_dense_manifest`, and `validate_dense_artifact`.
- `validate_dense_artifact(path, expectation)` returns parsed manifest and raises `ValueError` before any caller loads embeddings.

- [ ] **Step 1: Write failing integrity tests**

```python
def test_validate_dense_artifact_accepts_exact_model_and_page_identity(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path, page_ids=["law/1", "law/2"])
    result = validate_dense_artifact(artifact, _expectation(["law/1", "law/2"]))
    assert result["embedding"]["shape"] == [2, 3]

def test_validate_dense_artifact_rejects_hash_mismatch_before_loading(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path, page_ids=["law/1"])
    (artifact / "embeddings.npy").write_bytes(b"altered")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_dense_artifact(artifact, _expectation(["law/1"]))

def test_validate_dense_artifact_rejects_another_page_order(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path, page_ids=["law/2", "law/1"])
    with pytest.raises(ValueError, match="page_ids"):
        validate_dense_artifact(artifact, _expectation(["law/1", "law/2"]))
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/bench/test_dense_artifacts.py -q`

Expected: FAIL because `belge_gozu.bench.dense_artifacts` does not exist.

- [ ] **Step 3: Implement the schema and verifier**

```python
@dataclass(frozen=True)
class DenseArtifactExpectation:
    model: DenseModelSpec
    page_ids: Sequence[str]
    page_texts_sha256: str

def encoding_fingerprint(spec: DenseModelSpec) -> str:
    protocol = {"instruction": spec.instruction, "max_length": spec.max_length,
                "pooling": "last-token", "normalization": "l2", "dtype": "float32"}
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def validate_dense_artifact(path: Path, expectation: DenseArtifactExpectation) -> dict[str, object]:
    manifest = _read_manifest(path / "dense.json")
    _require_equal(manifest["model"], {"repo": expectation.model.repo, "revision": expectation.model.revision})
    _require_equal(manifest["page_ids_sha256"], page_ids_sha256(expectation.page_ids))
    _require_equal(manifest["page_texts_sha256"], expectation.page_texts_sha256)
    _require_equal(manifest["encoding_fingerprint"], encoding_fingerprint(expectation.model))
    values = np.load(path / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    _validate_embedding(values, manifest["embedding"], sha256_file(path / "embeddings.npy"))
    return manifest
```

`write_dense_manifest` serializes schema version `1`, model identity, source Dataset
repo/commit, both page identities, the encoding fingerprint, file hash/shape/dtype,
and producer Git commit. It rejects an incomplete `.partial.npy` or `progress.json`.

- [ ] **Step 4: Run contract tests and static checks**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/bench/test_dense_artifacts.py -q && uv run ruff check src/belge_gozu/bench/dense_artifacts.py tests/bench/test_dense_artifacts.py && uv run pyright src/belge_gozu/bench/dense_artifacts.py tests/bench/test_dense_artifacts.py`

Expected: all tests pass; Ruff and Pyright report no errors.

- [ ] **Step 5: Commit the independently usable contract**

```bash
git add src/belge_gozu/bench/dense_artifacts.py tests/bench/test_dense_artifacts.py
git commit -m "feat(bench): validate dense artefacts"
```

### Task 2: Immutable Hub transfer

**Files:**
- Create: `src/belge_gozu/bench/dense_artifact_hub.py`
- Create: `tests/bench/test_dense_artifact_hub.py`

**Interfaces:**
- Consumes `DenseArtifactExpectation` and `validate_dense_artifact` from Task 1.
- Produces `push_dense_artifact(artifact_dir, repo_id, model_key, ...) -> str` and `pull_dense_artifact(repo_id, revision, model_key, destination_root, expectation, ...) -> str`.

- [ ] **Step 1: Write failing immutable-revision and atomicity tests**

```python
def test_pull_rejects_a_branch_without_touching_existing_model(tmp_path: Path) -> None:
    destination = _existing_destination(tmp_path)
    with pytest.raises(ValueError, match="40 karakterli commit SHA"):
        pull_dense_artifact("user/repo", "main", "qwen3-embedding-4b", destination, _expectation())
    assert (destination / "old.marker").read_text() == "old"

def test_pull_validates_staged_artifact_before_atomic_replacement(tmp_path: Path) -> None:
    destination = _existing_destination(tmp_path)
    api = _api_copying(_invalid_artifact(tmp_path))
    with pytest.raises(ValueError, match="SHA-256"):
        pull_dense_artifact("user/repo", REVISION, "qwen3-embedding-4b", destination, _expectation(), api=api)
    assert (destination / "old.marker").read_text() == "old"

def test_push_uploads_only_completed_model_directory_and_returns_resolved_sha(tmp_path: Path) -> None:
    api = _api_with_sha(REVISION)
    assert push_dense_artifact(_valid_artifact(tmp_path), "user/repo", "qwen3-embedding-4b", api=api) == REVISION
    assert api.upload_folder.call_args.kwargs["path_in_repo"] == "artifacts/qwen3-embedding-4b"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/bench/test_dense_artifact_hub.py -q`

Expected: FAIL because the Hub adapter does not exist.

- [ ] **Step 3: Implement staged Hub operations**

```python
def pull_dense_artifact(repo_id: str, revision: str, model_key: str,
                        destination_root: Path, expectation: DenseArtifactExpectation, *,
                        api: HfApi | None = None, token: str = "") -> str:
    _require_commit_sha(revision)
    resolved = _resolved_commit(_api(api, token), repo_id, revision)
    if resolved != revision.lower():
        raise ValueError("istenen revision ile çözümlenen commit uyuşmuyor")
    with tempfile.TemporaryDirectory(dir=destination_root.parent) as temp:
        client.snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                                 allow_patterns=[f"artifacts/{model_key}/*"], local_dir=temp)
        staged = Path(temp) / "artifacts" / model_key
        validate_dense_artifact(staged, expectation)
        _replace_tree_atomically(staged, destination_root / model_key)
    return resolved
```

`push_dense_artifact` first calls `validate_dense_artifact` with the supplied
expectation, creates the Dataset if necessary, uploads just that model directory,
then resolves and returns the resulting 40-character commit SHA. It has no
`delete_patterns`, so uploading one completed model cannot remove another.

- [ ] **Step 4: Run Hub tests and static checks**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/bench/test_dense_artifact_hub.py -q && uv run ruff check src/belge_gozu/bench/dense_artifact_hub.py tests/bench/test_dense_artifact_hub.py && uv run pyright src/belge_gozu/bench/dense_artifact_hub.py tests/bench/test_dense_artifact_hub.py`

Expected: all tests pass without contacting Hugging Face.

- [ ] **Step 5: Commit Hub transfer**

```bash
git add src/belge_gozu/bench/dense_artifact_hub.py tests/bench/test_dense_artifact_hub.py
git commit -m "feat(bench): transfer verified dense artefacts"
```

### Task 3: Build artefacts and enforce them in evaluation

**Files:**
- Create: `scripts/build_dense_artifacts.py`
- Create: `tests/test_build_dense_artifacts.py`
- Modify: `scripts/eval_semantic_coverage.py:126-177, 286-325, 470-482`
- Modify: `tests/test_eval_semantic_coverage.py`

**Interfaces:**
- Consumes Task 1 manifest writer and `DENSE_MODELS`, `TransformerDenseEncoder`, `resume_dense_embeddings`.
- Produces `build_model_artifact(...) -> Path`; evaluator calls `validate_dense_artifact` before `np.load`.

- [ ] **Step 1: Write failing builder and evaluator tests**

```python
def test_builder_writes_a_manifest_only_after_resumable_matrix_finishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder, "TransformerDenseEncoder", _FakeEncoder)
    result = builder.build_model_artifact(_spec(), {"p1": "one", "p2": "two"}, tmp_path,
                                          source_repo="user/index", source_revision=REVISION)
    assert (result / "embeddings.npy").is_file()
    assert (result / "dense.json").is_file()

def test_evaluator_rejects_dense_vectors_with_the_wrong_page_text_hash(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="page_texts"):
        esc._load_verified_embeddings(_spec(), ["p1"], {"p1": "text"}, tmp_path)
```

- [ ] **Step 2: Run focused tests to verify they fail**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_build_dense_artifacts.py tests/test_eval_semantic_coverage.py -q`

Expected: FAIL because builder and verified loader are absent.

- [ ] **Step 3: Implement the minimal build and verification path**

```python
def build_model_artifact(spec: DenseModelSpec, page_texts: Mapping[str, str], artifact_root: Path,
                         *, source_repo: str, source_revision: str,
                         max_batches: int | None = None, device: str | None = None) -> Path:
    page_ids = list(page_texts)
    target = artifact_root / spec.repo.rsplit("/", 1)[-1]
    vectors = resume_dense_embeddings(encoder, [page_texts[key] for key in page_ids], target,
                                      _resume_identity(spec, page_ids), batch_size=encoder.batch_size,
                                      max_batches=max_batches)
    if vectors is None:
        raise IncompleteDenseArtifact(target)
    write_dense_manifest(target, spec=spec, page_ids=page_ids,
                         page_texts_sha256=sha256_file(page_texts_path),
                         source_repo=source_repo, source_revision=source_revision,
                         producer_git_commit=git_commit())
    return target
```

The builder CLI requires `--index-dir`, `--source-repo`, `--source-revision`,
`--model`, and `--artifact-root`. It rejects a non-commit source revision and
reports resumable `in_progress` rather than publishing a partial matrix. The evaluator
uses `_load_verified_embeddings` for both original and expanded dense candidate paths.

- [ ] **Step 4: Run regression tests and static checks**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_build_dense_artifacts.py tests/test_eval_semantic_coverage.py tests/bench/test_dense_artifacts.py -q && uv run ruff check scripts/build_dense_artifacts.py scripts/eval_semantic_coverage.py tests/test_build_dense_artifacts.py tests/test_eval_semantic_coverage.py && uv run pyright scripts/build_dense_artifacts.py scripts/eval_semantic_coverage.py`

Expected: all tests pass and a malformed local dense artefact is never loaded.

- [ ] **Step 5: Commit builder and evaluator enforcement**

```bash
git add scripts/build_dense_artifacts.py scripts/eval_semantic_coverage.py tests/test_build_dense_artifacts.py tests/test_eval_semantic_coverage.py
git commit -m "feat(bench): build verified dense artefacts"
```

### Task 4: Explicit pull command

**Files:**
- Create: `scripts/pull_dense_artifacts.py`
- Create: `tests/test_pull_dense_artifacts.py`

**Interfaces:**
- Consumes `pull_dense_artifact` from Task 2 and page identity from the caller's `--index-dir`.
- Produces a command that prints the resolved immutable Hub commit SHA only after atomic installation succeeds.

- [ ] **Step 1: Write failing CLI wiring tests**

```python
def test_pull_cli_passes_local_page_identity_and_prints_resolved_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(puller, "pull_dense_artifact", lambda **_: REVISION)
    monkeypatch.setattr(puller, "load_page_texts", lambda _: {"p1": "metin"})
    assert puller.main(["--repo", "user/repo", "--revision", REVISION, "--model", "qwen3-embedding-4b", "--index-dir", str(tmp_path)]) == 0
    assert REVISION in capsys.readouterr().out
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_pull_dense_artifacts.py -q`

Expected: FAIL because the pull script is absent.

- [ ] **Step 3: Implement the explicit CLI**

```python
parser.add_argument("--repo", required=True)
parser.add_argument("--revision", required=True)
parser.add_argument("--model", choices=sorted(DENSE_MODELS))
parser.add_argument("--index-dir", type=Path, required=True)
parser.add_argument("--artifact-root", type=Path, default=Path("data/bench/dense-artifacts"))
sha = pull_dense_artifact(args.repo, args.revision, args.model, args.artifact_root,
                          DenseArtifactExpectation(DENSE_MODELS[args.model], page_ids,
                                                   sha256_file(args.index_dir / "page_texts.parquet")),
                          token=os.environ.get("HF_TOKEN", ""))
print(f"dense artefact indirildi; commit={sha}")
```

- [ ] **Step 4: Run CLI tests, help, and static checks**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_pull_dense_artifacts.py -q && uv run python scripts/pull_dense_artifacts.py --help && uv run ruff check scripts/pull_dense_artifacts.py tests/test_pull_dense_artifacts.py && uv run pyright scripts/pull_dense_artifacts.py`

Expected: tests and type checks pass; help documents immutable `--revision` and target paths.

- [ ] **Step 5: Commit pull command**

```bash
git add scripts/pull_dense_artifacts.py tests/test_pull_dense_artifacts.py
git commit -m "feat(bench): pull dense artefacts by commit"
```

### Task 5: Colab notebook and operator documentation

**Files:**
- Create: `notebooks/build_dense_artifacts_colab.ipynb`
- Modify: `README.md`

**Interfaces:**
- Consumes the Task 3 builder and Task 2 uploader directly from a pinned checkout.
- Produces an operator-visible source index commit, source code commit, model-specific Hub commit, and local checkpoint location.

- [ ] **Step 1: Add notebook cells with no embedded token**

The first code cell must require the exact inputs and reject insufficient hardware:

```python
from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")
SOURCE_REPO = "barandincoguz/belge-gozu-index"
SOURCE_REVISION = "PASTE_40_CHARACTER_INDEX_COMMIT"
CODE_REVISION = "PASTE_40_CHARACTER_CODE_COMMIT"
ARTIFACT_REPO = "barandincoguz/belge-gozu-semantic-artifacts"
assert HF_TOKEN and len(SOURCE_REVISION) == len(CODE_REVISION) == 40
```

The GPU cell uses `torch.cuda.get_device_properties(0).total_memory`; before the 8B
model it raises a clear error below `24 * 1024**3`. Subsequent cells clone the source
at `CODE_REVISION`, download only `index/page_texts.parquet` from `SOURCE_REVISION`,
run `build_dense_artifacts.py`, validate the result, and call `push_dense_artifact`.
They print the returned Hub commit SHA immediately after each completed model.

- [ ] **Step 2: Add README operator instructions**

Document this exact sequence:

```bash
uv run python scripts/pull_dense_artifacts.py \
  --repo barandincoguz/belge-gozu-semantic-artifacts \
  --revision <artifact-hub-commit-sha> \
  --model qwen3-embedding-8b \
  --index-dir "$BG_INDEX_DIR"
uv run python scripts/eval_semantic_coverage.py --help
```

State that a Colab session needs a GPU with at least 24 GB for 8B, the token belongs
only in Colab Secrets, and no uploaded artefact authorizes a production change.

- [ ] **Step 3: Validate notebook JSON and documentation diff**

Run: `uv run python -m json.tool notebooks/build_dense_artifacts_colab.ipynb >/dev/null && git diff --check && rg -n 'HF_TOKEN|SOURCE_REVISION|ARTIFACT_REPO' notebooks/build_dense_artifacts_colab.ipynb README.md`

Expected: valid notebook JSON; no token literal; all required run identifiers present.

- [ ] **Step 4: Commit the Colab runbook**

```bash
git add notebooks/build_dense_artifacts_colab.ipynb README.md
git commit -m "docs(bench): add Colab dense artefact runbook"
```

### Task 6: Full local verification and handoff

**Files:**
- Modify only files created by Tasks 1-5 if verification finds a defect.

- [ ] **Step 1: Run the complete affected test suite**

Run: `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/bench/test_dense_artifacts.py tests/bench/test_dense_artifact_hub.py tests/test_build_dense_artifacts.py tests/test_pull_dense_artifacts.py tests/test_eval_semantic_coverage.py tests/bench/test_semantic_coverage.py -q`

Expected: all tests pass without network or model downloads.

- [ ] **Step 2: Run repository quality gates**

Run: `uv run ruff check . && uv run pyright && git diff --check && git status --short`

Expected: Ruff, Pyright and whitespace checks pass. Pre-existing untracked files remain unmodified.

- [ ] **Step 3: Record the handoff boundary**

Do not claim an embedding result until a user runs the notebook on a qualifying GPU,
records its returned Hub commit SHA, pulls that SHA locally, and runs the existing
offline evaluator. Report the exact Colab hardware and all three immutable identities
(code, source index, artefact) alongside any result.

## Plan self-review

- Spec coverage: Tasks 1-2 implement the portable, immutable and atomic artefact contract; Task 3 uses the project encoder and enforces verification; Task 4 provides the explicit local pull; Task 5 provides Colab/Secrets/GPU/Drive-resume runbook; Task 6 prevents unmeasured production claims.
- Placeholder scan: no deferred implementation markers remain; `PASTE_...` values are deliberate notebook inputs, never committed credentials or revisions.
- Type consistency: Tasks 2-4 use `DenseArtifactExpectation`; task 3 writes the `dense.json` verified by Tasks 1-2 and read by the evaluator.
