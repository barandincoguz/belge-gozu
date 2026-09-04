# Resumable Semantic Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the offline Qwen dense evaluation resume safely after an interrupted process.

**Architecture:** `scripts/eval_semantic_coverage.py` owns its offline checkpoint files. A partial matrix is paired with an atomic identity/progress record and becomes the final dense artifact only after all page rows exist. The runner treats a budget-exhausted arm as incomplete, never as selectable.

**Tech Stack:** Python 3.12, NumPy memmap, pytest, Ruff, Pyright.

## Global Constraints

- Keep all behavior offline and preserve the production retrieval path.
- Identity is exact: model repo/revision plus ordered page-id SHA-256.
- Use atomic JSON updates and never expose a partial file as `embeddings.npy`.
- Follow TDD; test the interrupted/resumed behavior before production code.

---

### Task 1: Add validated resumable dense batches

**Files:**
- Modify: `scripts/eval_semantic_coverage.py`
- Create: `tests/test_eval_semantic_coverage.py`

- [ ] Write a failing test where a two-batch fake encoder stops after one batch, a second invocation resumes only the remaining text, and finalizes the matrix.
- [ ] Run the test and observe the missing resumable helper failure.
- [ ] Implement the smallest checkpoint reader/writer and batch loop needed for the test.
- [ ] Run focused tests; add the mismatched-identity rejection test and make it green.
- [ ] Commit the isolated checkpoint behavior.

### Task 2: Make the runner report incomplete work safely

**Files:**
- Modify: `scripts/eval_semantic_coverage.py`
- Modify: `tests/test_eval_semantic_coverage.py`

- [ ] Write a failing test that verifies a batch budget yields `status="in_progress"` and no final artifact.
- [ ] Implement `--max-dense-batches`, visible progress output, and non-selectable incomplete arms.
- [ ] Run focused tests, Ruff, and Pyright.
- [ ] Restart the real experiment in bounded slices until both dense artifacts, expansion cache, JSON, and HTML exist.
