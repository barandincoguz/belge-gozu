# Evaluation Suite Naming — Design

## Goal

Replace ambiguous internal terminology with names that state each dataset and
test's purpose. The migration is repository-wide, including historical
documentation, because the old term is banned for future project language.

## Naming contract

| Old concept | New contract |
|---|---|
| answerable retrieval_eval dataset | `retrieval_eval_vN` / “retrieval evaluation suite” |
| unanswerable dataset | `abstention_eval_vN` / “abstention evaluation suite” |
| real-model semantic retrieval_eval tests | `retrieval_regression` |
| retrieval_eval expectation artefact | `retrieval_regression_expectations.json` |
| review utility | `verify_retrieval_eval.py` |
| unanswerable validator | `validate_abstention_eval.py` |

`retrieval_eval`, `abstention_eval`, and cryptic abbreviations are not used in new identifiers,
CLI flags, prose, or artifacts. Filenames remain lowercase snake_case.

## Migration boundary

Move benchmark files and executable/test files with `git mv`, then update all
imports, default paths, flags, fixtures, UI copy, docs, and research history.
The JSONL content and benchmark semantics are unchanged. No compatibility alias
is kept: a stale caller must fail clearly rather than silently retain banned
terminology.

## Guardrail

Root `AGENTS.md` records the vocabulary and requires a final case-insensitive
repository scan that finds no banned term, excluding only Git internals and
unrelated user-owned binary files.
