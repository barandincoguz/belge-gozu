# Belge-Gözü

**Visual document RAG for Turkish legal documents — no OCR, no text parsing.**
Pages are indexed as *images*. A ColPali-class vision-language model retrieves the
right pages straight from pixels, and a swappable VLM answerer looks at those page
images and answers strictly from what it sees — citing pages, or admitting it doesn't
know — instead of hallucinating an article number.

v0 corpus: 4,222 pages across 50 core Turkish statutes (Anayasa, TBK, TCK, İş Kanunu,
KVKK, TTK, TMK, tax and finance law, and more) plus 6 historical Official Gazette scans
spanning 1928–1975.

## Live demo

**Status: pending — not created yet, by design, not by omission.** The retrieval index
and all 4,222 page images are already public on HF Datasets:
**[barandincoguz/belge-gozu-index](https://huggingface.co/datasets/barandincoguz/belge-gozu-index)**.
This repo already has everything a Space needs (`Dockerfile`, `pyproject.toml`, `src/`).
But as of this writing, Hugging Face requires a PRO subscription to create *any* Docker
or Gradio Space — even on the free `cpu-basic` tier — confirmed live via the Hub API
(`402 Payment Required`; only the fully-static SDK is free). This account doesn't have
PRO, so `barandincoguz/belge-gozu` hasn't been created yet rather than deployed and
guessed at. Creating and pushing it is one `huggingface_hub` call away once that's
resolved.

In the meantime, the whole system runs locally in a few minutes — see
[Quickstart](#quickstart) — against the exact same public index, no GPU required.
(Once live: free-tier Spaces sleep after inactivity, so expect ~1-2 min on first load.)

## How it works

```mermaid
flowchart TD
  subgraph OFF["1 - OFFLINE PIPELINE (one-time)"]
    M["Manifest CSV<br/>50 statutes + historical Official Gazette list"] --> D["Downloader (download.py)<br/>polite: 1s delay, resumable"]
    D --> R["Renderer (render.py)<br/>PDF -> WebP page image<br/>+ meta.parquet (page ids)"]
    R --> E["Encoder: ColSmol-500M (vision-language model)<br/>each page -> ~1000 tokens x 128-dim vectors"]
    E --> P["f16 master (float_store.py)<br/>-> int8 (474MB, shipped) or 1-bit (58MB, ablation)<br/>quantize.py / store.py"]
  end

  P -->|"belge-gozu index push"| HUB[("HF Datasets - free storage<br/>barandincoguz/belge-gozu-index")]

  subgraph ON["2 - ONLINE SERVICE (Docker, CPU)"]
    IDX["In-memory index<br/>int8 (shipped), mmap"]
    U["User<br/>single-page web UI"] -->|"question"| API["FastAPI (app/main.py)"]
    API --> QE["Query encoder<br/>same VLM, on CPU"]
    QE --> S2["Exhaustive MaxSim (late interaction)<br/>per-token max-similarity over the int8 index<br/>whole corpus -> top-5 pages (~0.24s/query, 4,222 pages)"]
    IDX --> S2
    S2 --> G{"score >= threshold?"}
    G -->|"no"| AB["Says 'not found'<br/>hallucination brake"]
    G -->|"yes"| ANS["Answerer (pluggable)<br/>Gemini Flash: top-5 page IMAGES + question<br/>-> Turkish answer with page citations"]
    ANS --> API
    AB --> API
    API -->|"answer + page thumbnails"| U
    API -.-> LOG["Telemetry<br/>sqlite request log + /stats"]
  end

  HUB -->|"pull + mmap at startup"| IDX
```

Retrieval is exhaustive: every query is scored against the whole corpus with
late-interaction MaxSim — no elimination pass — which takes ~0.24 s/query over the
current 4,222-page int8 index (CPU, idle machine). An earlier two-stage design first narrowed the corpus to ~200 candidates with
a cheap mean-sign Hamming filter before re-ranking with MaxSim; that filter turned out
to be discarding good candidates (see [v0 limitations](#v0-limitations)) and was
removed from the production path — it survives only as an ablation option
(`BG_RETRIEVAL_PIPELINE=two-stage`, which needs the 1-bit index). Any quantized
index approximates native float ColPali scoring, and the P0 plan's quantization ablation (C1/C2: float16 oracle vs.
int8 vs. 1-bit) has now been run on the 48-question canary benchmark (43 answerable;
**not a human-validated set** — see the caveat below, so treat these numbers as
provisional), in the production query/document format: **int8 matches float16 exactly
at every k** (Recall@1/5/20/50/200 all identical); **1-bit loses 7.0 points of
Recall@20** relative to float16 (0.233 vs. 0.302). 1-bit is also **slower, not
faster**: scoring all 4,222 pages against a 40-token query takes 1.08 s at 1-bit vs.
0.24 s at int8 vs. 0.08 s at float16 (CPU, idle machine), because int8/float16 hit a
BLAS matmul path while the 1-bit path builds large temporaries for the popcount
reduction. Index size is the one axis where 1-bit still wins (58 MB vs. 474 MB for
int8 vs. 919 MB for float16). **int8 is now what ships**: serving was the only missing
piece (the retriever previously accepted the packed 1-bit index only), and it is now
representation-agnostic, so the measured winner is also the served one. 1-bit remains
available as the ablation / disk-budget option (`data/index-traincompat-1bit`, 58 MB)
via `BG_INDEX_DIR`. Full tables:
[`docs/research/findings/2026-08-27-p0-baseline.md`](docs/research/findings/2026-08-27-p0-baseline.md)
and
[`docs/research/findings/2026-08-27-p0-gate.md`](docs/research/findings/2026-08-27-p0-gate.md).
Separately, the single biggest P0 result to date: switching the document encoder to
the checkpoint's training-time prompt (instead of the format `colpali-engine==0.3.18`
emits by default) raised float16 Recall@5 from 0.093 to 0.233 on that same
canary set.

**Benchmark provenance caveat (applies to every canary number on this page).** The
canary questions were drafted by model agents reading the page images; of the 48 rows,
only **3 were verified by a human** — the other 45 were checked by an independent model
pass that re-read the same images (`verification_kind: "model-cross-check"`). That pass
was not a rubber stamp (it found 5 label/evidence corrections and one real page-span
error, `c213`), but a model-verified benchmark **cannot be cited as human-validated**,
and because the verifying model is the same family that drafted the questions,
correlated blind spots are possible. Full provenance, limitations and the list of
defects found: [`data/bench/canary_v1.README.md`](data/bench/canary_v1.README.md).

The resulting score is itself
an **uncalibrated similarity** — the per-query-token average MaxSim, normalized to
roughly [-1, 1] so that every representation (int8, float16, 1-bit) reports on one
scale — not a confidence or probability. If it doesn't clear a threshold, the
service returns "I couldn't find grounds for this in the corpus" *before* ever calling
the LLM — the abstain path costs nothing and can't hallucinate. The threshold
(`BG_MIN_SCORE_THRESHOLD=0.58`) is a **mechanical transfer** of the old binary-scale
60.0 onto that normalized scale — it reproduces the same operating point, question for
question — and is therefore still uncalibrated and still non-separating (see
[v0 limitations](#v0-limitations)); real calibration is P2 work.

## Example queries

> **Stale — these rows describe the v0 pipeline and are kept for the record only.**
> They were logged before the P0 work replaced the retrieval path (Stage-1 removed,
> training-compatible prompt format adopted, index rebuilt). Both the ranks and the
> abstain outcomes below have since changed, and the "clean abstain" reading in
> particular no longer holds — see the measured threshold behaviour in
> [v0 limitations](#v0-limitations). They will be re-measured against the current
> pipeline before any public claim is made.

Runs against the local server, v0 pipeline, before the P0 changes:

| Question | Result (v0, superseded) |
|---|---|
| *"Kişisel Verilerin Korunması Kanunu'na göre açık rızanın geçerlilik şartları nelerdir?"* (KVKK: conditions for valid explicit consent) | **Substantive, correctly cited.** Retrieval put the actual KVKK pages at rank 1-2; the answer states the three real statutory conditions (specific to a matter, based on being informed, freely given) with citations. |
| *"Katma Değer Vergisi Kanunu'na göre KDV oranını belirlemeye kim yetkilidir?"* (who sets the VAT rate) | **Abstained** — the top score fell under the 60.0 threshold. Read at the time as the hallucination brake working; the P0 measurements show the threshold does not actually separate answerable from unanswerable questions, so this outcome cannot be credited to a working brake. |
| *"Türk Medeni Kanunu'na göre yerleşim yeri nasıl tanımlanır?"* (definition of legal domicile) | **Abstained**, same mechanism — and this is the query P0 used as its root-cause probe: the correct page (`k4721:4`) was ranked 3127/4222 by the old Stage-1. It is now at rank 1221 under the current pipeline; still not in the top-5, which is P1's target. |

Those were honestly representative of v0, not the best 3 out of hundreds: across 17 varied
legal questions tried in that session (see [v0 limitations](#v0-limitations)), 1 produced
a fully substantive correct answer and 3 abstained; the rest got an honest "not
found in the given pages" from the answerer despite retrieval clearing the score gate.
Explicitly naming the statute in the question measurably helped ("KVKK'na göre..." beat
"KVKK ne der?"-style phrasing every time it was tried) — a hint, in hindsight, of the
query-format problem P0 later found and fixed.

## Quickstart

```bash
make setup                  # dev deps: ruff, pyright, pytest
uv sync --all-extras        # + ml deps (torch, colpali-engine) — needed for index/serve
uv run belge-gozu --help

# serve straight from the published index + images (no local corpus needed)
# serves data/index-traincompat-int8 by default (int8, 474 MB — see BG_INDEX_DIR)
BG_HF_DATASET_REPO=barandincoguz/belge-gozu-index BG_DEVICE=cpu \
  uv run belge-gozu serve --pull
# -> http://localhost:7860
```

To reproduce the corpus from scratch instead of pulling the published index:

```bash
uv run belge-gozu corpus download   # ~50 statutes + historical RG scans -> data/pdf/
uv run belge-gozu corpus render     # PDF -> WebP page images + data/meta.parquet
uv run belge-gozu index build --precision f16 --out data/index-traincompat-f16
                                   # ColSmol-500M embeddings -> f16 master (919 MB)
uv run belge-gozu index derive --from data/index-traincompat-f16 \
  --quant int8 --out data/index-traincompat-int8      # what serving loads (474 MB)
uv run belge-gozu index push        # optional: publish index/ + images/ to your own HF dataset repo
uv run belge-gozu serve
```

`GOOGLE_API_KEY` (or `BG_GEMINI_API_KEY`) must be set for `/ask` to call the answerer;
`/search` works without it.

## Telemetri

Every `/ask` and `/search` request is logged to `data/requests.sqlite` (stage-by-stage
latency — query encode, exhaustive MaxSim, answerer — plus token counts, estimated USD
cost, and whether the request abstained) and mirrored as Prometheus
metrics on `GET /metrics` (`bg_*` series: request/stage duration histograms, abstain
and token counters, in-flight gauge, `bg_app_info`). `make obs-up` starts a local
Prometheus + Grafana (`http://localhost:3001`, anonymous access, dashboard `belge-gozu`
pre-provisioned) reading that endpoint; `make obs-down` tears it down.
The event table's `stage1_ms`/`stage2_ms` columns are a leftover of the removed
two-stage pipeline and stay `NULL` under the default (`exhaustive`) one — the
exhaustive stage's latency is recorded in the event's `detail.stages` map (as
`exhaustive_maxsim`) and exported to Prometheus in `bg_stage_duration_seconds`.
`uv run belge-gozu metrics summary` prints a quick p95/abstain/cost readout from the
SQLite log; `uv run belge-gozu metrics export --out <path>.parquet` dumps the raw event
table for offline analysis. See `docs/research/` for a real baseline measurement session
(load test, live `/ask` calls, and an honest write-up including a concurrency crash
found while running it).

## v0 limitations

This is a working end-to-end system, not a finished product — v0's known gaps, honestly:

- **Retrieval precision on natural-language queries is the weak link, not answer
  honesty.** In this session's own live testing, correct-law pages sometimes ranked
  just outside the top-5 window (or well outside it) even though the relevant statute
  is in the corpus; the answerer and the score-threshold abstain both behaved correctly
  every time (no fabricated citations observed), but a narrow top-5 with no reranking
  means real answers get missed. Query rewriting and a VLM reranking pass are planned
  next, along with a proper retrieval benchmark (v0 has none — the numbers above are a
  qualitative session log, not a scored eval).
- **The score threshold (`BG_MIN_SCORE_THRESHOLD=0.58`) is a rough calibration** from a
  handful of observed scores, not a tuned operating point — and after the T11 query/
  document format change it no longer separates answerable from unanswerable questions
  at all. The 0.58 is a *mechanical* transfer of the old binary-scale 60.0 onto the
  normalized score scale, reproducing the same operating point (42/43 answerable and
  4/5 unanswerable questions clear it, exactly as before) — a unit change, not a
  recalibration. Measured on the canary set (2026-08-29, production int8 index; 3/48 rows
  human-verified, 45 model-cross-checked — see the provenance caveat above):
  answerable top-1 scores run min 0.5767 / median 0.6250 / max 0.7450, unanswerable ones
  min 0.5679 / median 0.6550 / max 0.6866 — overlapping distributions, so no single
  cut-off splits them, and all three out-of-corpus questions currently clear 0.58.
  Raising the threshold would just abstain on real questions instead. Proper
  calibration (and probably score normalization) is P2 work; the current state is
  pinned by an `xfail(strict=True)` canary test so it can neither rot further nor be
  quietly declared fixed.
- **P0 root-cause investigation found the old two-stage Stage-1 filter was discarding
  good candidates, not just approximating the ranking.** For the query *"Türk Medeni
  Kanunu'na göre yerleşim yeri nasıl tanımlanır?"*, the correct page (`k4721:4`) ranked
  3127/4222 under the old mean-sign Hamming Stage-1 filter but 1576/4222 under
  exhaustive binary MaxSim; for *"Yerleşim yeri nedir?"* it ranked 1768 under Stage-1
  but **2** under exhaustive. Stage-1's top-200 candidate set overlapped the exhaustive
  top-200 by only 11.5-19% across the queries checked — it was picking a mostly
  different set of pages, not a faster version of the same ranking. Separately, the
  index was found to contain 3,960 all-zero padding-token rows across 15 pages — a
  real correctness defect (padding embeddings collapsing to an all-zero bit vector and
  scoring as if it were a genuine token). This is now fixed and locked:
  `PackedIndex.build` rejects all-zero rows at build time, and the rebuilt index has
  0 such rows: 3,776,882 tokens in the same (cpe-0.3.18) format as the old index —
  exactly 3,960 fewer than its 3,780,842 — and 3,759,994 in the train-compat format
  that ships today. It was **not**, however, one of the causes of today's poor retrieval
  numbers: measured on the canary benchmark (3/48 rows human-verified, 45
  model-cross-checked — see the provenance caveat above), an
  index rebuilt in the same format without the padding rows produced byte-identical
  Recall at every k and an identical top-20 list for 42 of the 43 questions versus the
  old, padded index. Independently, the encoder's retrieval training data is
  English-only, which is the likely reason Turkish paraphrase queries score weaker
  than queries that name the statute explicitly. A hybrid text+visual retrieval path
  is the planned fix (P1); a full retrieval benchmark is in progress to quantify where
  things stand today.
- **Single retrieval mode, single answerer.** No query rewriting, no agentic
  multi-step retrieval, no local-VLM fallback — Gemini Flash is the only answerer
  implemented, behind a pluggable `Answerer` protocol.
- **50 statutes, not the full corpus of Turkish law.** Scoped deliberately for v0;
  broader coverage is a later-phase concern once retrieval quality is solid enough
  to be worth scaling.
- **Space hosting is pending**, per [Live demo](#live-demo) above — a platform
  billing gate discovered during this deployment, not a code or infra gap.

## Data & license

Corpus text and images are rendered from official Turkish statutes and Official
Gazette (Resmî Gazete) publications. Turkish official texts (laws, regulations,
court decisions, and other public documents issued by government bodies) are exempt
from copyright protection under **FSEK (Fikir ve Sanat Eserleri Kanunu) art. 31**.
Source URLs for every document are recorded in `data/manifest/v0_manifest.csv` and in
`meta.parquet`'s `source_url` column.

## Tech stack

Python 3.12 · FastAPI + uvicorn · PyMuPDF (PDF -> image rendering, no OCR) ·
colpali-engine / ColSmol-500M (visual late-interaction retrieval) · PyTorch (MPS/CUDA/CPU) ·
NumPy (int8 index, memory-mapped) · Gemini API via `google-genai` (pluggable
answerer) · Hugging Face Hub (dataset storage + Space hosting) · pandas/pyarrow ·
pytest + ruff + pyright, enforced in CI.
