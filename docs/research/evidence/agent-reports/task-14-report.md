# Task 14 Steps 1-2 Report — README/UI honesty corrections

Scope: README.md + index.html + config.py text/comment edits only (Steps 1-2 of
task-14-brief.md). Steps 3-4 (baseline/gate reports) explicitly NOT written — left to
controller per instructions. No index build run, nothing written under `data/`.

Commit: `9fc4235` on `feat/p0-retrieval-correctness`
(`docs: honest scoring language in README, UI and config (P0 T14 step 1-2)`)

## README.md

### 1. "exact ... real ColPali-style scoring, not an approximation" claim

Before (in the "How it works" prose, one paragraph):
```
Retrieval is two-stage: a cheap Hamming-distance pass over binarized page-summary
vectors narrows the whole corpus to ~200 candidates in milliseconds, then exact
late-interaction MaxSim (real ColPali-style scoring, not an approximation) re-ranks
those candidates to the top 5. If the best score doesn't clear a threshold, the service
returns "I couldn't find grounds for this in the corpus" *before* ever calling the LLM —
the abstain path costs nothing and can't hallucinate.
```

After:
```
Retrieval is exhaustive: every query is scored against the whole corpus with binary
late-interaction MaxSim (Hamming distance standing in for the float dot-product) — no
elimination pass — which takes ~1.2 s/query over the current 4,222-page index on an
M4 Pro. An earlier two-stage design first narrowed the corpus to ~200 candidates with
a cheap mean-sign Hamming filter before re-ranking with MaxSim; that filter turned out
to be discarding good candidates (see [v0 limitations](#v0-limitations)) and was
removed from the production path — it survives only as an ablation option
(`BG_RETRIEVAL_PIPELINE=two-stage`). MaxSim over the binarized codes is exact *within
that binary code space*, but relative to native float ColPali scoring it is an
approximation; the size of that loss is being quantified in an ongoing quantization
ablation. The resulting score is itself an **uncalibrated similarity**
(`128 − 2×Hamming`, averaged per query token) — not a confidence or probability — and
if it doesn't clear a threshold (a rough v0 cut-off, not a tuned operating point), the
service returns "I couldn't find grounds for this in the corpus" *before* ever calling
the LLM — the abstain path costs nothing and can't hallucinate.
```
This single rewritten paragraph covers items (a) the "exact within binary code space
vs. approximation of native float" correction, (b) the prose description of production
now being exhaustive (no Stage-1), and (c) the "uncalibrated similarity, not
confidence/probability" note — all requested together since they live in the same
paragraph.

### 2. Mermaid diagram — Stage-1 removed from production path

Before (inside the `2 - ONLINE SERVICE` subgraph):
```
    QE --> S1["STAGE 1 - Elimination<br/>page-summary vector, Hamming (XOR+popcount)<br/>whole corpus -> top-200 candidates (milliseconds)"]
    IDX --> S1
    S1 --> S2["STAGE 2 - MaxSim (late interaction)<br/>exact ranking: top-200 -> top-5 pages"]
    S2 --> G{"score >= threshold?"}
```

After:
```
    QE --> S2["Exhaustive binary MaxSim (late interaction)<br/>Hamming XOR+popcount in place of float dot-product<br/>whole corpus -> top-5 pages (~1.2s/query, 4,222 pages)"]
    IDX --> S2
    S2 --> G{"score >= threshold?"}
```
S1 node removed entirely; QE and IDX both feed directly into the single S2 node, which
now feeds the threshold gate exactly as before. All other nodes/edges (OFF subgraph,
HUB, U, API, G, AB, ANS, LOG) untouched. Verified by inspection: every `[`/`{` has a
matching `]`/`}`, every quoted label is closed, no dangling edge references a removed
node id (`S1` no longer appears anywhere in the file).

### 4. v0 limitations — new P0 root-cause paragraph

Added as a new bullet (after the existing "score threshold is a rough calibration"
bullet, before "Single retrieval mode, single answerer"):
```
- **P0 root-cause investigation found the old two-stage Stage-1 filter was discarding
  good candidates, not just approximating the ranking.** For the query *"Türk Medeni
  Kanunu'na göre yerleşim yeri nasıl tanımlanır?"*, the correct page (`k4721:4`) ranked
  3127/4222 under the old mean-sign Hamming Stage-1 filter but 1576/4222 under
  exhaustive binary MaxSim; for *"Yerleşim yeri nedir?"* it ranked 1768 under Stage-1
  but **2** under exhaustive. Stage-1's top-200 candidate set overlapped the exhaustive
  top-200 by only 11.5-19% across the queries checked — it was picking a mostly
  different set of pages, not a faster version of the same ranking. Separately, the
  index was found to contain 3,960 all-zero padding-token rows across 15 pages (now
  rejected at build time), and the encoder's retrieval training data is English-only,
  which is the likely reason Turkish paraphrase queries score weaker than queries that
  name the statute explicitly. A hybrid text+visual retrieval path is the planned fix
  (P1); a full retrieval benchmark is in progress to quantify where things stand today.
```
Only the numbers explicitly supplied in the task instructions were used; no
extrapolated/invented figures. The existing "honest v0 log" (17-query session table +
narrative) and the "Data & license" section were left untouched, per instructions.

## src/belge_gozu/app/static/index.html

### 5a. Score footnote

Before: `skor: sorgu jetonu başına ortalama MaxSim benzerliği (yüksek = daha yakın)`

After: `skor: kalibre edilmemiş benzerlik (MaxSim, sorgu jetonu başına ortalama) —
güven ya da doğruluk yüzdesi DEĞİLDİR`

### 5b. Threshold explanation text (near the retrieval chart)

Before:
```
İki aşamalı retrieval’ın seçtiği en yakın 5 sayfa. <b>Kırmızı çizgi cevap eşiği:</b>
en iyi skor çizgiyi geçemezse sistem, yanıt uydurmak yerine "dayanak bulunamadı" der ve
Gemini hiç çağrılmaz.
```

After:
```
Retrieval’ın seçtiği en yakın 5 sayfa. <b>Kırmızı çizgi cevap eşiği:</b>
kaba, kalibre edilmemiş bir kesme noktasıdır (güven ölçüsü değil) — en iyi skor bu
çizgiyi geçemezse sistem, yanıt uydurmak yerine "dayanak bulunamadı" der ve Gemini
hiç çağrılmaz.
```
Note: beyond the two explicitly requested edits, I also removed the leading "İki
aşamalı" ("two-stage") qualifier from this sentence, since production retrieval is no
longer two-stage and leaving it would reintroduce the same inaccuracy this task exists
to fix. This is a text-node-only change in the same sentence already being edited; no
HTML structure, ids, classes, or other markup were touched. Both `id="q"` and
`id="ask-btn"` are untouched (verified: still present, `test_root_serves_ui` passes).
No other text in `index.html` (e.g. the pipeline-stage labels "4.222 sayfada Hamming
taraması" / "200 aday → MaxSim ilk 5") was changed — those were out of the explicit
scope given for this file and are left for a follow-up if desired.

## src/belge_gozu/config.py

Before (comment above `min_score_threshold`):
```
    # kaba v0 ayarı; gerçek kalibrasyon Plan 2 (Task 13 smoke test: gerçek soru
    # top_score~70.6, saçma soru top_score~52.4 -- 20.0 hiçbir zaman tetiklemiyordu)
    min_score_threshold: float = 60.0
```

After:
```
    # kaba v0 kalıntısı (Task 13 smoke test: gerçek soru top_score~70.6, saçma soru
    # top_score~52.4 -- 20.0 hiçbir zaman tetiklemiyordu). Bu skor bir güven/olasılık
    # ölçüsü DEĞİLDİR (bkz. README "v0 limitations"); gerçek kalibrasyon P2'nin işi.
    min_score_threshold: float = 60.0
```
Value `60.0` unchanged, as required. (Note: `retrieval_pipeline` and its comment,
just above this block, already reflected the exhaustive/two-stage split before I
started — presumably from the concurrent src/ work — so I left that comment as-is and
only touched the `min_score_threshold` comment per the explicit instruction.)

## Verification

```
$ uv run pytest -q -m "not slow"
145 passed, 1 deselected in 1.38s

$ make lint
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
76 files already formatted
0 errors, 0 warnings, 0 informations

$ uv run pytest -q tests/app/test_api.py::test_root_serves_ui
1 passed in 0.25s
```
Mermaid syntax sanity-checked by manual bracket/quote matching (no mmdc render run,
per the "do not start heavy processes" constraint — this is pure text editing so a
render wasn't warranted; happy to run one if requested).

`git status --short` before commit showed only `README.md`, `src/belge_gozu/app/static/index.html`,
and `src/belge_gozu/config.py` modified; `.agents/` and `skills-lock.json` are
untracked and unrelated (likely from the concurrent index-build agent) and were not
staged. Staged and committed by explicit path only (`git add README.md
src/belge_gozu/app/static/index.html src/belge_gozu/config.py`), never `-A`/`.`.
No `git index.lock` contention was encountered — add and commit both succeeded on the
first try.

## Concerns

- None blocking. One judgment call: in index.html I dropped "İki aşamalı" from the
  threshold-explanation sentence in addition to the two literally-quoted edits,
  because leaving it would have been the same category of inaccuracy this task is
  fixing. If the controller wants a stricter "only the two quoted sentences, verbatim
  scope" interpretation, that one extra word-removal is trivially revertable.
- I did not touch the pipeline-stage progress labels in index.html
  ("4.222 sayfada Hamming taraması", "200 aday → MaxSim ilk 5") even though they also
  describe the now-removed Stage-1 filter, since they were not in the explicit list of
  index.html changes for this task and the brief said "change only text nodes" for the
  two named items. Flagging for the controller/Step 3-4 owner in case a follow-up
  pass is wanted.
- Steps 3-4 (baseline/gate reports) were not started, per the constraint that they
  depend on in-progress measurement runs and are the controller's job.

---

# Fix report — review R1 (Critical 1&2, Important 3)

Commit: `acf7432` on `feat/p0-retrieval-correctness`
(`docs(ui): default UI describes the exhaustive pipeline it actually runs (review R1)`)

Files touched this round: `README.md`, `src/belge_gozu/app/static/index.html` only
(`config.py` untouched — no findings against it this round). No index build run,
nothing written under `data/`.

## Critical 1 & 2 — index.html still narrated the removed two-stage pipeline

### Footer (was line ~266)

Before:
```
ColSmol-500M görsel dizin · Hamming eleme + MaxSim geç-etkileşim ·
Gemini Flash yanıtlayıcı · kaynaklar sayfa kartlarında
```

After:
```
ColSmol-500M görsel dizin · bütün korpusta exhaustive binary MaxSim (geç-etkileşim) ·
Gemini Flash yanıtlayıcı · kaynaklar sayfa kartlarında
```

### Pipeline progress strip (was lines ~234-238)

Before (5 `<li class="stage">` steps):
```html
<li class="stage" data-st="0"><span class="dot"></span><span>sorgu kodlanıyor</span></li>
<li class="stage" data-st="1"><span class="dot"></span><span id="st-scan">4.222 sayfada Hamming taraması</span></li>
<li class="stage" data-st="2"><span class="dot"></span><span>200 aday → MaxSim ilk 5</span></li>
<li class="stage" data-st="3"><span class="dot"></span><span id="st-th">eşik kontrolü</span></li>
<li class="stage" data-st="4"><span class="dot"></span><span id="st-llm">Gemini sayfaları okuyor</span></li>
```

After (4 `<li class="stage">` steps — the Hamming-elimination and MaxSim-rerank steps
are collapsed into the one honest step that actually runs):
```html
<li class="stage" data-st="0"><span class="dot"></span><span>sorgu kodlanıyor</span></li>
<li class="stage" data-st="1"><span class="dot"></span><span id="st-scan">4.222 sayfada exhaustive MaxSim → ilk 5</span></li>
<li class="stage" data-st="2"><span class="dot"></span><span id="st-th">eşik kontrolü</span></li>
<li class="stage" data-st="3"><span class="dot"></span><span id="st-llm">Gemini sayfaları okuyor</span></li>
```

### JS kept consistent with the 4-step strip

I audited every place in the `<script>` block that touches `.stage`/`data-st`/stage
count before editing, via `grep -n "data-st\|st-scan\|st-th\|st-llm\|stage\b\|pacing\|j === "`.
Finding: `data-st` is a plain data attribute never read by JS — all stage indexing in
the script is positional, via `document.querySelectorAll(".stage")` and the array
index `j`/`i`. So collapsing two `<li>`s into one required three JS updates to stay
consistent with the new 4-element NodeList:

1. Dynamic scan-label text (was line ~286):
   - Before: `` $("st-scan").textContent = `${h.pages.toLocaleString("tr-TR")} sayfada Hamming taraması`; ``
   - After: `` $("st-scan").textContent = `${h.pages.toLocaleString("tr-TR")} sayfada exhaustive MaxSim → ilk 5`; ``

2. Pacing array (was line ~300) — was 5 entries (one per old stage), now 4:
   - Before: `const pacing = [0, 900, 1700, 2100, 2600];`
   - After: `const pacing = [0, 900, 2100, 2600];`
   (Purely cosmetic animation timing for the progress strip, not tied to real
   measured latency — reduced from 5 to 4 waypoints to match the new stage count.)

3. Last-stage index check in `pipelineFinish` (was lines ~312-313) — the Gemini step
   was index 4 out of 5, now index 3 out of 4:
   - Before: `if (j === 4 && abstained) { ... } else if (j === 4) { ... }`
   - After: `if (j === 3 && abstained) { ... } else if (j === 3) { ... }`

Verified after edit: exactly 4 `<li class="stage">` elements, `pacing` has 4 entries
(indices 0-3), and both `j === 3` checks correctly address the last stage (Gemini) —
no empty/skipped-by-mistake step, no out-of-range index.

`id="q"` and `id="ask-btn"` are untouched (neither is inside the pipeline strip or
footer); `tests/app/test_api.py::test_root_serves_ui` re-verified passing after the
edit (see full suite run below).

## Important 3 — README quantization-ablation sentence over-claimed

Before:
```
... but relative to native float ColPali scoring it is an
approximation; the size of that loss is being quantified in an ongoing quantization
ablation. The resulting score is itself an **uncalibrated similarity** ...
```

After:
```
... but relative to native float ColPali scoring it is an
approximation; the size of that loss is meant to be measured by the P0 plan's
quantization ablation (C1/C2: float16 oracle vs. int8 vs. 1-bit), which has not been
run yet — no results exist for it as of this writing. The resulting score is itself
an **uncalibrated similarity** ...
```
This keeps the substance (the ablation exists and is the mechanism that will quantify
the loss) but removes the implication that it's already running or that results
exist, and names the specific comparison (C1/C2: float16 oracle vs. int8 vs. 1-bit)
per the controller's correction. The hedged "likely reason ... Turkish paraphrase
queries" sentence in the v0-limitations bullet was left exactly as-is, per the
controller's explicit instruction that the hedge already carries it.

## Verification (post-fix)

```
$ uv run pytest -q -m "not slow"
145 passed, 1 deselected in 1.33s

$ make lint
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
76 files already formatted
0 errors, 0 warnings, 0 informations
```
`git status --short` before staging showed only `README.md` and
`src/belge_gozu/app/static/index.html` modified (`.agents/` and `skills-lock.json`
remain untracked/unrelated, not staged). Staged and committed by explicit path only
(`git add README.md src/belge_gozu/app/static/index.html`), never `-A`/`.`. No
`index.lock` contention encountered.

## Concerns

- None blocking. The two-stage ablation capability in the codebase itself
  (`BG_RETRIEVAL_PIPELINE=two-stage`, `stage1_candidates`, etc. in `config.py`) was
  left fully intact — this round only changed what the default UI narrates, not any
  runtime behavior or the ablation path, per the coordinator's explicit instruction.
- The pacing array's new intervals (`[0, 900, 2100, 2600]`) are an arbitrary but
  reasonable re-spacing of the old 5-waypoint schedule down to 4 waypoints; they are
  cosmetic (drive a CSS "active/done" progress animation) and not measured against
  real stage latency in either the old or new version, so no behavior regression here.
