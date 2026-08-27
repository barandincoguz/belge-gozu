# Task 11 / Step 6 — Adopt the measured A/B winner as the production default

Branch: `feat/p0-retrieval-correctness` (unchanged, no switch/build performed).

## Manifest facts verified

Read `data/index-traincompat-1bit/manifest.json` directly (not taken on faith):

```json
{
 "model_name": "vidore/colSmol-500M",
 "model_revision": "650243e9bf299a5a082841ed2907da8b0b9ce553",
 "query_format": {
  "format_id": "train-compat-v1",
  "prefix": "Query: ",
  "suffix_token": "<end_of_utterance>",
  "n_suffix": 10,
  "trailing_newline": true
 },
 "doc_prompt_sha256": "3d11cdfb8bca21c81671b3d074f446b3de06904fe98d184dec3a4c3e096b5212",
 "quantization": "sign-1bit",
 "mask_policy": "drop-padding",
 "n_pages": 4222,
 "n_tokens": 3759994,
 "built_at": "2026-08-27T16:09:07.745259+00:00"
}
```

Confirmed: `query_format.format_id == "train-compat-v1"`, `quantization == "sign-1bit"`,
`n_pages == 4222` (matches the real corpus). `doc_prompt_sha256` is a hash, not a
readable string, but it is *not* the CPE_0_3_18 doc-prompt hash recorded in
`data/index-cpe0318-1bit/manifest.json` (`bb16e19c...`), consistent with the
index having been built with `TRAIN_COMPAT_DOC_PROMPT` rather than the
processor's own default. Also read `data/index-cpe0318-1bit/manifest.json`
(the losing format) for the fail-fast test: `query_format.format_id ==
"cpe-0.3.18"`, same `n_pages`/`corpus_checksum`, confirming both indexes are
built over the identical corpus and differ only on the format axis under test.

## Changes made

**`src/belge_gozu/index/manifest.py`** — moved `QueryFormatChoice`,
`DocPromptChoice`, and the two lookup dicts (renamed `_QUERY_FORMATS` /
`_DOC_PROMPTS` → public `QUERY_FORMATS` / `DOC_PROMPTS`) here from `cli.py`.
This is the single shared definition; both `cli.py` and `app/main.py` now
import from it instead of each keeping its own copy.

**`src/belge_gozu/cli.py`** — removed the local `QueryFormatChoice`,
`DocPromptChoice`, `_QUERY_FORMATS`, `_DOC_PROMPTS` definitions; imports the
shared ones from `belge_gozu.index.manifest` instead. `index_build` now
reads `QUERY_FORMATS[query_format]` / `DOC_PROMPTS[doc_prompt]` (same
behavior, same CLI flags/defaults — `--query-format cpe-0.3.18` /
`--doc-prompt processor-default` unchanged).

**`src/belge_gozu/config.py`**:
- `index_dir` default changed `Path("data/index")` → `Path("data/index-traincompat-1bit")`.
- Added `query_format_id: str = "train-compat-v1"`.
- Added `doc_prompt_id: Literal["processor-default", "train-compat"] = "train-compat"`
  (values mirror `cli.py`'s `DocPromptChoice` enum exactly).
- Each new/changed field has a Turkish comment recording the A/B measurement
  (float R@5 0.093→0.233, R@20 0.186→0.302; measured 2026-08-27; detail in
  the p0-gate report) as the reason.
- All three remain plain pydantic-settings fields with the `BG_` env prefix,
  so `BG_INDEX_DIR`, `BG_QUERY_FORMAT_ID`, `BG_DOC_PROMPT_ID` all work as
  overrides (no hardcoding, verified live via `BG_INDEX_DIR` in the fail-fast
  test below).

**`src/belge_gozu/app/main.py`** — `create_app` now resolves
`QUERY_FORMATS[QueryFormatChoice(s.query_format_id)]` and
`DOC_PROMPTS[DocPromptChoice(s.doc_prompt_id)]` from settings (constructing
the enum from the settings string gives free validation of the string value,
and satisfies pyright's dict-key typing) and passes both into
`ColSmolEncoder(s.retriever_model, s.device, query_format=..., visual_prompt_override=...)`
when no encoder is injected. The pre-existing serve-time compatibility check
(`getattr(encoder, "query_format", CPE_0_3_18).format_id` fallback for
injected/stub encoders that don't declare a `query_format` attribute) was
left untouched — see "fixture" note below for why that was safe.

No other files were touched. `.agents/`, `data/bench/results/`, and
`skills-lock.json` remain untracked/unstaged (pre-existing, not part of this
task) — nothing was staged from them.

## Fixture note (tiny_corpus / CPE_0_3_18)

Traced through carefully before touching anything: `check_compatibility`'s
`query_format_id` argument in `create_app` comes from
`getattr(encoder, "query_format", CPE_0_3_18).format_id` — i.e. it is driven
by the **injected encoder object**, not by `Settings.query_format_id`, except
on the `encoder is None` path (real `ColSmolEncoder`, never exercised by the
unit tests — every app/compat/telemetry test passes an explicit
`encoder=enc` where `enc` is `FakeEncoder` or a stub, none of which declare a
`query_format` attribute). So changing `Settings`'s default does not change
what those tests' compat check compares against; `tests/conftest.py`'s
`tiny_corpus` fixture (which still writes `CPE_0_3_18` into its manifest via
`make_manifest`'s default) continues to match the `CPE_0_3_18` fallback used
in every test. Ran the full suite to confirm empirically rather than assume —
154 passed, 0 failed, so **no fixture change was needed or made**. I did not
weaken the check; I simply confirmed by inspection + test run that config's
new default doesn't reach the test-injected-encoder path at all.

## Verification

### 1. `uv run pytest -q -m "not slow" && make lint`

```
154 passed, 1 deselected in 1.24s
```
```
uv run ruff check . && uv run ruff format --check . && uv run pyright
All checks passed!
78 files already formatted
0 errors, 0 warnings, 0 informations
```
Both green. (One intermediate pyright failure surfaced and was fixed along
the way: `QUERY_FORMATS[s.query_format_id]` / `DOC_PROMPTS[s.doc_prompt_id]`
didn't type-check against the `QueryFormatChoice`/`DocPromptChoice`-keyed
dicts because `Settings.query_format_id` is a plain `str` and
`doc_prompt_id` is a `Literal`; fixed by constructing
`QueryFormatChoice(s.query_format_id)` / `DocPromptChoice(s.doc_prompt_id)`
before indexing, which also validates the string at app-boot time.)

### 2. Real serve boot against the winning index, real retrieval query, no answerer/Gemini

Ran `create_app()` with default `Settings()` (which now resolves to
`data/index-traincompat-1bit`, `train-compat-v1`, `train-compat` doc prompt)
and a `StubAnswerer`, then `TestClient(app).post("/search", {"query":
"Yerleşim yeri nedir?", "k": 5})`. This loaded the real colSmol-500M model on
MPS and searched the real 4222-page index (~1-2 minutes).

```
index_dir=data/index-traincompat-1bit
query_format_id=train-compat-v1
doc_prompt_id=train-compat
[
  {"page_id": "k4721:80", "score": 78.5,              "doc_name": "Türk Medeni Kanunu", ...},
  {"page_id": "k4734:53", "score": 78.0,              "doc_name": "Kamu İhale Kanunu", ...},
  {"page_id": "k2004:21", "score": 73.33333333333333, "doc_name": "İcra ve İflas Kanunu", ...},
  {"page_id": "k4721:4",  "score": 73.16666666666667, "doc_name": "Türk Medeni Kanunu", ...},
  {"page_id": "k2918:38", "score": 72.83333333333333, "doc_name": "Karayolları Trafik Kanunu", ...}
]
page_ids: ['k4721:80', 'k4734:53', 'k2004:21', 'k4721:4', 'k2918:38']
k4721:4 in top-5: True
```

`k4721:4` appears at **rank 4** (0-indexed position 3), score **73.17** —
exactly matching the measured rank from the A/B controller.

### 3. Fail-fast still works against the losing index

Same script/settings, but with `BG_INDEX_DIR=data/index-cpe0318-1bit` (the
losing `cpe-0.3.18` index) and default `query_format_id` (still
`train-compat-v1`, proving `create_app()` builds the encoder with the
config's format and then correctly detects the mismatch against that index's
manifest):

```
index_dir=data/index-cpe0318-1bit
query_format_id=train-compat-v1
RAISED IndexCompatibilityError: indeks/serve uyumsuzluğu: query_format: indeks=cpe-0.3.18 serve=train-compat-v1
```

Raised `IndexCompatibilityError` mentioning `query_format`, as required.

## Concerns

- The compat check's fallback default (`getattr(encoder, "query_format",
  CPE_0_3_18)`) is now a slightly stale historical constant in spirit — it
  hardcodes the *old* default rather than resolving through
  `Settings.query_format_id` — but it is only reachable when a caller injects
  a stub/fake encoder without a `query_format` attribute (i.e., tests), never
  in real serving (`encoder is None` always yields a `ColSmolEncoder` with an
  explicit `query_format` attribute set from config). I left it as-is because
  (a) the task scope explicitly asked me to wire the `encoder is None`
  branch, not rework the compat-check fallback, and (b) empirically it
  doesn't cause any test to silently pass when it shouldn't — every test that
  cares about compatibility either supplies its own explicit
  `query_format_id` to `check_compatibility` directly, or uses `tiny_corpus`'s
  `CPE_0_3_18` manifest against that same `CPE_0_3_18` fallback. Flagging in
  case the controller wants that fallback changed to
  `resolved_query_format` for stricter consistency — happy to do it in a
  follow-up if desired, but did not make that call unilaterally since it
  wasn't in the required-changes list and touches a fixture the task warned
  me to be careful with.
- `doc_prompt_sha256` in the winning manifest could not be verified against a
  known plaintext (it's a hash of `TRAIN_COMPAT_DOC_PROMPT` computed at build
  time by `ColSmolEncoder`); I verified it differs from the losing index's
  `doc_prompt_sha256` rather than recomputing the hash independently, which
  is sufficient to confirm the two indexes used different doc prompts but
  does not independently prove which specific prompt string produced it.
- Two scratch verification scripts were written to
  `/private/tmp/.../scratchpad/` (outside the repo) to run the two live
  end-to-end checks; nothing was added under the repo for this.
