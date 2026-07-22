# Data engineering workflow

End-to-end documentation of how this project moves data from extraction through to loading,
so the pipeline can be understood and reproduced without reading the code. Every claim is
grounded in the file named beside it. Where this document and the code disagree, the code
wins and the discrepancy is flagged. Steps inferred rather than found in code are marked
**[VERIFY]**. File links are relative to this document in `docs/`.

## 1. Overview

The pipeline measures in-context visibility and verifiability across generative engines. It
collects grounded (web-searching) responses from five engines for a frozen query set,
repeated `k` times per (query, engine) cell, then computes natural-language-inference (NLI)
attribution metrics over each response. It follows a medallion architecture: **bronze** (raw,
immutable API captures, one JSON per cell), then **silver** (canonical, engine-agnostic
relational tables rebuilt purely from bronze), then **gold** (metric-ready aggregates and,
for scale-up, dimensional flat tables). Two run profiles share one codebase and one config,
switched by a single environment variable so pilot and scale-up never share files: scale-up
is 250 queries with `k=8`, pilot is 100 queries with `k=3`
([../thesis_config.py](../thesis_config.py) lines 144-148). The silver layer is a normalised
relational model in third normal form, not a snowflake schema: each entity (responses,
citations, sources) is its own table keyed by `query_id` / `engine` / `run_index` (and
`url_canonical` for sources), with no nested duplication of parent attributes.

## 2. Data extraction (bronze)

Extraction is driven by [../scripts/run_collection.py](../scripts/run_collection.py), an
engine-agnostic loop over the frozen query registry, the active engines and the `K` runs. Each
engine has a single adapter whose `fetch()` makes the grounded API call and writes one verbatim
bronze JSON per cell, named `{engine}__{query_id}__r{run}.json`
([../src/adapters/base.py](../src/adapters/base.py) lines 18-24). Citation parsing does not
happen here, only capture.

The five active engines are ChatGPT (`gpt-5.5`, Responses API with the `web_search` tool),
Claude (`claude-sonnet-4-6`, Messages API with `web_search_20250305`), Gemini
(`gemini-3.5-flash`, `generate_content` with `google_search`), Perplexity (`sonar`,
inherently grounded), and Kimi (`kimi-k2.6`, the `$web_search` builtin run as a tool-call
loop) ([../thesis_config.py](../thesis_config.py) lines 61-110). Grounding is a precondition:
a plain call returns no sources, so each engine is queried with its search mechanism enabled.
A sixth engine, Mistral, is defined in config but excluded from collection after it returned
218 HTTP 429 errors in the pilot; its adapter is retained for reproducibility only
([../thesis_config.py](../thesis_config.py) lines 106-115).

The query set is GEO-Bench (repo `GEO-Optim/geo-bench`, `test` split), with a `query_id`
derived as a 64-bit hash of the query text since the dataset has no native id
([../thesis_config.py](../thesis_config.py) lines 33-42). The scale-up registry is the frozen
`data/scaleup/queries/scaleup_queries_v2.parquet` of 250 queries; at `k=8` across five engines
the theoretical grid is 10,000 cells. **[VERIFY]** `k=8` is grounded in
[../thesis_config.py](../thesis_config.py) (`PRODUCTION_K = 8`, line 142) and in the launcher's
expected row count below; note that `methodology_notes.md` section 5 describes `k=7` in prose,
an internal-documentation inconsistency, whereas config and the row counts are `k=8`.

Rate and failure handling ([../scripts/run_collection.py](../scripts/run_collection.py) lines
56-131, 225-297):

- Ordering is query-major (query, then engine, then run) so a query's engines are collected
  close in time (comparable web state) and a cell's repeats run back-to-back.
- Each call retries with exponential backoff up to `MAX_RETRIES = 4`; the default inter-call
  delay is 0.5 seconds. An adaptive (AIMD) delay controller exists but is enabled only for
  Mistral; the five active engines use the fixed default.
- A per-engine circuit-breaker aborts an engine after 20 consecutive same-type failures (for
  example a run of HTTP 429s), leaving other engines untouched.
- Collection is resume-by-bronze-file: a cell whose JSON already exists is skipped and never
  re-called, so a re-run continues where it stopped and naturally retries missing cells.

Model-id versioning per call: every bronze record stores both `model_requested` and
`model_served`, the id asked for and the exact id the API returned, because the two can differ
and drift over the collection window; where a dated snapshot exists it is pinned (OpenAI
`gpt-5.5-2026-04-23`), and where not, the served-id log plus a tight collection window carry
the version-drift control ([../src/schema.py](../src/schema.py) lines 44-58;
[docs/methodology_notes.md](methodology_notes.md) section 1).

Failure logging: failed cells are written to `collection_failures.jsonl` under the profile's
data root, one JSON line each, carrying engine, query, run, the exception type and message,
and the best-effort HTTP status ([../scripts/run_collection.py](../scripts/run_collection.py)
lines 293-297).

Kimi content-filter rejection **[discrepancy]**: the prompt describes this as a
`collection_failures.jsonl` entry, but the repo does not record it that way. All eight scale-up
runs of one Kimi query (`gb_3fcf760b1a2ea4f8`, a Xi Jinping / PLA query) are simply absent from
bronze, and no `collection_failures.jsonl` was preserved for the scale-up run. The gap was found
during the pre-gold audit by reconciling silver row counts against the registry grid, and is
interpreted as most plausibly provider-side content filtering in
[docs/decision_logs/2026-06-12_kimi_missing_cell.md](decision_logs/2026-06-12_kimi_missing_cell.md).
Scale-up bronze is therefore 9,992 files (10,000 minus 8), and Kimi has 249 query-cells versus
250 for the other four engines.

## 3. Transformation (silver)

Silver is a pure transform with no network and no keys: each bronze record is run back through
its engine adapter's `normalise()` to produce a common `CanonicalRecord`, which is then
flattened into the relational tables ([../src/silver.py](../src/silver.py) lines 1-11, 80-91).
The adapter is instantiated without its `__init__`, so no API key is needed, which is what
makes silver rebuildable from immutable bronze at any time.

The per-engine adapter is the only engine-specific code. Its `normalise()` reads the captured
bronze payload and converts each engine's native citation format into the engine-agnostic
`CitedSource` shape, so all downstream metric code runs identically across engines. For
example, the OpenAI adapter walks the typed output array, keeps only `url_citation`
annotations, and de-dupes each URL to its first 1-based position
([../src/adapters/openai_adapter.py](../src/adapters/openai_adapter.py) lines 74-118); the
Kimi adapter, which has no structured channel, parses self-reported URLs out of the answer
text ([../src/adapters/kimi_adapter.py](../src/adapters/kimi_adapter.py) lines 132-147).
Provenance is recorded on each source: the four structured engines are `provider_certified`
and Kimi's parsed URLs are `self_reported_prompt_elicited`
([../src/schema.py](../src/schema.py) lines 23-33).

Cleaning and normalisation:

- Citation URLs are canonicalised: host lowercased, fragment and known tracking parameters
  (`utm_*`, `gclid`, `fbclid`, and similar) dropped, trailing slash trimmed; best-effort, and
  the original string is returned on any parse failure
  ([../src/silver.py](../src/silver.py) lines 37-59).
- Domains are reduced with a `www.` strip so `www.vogue.com` and `vogue.com` collapse to one.
- `build_silver()` refuses to build across mixed run profiles, and normalisation failures are
  collected and reported rather than silently dropped
  ([../src/silver.py](../src/silver.py) lines 102-112, 158-183).

Outputs ([../src/silver.py](../src/silver.py) lines 115-155): `responses_{profile}.parquet`
(one row per engine, query, run cell) and `citations_{profile}.parquet` (one row per cited
source, long form). For scale-up these hold 9,992 and 58,851 rows respectively (README data
tree). A further silver stage fetches source-page content so attribution can later run
offline: [../scripts/build_sources.py](../scripts/build_sources.py) fetches each unique
`url_canonical` in parallel with a per-domain rate limiter, cleans the HTML to text, and
atomically writes one JSON per URL, resuming by skipping URLs already on disk; then
[../scripts/aggregate_sources.py](../scripts/aggregate_sources.py) folds those into the
8-column `sources_{profile}.parquet` (columns: `url_canonical`, `domain`, `fetch_status`,
`http_status_code`, `content_length`, `cleaned_text`, `fetch_timestamp_utc`,
`title_from_html`), deduped on `url_canonical`. Scale-up sources is 25,373 URLs, about 81 per
cent fetched ok (README data tree).

## 4. Enrichment and gold layer

Gold is the NLI attribution stage, driven by
[../scripts/run_nli_pilot.py](../scripts/run_nli_pilot.py) (data-driven, so the same script
runs on smoke, pilot or scale-up silver) and computed by
[../src/nli_attribution.py](../src/nli_attribution.py). It processes every cell in the silver
responses table and writes one row per cell. Before scoring it pre-populates an in-memory
cache from `sources_{profile}.parquet` (streamed in batches, since the scale-up `cleaned_text`
column is several gigabytes uncompressed), so no live HTTP happens during NLI and the run is
reproducible from immutable artefacts ([../scripts/run_nli_pilot.py](../scripts/run_nli_pilot.py)
lines 52-102).

Method and model:

- NLI model `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, entailment threshold
  tau = 0.50 over the softmax entailment probability
  ([../thesis_config.py](../thesis_config.py) lines 230, 233).
- The answer is split into sentences by the pinned `regex` segmenter, which is treated as part
  of the measurement instrument because sentence count `N` is the AIS denominator and drives
  every PAWC position weight ([../src/attribution.py](../src/attribution.py) lines 38-53;
  [../thesis_config.py](../thesis_config.py) lines 236-244).
- Sentence-to-source pairing: for each (sentence, source) pair the source is scored and, if the
  entailment probability is at or above tau, the source is judged to support the sentence
  ([../src/nli_attribution.py](../src/nli_attribution.py) lines 81-129).

Source truncation **[discrepancy]**: the prompt states a 512-token source truncation with no
windowing. The repo does neither exactly. The documented limitation, described as such in the
README, is a **4,000-character** truncation of each source page (`JUDGE_SOURCE_CHAR_LIMIT`),
which the README shows exposes engines non-uniformly (85 per cent of Kimi and Claude sources
exceed 4,000 characters versus 63 per cent for Gemini), biasing entailment scores downward as a
known-direction lower bound (README "Source truncation — documented limitation";
[../thesis_config.py](../thesis_config.py) line 202). After that character cut, the code does
window: each source is chunked into overlapping 200-token windows with a 50-token stride, and
the maximum entailment across windows is taken
([../thesis_config.py](../thesis_config.py) lines 231-232;
[../src/nli_attribution.py](../src/nli_attribution.py) lines 45-60). The number 512 is the
tokenizer `max_length` applied per NLI pair (one source window plus the sentence), not a single
non-windowed source cut ([../src/nli_attribution.py](../src/nli_attribution.py) line 71). Treat
the 4,000-character limit plus 200/50 windowing as the ground truth for what produced the data.

Metrics computed ([../src/nli_attribution.py](../src/nli_attribution.py) lines 100-144;
[../src/attribution.py](../src/attribution.py) lines 75-78):

- Position weight is a linear decay, first sentence 1.0, last 1/N: `(N - i + 1) / N`.
- PAWC per source position is the sum, over sentences that source supports, of
  `word_count * position_weight`; `pawc_total` sums across positions.
- AIS is supported sentences divided by total sentences (a sentence counts as supported if at
  least one cited source supports it).
- CV is computed later by [../scripts/aggregate_cv_pilot.py](../scripts/aggregate_cv_pilot.py),
  which collapses the per-call table to the (query, engine) cell level and reports mean, SD and
  CV = SD / mean for PAWC and AIS across the `k` repeats; zero-mean cells yield CV = NaN by
  design and are kept, not dropped ([../scripts/aggregate_cv_pilot.py](../scripts/aggregate_cv_pilot.py)
  lines 46-63).

Output location: the driver writes `nli_dir / nli_{profile}.parquet`, where `nli_dir` is silver
for scale-up and gold for the frozen pilot, following the 2026-06-26 medallion realignment that
routes the scale-up per-response NLI table to silver
([../scripts/run_nli_pilot.py](../scripts/run_nli_pilot.py) lines 104-110).

## 5. Compute infrastructure

The NLI stage runs on a rented GPU virtual machine (RunPod). **[VERIFY]** there is no
`docs/RunPod_Scaleup_Guide.md` in the repo, so this section is summarised from the actual
runbook, the launcher script [../run_runpod_nli_scaleup.command](../run_runpod_nli_scaleup.command)
(the pilot equivalent is [../run_runpod_nli_pilot.command](../run_runpod_nli_pilot.command)),
which runs from the local Mac over SSH.

Deployment and environment setup ([../run_runpod_nli_scaleup.command](../run_runpod_nli_scaleup.command)
lines 38-83): the launcher verifies SSH reachability, clones the repo to `/workspace/thesis` if
missing and always hard-resets to `origin/main` so a stale clone never runs, creates a venv and
installs `requirements.lock`, asserts CUDA is reachable via `torch.cuda`, and pre-caches the
DeBERTa model. Profile and device are set by environment variable, `THESIS_RUN_PROFILE=scaleup`
and `NLI_DEVICE=cuda`, not by editing config. A config gate then asserts the config resolves to
profile `scaleup`, device `cuda` and segmenter `regex`, plus a guard that the metric code has
exactly four `str(k): round` lines. Only the three scale-up silver parquets (responses,
citations, sources, about 3.5 GB) are rsynced to the pod; bronze is not needed because the
driver reads silver plus the sources parquet.

Mandatory smoke test gate ([../run_runpod_nli_scaleup.command](../run_runpod_nli_scaleup.command)
lines 101-146): before launch, a single-cell NLI plus parquet write-and-read-back test runs
under the scale-up profile, prefetching that cell's sources from the parquet, and the script
refuses to launch unless it reports `WRITE_OK=True` and `KEYS_ALL_STRINGS=True`. This gate is
the single-cell write test, not [../scripts/smoke_nli.py](../scripts/smoke_nli.py), which is a
separate 50-cell local method-comparison smoke.

Launch **[discrepancy]**: the prompt (and this section's brief) describe a tmux-detached
launch, but the launchers do not use tmux to run the job; tmux is installed yet the run is
started with `nohup ... > nli_run.log 2>&1 &`, and a second `nohup` arms a hard-shutdown timer
(48 hours for scale-up, 6 for the pilot) as cost insurance
([../run_runpod_nli_scaleup.command](../run_runpod_nli_scaleup.command) lines 148-168). The
launcher then sleeps about 90 seconds, prints the live processes and the first log lines, and
aborts if the run process is not visible.

Monitoring is by tailing `nli_run.log` over SSH, with `nvidia-smi` used for GPU verification
during setup ([../run_runpod_nli_scaleup.command](../run_runpod_nli_scaleup.command) lines
42, 179-181). The driver checkpoints completed cells to `nli_{profile}.partial.parquet` every
200 cells and at the end of each run
([../scripts/run_nli_pilot.py](../scripts/run_nli_pilot.py) lines 104-141).

Recovery of partial results **[clarification]**: recovery is checkpoint-based, not purely
log-based. If the pod dies, the operator starts a new pod, re-runs the launcher (which
re-syncs code and silver), and copies the `nli_scaleup.partial.parquet` checkpoint back with
`scp` before relaunching; the driver resumes from it automatically, recomputing at most 200
cells. The log is how a hang is detected, but the recovery artefact is the partial parquet
([../run_runpod_nli_scaleup.command](../run_runpod_nli_scaleup.command) lines 182-186).

Download and termination: the finished gold parquet is pulled to the local machine with `scp`
over the SSH port, then the pod is terminated in the RunPod dashboard to stop charges
([../run_runpod_nli_scaleup.command](../run_runpod_nli_scaleup.command) lines 188-199).

## 6. Data loading and outputs

Gold outputs land under the profile's data tree. For scale-up, the per-response NLI table lands
in `data/scaleup/silver/` (`nli_scaleupv2.parquet`, and `nli_scaleup_cleaned.parquet` after
post-processing), while `data/scaleup/gold/` holds the four dimensional flat tables
(`flat_table1_responses`, `flat_table2_sentences`, `flat_table3_citations`,
`flat_table4_entailment_scores`, a star schema) plus the `cell_aggregates_scaleup.parquet`
metric mart. For the frozen pilot the per-call NLI table stays at
`data/pilot/gold/nli_pilot.parquet` with `cell_aggregates_pilot.parquet` beside it (README
section 1 and data tree).

Validation ([../run_runpod_nli_scaleup.command](../run_runpod_nli_scaleup.command) lines
193-197): the launcher prints the exact check to run, loading the gold parquet and grouping by
`(engine, run_index)`. Scale-up expects 9,992 rows (`EXPECTED_GOLD_ROWS`, line 24: 10,000 minus
the 8 absent Kimi runs, so Kimi has 249 query-cells and the others 250); the pilot expects about
1,500 rows, roughly 100 per (engine, run_index) cell. The local CV aggregation then
independently asserts the expected cell count, derived from the collected silver rather than the
theoretical grid, and prints a note for documented gaps, producing 1,249 cells for scale-up
([../scripts/aggregate_cv_pilot.py](../scripts/aggregate_cv_pilot.py) lines 76-95).

Download path back to local and commit: gold is `scp`-ed into the project's data tree (for
scale-up, `data/scaleup/gold/`). The `data/` tree is gitignored (`/data/**`) with narrow
re-includes for query registries and the pilot subtree
([../.gitignore](../.gitignore) lines 13-24), so scale-up data under `data/scaleup/**` is not
tracked by default and must be added with `git add -f` to commit it. The launchers do not push
to GitHub; committing is a manual step after validation.

## 7. Reproducibility notes

Config-driven profiles: [../thesis_config.py](../thesis_config.py) is the single source of
truth. `RUN_PROFILE` is read from the `THESIS_RUN_PROFILE` environment variable, defaulting to
the safe, cheap `pilot`; scale-up is a deliberate per-command act, never a committed edit, and
each profile sets `n_queries`, `k` and the paths derived from it, so one switch repoints the
whole pipeline ([../thesis_config.py](../thesis_config.py) lines 129-161). The build scripts
refuse to run under an implicit profile ([../scripts/build_silver.py](../scripts/build_silver.py)
lines 29-41).

Offline rebuild: silver and gold rebuild from the immutable bronze plus the sources parquet
with no network; collection is the only online stage. The `NLI_DEVICE` and
`SENTENCE_SEGMENTER` settings are pinned as part of the instrument, and the segmenter is
deliberately fixed to `regex` for pilot-to-scale-up comparability
([../thesis_config.py](../thesis_config.py) lines 234-244).

Checkpointing and provenance: the NLI driver checkpoints every 200 cells to a partial parquet
and removes it once the final parquet is written
([../scripts/run_nli_pilot.py](../scripts/run_nli_pilot.py) lines 104-141, 193-197). Run logs
are kept alongside the data: the scale-up NLI run log lives at
`data/scaleup/_ops/nli_scaleup_runv2.log` (moved out of the data layers in the 2026-06-26
realignment), and the on-pod run writes `nli_run.log`.

Known reproducibility gap **[VERIFY]**: per the README, the V2 cleaned gold parquets and the
threshold-sweep and entailment-distribution analyses were produced by eight `pp_step*`
post-processing scripts that are present in `scripts/` but described in the README as not yet
committed and slated for consolidation into a single `scripts/postprocess_scaleup.py`; until
that is done the cleaned gold artefacts cannot be regenerated end-to-end from committed code
(README "Reproducibility note — the post-processing script is uncommitted"). Confirm the commit
status against `git status` before relying on it.
