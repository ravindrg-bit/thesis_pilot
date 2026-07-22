# Pilot LLM Configurations Reference

**Purpose.** A single, authoritative record of every LLM setting used in the
pilot, so the scale-up (250 queries × k=8) can be launched without reconstructing
decisions from code archaeology. **This document is the source of truth.** Where
the code currently disagrees with it (notably `thesis_config.py`), the code is
wrong and must be reconciled — see the conflicts section below and the closing
"Code changes required before scale-up launch" checklist.

**Scope.** The five analysed engines (ChatGPT, Claude, Gemini, Kimi, Perplexity),
their request parameters, citation-extraction paths, measured pilot behaviour, and
the shared attribution infrastructure. Mistral is excluded — see
[decision_logs/2026-06-06_mistral_excluded.md](decision_logs/2026-06-06_mistral_excluded.md).
Audience: future me, about to start the scale-up.

**Last updated:** 2026-06-06.

**Provenance.** Every value here is sourced from code (file:line cited) or measured
from the pilot bronze/silver captures. Token means and search/citation rates are
computed across all pilot runs (300 per analysed engine).

---

## ⚠ Conflicts to resolve before scale-up

These three are real and will silently corrupt the scale-up if not fixed. They are
listed first on purpose. Full fixes are in the closing checklist.

1. **`thesis_config.py` encodes the wrong scale-up configuration.**
   `PROFILES["scaleup"]` is currently `{"n_queries": 1000, "k": 8,
   "registry_version": "v1"}` and the engine set still includes Mistral
   ([thesis_config.py:146](../thesis_config.py)). Running
   `THESIS_RUN_PROFILE=scaleup` **today** would launch **1000 queries × k=8 × 6
   engines** reading `scaleup_queries_v1.parquet` (the "v1" version is the
   10-query proof-of-concept schema). The **canonical target is 250 × k=8 × 5
   engines** (this document). The code must be changed to match.

2. **ChatGPT `include` drift between pilot and current code.**
   The current adapter sends `include: ["web_search_call.action.sources"]`
   ([openai_adapter.py:42](../src/adapters/openai_adapter.py)), but **299 of 300**
   pilot ChatGPT captures were made *without* it (only 1 carries it). If scale-up
   runs the current code unchanged, ChatGPT bronze will differ from the pilot
   (the CONSULTED-sources list becomes populated). This does **not** change
   citation extraction, but it is a pilot↔scale-up difference that must be a
   deliberate, documented decision. Default recommendation: **remove it** to keep
   scale-up directly comparable to the pilot.

3. **Mistral is excluded in analysis but still live in collection code.**
   `run_nli_pilot.py` already drops Mistral (`EXCLUDE_ENGINES = {"mistral"}`), but
   it remains in `ENGINES`/`ADAPTERS` and in the scale-up profile, and
   `run_collection.py` iterates all engines by default. It will be called again at
   scale-up unless removed. Rationale for exclusion:
   [decision_logs/2026-06-06_mistral_excluded.md](decision_logs/2026-06-06_mistral_excluded.md).

*(Minor, non-blocking: the `claude_adapter.py` docstring says
`web_search_20260209`, but the config and the actual transmitted value are
`web_search_20250305`. The comment is wrong; behaviour is correct. Fix listed in
the closing checklist.)*

---

## Global configuration

**Engine set — 5 engines.** ChatGPT, Claude, Gemini, Kimi, Perplexity. **Mistral
excluded** (218 × HTTP 429, 82/300 captured → no fair comparison); formal
rationale in
[decision_logs/2026-06-06_mistral_excluded.md](decision_logs/2026-06-06_mistral_excluded.md).

**Scale-up specification (canonical target):**
**250 queries × k=8 × 5 engines = 10,000 calls.** (Pilot was 100 × k=3 × 5 =
1,500 analysed cells; the pilot also collected an additional 82 Mistral cells, now
excluded.) See the dedicated scale-up section below for the prominent flag that the
code does not yet encode this.

**Sampling regime — all engines at provider defaults.**
No `temperature` is transmitted by any adapter (verified: 0 of 1,612 captures
carry `temperature` in `request_params`). `TEMPERATURE = 0` in
[thesis_config.py:126](../thesis_config.py) is **logged only as a record of
intent and is never sent** — the intent (0) and the actuality (unset → provider
default) differ. Where the default is observable, OpenAI echoes
`temperature = 1.0`, `top_p = 0.98` in all ChatGPT responses; the other four
providers do not echo the parameter. **Recommendation for scale-up: keep the same
regime (transmit nothing).**

**Determinism control.** Reproducibility rests on the **repeated-measures design**
(k repeats per cell + per-cell coefficient of variation), **not** on temperature
suppression. Scale-up raises k from 3 to 8 for exactly this reason
(`PRODUCTION_K = 8`, Schulte et al. 2026 minimum).

**Shared infrastructure (identical across engines; must not drift at scale-up):**

| Component | Setting | Source |
|---|---|---|
| Source fetch | `requests.get(timeout=20)`, UA `"Mozilla/5.0 (thesis-research; PAWC/AIS attribution)"`, `BeautifulSoup(html.parser)` | [attribution.py:25-50](../src/attribution.py) |
| HTML parser lib | `beautifulsoup4==4.14.3` | requirements.lock |
| Sentence segmentation | `nltk.sent_tokenize` | [attribution.py:28-34](../src/attribution.py) |
| NLTK | `nltk==3.9.4` | requirements.lock |
| NLI model | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | [thesis_config.py:229](../thesis_config.py) |
| NLI chunking / threshold / device | chunk 200 tok / stride 50 / **τ=0.5** / device `mps` | [thesis_config.py:230-233](../thesis_config.py) |
| ML stack | `transformers==5.9.0`, `torch==2.12.0`, `tokenizers==0.22.2` | requirements.lock |

> Note: `NLI_DEVICE = "mps"` (Apple GPU) is the local pilot setting; the RunPod
> launcher switches it to `cuda` for GPU runs. This is environment-specific, not a
> model-behaviour change.

---

## Per-engine configuration

Order: ChatGPT → Claude → Gemini → Kimi → Perplexity. All token means and
search/citation rates are measured across the 300 pilot runs for each engine
(100 queries × 3 repeats).

### ChatGPT

- **Model identifier:** requested `gpt-5.5-2026-04-23`; **served**
  `gpt-5.5-2026-04-23` (confirmed 300/300). No drift.
- **API / SDK:** OpenAI **Responses API** / `openai==2.38.0`.
- **Environment variable:** `OPENAI_API_KEY`.
- **Request parameters actually sent** (from bronze):
  `model`, `tools: [{"type": "web_search"}]`, `reasoning: {"effort": "low"}`
  (`OPENAI_REASONING_EFFORT`, [thesis_config.py:119](../thesis_config.py)).
- **⚠ Parameter in config/code but NOT transmitted in the pilot:** the current
  adapter code adds `include: ["web_search_call.action.sources"]`
  ([openai_adapter.py:42](../src/adapters/openai_adapter.py)), but **299/300**
  pilot captures lack it (1/300 has it). See conflict #2.
- **Web-search behaviour:** `web_search` tool enabled; **the model decides**
  whether to search. Natural-routing default — **not** forced (`tool_choice` is
  not set; the forced variant is a separate diagnostic adapter, see below).
- **Citation extraction path:** walk `raw_response.output[]`; for items where
  `type == "message"`, walk `content[].annotations[]`; collect those where
  `type == "url_citation"` → `.url`, `.title`. Dedupe by URL; 1-based position by
  first appearance. Parse by `.type`, never by array position.
  [openai_adapter.py:89-106](../src/adapters/openai_adapter.py).
- **Measured search-rate:** **93 / 300 runs (31.0%)** invoked `web_search`
  (output contains a `web_search_call` item).
- **Measured citation-rate:** **83 / 300 runs (27.7%)** returned ≥1 citation.
  Mean cited sources/run: **0.81**. (The 31.0% vs 27.7% gap is real: ChatGPT
  sometimes searches yet cites nothing.)
- **Pilot token usage:** mean **6,004 in / 462 out**.
- **Cost per call:** ≈ **$0.0121** (6,004 × \$1.25/Mtok + 462 × \$10.00/Mtok).
  **VERIFY** prices ([thesis_config.py:213](../thesis_config.py)).
- **Known quirks:** structural **zero-citation floor** — ~72% of runs cite
  nothing because the model answers from parametric memory. This motivated the
  forced-search diagnostic (below).

### Claude

- **Model identifier:** requested `claude-sonnet-4-6`; **served**
  `claude-sonnet-4-6` (confirmed 300/300). No drift.
- **API / SDK:** Anthropic **Messages API** / `anthropic==0.105.2`.
- **Environment variable:** `ANTHROPIC_API_KEY`.
- **Request parameters actually sent** (from bronze):
  `model`, `max_tokens: 2048`, `tools: [{"type": "web_search_20250305", "name":
  "web_search", "max_uses": 5}]`. `max_tokens` is **required** by the Messages
  API. ([claude_adapter.py:28,42-51](../src/adapters/claude_adapter.py))
- **Web-search behaviour:** `web_search` server tool (version
  `web_search_20250305`, ≤5 uses); **the model decides** whether to search.
- **Citation extraction path:** walk `raw_response.content[]`; for blocks where
  `type == "text"`, walk `citations[]`; collect those where
  `type == "web_search_result_location"` → `.url`, `.title`. Dedupe by URL.
  [claude_adapter.py:89-103](../src/adapters/claude_adapter.py).
- **Measured search-rate:** **226 / 300 runs (75.3%)** (content carries a
  `server_tool_use` / `web_search_tool_result` block).
- **Measured citation-rate:** **223 / 300 runs (74.3%)**; mean cited
  sources/run: **4.92**.
- **Pilot token usage:** mean **14,244 in / 1,035 out** (high input — search
  results are fed back into context). `usage` also exposes `server_tool_use` and
  cache-token fields.
- **Cost per call:** ≈ **$0.0683** (14,244 × \$3.00/Mtok + 1,035 × \$15.00/Mtok =
  \$0.0583, **plus** ~\$0.01 web-search fee). **VERIFY** prices
  ([thesis_config.py:214](../thesis_config.py)).
- **Known quirks:** citations always carry `cited_text`/`title`/`url` when
  present; docstring version string is stale (`20260209`) but transmitted value is
  `20250305` — fix queued.

### Gemini

- **Model identifier:** requested `gemini-3.5-flash`; **served**
  `gemini-3.5-flash` (confirmed 300/300; read from `response.model_version`). No
  drift.
- **API / SDK:** Google **generate_content** / `google-genai==2.7.0`.
- **Environment variable:** `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).
- **Request parameters actually sent** (from bronze): `model`,
  `tools: ["google_search"]` (constructed as
  `GenerateContentConfig(tools=[Tool(google_search=GoogleSearch())])`).
- **⚠ Parameter deliberately NOT set:** `thinking_level` is left at the model
  default (the SDK field was still settling for Gemini 3 at pilot time —
  [gemini_adapter.py:50](../src/adapters/gemini_adapter.py)).
- **Web-search behaviour:** `google_search` grounding enabled; **the model
  decides** whether to search.
- **Citation extraction path:**
  `raw_response.candidates[0].grounding_metadata.grounding_chunks[]`; each chunk's
  `.web` → `.uri`, `.title`, `.domain`. **`uri` is a `vertexaisearch.cloud.google.com`
  redirect, not the publisher URL; `title` is the publisher domain, not a page
  title** → stored in the canonical `domain` field. Tolerates snake_case and
  camelCase. [gemini_adapter.py:88-103](../src/adapters/gemini_adapter.py).
- **Measured search-rate:** **211 / 300 runs (70.3%)**
  (`grounding_metadata.web_search_queries` non-empty).
- **Measured citation-rate:** **211 / 300 runs (70.3%)**; mean cited
  sources/run: **7.02**. (Search-rate == citation-rate: when Gemini grounds, it
  always returns grounding chunks.)
- **Pilot token usage:** mean **11 in / 931 out** (candidates), **plus ~1,295
  mean "thoughts" tokens**; prompt/grounding tokens are billed separately via
  `tool_use_prompt_token_count`, so the recorded `prompt_token_count` is tiny.
- **Cost per call:** ≈ **$0.0023** (11 × \$0.30/Mtok + 931 × \$2.50/Mtok).
  **Understated** — excludes ~1,295 thoughts tokens and the separate grounding
  fee, neither captured in pilot pricing. **VERIFY** prices
  ([thesis_config.py:215](../thesis_config.py)).
- **Known quirks:** redirect URIs + domain-as-title (above); resolving redirects
  to true publisher URLs is a documented **scale-up step**, not done in the pilot.

### Kimi

- **Model identifier:** requested `kimi-k2.6`; **served** `kimi-k2.6` (confirmed
  300/300). No drift.
- **API / SDK:** Moonshot **OpenAI-compatible** endpoint
  (`base_url=https://api.moonshot.ai/v1`) / `openai==2.38.0`.
- **Environment variable:** `MOONSHOT_API_KEY`.
- **Request parameters actually sent** (from bronze): `model`,
  `tools: ["$web_search"]`, `thinking: "disabled"`,
  `citation_mode: "prompt_elicited"`, `system_prompt: <citation prompt>`,
  `max_tool_rounds: 5` (and `max_tokens: 4096` in code). Thinking mode **must** be
  disabled for the tool to work. ([kimi_adapter.py:30-39,63-124](../src/adapters/kimi_adapter.py))
- **Web-search behaviour:** `$web_search` builtin function run as a **tool-call
  loop** (≤5 rounds): on `finish_reason == "tool_calls"`, echo the tool arguments
  back as a `role="tool"` message and repeat.
- **Citation extraction path:** **no structured citation channel.** URLs are
  regex-parsed (`https?://…`) from the answer text, which the system prompt forces
  into a numbered "Sources:" list. Every source is flagged
  `provenance="self_reported_prompt_elicited"` (not provider-certified).
  [kimi_adapter.py:33-41,132-147](../src/adapters/kimi_adapter.py).
- **Measured search-rate:** **285 / 300 runs (95.0%)** (`n_search_rounds > 0`).
- **Measured citation-rate:** **294 / 300 runs (98.0%)**; mean cited
  sources/run: **9.35**. (Citation-rate > search-rate because URLs are
  prompt-elicited from the text and can appear even on runs where no tool round
  was detected.)
- **Pilot token usage:** mean **8,251 in / 919 out**.
- **Cost per call:** ≈ **$0.0072** (8,251 × \$0.60/Mtok + 919 × \$2.50/Mtok).
  **VERIFY** prices ([thesis_config.py:217](../thesis_config.py)).
- **Known quirks:** self-reported (prompt-elicited) citations — a methodological
  difference handled uniformly by computing AIS at cited-source-set granularity
  across all engines (see `methodology_notes.md`). Chinese-web sources introduce a
  region confound. The full conversation + system prompt are captured to bronze
  for auditability.

### Perplexity

- **Model identifier:** requested `sonar`; **served** `sonar` (confirmed
  300/300). No drift.
- **API / SDK:** Perplexity **OpenAI-compatible** endpoint
  (`base_url=https://api.perplexity.ai`) / `openai==2.38.0`.
- **Environment variable:** `PERPLEXITY_API_KEY`.
- **Request parameters actually sent** (from bronze): `model`, `messages`.
  **No tools and no sampling parameters** — Sonar is inherently grounded.
  ([perplexity_adapter.py:35-42](../src/adapters/perplexity_adapter.py))
- **Web-search behaviour:** **always grounded** — every answer carries sources;
  there is no tool to enable and no model decision.
- **Citation extraction path:** `raw_response.search_results[]` →
  `.url`, `.title` (fallback: flat `raw_response.citations[]` of URL strings).
  Dedupe by URL. [perplexity_adapter.py:77-98](../src/adapters/perplexity_adapter.py).
- **Measured search-rate:** **300 / 300 runs (100.0%)** — inherent grounding.
- **Measured citation-rate:** **300 / 300 runs (100.0%)**; mean cited
  sources/run: **7.11**.
- **Pilot token usage:** mean **12 in / 248 out**.
- **Cost per call:** **$0.00526** (API-reported `usage.cost`, mean over the 300
  pilot runs). **Perplexity bills per-call, not per-token; the token math
  understates its true cost**, so the API-reported figure is used. **VERIFY**
  current pricing ([thesis_config.py:216](../thesis_config.py)).
- **Known quirks:** answer text carries inline `[n]` markers; the canonical
  citations come from `search_results[]`, not the markers. The SDK's typed parsing
  drops Perplexity's non-standard top-level fields, so the adapter re-merges them
  from `model_extra` ([perplexity_adapter.py:45-49](../src/adapters/perplexity_adapter.py)).

---

## Diagnostic / sensitivity-analysis variants

**Forced-search adapter** — `src/adapters/openai_forced_search_adapter.py`.

- **What it does:** subclasses `OpenAIAdapter`; the **only** behavioural change is
  adding `tool_choice: {"type": "web_search"}`, forcing a web search on every call
  instead of letting the model decide. Same model
  (`cfg.ENGINES["chatgpt"]` → `gpt-5.5-2026-04-23`), same reasoning effort, same
  `include`. `normalise()` is inherited verbatim (identical citation parsing).
- **What differs from the default ChatGPT adapter:** `tool_choice` is set
  (default mode leaves it unset); bronze is written under engine tag
  `openai_forced` to a separate tree so it never collides with `chatgpt__*`
  captures.
- **Scope:** first **10** v2 queries × k=3 = **30 captures** only. Purpose: probe
  ChatGPT's zero-citation floor (forcing lifts coverage but degrades quality — a
  sensitivity probe, N=30, not a primary result).
- **Propagation to scale-up:** **None.** It is not registered in `ENGINES`, not
  called by `run_collection.py`, and is invoked only by its own test script. It
  will **not** run at scale-up unless deliberately re-run.

---

## Scale-up specification (250 × k=8)

> **⚠ FLAG — read first.** The canonical target below is **250 × k=8 × 5
> engines**. `thesis_config.py` currently encodes a **different** configuration
> (`PROFILES["scaleup"] = {"n_queries": 1000, "k": 8, "registry_version": "v1"}`,
> Mistral still present). **This document is the source of truth; the code is
> wrong and must be reconciled** before launch — see "Code changes required before
> scale-up launch" below.

**Canonical target:** 250 queries × k=8 repeats × 5 engines = **10,000 calls**.

**Per-engine model identifiers** (confirm each is still live before bulk run;
run one test call per engine):

| Engine | Pilot model (= scale-up target) | Deprecation risk |
|---|---|---|
| ChatGPT | `gpt-5.5-2026-04-23` | Dated snapshot id — confirm still served; OpenAI rotates snapshots |
| Claude | `claude-sonnet-4-6` | Low; confirm alias still maps |
| Gemini | `gemini-3.5-flash` | Was renamed once (3-flash-preview → 3.5-flash); reconfirm |
| Kimi | `kimi-k2.6` | Confirm Moonshot still serves k2.6 |
| Perplexity | `sonar` | Stable tier name; low risk |

**k=3 → k=8:** adds 5 extra runs per (query, engine) cell. Total
250 × 8 × 5 = **10,000 calls** (vs 1,500 analysed pilot cells).

**Query set:** the scale-up registry **does not yet exist**. It must be built as a
frozen 250-query parquet at
`data/scaleup/queries/scaleup_queries_<version>.parquet` (+ manifest + preview),
with a **new selection seed** documented in the manifest (the pilot v2 seed
`20260530` must not be reused). The config's current `registry_version: "v1"`
pointer is wrong (v1 is the 10-query proof-of-concept). **This is a hard blocker:
scale-up cannot launch without the registry.** Whether the pilot's 100 queries are
included in the 250 is an open decision (see Open questions).

**No other configuration changes from pilot.** Beyond `k` (3→8) and query count
(100→250), every setting in this document stays identical. Any further change
requires an explicit rationale row here:

| Changed item | Rationale | Status |
|---|---|---|
| ChatGPT `include` param | Drift between pilot (absent) and current code (present) | **Decision required** — default: remove to stay pilot-comparable (conflict #2) |
| Mistral removal from collection | 218 × 429, 82/300 captured | Decided — [decision log](decision_logs/2026-06-06_mistral_excluded.md) |
| `NLI_DEVICE` mps → cuda | RunPod GPU run, not a model change | Environment-only; acceptable |

---

## Anti-mistake checklist

Pre-flight items to tick line by line before launching the scale-up:

- [ ] Each adapter's model identifier matches this document; **test one call per
      engine** before the bulk run.
- [ ] `thesis_config.py` `PROFILES["scaleup"]` matches the **250 × k=8 × 5**
      specification documented here.
- [ ] The 250-query scale-up registry parquet **exists and is frozen** (with a new
      documented seed).
- [ ] `tool_choice` settings unchanged from pilot — ChatGPT in natural-routing
      default mode, **NOT** forced.
- [ ] Citation-extraction logic in every adapter unchanged from pilot and matches
      the field paths documented here.
- [ ] BeautifulSoup fetching code unchanged from pilot (timeout 20 s, same UA,
      `html.parser`).
- [ ] DeBERTa checkpoint and **τ=0.5** threshold unchanged
      (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, chunk 200/stride 50).
- [ ] `requirements.lock` used; no library version drift since pilot
      (`openai==2.38.0`, `anthropic==0.105.2`, `google-genai==2.7.0`,
      `transformers==5.9.0`, `torch==2.12.0`).
- [ ] Mistral adapter removed and **not** reintroduced in `ENGINES` or
      `run_collection.py`.
- [ ] `k=8` set in the scale-up profile — **not** k=3 (pilot) or k=7 (earlier draft).
- [ ] Bronze writes to `data/scaleup/bronze/`, **NOT** `data/pilot/bronze/`
      (verify via the profile banner before the run).
- [ ] ChatGPT `include` parameter decision documented **and** implemented
      (kept or removed — conflict #2).
- [ ] **Current API pricing verified against the `ENGINE_PRICES` constants**
      ([thesis_config.py:212-218](../thesis_config.py)) — every price in this doc
      carries a VERIFY tag for this reason.
- [ ] `tmux`/`screen` session used on RunPod so a disconnection does not kill the
      collection run.

---

## Code changes required before scale-up launch

Exactly five code changes. Each maps to a specific file (and line range where
known). **This document is doc-only; these edits are tracked here but not yet
applied.**

- [ ] **`thesis_config.py` → `PROFILES["scaleup"]`
      ([thesis_config.py:146](../thesis_config.py)):** change `n_queries` from
      `1000` to `250`; change the engine set from 6 to 5 (remove Mistral); change
      `registry_version` to point at the new 250-query scale-up registry file.
- [ ] **Build the 250-query scale-up registry parquet** at
      `data/scaleup/queries/scaleup_queries_<version>.parquet` (+ manifest +
      preview) with a new documented seed. It does not yet exist. **Hard blocker —
      scale-up cannot launch without it.**
- [ ] **`src/adapters/openai_adapter.py:42`** — decide and document whether
      `include: ["web_search_call.action.sources"]` **stays** (intentional
      scale-up improvement, accept the pilot↔scale-up difference) or is **removed**
      (preserve pilot-identical behaviour). **Default recommendation: REMOVE it**
      so scale-up is directly comparable to the pilot. Record the decision here
      whichever way it goes.
- [ ] **Remove Mistral from the collection wiring** so it is never called again:
      the `ADAPTERS` dict in `scripts/run_collection.py:44-51` **and** the
      `ENGINES` registry in `thesis_config.py:107-114` (the scale-up profile engine
      set is also covered by code-change #1). Keep `src/adapters/mistral_adapter.py`
      and the 82 pilot captures for reproducibility
      (per the [decision log](decision_logs/2026-06-06_mistral_excluded.md)).
- [ ] **`src/adapters/claude_adapter.py` docstring** — correct
      `web_search_20260209` to `web_search_20250305` so the comment matches the
      actual transmitted behaviour.

---

## Open questions

Resolve before launch:

1. **Query overlap:** does the 250-query scale-up registry **include** the pilot's
   100 queries, or is it a fresh independent draw? Affects the selection seed and
   any pilot-vs-scaleup comparison.
2. **ChatGPT `include`:** keep or remove (conflict #2 / code-change #3)? Decision
   pending; default recommendation is remove.
3. **Forced-search probe at scale:** re-run the diagnostic at scale-up volume, or
   leave it as a pilot-only N=30 probe? Currently undecided; no propagation by
   default.
4. **RunPod auto-shutdown:** the existing launcher has a 6-hour hard shutdown sized
   for 1,500 cells; 10,000 cells (8,750 collection + NLI) will need a re-estimated
   timeout before launch.
5. **Pricing:** confirm current provider list prices against `ENGINE_PRICES`
   (all marked VERIFY) and Perplexity's per-call billing before any cost
   reporting.

---

*Revision log: 2026-06-06 — initial version (pilot configuration captured from
code + bronze; scale-up target and pre-launch checklists defined). Add future
revisions below.*
