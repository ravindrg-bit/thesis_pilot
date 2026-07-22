# Decision Log — Mistral excluded from pilot analysis and scale-up

**Date:** 2026-06-06
**Decision:** Exclude the `mistral` engine (`mistral-medium-3-5`, Agents API) from
both the pilot analysis and the scale-up run.
**Status:** Final.
**Scope:** Applies to all downstream analysis (NLI attribution, CV aggregation,
results chapter) and to the scale-up collection. The adapter
(`src/adapters/mistral_adapter.py`) and the 82 collected pilot captures are
**retained** for reproducibility and provenance; they are simply not analysed.

---

## Facts

**(a) 218 × HTTP 429 rate-limit events observed.**
During pilot collection (`scripts/run_collection.py`), Mistral returned **218
HTTP 429** errors, every one carrying the body `web_search rate limit reached`.
All 218 are recorded in `data/pilot/collection_failures.jsonl` — they are the
*only* failures in the entire pilot (every other engine completed cleanly). The
Agents API enforces a hard ~1 request/second cap and the `web_search` connector
fans a single cell into several rapid back-end calls; bursts trip the limit even
with the dedicated adaptive throttle configured for Mistral
(`ENGINE_PACING["mistral"]`: 2 s base delay, AIMD back-off to 20 s, 15 s floor on
failure, ≤5 s jitter — `run_collection.py:69-72`).

**(b) 82-of-300 capture rate.**
The design called for 100 queries × 3 repeats = **300** Mistral cells. Only
**82** were captured (82 collected + 218 failed = 300 attempted). That is a
**27.3 %** completion rate, versus **100 %** for the other five engines
(chatgpt, claude, gemini, perplexity, kimi each at 300/300).

**(c) Fair cross-engine comparison is impossible at that capture rate.**
The protocol's metrics (AIS, PAWC) and its determinism control (per-cell
coefficient of variation over k repeats) both assume complete, balanced cells.
With only 82/300 captures, Mistral has many incomplete cells (fewer than k
repeats, or missing entirely), and the surviving 82 are **not missing at random**
— they are the calls that happened to slip through between rate-limit bursts,
biasing any per-engine estimate. Including Mistral would either contaminate the
balanced five-engine comparison or require unequal-n handling that undermines the
clean repeated-measures design.

**(d) Formal decision.**
Mistral is **excluded from both the pilot analysis and the scale-up**. This is
already reflected in the analysis layer (`scripts/run_nli_pilot.py`
`EXCLUDE_ENGINES = {"mistral"}`, which is why `pilot_nli_pilot.parquet` has 1,500
rows = 5 × 100 × 3, not 1,582). It must additionally be reflected in the
**collection** layer before scale-up: remove `mistral` from `ENGINES`/`ADAPTERS`
and from the scale-up profile so it is never called again (see the "Code changes
required before scale-up launch" section of
[pilot_llm_configurations.md](../pilot_llm_configurations.md)).

---

## Consequences

- Engine set for both pilot results and scale-up: **5 engines** — ChatGPT,
  Claude, Gemini, Kimi, Perplexity.
- `src/adapters/mistral_adapter.py` is **kept** (do not delete — pilot
  reproducibility depends on it) but is no longer wired into collection.
- `mistral` is intentionally absent from `ENGINE_PRICES` in `thesis_config.py`.
- The 82 pilot captures remain in `data/pilot/bronze/` and the silver tables for
  audit completeness; analysis code filters them out by engine key.

---

*Revision log: 2026-06-06 — entry created.*
