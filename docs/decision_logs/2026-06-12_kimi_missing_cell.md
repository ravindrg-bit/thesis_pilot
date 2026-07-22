# Decision log — Kimi missing cell in scaleup bronze (gb_3fcf760b1a2ea4f8)

**Date:** 2026-06-12
**Status:** Documented; analyses proceed with 1,249 cells (not 1,250).

## What is missing

All **8 runs** of a single (engine, query) cell are absent from `data/scaleup/bronze/`:

- **Engine:** kimi (Moonshot, kimi-k2.6)
- **Query:** `gb_3fcf760b1a2ea4f8` — *"How is Xi Jinping becoming the leader of China's
  armed forces so different from the US President being the Commander in Chief of the
  American Armed Forces?"*
- **Tags:** informational, complex, non-technical, law and government, question, comparison
- Scaleup bronze is therefore **9,992 files** (250 q × 8 k × 5 engines − 8), and kimi has
  **249** query-cells vs 250 for the other four engines.

No `collection_failures.jsonl` was preserved for the scaleup run, so the per-call error
records are not available. This gap was discovered during the pre-gold audit
(12 Jun 2026) by reconciling silver row counts against the registry grid.

## Most plausible cause

The failure pattern — **all 8 repeats of exactly one query, on the China-based provider,
where that query directly concerns Xi Jinping and PLA command authority** — is consistent
with **provider-side content filtering**, not with transient API failure (which would be
expected to scatter across queries/runs, as Mistral's 429s did in the pilot). This cannot
be proven without the original error payloads, but the systematic shape of the missingness
is itself evidence.

## Methodological implications

1. **Missingness is plausibly NOT random for kimi.** It is concentrated on politically
   sensitive (China) content. Cross-engine comparisons on such queries should note that
   kimi's sample is potentially censored on this content category.
2. **Shape expectations downstream:** silver = 9,992 rows; gold per-call = 9,992 rows;
   cell aggregates = **1,249** rows. `scripts/aggregate_cv_pilot.py` derives its expected
   cell count from the collected silver (not the theoretical grid) and prints a note when
   cells are absent — this log is the documentation that note refers to.
3. **No imputation.** The cell is simply absent; kimi means aggregate over 249 queries.

## Related

- `docs/decision_logs/2026-06-06_mistral_excluded.md` (precedent: engine-level exclusion)
- Region confound already documented for kimi in `thesis_config.py` ENGINES notes
  ("Chinese-web sources (region confound)") — this log adds the response-side analogue.
