"""
thesis_config.py — SINGLE SOURCE OF TRUTH for the data-sourcing pilot.

MSc Business Analytics, Trinity College Dublin (BU7170).
Thesis: "Measuring In-Context Visibility and Verifiability Across Generative
Engines: A Reproducible Multi-Metric Protocol."

Every run constant lives here and nowhere else. Scripts import from this module;
they never redefine engines, paths, seeds, or model ids locally. This is the code
counterpart to Dissertation Context V3 (the governing single source of truth):
change a value here and the whole pipeline changes with it.

NOTE — TEST-GRADE PILOT config (10 queries, reduced repeats). Production values
from V3 are recorded in comments beside each pilot value.
"""

import os
from pathlib import Path

# --- 0. Project identity (provenance/headers only) --------------------------
THESIS_TITLE = (
    "Measuring In-Context Visibility and Verifiability Across Generative "
    "Engines: A Reproducible Multi-Metric Protocol"
)
RESEARCHER = "Ganenthra Ravindran"
MODULE = "BU7170 - MSc Business Analytics, Trinity College Dublin"

# --- 1. Paths (profile-independent) -----------------------------------------
ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
# The data tree (bronze/silver/gold/queries) is profile-specific -> see section 5.

# --- 2. Source query set: GEO-Bench (V3 primary; Aggarwal et al. 2024) -------
# From the feasibility test:
#   - MULTI-CONFIG dataset (configs: ['train','test']) -> use config/split handling.
#   - Record schema: query, tags, sources, sugg_idx.
#   - NO native id field -> query_id is DERIVED (see QUERY_ID_* below).
GEOBENCH_REPO = "GEO-Optim/geo-bench"
GEOBENCH_SPLIT = "test"          # fall back to first available config if absent
QUERY_FIELD = "query"
QUERY_ID_PREFIX = "gb"           # e.g. "gb_1a2b3c4d5e6f7a8b"
QUERY_ID_HASH_LEN = 16           # 64-bit hash of the query text -> stable, collision-safe id

# --- 3. Engines under test (six generative engines; Change Log 30 May 2026) ---
# Six engines spanning three regions (US / China / Europe). Strings VERIFIED against the
# providers' live docs on 30 May 2026 (sources recorded in docs/data_provenance.md).
#
# Two corrections from the Batch-1 placeholders:
#   - Perplexity has no "sonar-2"; current API tiers are sonar / sonar-pro / etc.
#   - "Gemini 3 Flash" (gemini-3-flash-preview) was renamed to gemini-3.5-flash.
#
# CITATIONS REQUIRE WEB GROUNDING. A plain call returns NO sources (empty PAWC/AIS).
# Each engine must be queried with its search tool enabled (see 'grounding').
#
# DETERMINISM: temperature=0 / fixed seed is NOT available on this model generation
# (GPT-5.5 is a reasoning model; Gemini 3 recommends default temperature). The
# determinism control is the repeated-measures design (k repeats) + the per-query
# coefficient of variation -- exactly V3 sec.7.2 (Atil et al. 2025; Yuan et al. 2025;
# Schulte et al. 2026). We fix what IS fixable (reasoning effort) and log the served
# model id on every call (V3 sec.7.4).
ENGINES = {
    # --- US engines ---
    "chatgpt": {
        "provider": "openai", "model": "gpt-5.5-2026-04-23", "api": "responses",
        "grounding": {"tool": "web_search"},
        "supports_temperature": False, "supports_seed": False,
        "env_key": "OPENAI_API_KEY", "region": "US",
        "notes": "Responses API + web_search; model decides to search; fixed reasoning_effort.",
        "docs": "https://platform.openai.com/docs/guides/tools-web-search",
    },
    "claude": {
        "provider": "anthropic", "model": "claude-sonnet-4-6", "api": "anthropic_messages",
        "grounding": {"tool": "web_search", "tool_version": "web_search_20250305"},
        "supports_temperature": True, "supports_seed": False,
        "env_key": "ANTHROPIC_API_KEY", "region": "US",
        "notes": "Messages API + web_search; citations always on (cited_text/title/url); model decides to search. Uses web_search_20250305 (standalone, structured citations); 20260209 dynamic-filtering version suppresses inline citations without code execution.",
        "docs": "https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool",
    },
    "gemini": {
        "provider": "google", "model": "gemini-3.5-flash", "api": "generate_content",
        "grounding": {"tool": "google_search"},
        "supports_temperature": False, "supports_seed": False,
        "env_key": "GEMINI_API_KEY", "region": "US",
        "notes": "generate_content + google_search; sources in groundingMetadata.groundingChunks (uri=redirect, title=domain).",
        "docs": "https://ai.google.dev/gemini-api/docs/google-search",
    },
    "perplexity": {
        "provider": "perplexity", "model": "sonar", "api": "openai_compatible",
        "base_url": "https://api.perplexity.ai",
        "grounding": {"inherent": True},
        "supports_temperature": True, "supports_seed": False,
        "env_key": "PERPLEXITY_API_KEY", "region": "US",
        "notes": "Inherently grounded; citations in search_results[] (title/url/date). Switch to sonar-pro for ~2x citations.",
        "docs": "https://docs.perplexity.ai",
    },
    # --- China engine ---
    "kimi": {
        "provider": "moonshot", "model": "kimi-k2.6", "api": "openai_compatible",
        "base_url": "https://api.moonshot.ai/v1",
        "grounding": {"builtin_function": "$web_search", "requires_tool_loop": True},
        "supports_temperature": True, "supports_seed": False,
        "env_key": "MOONSHOT_API_KEY", "region": "China",
        "notes": "Web search via $web_search builtin_function in a TOOL-CALL LOOP; cites sources. Chinese-web sources (region confound).",
        "docs": "https://platform.moonshot.ai/docs/guide/use-web-search",
    },
    # --- Europe engine (mistral) EXCLUDED from collection ---
    # Rationale: 218×HTTP 429 in pilot → 82/300 captured, no fair cross-engine
    # comparison possible. src/adapters/mistral_adapter.py retained for pilot
    # reproducibility. See docs/decision_logs/2026-06-06_mistral_excluded.md
}

# Engines active in collection and analysis (Mistral excluded — see decision log above).
# Use this set everywhere a script needs "active engines" to avoid repeating the exclusion.
_EXCLUDED_ENGINES = {"mistral"}
COLLECTION_ENGINES = {k: v for k, v in ENGINES.items() if k not in _EXCLUDED_ENGINES}

# Fixed reasoning/thinking effort -- the determinism knob that DOES exist here.
# Pilot uses 'low' for cost/latency; choose a representative level for the real run.
OPENAI_REASONING_EFFORT = "low"   # one of: none, low, medium, high, xhigh
GEMINI_THINKING_LEVEL = "low"     # one of: low, medium, high

# --- 4. Measurement model (V3 sec.7.2) --------------------------------------
# NOTE: temperature=0 / fixed seed is NOT settable on the current models (see sec.3);
# TEMPERATURE is retained only as a record of intent and may be ignored/rejected.
# K (repeats) moved to section 5 (consolidated under the run-profile switch).
TEMPERATURE = 0      # informational only on this model generation
SEED = 20260530      # seeds LOCAL randomness (e.g. query sampling in repro.py)

# --- 5. Run profile (single switch separating pilot from scale-up) ----------
# Flip RUN_PROFILE to repoint the ENTIRE pipeline at a separate data tree:
#   data/pilot/{bronze,silver,gold,queries}  vs  data/scaleup/{...}
# Pilot and scale-up therefore never share files. The profile is also stamped INTO
# every record (schema.run_profile), so separation holds by field as well as by path.
# Selected per-invocation via the THESIS_RUN_PROFILE env var; defaults to the safe/cheap
# "pilot". Scale-up is a DELIBERATE per-command act, never a committed edit, so the file
# is never left pointing at the wrong tree:
#   pilot   (default):  python scripts/<script>.py
#   scaleup (explicit): THESIS_RUN_PROFILE=scaleup python scripts/<script>.py
RUN_PROFILE = os.environ.get("THESIS_RUN_PROFILE", "pilot")

PILOT_K = 3                    # repeats per (query, engine) for the pilot; full k=8 reserved for scale-up
PRODUCTION_K = 8               # Schulte et al. (2026) 7-8 minimum; used for scale-up

PROFILES = {
    "pilot":         {"n_queries": 100,  "k": PILOT_K,      "registry_version": "v2"},
    "scaleup":       {"n_queries": 250,  "k": PRODUCTION_K, "registry_version": "v2"},
    "scaleup_smoke": {"n_queries": 5,    "k": 8,            "registry_version": "v2"},
}
assert RUN_PROFILE in PROFILES, f"unknown RUN_PROFILE={RUN_PROFILE!r}"
_p = PROFILES[RUN_PROFILE]
N_QUERIES = _p["n_queries"]
K = _p["k"]
REGISTRY_VERSION = _p["registry_version"]

# Profiled data tree (depends on RUN_PROFILE above)
DATA = ROOT / "data" / RUN_PROFILE
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
QUERY_DIR = DATA / "queries"
QUERY_REGISTRY = QUERY_DIR / f"{RUN_PROFILE}_queries_{REGISTRY_VERSION}.parquet"


def profile_summary() -> str:
    """One-line banner for entry-point scripts to print before running."""
    return f"RUN PROFILE: {RUN_PROFILE.upper()}  (N={N_QUERIES}, K={K}, root={DATA})"

# --- 6. Run metadata --------------------------------------------------------
# A per-execution stamp partitions bronze output so re-runs never overwrite raw.
RUN_ID_FORMAT = "%Y%m%dT%H%M%SZ"


def resolved_summary() -> dict:
    """Plain dict of the active config for printing (never includes secrets)."""
    return {
        "thesis_title": THESIS_TITLE,
        "run_profile": RUN_PROFILE,
        "root": str(ROOT),
        "bronze": str(BRONZE),
        "silver": str(SILVER),
        "gold": str(GOLD),
        "geobench_repo": GEOBENCH_REPO,
        "engines": {k: v["model"] for k, v in ENGINES.items()},
        "n_queries": N_QUERIES,
        "k": K,
        "pilot_k": PILOT_K,
        "production_k": PRODUCTION_K,
        "temperature": TEMPERATURE,
        "seed": SEED,
    }


# --- 8. LLM-as-judge: support attribution for AIS (RQ4) + support-based PAWC (RQ1) ---
# ONE shared judge call per sentence feeds BOTH metrics (document: not fully independent):
#   AIS  = supported_sentences / total_sentences; supported = >=1 cited source supports it
#   PAWC(source) = sum over supported sentences of (word_count x position_weight)
# Attribution by SUPPORT (judge reads the source), NOT native markers -> uniform across all
# engines incl. perplexity/kimi. Position weight = linear decay (first=1.0, last=1/n).
# Lineage: Aggarwal 2024 (PAWC), Rashkin 2023 (AIS), Gao 2023 (ALCE), Luttgenau 2025 (decay).
# DEVIATION (support-based vs Aggarwal's declared attribution) - pending supervisor sign-off.
JUDGE_MODEL = "claude-haiku-4-5-20251001"   # Claude Haiku 4.5 - VERIFY id at docs.claude.com
JUDGE_SOURCE_CHAR_LIMIT = 4000              # snippet per source fed to judge (cost knob)

# Approx Haiku 4.5 prices for the cost extrapolation ONLY (VERIFY at anthropic.com/pricing;
# update these two numbers if wrong - they drive the projection, not the billing):
HAIKU_INPUT_USD_PER_MTOK = 1.00             # $ per 1M input tokens   <-- VERIFY
HAIKU_OUTPUT_USD_PER_MTOK = 5.00            # $ per 1M output tokens  <-- VERIFY


# --- 9. Pricing for the cost ledger (USD per 1M tokens) -- VERIFY ALL at provider pages ---
# Per-engine input/output token prices. Some engines also bill separate web-search fees;
# capture those as a flat per-search or per-call adder where known (else leave 0 and note).
ENGINE_PRICES = {
    "chatgpt":    {"in": 1.25,  "out": 10.00, "search_per_call": 0.0},   # VERIFY (GPT-5.5)
    "claude":     {"in": 3.00,  "out": 15.00, "search_per_call": 0.01},  # VERIFY (Sonnet 4.6 + web_search ~$10/1k)
    "gemini":     {"in": 0.30,  "out": 2.50,  "search_per_call": 0.0},   # VERIFY (3.5 Flash; grounding fee separate)
    "perplexity": {"in": 1.00,  "out": 1.00,  "search_per_call": 0.0},   # VERIFY (Sonar)
    "kimi":       {"in": 0.60,  "out": 2.50,  "search_per_call": 0.0},   # VERIFY (k2.6 / Moonshot)
}
JUDGE_PRICES = {"in": 1.00, "out": 5.00}   # Haiku 4.5 -- VERIFY at anthropic.com/pricing
# (JUDGE_PRICES should match HAIKU_INPUT/OUTPUT_USD_PER_MTOK from section 8.)


# --- 10. NLI attribution (Option 3 - canonical AIS method) -------------------
# Replaces the LLM judge's support-decision step with a specialist NLI classifier
# (entailment-equals-supports). PAWC + AIS computation unchanged downstream.
# Lineage: Rashkin et al. 2023 (AIS framework), Gao et al. 2023 (ALCE), Honovich
# et al. 2022 (AutoAIS / TRUE). DeBERTa-v3-large-MNLI-FEVER-ANLI is a standard
# attribution-research checkpoint; chunked-source aggregation per ALCE.
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
NLI_SOURCE_CHUNK_TOKENS = 200          # split each source into ~200-token windows
NLI_SOURCE_CHUNK_STRIDE = 50           # 50-token overlap so claims don't fall between chunks
NLI_ENTAILMENT_THRESHOLD = 0.5         # softmax prob over entailment label
NLI_DEVICE = os.environ.get("NLI_DEVICE", "mps")  # override: NLI_DEVICE=cuda on RunPod, NLI_DEVICE=cpu to force CPU

# Sentence segmentation backend — PART OF THE MEASUREMENT INSTRUMENT, pinned explicitly.
# N (sentence count) is the AIS denominator and drives every PAWC position weight, so the
# splitter must be identical across environments and profiles. VERIFIED 12 Jun 2026: pilot
# gold n_sentences matches the regex splitter on 1500/1500 cells (punkt only 466/1500,
# all coincidental) — the pilot pod lacked NLTK punkt data, so nltk.sent_tokenize never
# ran and the silent except-fallback selected regex. "regex" is therefore the instrument
# of record; switching to "punkt" would change N and break pilot<->scaleup comparability.
SENTENCE_SEGMENTER = os.environ.get("SENTENCE_SEGMENTER", "regex")   # "regex" | "punkt"
assert SENTENCE_SEGMENTER in ("regex", "punkt"), f"unknown SENTENCE_SEGMENTER={SENTENCE_SEGMENTER!r}"
