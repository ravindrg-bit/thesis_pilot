"""
run_scaleup_preflight.py — Phase 2 of the scaleup pre-flight check.

Makes exactly 5 API calls (one per engine) for a single confirmed query, writes:
  data/scaleup_preflight/bronze/{engine}__gb_f31a6963837959ce__r1.json  (5 files)
  data/scaleup_preflight/silver/responses_preflight.parquet
  data/scaleup_preflight/silver/citations_preflight.parquet

Schema of both parquets is required to match the pilot silver exactly.
Run with: THESIS_RUN_PROFILE=scaleup python scripts/run_scaleup_preflight.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --- 1. Force scaleup profile BEFORE importing thesis_config -----------------
os.environ["THESIS_RUN_PROFILE"] = "scaleup"

from src.env import load_env
load_env()

import thesis_config as cfg
import pandas as pd

from src.adapters.openai_adapter import OpenAIAdapter
from src.adapters.claude_adapter import ClaudeAdapter
from src.adapters.gemini_adapter import GeminiAdapter
from src.adapters.perplexity_adapter import PerplexityAdapter
from src.adapters.kimi_adapter import KimiAdapter
from src.silver import normalise_record, build_responses_table, build_citations_table

# --- 2. Target query (confirmed in Phase 1) ----------------------------------
QUERY_ID   = "gb_f31a6963837959ce"
QUERY_TEXT = "the most innovative augmented reality advancements to apply at work today"
RUN_INDEX  = 1       # 1-based, matching pilot convention

# --- 3. Output paths — NEVER write to data/pilot/ or data/scaleup/ -----------
PREFLIGHT_ROOT   = ROOT / "data" / "scaleup_preflight"
PREFLIGHT_BRONZE = PREFLIGHT_ROOT / "bronze"
PREFLIGHT_SILVER = PREFLIGHT_ROOT / "silver"

# Redirect ALL cfg.BRONZE writes to the preflight tree (runtime patch, no file edit)
cfg.BRONZE = PREFLIGHT_BRONZE
PREFLIGHT_BRONZE.mkdir(parents=True, exist_ok=True)
PREFLIGHT_SILVER.mkdir(parents=True, exist_ok=True)

# --- 4. Adapter map (same order as ENGINES; mistral excluded) ----------------
ADAPTERS = {
    "chatgpt":    OpenAIAdapter,
    "claude":     ClaudeAdapter,
    "gemini":     GeminiAdapter,
    "kimi":       KimiAdapter,
    "perplexity": PerplexityAdapter,
}

# --- 5. Five API calls -------------------------------------------------------
print(f"RUN PROFILE : {cfg.RUN_PROFILE}")
print(f"QUERY ID    : {QUERY_ID}")
print(f"QUERY TEXT  : {QUERY_TEXT}")
print(f"BRONZE DIR  : {PREFLIGHT_BRONZE}")
print(f"SILVER DIR  : {PREFLIGHT_SILVER}")
print()

bronze_records = []
for engine, adapter_cls in ADAPTERS.items():
    print(f"[{engine}] Calling adapter.fetch() ...", flush=True)
    adapter = adapter_cls()
    rec = adapter.fetch(
        query_id=QUERY_ID,
        query_text=QUERY_TEXT,
        run_index=RUN_INDEX,
    )
    bronze_records.append(rec)
    bronze_path = PREFLIGHT_BRONZE / f"{engine}__{QUERY_ID}__r{RUN_INDEX}.json"
    print(f"[{engine}] Bronze written: {bronze_path.name}")

print()

# --- 6. Normalise to canonical records and build silver tables ---------------
canonicals = []
for rec in bronze_records:
    canonical = normalise_record(rec)
    canonical.run_profile = "scaleup"   # stamp must match profile
    canonicals.append(canonical)

responses_df = build_responses_table(canonicals)
citations_df  = build_citations_table(canonicals)

# --- 7. Enforce pilot column order and dtypes --------------------------------
RESPONSES_COLS = [
    "query_id", "engine", "run_index", "model_served", "timestamp_utc",
    "run_profile", "answer_text", "answer_char_len", "answer_word_count",
    "n_cited_sources", "n_unique_domains", "has_citations", "citation_provenance",
]
CITATIONS_COLS = [
    "query_id", "engine", "run_index", "position",
    "url", "url_canonical", "domain", "title", "provenance",
]

responses_df = responses_df[RESPONSES_COLS]
citations_df  = citations_df[CITATIONS_COLS]

# Cast dtypes to match pilot exactly
responses_df = responses_df.astype({
    "run_index":        "int64",
    "answer_char_len":  "int64",
    "answer_word_count":"int64",
    "n_cited_sources":  "int64",
    "n_unique_domains": "int64",
    "has_citations":    "bool",
})
citations_df = citations_df.astype({
    "run_index": "int64",
    "position":  "int64",
})

# --- 8. Write silver parquets ------------------------------------------------
resp_path = PREFLIGHT_SILVER / "responses_preflight.parquet"
cit_path  = PREFLIGHT_SILVER / "citations_preflight.parquet"
responses_df.to_parquet(resp_path, index=False)
citations_df.to_parquet(cit_path,  index=False)

# --- 9. Verification report --------------------------------------------------
pilot_resp = pd.read_parquet(ROOT / "data/pilot/silver/responses_pilot.parquet")
pilot_cit  = pd.read_parquet(ROOT / "data/pilot/silver/citations_pilot.parquet")

def schema_matches(df_a, df_b, label):
    a = dict(zip(df_a.columns, df_a.dtypes))
    b = dict(zip(df_b.columns, df_b.dtypes))
    mismatches = []
    for col in b:
        if col not in a:
            mismatches.append(f"  MISSING col: {col}")
        elif str(a[col]) != str(b[col]):
            mismatches.append(f"  DTYPE MISMATCH [{col}]: preflight={a[col]}  pilot={b[col]}")
    for col in a:
        if col not in b:
            mismatches.append(f"  EXTRA col (not in pilot): {col}")
    if mismatches:
        print(f"[FAIL] {label} schema differs:")
        for m in mismatches:
            print(m)
        return False
    if list(df_a.columns) != list(df_b.columns):
        print(f"[FAIL] {label} column ORDER differs")
        print(f"  preflight: {list(df_a.columns)}")
        print(f"  pilot    : {list(df_b.columns)}")
        return False
    return True

print("=" * 60)
print("VERIFICATION REPORT")
print("=" * 60)

checks = {}

# Files exist
checks["responses_preflight.parquet exists"] = resp_path.exists()
checks["citations_preflight.parquet exists"] = cit_path.exists()

# Schema
checks["responses schema matches pilot"] = schema_matches(responses_df, pilot_resp, "responses")
checks["citations schema matches pilot"]  = schema_matches(citations_df,  pilot_cit,  "citations")

# Row counts
checks["responses row count == 5"] = len(responses_df) == 5
checks["all 5 engines present"] = set(responses_df["engine"]) == {"chatgpt","claude","gemini","kimi","perplexity"}
checks["citations row count >= 5"] = len(citations_df) >= 5

# Bronze files
expected_bronze = [PREFLIGHT_BRONZE / f"{e}__{QUERY_ID}__r{RUN_INDEX}.json" for e in ADAPTERS]
checks["bronze JSONs written (one per engine)"] = all(p.exists() for p in expected_bronze)

# No files outside scaleup_preflight
writes_outside = False
for p in list(Path(ROOT / "data/pilot").rglob("*_preflight*")) + \
         list(Path(ROOT / "data/scaleup").rglob("*_preflight*")):
    writes_outside = True
    print(f"  [WARN] Unexpected file outside preflight tree: {p}")
checks["no files written outside data/scaleup_preflight/"] = not writes_outside

for label, ok in checks.items():
    mark = "[x]" if ok else "[ ]"
    print(f"  {mark} {label}")

print()
print("Per-engine citation counts:")
print(f"  {'engine':<12} {'preflight':>10}  {'pilot (same query)':>18}")
pilot_cit_q = pilot_cit[pilot_cit["query_id"] == QUERY_ID]
pilot_counts = pilot_cit_q.groupby("engine").size().to_dict()
PILOT_K = 3
for engine in sorted(ADAPTERS.keys()):
    pf_n  = len(citations_df[citations_df["engine"] == engine])
    p_n   = pilot_counts.get(engine, 0)
    per_run = round(p_n / PILOT_K, 1)
    flag  = " <<< ZERO CITATIONS — INVESTIGATE" if pf_n == 0 and p_n > 0 else ""
    print(f"  {engine:<12} {pf_n:>10}  {p_n:>10} total / {per_run:>5}/run{flag}")

print()
# Flag any engine with 0 citations in preflight but >0 in pilot
zero_engines = [
    e for e in ADAPTERS
    if len(citations_df[citations_df["engine"] == e]) == 0 and pilot_counts.get(e, 0) > 0
]
if zero_engines:
    print("!!! ZERO-CITATION ALERT !!!")
    for e in zero_engines:
        print(f"  {e} returned 0 citations in preflight but cited {pilot_counts[e]} times in pilot.")
    print("  Investigate before launching scale-up.")
else:
    print("All engines returned citations. No zero-citation alerts.")

print()
print("responses_preflight schema:")
print(responses_df.dtypes.to_string())
print()
print("citations_preflight schema:")
print(citations_df.dtypes.to_string())
