"""
test_gemini_adapter.py — prove the Gemini adapter on ONE pilot query (v2 registry).
Like OpenAI/Claude, Gemini decides whether to search, so pick a groundable row via
--query-index. From the project root:  python scripts/test_gemini_adapter.py --query-index N
Costs a few cents (one grounded call).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env import load_env
load_env()

import pandas as pd

import thesis_config as cfg
from src.adapters.gemini_adapter import GeminiAdapter

REGISTRY = cfg.QUERY_REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-index", type=int, default=0)
    args = ap.parse_args()

    if not REGISTRY.exists():
        print(f"[FAIL] frozen registry not found at {REGISTRY}. Run build_pilot_queries.py first.")
        sys.exit(1)

    df = pd.read_parquet(REGISTRY)
    qi = args.query_index
    if qi < 0 or qi >= len(df):
        print(f"[FAIL] --query-index {qi} out of range (0..{len(df)-1}).")
        sys.exit(1)
    row = df.iloc[qi]
    qid, qtext = row["query_id"], row["query_text"]

    print("=" * 64)
    print("BATCH 5 - GEMINI ADAPTER, SINGLE-QUERY TEST (v2 registry)")
    print("=" * 64)
    print(f"registry : {REGISTRY.name}")
    print(f"index    : {qi}")
    print(f"query_id : {qid}")
    print(f"query    : {qtext[:120]}")
    print("Calling gemini-3.5-flash with google_search grounding (a few cents)...")

    adapter = GeminiAdapter()
    bronze = adapter.fetch(qid, qtext, run_index=1)
    canonical = adapter.normalise(bronze)

    bronze_file = cfg.BRONZE / f"{bronze.engine}__{qid}__r1.json"
    print("-" * 64)
    print(f"BRONZE written : {bronze_file.relative_to(ROOT)}  ({bronze_file.stat().st_size} bytes)")
    print(f"served model   : {bronze.model_served}")
    print(f"token usage    : {bronze.usage}")
    print("-" * 64)
    print(f"answer (first 300 chars):\n{canonical.answer_text[:300]}")
    print("-" * 64)
    print(f"CITED SOURCES extracted: {len(canonical.cited_sources)}")
    for s in canonical.cited_sources:
        print(f"  [{s.position}] {s.domain}  | {(s.title or '')[:40]}  | {s.url[:60]}")
    print("=" * 64)
    if canonical.cited_sources:
        print("RESULT: PASS - grounded, cited response captured and normalised.")
    else:
        print("RESULT: CHECK - response captured but NO citations parsed.")
        print("  Either Gemini did not search this query, or the metadata path differs.")
        print("  Inspect the bronze JSON candidates[0].grounding_metadata and tell me.")


if __name__ == "__main__":
    main()
