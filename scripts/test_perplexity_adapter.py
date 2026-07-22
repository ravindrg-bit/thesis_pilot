"""
test_perplexity_adapter.py — prove the Perplexity adapter on ONE pilot query.
Sonar grounds EVERY query, so this should return citations even for the conceptual
row 0 that gpt-5.5 declined to search -- a useful cross-engine contrast.
From the project root:  python scripts/test_perplexity_adapter.py [--query-index N]
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
from src.adapters.perplexity_adapter import PerplexityAdapter

REGISTRY = cfg.QUERY_REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-index", type=int, default=0)
    args = ap.parse_args()

    if not REGISTRY.exists():
        print(f"[FAIL] frozen registry not found at {REGISTRY}. Run build_pilot_queries.py first.")
        sys.exit(1)

    df = pd.read_parquet(REGISTRY)
    print("Frozen pilot queries:")
    for i, q in enumerate(df["query_text"]):
        print(f"  [{i}] {str(q)[:90]}")

    qi = args.query_index
    if qi < 0 or qi >= len(df):
        print(f"[FAIL] --query-index {qi} out of range (0..{len(df)-1}).")
        sys.exit(1)
    row = df.iloc[qi]
    qid, qtext = row["query_id"], row["query_text"]

    print("=" * 64)
    print("BATCH 5 - PERPLEXITY ADAPTER, SINGLE-QUERY TEST")
    print("=" * 64)
    print(f"index    : {qi}")
    print(f"query_id : {qid}")
    print(f"query    : {qtext[:120]}")
    print("Calling sonar (inherently grounded; a few cents)...")

    adapter = PerplexityAdapter()
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
        print(f"  [{s.position}] {s.domain}  | {(s.title or '')[:60]}  | {s.url}")
    print("=" * 64)
    if canonical.cited_sources:
        print("RESULT: PASS - grounded, cited response captured and normalised.")
    else:
        print("RESULT: CHECK - response captured but NO citations parsed.")
        print("  Sonar grounds every query, so empty here likely means the citation")
        print("  fields didn't survive capture. Inspect the bronze JSON and tell me.")


if __name__ == "__main__":
    main()
