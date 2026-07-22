"""
test_kimi_adapter.py — prove the (prompt-elicited) Kimi adapter on ONE pilot query (v2).
Kimi has no structured citation channel; sources are elicited into the answer text via a
system prompt and parsed out (self-reported). Defaults to row 50 (the "opioids" row where
Claude and Gemini also cited) as a generalisation check beyond the current/transactional probe.
From the project root:  python scripts/test_kimi_adapter.py --query-index N
Costs cents (one grounded answer + $0.005 per $web_search call).
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
from src.adapters.kimi_adapter import KimiAdapter

REGISTRY = cfg.QUERY_REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-index", type=int, default=50)
    args = ap.parse_args()

    if not REGISTRY.exists():
        print(f"[FAIL] frozen registry not found at {REGISTRY}.")
        sys.exit(1)

    df = pd.read_parquet(REGISTRY)
    qi = args.query_index
    if qi < 0 or qi >= len(df):
        print(f"[FAIL] --query-index {qi} out of range (0..{len(df)-1}).")
        sys.exit(1)
    row = df.iloc[qi]
    qid, qtext = row["query_id"], row["query_text"]

    print("=" * 64)
    print("BATCH 5 - KIMI ADAPTER (prompt-elicited citations), SINGLE-QUERY TEST")
    print("=" * 64)
    print(f"index    : {qi}")
    print(f"query_id : {qid}")
    print(f"query    : {qtext[:120]}")
    print("Calling kimi-k2.6 with $web_search + citation system prompt (cents)...")

    adapter = KimiAdapter()
    bronze = adapter.fetch(qid, qtext, run_index=1)
    canonical = adapter.normalise(bronze)

    bronze_file = cfg.BRONZE / f"{bronze.engine}__{qid}__r1.json"
    print("-" * 64)
    print(f"BRONZE written : {bronze_file.relative_to(ROOT)}  ({bronze_file.stat().st_size} bytes)")
    print(f"served model   : {bronze.model_served}")
    print(f"search rounds  : {bronze.raw_response.get('n_search_rounds')}")
    print("-" * 64)
    print(f"CITED SOURCES extracted: {len(canonical.cited_sources)}")
    for s in canonical.cited_sources:
        print(f"  [{s.position}] {s.domain}  | {s.url[:70]}  | {s.provenance}")
    print("-" * 64)
    print(f"answer tail (last 600 chars):\n...{(canonical.answer_text or '')[-600:]}")
    print("=" * 64)
    if canonical.cited_sources:
        print(f"RESULT: PASS - {len(canonical.cited_sources)} self-reported sources parsed "
              f"(provenance flagged). Technique generalises to this row.")
    elif (bronze.raw_response.get('n_search_rounds') or 0) > 0:
        print("RESULT: CHECK - Kimi searched but emitted no URLs in text this time.")
        print("  Inspect the answer tail; the model may have skipped the Sources list.")
    else:
        print("RESULT: CHECK - Kimi did not search this query. Try a current/transactional row.")


if __name__ == "__main__":
    main()
