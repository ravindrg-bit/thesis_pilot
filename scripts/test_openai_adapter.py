"""
test_openai_adapter.py — prove the OpenAI adapter on ONE real pilot query.
Sends one grounded query, writes a verbatim BronzeRecord, normalises it, and prints
what was captured. From the project root:
    python scripts/test_openai_adapter.py                  # default: row 0
    python scripts/test_openai_adapter.py --query-index N  # pick a different row
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
from src.adapters.openai_adapter import OpenAIAdapter

REGISTRY = cfg.QUERY_REGISTRY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-index", type=int, default=0,
                        help="row index into the frozen pilot registry (default: 0)")
    args = parser.parse_args()

    if not REGISTRY.exists():
        print(f"[FAIL] frozen registry not found at {REGISTRY}. Run build_pilot_queries.py first.")
        sys.exit(1)

    df = pd.read_parquet(REGISTRY)

    print("=" * 64)
    print("BATCH 4 - OPENAI ADAPTER, SINGLE-QUERY TEST")
    print("=" * 64)
    print(f"Frozen registry: {REGISTRY.relative_to(ROOT)}  ({len(df)} rows)")
    for i, q in enumerate(df["query_text"]):
        print(f"  [{i}] {str(q)[:90]}")
    print("-" * 64)

    qi = args.query_index
    if not (0 <= qi < len(df)):
        print(f"[FAIL] --query-index={qi} out of range. Valid: 0..{len(df) - 1}.")
        sys.exit(1)

    row = df.iloc[qi]
    qid, qtext = row["query_id"], row["query_text"]
    print(f"Selected index : {qi}")
    print(f"query_id       : {qid}")
    print(f"query          : {qtext[:120]}")
    print("Calling gpt-5.5 with web_search grounding (a few cents)...")

    adapter = OpenAIAdapter()
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
    # Diagnose a zero-citation result honestly: distinguish "model never searched"
    # and "searched but cited nothing" (both legitimate engine behaviour) from a real
    # parser miss (url_citation annotations present in bronze but none extracted).
    output = bronze.raw_response.get("output") or []
    item_types = [it.get("type") for it in output]
    searched = "web_search_call" in item_types
    annotations_present = any(
        ann.get("type") == "url_citation"
        for it in output if it.get("type") == "message"
        for block in (it.get("content") or [])
        for ann in (block.get("annotations") or [])
    )
    print(f"output[] item types: {item_types}")

    if canonical.cited_sources:
        print("RESULT: PASS - grounded, cited response captured and normalised.")
    elif annotations_present:
        print("RESULT: FAIL - bronze HAS url_citation annotations but none were "
              "extracted. This is a real parser bug; inspect normalise().")
    elif not searched:
        print("RESULT: CHECK - model did NOT search this query (no web_search_call), "
              "so zero citations is correct.")
        print("  This is the chatgpt zero-citation floor, not a parser problem.")
        print("  Indices 0-2 are trivia the model answers from memory. To see a PASS,")
        print("  pick a --query-index whose answer needs current/web info.")
    else:
        print("RESULT: CHECK - model searched but attached no url_citation annotations, "
              "so zero citations is correct (legitimate; not a parser problem).")


if __name__ == "__main__":
    main()
