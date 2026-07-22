"""
test_mistral_adapter.py — prove the Mistral adapter on ONE pilot query (v2 registry).
Mistral decides whether to search; defaults to row 57 (the current/transactional row that
grounded Gemini & Claude). Override with --query-index N.
From the project root:  python scripts/test_mistral_adapter.py --query-index N
Costs cents (one grounded Agents API call + connector tokens).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env import load_env
load_env()

import pandas as pd

import thesis_config as cfg
from src.adapters.mistral_adapter import MistralAdapter

REGISTRY = cfg.QUERY_REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-index", type=int, default=57)
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
    print("BATCH 5 - MISTRAL ADAPTER, SINGLE-QUERY TEST (v2 registry)")
    print("=" * 64)
    print(f"index    : {qi}")
    print(f"query_id : {qid}")
    print(f"query    : {qtext[:120]}")
    print("Calling mistral-medium-3-5 via Agents API + web_search connector (cents)...")

    adapter = MistralAdapter()
    bronze = adapter.fetch(qid, qtext, run_index=1)
    canonical = adapter.normalise(bronze)

    bronze_file = cfg.BRONZE / f"{bronze.engine}__{qid}__r1.json"
    raw = bronze.raw_response
    entry_types = [e.get("type") for e in (raw.get("outputs") or [])]

    first_ref = None
    for e in (raw.get("outputs") or []):
        if e.get("type") == "message.output" and isinstance(e.get("content"), list):
            for ch in e["content"]:
                if ch.get("type") == "tool_reference":
                    first_ref = ch
                    break
        if first_ref:
            break

    print("-" * 64)
    print(f"BRONZE written : {bronze_file.relative_to(ROOT)}  ({bronze_file.stat().st_size} bytes)")
    print(f"served model   : {bronze.model_served}")
    print(f"output entries : {entry_types}")
    print(f"token usage    : {bronze.usage}")
    print(f"first tool_reference chunk (raw): {json.dumps(first_ref)[:300] if first_ref else None}")
    print("-" * 64)
    print(f"answer (first 300 chars):\n{canonical.answer_text[:300]}")
    print("-" * 64)
    print(f"CITED SOURCES extracted: {len(canonical.cited_sources)}")
    for s in canonical.cited_sources:
        print(f"  [{s.position}] {s.domain}  | {(s.title or '')[:40]}  | {s.url[:60]}")
    print("=" * 64)
    if canonical.cited_sources:
        print("RESULT: PASS - grounded, provider-certified citations captured and normalised.")
    elif "tool.execution" in entry_types:
        print("RESULT: CHECK - Mistral searched but no tool_reference parsed.")
        print("  Inspect the 'first tool_reference chunk (raw)' line and the bronze content chunks.")
    else:
        print("RESULT: CHECK - Mistral did not search this query. Try another current/transactional row.")


if __name__ == "__main__":
    main()
