"""
test_openai_forced_adapter.py — sensitivity test: FORCE web_search on every call.

Mirrors scripts/test_openai_adapter.py but exercises OpenAIForcedSearchAdapter
over the first 10 frozen-registry queries x 3 runs = 30 grounded calls, then
joins each forced response to the matching auto-adapter bronze (same query_id +
run_index) so the two regimes can be compared directly.

The whole point: with tool_choice forcing web_search, EVERY call should search
(did_search=True). A zero-citation row then means "searched but cited nothing",
never "model declined to search" — the chatgpt zero-citation floor we're probing.

COST: 30 grounded gpt-5.5 calls ≈ $0.50–1.00. A confirmation prompt guards the
loop so you can abort before spending anything.

From the project root:
    python scripts/test_openai_forced_adapter.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env import load_env
load_env()

import csv

import pandas as pd

import thesis_config as cfg
from src.adapters.openai_adapter import ENGINE_KEY as AUTO_ENGINE_KEY
from src.adapters.openai_forced_search_adapter import OpenAIForcedSearchAdapter
from src.schema import BronzeRecord

REGISTRY = cfg.QUERY_REGISTRY
N_QUERIES = 10                 # first 10 queries from the frozen registry
K_RUNS = 3                     # repeats per query -> 30 calls total
CSV_OUT = cfg.DATA / "forced" / "openai_forced_vs_auto.csv"


def _diagnose(raw_response: dict):
    """Same diagnostic dissection as test_openai_adapter.py: walk the typed output
    array, report item types, whether a web_search_call fired, and whether any
    url_citation annotation is present in bronze (to tell a real parser miss apart
    from a legitimate zero-citation response)."""
    output = raw_response.get("output") or []
    item_types = [it.get("type") for it in output]
    searched = "web_search_call" in item_types
    annotations_present = any(
        ann.get("type") == "url_citation"
        for it in output if it.get("type") == "message"
        for block in (it.get("content") or [])
        for ann in (block.get("annotations") or [])
    )
    return item_types, searched, annotations_present


def _verdict(cited_sources, searched, annotations_present) -> str:
    """The four-way RESULT verdict from test_openai_adapter.py, adapted for the
    forced regime (a non-search is now UNEXPECTED, since tool_choice forces it)."""
    if cited_sources:
        return "PASS  - grounded, cited response captured and normalised."
    elif annotations_present:
        return ("FAIL  - bronze HAS url_citation annotations but none extracted "
                "(real parser bug; inspect normalise()).")
    elif not searched:
        return ("CHECK - model did NOT search despite tool_choice forcing it "
                "(unexpected; inspect the request).")
    else:
        return ("CHECK - searched but cited nothing (legitimate zero-citation "
                "result; the chatgpt floor surviving a forced search).")


def _bronze_from_dict(d: dict) -> BronzeRecord:
    """Rebuild a BronzeRecord from a loaded bronze JSON dict so we can run the
    adapter's normalise() over the existing auto-adapter files identically."""
    return BronzeRecord(
        query_id=d.get("query_id", ""),
        engine=d.get("engine", ""),
        run_index=d.get("run_index", 0),
        model_requested=d.get("model_requested", ""),
        model_served=d.get("model_served", ""),
        timestamp_utc=d.get("timestamp_utc", ""),
        query_text=d.get("query_text", ""),
        request_params=d.get("request_params", {}),
        raw_response=d.get("raw_response", {}),
        answer_text=d.get("answer_text", ""),
        usage=d.get("usage", {}),
    )


def main():
    if not REGISTRY.exists():
        print(f"[FAIL] frozen registry not found at {REGISTRY}. Run build_pilot_queries.py first.")
        sys.exit(1)

    df = pd.read_parquet(REGISTRY)
    queries = df.head(N_QUERIES)
    n_calls = len(queries) * K_RUNS

    print("=" * 64)
    print("OPENAI FORCED-SEARCH ADAPTER - SENSITIVITY TEST")
    print("=" * 64)
    print(cfg.profile_summary())
    print(f"Frozen registry: {REGISTRY.relative_to(ROOT)}  ({len(df)} rows)")
    print(f"Testing first {len(queries)} queries x {K_RUNS} runs = {n_calls} grounded calls")
    for i, q in enumerate(queries["query_text"]):
        print(f"  [{i}] {str(q)[:90]}")
    print("-" * 64)

    # Confirmation gate — these are LIVE, billed calls.
    print(f"COST: ~{n_calls} grounded gpt-5.5 calls (web_search forced) ≈ $0.50–1.00.")
    resp = input("Proceed with the live calls? [y/N]: ").strip().lower()
    if resp not in ("y", "yes"):
        print("Aborted - no calls made, no cost incurred.")
        return

    adapter = OpenAIForcedSearchAdapter()

    per_query = {}        # qid -> list of n_citations across runs
    searched_count = {}   # qid -> count of runs that searched
    csv_rows = []
    missing_auto = []

    print("-" * 64)
    print(f"{'query_id':<22}{'run':>4}{'did_search':>12}{'n_cit':>7}  {'1st domain':<22}verdict")
    print("-" * 64)

    for _, row in queries.iterrows():
        qid, qtext = row["query_id"], row["query_text"]
        per_query.setdefault(qid, [])
        searched_count.setdefault(qid, 0)
        for run in range(1, K_RUNS + 1):
            bronze = adapter.fetch(qid, qtext, run_index=run)
            canon = adapter.normalise(bronze)
            _, searched, annotations_present = _diagnose(bronze.raw_response)
            n_cit = len(canon.cited_sources)
            first_domain = canon.cited_sources[0].domain if canon.cited_sources else "—"
            verdict = _verdict(canon.cited_sources, searched, annotations_present)

            print(f"{qid:<22}{run:>4}{str(searched):>12}{n_cit:>7}  "
                  f"{(first_domain or '—'):<22}{verdict}")

            per_query[qid].append(n_cit)
            if searched:
                searched_count[qid] += 1

            # Join to the auto adapter's EXISTING bronze (same qid + run). Skip
            # (with a note) if it hasn't been captured yet — never fail.
            auto_path = cfg.BRONZE / f"{AUTO_ENGINE_KEY}__{qid}__r{run}.json"
            if auto_path.exists():
                auto_rec = _bronze_from_dict(json.loads(auto_path.read_text()))
                auto_canon = adapter.normalise(auto_rec)
                _, auto_searched, _ = _diagnose(auto_rec.raw_response)
                csv_rows.append({
                    "query_id": qid,
                    "run_index": run,
                    "auto_searched": auto_searched,
                    "forced_searched": searched,
                    "auto_n_citations": len(auto_canon.cited_sources),
                    "forced_n_citations": n_cit,
                })
            else:
                missing_auto.append(auto_path.name)

    # --- per-query summary --------------------------------------------------
    # citations_consistent: did all K runs return the SAME citation count? (a
    # crude repeat-to-repeat stability flag for the forced regime).
    print("-" * 64)
    print("PER-QUERY SUMMARY")
    print(f"{'query_id':<22}{'runs_searched':>15}{'mean_n_cit':>12}{'consistent':>13}")
    for qid, counts in per_query.items():
        runs_searched = searched_count[qid]
        mean_n = sum(counts) / len(counts) if counts else 0.0
        consistent = len(set(counts)) == 1
        print(f"{qid:<22}{f'{runs_searched}/{K_RUNS}':>15}{mean_n:>12.2f}{str(consistent):>13}")

    # --- write the forced-vs-auto join CSV ----------------------------------
    if missing_auto:
        print("-" * 64)
        print(f"NOTE: {len(missing_auto)} auto bronze file(s) not found — skipped in CSV join:")
        for name in missing_auto:
            print(f"  missing: {name}")

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query_id", "run_index", "auto_searched", "forced_searched",
            "auto_n_citations", "forced_n_citations",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)

    print("-" * 64)
    print(f"CSV written: {CSV_OUT.relative_to(ROOT)}  ({len(csv_rows)} joined rows)")
    print("=" * 64)


if __name__ == "__main__":
    main()
