"""
build_silver_forced_pilot.py — SILVER layer for the FORCED-search sensitivity slice.

Pure transform, identical in spirit to src.silver for the main pipeline: run every
forced bronze record through the forced adapter's normalise() (inherited verbatim
from OpenAIAdapter), then flatten into the SAME two analysis tables the full pilot
uses —
  responses : one row per (engine, query, run)   [src.silver.build_responses_table]
  citations : one row per cited source (long)     [src.silver.build_citations_table]

Reuses src.silver's table builders verbatim, so the silver schema is byte-identical
to the full pilot's; only the bronze source dir and the output paths differ (the
forced slice lives outside the profiled data tree). No API key / network needed —
normalise() is a pure read of the captured bronze.

Run:  python scripts/build_silver_forced_pilot.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env import load_env
load_env()

import thesis_config as cfg
from src.silver import (
    reconstruct_bronze, build_responses_table, build_citations_table,
)
from src.adapters.openai_forced_search_adapter import (
    OpenAIForcedSearchAdapter, BRONZE_FORCED, ENGINE_KEY,
)

# Forced-slice silver, parallel to data/bronze/openai_forced and
# data/gold/forced_pilot_chatgpt.parquet — outside the profiled cfg.SILVER so it
# never mixes with the main pipeline's responses_pilot/citations_pilot tables.
SILVER_FORCED = cfg.DATA / "forced" / "silver"
RESPONSES_OUT = SILVER_FORCED / "responses_forced.parquet"
CITATIONS_OUT = SILVER_FORCED / "citations_forced.parquet"


def _normalise_forced(rec):
    """Forced adapter's normalise() WITHOUT __init__ (no API key); pure bronze read.
    Same no-init pattern as src.silver.normalise_record; carries run_profile through."""
    inst = OpenAIForcedSearchAdapter.__new__(OpenAIForcedSearchAdapter)
    inst.spec = cfg.ENGINES["chatgpt"]   # safety belt if normalise ever reads self.spec
    canon = inst.normalise(rec)
    canon.run_profile = rec.run_profile
    return canon


def main():
    files = sorted(BRONZE_FORCED.glob(f"{ENGINE_KEY}__*.json"))
    if not files:
        print(f"[STOP] no forced bronze in {BRONZE_FORCED.relative_to(ROOT)}. "
              f"Run scripts/test_openai_forced_adapter.py first.")
        sys.exit(1)

    canonicals = [_normalise_forced(reconstruct_bronze(json.loads(f.read_text()))) for f in files]
    responses = build_responses_table(canonicals)
    citations = build_citations_table(canonicals)

    SILVER_FORCED.mkdir(parents=True, exist_ok=True)
    responses.to_parquet(RESPONSES_OUT, index=False)
    citations.to_parquet(CITATIONS_OUT, index=False)

    print("=" * 72)
    print(f"FORCED SILVER BUILD | {len(files)} bronze -> {len(responses)} response rows, "
          f"{len(citations)} citation rows")
    print(f"responses cols: {list(responses.columns)}")
    print(f"citations cols: {list(citations.columns)}")
    print(f"cells with >=1 citation: {int(responses['has_citations'].sum())}/{len(responses)}")
    print(f"written: {RESPONSES_OUT.relative_to(ROOT)}")
    print(f"         {CITATIONS_OUT.relative_to(ROOT)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
