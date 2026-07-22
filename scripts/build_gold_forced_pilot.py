"""
build_gold_forced_pilot.py — GOLD layer for the FORCED-search sensitivity slice.

A smoke test of the forced adapter + the EXISTING metrics pipeline, end-to-end:
does forcing web_search yield citations on every call, and are those citations at
least SOMETIMES supported by their sources? Not a confirmatory analysis.

Medallion flow — same as the full pilot (scripts/run_nli_pilot.py):
  silver  data/silver/responses_forced.parquet + citations_forced.parquet
          (built by scripts/build_silver_forced_pilot.py)
    -> gold  one row per (query_id, run_index) via src.nli_attribution.attribute_cell_nli
          -> data/forced/gold/forced_chatgpt.parquet, SAME columns as nli_*.parquet

Reuses, WITHOUT reimplementing anything:
  - src.nli_attribution.attribute_cell_nli -> PAWC + AIS via DeBERTa NLI (same fn as run_nli_pilot)
  - src.attribution.fetch_source_text      -> BeautifulSoup source fetch (called inside it)

No paid API: NLI runs locally; only source pages are fetched over the network.
did_search for the diagnostic is read from the forced bronze (web_search_call presence);
it is NOT added to the gold schema, which stays identical to the full pilot's.

Run (after the silver build):  python scripts/build_gold_forced_pilot.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env import load_env
load_env()

import pandas as pd

import thesis_config as cfg
from src import nli_attribution as nli
from src.adapters.openai_forced_search_adapter import BRONZE_FORCED, ENGINE_KEY

# Forced-slice silver inputs (written by build_silver_forced_pilot.py) and the gold
# output, all parallel to data/bronze/openai_forced — outside the profiled data tree.
SILVER_FORCED = cfg.DATA / "forced" / "silver"
RESPONSES_IN = SILVER_FORCED / "responses_forced.parquet"
CITATIONS_IN = SILVER_FORCED / "citations_forced.parquet"
GOLD_OUT = cfg.DATA / "forced" / "gold" / "forced_chatgpt.parquet"


def _did_search_map() -> dict:
    """(query_id, run_index) -> did the model emit a web_search_call? Read from the
    forced bronze for the forced-regime diagnostic (not a gold-schema column)."""
    m = {}
    for f in BRONZE_FORCED.glob(f"{ENGINE_KEY}__*.json"):
        d = json.loads(f.read_text())
        out = d.get("raw_response", {}).get("output") or []
        m[(d["query_id"], int(d["run_index"]))] = any(
            it.get("type") == "web_search_call" for it in out)
    return m


def main():
    if not (RESPONSES_IN.exists() and CITATIONS_IN.exists()):
        print(f"[STOP] forced silver not found in {SILVER_FORCED.relative_to(ROOT)}.")
        print("       Build it first:  python scripts/build_silver_forced_pilot.py")
        sys.exit(1)

    responses = pd.read_parquet(RESPONSES_IN)
    citations = pd.read_parquet(CITATIONS_IN)
    searched_map = _did_search_map()

    print("=" * 72)
    print(f"FORCED-SEARCH GOLD BUILD | {len(responses)} silver cells "
          f"({len(citations)} citation rows)")
    print(f"NLI: {cfg.NLI_MODEL} | threshold {cfg.NLI_ENTAILMENT_THRESHOLD} | device {cfg.NLI_DEVICE}")
    print("=" * 72)

    cache, rows, t0, n_searched = {}, [], time.time(), 0

    for _, cell in responses.sort_values(["query_id", "run_index"]).iterrows():
        eng, qid, run = cell["engine"], cell["query_id"], int(cell["run_index"])
        answer = cell["answer_text"]
        # sources for this cell, position-ordered — SAME selection as run_nli_pilot.py
        srcs = (citations[(citations.engine == eng) & (citations.query_id == qid)
                          & (citations.run_index == run)]
                .sort_values("position")[["position", "url"]].to_dict("records"))
        res = nli.attribute_cell_nli(answer, srcs, cache)
        res.update({"engine": eng, "query_id": qid, "run_index": run,
                    "n_response_words": len((answer or "").split())})
        rows.append(res)

        searched = bool(searched_map.get((qid, run), False))
        n_searched += int(searched)
        ais = res["ais_rate"]
        print(f"{qid[:20]:20s} r{run} | search {str(searched):5s} | sents {res['n_sentences']:3d} | "
              f"src {res['n_sources_fetched_ok']}/{res['n_sources_cited']} | "
              f"AIS {round(ais, 2) if ais is not None else None} | PAWC {res['pawc_total']}")

    df = pd.DataFrame(rows)

    # ---- write gold: SAME columns as scripts/run_nli_pilot.py (drop fetch_status;
    #      KEEP pawc_by_source_position for downstream per-source PAWC normalisation) ----
    GOLD_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["fetch_status"]).to_parquet(GOLD_OUT, index=False)

    # ---- diagnostic summary -------------------------------------------------
    total = len(df)
    n_cited = int((df["n_sources_cited"] >= 1).sum())
    cited = df[df["n_sources_cited"] >= 1]
    mean_pawc = cited["pawc_total"].mean() if len(cited) else float("nan")
    mean_ais = pd.to_numeric(cited["ais_rate"], errors="coerce").mean() if len(cited) else float("nan")
    verdict = "PASS" if n_cited >= 27 else ("PARTIAL" if n_cited >= 15 else "FAIL")
    flag = "" if n_searched == total else "   <-- FLAG: forcing should give every call a search"

    print("=" * 72)
    print(f"calls total              : {total}")
    print(f"did_search=True          : {n_searched}/{total}{flag}")
    print(f"calls with >=1 citation  : {n_cited}/{total}")
    print(f"mean PAWC  (cited calls) : {mean_pawc:.3f}")
    print(f"mean AIS rate (cited)    : {mean_ais:.3f}")
    print(f"VERDICT                  : {verdict}  (PASS>=27 / PARTIAL 15-26 / FAIL<15 cited)")
    print(f"elapsed                  : {(time.time() - t0) / 60:.1f} min")
    print(f"written: {GOLD_OUT.relative_to(ROOT)}  ({total} rows; schema = nli_*.parquet)")
    print("=" * 72)


if __name__ == "__main__":
    main()
