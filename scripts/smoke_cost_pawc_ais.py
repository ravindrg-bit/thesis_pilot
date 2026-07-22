"""
smoke_cost_pawc_ais.py - PAWC+AIS cost smoke test over 50 cells (10 queries x 5 engines,
ONE run each; Mistral excluded). Spends real money (cents-low dollars). Prints a per-engine
breakdown AND a cost extrapolation to the full pilot.
Run from the project root:  python scripts/smoke_cost_pawc_ais.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env import load_env
load_env()

import pandas as pd
from anthropic import Anthropic

import thesis_config as cfg
from src import attribution as attr

ENGINES = ["chatgpt", "claude", "gemini", "perplexity", "kimi"]   # Mistral excluded
N_QUERIES = 10
PILOT_K = 3   # for the extrapolation only


def main():
    responses = pd.read_parquet(cfg.SILVER / f"responses_{cfg.RUN_PROFILE}.parquet")
    citations = pd.read_parquet(cfg.SILVER / f"citations_{cfg.RUN_PROFILE}.parquet")

    # 10 query_ids present for ALL 5 engines, run_index == 1, deterministic pick
    base = responses[responses.engine.isin(ENGINES)]
    per_q = base.groupby("query_id")["engine"].nunique()
    full_qids = sorted(per_q[per_q >= len(ENGINES)].index)[:N_QUERIES]
    if len(full_qids) < N_QUERIES:
        print(f"[WARN] only {len(full_qids)} queries common to all 5 engines.")

    client = Anthropic()
    cache = {}
    rows = []
    t0 = time.time()

    print("=" * 70)
    print(f"PAWC+AIS COST SMOKE TEST | {len(full_qids)} queries x {len(ENGINES)} engines, 1 run each")
    print(f"judge: {cfg.JUDGE_MODEL} | snippet limit: {cfg.JUDGE_SOURCE_CHAR_LIMIT} chars")
    print("=" * 70)

    for qid in full_qids:
        for eng in ENGINES:
            cell = responses[(responses.engine == eng) & (responses.query_id == qid)
                             & (responses.run_index == 1)]
            if cell.empty:
                cell = responses[(responses.engine == eng) & (responses.query_id == qid)]
                if cell.empty:
                    continue
            run = int(cell.iloc[0]["run_index"])
            answer = cell.iloc[0]["answer_text"]
            srcs = (citations[(citations.engine == eng) & (citations.query_id == qid)
                              & (citations.run_index == run)]
                    .sort_values("position")[["position", "url"]].to_dict("records"))
            res = attr.attribute_cell(answer, srcs, client, cache)
            res.update({"engine": eng, "query_id": qid, "run_index": run})
            rows.append(res)
            print(f"  {eng:11s} {qid[:18]} | sents {res['n_sentences']:3d} | "
                  f"src {res['n_sources_fetched_ok']}/{res['n_sources_cited']} | "
                  f"AIS {res['ais_rate'] if res['ais_rate'] is None else round(res['ais_rate'],2)} | "
                  f"PAWC {res['pawc_total']} | calls {res['judge_calls']}")

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    cfg.GOLD.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["fetch_status", "pawc_by_source_position"]).to_parquet(
        cfg.GOLD / f"smoke_pawc_ais_{cfg.RUN_PROFILE}.parquet", index=False)

    print("\n" + "-" * 70)
    print("PER-ENGINE SUMMARY (smoke sample):")
    g = df.groupby("engine").agg(
        cells=("query_id", "count"),
        mean_sentences=("n_sentences", "mean"),
        mean_ais=("ais_rate", "mean"),
        mean_pawc=("pawc_total", "mean"),
        total_calls=("judge_calls", "sum"),
        total_in_tok=("input_tokens", "sum"),
        total_out_tok=("output_tokens", "sum"),
    ).round(2)
    print(g.to_string())

    # ---- cost extrapolation ----
    cells_smoke = len(df)
    calls_smoke = int(df["judge_calls"].sum())
    in_tok = int(df["input_tokens"].sum())
    out_tok = int(df["output_tokens"].sum())
    cost_smoke = (in_tok / 1e6) * cfg.HAIKU_INPUT_USD_PER_MTOK + (out_tok / 1e6) * cfg.HAIKU_OUTPUT_USD_PER_MTOK

    calls_per_cell = calls_smoke / cells_smoke if cells_smoke else 0
    cost_per_cell = cost_smoke / cells_smoke if cells_smoke else 0

    # full pilot for these 5 engines = 100 queries x 5 x k=3
    full_cells = 100 * len(ENGINES) * PILOT_K
    proj_calls = calls_per_cell * full_cells
    proj_cost_sync = cost_per_cell * full_cells
    proj_cost_batch = proj_cost_sync * 0.5

    print("\n" + "=" * 70)
    print("COST READOUT (smoke):")
    print(f"  cells judged      : {cells_smoke}")
    print(f"  judge calls       : {calls_smoke}  ({calls_per_cell:.1f}/cell)")
    print(f"  tokens in/out     : {in_tok:,} / {out_tok:,}")
    print(f"  smoke cost (sync) : ${cost_smoke:.4f}  (${cost_per_cell:.5f}/cell)")
    print(f"  elapsed           : {elapsed/60:.1f} min  ({elapsed/cells_smoke:.1f}s/cell)")
    print("\nEXTRAPOLATION to full 5-engine pilot (100 q x 5 x k=3 = "
          f"{full_cells} cells):")
    print(f"  projected judge calls : {proj_calls:,.0f}")
    print(f"  projected cost SYNC   : ${proj_cost_sync:,.2f}")
    print(f"  projected cost BATCH  : ${proj_cost_batch:,.2f}  (50% off, laptop-free)")
    print("=" * 70)
    print("NB: prices are the VERIFY-me constants in thesis_config; source-fetch is free;")
    print("    the human judge-validation subsample is separate (time, not API cost).")


if __name__ == "__main__":
    main()
