"""
smoke_cost_batched.py - PAWC+AIS cost smoke test over the SAME 50 cells as
smoke_cost_pawc_ais.py, but using the BATCHED judge (one call per answer, sources sent once).
Spends real money (cents). Prints a per-engine breakdown, a cost extrapolation, AND a
head-to-head vs the per-sentence run (smoke_pawc_ais_pilot.parquet) if present, so you can see
(a) cost dropped and (b) the AIS/PAWC metrics still match.
Run from the project root:  python scripts/smoke_cost_batched.py
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


def _cost(in_tok, out_tok):
    return (in_tok / 1e6) * cfg.HAIKU_INPUT_USD_PER_MTOK + (out_tok / 1e6) * cfg.HAIKU_OUTPUT_USD_PER_MTOK


def main():
    responses = pd.read_parquet(cfg.SILVER / f"responses_{cfg.RUN_PROFILE}.parquet")
    citations = pd.read_parquet(cfg.SILVER / f"citations_{cfg.RUN_PROFILE}.parquet")

    # SAME deterministic selection as smoke_cost_pawc_ais.py -> identical 50 cells
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
    print(f"PAWC+AIS COST SMOKE TEST (BATCHED) | {len(full_qids)} queries x {len(ENGINES)} engines, 1 run each")
    print(f"judge: {cfg.JUDGE_MODEL} | snippet limit: {cfg.JUDGE_SOURCE_CHAR_LIMIT} chars | ONE call/answer")
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
            res = attr.attribute_cell_batched(answer, srcs, client, cache)
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
        cfg.GOLD / f"smoke_pawc_ais_batched_{cfg.RUN_PROFILE}.parquet", index=False)

    print("\n" + "-" * 70)
    print("PER-ENGINE SUMMARY (batched smoke sample):")
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

    # ---- cost readout (batched) ----
    cells_smoke = len(df)
    calls_smoke = int(df["judge_calls"].sum())
    in_tok = int(df["input_tokens"].sum())
    out_tok = int(df["output_tokens"].sum())
    cost_smoke = _cost(in_tok, out_tok)
    cost_per_cell = cost_smoke / cells_smoke if cells_smoke else 0
    full_cells = 100 * len(ENGINES) * PILOT_K
    proj_cost_sync = cost_per_cell * full_cells

    print("\n" + "=" * 70)
    print("COST READOUT (batched smoke):")
    print(f"  cells judged      : {cells_smoke}")
    print(f"  judge calls       : {calls_smoke}  ({calls_smoke/cells_smoke:.1f}/cell)")
    print(f"  tokens in/out     : {in_tok:,} / {out_tok:,}")
    print(f"  smoke cost (sync) : ${cost_smoke:.4f}  (${cost_per_cell:.5f}/cell)")
    print(f"  elapsed           : {elapsed/60:.1f} min  ({elapsed/cells_smoke:.1f}s/cell)")
    print(f"  EXTRAPOLATION to full 5-engine pilot ({full_cells} cells): "
          f"sync ${proj_cost_sync:,.2f} | batch ${proj_cost_sync*0.5:,.2f}")

    # ---- head-to-head vs the per-sentence run ----
    orig_path = cfg.GOLD / f"smoke_pawc_ais_{cfg.RUN_PROFILE}.parquet"
    if not orig_path.exists():
        print("\n[no original smoke_pawc_ais_pilot.parquet found -> skipping comparison]")
        return
    o = pd.read_parquet(orig_path)
    o_in, o_out = int(o["input_tokens"].sum()), int(o["output_tokens"].sum())
    o_cost = _cost(o_in, o_out)

    def pct(new, old):
        return f"{100*(new-old)/old:+.1f}%" if old else "n/a"

    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD: per-sentence (orig) vs batched (same 50 cells)")
    print(f"{'':18s}{'per-sentence':>16s}{'batched':>14s}{'delta':>12s}")
    print(f"{'cells':18s}{len(o):>16d}{len(df):>14d}")
    print(f"{'judge calls':18s}{int(o['judge_calls'].sum()):>16d}{calls_smoke:>14d}"
          f"{pct(calls_smoke, int(o['judge_calls'].sum())):>12s}")
    print(f"{'input tokens':18s}{o_in:>16,d}{in_tok:>14,d}{pct(in_tok, o_in):>12s}")
    print(f"{'output tokens':18s}{o_out:>16,d}{out_tok:>14,d}{pct(out_tok, o_out):>12s}")
    print(f"{'smoke cost $':18s}{o_cost:>16.4f}{cost_smoke:>14.4f}{pct(cost_smoke, o_cost):>12s}")

    print("-" * 70)
    print("METRICS MATCH? per-engine mean AIS / mean PAWC (orig -> batched):")
    om = o.groupby("engine").agg(ais=("ais_rate", "mean"), pawc=("pawc_total", "mean"))
    bm = df.groupby("engine").agg(ais=("ais_rate", "mean"), pawc=("pawc_total", "mean"))
    print(f"{'engine':12s}{'ais_orig':>10s}{'ais_batch':>11s}{'d_ais':>8s}{'pawc_orig':>11s}{'pawc_batch':>12s}{'d_pawc':>9s}")
    for e in ENGINES:
        if e in om.index and e in bm.index:
            ao, ab = om.loc[e, "ais"], bm.loc[e, "ais"]
            po, pb = om.loc[e, "pawc"], bm.loc[e, "pawc"]
            print(f"{e:12s}{ao:>10.3f}{ab:>11.3f}{ab-ao:>+8.3f}{po:>11.1f}{pb:>12.1f}{pb-po:>+9.1f}")
    print("=" * 70)
    print("Cost should drop sharply (sources sent once, not per sentence); AIS/PAWC should be")
    print("close (identical logic; small drift = judge stochasticity / long-answer JSON limits).")


if __name__ == "__main__":
    main()
