"""
smoke_cost_chunked.py - PAWC+AIS cost smoke over the SAME 50 cells as the per-sentence and
full-batch smokes, using the CHUNKED judge (judge N sentences per call, sources sent once per
chunk). Spends real money (cents). Prints a per-engine breakdown, a cost readout, AND a
THREE-WAY head-to-head vs per-sentence (smoke_pawc_ais_pilot.parquet) and full-batch
(smoke_pawc_ais_batched_pilot.parquet) if present.
Run:  python scripts/smoke_cost_chunked.py --chunk-size 10
"""

import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=10, help="sentences judged per call")
    args = ap.parse_args()
    cs = args.chunk_size

    responses = pd.read_parquet(cfg.SILVER / f"responses_{cfg.RUN_PROFILE}.parquet")
    citations = pd.read_parquet(cfg.SILVER / f"citations_{cfg.RUN_PROFILE}.parquet")

    # SAME deterministic selection as the prior two smokes -> identical 50 cells
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
    print(f"PAWC+AIS COST SMOKE (CHUNKED, chunk_size={cs}) | {len(full_qids)} queries x {len(ENGINES)} engines, 1 run each")
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
            res = attr.attribute_cell_chunked(answer, srcs, client, cache, chunk_size=cs)
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
        cfg.GOLD / f"smoke_pawc_ais_chunked{cs}_{cfg.RUN_PROFILE}.parquet", index=False)

    print("\n" + "-" * 70)
    print(f"PER-ENGINE SUMMARY (chunked cs={cs}):")
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

    cells_smoke = len(df)
    in_tok = int(df["input_tokens"].sum())
    out_tok = int(df["output_tokens"].sum())
    cost_smoke = _cost(in_tok, out_tok)
    full_cells = 100 * len(ENGINES) * PILOT_K
    print("\nCOST READOUT (chunked):")
    print(f"  judge calls {int(df['judge_calls'].sum())} ({df['judge_calls'].sum()/cells_smoke:.1f}/cell) | "
          f"tokens in/out {in_tok:,}/{out_tok:,} | cost ${cost_smoke:.4f} | "
          f"elapsed {elapsed/60:.1f} min")
    print(f"  EXTRAPOLATION to {full_cells} cells: sync ${_cost(in_tok, out_tok)/cells_smoke*full_cells:,.2f} | "
          f"batch ${_cost(in_tok, out_tok)/cells_smoke*full_cells*0.5:,.2f}")

    # ---- THREE-WAY comparison ----
    def load(name):
        p = cfg.GOLD / name
        return pd.read_parquet(p) if p.exists() else None

    ps = load(f"smoke_pawc_ais_{cfg.RUN_PROFILE}.parquet")          # per-sentence
    fb = load(f"smoke_pawc_ais_batched_{cfg.RUN_PROFILE}.parquet")  # full-batch
    ch = df                                                          # chunked (this run)

    def totals(d):
        if d is None:
            return None
        it, ot = int(d["input_tokens"].sum()), int(d["output_tokens"].sum())
        return {"calls": int(d["judge_calls"].sum()), "in_tok": it, "out_tok": ot, "cost": _cost(it, ot)}

    tp, tf, tc = totals(ps), totals(fb), totals(ch)
    print("\n" + "=" * 70)
    print(f"THREE-WAY TOTALS (same {cells_smoke} cells):")
    print(f"{'':16s}{'per-sentence':>15s}{'full-batch':>14s}{f'chunked(cs={cs})':>16s}")
    def row(label, key, fmt):
        a = format(tp[key], fmt) if tp else "n/a"
        b = format(tf[key], fmt) if tf else "n/a"
        c = format(tc[key], fmt) if tc else "n/a"
        print(f"{label:16s}{a:>15s}{b:>14s}{c:>16s}")
    row("judge calls", "calls", ",d")
    row("input tokens", "in_tok", ",d")
    row("output tokens", "out_tok", ",d")
    row("smoke cost $", "cost", ".4f")

    print("-" * 70)
    print("PER-ENGINE mean AIS  (per-sentence | full-batch | chunked)  +  |chunked - per-sentence|:")
    def eng_means(d, col):
        return d.groupby("engine")[col].mean() if d is not None else None
    ais_ps, ais_fb, ais_ch = eng_means(ps, "ais_rate"), eng_means(fb, "ais_rate"), eng_means(ch, "ais_rate")
    print(f"{'engine':12s}{'ais_ps':>9s}{'ais_fb':>9s}{'ais_ch':>9s}{'|ch-ps|':>10s}")
    for e in ENGINES:
        p_ = ais_ps.get(e) if ais_ps is not None else None
        f_ = ais_fb.get(e) if ais_fb is not None else None
        c_ = ais_ch.get(e) if ais_ch is not None else None
        d_ = abs(c_ - p_) if (p_ is not None and c_ is not None) else None
        print(f"{e:12s}{(f'{p_:.3f}' if p_ is not None else 'n/a'):>9s}"
              f"{(f'{f_:.3f}' if f_ is not None else 'n/a'):>9s}"
              f"{(f'{c_:.3f}' if c_ is not None else 'n/a'):>9s}"
              f"{(f'{d_:.3f}' if d_ is not None else 'n/a'):>10s}")

    print("-" * 70)
    print("PER-ENGINE mean PAWC  (per-sentence | full-batch | chunked):")
    pawc_ps, pawc_fb, pawc_ch = eng_means(ps, "pawc_total"), eng_means(fb, "pawc_total"), eng_means(ch, "pawc_total")
    print(f"{'engine':12s}{'pawc_ps':>10s}{'pawc_fb':>10s}{'pawc_ch':>10s}")
    for e in ENGINES:
        p_ = pawc_ps.get(e) if pawc_ps is not None else None
        f_ = pawc_fb.get(e) if pawc_fb is not None else None
        c_ = pawc_ch.get(e) if pawc_ch is not None else None
        print(f"{e:12s}{(f'{p_:.1f}' if p_ is not None else 'n/a'):>10s}"
              f"{(f'{f_:.1f}' if f_ is not None else 'n/a'):>10s}"
              f"{(f'{c_:.1f}' if c_ is not None else 'n/a'):>10s}")
    print("=" * 70)
    print(f"Chunked (cs={cs}) is the sweet spot IF |ch-ps| AIS is small AND cost << per-sentence.")


if __name__ == "__main__":
    main()
