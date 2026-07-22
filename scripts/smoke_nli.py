"""
smoke_nli.py - Option 3 (NLI) smoke test, 50 cells (10 queries x 5 engines, 1 run each;
Mistral excluded). NEAR-ZERO marginal cost - just local CPU/GPU compute. Prints side-by-side
comparison against the per-sentence judge smoke if it's on disk.

PAWC and AIS values come out of the SAME formulas as the judge smoke - only the support-
decision step changed (NLI entailment instead of Haiku verdict).
Run:  python scripts/smoke_nli.py
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.env import load_env
load_env()
import pandas as pd
import thesis_config as cfg
from src import nli_attribution as nli

ENGINES = ["chatgpt", "claude", "gemini", "perplexity", "kimi"]
N_QUERIES = 10

responses = pd.read_parquet(cfg.SILVER / f"responses_{cfg.RUN_PROFILE}.parquet")
citations = pd.read_parquet(cfg.SILVER / f"citations_{cfg.RUN_PROFILE}.parquet")

# same deterministic selection as the prior smokes -> fair head-to-head
base = responses[responses.engine.isin(ENGINES)]
per_q = base.groupby("query_id")["engine"].nunique()
full_qids = sorted(per_q[per_q >= len(ENGINES)].index)[:N_QUERIES]

cache, rows = {}, []
t0 = time.time()
print("=" * 72)
print(f"NLI ATTRIBUTION SMOKE (Option 3) | {len(full_qids)} q x {len(ENGINES)} engines, 1 run")
print(f"model: {cfg.NLI_MODEL}  |  threshold: {cfg.NLI_ENTAILMENT_THRESHOLD}  |  device: {cfg.NLI_DEVICE}")
print("=" * 72)

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
        res = nli.attribute_cell_nli(answer, srcs, cache)
        res.update({"engine": eng, "query_id": qid, "run_index": run})
        rows.append(res)
        ais = res['ais_rate']
        print(f"  {eng:11s} {qid[:18]} | sents {res['n_sentences']:3d} | "
              f"src {res['n_sources_fetched_ok']}/{res['n_sources_cited']} | "
              f"AIS {round(ais,2) if ais is not None else None} | "
              f"PAWC {res['pawc_total']} | NLI evals {res['nli_evaluations']}")

elapsed = time.time() - t0
df = pd.DataFrame(rows)
cfg.GOLD.mkdir(parents=True, exist_ok=True)
df.drop(columns=["fetch_status", "pawc_by_source_position"]).to_parquet(
    cfg.GOLD / f"smoke_nli_{cfg.RUN_PROFILE}.parquet", index=False)

print("\n" + "-" * 72)
print("PER-ENGINE SUMMARY (Option 3 / NLI):")
g = df.groupby("engine").agg(
    cells=("query_id", "count"),
    mean_sents=("n_sentences", "mean"),
    mean_ais=("ais_rate", "mean"),
    mean_pawc=("pawc_total", "mean"),
    total_evals=("nli_evaluations", "sum"),
).round(2)
print(g.to_string())

print("\nRUNTIME / COST:")
print(f"  cells          : {len(df)}")
print(f"  NLI evaluations: {int(df['nli_evaluations'].sum())}  (no API cost)")
print(f"  elapsed        : {elapsed/60:.1f} min  ({elapsed/len(df):.1f} s/cell)")
print(f"  $ API spend    : $0.00  (local model)")

# side-by-side vs per-sentence judge smoke if present
judge_path = cfg.GOLD / f"smoke_pawc_ais_{cfg.RUN_PROFILE}.parquet"
if judge_path.exists():
    js = pd.read_parquet(judge_path)
    j = js.groupby("engine").agg(j_ais=("ais_rate", "mean"),
                                 j_pawc=("pawc_total", "mean")).round(3)
    side = g[["mean_ais", "mean_pawc"]].rename(columns={"mean_ais": "nli_ais",
                                                        "mean_pawc": "nli_pawc"}).join(j)
    side["d_ais"] = (side["nli_ais"] - side["j_ais"]).round(3)
    side["d_pawc"] = (side["nli_pawc"] - side["j_pawc"]).round(2)
    print("\nSIDE-BY-SIDE vs per-sentence judge smoke:")
    print(side.to_string())
    # ranking check
    print("\nRanking by AIS:")
    print(f"  judge: {' > '.join(j.sort_values('j_ais', ascending=False).index.tolist())}")
    print(f"  nli:   {' > '.join(g.sort_values('mean_ais', ascending=False).index.tolist())}")
else:
    print("\n[per-sentence judge smoke not found; skipping side-by-side]")
print("=" * 72)
